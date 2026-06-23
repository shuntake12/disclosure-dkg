"""LLM 抽出器: Claude API で五つ組を抽出（ICKG 日本語版）。

テキストチャンクからエンティティ・関係・数値を構造化抽出し、
XBRL 由来の数値事実とマージする。
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import httpx
from tqdm import tqdm

from schema import (
    EntityRef,
    EntityType,
    ExtractedFact,
    FiscalPeriod,
    Quintuple,
    RelationType,
    SourceSpan,
    TextChunk,
    schema_hint_for_prompt,
)

EXTRACTED_DIR = Path(__file__).parent / "extracted"
EXTRACTED_DIR.mkdir(exist_ok=True)

# 抽出モデル（Foundry: gpt-5.4-nano=高速低コスト, gpt-5.4=高品質）
DEFAULT_MODEL = "gpt-5.4-nano"

# Azure AI Foundry 設定
FOUNDRY_BASE_URL = "https://superds26sfoundry.services.ai.azure.com"
FOUNDRY_API_VERSION = "2025-04-01-preview"


def _foundry_api_key() -> str:
    key = os.environ.get("FOUNDRY_API_KEY", "")
    if not key:
        raise RuntimeError("FOUNDRY_API_KEY が設定されていません。")
    return key

EXTRACTION_PROMPT_TEMPLATE = """\
あなたは金融開示文書から知識グラフの事実（五つ組）を抽出する専門家です。

{schema_hint}

## 入力
- 文書ID: {doc_id}
- セクション: {section}
- テキスト:
```
{text}
```

## タスク
上記テキストから、以下の形式で事実を抽出してください。

出力は **JSON配列** のみ。説明文は不要です。

```json
[
  {{
    "subject": "{{"name": "企業名", "entity_type": "company"}}",
    "relation": "revenue_is",
    "object_text": "3兆7100億円",
    "object_entity": null,
    "value": 3710000,
    "unit": "百万円",
    "confidence": 0.95,
    "source_snippet": "売上高は3兆7,100億円（前年同期比12.3%増）"
  }}
]
```

## ルール
1. 数値を含む事実は必ず `value`（百万円単位に統一）と `unit` を記録
2. `source_snippet` に原文の該当箇所（50文字以内）を含める
3. エンティティ名は正式名称を使用（略称ではなく）
4. 関係型は上記スキーマから選択（該当なしなら "other"）
5. 確信度 `confidence` は 0.0-1.0 で（数値事実は高め、推測は低め）
6. 抽出できる事実がない場合は空配列 [] を返す
7. 業績予想と実績を区別する（forecast_* vs *_is）
"""


def _build_prompt(chunk: TextChunk) -> str:
    return EXTRACTION_PROMPT_TEMPLATE.format(
        schema_hint=schema_hint_for_prompt(),
        doc_id=chunk.doc_id,
        section=chunk.section,
        text=chunk.text[:4000],
    )


def _parse_entity_type(s: str) -> EntityType:
    try:
        return EntityType(s)
    except ValueError:
        return EntityType.COMPANY


def _parse_relation_type(s: str) -> RelationType:
    try:
        return RelationType(s)
    except ValueError:
        return RelationType.OTHER


def _parse_llm_response(
    raw: str,
    chunk: TextChunk,
    fiscal_period: FiscalPeriod,
) -> list[ExtractedFact]:
    """LLM のJSON応答を ExtractedFact リストに変換。"""
    # JSON配列を抽出
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        text = text.rsplit("```", 1)[0]
    text = text.strip()

    try:
        items = json.loads(text)
    except json.JSONDecodeError:
        # 配列部分だけ抽出
        start = text.find("[")
        end = text.rfind("]")
        if start >= 0 and end > start:
            try:
                items = json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return []
        else:
            return []

    if not isinstance(items, list):
        items = [items]

    facts = []
    for item in items:
        if not isinstance(item, dict):
            continue

        # subject パース
        subj = item.get("subject", {})
        if isinstance(subj, str):
            try:
                subj = json.loads(subj)
            except json.JSONDecodeError:
                subj = {"name": subj, "entity_type": "company"}

        subject = EntityRef(
            name=subj.get("name", ""),
            entity_type=_parse_entity_type(subj.get("entity_type", "company")),
        )
        if not subject.name:
            continue

        # object_entity パース
        obj_ent = item.get("object_entity")
        object_entity = None
        if obj_ent and isinstance(obj_ent, dict):
            object_entity = EntityRef(
                name=obj_ent.get("name", ""),
                entity_type=_parse_entity_type(obj_ent.get("entity_type", "company")),
            )

        quintuple = Quintuple(
            subject=subject,
            relation=_parse_relation_type(item.get("relation", "other")),
            object_text=str(item.get("object_text", "")),
            object_entity=object_entity,
            value=item.get("value"),
            unit=item.get("unit"),
            timestamp=fiscal_period,
            confidence=float(item.get("confidence", 0.8)),
        )

        source = SourceSpan(
            doc_id=chunk.doc_id,
            section=chunk.section,
            char_start=chunk.char_start,
            char_end=chunk.char_end,
            text_snippet=str(item.get("source_snippet", ""))[:100],
        )

        facts.append(
            ExtractedFact(
                quintuple=quintuple,
                source=source,
                extraction_method="llm_direct",
            )
        )

    return facts


def extract_from_chunks(
    chunks: list[TextChunk],
    fiscal_period: FiscalPeriod,
    model: str = DEFAULT_MODEL,
    sleep_sec: float = 0.2,
) -> list[ExtractedFact]:
    """テキストチャンクから五つ組を抽出（Azure AI Foundry 経由）。"""
    api_key = _foundry_api_key()
    url = (
        f"{FOUNDRY_BASE_URL}/openai/deployments/{model}"
        f"/chat/completions?api-version={FOUNDRY_API_VERSION}"
    )
    headers = {"api-key": api_key, "Content-Type": "application/json"}
    all_facts: list[ExtractedFact] = []

    for chunk in tqdm(chunks, desc="Extracting quintuples"):
        if len(chunk.text.strip()) < 50:
            continue

        prompt = _build_prompt(chunk)

        try:
            resp = httpx.post(
                url,
                headers=headers,
                json={
                    "messages": [{"role": "user", "content": prompt}],
                    "max_completion_tokens": 4096,
                },
                timeout=60,
            )
            resp.raise_for_status()
            raw_text = resp.json()["choices"][0]["message"]["content"]
            facts = _parse_llm_response(raw_text, chunk, fiscal_period)
            all_facts.extend(facts)
        except Exception as e:
            print(f"  Extraction error ({chunk.section}): {e}")

        time.sleep(sleep_sec)

    return all_facts


def xbrl_facts_to_extracted(
    xbrl_facts: list[dict],
    doc_id: str,
    filer_name: str,
    fiscal_period: FiscalPeriod,
) -> list[ExtractedFact]:
    """XBRL 数値事実を ExtractedFact に変換。"""
    results = []
    for f in xbrl_facts:
        try:
            val = float(f["value"])
        except (ValueError, KeyError):
            continue

        quintuple = Quintuple(
            subject=EntityRef(name=filer_name, entity_type=EntityType.COMPANY),
            relation=_parse_relation_type(f.get("relation", "other")),
            object_text=f.get("value", ""),
            value=val,
            unit=f.get("unit", "JPY"),
            timestamp=fiscal_period,
            confidence=1.0,  # XBRL は確実
        )

        source = SourceSpan(
            doc_id=doc_id,
            section="XBRL",
            text_snippet=f"{f.get('element', '')}: {f.get('value', '')}",
        )

        results.append(
            ExtractedFact(
                quintuple=quintuple,
                source=source,
                extraction_method="xbrl_parsed",
            )
        )

    return results


def merge_facts(
    xbrl_facts: list[ExtractedFact],
    llm_facts: list[ExtractedFact],
) -> list[ExtractedFact]:
    """XBRL事実とLLM事実をマージ（重複排除）。

    同一 subject + relation + timestamp の場合、XBRL を優先。
    """
    seen: set[str] = set()
    merged: list[ExtractedFact] = []

    # XBRL 事実を先に追加（高信頼）
    for f in xbrl_facts:
        key = (
            f"{f.quintuple.subject.name}|"
            f"{f.quintuple.relation.value}|"
            f"{f.quintuple.object_text}|"
            f"{f.quintuple.timestamp.fiscal_year}"
        )
        if key not in seen:
            seen.add(key)
            merged.append(f)

    # LLM 事実で重複しないものを追加
    for f in llm_facts:
        key = (
            f"{f.quintuple.subject.name}|"
            f"{f.quintuple.relation.value}|"
            f"{f.quintuple.object_text}|"
            f"{f.quintuple.timestamp.fiscal_year}"
        )
        if key not in seen:
            seen.add(key)
            merged.append(f)

    return merged


def save_extracted(facts: list[ExtractedFact], doc_id: str) -> Path:
    """抽出結果を JSON ファイルに保存。"""
    out_path = EXTRACTED_DIR / f"{doc_id}.json"
    data = [f.model_dump() for f in facts]
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  Saved {len(facts)} facts to {out_path}")
    return out_path


def load_extracted(doc_id: str) -> list[ExtractedFact]:
    """保存済み抽出結果を読み込み。"""
    path = EXTRACTED_DIR / f"{doc_id}.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return [ExtractedFact.model_validate(d) for d in data]
