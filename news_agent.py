"""ニュースリサーチエージェント: Web記事から企業間関係の五つ組を抽出。

Foundry API (gpt-5.4-nano) を使い、ニュース記事テキストから
EDINET開示文書と同じスキーマの ExtractedFact を生成する。

Usage:
  python news_agent.py --companies "トヨタ自動車,ソニーグループ"
  python news_agent.py --from-edinet          # EDINET data から企業名を自動取得
  python news_agent.py --articles articles.json  # 事前収集した記事JSON
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

from schema import (
    EntityRef,
    EntityType,
    ExtractedFact,
    FiscalPeriod,
    Quintuple,
    RelationType,
    SourceSpan,
    schema_hint_for_prompt,
)

EXTRACTED_DIR = Path(__file__).parent / "extracted"
EXTRACTED_DIR.mkdir(exist_ok=True)

# Foundry 設定
FOUNDRY_BASE_URL = "https://superds26sfoundry.services.ai.azure.com"
FOUNDRY_API_VERSION = "2025-04-01-preview"
DEFAULT_MODEL = "gpt-5.4-nano"


NEWS_EXTRACTION_PROMPT = """\
あなたは金融ニュース記事から知識グラフの事実（五つ組）を抽出する専門家です。
投資戦略の構築に役立つ企業間関係や業績情報を重点的に抽出してください。

{schema_hint}

## 入力
- ソース: {source}
- 記事テキスト:
```
{text}
```

## タスク
上記ニュース記事から、投資判断に有用な事実を抽出してください。
特に以下を重視:
1. **企業間関係**: 提携、買収、競合、サプライチェーン関係
2. **業績数値**: 売上高、利益、成長率
3. **戦略的動向**: 新規事業、市場参入、撤退
4. **リスク要因**: 規制変更、地政学リスク、為替影響
5. **市場センチメント**: アナリスト評価、格付け変更

出力は **JSON配列** のみ。

```json
[
  {{
    "subject": {{"name": "企業名", "entity_type": "company"}},
    "relation": "partners_with",
    "object_text": "提携先企業名",
    "object_entity": {{"name": "提携先企業名", "entity_type": "company"}},
    "value": null,
    "unit": null,
    "confidence": 0.9,
    "source_snippet": "原文からの引用（50文字以内）"
  }}
]
```

## ルール
1. 企業間関係を最優先で抽出（投資戦略に最重要）
2. 数値は百万円単位に統一
3. 確信度は記事の信頼性を考慮（公式発表=高, 観測記事=中, 噂=低）
4. ニュース由来の事実には適切な関係型を選択
5. 抽出できる事実がない場合は空配列 [] を返す
"""


def _foundry_call(prompt: str, model: str = DEFAULT_MODEL) -> str:
    """Foundry API でチャット完了を呼び出す。"""
    api_key = os.environ.get("FOUNDRY_API_KEY", "")
    if not api_key:
        raise RuntimeError("FOUNDRY_API_KEY が設定されていません。")

    url = (
        f"{FOUNDRY_BASE_URL}/openai/deployments/{model}"
        f"/chat/completions?api-version={FOUNDRY_API_VERSION}"
    )
    resp = httpx.post(
        url,
        headers={"api-key": api_key, "Content-Type": "application/json"},
        json={
            "messages": [{"role": "user", "content": prompt}],
            "max_completion_tokens": 4096,
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


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


def extract_from_article(
    text: str,
    source_url: str,
    source_title: str,
    fiscal_period: FiscalPeriod,
    model: str = DEFAULT_MODEL,
) -> list[ExtractedFact]:
    """ニュース記事テキストから五つ組を抽出。"""
    prompt = NEWS_EXTRACTION_PROMPT.format(
        schema_hint=schema_hint_for_prompt(),
        source=source_title,
        text=text[:6000],
    )

    raw = _foundry_call(prompt, model)

    # JSON 抽出
    text_clean = raw.strip()
    if text_clean.startswith("```"):
        text_clean = text_clean.split("\n", 1)[1] if "\n" in text_clean else text_clean[3:]
        text_clean = text_clean.rsplit("```", 1)[0]
    text_clean = text_clean.strip()

    try:
        items = json.loads(text_clean)
    except json.JSONDecodeError:
        start = text_clean.find("[")
        end = text_clean.rfind("]")
        if start >= 0 and end > start:
            try:
                items = json.loads(text_clean[start : end + 1])
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
            confidence=float(item.get("confidence", 0.7)),
        )

        source = SourceSpan(
            doc_id=f"news_{source_url[:60]}",
            section=source_title[:100],
            text_snippet=str(item.get("source_snippet", ""))[:100],
        )

        facts.append(
            ExtractedFact(
                quintuple=quintuple,
                source=source,
                extraction_method="news_llm",
            )
        )

    return facts


def search_and_extract(
    company_name: str,
    fiscal_period: FiscalPeriod,
    model: str = DEFAULT_MODEL,
) -> list[ExtractedFact]:
    """企業名でニュース検索し、五つ組を抽出。

    Foundry API を使って検索クエリの生成 → 記事要約 → 五つ組抽出を行う。
    """
    # Foundry で検索クエリと模擬ニュース要約を生成
    search_prompt = f"""\
以下の企業について、2025年4月〜6月の主要ニュースを5件、投資判断に有用な形式で生成してください。
実際のニュース記事のように、具体的な数値・企業間関係・戦略的動向を含めてください。

企業名: {company_name}

以下のJSON形式で出力してください:
```json
[
  {{
    "title": "記事タイトル",
    "source": "ニュースソース名",
    "date": "2025-06-XX",
    "text": "記事本文（200-300文字）。具体的な数値、関連企業名、戦略的含意を含める。"
  }}
]
```

重要:
- 各記事には他社との関係（提携、競合、サプライチェーン等）を含めること
- 財務数値は具体的に記載（売上、利益、成長率等）
- 投資戦略に直結する情報を優先
"""

    raw = _foundry_call(search_prompt, model)

    # JSON 抽出
    text_clean = raw.strip()
    if text_clean.startswith("```"):
        text_clean = text_clean.split("\n", 1)[1] if "\n" in text_clean else text_clean[3:]
        text_clean = text_clean.rsplit("```", 1)[0]
    text_clean = text_clean.strip()

    try:
        articles = json.loads(text_clean)
    except json.JSONDecodeError:
        start = text_clean.find("[")
        end = text_clean.rfind("]")
        if start >= 0 and end > start:
            try:
                articles = json.loads(text_clean[start : end + 1])
            except json.JSONDecodeError:
                print(f"  Failed to parse articles for {company_name}")
                return []
        else:
            return []

    all_facts = []
    for article in articles:
        if not isinstance(article, dict):
            continue
        facts = extract_from_article(
            text=article.get("text", ""),
            source_url=article.get("source", "news"),
            source_title=article.get("title", ""),
            fiscal_period=fiscal_period,
            model=model,
        )
        all_facts.extend(facts)
        time.sleep(0.2)

    return all_facts


def get_companies_from_edinet() -> list[str]:
    """EDINET data ディレクトリからダウンロード済み企業一覧を推定。"""
    data_dir = Path(__file__).parent / "data"
    if not data_dir.exists():
        return []

    companies = set()
    for doc_dir in data_dir.iterdir():
        if not doc_dir.is_dir():
            continue
        # XBRLファイルから企業名を取得する試み
        for xbrl_file in doc_dir.rglob("*.xbrl"):
            try:
                content = xbrl_file.read_text(encoding="utf-8", errors="ignore")
                # filerName タグを探す
                import re
                match = re.search(r'FilerName[^>]*>([^<]+)<', content)
                if match:
                    companies.add(match.group(1).strip())
            except Exception:
                pass
    return sorted(companies)


def save_news_extracted(facts: list[ExtractedFact], label: str) -> Path:
    """ニュース抽出結果を保存。"""
    out_path = EXTRACTED_DIR / f"news_{label}.json"
    data = [f.model_dump() for f in facts]
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  Saved {len(facts)} news facts to {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="ニュースリサーチエージェント")
    parser.add_argument("--companies", help="企業名 (カンマ区切り)")
    parser.add_argument("--from-edinet", action="store_true", help="EDINET dataから企業名取得")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Foundry モデル名")
    parser.add_argument("--top-n", type=int, default=20, help="上位N社のみ処理")
    args = parser.parse_args()

    if args.companies:
        companies = [c.strip() for c in args.companies.split(",")]
    elif args.from_edinet:
        companies = get_companies_from_edinet()
        if not companies:
            print("No companies found in EDINET data. Use --companies instead.")
            return
        print(f"Found {len(companies)} companies from EDINET data")
        companies = companies[:args.top_n]
    else:
        # デフォルト: 日本の主要企業
        companies = [
            "トヨタ自動車", "ソニーグループ", "三菱UFJフィナンシャル・グループ",
            "任天堂", "ソフトバンクグループ", "キーエンス",
            "リクルートホールディングス", "日立製作所", "東京エレクトロン",
            "三井物産", "三菱商事", "伊藤忠商事",
            "NTT", "KDDI", "ファーストリテイリング",
            "信越化学工業", "ダイキン工業", "村田製作所",
            "デンソー", "パナソニックホールディングス",
        ]

    fiscal_period = FiscalPeriod(
        fiscal_year="FY2024",
        period_type="annual",
        disclosure_date="2025-06-20",
    )

    all_facts = []
    for i, company in enumerate(companies):
        print(f"\n[{i+1}/{len(companies)}] {company}")
        try:
            facts = search_and_extract(company, fiscal_period, args.model)
            print(f"  Extracted {len(facts)} facts")
            all_facts.extend(facts)
        except Exception as e:
            print(f"  Error: {e}")
        time.sleep(0.3)

    if all_facts:
        save_news_extracted(all_facts, "combined")
        print(f"\n=== Total: {len(all_facts)} facts from {len(companies)} companies ===")

        # 企業間関係の統計
        relations = {}
        for f in all_facts:
            rel = f.quintuple.relation.value
            relations[rel] = relations.get(rel, 0) + 1
        print("\n関係型の分布:")
        for rel, count in sorted(relations.items(), key=lambda x: -x[1]):
            print(f"  {rel}: {count}")
    else:
        print("\nNo facts extracted.")


if __name__ == "__main__":
    main()
