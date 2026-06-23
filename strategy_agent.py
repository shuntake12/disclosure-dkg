"""投資戦略分析エージェント: DKG を分析して投資戦略を提案。

構築済みの DKG（EDINET開示 + ニュース）を Foundry API (gpt-5.4) で分析し、
企業間関係・業績トレンド・リスク要因から投資戦略を提案する。

Usage:
  python strategy_agent.py                    # フル分析
  python strategy_agent.py --focus "半導体"   # セクター指定
  python strategy_agent.py --company "トヨタ自動車"  # 企業指定
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import httpx
import networkx as nx
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

from kg_builder import graph_stats, load_graph, query_facts

# Foundry 設定
FOUNDRY_BASE_URL = "https://superds26sfoundry.services.ai.azure.com"
FOUNDRY_API_VERSION = "2025-04-01-preview"
ANALYSIS_MODEL = "gpt-5.4"  # 高品質モデルで戦略分析
FAST_MODEL = "gpt-5.4-nano"  # 高速モデルでデータ整理

OUTPUT_DIR = Path(__file__).parent / "analysis"
OUTPUT_DIR.mkdir(exist_ok=True)


def _foundry_call(prompt: str, model: str = ANALYSIS_MODEL) -> str:
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
            "max_completion_tokens": 8192,
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def extract_graph_context(G: nx.MultiDiGraph, focus: str | None = None) -> str:
    """グラフから分析用のコンテキストを構築。"""
    stats = graph_stats(G)
    lines = [
        f"## グラフ概要",
        f"- ノード数: {stats['nodes']}",
        f"- エッジ数: {stats['edges']}",
        f"- エンティティ型: {json.dumps(stats['entity_types'], ensure_ascii=False)}",
        f"- 関係型: {json.dumps(stats['relation_types'], ensure_ascii=False)}",
        f"- 会計年度: {stats['fiscal_years']}",
    ]

    # 主要企業（次数上位）
    degree_sorted = sorted(G.degree(), key=lambda x: x[1], reverse=True)
    top_entities = degree_sorted[:30]
    lines.append(f"\n## 主要エンティティ (次数上位30)")
    for name, deg in top_entities:
        node_data = G.nodes[name]
        etype = node_data.get("entity_type", "unknown")
        lines.append(f"- {name} (type={etype}, degree={deg})")

    # 企業間関係エッジ
    inter_company_edges = []
    for u, v, data in G.edges(data=True):
        u_type = G.nodes.get(u, {}).get("entity_type", "")
        v_type = G.nodes.get(v, {}).get("entity_type", "")
        if u_type == "company" and v_type == "company":
            inter_company_edges.append((u, v, data))

    lines.append(f"\n## 企業間関係 ({len(inter_company_edges)}件)")
    for u, v, data in inter_company_edges[:50]:
        rel = data.get("relation", "unknown")
        val = data.get("value", "")
        conf = data.get("confidence", "")
        val_str = f" value={val}" if val else ""
        lines.append(f"- {u} --[{rel}]--> {v}{val_str} (conf={conf})")

    # 財務数値エッジ
    financial_edges = []
    financial_rels = {
        "revenue_is", "operating_profit_is", "net_income_is",
        "forecast_revenue", "forecast_profit", "revenue_change",
    }
    for u, v, data in G.edges(data=True):
        if data.get("relation", "") in financial_rels:
            financial_edges.append((u, v, data))

    lines.append(f"\n## 主要財務データ ({len(financial_edges)}件)")
    for u, v, data in financial_edges[:80]:
        rel = data.get("relation", "")
        val = data.get("value", "")
        unit = data.get("unit", "")
        fy = data.get("fiscal_year", "")
        lines.append(f"- {u}: {rel}={val} {unit} ({fy})")

    # フォーカスがある場合は関連ノードを重点表示
    if focus:
        lines.append(f"\n## フォーカス: {focus} 関連")
        for node in G.nodes():
            if focus in str(node):
                neighbors = list(G.neighbors(node)) + list(G.predecessors(node))
                for n in neighbors[:10]:
                    edges = list(G.edges(node, data=True)) + list(G.in_edges(node, data=True))
                    for u, v, d in edges:
                        lines.append(f"  {u} --[{d.get('relation','')}]--> {v}")

    return "\n".join(lines)


def analyze_investment_strategy(
    G: nx.MultiDiGraph,
    focus: str | None = None,
    company: str | None = None,
) -> str:
    """DKGを分析し投資戦略を提案。"""
    context = extract_graph_context(G, focus=focus or company)

    prompt = f"""\
あなたは日本株式市場に精通した定量・定性分析の専門アナリストです。
以下の動的知識グラフ（DKG）データを分析し、投資戦略を提案してください。

{context}

## 分析タスク

以下の5つの観点から包括的に分析してください:

### 1. ネットワーク構造分析
- 中心性の高い企業とその意味（ハブ企業の特定）
- 企業クラスターの特定（提携・サプライチェーンで密結合したグループ）
- 孤立した企業の投資機会（過小評価の可能性）

### 2. バリューチェーン分析
- サプライチェーン上の重要ポジション
- 川上・川下の支配力を持つ企業
- ボトルネックリスクのある依存関係

### 3. 業績トレンド分析
- 売上高・利益の成長率が高い企業
- 業績予想と実績の乖離（ポジティブ/ネガティブサプライズ）
- セグメント別の成長ドライバー

### 4. リスク分析
- 集中リスク（特定顧客/サプライヤーへの依存）
- セクター横断的なリスク要因
- ESG関連のリスクと機会

### 5. 投資戦略提案
具体的な投資アイデアを3-5つ提案してください:
- **テーマ名**: 簡潔なテーマ名
- **対象企業**: 具体的な企業名
- **根拠**: DKGデータに基づく論拠
- **リスク**: 主要リスク要因
- **確信度**: 高/中/低
- **時間軸**: 短期(〜3ヶ月)/中期(3-12ヶ月)/長期(1年〜)

{"フォーカス: " + (focus or company) + " に特に注目してください。" if focus or company else ""}

JSON形式で出力してください:
```json
{{
  "network_analysis": {{
    "hub_companies": ["..."],
    "clusters": [{{"name": "...", "companies": ["..."], "theme": "..."}}],
    "isolated_opportunities": ["..."]
  }},
  "value_chain": {{
    "key_positions": [{{"company": "...", "role": "...", "strength": "..."}}],
    "bottleneck_risks": ["..."]
  }},
  "performance": {{
    "growth_leaders": [{{"company": "...", "metric": "...", "value": "..."}}],
    "surprises": [{{"company": "...", "type": "positive/negative", "detail": "..."}}]
  }},
  "risks": {{
    "concentration_risks": ["..."],
    "sector_risks": ["..."],
    "esg_factors": ["..."]
  }},
  "strategies": [
    {{
      "theme": "...",
      "companies": ["..."],
      "rationale": "...",
      "risks": "...",
      "confidence": "高/中/低",
      "timeframe": "短期/中期/長期"
    }}
  ],
  "summary": "全体要約（3-5文）"
}}
```
"""

    return _foundry_call(prompt, ANALYSIS_MODEL)


def analyze_company_deep(G: nx.MultiDiGraph, company: str) -> str:
    """特定企業の深掘り分析。"""
    # 企業の全エッジを取得
    if company not in G:
        # 部分一致で検索
        matches = [n for n in G.nodes() if company in n]
        if not matches:
            return json.dumps({"error": f"企業 '{company}' が見つかりません"})
        company = matches[0]

    edges_out = list(G.out_edges(company, data=True))
    edges_in = list(G.in_edges(company, data=True))

    context_lines = [f"## {company} の全関係"]
    for u, v, d in edges_out:
        context_lines.append(f"  OUT: {u} --[{d.get('relation','')}]--> {v} (value={d.get('value','')}, unit={d.get('unit','')})")
    for u, v, d in edges_in:
        context_lines.append(f"  IN:  {u} --[{d.get('relation','')}]--> {v} (value={d.get('value','')}, unit={d.get('unit','')})")

    context = "\n".join(context_lines)

    prompt = f"""\
以下の知識グラフデータに基づき、{company} の投資分析を行ってください。

{context}

以下を含むJSON形式で分析結果を出力:
1. 企業概要（KGから読み取れる事業構造）
2. 財務ハイライト（売上、利益、成長率）
3. 競争ポジション（競合・提携・サプライチェーン）
4. 投資判断（買い/中立/売り + 根拠）
5. カタリスト（株価変動の触媒となりうるイベント）
"""

    return _foundry_call(prompt, ANALYSIS_MODEL)


def main():
    parser = argparse.ArgumentParser(description="投資戦略分析エージェント")
    parser.add_argument("--graph", default="dkg", help="グラフ名")
    parser.add_argument("--focus", help="フォーカスセクター/テーマ")
    parser.add_argument("--company", help="特定企業の深掘り分析")
    parser.add_argument("--model", default=ANALYSIS_MODEL, help="分析モデル")
    args = parser.parse_args()

    print("Loading graph...")
    G = load_graph(args.graph)

    if G.number_of_nodes() == 0:
        print("Graph is empty. Run pipeline first.")
        return

    stats = graph_stats(G)
    print(f"Graph: {stats['nodes']} nodes, {stats['edges']} edges")

    if args.company:
        print(f"\n=== Deep analysis: {args.company} ===")
        result = analyze_company_deep(G, args.company)
    else:
        print(f"\n=== Investment Strategy Analysis ===")
        if args.focus:
            print(f"Focus: {args.focus}")
        result = analyze_investment_strategy(G, focus=args.focus)

    # 結果を保存
    label = args.company or args.focus or "full"
    out_path = OUTPUT_DIR / f"strategy_{label}.json"

    # JSONパース試行
    try:
        text = result.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
            text = text.rsplit("```", 1)[0]
        parsed = json.loads(text)
        out_path.write_text(json.dumps(parsed, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nAnalysis saved to {out_path}")

        # サマリー表示
        if isinstance(parsed, dict) and "summary" in parsed:
            print(f"\n=== サマリー ===\n{parsed['summary']}")
        if isinstance(parsed, dict) and "strategies" in parsed:
            print(f"\n=== 投資戦略 ===")
            for s in parsed["strategies"]:
                print(f"  [{s.get('confidence','')}] {s.get('theme','')}")
                print(f"    対象: {', '.join(s.get('companies', []))}")
                print(f"    時間軸: {s.get('timeframe','')}")
    except json.JSONDecodeError:
        out_path = OUTPUT_DIR / f"strategy_{label}.md"
        out_path.write_text(result, encoding="utf-8")
        print(f"\nAnalysis saved to {out_path}")
        print(result[:500])


if __name__ == "__main__":
    main()
