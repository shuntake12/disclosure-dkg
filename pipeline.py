"""CLI パイプライン: collect / extract / build / viz / query / run。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

# .env 読込
load_dotenv(Path(__file__).parent / ".env")

from schema import DocumentMeta, FiscalPeriod


def cmd_collect(args: argparse.Namespace) -> None:
    """EDINET から文書を取得・ダウンロード。"""
    from edinet_client import (
        download_documents,
        list_documents,
        list_documents_range,
    )

    doc_types = args.doc_types.split(",") if args.doc_types else None

    if args.date:
        print(f"Fetching documents for {args.date}...")
        docs = list_documents(args.date, doc_types)
    elif args.start and args.end:
        print(f"Fetching documents from {args.start} to {args.end}...")
        docs = list_documents_range(args.start, args.end, doc_types)
    else:
        print("Error: --date or --start/--end required")
        return

    print(f"  Found {len(docs)} documents")
    for d in docs[:10]:
        print(f"    {d.doc_id}: {d.filer_name} — {d.doc_description}")
    if len(docs) > 10:
        print(f"    ... and {len(docs) - 10} more")

    if not args.list_only:
        print(f"\nDownloading {len(docs)} documents...")
        download_documents(docs)


def cmd_extract(args: argparse.Namespace) -> None:
    """文書から五つ組を抽出。"""
    from document_parser import extract_xbrl_numeric_facts, parse_document
    from extractor import (
        extract_from_chunks,
        merge_facts,
        save_extracted,
        xbrl_facts_to_extracted,
    )

    data_dir = Path(__file__).parent / "data"

    if args.doc_id:
        doc_ids = [args.doc_id]
    elif args.all:
        doc_ids = [d.name for d in data_dir.iterdir() if d.is_dir()]
    else:
        print("Error: --doc-id or --all required")
        return

    fiscal_period = FiscalPeriod(
        fiscal_year=args.fiscal_year or "FY2024",
        period_type=args.period_type or "annual",
    )

    model = args.model or "gpt-5.4-nano"

    for doc_id in doc_ids:
        doc_dir = data_dir / doc_id
        if not doc_dir.exists():
            print(f"  Skipping {doc_id}: not downloaded")
            continue

        print(f"\n=== Extracting: {doc_id} ===")

        # テキストチャンク
        chunks = parse_document(doc_dir, doc_id)
        print(f"  {len(chunks)} text chunks")

        # XBRL 数値
        xbrl_raw = extract_xbrl_numeric_facts(doc_dir)
        xbrl_facts = xbrl_facts_to_extracted(
            xbrl_raw, doc_id, args.filer_name or doc_id, fiscal_period
        )
        print(f"  {len(xbrl_facts)} XBRL facts")

        # LLM 抽出
        llm_facts = extract_from_chunks(chunks, fiscal_period, model=model)
        print(f"  {len(llm_facts)} LLM facts")

        # マージ
        merged = merge_facts(xbrl_facts, llm_facts)
        print(f"  {len(merged)} merged facts (deduplicated)")

        save_extracted(merged, doc_id)


def cmd_build(args: argparse.Namespace) -> None:
    """抽出結果から KG を構築。"""
    from extractor import load_extracted
    from kg_builder import build_graph, graph_stats, save_graph

    extracted_dir = Path(__file__).parent / "extracted"
    all_facts = []

    for json_file in sorted(extracted_dir.glob("*.json")):
        doc_id = json_file.stem
        facts = load_extracted(doc_id)
        all_facts.extend(facts)
        print(f"  Loaded {len(facts)} facts from {doc_id}")

    if not all_facts:
        print("No extracted facts found. Run 'extract' first.")
        return

    print(f"\nBuilding graph from {len(all_facts)} facts...")
    G = build_graph(all_facts)

    stats = graph_stats(G)
    print(f"\n=== Graph Statistics ===")
    print(f"  Nodes: {stats['nodes']}")
    print(f"  Edges: {stats['edges']}")
    print(f"  Entity types: {json.dumps(stats['entity_types'], ensure_ascii=False)}")
    print(f"  Relation types: {json.dumps(stats['relation_types'], ensure_ascii=False)}")
    print(f"  Fiscal years: {stats['fiscal_years']}")

    name = args.name or "dkg"
    save_graph(G, name)


def cmd_viz(args: argparse.Namespace) -> None:
    """KG を可視化。"""
    from kg_builder import load_graph
    from visualizer import (
        export_to_csv,
        plot_metric_timeline,
        visualize_full_graph,
        visualize_neighborhood,
    )

    G = load_graph(args.graph or "dkg")
    if G.number_of_nodes() == 0:
        print("Graph is empty. Run 'build' first.")
        return

    if args.entity:
        visualize_neighborhood(G, args.entity, depth=args.depth, fiscal_year=args.fiscal_year)
        if args.metric:
            plot_metric_timeline(G, args.entity, args.metric)
    else:
        visualize_full_graph(G, fiscal_year=args.fiscal_year)

    if args.csv:
        export_to_csv(G)


def cmd_query(args: argparse.Namespace) -> None:
    """KG をクエリ。"""
    from kg_builder import entity_timeline, load_graph, query_facts

    G = load_graph(args.graph or "dkg")
    if G.number_of_nodes() == 0:
        print("Graph is empty. Run 'build' first.")
        return

    if args.timeline:
        results = entity_timeline(G, args.entity)
    else:
        results = query_facts(G, args.entity, args.relation, args.fiscal_year)

    print(f"\n{len(results)} results:")
    for r in results:
        print(json.dumps(r, ensure_ascii=False, default=str, indent=2))


def cmd_run(args: argparse.Namespace) -> None:
    """フルパイプライン実行。"""
    print("=== Step 1: Collect ===")
    cmd_collect(args)

    print("\n=== Step 2: Extract ===")
    args.all = True
    args.doc_id = None
    cmd_extract(args)

    print("\n=== Step 3: Build ===")
    args.name = "dkg"
    cmd_build(args)

    print("\n=== Step 4: Visualize ===")
    args.entity = None
    args.csv = True
    args.graph = "dkg"
    args.depth = 1
    args.metric = None
    cmd_viz(args)

    print("\n=== Pipeline Complete ===")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Disclosure DKG — 決算短信・統合報告書の動的知識グラフ化",
    )
    sub = parser.add_subparsers(dest="command")

    # collect
    p_collect = sub.add_parser("collect", help="EDINET から文書取得")
    p_collect.add_argument("--date", help="対象日 (YYYY-MM-DD)")
    p_collect.add_argument("--start", help="開始日")
    p_collect.add_argument("--end", help="終了日")
    p_collect.add_argument("--doc-types", help="文書種別コード (カンマ区切り, 例: 120,140)")
    p_collect.add_argument("--list-only", action="store_true", help="一覧表示のみ")

    # extract
    p_extract = sub.add_parser("extract", help="五つ組抽出")
    p_extract.add_argument("--doc-id", help="対象文書ID")
    p_extract.add_argument("--all", action="store_true", help="全ダウンロード済み文書")
    p_extract.add_argument("--fiscal-year", help="会計年度 (例: FY2024)")
    p_extract.add_argument("--period-type", help="期間種別 (annual/q1/q2/q3/q4)")
    p_extract.add_argument("--filer-name", help="提出者名")
    p_extract.add_argument("--model", help="Claude モデル名")

    # build
    p_build = sub.add_parser("build", help="KG 構築")
    p_build.add_argument("--name", default="dkg", help="グラフ名")

    # viz
    p_viz = sub.add_parser("viz", help="可視化")
    p_viz.add_argument("--entity", help="対象エンティティ名")
    p_viz.add_argument("--depth", type=int, default=1, help="近傍の深さ")
    p_viz.add_argument("--fiscal-year", help="対象会計年度")
    p_viz.add_argument("--metric", help="時系列チャートの指標")
    p_viz.add_argument("--graph", default="dkg", help="グラフ名")
    p_viz.add_argument("--csv", action="store_true", help="CSV エクスポート")

    # query
    p_query = sub.add_parser("query", help="KG クエリ")
    p_query.add_argument("--entity", help="エンティティ名")
    p_query.add_argument("--relation", help="関係型")
    p_query.add_argument("--fiscal-year", help="会計年度")
    p_query.add_argument("--timeline", action="store_true", help="タイムライン表示")
    p_query.add_argument("--graph", default="dkg", help="グラフ名")

    # run (full pipeline)
    p_run = sub.add_parser("run", help="フルパイプライン実行")
    p_run.add_argument("--date", help="対象日")
    p_run.add_argument("--start", help="開始日")
    p_run.add_argument("--end", help="終了日")
    p_run.add_argument("--doc-types", help="文書種別コード")
    p_run.add_argument("--list-only", action="store_true")
    p_run.add_argument("--fiscal-year", help="会計年度")
    p_run.add_argument("--period-type", help="期間種別")
    p_run.add_argument("--filer-name", help="提出者名")
    p_run.add_argument("--model", help="Claude モデル名")
    p_run.add_argument("--name", default="dkg", help="グラフ名")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    commands = {
        "collect": cmd_collect,
        "extract": cmd_extract,
        "build": cmd_build,
        "viz": cmd_viz,
        "query": cmd_query,
        "run": cmd_run,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
