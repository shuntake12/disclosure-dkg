"""時系列知識グラフ構築: NetworkX MultiDiGraph ベース。

ExtractedFact のリストから時系列 KG を構築・クエリ・保存。
"""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx

from schema import ExtractedFact, FiscalPeriod

GRAPHS_DIR = Path(__file__).parent / "graphs"
GRAPHS_DIR.mkdir(exist_ok=True)


def build_graph(facts: list[ExtractedFact]) -> nx.MultiDiGraph:
    """ExtractedFact リストから時系列 KG を構築。"""
    G = nx.MultiDiGraph()

    for fact in facts:
        q = fact.quintuple
        src = q.subject.name
        tgt = q.object_text

        # ノード追加（属性付き）
        if src not in G:
            G.add_node(
                src,
                entity_type=q.subject.entity_type.value,
                identifiers=q.subject.identifiers,
            )

        if q.object_entity and q.object_entity.name:
            tgt = q.object_entity.name
            if tgt not in G:
                G.add_node(
                    tgt,
                    entity_type=q.object_entity.entity_type.value,
                    identifiers=q.object_entity.identifiers,
                )
        elif tgt not in G:
            G.add_node(tgt, entity_type="value")

        # エッジ追加（時間・数値・出典付き）
        G.add_edge(
            src,
            tgt,
            relation=q.relation.value,
            value=q.value,
            unit=q.unit,
            fiscal_year=q.timestamp.fiscal_year,
            period_type=q.timestamp.period_type,
            disclosure_date=q.timestamp.disclosure_date or "",
            confidence=q.confidence,
            source_doc=fact.source.doc_id,
            source_section=fact.source.section,
            source_snippet=fact.source.text_snippet,
            extraction_method=fact.extraction_method,
        )

    return G


def add_facts(G: nx.MultiDiGraph, facts: list[ExtractedFact]) -> None:
    """既存グラフに事実を追加。"""
    new_g = build_graph(facts)
    G.update(new_g)


def snapshot_at(G: nx.MultiDiGraph, fiscal_year: str) -> nx.DiGraph:
    """特定時点のスナップショットを返す。"""
    snap = nx.DiGraph()

    for u, v, data in G.edges(data=True):
        if data.get("fiscal_year") == fiscal_year:
            snap.add_node(u, **{k: v for k, v in G.nodes[u].items()})
            snap.add_node(v, **{k: v for k, v in G.nodes[v].items()})
            snap.add_edge(u, v, **data)

    return snap


def entity_timeline(G: nx.MultiDiGraph, entity_name: str) -> list[dict]:
    """エンティティに関する全事実を時系列順で返す。"""
    timeline = []

    # outgoing edges
    for _, tgt, data in G.out_edges(entity_name, data=True):
        timeline.append({
            "direction": "out",
            "target": tgt,
            **data,
        })

    # incoming edges
    for src, _, data in G.in_edges(entity_name, data=True):
        timeline.append({
            "direction": "in",
            "source": src,
            **data,
        })

    timeline.sort(key=lambda x: x.get("fiscal_year", ""))
    return timeline


def query_facts(
    G: nx.MultiDiGraph,
    entity: str | None = None,
    relation: str | None = None,
    fiscal_year: str | None = None,
) -> list[dict]:
    """条件指定でエッジ（事実）を検索。"""
    results = []
    for u, v, data in G.edges(data=True):
        if entity and entity not in (u, v):
            continue
        if relation and data.get("relation") != relation:
            continue
        if fiscal_year and data.get("fiscal_year") != fiscal_year:
            continue
        results.append({"subject": u, "object": v, **data})
    return results


def graph_stats(G: nx.MultiDiGraph) -> dict:
    """グラフの統計情報。"""
    entity_types: dict[str, int] = {}
    for _, data in G.nodes(data=True):
        et = data.get("entity_type", "unknown")
        entity_types[et] = entity_types.get(et, 0) + 1

    relation_types: dict[str, int] = {}
    for _, _, data in G.edges(data=True):
        rt = data.get("relation", "unknown")
        relation_types[rt] = relation_types.get(rt, 0) + 1

    fiscal_years: set[str] = set()
    for _, _, data in G.edges(data=True):
        fy = data.get("fiscal_year", "")
        if fy:
            fiscal_years.add(fy)

    return {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "entity_types": entity_types,
        "relation_types": relation_types,
        "fiscal_years": sorted(fiscal_years),
    }


def save_graph(G: nx.MultiDiGraph, name: str = "dkg") -> Path:
    """グラフを JSON (node-link format) で保存。"""
    path = GRAPHS_DIR / f"{name}.json"
    data = nx.node_link_data(G)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  Saved graph ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges) to {path}")
    return path


def load_graph(name: str = "dkg") -> nx.MultiDiGraph:
    """保存済みグラフを読み込み。"""
    path = GRAPHS_DIR / f"{name}.json"
    if not path.exists():
        return nx.MultiDiGraph()
    data = json.loads(path.read_text(encoding="utf-8"))
    return nx.node_link_graph(data, multigraph=True, directed=True)
