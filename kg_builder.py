"""時系列知識グラフ構築: NetworkX MultiDiGraph ベース。

ExtractedFact のリストから時系列 KG を構築・クエリ・保存。
"""

from __future__ import annotations

import csv
import json
import unicodedata
from pathlib import Path

import networkx as nx

from schema import ExtractedFact, FiscalPeriod, RelationType

GRAPHS_DIR = Path(__file__).parent / "graphs"
GRAPHS_DIR.mkdir(exist_ok=True)


def normalize_entity_name(name: str) -> str:
    """エンティティ名を正規化（FinDKG Sentence-BERT相当の前段処理）。

    NFKC 正規化（全角→半角）+ 表記揺れ統一。
    DyGFormer のノード埋め込みテーブルで同一エンティティが
    別ノードにならないことを保証する。
    """
    # NFKC: 全角英数→半角、㈱→(株) 等
    name = unicodedata.normalize("NFKC", name)
    # 前後空白
    name = name.strip()
    # 株式会社の表記揺れ統一
    name = name.replace("(株)", "株式会社")
    name = name.replace("（株）", "株式会社")
    # 末尾の株式会社を前置に統一
    if name.endswith("株式会社") and not name.startswith("株式会社"):
        name = "株式会社" + name[: -len("株式会社")]
    return name


def build_graph(facts: list[ExtractedFact]) -> nx.MultiDiGraph:
    """ExtractedFact リストから時系列 KG を構築。"""
    G = nx.MultiDiGraph()

    for fact in facts:
        q = fact.quintuple
        src = normalize_entity_name(q.subject.name)
        tgt = normalize_entity_name(q.object_text)

        # ノード追加（属性付き）
        if src not in G:
            G.add_node(
                src,
                entity_type=q.subject.entity_type.value,
                identifiers=q.subject.identifiers,
            )

        if q.object_entity and q.object_entity.name:
            tgt = normalize_entity_name(q.object_entity.name)
            if tgt not in G:
                G.add_node(
                    tgt,
                    entity_type=q.object_entity.entity_type.value,
                    identifiers=q.object_entity.identifiers,
                )
        elif tgt not in G:
            G.add_node(tgt, entity_type="value")

        # エッジ追加（時間・数値・出典付き + CTDG連続タイムスタンプ）
        G.add_edge(
            src,
            tgt,
            relation=q.relation.value,
            value=q.value,
            unit=q.unit,
            fiscal_year=q.timestamp.fiscal_year,
            period_type=q.timestamp.period_type,
            disclosure_date=q.timestamp.disclosure_date or "",
            continuous_ts=q.timestamp.to_unix_timestamp(),
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


def snapshot_at(G: nx.MultiDiGraph, fiscal_year: str) -> nx.MultiDiGraph:
    """特定時点のスナップショットを返す（MultiDiGraph 保持）。

    DiGraph ではなく MultiDiGraph を返すことで、同一エンティティペア間の
    複数エッジ（異なる relation/source_doc）が保持される。
    """
    snap = nx.MultiDiGraph()

    for u, v, _key, data in G.edges(data=True, keys=True):
        if data.get("fiscal_year") == fiscal_year:
            if u not in snap:
                snap.add_node(u, **dict(G.nodes[u]))
            if v not in snap:
                snap.add_node(v, **dict(G.nodes[v]))
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


def export_dyglib_format(
    G: nx.MultiDiGraph,
    name: str = "dkg_dyglib",
) -> Path:
    """DyGFormer/DyGLib 準拠のエッジリスト CSV を出力。

    形式: source_id, target_id, timestamp, label, feat_0, ..., feat_d
    - ノード ID は整数マッピング
    - timestamp は連続 Unix タイムスタンプ
    - edge features: 関係型 one-hot + 正規化 value + confidence
    """
    # ノード → 整数 ID マッピング
    node_list = sorted(G.nodes())
    node_to_id = {n: i for i, n in enumerate(node_list)}

    # 関係型 → one-hot インデックス
    all_relations = sorted(set(r.value for r in RelationType))
    rel_to_idx = {r: i for i, r in enumerate(all_relations)}
    n_rels = len(all_relations)

    # エッジを時系列順にソート
    edges = []
    for u, v, data in G.edges(data=True):
        edges.append((u, v, data))
    edges.sort(key=lambda x: x[2].get("continuous_ts", 0))

    # CSV 出力
    csv_path = GRAPHS_DIR / f"{name}.csv"
    id_map_path = GRAPHS_DIR / f"{name}_node_ids.json"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["source_id", "target_id", "timestamp", "label"]
        header += [f"rel_{r}" for r in all_relations]
        header += ["value_norm", "confidence"]
        writer.writerow(header)

        for u, v, data in edges:
            src_id = node_to_id[u]
            tgt_id = node_to_id[v]
            ts = data.get("continuous_ts", 0)
            label = 1  # 正例（観測されたエッジ）

            # 関係型 one-hot
            rel_onehot = [0] * n_rels
            rel_name = data.get("relation", "other")
            if rel_name in rel_to_idx:
                rel_onehot[rel_to_idx[rel_name]] = 1

            # 正規化 value（対数スケール）
            raw_val = data.get("value")
            if raw_val and isinstance(raw_val, (int, float)) and raw_val > 0:
                import math
                val_norm = round(math.log10(raw_val + 1) / 10, 4)
            else:
                val_norm = 0.0

            conf = data.get("confidence", 0.5)

            row = [src_id, tgt_id, ts, label] + rel_onehot + [val_norm, conf]
            writer.writerow(row)

    # ノード ID マッピングも保存
    id_map_path.write_text(
        json.dumps(node_to_id, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"  DyGLib format: {csv_path} ({len(edges)} edges, {len(node_list)} nodes)")
    print(f"  Node ID map: {id_map_path}")
    return csv_path


def load_graph(name: str = "dkg") -> nx.MultiDiGraph:
    """保存済みグラフを読み込み。"""
    path = GRAPHS_DIR / f"{name}.json"
    if not path.exists():
        return nx.MultiDiGraph()
    data = json.loads(path.read_text(encoding="utf-8"))
    return nx.node_link_graph(data, multigraph=True, directed=True)
