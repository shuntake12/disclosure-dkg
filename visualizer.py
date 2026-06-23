"""可視化: pyvis インタラクティブ HTML + plotly 時系列チャート。"""

from __future__ import annotations

from pathlib import Path

import networkx as nx

from kg_builder import GRAPHS_DIR

# エンティティ型ごとの色
ENTITY_COLORS = {
    "company": "#4472C4",
    "person": "#ED7D31",
    "segment": "#70AD47",
    "financial_metric": "#FFC000",
    "region": "#5B9BD5",
    "product": "#A5A5A5",
    "industry": "#264478",
    "event": "#C00000",
    "value": "#CCCCCC",
    "unknown": "#999999",
}


def visualize_neighborhood(
    G: nx.MultiDiGraph,
    entity_name: str,
    depth: int = 1,
    fiscal_year: str | None = None,
    output_name: str | None = None,
) -> Path:
    """エンティティの近傍を pyvis で可視化。"""
    try:
        from pyvis.network import Network
    except ImportError:
        raise ImportError("pyvis が必要です: pip install pyvis")

    # サブグラフ抽出
    if entity_name not in G:
        raise ValueError(f"Entity '{entity_name}' not found in graph")

    # BFS で近傍ノード取得
    neighbors = {entity_name}
    frontier = {entity_name}
    for _ in range(depth):
        next_frontier = set()
        for n in frontier:
            next_frontier.update(G.successors(n))
            next_frontier.update(G.predecessors(n))
        neighbors.update(next_frontier)
        frontier = next_frontier

    # サブグラフ
    subgraph = G.subgraph(neighbors)

    # pyvis ネットワーク
    net = Network(height="700px", width="100%", directed=True, notebook=False)
    net.barnes_hut(gravity=-3000, central_gravity=0.3, spring_length=200)

    # ノード追加
    for node, data in subgraph.nodes(data=True):
        et = data.get("entity_type", "unknown")
        color = ENTITY_COLORS.get(et, "#999999")
        size = 30 if node == entity_name else 20
        net.add_node(
            node,
            label=node,
            color=color,
            size=size,
            title=f"{node}\n型: {et}",
        )

    # エッジ追加
    for u, v, data in subgraph.edges(data=True):
        if fiscal_year and data.get("fiscal_year") != fiscal_year:
            continue
        relation = data.get("relation", "")
        value = data.get("value")
        unit = data.get("unit", "")
        fy = data.get("fiscal_year", "")

        label = relation
        if value is not None:
            label += f"\n{value:,.0f} {unit}" if isinstance(value, (int, float)) else f"\n{value}"
        if fy:
            label += f"\n({fy})"

        title = (
            f"関係: {relation}\n"
            f"値: {value} {unit}\n"
            f"期間: {fy}\n"
            f"出典: {data.get('source_snippet', '')}\n"
            f"信頼度: {data.get('confidence', '')}"
        )

        net.add_edge(u, v, label=label, title=title, arrows="to")

    # HTML 出力
    if not output_name:
        output_name = f"viz_{entity_name.replace(' ', '_')}"
    out_path = GRAPHS_DIR / f"{output_name}.html"
    net.save_graph(str(out_path))
    print(f"  Visualization saved to {out_path}")
    return out_path


def visualize_full_graph(
    G: nx.MultiDiGraph,
    fiscal_year: str | None = None,
    max_nodes: int = 100,
    output_name: str = "full_graph",
) -> Path:
    """グラフ全体を可視化（ノード数制限付き）。"""
    try:
        from pyvis.network import Network
    except ImportError:
        raise ImportError("pyvis が必要です: pip install pyvis")

    # ノード数が多い場合は次数上位を取る
    if G.number_of_nodes() > max_nodes:
        degrees = sorted(G.degree(), key=lambda x: x[1], reverse=True)
        top_nodes = {n for n, _ in degrees[:max_nodes]}
        subgraph = G.subgraph(top_nodes)
    else:
        subgraph = G

    net = Network(height="800px", width="100%", directed=True, notebook=False)
    net.barnes_hut(gravity=-5000, central_gravity=0.3, spring_length=250)

    for node, data in subgraph.nodes(data=True):
        et = data.get("entity_type", "unknown")
        color = ENTITY_COLORS.get(et, "#999999")
        deg = subgraph.degree(node)
        net.add_node(node, label=node, color=color, size=10 + deg * 3, title=f"{node}\n型: {et}")

    for u, v, data in subgraph.edges(data=True):
        if fiscal_year and data.get("fiscal_year") != fiscal_year:
            continue
        relation = data.get("relation", "")
        fy = data.get("fiscal_year", "")
        net.add_edge(u, v, label=f"{relation} ({fy})", arrows="to")

    out_path = GRAPHS_DIR / f"{output_name}.html"
    net.save_graph(str(out_path))
    print(f"  Full graph visualization saved to {out_path}")
    return out_path


def plot_metric_timeline(
    G: nx.MultiDiGraph,
    entity_name: str,
    relation: str = "revenue_is",
    output_name: str | None = None,
) -> Path | None:
    """特定指標の時系列チャート（plotly）。"""
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("  plotly が必要です: pip install plotly")
        return None

    points = []
    for _, tgt, data in G.out_edges(entity_name, data=True):
        if data.get("relation") != relation:
            continue
        fy = data.get("fiscal_year", "")
        val = data.get("value")
        if fy and val is not None:
            points.append((fy, float(val)))

    if not points:
        print(f"  No data for {entity_name} / {relation}")
        return None

    points.sort(key=lambda x: x[0])
    years, values = zip(*points)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(years),
        y=list(values),
        mode="lines+markers",
        name=relation,
    ))
    fig.update_layout(
        title=f"{entity_name} — {relation}",
        xaxis_title="Fiscal Year",
        yaxis_title=f"Value",
    )

    if not output_name:
        output_name = f"timeline_{entity_name}_{relation}"
    out_path = GRAPHS_DIR / f"{output_name}.html"
    fig.write_html(str(out_path))
    print(f"  Timeline chart saved to {out_path}")
    return out_path


def export_to_csv(G: nx.MultiDiGraph, output_name: str = "dkg_triples") -> Path:
    """全トリプルを CSV でエクスポート。"""
    import csv

    out_path = GRAPHS_DIR / f"{output_name}.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "subject", "relation", "object", "value", "unit",
            "fiscal_year", "confidence", "source_doc", "extraction_method",
        ])
        for u, v, data in G.edges(data=True):
            writer.writerow([
                u,
                data.get("relation", ""),
                v,
                data.get("value", ""),
                data.get("unit", ""),
                data.get("fiscal_year", ""),
                data.get("confidence", ""),
                data.get("source_doc", ""),
                data.get("extraction_method", ""),
            ])

    print(f"  Exported {G.number_of_edges()} triples to {out_path}")
    return out_path
