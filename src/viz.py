"""Plotting helpers. Every function returns the figure and saves a PNG into figures/."""
from __future__ import annotations

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
from matplotlib.ticker import FuncFormatter

from . import config
from .vsm import Interactions

PALETTE = {"user": "#4C78A8", "item": "#F58518", "focus": "#E45756", "second": "#54A24B", "grey": "#9AA0A6"}
_thousands = FuncFormatter(lambda x, _: f"{x:,.0f}")


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 110, "savefig.dpi": 160, "axes.spines.top": False, "axes.spines.right": False,
            "axes.grid": True, "grid.alpha": 0.25, "axes.titleweight": "bold", "axes.titlesize": 11,
            "axes.labelsize": 9.5, "legend.frameon": False,
        }
    )


def _save(fig: plt.Figure, name: str) -> plt.Figure:
    config.FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.FIG_DIR / f"{name}.png", bbox_inches="tight")
    return fig


def plot_timeline(events: pd.DataFrame, p1: tuple[str, str], p2: tuple[str, str]) -> plt.Figure:
    _style()
    daily = events.set_index("datetime").groupby([pd.Grouper(freq="D"), "event"], observed=True).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.plot(daily.index, daily["view"], color=PALETTE["user"], lw=1.4, label="view")
    ax.set_ylabel("views per day")
    ax2 = ax.twinx()
    ax2.grid(False)
    ax2.spines["right"].set_visible(True)
    for name, colour in (("addtocart", PALETTE["item"]), ("transaction", PALETTE["focus"])):
        ax2.plot(daily.index, daily[name], color=colour, lw=1.2, label=name)
    ax2.set_ylabel("cart adds / transactions per day")
    for (start, end), label, colour in ((p1, "P1 training", PALETTE["second"]), (p2, "P2 testing", PALETTE["focus"])):
        ax.axvspan(pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC"), color=colour, alpha=0.13)
        ax.text(pd.Timestamp(start, tz="UTC"), ax.get_ylim()[1] * 0.97, f" {label}", va="top", fontsize=9, color=colour, fontweight="bold")
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3)
    ax.set_title("Daily activity and the two chosen periods")
    return _save(fig, "01_timeline")


def plot_event_mix(events: pd.DataFrame) -> plt.Figure:
    _style()
    counts = events["event"].value_counts().reindex(["view", "addtocart", "transaction"])
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    bars = ax.bar(counts.index, counts.values, color=[PALETTE["user"], PALETTE["item"], PALETTE["focus"]])
    ax.set_yscale("log")
    ax.set_title("Event types (log scale)")
    ax.set_ylabel("events")
    for bar, value in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:,}", ha="center", va="bottom", fontsize=9)
    return _save(fig, "02_event_mix")


def plot_ccdf(events: pd.DataFrame) -> plt.Figure:
    """Complementary CDF of activity: how heavy is the tail of visitors and of items?"""
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, column, colour, title in (
        (axes[0], "visitorid", PALETTE["user"], "Events per visitor"),
        (axes[1], "itemid", PALETTE["item"], "Events per item"),
    ):
        values = np.sort(events.groupby(column).size().to_numpy())
        share = 1 - np.arange(len(values)) / len(values)
        ax.loglog(values, share, color=colour, lw=1.8)
        ax.set_xlabel("events (log)")
        ax.set_ylabel("share of entities with at least this many (log)")
        ax.set_title(title)
    return _save(fig, "03_long_tail")


def plot_composition(nodes: pd.DataFrame, rels: pd.DataFrame) -> plt.Figure:
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, frame, col, colours, title in (
        (axes[0], nodes, "label", [PALETTE["user"], PALETTE["item"]], "Nodes by label"),
        (axes[1], rels, "relationship", [PALETTE["user"], PALETTE["item"], PALETTE["focus"]], "Relationships by type"),
    ):
        bars = ax.bar(frame[col], frame["total"], color=colours[: len(frame)])
        ax.yaxis.set_major_formatter(_thousands)
        ax.set_title(title)
        for bar, value in zip(bars, frame["total"]):
            ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:,}", ha="center", va="bottom", fontsize=9)
    return _save(fig, "04_composition")


def plot_density_block(matrix: sp.csr_matrix, size: int = 300) -> plt.Figure:
    """Sparsity pattern of the `size` busiest users x `size` most popular items."""
    _style()
    user_order = np.argsort(-np.diff(matrix.indptr))[:size]
    item_order = np.argsort(-np.diff(matrix.tocsc().indptr))[:size]
    block = matrix[user_order][:, item_order]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].spy(block, markersize=1.2, color=PALETTE["user"])
    axes[0].set_title(f"Busiest {size} users x most popular {size} items")
    axes[0].set_xlabel("items (most popular first)")
    axes[0].set_ylabel("users (most active first)")
    axes[0].grid(False)
    sample = matrix[np.random.default_rng(config.SEED).choice(matrix.shape[0], size=min(size, matrix.shape[0]), replace=False)]
    sample = sample[:, np.argsort(-np.diff(sample.tocsc().indptr))[:size]]
    axes[1].spy(sample, markersize=1.2, color=PALETTE["item"])
    axes[1].set_title(f"{sample.shape[0]} random users x {sample.shape[1]} busiest columns")
    axes[1].set_xlabel("items")
    axes[1].grid(False)
    return _save(fig, "05_density_block")


def plot_degree_distributions(inter: Interactions) -> plt.Figure:
    _style()
    user_deg = np.diff(inter.matrix.indptr)
    item_deg = np.diff(inter.matrix.tocsc().indptr)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, deg, colour, title, xlabel in (
        (axes[0], user_deg, PALETTE["user"], "Distinct items per user (row length)", "distinct items"),
        (axes[1], item_deg, PALETTE["item"], "Distinct users per item (row length)", "distinct users"),
    ):
        bins = np.unique(np.logspace(0, np.log10(max(deg.max(), 2)), 40).astype(int))
        ax.hist(deg, bins=bins, color=colour)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.axvline(np.median(deg), color="black", ls="--", lw=1)
        ax.text(np.median(deg), ax.get_ylim()[1], f" median = {np.median(deg):.0f}", va="top", fontsize=9)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("count (log)")
    return _save(fig, "06_degree_distributions")


def ego_graph(inter: Interactions, space: str, focus: int, max_first: int = 8, max_second: int = 10) -> nx.Graph:
    """Focus entity, its direct neighbours (other side) and the 2-hop entities on its own side.

    For space='user' the 2-hop users are exactly the neighbours in the user-user projection.
    """
    matrix = inter.matrix if space == "user" else inter.matrix.T.tocsr()
    csc = matrix.T.tocsr()
    own_ids = inter.user_ids if space == "user" else inter.item_ids
    other_ids = inter.item_ids if space == "user" else inter.user_ids
    first = matrix[focus].indices
    first = first[np.argsort(-matrix[focus].data, kind="stable")][:max_first]
    shared = np.asarray((matrix[:, first] > 0).sum(axis=1)).ravel()
    shared[focus] = 0
    second = np.argsort(-shared, kind="stable")[:max_second]
    second = second[shared[second] > 0]

    graph = nx.Graph()
    own, other = ("U", "I") if space == "user" else ("I", "U")
    graph.add_node(f"{own}{own_ids[focus]}", role="focus")
    for col in first:
        graph.add_node(f"{other}{other_ids[col]}", role="first")
        graph.add_edge(f"{own}{own_ids[focus]}", f"{other}{other_ids[col]}", weight=matrix[focus, col])
    for row in second:
        graph.add_node(f"{own}{own_ids[row]}", role="second")
        for col in first:
            if matrix[row, col] > 0:
                graph.add_edge(f"{own}{own_ids[row]}", f"{other}{other_ids[col]}", weight=matrix[row, col])
    return graph


def plot_ego(graph: nx.Graph, space: str, title: str, name: str) -> plt.Figure:
    _style()
    own_name, other_name = ("user", "item") if space == "user" else ("item", "user")
    colours = {"focus": PALETTE["focus"], "first": PALETTE[other_name], "second": PALETTE[own_name]}
    sizes = {"focus": 900, "first": 520, "second": 520}
    pos = nx.spring_layout(graph, seed=config.SEED32, k=0.9)
    fig, ax = plt.subplots(figsize=(8.2, 6))
    weights = np.array([graph[u][v]["weight"] for u, v in graph.edges])
    nx.draw_networkx_edges(graph, pos, ax=ax, width=0.8 + 2.2 * weights / weights.max(), edge_color=PALETTE["grey"], alpha=0.8)
    for role in colours:
        nodes = [n for n, d in graph.nodes(data=True) if d["role"] == role]
        nx.draw_networkx_nodes(graph, pos, nodelist=nodes, node_color=colours[role], node_size=sizes[role], ax=ax, edgecolors="white", linewidths=1.2)
    nx.draw_networkx_labels(graph, pos, font_size=6.5, font_color="white", ax=ax)
    ax.axis("off")
    ax.set_title(title)
    handles = [
        plt.Line2D([], [], marker="o", ls="", color=colours["focus"], label=f"focus {own_name}", markersize=9),
        plt.Line2D([], [], marker="o", ls="", color=colours["first"], label=f"{other_name}s it touched", markersize=9),
        plt.Line2D([], [], marker="o", ls="", color=colours["second"], label=f"other {own_name}s sharing them", markersize=9),
    ]
    ax.legend(handles=handles, loc="lower left")
    return _save(fig, name)


def plot_before_after(edges_before: pd.DataFrame, pairs_after: pd.DataFrame, space: str, ref: int) -> plt.Figure:
    """Left: raw interaction edges around the reference node. Right: SIMILAR_TO edges with their scores."""
    _style()
    own, other = ("Visitor", "Item") if space == "user" else ("Item", "Visitor")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))

    before = nx.Graph()
    centre = f"{own}\n{ref}"
    for _, row in edges_before.iterrows():
        before.add_edge(centre, f"{other}\n{row['other']}", rel=row["rel"])
    pos = nx.spring_layout(before, seed=config.SEED32)
    nx.draw_networkx_nodes(before, pos, nodelist=[centre], node_color=PALETTE["focus"], node_size=1100, ax=axes[0])
    others = [n for n in before.nodes if n != centre]
    nx.draw_networkx_nodes(before, pos, nodelist=others, node_color=PALETTE["user" if other == "Visitor" else "item"], node_size=420, ax=axes[0])
    nx.draw_networkx_edges(before, pos, edge_color=PALETTE["grey"], ax=axes[0])
    nx.draw_networkx_labels(before, pos, font_size=6, ax=axes[0])
    axes[0].set_title(f"BEFORE: {len(others)} raw interaction edges, no similarity relationship")

    after = nx.DiGraph()
    centre = str(ref)
    for _, row in pairs_after.iterrows():
        after.add_edge(centre, str(int(row["target"])), score=row["score"])
    pos = nx.spring_layout(after, seed=config.SEED32)
    colour = PALETTE["user" if space == "user" else "item"]
    nx.draw_networkx_nodes(after, pos, nodelist=[centre], node_color=PALETTE["focus"], node_size=1100, ax=axes[1])
    nx.draw_networkx_nodes(after, pos, nodelist=[n for n in after.nodes if n != centre], node_color=colour, node_size=700, ax=axes[1])
    nx.draw_networkx_edges(after, pos, edge_color=PALETTE["second"], width=2.2, arrows=True, arrowsize=16, node_size=700, ax=axes[1])
    nx.draw_networkx_labels(after, pos, font_size=7, ax=axes[1])
    nx.draw_networkx_edge_labels(after, pos, edge_labels={(u, v): f"{d['score']:.3f}" for u, v, d in after.edges(data=True)}, font_size=8, ax=axes[1])
    axes[1].set_title("AFTER: top-5 SIMILAR_TO edges with cosine score")
    for ax in axes:
        ax.axis("off")
    return _save(fig, "08_before_after_similarity")


def plot_metric_comparison(summary: pd.DataFrame, n: int = 5) -> plt.Figure:
    """summary: index = method, columns = hit@n, ndcg@n and their CI bounds."""
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    colours = [PALETTE["grey"], PALETTE["user"], PALETTE["item"]]
    for ax, metric in zip(axes, (f"hit@{n}", f"ndcg@{n}")):
        mean = summary[metric]
        err = np.vstack([mean - summary[f"{metric}_lo"], summary[f"{metric}_hi"] - mean])
        ax.bar(summary.index, mean, yerr=err, capsize=4, color=colours[: len(summary)])
        ax.set_title(f"{metric.upper()} (mean and 95% bootstrap CI)")
        for x, value in enumerate(mean):
            ax.text(x, value, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
        ax.tick_params(axis="x", labelsize=8.5)
    return _save(fig, "09_metric_comparison")
