"""Create the geofile with data of coverage."""

import geopandas as geopd
import h3pandas
import networkx as nx
import numpy as np
import pandas as pd
from adjustText import adjust_text
from matplotlib import axes
from matplotlib import pyplot as plt
from scipy import stats

import base

ITA = base.ita()
GRAPH = base.load_graph(full=False).drop_duplicates()


def lin_map(vals: float | np.ndarray | pd.Series, p1: tuple, p2: tuple) -> np.ndarray:
    """Linear map from vals to the line betwenn p1 and p2.

    p1: two points in the domain space
    p2: two points in the codomain space
    """
    vals = vals.to_numpy() if isinstance(vals, pd.Series) else vals
    return p2[0] + (vals - p1[0]) * (p2[1] - p2[0]) / (p1[1] - p1[0])


def prepare_text(text: str) -> str:
    parts = text.split()
    if parts[0] in {"san"}:
        return (
            " ".join(parts[:2]).title()
            + " "
            + "".join([s[0].title() for s in parts[2:]])
        )
    return parts[0].title() + " " + "".join([s[0].title() for s in parts[1:]])


def plot_geo_perturbation(coverage: geopd.GeoDataFrame, name: str) -> None:
    """Plot."""
    labelmap = {
        "stressor": "Local Stressor",
        "risk_stressor": "Local Stressor",
    }
    rname = "risk_" + name

    print(f"Full {name}")
    fig = plt.figure(figsize=(12, 10))
    common = {"top": 0.95, "bottom": 0.07, "wspace": 0, "hspace": 0.15}
    map_axs = fig.subplots(
        ncols=2,
        nrows=2,
        sharey=True,
        sharex=True,
        gridspec_kw={"left": 0.05, "right": 0.6, **common},
    )
    sc_axs = fig.subplots(
        nrows=2, sharex=True, gridspec_kw={"left": 0.67, "right": 0.95, **common}
    )
    base.add_axis_label(map_axs[0, 0], "a")
    base.add_axis_label(sc_axs[0], "b")
    base.add_axis_label(map_axs[1, 0], "c")
    base.add_axis_label(sc_axs[1], "d")
    _plot_geo_perturbation(
        coverage=coverage,
        name=name,
        ax_scatter=sc_axs[0],
        ax_map=map_axs[0, 0],
        ax_aggr=map_axs[0, 1],
    )
    _plot_geo_perturbation(
        coverage=coverage,
        name=rname,
        ax_scatter=sc_axs[1],
        ax_map=map_axs[1, 0],
        ax_aggr=map_axs[1, 1],
    )
    sc_axs[0].set_xlabel("")
    fig.savefig(base.PLOTS / f"coverage_{name}_combined.pdf")
    fig.savefig(base.PLOTS / f"coverage_{name}_combined.png", dpi=300)


def _plot_geo_perturbation(
    coverage: geopd.GeoDataFrame,
    name: str,
    ax_scatter: axes.Axes,
    ax_map: axes.Axes,
    ax_aggr: axes.Axes,
) -> None:
    """Plot."""
    # Prepare data
    measure = geopd.GeoDataFrame(
        coverage[
            coverage.columns.intersection(
                [
                    name,
                    "node_degree",
                    "node_betweenness",
                    "capacity",
                    "geometry",
                    "ego1",
                    "ego2",
                ]
            )
        ]
        .dropna(subset=name)
        .sort_values(by=name),
        geometry="geometry",
    )
    print(stats.spearmanr(measure[name].fillna(0), measure["node_degree"]))
    print(stats.spearmanr(measure[name].fillna(0), measure["node_betweenness"]))
    qlow, qhigh = np.quantile(measure[name], [0.01, 0.99])

    measure.h3.geo_to_h3_aggregate(4, operation="sum").plot(
        column=name, ax=ax_aggr, cmap="Reds", aspect=None
    )

    # Prepare basemap
    for ax in (ax_aggr, ax_map):
        ITA.plot(ax=ax, facecolor="none", lw=0.1, edgecolor="k", aspect=None)
        GRAPH.edges().plot(
            ax=ax, lw=0.1, color="C1", alpha=0.2, rasterized=True, aspect=None
        )

    points = ax_map.scatter(
        measure["geometry"].x,
        measure["geometry"].y,
        s=lin_map(measure[name], (qlow, qhigh), (1, 100)),
        c=measure[name],
        cmap="Reds",
        vmin=qlow,
        vmax=qhigh,
        lw=0.2,
        edgecolor="k",
    )

    text = [
        ax_map.annotate(
            prepare_text(str(stname)),
            (st["geometry"].x, st["geometry"].y),
            fontsize="x-small",
            color="#444444",
        )
        for stname, st in measure.sort_values(by=name).tail(10).iterrows()
    ]
    adjust_text(
        text,
        objects=points,
        prevent_crossings=True,
        force_text=(0.9, 1.5),
        force_pull=(0.001, 0.001),
        max_move=(1000, 1000),
        arrowprops=dict(arrowstyle="->", color="C5", alpha=0.5),
        ax=ax_map,
    )

    sc = ax_scatter.scatter(
        measure["capacity"],
        measure[name],
        s=lin_map(measure["node_degree"], (0, 10), (10, 50)),
        c=measure["node_betweenness"],
        cmap="PRGn",
        alpha=0.3,
        edgecolor="k",
        lw=0.2,
    )
    ax_scatter.set(
        xlabel="Capacity",
        ylabel="Social risk" if "risk" in name else "Perturbability",
    )

    ax_map.text(
        1,
        1,
        "Social risk" if "risk" in name else "Perturbability",
        transform=ax_map.transAxes,
        va="bottom",
        ha="center",
        fontsize="large",
    )
    hdl1, lbl1 = sc.legend_elements(prop="colors", num=3)
    ax_scatter.legend(hdl1, lbl1, title="Betweenness", fontsize="x-small")

    to_annotate = pd.concat(
        [measure.sort_values("capacity").tail(7), measure.sort_values(name).tail(8)]
    ).drop_duplicates()
    text2 = [
        ax_scatter.annotate(
            prepare_text(str(stname)), (st["capacity"], st[name]), fontsize="x-small"
        )
        for stname, st in to_annotate.iterrows()
    ]
    adjust_text(
        text2,
        objects=sc,
        prevent_crossings=True,
        force_text=(0.9, 1.5),
        force_pull=(0.001, 0.001),
        max_move=(100, 100),
        arrowprops=dict(arrowstyle="->", color="C5", alpha=0.5),
        ax=ax_scatter,
    )


def load_data(
    degree: bool | None = None, betweenness: bool | None = None
) -> geopd.GeoDataFrame:
    print("Loading data")
    coverage = pd.read_csv(
        base.CACHE / "coverage_data.csv.gz", index_col=["time", "station"]
    )
    coverage = coverage.loc["2024-01-01T00:00:00"]
    print("Converting to graph")
    graph = GRAPH
    print("To networkx")
    nxg = graph.to_networkx()
    nodes = graph.nodes()

    nodes[coverage.columns] = coverage

    if degree:
        print("Degree")
        nodes["node_degree"] = [
            len([e for e in graph.edges().index if s in e]) for s in nodes.index
        ]
        nodes["ego1"] = [
            nx.ego_graph(nxg, s, radius=1).number_of_edges() for s in nodes.index
        ]
        nodes["ego2"] = [
            nx.ego_graph(nxg, s, radius=2).number_of_edges() for s in nodes.index
        ]

    print(nodes)
    if betweenness:
        print("Betweenness")
        bet = nx.betweenness_centrality(nxg, weight="count")
        nodes["node_betweenness"] = [bet[n] for n in nodes.index]
    return nodes


def main() -> None:
    """Do the main."""
    nodes = load_data(degree=True, betweenness=True)

    geofile = base.CACHE / "coverage_data.geojson"
    if not geofile.is_file():
        nodes.to_file(geofile)

    for stress in ["stressor"]:
        nodes[f"risk_{stress}"] = nodes[f"{stress}_risk"]

    for stress in ["stressor"]:
        plot_geo_perturbation(nodes, stress)


if __name__ == "__main__":
    main()
