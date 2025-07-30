"""Test the fitted parameters on the last year: 2024.

WARNING: this will take 12 cores and 16 Gb or RAM for approx 2 hours.
"""

from multiprocessing import Pool

import geopandas as geopd
import numpy as np
import pandas as pd
import xarray
from adjustText import adjust_text
from matplotlib import axes, collections, dates, legend_handler, lines, patches
from matplotlib import pyplot as plt
from scipy import stats
from tqdm import tqdm

import base
from diffsys.models import Diffusion

ITA = base.ita()


# %%

print("Loading Graphs")
GRAPH_ADJ = base.load_graph(full=False).drop_duplicates()
GRAPH_TMP = base.load_graph(full=True)
print("Loaded Graphs")

# %%

pars = base.params()
print(pars)
peaks = base.load_peaks(2024, kind="full")

baseline = base.load_nodes()["delay_q50"].sort_index()

delay = (base.load_real_delay(2024) - baseline).clip(lower=0)
delay = delay.loc[peaks.index]
print(delay)

# %%


def simulate(pars: dict[str, float], peaks: pd.DataFrame, **kwargs):
    ef = base.load_extfield(2024)
    with Pool(10) as pool:
        cascades = list(
            # `starmap` keeps the order
            pool.starmap(
                base.sim,
                tqdm(
                    [
                        (
                            Diffusion(
                                GRAPH_ADJ,
                                ef.get(day=peak_day),
                                alpha=pars["alpha"],
                                beta=pars["beta"],
                                gamma=pars["gamma"],
                                weight="count",
                            ),
                            GRAPH_TMP,
                        )
                        for peak_day in peaks.index
                    ],
                    total=len(peaks),
                    dynamic_ncols=True,
                ),
                chunksize=1,
            )
        )

    return pd.concat(cascades)


delays = simulate(pars, peaks)
delays["time"] = pd.DatetimeIndex(delays["time"])


# %%

from time import time

base.LOC_CACHE = {}
ef = base.load_extfield(2024)

a = time()
tmp = base.sim(
    Diffusion(
        GRAPH_ADJ,
        ef.get(day="2024-09-09"),
        alpha=pars["alpha"],
        beta=pars["beta"],
        gamma=pars["gamma"],
        weight="count",
    ),
    GRAPH_TMP,
)
print(time() - a)
print(tmp)
print(tmp.value.sum())

# %%


def _exp_ax(ax: np.ndarray, expand_axes: float = 0.2):
    if len(ax) == 1:
        val = ax[0]
        return np.array([val - val * expand_axes, val + val * expand_axes])

    diff = ax[-1] - ax[0]
    x0, x1 = ax[0], ax[-1]
    ax = ax[:-1] + (ax[1:] - ax[:-1]) / 2
    return np.asarray([x0 - diff * expand_axes] + list(ax) + [x1 + diff * expand_axes])


def add_backgroundgrad(
    ax: axes.Axes,
    mat: np.ndarray,
    xy: tuple[np.ndarray, np.ndarray],
    expand_axes=(0.2, 0.2),
    **kwargs,
):
    x = _exp_ax(xy[0], expand_axes=expand_axes[0])
    y = _exp_ax(xy[1], expand_axes=expand_axes[1])
    ax.pcolormesh(x, y, mat, rasterized=True, **kwargs)


class GradHandler(legend_handler.HandlerBase):
    """This is needed to add a rectangle with a gradient in the legend."""

    def __init__(self, cmap, num_stripes=8, **kw):
        legend_handler.HandlerBase.__init__(self, **kw)
        self.cmap = cmap
        self.num_stripes = num_stripes

    def create_artists(
        self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans
    ):
        stripes = []
        for i in range(self.num_stripes):
            s = patches.Rectangle(
                (xdescent + i * width / self.num_stripes, ydescent),
                width / self.num_stripes,
                height,
                fc=self.cmap((2 * i + 1) / (2 * self.num_stripes)),
                ec=self.cmap((2 * i + 1) / (2 * self.num_stripes)),
                transform=trans,
                lw=1,
            )
            stripes.append(s)
        return stripes


def plot_all_days(results: pd.DataFrame):
    pd.plotting.register_matplotlib_converters()
    fig = plt.figure(figsize=(10, 4))
    ax_scat = fig.subplots(
        gridspec_kw={"top": 0.85, "bottom": 0.15, "right": 0.98, "left": 0.6}
    )
    ax_rep, ax_rain = fig.subplots(
        nrows=2,
        gridspec_kw={
            "top": 0.85,
            "bottom": 0.15,
            "right": 0.58,
            "left": 0.08,
            "hspace": 0,
        },
        sharex=True,
        height_ratios=[5, 1],
    )

    ax_rep.fill_between(
        results.index,
        results["real"] / 60,
        color="C8",
        label="Reported excess",
        lw=1,
        alpha=0.8,
    )
    ax_rep.fill_between(
        results.index,
        results["value"] / 60,
        color="C2",
        ls="solid",
        label="Predicted",
        alpha=0.8,
        lw=1,
    )
    add_backgroundgrad(
        ax_rain,
        results["rain"].to_numpy().reshape((1, -1)),
        (results.index.to_numpy(), np.asarray([-400])),
        expand_axes=[0.02, 1],
        cmap="RdBu",
        vmin=200,
        vmax=800,
        alpha=0.8,
    )
    ax_rain.set(yticks=[], xlabel="Time")

    handles, labels = ax_rep.get_legend_handles_labels()
    t = np.column_stack([np.linspace(0, 1, 10), np.zeros(10)])
    lc = collections.LineCollection([t])
    ax_rep.add_collection(lc)
    handles.append(lc)
    labels.append("Rain")
    ax_rep.legend(
        handles=handles,
        labels=labels,
        fontsize="small",
        handler_map={lc: GradHandler(plt.get_cmap("RdBu"), num_stripes=50)},
    )
    ax_rep.set(ylabel="Delay (hours)", ylim=(0, 3400))
    ax_rep.xaxis.set_major_formatter(dates.DateFormatter("%b"))
    # fig.align_labels([ax_rain, ax_rep])

    corr = stats.spearmanr(
        results[results["rain"] > 500]["value"], results[results["rain"] > 500]["real"]
    )
    scttr = ax_scat.scatter(
        results["value"] / 60,
        results["real"] / 60,
        c=results["rain"],
        s=results["rain"] / 3 + 10,
        alpha=0.5,
        cmap="RdBu",
        lw=0.1,
        edgecolors="k",
        vmin=200,
        vmax=800,
    )
    ax_scat.set(
        xlabel="Predicted delay (hours)",
        title="Cumulative daily delay",
        ylabel="Reported excess of delay (hours)",
        # aspect=1,
        # xlim=(-200, 2800),
        ylim=(-200, 3500),
    )
    ax_scat.annotate(
        f"Spearman: {corr.statistic:3.2f}\np-value: {str(corr.pvalue)[:5] if corr.pvalue > 0.01 else '<0.01'}",  # type: ignore
        (0.95, 0.95),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize="small",
        color="#666666",
    )
    l1 = ax_scat.legend(
        *scttr.legend_elements(num=5),
        markerscale=2,
        fontsize="small",
        handletextpad=0,
        loc=(0.1, 0.6),
    )
    l1 = ax_scat.legend(
        handles=[
            lines.Line2D(
                [0],
                [0],
                marker="o",
                markersize=np.sqrt(x / 3 + 10),
                lw=0,
                color=scttr.cmap(x / results["rain"].max()),
                label=f"{x} mm",
                alpha=0.5,
                mew=0.1,
                mec="k",
            )
            for x in range(200, 1000, 200)
        ],
        fontsize="x-small",
        ncols=1,
        handletextpad=0,
        title="Rain",
        bbox_to_anchor=(1.02, 0.5),
        loc="center left",
        borderaxespad=0.0,
    )
    ax_scat.add_artist(l1)

    text = [
        ax_scat.annotate(
            str(day)[:10],
            (p["value"] / 60, p["real"] / 60),
            fontsize="xx-small",
            color="#999999",
        )
        for day, p in results.iterrows()
        if p["rain"] > 500 and p["value"] > 500 * 60
    ]
    adjust_text(
        text,
        objects=scttr,
        arrowprops=dict(arrowstyle="->", color="grey", alpha=0.3),
        ax=ax_scat,
        max_move=(100, 100),
    )
    base.add_axis_label(ax_rep, "a")
    base.add_axis_label(ax_scat, "b ")
    ax_rep.grid(False)

    fig.savefig(base.PLOTS / "validation_2024.pdf")
    fig.savefig(base.PLOTS / "validation_2024.png")


daily_d = (
    delays.drop(columns=["node"])
    .set_index("time", drop=True)
    .groupby(pd.Grouper(freq="1D"))
    .sum()
    .dropna(axis=0, how="all")
)

res = pd.concat([daily_d, delay.sum(1).rename("real"), peaks], axis=1).fillna(0)
plot_all_days(res)
print(res)
print(res[["real", "value"]].max() / 60)
# %%


def plot_multi_days(isodays: list[str]):
    """"""
    fig = plt.figure(figsize=(12, 4.4 * len(isodays)))

    axss = fig.subplots(
        ncols=3,
        nrows=len(isodays),
        sharey=True,
        sharex=True,
        gridspec_kw={
            "wspace": 0,
            "hspace": 0.07,
            "left": 0.05,
            "right": 0.6,
            "bottom": 0.1,
            "top": 0.95,
        },
    )
    axs = fig.subplots(
        nrows=len(isodays),
        gridspec_kw={
            "left": 0.67,
            "right": 0.99,
            "top": 0.95,
            "bottom": 0.1,
            "hspace": 0.07,
        },
        sharex=True,
        sharey=True,
    )

    for _axss, _axs, isoday in zip(axss, axs, isodays):
        _plot_one_day(isoday, *_axss, _axs)
        _axss[2].set(xlabel="", ylabel="")
        _axss[0].set_ylabel(isoday, fontsize="large")

    axss[0, 0].set(title="Real data")
    axss[0, 1].set(title="Prediction")
    axss[0, 2].set(title="Stressor field")
    axs[0].set(ylabel="Reported excess delay (mins)")
    axs[1].set(xlabel="Predicted delay (mins)", ylabel="Reported excess delay (mins)")

    base.add_axis_label(axss[0, 0], "a")
    base.add_axis_label(axs[0], "b")

    title = "_".join(isodays)
    fig.savefig(base.PLOTS / f"validation_2024_oneday_{title}.pdf")
    fig.savefig(base.PLOTS / f"validation_2024_oneday_{title}.png")
    fig.savefig(base.PLOTS / f"validation_2024_oneday_{title}.svg")


def _plot_one_day(isoday: str, ax_real, ax_pred, ax_stress, ax_scat):
    day, raindata = _get_one_day(isoday)

    kwargs = dict(
        rasterized=False,
        cmap="RdBu",
        vmin=0.0,
        vmax=0.5,
        lw=0.1,
        edgecolor="k",
        alpha=0.5,
    )
    day = geopd.GeoDataFrame(
        day.sort_values("rain"), geometry=GRAPH_ADJ.nodes().loc[day.index, "geometry"]
    )

    raindata.plot.pcolormesh(
        ax=ax_stress, add_colorbar=False, lw=0, rasterized=True, vmax=1, cmap="Blues"
    )
    for ax in [ax_real, ax_pred, ax_stress]:
        ITA.plot(
            ax=ax,
            lw=0.3,
            edgecolor="k",
            facecolor="none",
            aspect=None,
            rasterized=False,
        )

    ax_real.scatter(
        day["geometry"].x,
        day["geometry"].y,
        s=lin_map(day["real"], (0, 100), (0, 30)),
        c=day["rain"],
        **kwargs,
    )
    ax_pred.scatter(
        day["geometry"].x,
        day["geometry"].y,
        s=lin_map(day["value"], (0, 200), (0, 30)) if day["value"].max() > 0 else 0,
        c=day["rain"],
        **kwargs,
    )

    points = ax_scat.scatter(
        day["value"],
        day["real"],
        c=day["rain"],
        s=[GRAPH_ADJ.nodes().loc[s, "capacity"] * 2 for s in day.index],  # type: ignore
        **kwargs,
    )
    spear = stats.spearmanr(
        day["value"][day["value"] > 0], day["real"][day["value"] > 0]
    )
    ax_scat.annotate(
        f"Spearman: {spear.statistic:3.2f}\np-value: {str(spear.pvalue)[:5] if spear.pvalue > 0.01 else '<0.01'}",  # type: ignore
        (0.95, 0.05),
        xycoords="axes fraction",
        ha="right",
        va="bottom",
    )
    text = [
        ax_scat.text(
            d["value"], d["real"], base.shorten_name(str(node)), fontsize="small"
        )
        for node, d in day.iterrows()
        if d["value"] > day["value"].quantile(0.998)
        or d["real"] > day["real"].quantile(0.998)
    ]
    ax_scat.set(
        xscale="symlog",
        yscale="symlog",
        aspect=1,
        xlim=(-0.4, 2e5),
        ylim=(-0.4, 9e4),
    )
    adjust_text(
        text,
        prevent_crossings=True,
        objects=points,
        force_text=(0.5, 1.5),
        force_pull=(0.01, 0.001),
        max_move=(100, 100),
        arrowprops=dict(arrowstyle="->", color="grey", alpha=0.3),
        ax=ax_scat,
        expand_axes=True,
    )


def plot_one_day(day: pd.DataFrame, raindata: xarray.DataArray, title: str = "One day"):
    fig = plt.figure(figsize=(12, 4.4))

    ax_real, ax_pred, ax0 = fig.subplots(
        ncols=3,
        sharey=True,
        sharex=True,
        gridspec_kw={"wspace": 0, "left": 0.05, "right": 0.6, "bottom": 0.15},
    )
    ax1 = fig.subplots(gridspec_kw={"left": 0.67, "right": 0.99, "bottom": 0.15})

    _plot_one_day(title, ax_real, ax_pred, ax0, ax1)
    ax_real.set_title("Real data")
    ax_pred.set_title("Prediction")
    ax0.set(title="Stressor field", xlabel="", ylabel="")

    ax1.set(
        title=title,
        ylabel="Reported excess delay (mins)",
        xlabel="Predicted delay (mins)",
    )

    fig.savefig(base.PLOTS / f"validation_2024_oneday_{title}.pdf")
    fig.savefig(base.PLOTS / f"validation_2024_oneday_{title}.png")
    fig.savefig(base.PLOTS / f"validation_2024_oneday_{title}.svg")


def lin_map(vals: np.ndarray | pd.Series, p1: tuple, p2: tuple) -> np.ndarray:
    """Linear map from vals to the line betwenn p1 and p2.

    p1: two points in the domain space
    p2: two points in the codomain space
    """
    vals = vals.to_numpy() if isinstance(vals, pd.Series) else vals
    return p2[0] + (vals - p1[0]) * (p2[1] - p2[0]) / (p1[1] - p1[0])


def main() -> None:
    """Do the main."""
    # daily_d = (
    #     delays.drop(columns=["failing", "node"])
    #     .set_index("time", drop=True)
    #     .groupby(pd.Grouper(freq="1D"))
    #     .sum()
    #     .dropna(axis=0, how="all")
    # )

    # res = pd.concat([daily_d, delay.sum(1).rename("real"), peaks], axis=1).fillna(0)
    # plot_all_days(res)

    plot_multi_days(["2024-10-03", "2024-09-09"])
    for isoday in [
        "2024-02-27",
        "2024-03-10",
        "2024-03-27",
        "2024-04-01",
        "2024-09-05",
        "2024-09-09",
        "2024-10-03",
        "2024-10-08",
        "2024-10-18",
        "2024-10-19",
        "2024-12-08",
    ]:
        plot_one_day(*_get_one_day(isoday=isoday), isoday)


def _get_one_day(isoday: str) -> tuple[pd.DataFrame, xarray.DataArray]:
    print(isoday)
    try:
        daily_d = (
            delays.drop(columns=["failing"])
            .set_index("time", drop=True)
            .groupby([pd.Grouper(freq="1D"), "node"])
            .sum()
            .dropna(axis=0, how="all")
            .loc[isoday]
        )
    except KeyError:
        daily_d = pd.DataFrame([], columns=["value"])
    raindata = base.load_extfield(2024).data.sel(time=isoday)
    raindata = raindata.sum(dim="time")
    nodes = base.load_nodes()
    daily_d["rain"] = [
        raindata.sel(
            latitude=nodes.loc[st, "geometry"].y,
            longitude=nodes.loc[st, "geometry"].x,
            method="nearest",
        ).data
        for st in daily_d.index
    ]

    return pd.concat([daily_d, delay.loc[isoday].rename("real")], axis=1).fillna(
        0
    ), raindata


if __name__ == "__main__":
    main()

# %%
