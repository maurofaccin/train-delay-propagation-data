"""Test the fitted parameters on the last year: 2024.
Plot predicted delays in various ways.
"""

from functools import partial
from pathlib import Path

import geopandas as geopd
import numpy as np
import pandas as pd
import xarray as xr
from adjustText import adjust_text
from matplotlib import axes, collections, dates, legend_handler, lines, patches
from matplotlib import patheffects as pe
from matplotlib import pyplot as plt
from scipy import stats
from tqdm.contrib.concurrent import process_map

import base
from diffsys.models import Diffusion

# %%


def simulate(pars: dict[str, float], peaks: pd.DataFrame, **kwargs) -> pd.DataFrame:
    ef = base.load_extfield(2024)
    print("Loading Graphs")
    graph_tmp = base.load_graph(full=True)
    graph_adj = base.load_graph(full=False).drop_duplicates()
    print("Loaded Graphs")

    print(f"Using:: α={pars['alpha']}, β={pars['beta']}, γ={pars['gamma']}")
    func = partial(base.sim, full_graph=graph_tmp, usecache=True)
    cascades = process_map(
        func,
        [
            Diffusion(
                graph_adj,
                ef.get(day=peak_day),
                alpha=pars["alpha"],
                beta=pars["beta"],
                gamma=pars["gamma"],
                weight="count",
            )
            for peak_day in peaks.index
        ],
        total=len(peaks),
        max_workers=10,
        chunksize=1,
    )

    return pd.concat(cascades)


# %%


class Delays:
    def __init__(self):
        pass

    def prepare(self):
        # Load parameters
        self.pars = base.params(
            sorted(Path("./cache/optimize_pars").glob("fitted_params_KFO*.jsonl")), df="kfold"
        )
        assert isinstance(self.pars, pd.DataFrame)
        self.pars = pd.concat(
            [self.pars, self.pars.median(axis=0).rename("stats").to_frame().T], axis=0
        )

        # Load rain
        self.peaks = base.load_peaks(2024, kind="full") * 1000

        # Load real delays
        self.real_delay = base.load_real_delay(2024) / 60
        self.baseline = self.real_delay.median().sort_index()

        delays = {
            k: simulate(
                self.pars.loc[k].to_dict(),
                self.peaks,
                cachename=base.CACHE / f"job_fit_params_validate_sims_KFO_{k}.csv.gz",
            )
            for k in self.pars.index.tolist()
        }
        for k in delays:
            delays[k]["time"] = pd.to_datetime(delays[k]["time"])
        self.delays = (
            pd.concat(
                [v.set_index(["time", "node"])["value"].rename(k) for k, v in delays.items()],
                axis=1,
            )
            .fillna(0.0)
            .groupby([pd.Grouper(level="time", freq="1d"), pd.Grouper(level="node")])
            .sum()
            / 60
        )

        self.nodes = base.load_nodes()
        self.raindata = base.load_extfield(2024).data * 1000

    def write(self, basedir: Path):
        basedir.mkdir(parents=True, exist_ok=True)
        self.pars.to_csv(basedir / "pars.csv.gz")
        self.peaks.to_csv(basedir / "peaks.csv.gz")
        self.real_delay.to_csv(basedir / "real_delay.csv.gz")
        self.baseline.to_csv(basedir / "baseline.csv.gz")
        self.delays.to_csv(basedir / "delays.csv.gz")
        self.nodes.to_file(basedir / "nodes.geojson")
        self.raindata.to_netcdf(basedir / "raindata.nc")

    def load(self, basedir: Path):
        self.pars = pd.read_csv(basedir / "pars.csv.gz", index_col=0)
        self.peaks = pd.read_csv(
            basedir / "peaks.csv.gz", index_col="valid_time", parse_dates=["valid_time"]
        )
        self.real_delay = pd.read_csv(
            basedir / "real_delay.csv.gz", index_col="time", parse_dates=["time"]
        )
        self.baseline = pd.read_csv(basedir / "baseline.csv.gz", index_col=0)
        self.delays = pd.read_csv(basedir / "delays.csv.gz", index_col="time", parse_dates=["time"])
        self.nodes = geopd.read_file(basedir / "nodes.geojson")
        self.raindata = xr.load_dataarray(basedir / "raindata.nc")

    def excess_delay(self):
        return (self.real_delay - self.baseline).clip(lower=0).loc[self.peaks.index]

    def _delays_by(self, by: str = "time"):
        return self.delays.groupby(pd.Grouper(level=by)).sum()

    @property
    def index(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.peaks.index, name="time")

    @property
    def real(self) -> pd.Series:
        return self.excess_delay().sum(1)

    @property
    def predicted(self) -> pd.Series:
        return self._delays_by("time")["stats"].reindex(self.index, fill_value=0.0)

    def ci(self, k: str = "low") -> pd.Series:
        _ci = self._delays_by("time").reindex(self.index, fill_value=0.0)
        _real = _ci["stats"]
        _ci = _ci.drop(columns="stats")
        if k == "low":
            _ci = _ci.min(axis=1)
        elif k == "high":
            _ci = _ci.max(axis=1)

        if k == "low":
            return (self.predicted - _ci).clip(lower=0.0)
        return (_ci - self.predicted).clip(lower=0.0)

    @property
    def rain(self) -> pd.Series:
        return self.peaks["median_rain"]

    def get_day(self, isoday: str) -> tuple[pd.DataFrame, xr.DataArray]:
        raindata = self.raindata.sel(time=isoday).sum("time")
        xes = xr.DataArray(self.nodes.geometry.x, dims="nodes")
        yes = xr.DataArray(self.nodes.geometry.y, dims="nodes")

        pred = self.delays.loc[isoday].reindex(self.nodes.index, fill_value=0.0)
        outout = {
            "excess_delay": self.excess_delay().loc[isoday],
            "predicted": pred["stats"],
            "low": pred["stats"] - pred.min(1),
            "high": pred.max(1) - pred["stats"],
            "rain": raindata.sel(longitude=xes, latitude=yes, method="nearest").to_dataframe()[
                "tp"
            ],
        }
        return pd.DataFrame(outout), raindata


DELAYS = Delays()
DELAYS.prepare()
DELAYS.write(Path("/tmp/ita_trains"))

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


def plot_all_days(results: Delays):
    pd.plotting.register_matplotlib_converters()
    fig = plt.figure(figsize=(10, 4))

    # Lineplots
    ax_rep, ax_rain = fig.subplots(
        nrows=2,
        gridspec_kw={"top": 0.85, "bottom": 0.15, "right": 0.52, "left": 0.08, "hspace": 0},
        sharex=True,
        height_ratios=[5, 1],
    )

    ax_rep.fill_between(
        results.index, results.real, color="C8", label="Reported excess", lw=1, alpha=0.8
    )
    ax_rep.fill_between(
        results.index, results.predicted, color="C2", ls="solid", label="Predicted", alpha=0.8, lw=1
    )
    print(results.rain.quantile([0.25, 0.5, 0.75, 0.8, 0.95, 1]))
    add_backgroundgrad(
        ax_rain,
        results.rain.to_numpy().reshape((1, -1)),
        (results.index.to_numpy(), np.asarray([-400])),
        expand_axes=[0.02, 1],
        cmap="RdBu",
        vmin=0,
        vmax=144,
        alpha=0.8,
    )
    ax_rain.set(yticks=[], xlabel="Time")

    handles, labels = ax_rep.get_legend_handles_labels()
    # create a new one
    lc = collections.LineCollection([np.column_stack([np.linspace(0, 1, 10), np.zeros(10)])])
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

    # Scatter plot
    ax_scat = fig.subplots(gridspec_kw={"top": 0.85, "bottom": 0.15, "right": 0.90, "left": 0.6})
    corr = stats.spearmanr(results.predicted[results.rain > 144], results.real[results.rain > 144])
    print("FRAC")
    print(
        pd.DataFrame(
            [
                {
                    "threshold": x,
                    "upper percentile": 100
                    * len(results.rain[results.rain > x])
                    / len(results.rain),
                    "num of rainy days": len(results.rain[results.rain > x]),
                }
                for x in [6 * 24, 7 * 24, 140, 145, 150]
            ]
        )
    )
    scttr = ax_scat.scatter(
        results.predicted,
        results.real,
        c=results.rain,
        s=results.rain + 10,
        alpha=0.5,
        cmap="RdBu",
        lw=0.1,
        edgecolors="k",
        vmin=2,
        vmax=200,
    )
    print(results.ci("low"))
    print(results.ci("high"))
    ax_scat.errorbar(
        results.predicted,
        results.real,
        xerr=[results.ci("low"), results.ci("high")],
        fmt="none",
        alpha=0.3,
    )
    cci = results.ci()
    print(cci[cci > 0])
    print(results.delays)
    ax_scat.set(
        xlabel="Predicted delay (hours)",
        title="Cumulative daily delay",
        ylabel="Reported excess of delay (hours)",
        # aspect=1,
        # xlim=(-200, 3800),
        # ylim=(-200, 3500),
    )
    ax_scat.annotate(
        f"Spearman: {corr.statistic:3.2f}\np-value: {str(corr.pvalue)[:5] if corr.pvalue > 0.01 else '<0.01'}",
        (0.95, 0.95),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize="small",
        color="#666666",
        path_effects=[pe.withStroke(linewidth=2, foreground="w")],
    )
    ax_scat.legend(
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
                markersize=np.sqrt(x + 10),
                lw=0,
                color=scttr.cmap(x / 200),
                label=f"{x} mm",
                alpha=0.5,
                mew=0.1,
                mec="k",
            )
            for x in range(0, 201, 50)
        ],
        fontsize="x-small",
        ncols=1,
        handletextpad=0,
        title="Daily Rain",
        bbox_to_anchor=(1.02, 0.5),
        loc="center left",
        borderaxespad=0.0,
    )
    ax_scat.add_artist(l1)

    text = [
        ax_scat.annotate(
            str(day)[:10],
            (pred, real),
            fontsize="xx-small",
            color="#999999",
            path_effects=[pe.withStroke(linewidth=1, foreground="w")],
        )
        for day, pred, real, rain in zip(
            results.index, results.predicted, results.real, results.rain
        )
        if rain > 150 and pred > 500
    ]
    adjust_text(
        text,
        objects=scttr,
        arrowprops=dict(arrowstyle="->", color="grey", alpha=0.3),
        ax=ax_scat,
        force_text=1.5,
        max_move=(300, 300),
    )
    base.add_axis_label(ax_rep, "a")
    base.add_axis_label(ax_scat, "b ")
    ax_rep.grid(False)

    fig.savefig(base.PLOTS / "validation_2024_errorbars.pdf")
    fig.savefig(base.PLOTS / "validation_2024_errorbars.png")
    plt.close()


# %%


def plot_multi_days(isodays: list[str]):
    """"""
    fig = plt.figure(figsize=(13, 4.4 * len(isodays)))

    axss = fig.subplots(
        ncols=3,
        nrows=len(isodays),
        sharey=True,
        sharex=True,
        gridspec_kw={
            "wspace": 0,
            "hspace": 0.07,
            "left": 0.05,
            "right": 0.55,
            "bottom": 0.1,
            "top": 0.95,
        },
    )
    axs = fig.subplots(
        nrows=len(isodays),
        gridspec_kw={"left": 0.62, "right": 0.94, "top": 0.95, "bottom": 0.1, "hspace": 0.07},
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
    axs[0].set(ylabel="Reported excess delay (hours)")
    axs[1].set(xlabel="Predicted delay (hours)", ylabel="Reported excess delay (hours)")

    base.add_axis_label(axss[0, 0], "a")
    base.add_axis_label(axs[0], "b")

    title = "_".join(isodays)
    fig.savefig(base.PLOTS / f"validation_2024_oneday_{title}.pdf")
    fig.savefig(base.PLOTS / f"validation_2024_oneday_{title}.png")
    plt.close()


def _plot_one_day(isoday: str, ax_real, ax_pred, ax_stress, ax_scat):
    day, raindata = DELAYS.get_day(isoday)
    day = day.sort_values(by="rain")

    kwargs = dict(
        rasterized=False, cmap="RdBu", vmin=0.0, vmax=500, lw=0.1, edgecolor="k", alpha=0.5
    )
    day = geopd.GeoDataFrame(day, geometry=DELAYS.nodes.geometry)

    raindata.plot.pcolormesh(
        ax=ax_stress, add_colorbar=False, lw=0, rasterized=True, vmax=kwargs["vmax"], cmap="Blues"
    )

    ax_real.scatter(
        day["geometry"].x,
        day["geometry"].y,
        s=lin_map(day["excess_delay"], (0, 20), (0, 40)),
        c=day["rain"],
        **kwargs,
    )
    ax_pred.scatter(
        day["geometry"].x,
        day["geometry"].y,
        s=lin_map(day["predicted"], (0, 20), (0, 40)) if day["predicted"].max() > 0 else 0,
        c=day["rain"],
        **kwargs,
    )

    points = ax_scat.scatter(
        day["predicted"],
        day["excess_delay"],
        c=day["rain"],
        s=[DELAYS.nodes.loc[s, "capacity"] * 2 for s in day.index],
        **kwargs,
    )

    print(day)
    ax_scat.errorbar(
        day["predicted"], day["excess_delay"], xerr=[day["low"], day["high"]], fmt="none", alpha=0.3
    )
    l1 = ax_scat.legend(
        handles=[
            lines.Line2D(
                [0],
                [0],
                marker="o",
                markersize=10,
                lw=0,
                color=points.cmap(x / kwargs["vmax"]),
                label=f"{x} mm",
                alpha=0.5,
                mew=0.1,
                mec="k",
            )
            for x in range(100, kwargs["vmax"] + 1, 100)
        ],
        fontsize="x-small",
        ncols=1,
        handletextpad=0,
        title="Daily Rain",
        bbox_to_anchor=(1.02, 0.5),
        loc="center left",
        borderaxespad=0.0,
    )
    ax_scat.add_artist(l1)
    print(day["rain"].quantile([0.25, 0.5, 0.75, 0.9, 1.0]))
    spear_data = day.loc[(day["predicted"] > 0) & (day["excess_delay"] > 0)]
    if len(spear_data) > 1:
        spear = stats.spearmanr(spear_data["predicted"], spear_data["excess_delay"])
        ax_scat.annotate(
            f"Spearman: {spear.statistic:3.2f}\np-value: {str(spear.pvalue)[:5] if spear.pvalue > 0.01 else '<0.01'}",  # type: ignore
            (0.95, 0.95),
            xycoords="axes fraction",
            ha="right",
            va="top",
            path_effects=[pe.withStroke(linewidth=3, foreground="w")],
        )
    text = [
        ax_scat.text(
            d["predicted"], d["excess_delay"], base.shorten_name(str(node)), fontsize="small"
        )
        for node, d in day.iterrows()
        if d["predicted"] > day["predicted"].quantile(0.998)
        or d["excess_delay"] > day["excess_delay"].quantile(0.998)
    ]
    ax_scat.set(xscale="symlog", yscale="symlog", aspect=1, xlim=(-0.4, 2e3), ylim=(-0.4, 2e3))
    adjust_text(
        text,
        prevent_crossings=True,
        objects=points,
        force_text=(0.5, 1.5),
        force_pull=(0.01, 0.001),
        max_move=(80, 80),
        arrowprops=dict(arrowstyle="->", color="grey", alpha=0.3),
        ax=ax_scat,
        expand_axes=True,
    )


def plot_one_day(day: pd.DataFrame, raindata: xr.DataArray, title: str = "One day"):
    fig = plt.figure(figsize=(13, 4.4))

    ax_real, ax_pred, ax0 = fig.subplots(
        ncols=3,
        sharey=True,
        sharex=True,
        gridspec_kw={"wspace": 0, "left": 0.05, "right": 0.55, "bottom": 0.15},
    )
    ax1 = fig.subplots(gridspec_kw={"left": 0.62, "right": 0.94, "bottom": 0.15})

    _plot_one_day(title, ax_real, ax_pred, ax0, ax1)
    ax_real.set_title("Real data")
    ax_pred.set_title("Prediction")
    ax0.set(title="Stressor field", xlabel="", ylabel="")

    ax1.set(title=title, ylabel="Reported excess delay (hours)", xlabel="Predicted delay (hours)")

    fig.savefig(base.PLOTS / f"validation_2024_oneday_{title}.pdf")
    fig.savefig(base.PLOTS / f"validation_2024_oneday_{title}.png")
    plt.close()


def lin_map(vals: np.ndarray | pd.Series | float, p1: tuple, p2: tuple) -> np.ndarray | float:
    """Linear map from vals to the line betwenn p1 and p2.

    p1: two points in the domain space
    p2: two points in the codomain space
    """
    vals = vals.to_numpy() if isinstance(vals, pd.Series) else vals
    return p2[0] + (vals - p1[0]) * (p2[1] - p2[0]) / (p1[1] - p1[0])


def main() -> None:
    """Do the main."""
    plot_all_days(DELAYS)
    plot_multi_days(["2024-10-03", "2024-09-09"])
    for isoday in [
        "2024-01-06",
        "2024-01-07",
        "2024-02-10",
        "2024-02-23",
        "2024-02-27",
        "2024-03-01",
        "2024-03-04",
        "2024-03-10",
        "2024-03-27",
        "2024-04-01",
        "2024-05-01",
        "2024-05-02",
        "2024-05-15",
        "2024-05-16",
        "2024-05-21",
        "2024-05-31",
        "2024-09-05",
        "2024-09-09",
        "2024-09-12",
        "2024-10-03",
        "2024-10-08",
        "2024-10-10",
        "2024-10-18",
        "2024-10-19",
        "2024-10-20",
        "2024-12-08",
    ]:
        plot_one_day(*DELAYS.get_day(isoday=isoday), isoday)


if __name__ == "__main__":
    main()

# %%
