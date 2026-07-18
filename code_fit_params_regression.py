"""Predict delays using a linear model (Ridge) from rain.

Plot it and compute Spearman.
"""

from pathlib import Path
from typing import Literal

import pandas as pd
from matplotlib import patheffects as pe
from matplotlib import pyplot as plt
from scipy import stats
from sklearn import linear_model


def resample(fl: Path) -> pd.DataFrame:
    df = pd.read_csv(fl, index_col=0).T
    df.index = pd.DatetimeIndex(df.index)
    df = df.resample("1D").sum()
    return df


def load_rain(dataset: Literal["train", "test"] = "train") -> pd.DataFrame:
    return pd.concat(
        [
            resample(fl)
            for fl in Path("stats/rain_cache").glob(
                "*202[123]*" if dataset == "train" else "*2024*"
            )
        ]
    ).sort_index()


def load_peaks(year: int | list[int] | None = None, kind: str = "both") -> pd.DataFrame:
    """Load the peaks."""
    data: pd.DataFrame = pd.read_csv("./data/rainy_peaks.csv.gz", index_col=0, parse_dates=True)
    if kind == "both":
        data = data.loc[data["peak"].isin(["high", "low"])]
    elif kind in {"high", "low"}:
        data = data.loc[data["peak"] == kind]
    elif kind == "full":
        data = data
    else:
        raise NotImplementedError()

    if year is not None:
        if isinstance(year, int):
            data = data.loc[data.index.year == year]  # type: ignore
        else:
            data = data.loc[data.index.year.isin(year)]  # type: ignore

    return data.sort_index()


def load_delay(dataset: Literal["train", "test"] = "train") -> pd.DataFrame:
    if dataset == "train":
        d = pd.concat(
            [
                pd.read_csv(fl, index_col=0, parse_dates=[0])
                for fl in Path("data").glob("delays_per_stations_202[123].csv.gz")
            ]
        ).sort_index()
        return (d - d.median(0)).clip(lower=0.0)

    d = pd.read_csv(
        Path("data/delays_per_stations_2024.csv.gz"), index_col=0, parse_dates=[0]
    ).sort_index()
    return (d - d.median(0)).clip(lower=0.0)


# %%


def main() -> None:
    """Do the main."""
    # rain = load_rain()
    rain = (
        load_peaks([2021, 2022, 2023], kind="full")["median_rain"].sort_index().rename("rain")
        * 1000
    )
    test_rain = load_peaks(2024, kind="full")["median_rain"].sort_index().rename("rain") * 1000
    delay = load_delay().sum(1)

    test_delay = load_delay("test").sum(1)

    regr = linear_model.Ridge(fit_intercept=False)
    regr = regr.fit(rain.to_frame(), delay.to_frame())
    new = regr.predict(test_rain.to_frame())
    results = pd.DataFrame({"rain": test_rain, "real": test_delay / 60, "pred": new.squeeze() / 60})
    print(results)

    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))

    ax = axs
    results.plot.scatter(
        "pred",
        "real",
        c="rain",
        s=results["rain"] / 3 + 10,
        cmap="RdBu",
        alpha=0.5,
        ax=ax,
        vmin=0,
        vmax=144,
        lw=0,
    )
    _, cax = fig.get_axes()
    cax.set(ylabel="Rainfall (mm)")
    ax.set(
        title="Ridge (IRN)",
        xlabel="Predicted delay (hours)",
        ylabel="Reported excess delay (hours)",
    )
    corr = stats.spearmanr(
        results[results["rain"] > 144]["pred"], results[results["rain"] > 144]["real"]
    )
    print(corr)
    ax.annotate(
        f"Spearman: {corr.statistic:3.2f}\np-value: {str(corr.pvalue)[:5] if corr.pvalue > 0.01 else '<0.01'}",
        (0.95, 0.95),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize="small",
        color="#666666",
        path_effects=[pe.withStroke(linewidth=2, foreground="w")],
    )

    ax.set_autoscale_on(False)
    xlim = ax.get_xlim()
    ax.plot(xlim, xlim, "k-.", alpha=0.5)

    fig.tight_layout()
    fig.savefig("./plots/validation_regression_2024.pdf", dpi=300)
    fig.savefig("./plots/validation_regression_2024.png", dpi=300)


if __name__ == "__main__":
    main()
