"""Compute the real delay-rain correlation.
Plot a map of Spearman coefficients for each station.
"""

import numpy as np
import pandas as pd
import xarray as xr
from cartopy import crs as ccrs
from cartopy import feature as cfeature
from matplotlib import pyplot as plt
from scipy import stats

import base


def load_delay():
    d = pd.read_csv(
        "./train_stats/delays_per_stations_2023.csv.gz", index_col="time", parse_dates=["time"]
    )
    return d


def load_rain():
    xrain = (
        xr.load_dataarray(
            base.COPERNICUS / "IT_total_precipitation_land/total_precipitation_IT_2023.nc"
        )
        .resample(valid_time="1D")
        .sum()
        .rio.write_crs("4326")
        .fillna(0.0)
    )
    nodes = base.load_nodes()

    print(xrain.max().data)
    rain = (
        xrain.sel(
            longitude=xr.DataArray(nodes.geometry.x, dims="nodes"),
            latitude=xr.DataArray(nodes.geometry.y, dims="nodes"),
            method="nearest",
        )
        .drop_vars(["number", "latitude", "longitude", "expver", "spatial_ref"], errors="ignore")
        .to_dataframe()["tp"]
        .unstack(fill_value=0.0)
    )

    return rain


def main() -> None:
    """Do the main."""
    d = load_delay()
    print(d)
    r = load_rain()
    print(r)

    nodes = base.load_nodes()
    nodes["spearman"] = [
        stats.spearmanr(d[station], r[station]).statistic for station in nodes.index
    ]
    nodes["pvalue"] = [stats.spearmanr(d[station], r[station]).pvalue for station in nodes.index]
    nodes = nodes.fillna({"spearman": 0.0, "pvalue": 1.0})

    print(nodes["spearman"].quantile([0, 0.25, 0.5, 0.75, 0.9, 1]))
    print(nodes["pvalue"].quantile([0, 0.25, 0.5, 0.75, 0.9, 1]))

    fig, axs = plt.subplots(
        nrows=1, ncols=1, figsize=(5, 4), subplot_kw={"projection": ccrs.PlateCarree()}
    )

    base.load_graph(full=False).edges().plot(
        ax=axs, lw=1, alpha=0.2, rasterized=True, zorder=0.01, color="#999999"
    )
    axs.add_feature(cfeature.OCEAN, alpha=0.5, rasterized=True)
    axs.add_feature(cfeature.BORDERS, linestyle="-", lw=0.2, alpha=0.2)
    nodes.plot(
        ax=axs,
        column="spearman",
        markersize=np.clip(-np.log10(nodes["pvalue"]), a_min=0, a_max=5) * 10,
        cmap="Spectral",
        vmin=-0.3,
        vmax=0.3,
        legend=True,
        legend_kwds={"label": "Spearman"},
        alpha=0.8,
        lw=0,
    )
    fig.tight_layout()
    fig.savefig("./plots/delay_rain_corr.pdf", dpi=300)


if __name__ == "__main__":
    main()
