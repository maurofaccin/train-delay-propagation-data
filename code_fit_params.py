"""Fit the parameter alpha, beta and gamma to the learning dataset.

Warning: will take a lot of memory and cores

This should be run 4 times changing the value of KFOLD \in {0, 1, 2, 3}
"""

import json
import logging
from functools import partial
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy import optimize
from sklearn import model_selection
from tqdm.contrib.concurrent import process_map

import base
import diffsys
from base import CACHE, load_peaks
from diffsys.models import Diffusion

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TCACHE = Path("train_stats/")
NJOBS = 20
YEARS = [2021, 2022, 2023]
PEAKS = load_peaks(YEARS, kind="full")
KFOLD = 1
print(f"Fitting the {KFOLD} kfold.")

# %%

kfold = model_selection.StratifiedKFold(n_splits=4, shuffle=True, random_state=1996)
for i, (train_index, test_index) in enumerate(kfold.split(PEAKS.index, PEAKS["rain"] > 500)):
    if i == KFOLD:
        train_days = PEAKS.index[train_index]
        test_days = PEAKS.index[test_index]
        break

# %%

REAL_DELAY = pd.DataFrame({})
GRAPH_ADJ = diffsys.Graph.empty()
GRAPH_TMP = diffsys.Graph.empty()
EF = diffsys.ExternalField.empty()


def prepare_data():
    global REAL_DELAY
    global GRAPH_ADJ
    global GRAPH_TMP
    global EF
    REAL_DELAY = base.load_real_delay(YEARS)
    logger.info(f"Loading Files for {len(REAL_DELAY)} days")
    baseline = REAL_DELAY.loc[train_days].median()

    GRAPH_ADJ = base.load_graph(full=False, days=train_days).drop_duplicates()
    logger.info(f"Full graph: {GRAPH_ADJ}")
    GRAPH_ADJ._nodes["delay_q50"] = baseline

    GRAPH_TMP = base.load_graph(full=True, days=train_days)
    logger.info(f"Graph {GRAPH_TMP}")

    EF = base.load_extfield(YEARS)
    print(f"ExternalField: {EF}")


# %%


def _simulate(pars: tuple, days: pd.DatetimeIndex | None = None):
    if len(pars) == 2:
        alpha = 1.0
        beta, gamma = pars
    elif len(pars) == 3:
        alpha, beta, gamma = pars
    else:
        raise ValueError()

    if days is None:
        days = pd.DatetimeIndex(PEAKS.index)

    logger.info(f"Using:: α={alpha}, β={beta}, γ={gamma}")
    func = partial(base.sim, full_graph=GRAPH_TMP, usecache=True)
    cascades = process_map(
        func,
        [
            Diffusion(
                GRAPH_ADJ, EF.get(day=peak_day), alpha=alpha, beta=beta, gamma=gamma, weight="count"
            )
            for peak_day in days
        ],
        total=len(days),
        max_workers=NJOBS,
        chunksize=5,
    )
    return cascades


def _test_cascasdes(cascades: list[pd.DataFrame]):
    real_delay = REAL_DELAY
    baselines = GRAPH_ADJ.nodes()["delay_q50"]
    delta = 0
    n = 0

    daily_rain = EF.data.resample(time="D").sum()

    # Save position in `DataArray` to use as point coordinates
    st_lons = xr.DataArray([p.x for p in GRAPH_ADJ.nodes()["geometry"]])
    st_lats = xr.DataArray([p.y for p in GRAPH_ADJ.nodes()["geometry"]])
    for cascade, day in zip(cascades, PEAKS.index):
        rainy = daily_rain.sel(
            longitude=st_lons,
            latitude=st_lats,
            time=xr.DataArray([day] * len(st_lats)),
            method="nearest",
        )

        # Find stations where there was at least a bit of rain
        rainy_stats = [s for s, r in zip(GRAPH_ADJ.nodes().index, rainy.data) if r > 0]
        excess_delay = (real_delay.loc[day] - baselines).clip(lower=0)
        if len(cascade) == 0:
            delta_v = excess_delay.fillna(0.0)

            n += 1
        else:
            delta_v = (
                (cascade.set_index("node", drop=True)["value"] - excess_delay).fillna(0.0).abs()
            )

        delta += np.sum(np.power(np.abs(delta_v.loc[rainy_stats].to_numpy()), 2.0))

    logger.info(f"Got Δ={delta:g} (zeros = {n} / {len(PEAKS)})")
    return delta


# %%


def simulate(pars: tuple):
    cascades = _simulate(pars, days=test_days)
    delta = _test_cascasdes(cascades)

    if len(pars) == 3:
        data = {k: x for k, x in zip(["alpha", "beta", "gamma"], pars)}
    else:
        data = {k: x for k, x in zip(["beta", "gamma"], pars)}
        data["alpha"] = 1.0
    data["value"] = delta

    folder = CACHE / "optimize_pars"
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / f"fitted_params_KFOLD-{KFOLD}.jsonl").open("a") as fout:
        json.dump(data, fout, separators=(",", ":"))
        fout.write("\n")
        fout.flush()

    return delta


# %%


def main() -> None:
    """Do the main."""
    prepare_data()
    delta = np.inf
    for a, b, g in product(np.linspace(1.01, 1.03, 2), np.linspace(5, 7, 4), np.linspace(1, 2, 4)):
        d = simulate((a, b, g))
        print(d, a, b, g)
        if d < delta:
            opt_pars = {"alpha": a, "beta": b, "gamma": g}
            delta = d

    optimize.minimize(
        simulate,
        (opt_pars["alpha"], opt_pars["beta"], opt_pars["gamma"]),
        bounds=[(1.0, 100.0), (0, 1000.0), (0, 100.0)],
        method="Nelder-Mead",
    )


if __name__ == "__main__":
    main()
