"""Study of the mutual influence of the nodes.

How much an external perturbation on one node influence other nodes.

1. The perturbation is on the state -> the evolution of the perturbation on other nodes depends on the transition matrix and the removal process
2. The perturbation is on the external stressor.
3. Compute also the steady state
"""

import json
from pathlib import Path

import geopandas as geopd
import numpy as np
import pandas as pd
import xarray as xr
from scipy import optimize, stats
from tqdm import trange

import base
import diffsys
from diffsys.models import Diffusion

# %%
graph_adj = base.load_graph(
    full=False, days=pd.date_range("2021-01-01", "2023-12-31").to_flat_index()
).drop_duplicates()
graph_tmp = base.load_graph(
    full=True, days=pd.date_range("2021-01-01", "2023-12-31").to_flat_index()
)
LOC_CACHE = {}


def exfield_like(exfiel: diffsys.ExternalField, fill_values: float):
    """Make a new external stressor with the same coords but filled by a fixed value."""
    data = xr.DataArray(
        np.full(exfiel.shape, fill_values), coords=exfiel.data.coords
    )  # From 50mm it's heavy rain
    return diffsys.ExternalField(data)


# %%


def get_stress(exf, rain):
    global graph_adj
    graph = graph_tmp.subset([("weekday", False), ("month", False), ("hour", 8)])
    stressor = base.generated_delay(
        graph, exf, trange=exf.trange()[0], weight="count", integral=rain
    )

    capacity = graph_adj.nodes()["capacity"].to_numpy()

    return stressor, capacity


def sim_all(
    data: tuple[diffsys.ExternalField, pd.Series, dict], fname: str | Path | None = None
) -> geopd.GeoDataFrame:
    global graph_adj
    global graph_tmp

    stressor, rain, params = data

    mod = Diffusion(graph_adj, stressor, **params)
    exf = next(mod.ex_field.extreme_events(kind="simple"))
    graph = graph_tmp.subset([("weekday", False), ("month", False), ("hour", 8)])

    # Initialize
    delta = np.inf
    state = np.zeros(graph.nn)

    loc_stressor, capacity = get_stress(exf, rain)

    delay_evolution = [(0, 0, 0)]
    for i in trange(100000, dynamic_ncols=True, leave=False):
        state = mod.state.copy()

        mod.evolve()
        mod.generate(loc_stressor, "beta")
        mod.generate(-capacity, "gamma")
        mod.conclude_step(exf.trange()[0].to_datetime64(), threshold=0, keep_cascade=False)

        delay_evolution.append((i + 1, mod.state.mean(), mod.state.max()))

        new_state = mod.state.copy()
        delta = np.abs((new_state - state)).sum()

        if delta <= 1e-5:
            break
        if new_state.max() > 1e5:
            break

    if fname is not None:
        pd.DataFrame(
            delay_evolution, columns=pd.Index(["hours", "mean_delay", "max_delay"])
        ).to_csv(fname, index=False)

    nodes = graph_adj.nodes().copy()
    if delta and delta > 1e-5:
        print("Warning, not converged:", delta)
    else:
        nodes["ss"] = np.asarray(state)

    nodes["ext_s"] = loc_stressor
    nodes["diff"] = params["beta"] * loc_stressor - params["gamma"] * capacity

    mod.set_initial_state(np.full_like(mod.state, 1000))

    for i in trange(100000, dynamic_ncols=True, leave=False):
        state = mod.state.copy()

        mod.evolve()
        mod.generate(loc_stressor, "beta")
        mod.generate(-capacity, "gamma")
        mod.conclude_step(exf.trange()[0].to_datetime64(), threshold=0, keep_cascade=False)

        new_state = mod.state.copy()
        delta = np.abs((new_state - state)).sum()

        if delta <= 1e-5:
            break
        if new_state.max() > 1e5:
            break

    if delta and delta > 1e-5:
        print("Warning, not converged:", delta)
    else:
        nodes["ss_up"] = np.asarray(state)

    return geopd.GeoDataFrame(nodes, geometry="geometry", crs=4326)


def main() -> None:
    """Do the main."""
    # Load the network
    global graph_adj
    global graph_tmp

    # build the stressor
    stressor = exfield_like(base.load_extfield(2024).get(day="2024-01-01"), fill_values=0.01)

    # Params
    params = base.params()
    print(params)

    rain = graph_adj.integrate(stressor, trange=pd.Timestamp("2024-01-01 08:00:00"), ds=1.0)

    delays = sim_all((stressor, rain, params))
    print(delays.sort_values("ss"))
    print(delays["ss"].mean())

    delays.to_file(base.CACHE / "ext_field_steady_state.geojson")


def __find_bounds(stressor, rain, params) -> tuple[float, float]:
    """find lower and upper bounds to the region where steady state converge."""

    def opt(s, stressor, capacity, params, stat):
        return stat(params["beta"] * s * stressor - params["gamma"] * capacity)

    s, c = get_stress(next(stressor.extreme_events()), rain)

    lbound = optimize.fsolve(opt, 0.0005, args=(s, c, params, np.max))[0]
    ubound = optimize.fsolve(opt, 0.0025, args=(s, c, params, np.sum))[0]
    return lbound, ubound


def find_bounds() -> tuple[float, float]:
    """find lower and upper bounds to the region where steady state converge."""
    print("Computing steady state bounds.")
    stressor = exfield_like(base.load_extfield(2024).get(day="2024-01-01"), fill_values=1)
    rain = graph_adj.integrate(stressor, trange=pd.Timestamp("2024-01-01 08:00:00"), ds=1.0)
    params = base.params(df="ci")

    lbound, ubound = __find_bounds(stressor, rain, params["stats"].to_dict())
    res = {"upper": ubound, "lower": lbound, "low": {}, "high": {}}
    print(f"Lower bound: {lbound}")
    print(f"Upper bound: {ubound}")

    l, u = [], []
    for name, pars in base.params(df="kfold").iterrows():
        print(name)
        _l, _u = __find_bounds(stressor, rain, pars)
        l.append(_l)
        u.append(_u)
    res["low"] = {"upper": min(u), "lower": min(l)}
    res["high"] = {"upper": max(u), "lower": max(l)}
    print(f"bound: {res['low']}")
    print(f"bound: {res['high']}")

    with open("./data/ext_field_steady_state_bounds.json", "wt") as fout:
        json.dump(res, fout)

    return lbound, ubound


def find_critical_value():
    global graph_adj
    global graph_tmp

    # Params
    params = base.params()
    print(params)

    stressor = exfield_like(base.load_extfield(2024).get(day="2024-01-01"), fill_values=1)
    rain = graph_adj.integrate(stressor, trange=pd.Timestamp("2024-01-01 08:00:00"), ds=1.0)
    print(rain)

    res = []
    for s in np.linspace(0, 0.03, 151):
        _res = {"stressor": s}

        delays = sim_all((stressor * s, rain * s, params))

        for col in ["ss", "ss_up"]:
            if col in delays.columns:
                if len(np.unique(delays[col])) > 1:
                    ci = stats.bootstrap([delays[col]], np.mean)
                    _res[f"{col}_delay_ci_low"] = ci.confidence_interval[0]
                    _res[f"{col}_delay_ci_high"] = ci.confidence_interval[1]
                else:
                    _res[f"{col}_delay_ci_low"] = delays[col].mean()
                    _res[f"{col}_delay_ci_high"] = delays[col].mean()
                _res[f"{col}_delay"] = delays[col].mean()
                _res[f"{col}_delay_std"] = delays[col].std()
                _res[f"{col}_delay_q025"] = delays[col].quantile(0.025)
                _res[f"{col}_delay_q50"] = delays[col].quantile(0.5)
                _res[f"{col}_delay_q975"] = delays[col].quantile(0.975)
                _res[f"{col}_delay_max"] = delays[col].max()
            else:
                _res[f"{col}_delay"] = None
                _res[f"{col}_delay_std"] = None
                _res[f"{col}_delay_ci_low"] = None
                _res[f"{col}_delay_ci_high"] = None
                _res[f"{col}_delay_q025"] = None
                _res[f"{col}_delay_q50"] = None
                _res[f"{col}_delay_q975"] = None
                _res[f"{col}_delay_max"] = None

        _res["ext_min"] = delays["ext_s"].min()
        _res["ext_max"] = delays["ext_s"].max()
        _res["diff_min"] = delays["diff"].min()
        _res["diff_max"] = delays["diff"].max()

        res.append(_res)

        if "ss" not in delays.columns:
            break

    res_df = pd.DataFrame(res)
    res_df.to_csv(base.CACHE / "ext_field_steady_state_criticalpoint.csv.gz")


if __name__ == "__main__":
    lb, ub = find_bounds()
    find_critical_value()
    main()

# %%
