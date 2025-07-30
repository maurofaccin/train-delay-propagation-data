"""Study of the mutual influence of the nodes.

How much an external perturbation on one node influence other nodes.

1. The perturbation is on the state -> the evolution of the perturbation on other nodes depends on the transition matrix and the removal process
2. The perturbation is on the external stressor.
3. Compute also the steady state
"""

import numpy as np
import pandas as pd
import xarray as xr
from tqdm.contrib import concurrent

import base
import diffsys
from diffsys.models import Diffusion

PERTURBATION_STRESSOR = 0.1  # 0.1m/h rain
PERTURBATION_RADIUS = 50  # km

graph_adj = base.load_graph(full=False).drop_duplicates()
graph_tmp = base.load_graph(full=True)

# %%


def exfield_like(exfiel: diffsys.ExternalField, fill_values: float):
    """Make a new external stressor with the same coords but filled by a fixed value."""
    data = xr.DataArray(
        np.full(exfiel.shape, fill_values), coords=exfiel.data.coords
    )  # From 50mm it's heavy rain
    return diffsys.ExternalField(data)


def sim_station(
    model: Diffusion,
    station: str,
    kind: str,
) -> pd.DataFrame:
    # save initial state
    # model.conclude_step(model.ex_field.trange()[0].to_datetime64(), threshold=0)

    for ilef, lef in enumerate(model.ex_field.extreme_events("simple")):
        hour = lef.trange()[1]

        # We do not use the hourly transition matrix
        # (in the first hours in the morning there are no trains)
        model.evolve()

        model.generate(-model.graph.nodes()["capacity"].to_numpy(), "gamma")
        model.conclude_step(hour.to_datetime64(), threshold=0)
        if model.state.sum() < 1e-10:
            break

    model.conclude_cascade()

    # Transform to DataFrame
    delays = list(model.cascades())[0].df()
    if len(delays) == 0:
        return pd.DataFrame(
            {
                kind: [],
                kind + "_count": [],
                kind + "_risk": [],
                "time": [],
                "station": [],
            },
        ).set_index(["time", "station"], drop=True)

    delays = (
        delays.drop(columns=["failing"])
        .rename(columns={"value": station})
        .set_index("node", drop=True)
        .rename_axis("neighbors")
    )

    delays_cumul = (
        delays.groupby("time")
        .sum()
        .stack()
        .rename_axis(["time", "station"])
        .rename(index=kind)
    )
    delays_count = (
        delays.groupby("time")
        .count()
        .stack()
        .rename_axis(["time", "station"])
        .rename(index=kind + "_count")
    )
    res = pd.concat([delays_cumul, delays_count], axis=1)
    res[kind + "_risk"] = [
        d[kind] * graph_adj.nodes().loc[s, "pop"] for (t, s), d in res.iterrows()
    ]

    return res


# %%


def prepare_initial_state(
    kind: str,
    model: Diffusion,
    station: str,
    loc_graph: diffsys.Graph | None = None,
    rain: pd.Series | None = None,
) -> None:
    if kind == "state":
        init = np.zeros(model.graph.nn)
        # init[model.graph.nodes().index.get_loc(station)] = PERTURBATION_STATE
        model.set_initial_state(init)
    elif kind == "degree":
        init = np.zeros(model.graph.nn)
        # init[model.graph.nodes().index.get_loc(station)] = (
        #     PERTURBATION_STATE / 4 * float(model.graph.nodes().loc[station, "capacity"])
        # )
        model.set_initial_state(init)
    elif kind == "stressor" and loc_graph is not None and rain is not None:
        gg = loc_graph.subset(
            [("hour", 9), ("weekday", False), ("month", False)]
        ).drop_duplicates()
        # Sum the rain along all incoming links
        integral = (
            gg.to_matrix(weight=rain)
            .multiply(gg.to_matrix(weight="count"))
            .sum(1)
            .A.ravel()
        )
        model.generate(integral, "beta")
    else:
        raise NotImplementedError


def sim_all(data):
    global graph_adj
    global graph_tmp

    stressor, station, rain, params = data

    mod = Diffusion(graph_adj, stressor, **params)
    prepare_initial_state(
        "stressor",
        mod,
        station,
        graph_tmp,
        rain.loc[[station in link for link in rain.index]],
    )
    sim_stressor = sim_station(mod, station, "stressor")

    # # Perturbation on the state
    # mod = Diffusion(graph_adj, stressor, **params)
    # prepare_initial_state("state", mod, station)
    # sim_state = sim_station(mod, station, "state")
    #
    # # Perturbation on the state
    # mod = Diffusion(graph_adj, stressor, **params)
    # prepare_initial_state("degree", mod, station)
    # sim_degree = sim_station(mod, station, "degree")

    return pd.concat(
        [
            sim_stressor,
            # sim_state,
            # sim_degree
        ],
        axis=1,
    )


def main() -> None:
    """Do the main."""
    # Load the network
    # graph_adj = base.load_graph(full=False).drop_duplicates()
    # graph_tmp = base.load_graph(full=True)
    global graph_adj
    global graph_tmp

    # build the stressor
    stressor = exfield_like(
        base.load_extfield(2024).get(day="2024-01-01"),
        fill_values=PERTURBATION_STRESSOR,
    )

    rain = graph_adj.integrate(
        stressor, trange=pd.Timestamp("2024-01-01 01:00:00"), ds=1.0
    )
    # Put a radius of about 20 km
    rain_threshold = PERTURBATION_STRESSOR * PERTURBATION_RADIUS
    rain[rain > rain_threshold] = rain_threshold
    print(rain.sort_values())

    # Params
    params = base.params()
    print(params)

    delays = concurrent.process_map(
        sim_all,
        [(stressor, station, rain, params) for station in graph_adj.nodes().index],
        max_workers=8,
        chunksize=2,
    )

    delays_df = pd.concat(delays, ignore_index=False)
    delays_df.to_csv(base.CACHE / "coverage_data.csv.gz")
    print(delays_df)


if __name__ == "__main__":
    main()

# %%
