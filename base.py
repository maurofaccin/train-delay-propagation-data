from pathlib import Path

import geopandas as geopd
import numpy as np
import pandas as pd
import xarray as xr

import diffsys
from diffsys.models import Diffusion

COPERNICUS = Path("~/curro/data/copernicus/")
TCACHE = Path("data")
CACHE = Path("cache")
CACHE.mkdir(exist_ok=True)
PLOTS = Path("plots")
PLOTS.mkdir(exist_ok=True)

PARAMS = {"alpha": 1.0, "beta": 19.126156378600818, "gamma": 7.788523491083679}

LOC_CACHE = {}


def params():
    return PARAMS


def ita() -> geopd.GeoSeries:
    return geopd.GeoSeries(
        [geopd.read_file(TCACHE / "ITA_regions.geojson").union_all(method="coverage")]
    ).simplify(0.01)


def load_extfield(years: list | int | None = None) -> diffsys.ExternalField:
    if years is None:
        return load_extfield(list(range(2020, 2025)))

    if isinstance(years, int):
        data = xr.load_dataarray(
            COPERNICUS
            / f"IT_total_precipitation_land/total_precipitation_IT_{years}.nc",
            decode_coords="all",
        )
    else:
        data = xr.concat(
            [
                xr.load_dataarray(
                    COPERNICUS
                    / f"IT_total_precipitation_land/total_precipitation_IT_{year}.nc",
                    decode_coords="all",
                )
                for year in years
            ],
            dim="valid_time",
        )

    return diffsys.ExternalField(
        data.fillna(0.0).rename({"valid_time": "time"}).squeeze().drop_vars("number")
    )


def load_nodes() -> geopd.GeoDataFrame:
    return geopd.read_file(TCACHE / "graph_nodes_metadata.geojson").set_index(
        "index", drop=True
    )


def load_graph(full: bool = True) -> diffsys.Graph:
    """Load the graph.

    This may contain duplicated links.

    If `full == True` all links are kept separated by month, weekday, hour,
    otherwise a global average is returned.
    """
    nodes = load_nodes()

    edges = geopd.read_file(TCACHE / "graph_edges.geojson", rows=None)
    edges = edges.set_index(["source", "target"], drop=True)

    transitions = pd.read_csv(
        TCACHE / "aggregate_transitions_learn_ita.csv.gz", index_col=0
    ).drop(
        columns=["cumul", "type"],
    )

    tcount = pd.DataFrame(
        [
            {"val": 1, "m": d.month, "w": d.weekday()}
            for d in pd.date_range("2021-01-01", "2023-12-31")
        ]
    )
    if full:
        transitions["month"] = transitions["month"] == 8  # August
        transitions["weekday"] = transitions["weekday"] == 6  # Sunday
        transitions = (
            transitions.groupby(["start", "end", "month", "weekday", "hour"])
            .sum()
            .reset_index()
        )

        # Normalize by the number of time each combination appears
        tcount["m"] = tcount["m"] == 8
        tcount["w"] = tcount["w"] == 6
        tcount = tcount.groupby(["m", "w"]).sum()

        transitions["count"] = [
            t["count"] / tcount.loc[(t["month"], t["weekday"]), "val"]
            for _, t in transitions.iterrows()
        ]
    else:
        transitions = (
            transitions.drop(columns=["month", "weekday", "hour"])
            .groupby(["start", "end"])
            .sum()
            .reset_index()
        )
        # Normalize by the number of time each combination appears
        transitions["count"] /= len(tcount) * 24

    # Remove self loops
    transitions = transitions[transitions["start"] != transitions["end"]]

    transitions["geometry"] = [
        edges.loc[(st["start"], st["end"])].geometry for _, st in transitions.iterrows()
    ]

    return diffsys.Graph(
        geopd.GeoDataFrame(nodes, geometry="geometry", crs=4326),
        geopd.GeoDataFrame(transitions, geometry="geometry", crs=4326),
        directed=False,
        edge_cols=["start", "end"],
    )


def load_peaks(year: int | list[int] | None = None, kind: str = "both") -> pd.DataFrame:
    """Load the peaks."""
    data: pd.DataFrame = pd.read_csv(
        TCACHE / "rainy_peaks.csv.gz", index_col=0, parse_dates=True
    )
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


def sim(
    model: Diffusion,
    full_graph: diffsys.Graph,
    usecache: bool = True,
    cached: dict | None = None,
) -> pd.DataFrame:
    """Simulate a day.

    Pick the model and simulate **one°° cascade.
    `full_graph` should contain all multiple links for each hour, month, day
    """
    # Get the date
    day = model.ex_field.trange()[0]

    # Prepare cache for the rain.
    cache = TCACHE / "rain_cache"
    cache.mkdir(parents=True, exist_ok=True)

    raincache = cache / f"rain_cache_{day.isoformat()[:10]}.csv.gz"
    if raincache.is_file() and usecache:
        event_cache = pd.read_csv(raincache, index_col=0)
        event_cache.columns = [pd.Timestamp(c) for c in event_cache.columns]
    else:
        event_cache = {}

    node_capacity = model.graph.nodes()["capacity"].to_numpy()
    _full_graph = full_graph.subset(
        [("weekday", day.day_of_week == 6), ("month", day.month == 8)]
    )

    sts = {}
    for ev_hour in model.ex_field.extreme_events(kind="simple"):
        hour = ev_hour.trange()[0]

        if hour not in LOC_CACHE:
            if cached is not None and "rain" in cached:
                integral = cached["rain"]
            else:
                integral = None
            ggg = _full_graph.subset([("hour", hour.hour)])
            ggg = ggg.drop_duplicates()
            LOC_CACHE[hour] = generated_delay(
                ggg,
                ev_hour,
                trange=hour,
                weight="count",
                integral=integral,
            )
        else:
            print("Using cache.")
        model.evolve()
        model.generate(LOC_CACHE[hour], "beta")
        model.generate(-node_capacity, "gamma")
        model.conclude_step(hour.to_datetime64(), threshold=0.0, keep_cascade=False)

        sts[hour] = model.state_df

    states = pd.DataFrame(sts).stack()
    states = (
        states.loc[states > 0]
        .to_frame()
        .reset_index()
        .rename(columns={"level_0": "node", "level_1": "time", 0: "value"})
    )

    if not raincache.is_file() and usecache:
        pd.DataFrame(event_cache, index=model.graph.nodes().index).to_csv(raincache)

    # Return the first and only cascade.
    return states


def generated_delay(
    graph: diffsys.Graph,
    external_field: diffsys.ExternalField,
    trange: pd.Timestamp | tuple[pd.Timestamp, pd.Timestamp] | None = None,
    weight: str | float = "weight",
    integral: pd.Series | None = None,
    **kwargs: float,
) -> np.ndarray:
    """Compute the generated delay (before multipling by the parameter)."""
    if integral is None:
        if trange is None:
            trange = external_field.trange()
        integral = graph.integrate(trange=trange, ex_field=external_field, **kwargs)

    # Load the weight of each link (e.g. the number of trains going through).
    pass_count = graph.to_matrix(weight=weight)
    edge_weight = graph.to_matrix(weight=integral).multiply(pass_count)
    return edge_weight.sum(1).A.ravel()


def load_real_delay(year: int | list[int] | None):
    if year is None:
        return load_real_delay(list(range(2021, 2025)))

    if isinstance(year, int):
        return pd.read_csv(
            TCACHE / f"delays_per_stations_{year}.csv.gz",
            parse_dates=["time"],
            index_col="time",
        )

    return pd.concat([load_real_delay(y) for y in year], axis=0)


def add_axis_label(ax, text: str):
    ax.set_title(
        text,
        loc="left",
        fontsize="xx-large",
        fontweight="bold",
        ha="right",
    )
