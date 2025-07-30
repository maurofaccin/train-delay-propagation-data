# Train propagation data and code

In this repository we held the code and data needed to reproduce the figures of the paper:

> Transport network response to external stressor fields\
> M. Faccin, T. Scagliarini and M. De Domenico

Additional data should be downloaded from other public sources.

## Data

```
.
├── data
│   ├── aggregate_transitions_learn_ita.csv.gz
│   ├── delays_per_stations_2020.csv.gz
│   ├── delays_per_stations_2021.csv.gz
│   ├── delays_per_stations_2022.csv.gz
│   ├── delays_per_stations_2023.csv.gz
│   ├── delays_per_stations_2024.csv.gz
│   ├── graph_edges.geojson
│   ├── graph_nodes_metadata.geojson
│   ├── ITA_regions.geojson
│   ├── rain_cache
│   │   ├── rain_cache_2024-01-01.csv.gz
│   │   ├── rain_cache_2024-01-02.csv.gz
│   │   ├── rain_cache_2024-01-03.csv.gz
│   │   ├── rain_cache_2024-01-04.csv.gz
│   │   ├── rain_cache_2024-01-05.csv.gz
│   │   └── ...
│   └── rainy_peaks.csv.gz
└── ...
```

This repository contains part of the data that are necessary to reproduce the results.
In particular:

#### The network

The files `./data/graph_edges.geojson` and `./data/graph_nodes_metadata.geojson` contains the edges and the nodes of the network in `geojson` format.
In particular, the edge file is just a list of `LineString`s describing the shape of each line, with `source` and `target` nodes.
The node file is a list of `Point`s (stations) with the following metadata:

- `index`: the name of the node
- `osm_name`: the name as reported by OpenStreetMap
- `region`: the region in which the station is located
- `pop`: the population associated to the station
- `delay_q10`: delays experienced by each station (10% percentile)
- `delay_q50`: delays experienced by each station (50% percentile)
- `delay_q90`: delays experienced by each station (90% percentile)
- `delay_mean`: mean delays experienced by each station
- `capacity`: the maximum number of trains going through a station in one hour
- `rain`: average rain at each station

#### The dynamics and perturbations

`./data/aggregate_transitions_learn_ita.csv.gz` and `./data/delays_per_stations_YYYY.csv.gz` contains aggregated dynamics and train delays.
The former contains the `count` times one train has gone through the line between `start` and `end` in the given `month`, `weekday` and `hour`.
The latter contains the cumulative real delay experienced each day by each station.

## Additional data

### Climate data

Climate data should be downloaded from [the Copernicus project](https://atlas.climate.copernicus.eu/atlas).

To ease your task, you can use a [Copernicus helper](https://github.com/maurofaccin/copernicus_helper).


## Code & Figures

This is the code to reproduce the figures in the manuscript.
The following is the structure of the code:

```
.
├── base.py
├── code_coverage_plot.py
├── code_coverage.py
├── code_steady_state_plot.py
├── code_steady_state.py
├── code_validation.py
├── data
│   └── ...
├── diffsys
│   ├── __init__.py
│   ├── main.py
│   ├── models.py
│   └── utils.py
├── pyproject.toml
├── README.md
└── uv.lock
```

One easy way to get the project up and running is to install [`uv`](https://docs.astral.sh/uv/) and run:

```
uv sync
```
this will install all the dependencies in a virtual environment (under the `.venv` folder).
To run each script you can run it with the following command:

```
uv run scriptname.py
```


### Model validation

Usage:

```
uv run code_validation.py
```

Warnings: it will take approx a couple of hours in a 10 cores 16 GB RAM laptop.

### Perturbability

Usage:

```
uv run code_coverage.py
uv run code_coverage_plot.py
```

### Steady state

Usage:

```
uv run code_steady_state.py
uv run code_steady_state_plot.py
```

Warning: Before running this you should run both the scripts for computing the perturbability.
