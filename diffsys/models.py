"""Models to simulate the evolution of the dinamical systems."""

from __future__ import annotations

import json
import typing
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

if typing.TYPE_CHECKING:
    from collections.abc import Callable, Generator
    from typing import Any

    import geopandas as geopd

    from . import main as diffsys


@dataclass
class Cascade:
    """A small dataclass to collect results from cascades.

    The cascades can account for the disruption of either nodes or edges
    (although the identifier should be a `str` for storage formatting).
    It also considers partial disruption: each node/edge is accompanied by
    a floating value representing its fractional working state.

    The `failing_0` and `failing_1` parameters can be either
    a list of failing nodes:

    ```
    ["node1", "node2", ...]
    ```

    or a list of dictionaries:

    ```
    [
        {'node': 'node1', 'value': val1, 'time': '2020-01-01T04:30', 'primary': True},
        {'node': 'node2', 'value': val2, 'time': '2020-01-01T04:45', 'primary': False},
        …
    ]
    ```

    In the latter case, `value` is used for nodes that do not completely fails,
    `time` for the time at which the node fails,
    `primary` is `True` if the node fails as direct consequence of the
    external field.

    Any of those entries (except `node`) can be omitted,
    and additional new entries can be added by each model.

    Parameters
    ----------
    failing_0 :
        The nodes (or edges) failing as direct consequence of the ExternalField.
        It may be a list of node ids or a list of dicts.
    failing_1 :
        The nodes (or edges) failing as consequence of the cascade.
        Same type as `failing_0`
    time :
        The time steps involved into this cascade.
    gcc :
        The fraction of nodes left in the greatest connected component.

    """

    failing_0: list[str | dict[str, Any]] = field(default_factory=list)
    failing_1: list[str | dict[str, Any]] = field(default_factory=list)
    time: list[np.datetime64] | np.datetime64 | None = None
    gcc: float | None = None

    def to_json(self) -> dict[str, Any]:
        """Convert to JSON serializable types."""
        return {
            "time": np.datetime_as_string(self.time).tolist()
            if self.time is not None
            else None,
            "failing_0": self.failing_0,
            "failing_1": self.failing_1,
            "gcc": self.gcc,
        }

    def df(self) -> pd.DataFrame:
        """Return the cascades as DataFrame."""
        if len(self) == 0:
            return pd.DataFrame([])

        return pd.DataFrame(
            [
                {"node": node, "failing": i}
                if isinstance(node, str)
                else node | {"failing": i}
                for i, failing in enumerate([self.failing_0, self.failing_1])
                for node in failing
            ],
        )

    def _failing_is_dict_(self) -> bool:
        return bool(isinstance(self.failing_0[0], dict))

    def add(
        self,
        failing_0: str | dict[str, Any] | None = None,
        failing_1: str | dict[str, Any] | None = None,
        timestep: np.datetime64 | None = None,
        gcc: int | None = None,
    ) -> Cascade:
        """Add new data."""
        if failing_0 is not None:
            self.failing_0.append(failing_0)
        if failing_1 is not None:
            self.failing_1.append(failing_1)
        if timestep is not None:
            if self.time is None:
                self.time = [timestep]
            elif isinstance(self.time, np.datetime64):
                self.time = [self.time, timestep]
            else:
                self.time.append(timestep)
        if gcc is not None:
            self.gcc = gcc
        return self

    def extend(
        self,
        failing_0: list[str] | list[dict[str, Any]] | None = None,
        failing_1: list[str] | list[dict[str, Any]] | None = None,
        timestep: list[np.datetime64] | None = None,
    ) -> Cascade:
        """Extend with new data (from iterables)."""
        if failing_0 is not None:
            self.failing_0.extend(failing_0)
        if failing_1 is not None:
            self.failing_1.extend(failing_1)
        if timestep is not None:
            if self.time is None:
                self.time = list(timestep)
            elif isinstance(self.time, np.datetime64):
                self.time = [self.time]
                self.time.extend(timestep)
            else:
                self.time.extend(timestep)
        return self

    def set_gcc(self, gcc: float) -> Cascade:
        """Set the great connected component size."""
        self.gcc = gcc
        return self

    def sum(self, key: str = "none") -> float:
        """Compute the sum (or number) of nodes involved."""
        if len(self) == 0:
            return 0

        s0 = sum([1 if isinstance(d, str) else d.get(key, 0) for d in self.failing_0])
        s0 += sum([1 if isinstance(d, str) else d.get(key, 0) for d in self.failing_1])

        return s0

    def __len__(self) -> int:
        """Return the size of the cascade."""
        return len(self.failing_0) + len(self.failing_1)


class PostInitCaller(type):
    """A metaclass to add `__post_init__` to the Models."""

    def __call__(cls, *args, **kwargs) -> PostInitCaller:
        obj = super().__call__(*args, **kwargs)
        obj.__post_init__()
        return obj


class Model(metaclass=PostInitCaller):
    """Represents a base class for all models.

    This class take as input the data and the climate to simulate the evolution of the dynanics.

    For each extreme event as defined in `ExternalField.extreme_events()` it should:

    - compute the nodes and/or edges that are interested by the extreme event
    - define the probability of those nodes edges to fail
    - start the simulation
    - return for each time-step the nodes failed, distinguising the nodes failed at each cycle
        (ex. nodes failed directly from the weather condition and node failed as a cascade).

    The above assumes that after the extreme event everything comes back to normality.
    This may be not the case, especially in cases of larger breaks of the network
    where a longer time is needed to return to normality (e.g. reconstraction of lines).
    Maybe a restoring mechanism should be put in place but this will increase the model complexity.
    In any case most of the extreme events last one day.
    """

    def __init__(
        self,
        graph: diffsys.Graph,
        ex_field: diffsys.ExternalField,
        **kwargs: str | float,
    ) -> None:
        """Initialize the Model.

        This is a `class` where all the modelling should occur.
        One needs to subclass this and code `self.run()` accordingly.

        Parameters
        ----------
        graph : adaptsys.Graph
            The network
        ex_field : adaptsys.ExternalField
            The failing probs at any point in time-space
        kwargs : dict
            Other parameters needed to perform the simulation
            Need to be json-serializable.

        """
        # Save the full graph
        self.graph = graph

        # External field
        self.ex_field = ex_field

        # Here we will collect the Cascades
        self._results: list[Cascade] = []

        # Here we will save the Model specific parameters
        # these will be saved in the `write` method.
        self._params = kwargs

        # placeholder for a filtering function
        self._graph_filter: Callable | None = None

    def __post_init__(self) -> None:
        """Redefine this method to fix parameters or set default values."""

    def graph_filter(
        self,
        filter_func: Callable[[np.datetime64, geopd.GeoDataFrame], geopd.GeoDataFrame],
    ) -> None:
        """Add a filtering function to the Model.

        This function will be used to filter the full graph
        if the edges are available only at a given time.
        """
        self._graph_filter = filter_func

    def cascades(self) -> Generator[Cascade]:
        yield from self._results

    def run(self) -> None:
        """Run the simulations.

        Each model needs to overwrite this funciton.

        **Important**: for each cascade, append to `self._results` a Cascade class
        """
        msg = "This method is not implemented for this Model yet."
        raise NotImplementedError(msg)

    def write(self, filepath: str | Path) -> None:
        """Write the Cascade result to json file.

        Parameters
        ----------
        filepath : str
            filename output
        save_nodes: bool
            if True save the list of initial nodes failed (failing_0)
            and the nodes failed due to the cascade (failed_1)
        **kwargs : dictionary
            Parameters of the simulation

        """
        with Path(filepath).open("w") as fout:
            json.dump(self.to_json(), fout, indent=4)

    def to_json(self) -> dict:
        """Return a json representation of the results and the parameters."""
        return {
            "parameters": self._params
            | {"num_nodes": self.graph.nn, "num_edges": self.graph.ne},
            "events": [res.to_json() for res in self._results if len(res) > 0],
        }

    def clear_cascades(self) -> None:
        """Clear the results. Free space."""
        self._results = []

    def __getitem__(self, key: str) -> str | float | None:
        """Get access to the parameters."""
        return self._params.get(key, None)

    def __len__(self) -> int:
        """Get the number of nodes."""
        return len(self.graph)

    @classmethod
    def load_cascades(cls, filepath: Path) -> dict[str, Any]:
        """Load the cascades from a file and return them."""
        with filepath.open("rt") as fin:
            data = json.load(fin)

        data["events"] = [Cascade(**ev) for ev in data["events"]]

        return data


class Diffusion(Model):
    """Propagate the external field influence according to a transfer matrix.

    Parameters
    ----------
    ppp: dict
        {'month': 'month-column', 'day': 'day-column'}

    """

    def __post_init__(self) -> None:
        """Fix the parameters."""
        self._params.setdefault("weight", 1.0)
        self._params.setdefault("alpha", 1.0)
        self._params.setdefault("beta", 1.0)
        self._params.setdefault("gamma", 1.0)

        # i -> node
        self._nodemap = dict(enumerate(self.graph.nodes().index))

        # placeholders
        self._cache: dict[str, Any] = {}
        self._new_cascade = Cascade()
        self._state = np.zeros(self.graph.nn)

    def set_initial_state(self, initial_state: np.ndarray) -> None:
        """Set the initial dynamical state to a given value."""
        self._state = initial_state

    def run(self) -> None:
        # Stations are assumed to have accumulated zero delay at the beginning
        raise NotImplementedError

    def run_event(self) -> None:
        raise NotImplementedError

    def evolve(self, alt_graph: Graph | None = None) -> None:
        """Evolve one timestep."""
        g = self.graph if alt_graph is None else alt_graph
        if "transition" in self._cache and alt_graph is None:
            transition = self._cache["transition"]
        else:
            transition = g.to_matrix(weight=self._params["weight"], normalize=True)
            self._cache["transition"] = transition

        if float(self._params["alpha"]) > 0.0:
            self._state = np.asarray(
                float(self._params["alpha"]) * transition @ self._state
            )

    def generate(
        self,
        node_gen: np.ndarray | float,
        param: str | float = 1.0,
    ) -> None:
        r"""Generate or degrade walkers.

        This step should not depend on the actual state.

        will perform:
        $$ s += p * v_\text{add} $$

        Parameters
        ----------
        node_gen : np.ndarray|float
            vector to be added to the state.
        param : str | int
            multiplicative factor
            If it is a string, it's the name of the parameter passed upon class generation.

        """
        use_param = self._params[param] if isinstance(param, str) else param
        self._state += use_param * node_gen
        self._state = np.clip(self._state, 0.0, np.inf)

    def adjust(
        self,
        alt_graph: diffsys.Graph | None = None,
        trange: pd.Timestamp | tuple[pd.Timestamp, pd.Timestamp] | None = None,
        integral: pd.Series | None = None,
        node_capacity: np.ndarray | None = None,
        edge_weight: np.ndarray | float | None = None,
        **kwargs: float | str,
    ) -> np.ndarray:
        r"""Generate or degrade walkers.

        This step should not depend on the actual state.

        will perform:
        $$ s = v_\text{mult} s + v_\text{add} $$
        """
        warnings.warn("Do not use adjust if possible", DeprecationWarning)
        if edge_weight is None:
            g = self.graph if alt_graph is None else alt_graph
            if integral is None:
                integral = g.integrate(ex_field=self.ex_field, trange=trange, **kwargs)

            # Load the weight of each link (e.g. the number of trains going through).
            if "pass_count" in self._cache and alt_graph is None:
                pass_count = self._cache["pass_count"]
            else:
                pass_count = g.to_matrix(weight=self._params["weight"])
                self._cache["pass_count"] = pass_count

            edge_weight = g.to_matrix(weight=integral).multiply(pass_count)
            edge_weight = edge_weight.sum(1).A.ravel()

        self._state += self._params["beta"] * edge_weight
        if node_capacity is not None:
            self._state -= self._params["gamma"] * node_capacity

        return edge_weight

    def conclude_step(
        self,
        time: np.datetime64,
        threshold: float = 0,
        keep_cascade: bool | None = None,
    ) -> None:
        """Update cascade with newly affected nodes."""
        # clip negative occupancy!
        self._state = np.clip(self._state, a_min=0, a_max=np.inf)
        if keep_cascade is None or keep_cascade:
            self._new_cascade.extend(
                failing_0=[
                    {
                        "node": self.graph.nodes().index[i],
                        "value": float(val),
                        "time": pd.Timestamp(time).isoformat(),
                    }
                    for i, val in enumerate(self._state)
                    if val > threshold
                ],
            )

    @property
    def state(self) -> np.ndarray:
        return self._state

    @property
    def state_df(self) -> pd.Series:
        return pd.Series(self._state, index=self._nodemap.values())

    def conclude_cascade(self) -> None:
        self._results.append(self._new_cascade)
        self._state = np.zeros(self.graph.nn)
        self._new_cascade = Cascade()
