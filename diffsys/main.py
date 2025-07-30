"""Main packages with most important classes and functions."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import TYPE_CHECKING

import geopandas as geopd
import networkx as nx
import numpy as np
import pandas as pd
import shapely
import xarray
from pyproj import Transformer
from scipy import ndimage, sparse
from shapely import ops
from shapely.geometry import LineString, Point
from sklearn import preprocessing

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Hashable

_LINK_SIMB = {"auto": "↺ ", "dir": "->", "sym": "<->"}
_LINK_RE = {k: re.compile(v) for k, v in _LINK_SIMB.items()}


class Link:
    """A class to manage the link."""

    def __init__(
        self,
        source: Hashable,
        target: Hashable,
        *args: Iterable,
        directed: bool = False,
    ) -> None:
        """Initialize the class.

        Tip: This class is made such that if the link is directed we use
        a tuple otherwise we use a frozenset

        If more than two labels are passed,
        source will be the first and target the last.

        Attributes
        ----------
        source : Hashable
            The label of the source node.
        target : Hashable
            The label of the target node.
        … : Iterable[Hashable] (optional)
            Other node labels of the path
        directed : bool (default=False)
            If true, the link will be directed.

        """
        self.source = source
        self.target = target if len(args) == 0 else args[-1]

        self.path: tuple | frozenset = (source, target, *args)
        if not directed:
            if len(args) > 0:
                msg = (
                    "A symmetric link has to be of leght 2,"
                    f" not {len(args) + 2} for {source}, {target}, {args}."
                )
                raise ValueError(msg)
            self.path = frozenset(self.path)

        self.directed = directed

    @property
    def base(self) -> tuple | frozenset:
        """Return a simple form of the link.

        As a tuple (directed) or frozenset (undirected).
        """
        return self.path

    def __hash__(self) -> int:
        """Hash."""
        return hash(self.base)

    def __eq__(self, other: object) -> bool:
        """Equal."""
        if isinstance(other, Link):
            return self.base == other.base

        if self.directed and isinstance(other, tuple):
            return self.base == other

        if not self.directed and isinstance(other, frozenset):
            return self.base == other
        return False

    def __contains__(self, other: ...) -> bool:
        """Check if node is part of the link."""
        return other in self.base

    @classmethod
    def from_str(cls, string: str) -> Link:
        """Infer from a string."""
        if string.endswith(_LINK_SIMB["auto"]):
            return Link(
                string.removesuffix(_LINK_SIMB["auto"]),
                string.removesuffix(_LINK_SIMB["auto"]),
            )

        splitted = _LINK_RE["sym"].split(string, maxsplit=1)
        if len(splitted) > 1:
            return Link(*splitted)
        splitted = _LINK_RE["dir"].split(string, maxsplit=1)
        return Link(*splitted)

    def __str__(self) -> str:
        """Represent as a string."""
        if self.source == self.target:
            return f"{self.source}{_LINK_SIMB['auto']}"
        if isinstance(self.base, tuple):
            # Directed link
            return f"{self.source}{_LINK_SIMB['dir']}{self.target}"
        # Symmetric link
        return f"{self.source}{_LINK_SIMB['sym']}{self.target}"

    def __repr__(self) -> str:
        """Represent as a string."""
        return self.__str__()


class Graph:
    """Represent the graph, with edges between interacting nodes.

    It contrains the edges and vertices ad `GeoDataFrame`s.
    Nodes will be a list of (geographic) points.

    Tips: edges may include all the trajectory of the **real line** such that
    we can compute the exteranl influence on the entire line.

    """

    def __init__(
        self,
        nodes: geopd.GeoDataFrame,
        edges: geopd.GeoDataFrame,
        directed: bool | None = None,
        edge_cols: list[str] | None = None,
    ) -> None:
        """Initialize te Graph.

        Attributes
        ----------
        nodes : geopandas.GeoDataFrame
            a geodataframe of nodes with all metatada and a location
            the index is the node label
            (for the geometry column, only `shapely.Point`s are allowed)
        edges : geopandas.GeoDataFrame
            a geodataframe of edges.
            The source and target of the link should be stored in two columns
            (by default named `source` and `target`).
            The geometry can be a straight line (from source to target node)
            or a shaped line (still starting from source and ending in target)
            of type `shapely.LineString`.
            Additional metatada can be saved in extra columns.
        directed : bool
            this is to determine whether the links are directed or symmetric
            (default: directed).
        edge_cols : list[str]
            a list of column names that represent the source and target of the edges.
            This may contains more than two column names for paths (in that case the
            graph is forced to be directed).
            (default: `["source", "target"]`)

        """
        self._ec = ["source", "target"] if edge_cols is None else edge_cols
        self._nodes = nodes
        directed = False if directed is None else directed

        # save edges
        to_rename = not isinstance(edges.index[0], Link)
        self._edges: geopd.GeoDataFrame = (
            geopd.GeoDataFrame(
                edges.rename(
                    {
                        old_indx: Link(
                            *[edge_data.loc[col] for col in self._ec], directed=directed
                        )
                        for old_indx, edge_data in edges.iterrows()
                    },
                ),
                geometry="geometry",
                crs="4326",
            )
            if to_rename
            else edges
        )
        self._directed = directed

        # Here we save the neighbors of nodes and edges for easier lookup
        self._node_successors: dict[Hashable, set[Hashable]] = {}
        self._node_predecessors: dict[Hashable, set[Hashable]] = {}

    def subset(self, filters: list[tuple]) -> Graph:
        """Return a subset of graph.

        The new graph will have the same nodes and filtered edges.

        Parameters
        ----------
        filters : list[tuple]
            list of columns to check. Each `tuple` should be of the type
            `(column: str, value: Any)`.
            Only edges in which `edges[column] == value` are kept.

        Return
        ------
        graph : Graph
            The filtered graph.

        """
        edges = self._edges
        for column, value in filters:
            edges = edges[edges[column] == value]

        return Graph(
            self.nodes(),
            geopd.GeoDataFrame(
                edges, geometry="geometry"
            ),  # Force to be of time `GeoDataFrame`
            directed=self._directed,
            edge_cols=self._ec,
        )

    def filter(
        self, filter_func: Callable[[geopd.GeoDataFrame], geopd.GeoDataFrame]
    ) -> Graph:
        """Filter the edges based on a function.

        Parameters
        ----------
        filter_func : func
            this should be a function that takes the full list of edges
            (a GeoDataFrame) and spits a filtered list of edges.
            The only parameter should be the GeoDataFrame.
            (If you need more parameters, you can use functools.partial).

        Return
        ------
        graph : Graph
            The new graph with the same nodes and parameters and a subset of edges.

        """
        return Graph(
            self._nodes,
            filter_func(self._edges),
            directed=self._directed,
            edge_cols=self._ec,
        )

    def _compute_node_neighbors(self) -> None:
        """Compute the list of successors and predecessors for nodes.

        The symmetric case has no predecessors and all neighbors are successors.
        """
        for edge in self._edges.index:
            source, target = edge.source, edge.target
            if source == target:
                continue

            self._node_successors.setdefault(source, set()).add(target)
            if self._directed:
                self._node_predecessors.setdefault(target, set()).add(source)
            else:
                self._node_successors.setdefault(target, set()).add(source)

    def successors(self, node: Hashable) -> set[Hashable]:
        """Successors of nodes.

        In the symmetric case, it represents the neighbors of nodes.

        In the directed case it represents the nodes connected to outgoing edges.
        """
        if len(self._node_successors) == 0:
            # compute lazely
            self._compute_node_neighbors()
        return self._node_successors.get(node, set())

    def predecessors(self, node: Hashable) -> set[Hashable]:
        """Predecessors of nodes.

        In the symmetric case it's empty.

        In the directed case it represents the nodes connected to incoming edges.
        """
        if len(self._node_successors) == 0:
            # compute lazely
            self._compute_node_neighbors()
        return self._node_predecessors.get(node, set())

    def nodes(self, subset: list[Hashable] | None = None) -> geopd.GeoDataFrame:
        """Get the nodes dataframe."""
        if subset is None:
            return self._nodes

        return self._nodes.loc[subset]

    def node(self, label: Hashable) -> geopd.GeoSeries:
        """Get one node."""
        return self._nodes.loc[label]

    def edges(self, subset: Iterable | None = None) -> geopd.GeoDataFrame:
        """Get the edges ad dataframe."""
        if subset is None:
            return self._edges

        return self._edges.loc[[Link(s, t, directed=self._directed) for s, t in subset]]

    def edge(self, source: Hashable, target: Hashable) -> geopd.GeoSeries:
        """Get one edge."""
        return self._edges.loc[Link(source, target, directed=self._directed)]

    def drop_duplicates(self) -> Graph:
        """Remove duplicated edges."""
        return Graph(
            nodes=self._nodes,
            edges=self._edges.loc[~self._edges.index.duplicated()],
            edge_cols=self._ec,
            directed=self._directed,
        )

    def remove_edges(self, removed_edges: Iterable[Link]) -> None:
        """Remove edges from list."""
        self._edges = geopd.GeoDataFrame(self._edges.drop(removed_edges, inplace=False))
        # Clean successors and predecessors (will be computed when/if needed).
        self._node_successors = {}
        self._node_predecessors = {}

    def adjacency_matrix(self) -> sparse.spmatrix:
        """Return the adjacency matrix as sparce array."""
        max_index = self.nodes().index.max() + 1
        adj = sparse.csr_matrix(
            (
                np.ones(shape=(len(self.edges()),)),
                (self.edges().source, self.edges().target),
            ),
            shape=(max_index, max_index),
        )

        return adj + adj.transpose()

    def gcc(self, removed_nodes_edges: list[Hashable | Link] | None = None) -> int:
        """Return the size of the greatest connected component.

        Parameters
        ----------
        removed_nodes_edges : list
            Additional nodes or edges to be removed.

        Return
        ------
        size_gcc : int
            the size of the giant connected component.

        Return
        ------
        size_gcc : int
            Size of the giant connected component.

        """
        new_nx_graph = self.to_networkx()

        if removed_nodes_edges is None:
            components = nx.connected_components(new_nx_graph)
            return max(list(map(len, components)))

        new_nx_graph.remove_nodes_from(
            [x for x in removed_nodes_edges if not isinstance(x, Link)]
        )
        new_nx_graph.remove_edges_from(
            [x for x in removed_nodes_edges if isinstance(x, Link)]
        )

        if new_nx_graph.number_of_nodes() == 0 or new_nx_graph.number_of_edges() == 0:
            return 0

        components = nx.connected_components(new_nx_graph)
        return max(list(map(len, components)))

    @classmethod
    def read_topology(
        cls, path_nodelist: str, path_edgelist: str, directed: bool = False
    ) -> Graph:
        """Read topology from files  (edgelist and nodelist).

        Parameters
        ----------
        path_nodelist : str
            path to nodelist
            must be in format: `node_label lon lat metadata …`
        path_edgelist : str
            path to edgelist
            must have the format: `node_label1 node_label2 metadata …`
        directed : bool (default=False)
            if the graph should be considered as directed.

        Returns
        -------
        graph : Graph
            a graph from nodes and edges: graph(nodes, edges).

        """
        # TODO: per adesso stiamo considerando gli edges come linee dritte
        #       in futuro implementare una shape

        # load nodes.
        # First column is label
        # second and third is lon/lat
        df_nodes = (
            pd.read_csv(
                path_nodelist, sep=" ", dtype={0: str, 1: float, 2: float}, header=None
            )
            .rename({0: "label", 1: "lon", 2: "lat"}, axis=1)
            .set_index("label", drop=True)
        )
        gdf_nodes = geopd.GeoDataFrame(
            df_nodes.drop(columns=["lon", "lat"]),
            geometry=geopd.points_from_xy(df_nodes["lon"], df_nodes["lat"]),
        )

        # Load edges
        # first and second columns are the node labels
        df_edges = pd.read_csv(
            path_edgelist, sep=" ", header=None, dtype={0: str, 1: str}
        ).rename(
            {0: "source", 1: "target"},
            axis=1,
        )
        gdf_edges = geopd.GeoDataFrame(
            df_edges,
            geometry=[
                LineString(
                    [
                        gdf_nodes.loc[edge.source]["geometry"],
                        gdf_nodes.loc[edge.target]["geometry"],
                    ],
                )
                for _, edge in df_edges.iterrows()
            ],
        )

        return cls(gdf_nodes, gdf_edges, directed=directed)

    @classmethod
    def from_networkx(
        cls,
        network: nx.Graph | nx.DiGraph,
        longitude: str = "x",
        latitude: str = "y",
        elevation: str | None = None,
    ) -> Graph:
        """Read a networkx Graph or DiGraph.

        Parameters
        ----------
        network : nx.Graph | nx.DiGraph
            the network from networkx
        longitude: str
            the node attribute with the longitude variable (default: 'x')
        latitude: str
            the node attribute with the latitude variable (default: 'y')
        elevation : str
            the attribute that describes the elevation of the node.

        Returns
        -------
        graph : Graph
            The graph with all metadata saved

        """
        geo_nodes = geopd.GeoDataFrame(
            [data for _, data in network.nodes.data()],
            index=network.nodes(),
            geometry=[
                shapely.geometry.Point(p[longitude], p[latitude], p[elevation])
                if elevation
                else shapely.geometry.Point(p[longitude], p[latitude])
                for _, p in network.nodes.data()
            ],
            crs=3246,
        ).drop(columns=[longitude, latitude])

        geo_edges = geopd.GeoDataFrame(
            [
                data | {"source": n1, "target": n2}
                for n1, n2, data in network.edges.data()
            ],
            geometry=[
                shapely.geometry.LineString(
                    [
                        geo_nodes["geometry"].loc[source],
                        geo_nodes["geometry"].loc[target],
                    ],
                )
                for source, target, data in network.edges.data()
            ],
            crs=3246,
        )
        return cls(geo_nodes, geo_edges, directed=isinstance(network, nx.DiGraph))

    def to_networkx(self) -> nx.Graph | nx.DiGraph:
        """Return a networkx Graph.

        Returns
        -------
        graph : nx.Graph or nx.DiGraph
            The corresponding graph with all the metadata.

        """
        graph = nx.DiGraph() if self._directed else nx.Graph()
        graph.add_nodes_from(
            [
                (
                    node["id"],
                    node["properties"]
                    | dict(zip("xyz", node["geometry"]["coordinates"], strict=False)),
                )
                for node in self._nodes.iterfeatures()
            ],
        )
        graph.add_edges_from(
            [
                (edge.source, edge.target, edge_data)
                for edge, edge_data in self._edges.drop(
                    columns=["geometry"] + self._ec,
                ).iterrows()
            ],
        )
        return graph

    def to_matrix(
        self,
        weight: str | pd.Series | float = 1.0,
        normalize: bool | None = None,
    ) -> sparse.csr_array:
        """Tranform to a sparse matrix.

        Return the matrix nodes x nodes that encodes the given quantity,
        from the edges dataframe's column (e.g. `weight`, that column must be numerical).
        If no `weight` is given, the adjacency matrix is returned.

        Parameters
        ----------
        weight : str | None
            the column name to use as weight for the edges
        normalize : bool | None
            if True returns the right stochastic matrix (columns normalized).

        Return
        ------
        transfer matrix : sparse.spmatrix

        """
        node_map = pd.Series(np.arange(self.nn), index=self._nodes.index)

        if isinstance(weight, (int, float)):
            sources = [e.source for e in self._edges.index]
            targets = [e.target for e in self._edges.index]
            w = np.full(len(sources), fill_value=weight)
        elif isinstance(weight, str):
            sources = [e.source for e in self._edges.index]
            targets = [e.target for e in self._edges.index]
            w = self._edges[weight].fillna(1.0)
        elif isinstance(weight, pd.Series):
            _weight: pd.Series = weight.loc[weight > 0.0]
            sources = [e.source for e in _weight.index]
            targets = [e.target for e in _weight.index]
            w = _weight
        else:
            raise NotImplementedError

        spm = sparse.coo_matrix(
            (w, (node_map.loc[targets], node_map.loc[sources])),
            shape=(self.nn, self.nn),
            dtype=np.float64,
        )
        if not self._directed:
            spm += spm.T

        if normalize:
            spm = preprocessing.normalize(spm, norm="l1", axis=0)
        return spm.tocsr()

    def to_array(
        self, weight: str | None = None, normalize: bool | None = None
    ) -> np.ndarray:
        """Put nodes metadata in a vector. Possibly normalize."""
        arr = np.ones(len(self)) if weight is None else self._nodes[weight].to_numpy()
        if normalize is True:
            arr = preprocessing.normalize(arr, norm="l1")
        return arr

    def integrate(
        self,
        ex_field: ExternalField,
        trange: pd.Timestamp | tuple[pd.Timestamp, pd.Timestamp] | None = None,
        ds: float = 1.0,
        kind: str = "edges",
        verbose: bool | None = None,
    ) -> pd.Series:
        """Integrate the ExternalField."""
        trange = ex_field.trange() if trange is None else trange
        if verbose:
            from tqdm import tqdm

        if kind == "edges":
            edges = self._edges.geometry

            dlim = trange if isinstance(trange, Iterable) else (trange, trange)
            if ex_field.data.sel(time=slice(dlim[0], dlim[1])).max().data <= 0.0:
                return pd.Series(np.zeros(len(self._edges)), index=self._edges.index)
            objects = tqdm(edges) if verbose else edges
            return pd.Series(
                [
                    ex_field.path_integral(
                        p, date_lims=dlim, ds=ds, crs=self._edges.crs
                    )
                    for p in objects
                ],
                index=self._edges.index,
                name="path integral",
            )

        if kind == "nodes":
            tp = trange[0] if isinstance(trange, Iterable) else trange
            if ex_field.data.sel(time=tp).sum() <= 0.0:
                return pd.Series(np.zeros(len(self._nodes)), index=self._nodes.index)
            return pd.Series(
                [ex_field.get_point(p, timepoint=tp) for p in self._nodes.geometry],
                index=self._nodes.index,
                name="point integral",
            )

        msg = f"Not implemented integration `{kind}`, use `edges` or `nodes`"
        raise NotImplementedError(msg)

    def to_laplacian(
        self, kind: str = "RW", weight: str | float = 1.0
    ) -> sparse.spmatrix:
        """Return the laplacian."""
        if kind == "RW":
            return sparse.eye(self.nn) - self.to_matrix(weight=weight, normalize=True)
        if kind == "normal":
            adj = self.to_matrix(weight=weight, normalize=False)
            return sparse.diag(adj.sum(0).flatten()) - adj
        raise NotImplementedError

    def copy(self):
        # Return a new instance with the same data
        return Graph(self._nodes, self._edges, self._directed)

    @property
    def nn(self) -> int:
        """Number of nodes."""
        return len(self._nodes)

    @property
    def ne(self) -> int:
        """Number of edges."""
        return len(self._edges)

    def __str__(self) -> str:
        """Return a string description."""
        return f"A graph with {len(self._nodes)} nodes and {len(self._edges)} edges"

    def __len__(self) -> int:
        """Return the number of nodes."""
        return len(self._nodes)


class ExternalField:
    """Failing probability of nodes or part of edges in a gridded spatio-temporal field.

    Each cell of the spatio-temporal tensor described by this object represents
    the probability of failing of an object in that location and time.
    The object might be a node, an edge or a cetrain kind of nodes or edges.
    """

    def __init__(self, data: xarray.DataArray) -> None:
        """External Field.

        dims: "longitude", "latitude" "time"

        This is a spatio-temporal tensor.
        I don't know if there is a need for a fourth dimension as elevation.

        Parameters
        ----------
        data : xarray.DataArray
            a tensor with 3 dimensions: `longitude`, `latitude` and `time`.

        TODO
        ----
        Implement the fourth dimension: **elevation**.

        """
        if "longitude" not in data.coords:
            msg = "One of the coordinates should be `longitude`."
            raise ValueError(msg)
        if "latitude" not in data.coords:
            msg = "One of the coordinates should be `latitude`."
            raise ValueError(msg)
        if "time" not in data.coords:
            msg = "One of the coordinates should be `time`."
            raise ValueError(msg)
        # reoder coordinates such that time is first
        self._data: xarray.DataArray = data.transpose("time", ...)

    def get(self, day: str | pd.Timestamp | None = None) -> ExternalField:
        """Filter data."""
        if day is None:
            return self

        if isinstance(day, str):
            day = pd.Timestamp(day)
        start = pd.Timestamp(day.isoformat()[:10])
        end = start + pd.Timedelta(minutes=24 * 60 - 0.001)
        return ExternalField(self._data.sel(time=slice(start, end)))

    def clip(self, vmin: float = 30.0, vmax: float = 50.0) -> ExternalField:
        """Clip and respale values within vmin and vmax.

        Internal data are modified as follows:
              0                           if p < vmin
        p ->  (p - vmin) / (vmax - vmin)  if vmin < p < vmax
              1                           if p > vmax
        """
        self._data = ((self._data - vmin) / (vmax - vmin)).clip(0.0, 1.0)
        return self

    def add_min_prob(self, min_p: float) -> ExternalField:
        """Add a default probability of failure to all nodes.

        Attributes:
        ----------
        min_p : float
            the failing probability

        Return:
        ------
        field : ExternalField
            return self with min prob set as `min_p`

        """
        self._data = xarray.ufuncs.maximum(self._data, min_p)
        return self

    def threshold(self, value: float) -> ExternalField:
        """Apply a threshold.

        All places below the threshold have prob 0, otherwise 1.
        """
        self._data = (self._data >= value).astype(np.float64)
        return self

    def trange(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        """Return the fist and last datetime of the interval."""
        return tuple(pd.Timestamp(x) for x in self.data.time.to_numpy()[[0, -1]])

    def extreme_events(
        self,
        kind: str = "simple",
        threshold: float = 0.1,
    ) -> Generator[ExternalField]:
        """Find connected blobs in the climate tensor.

        Extreme events are computed following simple euristics:

        **Simple**:

        Each non empty timeslice is an extreme event.
        drawback: it consider far-apart events as one.

        **Blob**:

        Extreme events detected if temporally and spatially isolated extreme.
        This assumes that extreme events have independent effects on the network
        (which is safe to assume in most cases when events are rares).

        Parameters
        ----------
        kind :  str (otional, default='simple')
            Type of events:
                - simple: aggregated events per time-step
                - blob: find connected blobs and aggregate them (assigned to the first time).
        threshold : float (default: 0.1)
            The threshold for a probability to be considered in a blob.
            Ignored in `simple` kind.

        Yield
        -----
        events : iterator of ExternalField
            returns an iterator over events

        """
        if kind == "simple":
            for time in self.time:
                yield (
                    ExternalField(
                        self._data.sel(time=time, drop=True)
                        .assign_attrs(time=time)
                        .expand_dims(dim={"time": [time]}, axis=2)
                        .fillna(0),
                    )
                )

        elif kind == "daily":
            for _, day in self._data.groupby(["time.year", "time.month", "time.day"]):
                yield ExternalField(day)

        elif kind == "blob":
            labels, num_labels = ndimage.label(
                (self._data.fillna(0) > threshold).astype(int)
            )
            indxes = ndimage.value_indices(labels)
            del indxes[0]

            for indx in indxes.values():
                # time coords for this blob
                time_coords = self._data.time[np.sort(np.unique(indx[0]))]

                # restrict to the timesteps of this blob
                new_indx = indx[0] - np.min(indx[0])
                this_blob = np.zeros(
                    # Time, longitude, and latitude indices should be consecutive
                    (len(set(indx[0])), self._data.shape[1], self._data.shape[2]),
                )
                this_blob[new_indx, indx[1], indx[2]] = self._data.data[
                    indx[0], indx[1], indx[2]
                ]

                yield ExternalField(
                    xarray.DataArray(
                        this_blob,
                        coords={
                            "time": time_coords,
                            "longitude": self._data["longitude"],
                            "latitude": self._data["latitude"],
                        },
                        dims=["time", "longitude", "latitude"],
                        attrs={"time": time_coords[0].data},
                    ),
                )

        else:
            raise NotImplementedError("Not `simple` neither `blob` but " + str(kind))

    def get_point(
        self, point: Point, timepoint: pd.Timestamp | None = None
    ) -> xarray.DataArray:
        """Get the probability of failure of a `shapely.geometry.Point`."""
        if timepoint is None:
            return self._data.sel(
                longitude=point.x, latitude=point.y, method="nearest", drop=True
            )
        return self._data.sel(
            longitude=point.x,
            latitude=point.y,
            time=timepoint,
            method="nearest",
            drop=True,
        ).data.tolist()

    def get_line(self, line: LineString) -> xarray.DataArray:
        r"""Get the probability of failure of a line along the temporal coordinate.

        At each time step, the probability of failing is:

        ..math::
            Prob(fail) = 1 - \prod_i (1-p_i)

        """
        x, y = shapely.segmentize(line, self.step / 10).xy
        sel = self._data.sel(longitude=x, latitude=y, method="nearest", drop=True)

        # take just one point per cell
        xy = pd.DataFrame([sel.longitude.data, sel.latitude.data]).T.drop_duplicates()

        # failing probability time x cells
        probs = np.diagonal(
            self._data.sel(
                longitude=xy[0].to_numpy(), latitude=xy[1].to_numpy(), drop=True
            ),
            axis1=1,  # longitude
            axis2=2,  # latitude
        )
        # probs are computed as 1 - (1-p1)(1-p2)(1-p3)…
        probs = 1 - (1 - probs).prod(axis=1)

        return xarray.DataArray(
            probs,
            dims=["time"],
            coords={"time": self._data.time},
        )

    def path_integral(self, line: LineString, **kwargs) -> float:
        r"""Path integral of the External Field along the Line.

        Trapezoidal approximation

        Parameters
        ----------
        line : LineString
            path along which to perform the integral
        **kwargs :
            args to be passed to `self.path_values`

        Return
        ------
        integral : float
            integral value using the Trapezoidal rule
            1/S \int_0^S E(x,y,t)ds

        """
        vals, real_ds = self.path_values(line=line, **kwargs)
        # use the Trapezoidal rule
        intgr = vals[1:-1].sum() + 0.5 * vals[[0, -1]].sum()
        return intgr * real_ds

    def path_values(
        self,
        line: LineString,
        date_lims: tuple[pd.Timestamp, pd.Timestamp],
        ds: float = 1,
        crs: str = "EPSG:4326",
    ) -> tuple[np.ndarray, float]:
        r"""Path values of the External Field along the Line.

        Parameters
        ----------
        line : LineString
            path along which to perform the integral
        date_lims :
            initial and final datetime where to perform the integral
            middle points will be interpolated from those.
        ds : float
            unitary distance (in km).
            The line will be choppend in many pieces of approx `ds` length.
        crs: the crs of the original shape
            (will be transformed to 3857 to compute the distance in km)

        Return
        ------
        values : np.ndarray
            values at regular intervals
        real_ds :  float
            real interval lenght (km)

        """
        # Compute a series of point at approximately the same distance.
        # The problem here is that we are bound to the user `crs`
        # which may not provide exactly the same distance between interpolated points.
        n_segments = int(line.length / ds) + 1
        points = [
            line.interpolate(dist)
            for dist in np.linspace(0, line.length, n_segments + 1)
        ]
        dates = pd.date_range(date_lims[0], date_lims[1], periods=len(points))

        # compute the distance in Km
        km_transf = Transformer.from_crs(
            crs_from=crs, crs_to="EPSG:3857", always_xy=True
        )
        line_transf = ops.transform(km_transf.transform, line)
        # WARN: this is not the best since each little piece length
        #       is still computed as Euclidean distance.

        return self._data.sel(
            longitude=xarray.DataArray([p.x for p in points]),
            latitude=xarray.DataArray([p.y for p in points]),
            time=xarray.DataArray(dates),
            method="nearest",
        ).data, line_transf.length / 1000 / n_segments

    def extend(self, other: ExternalField, dim: str = "time") -> ExternalField:
        """Return an extended external field on the given dimension.

        Defaults to concat on the temporal dimension.
        """
        return ExternalField(xarray.concat([self._data, other.data], dim))

    @property
    def data(self) -> xarray.DataArray:
        """Return the data."""
        return self._data

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the shape."""
        return self._data.shape

    @property
    def time(self) -> np.ndarray:
        """Time as numpy array."""
        return self._data.time.to_numpy().astype(np.datetime64)

    @property
    def longitude(self) -> np.ndarray:
        """Longitude as numpy array."""
        return self._data.longitude.to_numpy()

    @property
    def latitude(self) -> np.ndarray:
        """Latitude as numpy array."""
        return self._data.latitude.to_numpy()

    @property
    def data_var(self) -> str:
        """Data variable as string."""
        # Fails when there are more data variables
        if len(list(self._data.data_vars)) > 1:
            msg = "We only want ONE var."
            raise ValueError(msg)
        return next(self._data.data_vars)

    @property
    def step(self) -> float:
        """Step of the grid in degree as float."""
        return float(np.abs(self._data["latitude"][1] - self._data["latitude"][0]))

    def sum(self) -> float:
        """Sum."""
        return self.data.sum().data

    def __len__(self) -> int:
        """Return the number of time steps."""
        return len(self._data.time)

    def __str__(self) -> str:
        """Return a string representation."""
        return f"External Field of {self.shape}."

    def __mul__(self, other: object) -> ExternalField:
        """Perform multiplication."""
        if isinstance(other, (int, float, xarray.DataArray)):
            return ExternalField(other * self._data)

        if isinstance(other, ExternalField):
            return ExternalField(self._data * other._data)

        return NotImplemented

    __rmul__ = __mul__
