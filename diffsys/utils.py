"""Utility functions."""

from pathlib import Path

import geopandas as geopd
import numpy as np
import pandas as pd
import rasterio as rio
import shapely
from pyproj import Geod
from rasterio import mask
from scipy.spatial import KDTree
from shapely.geometry import LineString, MultiPoint, Point, mapping

from .main import ExternalField

############################


def failing_probability(ext_field: ExternalField, objects: geopd.GeoDataFrame) -> pd.DataFrame:
    r"""Compute the probability of failing of nodes and edges in time.

    Attributes:
    ----------
    ext_field : ExternalField
        The externa Field that represents the probability of failing in each location
    objects : geopd.GeoDataFrame
        The nodes or edges where to compute the probability of failing
        They can be drawn from `Graph` (be it `Graph.nodes` or `Graph.edges`).
        It can be any `geopandas.GeoDataFrame` of `Point`s or `LineString`s.

    Returns:
    -------
    failing_prob : pd.DataFrame
        The failing prob per each node and edge at each temporal point.
        The index of the dataframe is the same of the provided `objects`.
        The columns of the dataframe represents the time steps (in `np.datetime64`)


    Note:
    ----
    For nodes, only the probability at position `Point(x, y)` is returned.
    For edges, the probability is:

    .. math::
        p = 1 - \prod_i (1-p_i)

    where \(p_i\) is the failing probability for lines at grid cell \(i\).

    """
    return pd.DataFrame(
        [
            ext_field.get_point(geom).data
            if isinstance(geom, Point)
            else ext_field.get_line(geom).data
            for geom in objects.geometry
        ],
        index=objects.index,
        columns=ext_field.time,
    )


def geodesic(nodes: geopd.GeoDataFrame, edges: pd.DataFrame, min_lenght: float = 1.0) -> float:
    """Compute the geodesic path for a link between two nodes.

    Parameters
    ----------
    nodes : geopd.GeoDataFrame
        The nodes ad Points
    edges : pd.DataFrame
        The edges for which to compute the geodesic line.
        Should contain the `source` and `target` columns.
    min_lenght : float
        the minimum lenght between points in the geodesic line.
        may be ExternalField.step

    Returns
    -------
    edges : geopd.GeoDataFrame
        The edges in a GeoDataFrame with the geodesic line as LineString

    TODO
    ----
    Possible improvements:

    - add geodesic lines in LineStrings segments
    - Interpolate the elevation

    """
    return geopd.GeoDataFrame(
        data=edges,
        geometry=[
            geodesic_line(
                nodes.loc[source]["geometry"],
                nodes.loc["target"]["geometry"],
                min_lenght=min_lenght,
            )
            for eid, source, target in edges[["source", "target"]].iterrows()
        ],
    )


def geodesic_line(p1: Point, p2: Point, min_lenght: float) -> LineString:
    """Compute the geodesic line between two points."""
    geod = Geod(ellps="clrk66")
    distance = shapely.distance(p1, p2)
    return LineString(
        geod.npts(
            p1.x,
            p1.y,
            p2.x,
            p2.y,
            int(distance / min_lenght) + 2,
            initial_idx=0,
            terminus_idx=0,
        ),
    )


def ckdnearest_points(gda: geopd.GeoDataFrame, gdb: geopd.GeoSeries) -> geopd.GeoDataFrame:
    """Find points in the second GeoDataFrame that are closer to those in the first.

    Will append a few columns to `gda`:
    - `_orig_indx_`: its index
    - `_dist_`: the distance between the points
    """
    na = np.array(list(gda.geometry.apply(lambda x: (x.x, x.y))))
    nb = np.array(list(gdb.geometry.apply(lambda x: (x.x, x.y))))
    btree = KDTree(nb)
    dist, idx = btree.query(na, k=1, workers=-1)
    return pd.concat(
        [
            gda.reset_index(drop=True),
            pd.Series(idx, name="_orig_indx_"),
            pd.Series(dist, name="_dist_"),
        ],
        axis=1,
    ).set_index(gda.index, drop=True)


def compute_voronois(
    points: geopd.GeoDataFrame,
    clip: shapely.Polygon | shapely.MultiPolygon | None = None,
) -> geopd.GeoDataFrame:
    """Compute the voronoi polygons of each point.

    Parameters
    ----------
    points : geopd.GeoDataFrame
        a geodataframe of points (shapely.geometry.Point)
    clip :
        a shapely geometry to clip the voronoi cells

    Returns
    -------
    voronois : geopd.GeoDataFrame
        the same geodataframe as input with an additional column named `voronoi`
        (shapely.geometry.Polygon)

    """
    # Just put all voronoi polygons in a GeoSeries
    polys = geopd.GeoDataFrame(
        {"voronoi": shapely.voronoi_polygons(MultiPoint(points["geometry"].tolist())).geoms},
        crs="4326",
        geometry="voronoi",
    )
    if clip is not None:
        polys = polys.clip(clip)
    # pick representative points
    polys["repr"] = geopd.GeoSeries([p.point_on_surface() for p in polys.geometry])

    # find the closer withing the representative points
    nearest = ckdnearest_points(points, geopd.GeoSeries(polys["repr"]))

    # check if the closer representative point fall within the voronoi cell
    # otherwise search for the right one.
    real_containing_poly = []
    polygon_taken = set()
    for point_indx, point in nearest.iterrows():
        poly = polys.loc[point["_orig_indx_"]]
        if poly["voronoi"].contains(point.geometry):
            # it's OK: the point fall within the polygon
            polygon_taken.add(poly.name)
            real_containing_poly.append({"indx": point_indx, "voronoi": poly["voronoi"]})
            continue

        # find the containing polygon the hard way
        for pl_indx, pl in polys.iterrows():
            if pl_indx not in polygon_taken and pl["voronoi"].contains(point.geometry):
                polygon_taken.add(pl_indx)
                real_containing_poly.append({"indx": point_indx, "voronoi": pl["voronoi"]})
                break

    real_containing_poly_df = pd.DataFrame(real_containing_poly).set_index("indx", drop=True)
    nearest.loc[real_containing_poly_df.index, "voronoi"] = real_containing_poly_df["voronoi"]
    return nearest.drop(columns=["_dist_", "_orig_indx_"])


def integrate_raster(polygons: geopd.GeoSeries, raster_file: str | Path) -> pd.Series:
    """Integrate the given raster in each polygon.

    Parameters
    ----------
    polygons : geopd.GeoSeries
        a Series of polygons (e.g. voronoi cells)
    raster_file : str | Path
        the path to a raster file (GeoTiff format).

    Returns
    -------
    integral : pd.Series
        A series with the raster integrated within polygons with the same index as `polygons`.

    """
    pop = []
    with rio.open(raster_file) as raster:
        for poly in polygons:
            # load population in that window
            try:
                win_pop, win_transform = mask.mask(
                    raster,
                    [mapping(poly)],
                    nodata=0.0,
                    filled=True,
                    crop=True,
                )
            except ValueError:
                pop.append(0)
                continue

            # Compute the population inside the polygon
            pop.append(np.nansum(win_pop))

    return pd.Series(pop, index=polygons.index, name="population")
