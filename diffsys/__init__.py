"""The model.

The aim of this project is to provide a straightforward module for analysing network resilience.
This will:

- consider a graph in a spatial context,
  where nodes represent points in space,
  and edges are either straight lines or curves connecting the nodes.
- control the disruption of nodes and edges through an external field.
- implement various network dynamics to simulate the cascading effects of disruptions.

```mermaid

flowchart TD
    top[Topology]
    ext[External Field]
    sym{Symulation}
    dyn[Dynamical System Disruption]
    top --> sym
    ext --> sym
    sym --> dyn
```
"""

from .main import ExternalField, Graph, Link

__all__ = [
    "ExternalField",
    "Graph",
    "Link",
    "copernicus",
    "models",
    "utils",
]
