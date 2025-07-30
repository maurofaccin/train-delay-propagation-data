# Train propagation data and code

In this repository we held the code and data needed to reproduce the figures of the paper:

> Transport network response to external stressor fields\
> M. Faccin, T. Scagliarini and M. De Domenico

Additional data should be downloaded from other public sources.

## Additional data

### Climate data

Climate data should be downloaded from [the Copernicus project](https://atlas.climate.copernicus.eu/atlas).

To ease your task, you can use a [Copernicus helper](https://github.com/maurofaccin/copernicus_helper).


## Figures

### Model validation

Usage:

```
./code_validation.py
```

Warnings: it will take approx a couple of hours in a 10 cores 16 GB RAM laptop.

### Perturbability

Usage:

```
./code_coverage.py
./code_coverage_plot.py
```

### Steady state

Usage:

```
./code_steady_state.py
./code_steady_state_plot.py
```

Warning: Before running this you should run both the scripts for computing the perturbability.
