from pathlib import Path
import itertools
import numpy as np
import xarray as xr
from SimPEG import maps

from utilities import (
    SimulationBenchmarker,
    get_region,
    create_observation_points,
    create_tensor_mesh_and_density,
    create_survey,
)

PLOT = False

# Configure benchmarks
# --------------------
n_runs = 3
height = 100
shape = (50, 50)  # grid of receivers
n_cells_per_axis = [20, 40, 60, 80, 100]

# Define iterator over different scenarios
mesh_shapes = [(n, n, n) for n in n_cells_per_axis]
simulation_types = ["ram", "forward_only"]
engines = ["choclo", "geoana"]
parallelism = [False, True]

# Build iterator
iterators = (parallelism, simulation_types, engines, mesh_shapes)
pool = itertools.product(*iterators)

# Allocate results arrays
array_shape = tuple(len(i) for i in iterators)
times = np.empty(array_shape)
errors = np.empty(array_shape)

for index, (parallel, store_sensitivities, engine, mesh_shape) in enumerate(pool):
    print(
        f"parallel: {parallel}, store_sens: {store_sensitivities}, "
        f"engine: {engine}, mesh_shape: {mesh_shape}"
    )

    # Define mesh
    mesh_spacings = (10, 10, 5)
    mesh, active_cells, density = create_tensor_mesh_and_density(
        mesh_shape, mesh_spacings
    )

    grid_coords = create_observation_points(get_region(mesh), shape, height)
    survey = create_survey(grid_coords)
    model_map = maps.IdentityMap(nP=density.size)

    kwargs = dict(
        survey=survey,
        mesh=mesh,
        ind_active=active_cells,
        rhoMap=model_map,
        engine=engine,
        store_sensitivities=store_sensitivities,
    )
    if engine == "choclo":
        kwargs["numba_parallel"] = parallel
    else:
        if parallel:
            kwargs["n_processes"] = None
        else:
            kwargs["n_processes"] = 1

    benchmarker = SimulationBenchmarker(n_runs=n_runs, **kwargs)
    runtime, std = benchmarker.benchmark(density)

    # Save result to arrays
    indices = np.unravel_index(index, array_shape)
    times[indices] = runtime
    errors[indices] = std


# Build Dataset
dims = ["parallel", "simulation_type", "engine", "n_cells"]
coords = {
    "parallel": parallelism,
    "simulation_type": simulation_types,
    "engine": engines,
    "n_cells": [np.prod(shape) for shape in mesh_shapes],
}

data_vars = {"times": (dims, times), "errors": (dims, errors)}
attrs = dict(n_receivers=np.prod(shape))
dataset = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)

# Save to file
results_dir = Path(__file__).parent / ".." / "results"
if not results_dir.is_dir():
    results_dir.mkdir(parents=True)
dataset.to_netcdf(results_dir / "benchmark_n-cells.nc")
