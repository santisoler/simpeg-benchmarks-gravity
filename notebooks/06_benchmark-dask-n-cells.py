"""
Benchmark SimPEG running with Dask.
"""
from pathlib import Path
import itertools
import numpy as np
import xarray as xr
import SimPEG.dask
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
engine = "geoana"
height = 100
shape = (50, 50)  # grid of receivers
n_cells_per_axis = [20, 40, 60, 80, 100]

# Define mesh
mesh_spacings = (10, 10, 5)

# Define iterator over different scenarios
mesh_shapes = [(n, n, n) for n in n_cells_per_axis]
simulation_types = ["ram", "forward_only"]

# Build iterator
iterators = (simulation_types, mesh_shapes)
pool = itertools.product(*iterators)

# Allocate results arrays
array_shape = tuple(len(i) for i in iterators)
times = np.empty(array_shape)
errors = np.empty(array_shape)

for index, (store_sensitivities, mesh_shape) in enumerate(pool):
    print(f"store_sens: {store_sensitivities}, shape: {mesh_shape}")

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

    benchmarker = SimulationBenchmarker(n_runs=n_runs, verbose=False, **kwargs)
    runtime, std = benchmarker.benchmark(density)

    # Save result to arrays
    indices = np.unravel_index(index, array_shape)
    times[indices] = runtime
    errors[indices] = std


# Build Dataset
dims = ["simulation_type", "n_cells"]
coords = {
    "simulation_type": simulation_types,
    "n_cells": [np.prod(s) for s in mesh_shapes],
}

data_vars = {"times": (dims, times), "errors": (dims, errors)}
dataset = xr.Dataset(data_vars=data_vars, coords=coords)

# Save to file
results_dir = Path(__file__).parent / ".." / "results"
if not results_dir.is_dir():
    results_dir.mkdir(parents=True)
dataset.to_netcdf(results_dir / "benchmark_dask-n-cells.nc")
