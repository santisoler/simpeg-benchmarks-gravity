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


# Define mesh
mesh_shape = (30, 30, 30)
mesh_spacings = (10, 10, 5)
mesh, active_cells, density = create_tensor_mesh_and_density(mesh_shape, mesh_spacings)


# Configure benchmarks
# --------------------
n_runs = 3
engine = "geoana"
height = 100
n_receivers_per_side = [20, 40, 60, 80, 100, 120]

# Define iterator over different scenarios
shapes = [(n, n) for n in n_receivers_per_side]
simulation_types = ["ram", "forward_only"]

# Build iterator
iterators = (simulation_types, shapes)
pool = itertools.product(*iterators)

# Allocate results arrays
array_shape = tuple(len(i) for i in iterators)
times = np.empty(array_shape)
errors = np.empty(array_shape)

for index, (store_sensitivities, shape) in enumerate(pool):
    print(f"store_sens: {store_sensitivities}, shape: {shape}")

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

    benchmarker = SimulationBenchmarker(n_runs=n_runs, **kwargs)
    runtime, std = benchmarker.benchmark(density)

    # Save result to arrays
    indices = np.unravel_index(index, array_shape)
    times[indices] = runtime
    errors[indices] = std


# Build Dataset
dims = ["simulation_type", "n_receivers"]
coords = {
    "simulation_type": simulation_types,
    "n_receivers": [np.prod(shape) for shape in shapes],
}

data_vars = {"times": (dims, times), "errors": (dims, errors)}
dataset = xr.Dataset(data_vars=data_vars, coords=coords)

# Save to file
results_dir = Path(__file__).parent / ".." / "results"
if not results_dir.is_dir():
    results_dir.mkdir(parents=True)
dataset.to_netcdf(results_dir / "benchmark_dask-n-receivers.nc")
