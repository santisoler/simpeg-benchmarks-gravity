"""
Benchmark a large problem: high number of cells and receivers
"""
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


# Define mesh
mesh_shape = (100, 100, 100)
mesh_spacings = (10, 10, 5)
mesh, active_cells, density = create_tensor_mesh_and_density(mesh_shape, mesh_spacings)

# Define survey
height = 100
shape = (300, 300)
grid_coords = create_observation_points(get_region(mesh), shape, height)
survey = create_survey(grid_coords)

# Define model map
model_map = maps.IdentityMap(nP=density.size)


# Configure benchmarks
# --------------------
n_runs = 1
store_sensitivities = "ram"

# Define iterator over different scenarios
engines = ["choclo", "geoana"]

# Build iterator
iterators = (engines,)
pool = itertools.product(*iterators)

# Allocate results arrays
array_shape = tuple(len(i) for i in iterators)
times = np.empty(array_shape)
errors = np.empty(array_shape)


for index, (engine,) in enumerate(pool):
    print(f"engine: {engine}")

    kwargs = dict(
        survey=survey,
        mesh=mesh,
        ind_active=active_cells,
        rhoMap=model_map,
        engine=engine,
        store_sensitivities=store_sensitivities,
    )

    if engine == "choclo":
        kwargs["numba_parallel"] = True
    else:
        kwargs["n_processes"] = None

    benchmarker = SimulationBenchmarker(n_runs=n_runs, **kwargs)
    runtime, std = benchmarker.benchmark(density)

    # Save result to arrays
    indices = np.unravel_index(index, array_shape)
    times[indices] = runtime
    errors[indices] = std


# Build Dataset
dims = ["engine"]
coords = {"engine": engines}

data_vars = {"times": (dims, times), "errors": (dims, errors)}
attrs = dict(n_cells=np.prod(mesh_shape), n_receivers=np.prod(shape))
dataset = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)

# Save to file
results_dir = Path(__file__).parent / ".." / "results"
if not results_dir.is_dir():
    results_dir.mkdir(parents=True)
dataset.to_netcdf(results_dir / "benchmark_large-problem.nc")
