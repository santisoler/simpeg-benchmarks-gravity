from pathlib import Path
import itertools
import numba
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
n_runs = 10
height = 100
shape = (50, 50)  # grid of receivers
mesh_shape = (100, 100, 100)

# Define iterator over different scenarios
simulation_types = ["ram", "forward_only"]
engines = ["geoana", "choclo"]
number_of_processes = [1, 5, 10, 20, 30, 40, None]

# Build iterator
iterators = (simulation_types, engines, number_of_processes)
pool = itertools.product(*iterators)

# Allocate results arrays
array_shape = tuple(len(i) for i in iterators)
times = np.empty(array_shape)
errors = np.empty(array_shape)

for index, (store_sensitivities, engine, n_processes) in enumerate(pool):
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
        numba.set_num_threads(n_processes)
        if n_processes == 1:
            kwargs["choclo_parallel"] = False
        else:
            kwargs["choclo_parallel"] = True
    else:
        kwargs["n_processes"] = n_processes

    benchmarker = SimulationBenchmarker(n_runs=n_runs, **kwargs)
    runtime, std = benchmarker.benchmark(density)

    # Save result to arrays
    indices = np.unravel_index(index, array_shape)
    times[indices] = runtime
    errors[indices] = std


# Build Dataset
dims = ["simulation_type", "engine", "n_processes"]
coords = {
    "simulation_type": simulation_types,
    "engine": engines,
    "n_processes": number_of_processes,
}

data_vars = {"times": (dims, times), "errors": (dims, errors)}
dataset = xr.Dataset(data_vars=data_vars, coords=coords)

# Save to file
results_dir = Path(__file__).parent / ".." / "results"
if not results_dir.is_dir():
    results_dir.mkdir(parents=True)
dataset.to_netcdf(results_dir / "benchmark_n-processes.nc")
