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
n_runs = 3
height = 100
shape = (70, 70)  # grid of receivers
mesh_spacings = (10, 10, 5)
mesh_shape = (50, 50, 50)

# Define iterator over different scenarios
simulation_types = ["ram", "forward_only"]
engines = ["choclo", "geoana"]
number_of_processes = [1, 5, 10, 20, 30, numba.config.NUMBA_NUM_THREADS]

# Build iterator
iterators = (engines, simulation_types, number_of_processes)
pool = itertools.product(*iterators)

# Allocate results arrays
array_shape = tuple(len(i) for i in iterators)
times = np.empty(array_shape)
errors = np.empty(array_shape)

for index, (engine, store_sensitivities, n_processes) in enumerate(pool):
    print(
        f"engine: {engine}, store_sens: {store_sensitivities}, "
        f"n_processes: {n_processes}"
    )

    # Define mesh
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
        if n_processes == 1:
            kwargs["choclo_parallel"] = False
        else:
            kwargs["choclo_parallel"] = True
        numba.set_num_threads(n_processes)
    else:
        kwargs["n_processes"] = n_processes

    benchmarker = SimulationBenchmarker(n_runs=n_runs, **kwargs)
    runtime, std = benchmarker.benchmark(density)

    # Save result to arrays
    indices = np.unravel_index(index, array_shape)
    times[indices] = runtime
    errors[indices] = std

# Build Dataset
dims = ["engine", "simulation_type", "n_processes"]
coords = {
    "engine": engines,
    "simulation_type": simulation_types,
    "n_processes": number_of_processes,
}

data_vars = {"times": (dims, times), "errors": (dims, errors)}
attrs = dict(n_cells=np.prod(mesh_shape), n_receivers=np.prod(shape))
dataset = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)

# Save to file
results_dir = Path(__file__).parent / ".." / "results"
if not results_dir.is_dir():
    results_dir.mkdir(parents=True)
dataset.to_netcdf(results_dir / "benchmark_n-processes.nc")
