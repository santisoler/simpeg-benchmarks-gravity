from pathlib import Path
import itertools
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from SimPEG import maps
from SimPEG.potential_fields import gravity as simpeg_gravity

from utilities import (
    SimulationBenchmarker,
    delete_simulation,
    get_region,
    create_observation_points,
    create_tensor_mesh_and_density,
    create_survey,
)


# Define mesh
mesh_shape = (30, 30, 30)
mesh_spacings = (10, 10, 5)
mesh, active_cells, density = create_tensor_mesh_and_density(mesh_shape, mesh_spacings)


# Configure benchmarks
# --------------------
n_runs = 3
height = 100
n_receivers_per_side = [10, 20, 30, 40]

# Define iterator over different scenarios
shapes = [(n, n) for n in n_receivers_per_side]
simulation_types = ["ram", "forward_only"]
engines = ["geoana", "choclo"]

# Build iterator
iterators = (simulation_types, engines, shapes)
pool = itertools.product(*iterators)

# Allocate results arrays
array_shape = tuple(len(i) for i in iterators)
times = np.empty(array_shape)
errors = np.empty(array_shape)

for index, (store_sensitivities, engine, shape) in enumerate(pool):
    # Run on single thread
    if engine == "choclo":
        kwargs = dict(choclo_parallel=False)
    else:
        kwargs = dict(n_processes=1)

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
        kwargs["choclo_parallel"] = False
    else:
        kwargs["n_processes"] = 1

    benchmarker = SimulationBenchmarker(n_runs=n_runs, **kwargs)
    runtime, std = benchmarker.benchmark(density)

    # Save result to arrays
    indices = np.unravel_index(index, array_shape)
    times[indices] = runtime
    errors[indices] = std


# Build Dataset
dims = ["simulation_type", "engine", "n_receivers"]
coords = {
    "n_receivers": [np.prod(shape) for shape in shapes],
    "engine": engines,
    "simulation_type": simulation_types,
}

data_vars = {"times": (dims, times), "errors": (dims, errors)}
dataset = xr.Dataset(data_vars=data_vars, coords=coords)

# Save to file
results_dir = Path(__file__).parent / ".." / "results"
if not results_dir.is_dir():
    results_dir.mkdir(parents=True)
dataset.to_netcdf(results_dir / "benchmark_n-receivers_serial.nc")

# Plot
for simulation_type in simulation_types:
    for engine in engines:
        results = dataset.sel(engine=engine, simulation_type=simulation_type)
        plt.errorbar(
            x=results.n_receivers,
            y=results.times,
            yerr=results.errors,
            marker="o",
            linestyle="none",
            label=engine,
        )
    plt.title(f"{simulation_type}")
    plt.legend()
    plt.show()
