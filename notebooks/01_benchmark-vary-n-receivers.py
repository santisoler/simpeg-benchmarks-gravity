from pathlib import Path
import time
import itertools
import numpy as np
import pandas as pd
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
n_runs = 3

height = 100
n_receivers_per_side = [10, 20, 30, 40, 50, 60, 100]
shapes = [(n, n) for n in n_receivers_per_side]
simulation_types = ["ram", "forward_only"]
engines = ["geoana", "choclo"]
pool = itertools.product(simulation_types, engines)

results = {
    "n_receivers": [np.prod(shape) for shape in shapes],
}

for store_sensitivities, engine in pool:
    # Run on single thread
    if engine == "choclo":
        kwargs = dict(choclo_parallel=False)
    else:
        kwargs = dict(n_processes=1)

    times = []
    times_std = []
    for shape in shapes:
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

        benchmarker = SimulationBenchmarker(n_runs=1, **kwargs)
        runtime, time_std = benchmarker.benchmark(density)

        times.append(runtime)
        times_std.append(time_std)

    # Get mean and std
    key = f"{engine}-{store_sensitivities}"
    results[key] = times
    results[key + "-std"] = times_std


# Generate a dataframe
df = pd.DataFrame(results).set_index("n_receivers")

# Save to a csv file
results_dir = Path(__file__).parent / ".." / "results"
if not results_dir.is_dir():
    results_dir.mkdir(parents=True)
df.to_csv(results_dir / "benchmark_n-receivers_serial.csv")


for simulation_type in simulation_types:
    for engine in engines:
        key = f"{engine}-{simulation_type}"
        times = df[key]
        errors = df[key + "-std"]
        plt.errorbar(
            x=times.index,
            y=times,
            yerr=errors,
            marker="o",
            linestyle="none",
            label=engine,
        )
    plt.legend()
    plt.show()
