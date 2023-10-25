from pathlib import Path
import time
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from SimPEG import maps
from SimPEG.potential_fields import gravity as simpeg_gravity

from utilities import (
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

    for shape in shapes:
        grid_coords = create_observation_points(get_region(mesh), shape, height)
        survey = create_survey(grid_coords)
        model_map = maps.IdentityMap(nP=density.size)

        times = []
        print(store_sensitivities, engine, shape)
        for i in range(n_runs):
            if i == 0 and engine == "choclo":
                # Compile first
                simulation = simpeg_gravity.simulation.Simulation3DIntegral(
                    survey=survey,
                    mesh=mesh,
                    ind_active=active_cells,
                    rhoMap=model_map,
                    engine=engine,
                    store_sensitivities=store_sensitivities,
                    **kwargs,
                )
                simulation.fields(density)
                delete_simulation(simulation)

            simulation = simpeg_gravity.simulation.Simulation3DIntegral(
                survey=survey,
                mesh=mesh,
                ind_active=active_cells,
                rhoMap=model_map,
                engine=engine,
                store_sensitivities=store_sensitivities,
                **kwargs,
            )

            # Benchmark
            start = time.perf_counter()
            result_simpeg = simulation.fields(density)
            end = time.perf_counter()
            delete_simulation(simulation)

            times.append(end - start)

        # Get mean and std
        times = np.array(times)
        key = f"{engine}-{store_sensitivities}"
        if key not in results:
            results[key] = []
            results[key + "-std"] = []
        results[key].append(np.mean(times))
        results[key + "-std"].append(np.std(times))


# Generate a dataframe
df = pd.DataFrame(results).set_index("n_receivers")

# Save to a csv file
results_dir = Path(__file__) / ".." / "results"
if not results_dir.is_dir():
    results_dir.mkdir(parents=True)
df.to_csv(results_dir / "benchmark_n-receivers_serial.csv")


simulation_type = "ram"
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
