"""
Benchmark memory usage on a large problem: high number of cells and receivers
"""
import sys
from SimPEG import maps
from SimPEG.potential_fields.gravity.simulation import Simulation3DIntegral

from utilities import (
    get_region,
    create_observation_points,
    create_tensor_mesh_and_density,
    create_survey,
)


valid_engines = ("choclo", "geoana")
engine = str(sys.argv[1]).strip()
if engine not in valid_engines:
    raise ValueError(f"Invalid engine '{engine}'.")


# Define mesh
mesh_shape = (100, 100, 100)
mesh_spacings = (10, 10, 5)
mesh, active_cells, density = create_tensor_mesh_and_density(mesh_shape, mesh_spacings)

# Define survey
height = 100
shape = (120, 120)
grid_coords = create_observation_points(get_region(mesh), shape, height)
survey = create_survey(grid_coords)

# Define model map
model_map = maps.IdentityMap(nP=density.size)


# Configure benchmarks
# --------------------
store_sensitivities = "ram"


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
    kwargs["choclo_parallel"] = True
else:
    kwargs["n_processes"] = None

simulation = Simulation3DIntegral(**kwargs)
simulation.dpred(density)
