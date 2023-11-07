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

ENGINES = ("choclo", "geoana", "dask")
STORE_SENSITIVITIES = ("ram", "forward_only")

# Handle shell arguments
engine = str(sys.argv[1]).strip()
store_sensitivities = str(sys.argv[2]).strip()
if engine not in ENGINES:
    raise ValueError(f"Invalid engine '{engine}'.")
if store_sensitivities not in STORE_SENSITIVITIES:
    raise ValueError(f"Invalid store_sensitivities '{store_sensitivities}'.")

# Use dask if required
if engine == "dask":
    engine = "geoana"
    import SimPEG.dask


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
print(f"engine: {engine}, store_sensitivities: {store_sensitivities}")

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
