import gc
import numpy as np
import verde as vd
import discretize
from discretize.utils import mkvc
from SimPEG.potential_fields import gravity as simpeg_gravity


def delete_simulation(simulation):
    """Properly delete a simulation"""
    del simulation._G
    del simulation
    gc.collect()


def get_region(mesh):
    """Get horizontal boundaries of the mesh."""
    xmin, xmax = mesh.nodes_x.min(), mesh.nodes_x.max()
    ymin, ymax = mesh.nodes_y.min(), mesh.nodes_y.max()
    return (xmin, xmax, ymin, ymax)


def create_tensor_mesh_and_density(shape, spacings):
    """Create a sample TensorMesh and a density array for it."""
    # Create the TensorMesh
    h = [d * np.ones(s) for d, s in zip(spacings, shape)]
    origin = (0, 0, -shape[-1] * spacings[-1])
    mesh = discretize.TensorMesh(h, origin=origin)
    # Create active cells
    active_cells = np.ones(mesh.n_cells, dtype=bool)
    # Create random density array
    rng = np.random.default_rng(seed=42)
    density = rng.uniform(low=-1.0, high=1.0, size=mesh.n_cells)
    return mesh, active_cells, density


def create_observation_points(region, shape, height):
    """Create sample observation points."""
    grid_coords = vd.grid_coordinates(
        region=region, shape=shape, adjust="spacing", extra_coords=height
    )
    return grid_coords


def create_survey(grid_coords, components=None):
    """Create a SimPEG gravity survey with the observation points."""
    if components is None:
        components = ["gz"]
    receiver_locations = np.array([mkvc(c.ravel().T) for c in grid_coords])
    receivers = simpeg_gravity.receivers.Point(
        receiver_locations.T, components=components
    )
    source_field = simpeg_gravity.sources.SourceField(receiver_list=[receivers])
    survey = simpeg_gravity.survey.Survey(source_field)
    return survey
