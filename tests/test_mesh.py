import numpy as np
import pytest
from pyblenderSDIC.meshes import TriMesh3D, IntersectPoints, create_axisymmetric_mesh, create_xy_heightmap_mesh

@pytest.fixture
def mesh():
    """Fixture to create a fresh TriMesh3D with some data for each test."""
    nodes = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]])
    elements = np.array([[0, 2, 1], [0, 3, 2], [0, 1, 3], [1, 4, 3], [2, 3, 4], [2, 4, 1]])
    uvmap = np.array([[0.0, 0.0], [0.1, 0.1], [0.2, 0.2], [0.3, 0.3], [0.4, 0.4]])

    tmesh = TriMesh3D(points=nodes, cells={"triangle": elements})
    tmesh.uvmap2D = uvmap

    return tmesh

def test_trimesh3d_initialization(mesh):
    assert mesh.Nelements == 6
    assert mesh.nodes.shape == (5, 3)
    assert mesh.elements.shape == (6, 3)

def test_trimesh3d_setters(mesh):
    new_nodes = np.array([[10.0, 11.0, 12.0], [10.1, 11.1, 12.1], [10.2, 11.2, 12.2], [10.3, 11.3, 12.3], [10.4, 11.4, 12.4]])
    new_elements = np.array([[2, 1, 0], [3, 2, 1], [3, 0, 2], [4, 3, 1], [4, 2, 3], [4, 1, 2]])
    new_uvmap = np.array([[0.5, 0.5], [0.6, 0.6], [0.7, 0.7], [0.8, 0.8], [0.9, 0.9]])
    mesh.elements = new_elements.copy()
    mesh.nodes = new_nodes.copy()
    mesh.uvmap2D = new_uvmap.copy()
    assert np.allclose(mesh.nodes, new_nodes)
    assert np.allclose(mesh.elements, new_elements)
    assert np.allclose(mesh.uvmap2D, new_uvmap)
    assert mesh.Nelements == 6

    # Setter without copy
    nodes = mesh.nodes
    elements = mesh.elements

    nodes[0, 0] = 100.0
    elements[0, 0] = 100
    assert np.allclose(mesh.nodes, nodes)
    assert np.allclose(mesh.elements, elements)

def test_trimesh3d_validate(mesh):
    mesh.validate()

def test_trimesh3d_save_load_vtk(tmp_path, mesh):
    """Test saving and loading a TriMesh3D object to and from a file."""
    filepath = tmp_path / "test_mesh.vtk"
    mesh.save_to_vtk(str(filepath))

    loaded_mesh = TriMesh3D.load_from_vtk(str(filepath))
    assert np.allclose(loaded_mesh.nodes, mesh.nodes)
    assert np.array_equal(loaded_mesh.elements, mesh.elements)

    if mesh.uvmap is not None:
        assert np.allclose(loaded_mesh.uvmap, mesh.uvmap)

def test_trimesh3d_save_load_dict(mesh):
    """Test saving and loading a TriMesh3D object to and from a dictionary."""
    dict_data = mesh.save_to_dict()
    
    loaded_mesh = TriMesh3D.load_from_dict(dict_data)
    assert np.allclose(loaded_mesh.nodes, mesh.nodes)
    assert np.array_equal(loaded_mesh.elements, mesh.elements)

    if mesh.uvmap is not None:
        assert np.allclose(loaded_mesh.uvmap, mesh.uvmap)

def test_trimesh3d_save_load_json(tmp_path, mesh):
    """Test saving and loading a TriMesh3D object to and from a JSON file."""
    filepath = tmp_path / "test_mesh.json"
    mesh.save_to_json(str(filepath))
    
    loaded_mesh = TriMesh3D.load_from_json(str(filepath))
    assert np.allclose(loaded_mesh.nodes, mesh.nodes)
    assert np.array_equal(loaded_mesh.elements, mesh.elements)

    if mesh.uvmap is not None:
        assert np.allclose(loaded_mesh.uvmap, mesh.uvmap)

def test_open3d_visualize(mesh):
    """Test the Open3D visualization method with optional highlighting."""
    # Compute some intersection points to display
    origin = np.array([[0.4, 0.4, 0.5]])
    direction = np.array([[1.0, 0.0, 0.0]])
    rays = np.hstack((origin, direction))

    intersect_points = mesh.compute_intersect_points(rays)

    # Show the mesh highlighting the intersected element and the intersection points
    mesh.visualize(element_highlighted=intersect_points.id, intersect_points=intersect_points)

def test_compute_intersect_points(mesh):
    """Test the ray intersection method using compute_intersect_points."""

    # Case 1: Rays with expected intersections
    origin_intersect = np.array([
        [0.5, 0.5, 0.5],   # Ray 1
        [0.7, 0.7, 0.7]    # Ray 2
    ])
    direction_intersect = np.array([
        [1.0, 0.0, 0.0],   # Direction 1
        [0.0, 1.0, 0.0]    # Direction 2
    ])
    rays_intersect = np.hstack((origin_intersect, direction_intersect))

    # Case 2: Rays without expected intersections
    origin_no_intersect = np.array([
        [2.0, 2.0, 2.0],   # Ray 3
        [1.5, 1.5, 1.5]    # Ray 4
    ])
    direction_no_intersect = np.array([
        [0.0, 0.0, 1.0],   # Direction 3
        [0.0, 0.0, -1.0]   # Direction 4
    ])
    rays_no_intersect = np.hstack((origin_no_intersect, direction_no_intersect))

    # Results for rays with intersections
    intersect_pts = mesh.compute_intersect_points(rays_intersect)
    assert np.all(intersect_pts.element_indices >= 0), "Intersections were expected but not found."

    # Results for rays without intersections
    no_intersect_pts = mesh.compute_intersect_points(rays_no_intersect)
    assert np.all(no_intersect_pts.element_indices == -1), "Intersections were detected where none were expected."

def test_create_axisymmetric_mesh():
    """Test the creation of an axisymmetric mesh."""
    # Define parameters for the axisymmetric mesh (demi-cylinder example)
    cylinder_mesh = create_axisymmetric_mesh(
        profile_curve=lambda z: 1.0,
        height_bounds=(-1.0, 1.0),
        theta_bounds=(-np.pi/4, np.pi/4),
        Nheight=10,
        Ntheta=20,
    )

    # Vizualize the created mesh
    cylinder_mesh.visualize()

    # Define parameters for the axisymmetric mesh (full cylinder example)
    cylinder_mesh = create_axisymmetric_mesh(
        profile_curve=lambda z: 1.0,
        height_bounds=(-1.0, 1.0),
        theta_bounds=(0.0, 2.0*np.pi*(1-1.0/50)),
        Nheight=10,
        Ntheta=50,
        closed=True,
        first_diagonal=True,
        direct=True,
    )

    # Vizualize the created mesh
    cylinder_mesh.visualize()

def test_create_xy_heightmap_mesh():
    """Test the creation of a XY heightmap mesh."""
    # Define parameters for the XY heightmap mesh
    height_function = lambda x, y: 0.5 * np.sin(np.pi * x) * np.cos(np.pi * y)
    xy_mesh = create_xy_heightmap_mesh(
        height_function=height_function,
        x_bounds=(-1.0, 1.0),
        y_bounds=(-1.0, 1.0),
        Nx=50,
        Ny=50,
        uv_layout=0,
    )

    # Vizualize the created mesh
    xy_mesh.visualize()


def test_element_data(mesh):
    """Test the element data functionality of TriMesh3D."""
    normals = mesh.compute_element_normals()
    centroides = mesh.compute_element_centroids()
    areas = mesh.compute_element_areas()

    assert normals.shape == (mesh.Nelements, 3)
    assert centroides.shape == (mesh.Nelements, 3)
    assert areas.shape == (mesh.Nelements,)