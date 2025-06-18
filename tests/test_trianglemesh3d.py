import numpy
import pytest
from pyblenderSDIC.meshes import TriangleMesh3D, IntersectPoints, create_xy_heightmap_mesh, create_axisymmetric_mesh
import meshio
import open3d

@pytest.fixture
def tetra_mesh():
    """Fixture to create a fresh TriangleMesh3D with some data for each test."""
    vertices = numpy.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]])
    triangles = numpy.array([[0, 2, 1], [0, 3, 2], [0, 1, 3], [1, 4, 3], [2, 3, 4], [2, 4, 1]])
    uvmap = numpy.array([[0.0, 0.0], [0.1, 0.1], [0.2, 0.2], [0.3, 0.3], [0.4, 0.4]])

    tmesh = TriangleMesh3D(vertices=vertices, triangles=triangles)
    tmesh.set_vertices_uvmap(uvmap)
    tmesh.validate()

    return tmesh




def test_trianglemesh3d_initialization(tetra_mesh):
    assert tetra_mesh.Ntriangles == 6
    assert tetra_mesh.vertices.shape == (5, 3)
    assert tetra_mesh.triangles.shape == (6, 3)
    assert tetra_mesh.uvmap.shape == (6, 6)
    

def test_trianglemesh3d_vertices_setters(tetra_mesh):
    new_vertices = numpy.array([[10.0, 11.0, 12.0], [10.1, 11.1, 12.1], [10.2, 11.2, 12.2], [10.3, 11.3, 12.3], [10.4, 11.4, 12.4]])
    
    tetra_mesh.vertices = new_vertices.copy()
    
    assert numpy.allclose(tetra_mesh.vertices, new_vertices)
    assert tetra_mesh.Ntriangles == 6

    # Setter without copy
    vertices = tetra_mesh.vertices
    vertices[0, 0] = 100.0
    assert numpy.allclose(tetra_mesh.vertices, vertices)


def test_trianglemesh3d_triangles_setters(tetra_mesh):
    new_triangles = numpy.array([[2, 1, 0], [3, 2, 1], [3, 0, 2], [4, 3, 1], [4, 2, 3], [4, 1, 2]])
    
    tetra_mesh.triangles = new_triangles.copy()
    
    assert numpy.allclose(tetra_mesh.triangles, new_triangles)
    assert tetra_mesh.Ntriangles == 6

    # Setter without copy
    triangles = tetra_mesh.triangles
    triangles[0, 0] = 100
    assert numpy.allclose(tetra_mesh.triangles, triangles)


def test_trianglemesh3d_bounding_box(tetra_mesh):
    assert tetra_mesh.bounding_box is None
    tetra_mesh.compute_bounding_box()
    assert tetra_mesh.bounding_box is not None
    assert tetra_mesh.bounding_box.shape == (2, 3)


def test_trianglemesh3d_volume(tetra_mesh):
    assert tetra_mesh.volume is None
    tetra_mesh.compute_volume()
    assert tetra_mesh.volume is not None
    assert tetra_mesh.volume > 0.0


def test_trianglemesh3d_vertex_normals(tetra_mesh):
    assert tetra_mesh.vertex_normals is None
    tetra_mesh.compute_triangle_normals()
    tetra_mesh.compute_vertex_normals()
    assert tetra_mesh.vertex_normals is not None
    assert tetra_mesh.vertex_normals.shape == (5, 3)


def test_trianglemesh3d_triangle_normals(tetra_mesh):
    assert tetra_mesh.triangle_normals is None
    tetra_mesh.compute_triangle_normals()
    assert tetra_mesh.triangle_normals is not None
    assert tetra_mesh.triangle_normals.shape == (6, 3)


def test_trianglemesh3d_triangle_centroids(tetra_mesh):
    assert tetra_mesh.triangle_centroids is None
    tetra_mesh.compute_triangle_centroids()
    assert tetra_mesh.triangle_centroids is not None
    assert tetra_mesh.triangle_centroids.shape == (6, 3)


def test_trianglemesh3d_triangle_areas(tetra_mesh):
    assert tetra_mesh.triangle_areas is None
    tetra_mesh.compute_triangle_areas()
    assert tetra_mesh.triangle_areas is not None
    assert tetra_mesh.triangle_areas.shape == (6, )


def test_trianglemesh3d_from_to_meshio(tetra_mesh):
    meshio_obj = tetra_mesh.to_meshio()
    new_mesh = TriangleMesh3D.from_meshio(meshio_obj)
    
    assert numpy.allclose(new_mesh.vertices, tetra_mesh.vertices)
    assert numpy.array_equal(new_mesh.triangles, tetra_mesh.triangles)
    assert numpy.allclose(new_mesh.uvmap, tetra_mesh.uvmap)


def test_trianglemesh3d_from_to_open3d(tetra_mesh):
    open3d_mesh = tetra_mesh.to_open3d(legacy=False)
    new_mesh = TriangleMesh3D.from_open3d(open3d_mesh)
    
    assert numpy.allclose(new_mesh.vertices, tetra_mesh.vertices)
    assert numpy.array_equal(new_mesh.triangles, tetra_mesh.triangles)
    assert numpy.allclose(new_mesh.uvmap, tetra_mesh.uvmap)

    # With legacy=True
    open3d_mesh_legacy = tetra_mesh.to_open3d(legacy=True)
    new_mesh_legacy = TriangleMesh3D.from_open3d(open3d_mesh_legacy)

    assert numpy.allclose(new_mesh_legacy.vertices, tetra_mesh.vertices)
    assert numpy.array_equal(new_mesh_legacy.triangles, tetra_mesh.triangles)


def test_trianglemesh3d_from_to_vtk(tetra_mesh, tmp_path):
    filepath = tmp_path / "test_mesh.vtk"
    tetra_mesh.to_vtk(str(filepath))

    loaded_mesh = TriangleMesh3D.from_vtk(str(filepath))
    assert numpy.allclose(loaded_mesh.vertices, tetra_mesh.vertices)
    assert numpy.array_equal(loaded_mesh.triangles, tetra_mesh.triangles)
    assert numpy.allclose(loaded_mesh.uvmap, tetra_mesh.uvmap)


def test_trianglemesh3d_from_to_dict(tetra_mesh):
    dict_data = tetra_mesh.to_dict()
    
    loaded_mesh = TriangleMesh3D.from_dict(dict_data)
    assert numpy.allclose(loaded_mesh.vertices, tetra_mesh.vertices)
    assert numpy.array_equal(loaded_mesh.triangles, tetra_mesh.triangles)
    assert numpy.allclose(loaded_mesh.uvmap, tetra_mesh.uvmap)


def test_trianglemesh3d_from_to_json(tetra_mesh, tmp_path):
    filepath = tmp_path / "test_mesh.json"
    tetra_mesh.to_json(str(filepath))
    
    loaded_mesh = TriangleMesh3D.from_json(str(filepath))
    assert numpy.allclose(loaded_mesh.vertices, tetra_mesh.vertices)
    assert numpy.array_equal(loaded_mesh.triangles, tetra_mesh.triangles)
    assert numpy.allclose(loaded_mesh.uvmap, tetra_mesh.uvmap)


def test_trianglemesh3d_properties(tetra_mesh):
    assert tetra_mesh.is_edge_manifold == True
    assert tetra_mesh.is_vertex_manifold == True
    assert tetra_mesh.is_edge_manifold_with_boundary == True
    assert tetra_mesh.is_self_intersecting == False
    assert tetra_mesh.is_watertight == True
    assert tetra_mesh.is_orientable == True


def test_trianglemesh3d_intersect_points(tetra_mesh):
    """Test the ray intersection method using compute_intersect_points."""

    # Case 1: Rays with expected intersections
    origin_intersect = numpy.array([
        [0.5, 0.5, 0.5],   # Ray 1
        [0.7, 0.7, 0.7]    # Ray 2
    ])
    direction_intersect = numpy.array([
        [1.0, 0.0, 0.0],   # Direction 1
        [0.0, 1.0, 0.0]    # Direction 2
    ])
    rays_intersect = numpy.hstack((origin_intersect, direction_intersect))

    # Case 2: Rays without expected intersections
    origin_no_intersect = numpy.array([
        [2.0, 2.0, 2.0],   # Ray 3
        [1.5, 1.5, 1.5]    # Ray 4
    ])
    direction_no_intersect = numpy.array([
        [0.0, 0.0, 1.0],   # Direction 3
        [0.0, 0.0, -1.0]   # Direction 4
    ])
    rays_no_intersect = numpy.hstack((origin_no_intersect, direction_no_intersect))

    # Results for rays with intersections
    intersect_pts = tetra_mesh.cast_rays(rays_intersect)
    assert numpy.all(intersect_pts.triangle_indices >= 0), "Intersections were expected but not found."

    # Results for rays without intersections
    no_intersect_pts = tetra_mesh.cast_rays(rays_no_intersect)
    assert numpy.all(no_intersect_pts.triangle_indices == -1), "Intersections were detected where none were expected."

    points_coords = tetra_mesh.calculate_intersect_coordinates(intersect_pts)
    assert points_coords.shape == (intersect_pts.id.size, 3), "Coordinates shape mismatch."


def test_trianglemesh3d_intersect_points_visualization(tetra_mesh):
    """Test the Open3D visualization method with optional highlighting."""
    # Compute some intersection points to display
    origin = numpy.array([[0.4, 0.4, 0.5]])
    direction = numpy.array([[1.0, 0.0, 0.0]])
    rays = numpy.hstack((origin, direction))

    intersect_points = tetra_mesh.cast_rays(rays)

    # Show the mesh highlighting the intersected element and the intersection points
    tetra_mesh.visualize()
    tetra_mesh.visualize(
        pattern_path=None,
        highlighted_triangles=[0,3],
        highlight_color=(1.0, 0.0, 0.0),  # Red color for highlights
        intersect_points=intersect_points,
        intersect_color=(0.0, 1.0, 0.0),  # Green color for intersection points
        display_edges=True,
        edges_color=(0.0, 0.0, 1.0)  # Blue color for edges
    )




def test_create_xy_heightmap_mesh():
    """Test the creation of a XY heightmap mesh."""
    # Define parameters for the XY heightmap mesh
    height_function = lambda x, y: 0.5 * numpy.sin(numpy.pi * x) * numpy.cos(numpy.pi * y)

    xy_mesh = create_xy_heightmap_mesh(
        height_function=height_function,
        x_bounds=(-1.0, 1.0),
        y_bounds=(-1.0, 1.0),
        Nx=50,
        Ny=50,
        uv_layout=0,
    )

    # Check the properties of the created mesh
    assert xy_mesh.Nvertices == 2500  # 50 x 50 grid
    assert xy_mesh.Ntriangles == 2 * (50 - 1) * (50 - 1)  # 2 triangles per quad in a 50x50 grid
    assert xy_mesh.is_edge_manifold == False
    assert xy_mesh.is_edge_manifold_with_boundary == True
    assert xy_mesh.is_vertex_manifold == True
    assert xy_mesh.is_self_intersecting == False
    assert xy_mesh.is_watertight == False  # Not watertight because flat mesh
    assert xy_mesh.is_orientable == True

    # Vizualize the created mesh
    xy_mesh.visualize()



def test_create_axisymmetric_mesh():
    """Test the creation of an axisymmetric mesh."""
    # Define parameters for the axisymmetric mesh (demi-cylinder example)
    cylinder_mesh = create_axisymmetric_mesh(
        profile_curve=lambda z: 1.0,
        height_bounds=(-1.0, 1.0),
        theta_bounds=(-numpy.pi/4, numpy.pi/4),
        Nheight=10,
        Ntheta=20,
    )

    # Check the properties of the created mesh
    assert cylinder_mesh.Nvertices == 200  # 10 height vertices * 20 theta vertices
    assert cylinder_mesh.Ntriangles == 2 * (10 - 1) * (20 - 1)  # 2 triangles per quad in a 10x20 grid
    assert cylinder_mesh.is_edge_manifold == False
    assert cylinder_mesh.is_edge_manifold_with_boundary == True
    assert cylinder_mesh.is_vertex_manifold == True
    assert cylinder_mesh.is_self_intersecting == False
    assert cylinder_mesh.is_watertight == False  # Not watertight because flat mesh
    assert cylinder_mesh.is_orientable == True 


    # Vizualize the created mesh
    cylinder_mesh.visualize()

    # Define parameters for the axisymmetric mesh (full cylinder example)
    cylinder_mesh = create_axisymmetric_mesh(
        profile_curve=lambda z: 1.0,
        height_bounds=(-1.0, 1.0),
        theta_bounds=(0.0, 2.0*numpy.pi*(1-1.0/50)),
        Nheight=10,
        Ntheta=50,
        closed=True,
        first_diagonal=True,
        direct=True,
    )

    # Check the properties of the created mesh
    assert cylinder_mesh.Nvertices == 500  # 10 height vertices * 50 theta vertices
    assert cylinder_mesh.Ntriangles == 2 * (10 - 1) * (50 - 1 + 1)  # 2 triangles per quad in a 10x50 grid with closure
    assert cylinder_mesh.is_edge_manifold == False
    assert cylinder_mesh.is_edge_manifold_with_boundary == True
    assert cylinder_mesh.is_vertex_manifold == True
    assert cylinder_mesh.is_self_intersecting == False
    assert cylinder_mesh.is_watertight == False  # Not watertight because flat mesh
    assert cylinder_mesh.is_orientable == True

    # Vizualize the created mesh
    cylinder_mesh.visualize()














