from pyblenderSDIC.meshes import create_axisymmetric_mesh, TriMesh3D, create_xy_heightmap_mesh
from py3dframe import Frame
import numpy as np
import os

"""
Lets create a connical mesh along the X axis (from X = 0 to X = 1) where the radius at a given X is (X+1)**2
Many solution can be used but for now, we will use a simple method of pyblenderSDIC
"""

# ==========
# 1. Create the frame to orient the mesh
# ==========

# The axis of revolution is the X axis -> it must be oriented along the Z-axis of the frame.

z_axis = np.array([1.0, 0.0, 0.0])

# Because the meshwill be closed, we dont matter for the x and y axis of the frame. Lets fix them to the canonical basis
x_axis = np.array([0.0, 1.0, 0.0])
y_axis = np.array([0.0, 0.0, 1.0])

# To simplify, set the origin of the frame to the origin of the mesh
origin = np.array([0.0, 0.0, 0.0])

frame = Frame(
    origin=origin,
    x_axis=x_axis,
    y_axis=y_axis,
    z_axis=z_axis
)


# ==========
# 2. Create the mesh
# ==========

# The radius of the mesh is given by the function f(x) = (x+1)**2
def f(x):
    return (x + 1)**2

# The mesh is defined between x = 0 and x = 1
height_bounds = (0, 1)

# According the documentation to close the mesh we need to set the theta bounds to (theta_0, theta_0 + 2*pi*(1 - 1/Ntheta)).
mesh = create_axisymmetric_mesh(
    profile_curve=f,
    frame=frame,
    height_bounds=height_bounds,
    theta_bounds=(0, 2*np.pi*(1 - 1/20)),
    Nheight=10, # Number of node along the height
    Ntheta=20, # Number of node along the theta
    closed=True,
    first_diagonal=False,
    direct=True,
)

mesh.visualize()

# Save the mesh to a file
filepath = os.path.join(os.path.dirname(__file__), "cone.vtk")
mesh.save_to_vtk(filepath)










"""
Lets create a wave plane mesh with a normal vector along the Z axis.
The wave is defined by the function f(x, y) = 0.5 * np.sin(np.pi * x) * np.cos(np.pi * y).
The mesh is defined between x = -1 and x = 1 and y = -1 and y = 1.
"""

# The frame is the default one with the Z axis along the normal vector of the mesh, we don't need to give it to the function

# The wave is defined by the function f(x, y) = 0.5 * np.sin(np.pi * x) * np.cos(np.pi * y)

def f(x, y):
    return 0.5 * np.sin(np.pi * x) * np.cos(np.pi * y)

# The mesh is defined between x = -1 and x = 1 and y = -1 and y = 1
x_bounds = (-1, 1)
y_bounds = (-1, 1)

# Create the mesh
mesh = create_xy_heightmap_mesh(
    height_function=f,
    x_bounds=x_bounds,
    y_bounds=y_bounds,
    Nx=50, # Number of node along the x axis
    Ny=50, # Number of node along the y axis
)

# Visualize the mesh
mesh.visualize()

# Save the mesh to a file
filepath = os.path.join(os.path.dirname(__file__), "wave_plane.vtk")
mesh.save_to_vtk(filepath)
