import numpy as np
import os
import meshio

from py3dframe import Frame

from pyblenderSDIC import Camera, SpotLight, BlenderExperiment

from pyblenderSDIC.meshes import create_axisymmetric_mesh
from pyblenderSDIC.materials import MaterialBSDF, get_iron_material, get_copper_material
from pyblenderSDIC.patterns import get_mouchtichu_path, get_speckle_path

"""
We observe a demi-cylinder with a mouchtichu pattern on it.
The cylinder evolves in time so we need 2 cylinders instances.

To observe the cylinder, we use two cameras fixed in the scene.
The cameras are the same for time 0 and time 1.
"""

# =======================================================================
# ========== Example of how to use the package `pyblenderSDIC` ==========
# =======================================================================

# ====================
# 1. CREATE THE MESH
# ====================

# Set the cylinder parameters
cylinder_center = np.array([0.0, 0.0, 10.0])
cylinder_z_axis = np.array([0.0, 0.0, 1.0])
cylinder_x_axis = np.array([1.0, 0.0, 0.0])
cylinder_y_axis = np.cross(cylinder_z_axis, cylinder_x_axis)

cylinder_frame = Frame(
    origin=cylinder_center,
    x_axis=cylinder_x_axis,
    y_axis=cylinder_y_axis,
    z_axis=cylinder_z_axis
)

cylinder_radius = 5.0
cylinder_height_min = -2.0
cylinder_height_max = 2.0
cylinder_theta_min = -0.25
cylinder_theta_max = 0.25
cylinder_Nheight = 100
cylinder_Ntheta = 100

cylinder_mesh_time_0 = create_axisymmetric_mesh(
    profile_curve=lambda z: cylinder_radius,
    frame=cylinder_frame,
    height_bounds=(cylinder_height_min, cylinder_height_max),
    theta_bounds=(cylinder_theta_min, cylinder_theta_max),
    Nheight=cylinder_Nheight,
    Ntheta=cylinder_Ntheta,
    closed=False,
)

# Assume the cylinder is extended at time 1 and it radius is reduced
# The total volume of the cylinder is conserved
# Volume = pi * R^2 * H
deformation_factor = 0.1
new_radius = cylinder_radius * (1 - deformation_factor)
old_height = cylinder_height_max - cylinder_height_min
new_height = old_height * (cylinder_radius / new_radius)**2

cylinder_mesh_time_1 = create_axisymmetric_mesh(
    profile_curve=lambda z: new_radius,
    frame=cylinder_frame,
    height_bounds=(-new_height/2, new_height/2),
    theta_bounds=(cylinder_theta_min, cylinder_theta_max),
    Nheight=cylinder_Nheight,
    Ntheta=cylinder_Ntheta,
    closed=False,
)


# ====================
# 2. CREATE THE CAMERAS
# ====================
camera_1_position = np.array([120.0, 0.0, 50.0])
camera_1_target = cylinder_center + cylinder_radius*cylinder_x_axis
camera_1_z_axis = camera_1_target - camera_1_position
camera_1_y_axis = np.array([0.0, 1.0, 0.0])
camera_1_x_axis = np.cross(camera_1_y_axis, camera_1_z_axis)
camera_1_frame = Frame(
    origin=camera_1_position,
    x_axis=camera_1_x_axis,
    y_axis=camera_1_y_axis,
    z_axis=camera_1_z_axis
)

camera_1 = Camera(
    frame=camera_1_frame,
    intrinsic_matrix=np.array([[20000.0, 0.0, 499.5], [0.0, 20000.0, 503.5], [0.0, 0.0, 1.0]]),
    resolution=(1000, 1000), # Image resolution
    pixel_size=(0.01, 0.01), # Pixel size in mm
)

camera_2_position = np.array([120.0, 0.0, -20.0])
camera_2_target = cylinder_center + cylinder_radius*cylinder_x_axis
camera_2_z_axis = camera_2_target - camera_2_position
camera_2_y_axis = np.array([0.0, 1.0, 0.0])
camera_2_x_axis = np.cross(camera_2_y_axis, camera_2_z_axis)
camera_2_frame = Frame(
    origin=camera_2_position,
    x_axis=camera_2_x_axis,
    y_axis=camera_2_y_axis,
    z_axis=camera_2_z_axis
)

camera_2 = Camera(
    frame=camera_2_frame,
    intrinsic_matrix=np.array([[23000.0, 0.0, 499.5], [0.0, 21000.0, 503.5], [0.0, 0.0, 1.0]]),
    resolution=(1000, 1000), # Image resolution
    pixel_size=(0.01, 0.01), # Pixel size in mm
)





# ====================
# 3. CREATE THE LIGHT
# ====================
light_position = np.array([100.0, 0.0, 0.0])
light_target = cylinder_center
light_z_axis = light_target - light_position
light_y_axis = np.array([0.0, 1.0, 0.0])
light_x_axis = np.cross(light_y_axis, light_z_axis)

light_frame = Frame(
    origin=light_position,
    x_axis=light_x_axis,
    y_axis=light_y_axis,
    z_axis=light_z_axis
)

light = SpotLight(
    frame=light_frame,
    energy=50000.0,
    spot_size=0.5,
    spot_blend=0.5,
)



# ======================
# 4. CREATE THE MATERIAL
# ======================
material = get_iron_material()




# ==========================
# 5. CREATE THE EXPERIMENT
# ==========================
STOP_BEFORE_RENDER = False
dat_folder = os.path.dirname(__file__)

experiment = BlenderExperiment(Nb_frames=2) # We want Time 0 and Time 1 -> 2 frames (WARNING : BLENDER frames = rendering time, OBJECT frame = orientation in the scene)
experiment.set_default_background()

# Reading the mesh and adding it to the experiment
print("Adding the mesh to the experiment...")
experiment.add_mesh("Cylinder Time 0", cylinder_mesh_time_0, frames=[True, False]) # The mesh TIME 0 is only active for the first frame
experiment.add_mesh_material("Cylinder Time 0", material)
experiment.add_mesh_pattern("Cylinder Time 0", get_mouchtichu_path())
experiment.add_mesh("Cylinder Time 1", cylinder_mesh_time_1, frames=[False, True]) # The mesh TIME 1 is only active for the second frame
experiment.add_mesh_material("Cylinder Time 1", material) # Same pattern and material for both meshes
experiment.add_mesh_pattern("Cylinder Time 1", get_mouchtichu_path()) 

# Adding the camera to the experiment
print("Adding the camera to the experiment...")
experiment.add_camera("Camera 1", camera_1, frames=[True, True]) # The cameras are the same for both frames
experiment.add_camera("Camera 2", camera_2, frames=[True, True]) # The cameras are the same for both frames

# Adding the light to the experiment
print("Adding the light to the experiment...")
experiment.add_spotlight("Light", light)





# ==========================
# 6. RENDERING THE SCENE
# ==========================

if not STOP_BEFORE_RENDER:
    for frame in [0, 1]:
        for camera in ["Camera 1", "Camera 2"]:
            # Selecting the frame and the camera for rendering
            experiment.set_active_camera(camera)
            experiment.set_active_frame(frame + 1) # Blender frame start at 1
            # Rendering the scene
            print("Rendering the scene...")
            experiment.render(
                os.path.join(dat_folder, "SDIC_render_{}_{}.tiff".format(frame, camera)),
                N_samples=200,
            )
