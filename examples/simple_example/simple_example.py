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
To observe the pattern, we need to set a camera and a light.
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

cylinder_mesh = create_axisymmetric_mesh(
    profile_curve=lambda z: cylinder_radius,
    frame=cylinder_frame,
    height_bounds=(cylinder_height_min, cylinder_height_max),
    theta_bounds=(cylinder_theta_min, cylinder_theta_max),
    Nheight=cylinder_Nheight,
    Ntheta=cylinder_Ntheta,
    closed=False,
)


# ====================
# 2. CREATE THE CAMERA
# ====================
camera_position = np.array([120.0, 0.0, 50.0])
camera_target = cylinder_center + cylinder_radius*cylinder_x_axis
camera_z_axis = camera_target - camera_position
camera_y_axis = np.array([0.0, 1.0, 0.0])
camera_x_axis = np.cross(camera_y_axis, camera_z_axis)
camera_frame = Frame(
    origin=camera_position,
    x_axis=camera_x_axis,
    y_axis=camera_y_axis,
    z_axis=camera_z_axis
)

camera = Camera(
    frame=camera_frame,
    intrinsic_matrix=np.array([[20000.0, 0.0, 499.5], [0.0, 20000.0, 503.5], [0.0, 0.0, 1.0]]),
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
material = MaterialBSDF(
    base_color=(1.0, 1.0, 1.0, 1.0), # White color
    roughness=0.8,
    specular_IOR_level=1.0,
    metallic=0.7,
)


material = get_copper_material()



# ==========================
# 5. CREATE THE EXPERIMENT
# ==========================
STOP_BEFORE_RENDER = False
dat_folder = os.path.dirname(__file__)

experiment = BlenderExperiment()
experiment.set_default_background()

# Reading the mesh and adding it to the experiment
print("Adding the mesh to the experiment...")
experiment.add_mesh("Cylinder", cylinder_mesh)
experiment.add_mesh_material("Cylinder", material)
experiment.add_mesh_pattern("Cylinder", get_mouchtichu_path())

# Adding the camera to the experiment
print("Adding the camera to the experiment...")
experiment.add_camera("Camera", camera)

# Adding the light to the experiment
print("Adding the light to the experiment...")
experiment.add_spotlight("Light", light)





# ==========================
# 6. RENDERING THE SCENE
# ==========================

# Selecting the frame and the camera for rendering
experiment.set_active_camera("Camera")
experiment.set_active_frame(1)

if not STOP_BEFORE_RENDER:
    # Rendering the scene
    print("Rendering the scene...")
    experiment.render(
        os.path.join(dat_folder, "simple_render.tiff"),
        N_samples=200,
    )
