import numpy as np
import os
import meshio

from py3dframe import Frame

from pyblenderSDIC import Camera, SpotLight, BlenderExperiment

from pyblenderSDIC.meshes import create_axisymmetric_mesh, create_xy_heightmap_mesh
from pyblenderSDIC.materials import MaterialBSDF, get_iron_material, get_copper_material, get_mirror_material
from pyblenderSDIC.patterns import get_mouchtichu_path, get_speckle_path

"""
We observe a demi-cylinder with a mouchtichu pattern on it.

The cylinder is observed by a camera using a indirect view.
The camera observes a mirror reflexion of the cylinder.

We place all the obsects at y=0 to simplify the creation of the scene. 
Because y-axis will be fixed for all the objects.
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

# ========================
# 2. Preparing the camera (part 1)
# ========================

# We select the position of the camera to allow defining the mirror
camera_position = np.array([20.0, 0.0, 60.0])
camera_target = cylinder_center + cylinder_radius*cylinder_x_axis


# ====================
# 3. CREATE THE MIRROR
# ====================

# The mirror is a plane with a normal vector allowing to reflect the cylinder on the camera
mirror_center = np.array([75.0, 0.0, 20.0])
mirror_y_axis = np.array([0.0, 1.0, 0.0])

# The normal vector of the mirror must reflect "camera_target" on "camera_position".
input_ray = mirror_center - camera_target
input_ray /= np.linalg.norm(input_ray)

output_ray = camera_position - mirror_center
output_ray /= np.linalg.norm(output_ray)

mirror_z_axis = output_ray - input_ray
mirror_z_axis /= np.linalg.norm(mirror_z_axis)

mirror_x_axis = np.cross(mirror_y_axis, mirror_z_axis)
mirror_frame = Frame(
    origin=mirror_center,
    x_axis=mirror_x_axis,
    y_axis=mirror_y_axis,
    z_axis=mirror_z_axis
)

mirror_mesh = create_xy_heightmap_mesh(
    height_function=lambda x, y: 0.0,
    frame=mirror_frame,
    x_bounds=(-8.0, 8.0),
    y_bounds=(-8.0, 8.0),
    Nx=2,
    Ny=2,
)

# ====================
# 4. CREATE THE CAMERA (part 2)
# ====================

# The camera is observing the cylinder through the mirror
camera_direction = mirror_center - camera_position
camera_z_axis = camera_direction / np.linalg.norm(camera_direction)
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
# 5. CREATE THE LIGHT
# ====================

light_position = np.array([70.0, 0.0, -20.0])
light_target = cylinder_center + cylinder_radius*cylinder_x_axis
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
# 6. CREATE THE MATERIAL
# ======================

cylinder_material = get_iron_material()
mirror_material = get_mirror_material()




# ==========================
# 7. CREATE THE EXPERIMENT
# ==========================
STOP_BEFORE_RENDER = False
dat_folder = os.path.dirname(__file__)

experiment = BlenderExperiment(Nb_frames=1)
experiment.set_default_background()

# Reading the mesh and adding it to the experiment
print("Adding the mesh to the experiment...")
experiment.add_mesh("Cylinder", cylinder_mesh) # The mesh TIME 0 is only active for the first frame
experiment.add_mesh_material("Cylinder", cylinder_material)
experiment.add_mesh_pattern("Cylinder", get_mouchtichu_path())
experiment.add_mesh("Mirror", mirror_mesh) # The mesh TIME 0 is only active for the first frame
experiment.add_mesh_material("Mirror", mirror_material) # No pattern for the mirror

# Adding the camera to the experiment
print("Adding the camera to the experiment...")
experiment.add_camera("Camera", camera)

# Adding the light to the experiment
print("Adding the light to the experiment...")
experiment.add_spotlight("Light", light)




# ==========================
# 6. RENDERING THE SCENE
# ==========================

if not STOP_BEFORE_RENDER:
    # Selecting the frame and the camera for rendering
    experiment.set_active_camera("Camera")
    experiment.set_active_frame(1) # Blender frame start at 1
    # Rendering the scene
    print("Rendering the scene...")
    experiment.render(
        os.path.join(dat_folder, "mirror_reflexion.tiff"),
        N_samples=200,
    )
