import numpy as np
import os
import meshio
import matplotlib.pyplot as plt

from py3dframe import Frame

from pyblenderSDIC import Camera, SpotLight, BlenderExperiment

from pyblenderSDIC.meshes import create_axisymmetric_mesh, create_xy_heightmap_mesh
from pyblenderSDIC.materials import MaterialBSDF, get_iron_material, get_copper_material, get_mirror_material
from pyblenderSDIC.patterns import create_speckle_BW_image

"""
We observe a closed cylinder with a speckle pattern on it.
The cylinder evolves in time so we need 2 cylinders instances.

For this example we use a realistic pattern and a realistic material.
Realistic distance are also used.

To observe the cylinder, we use only one camera placed at around 4 meters from the cylinder.
The camera is fixed in the scene and we want to acquire two points of view of the same object in one image.

To achive this, we use 4 mirrors placed to observe the cylinder from two different angles.

Camera -> Mirror B1 -> Mirror A1 -> Cylinder
Camera -> Mirror B2 -> Mirror A2 -> Cylinder


TO solve no rendering,

remove mirror B and use cm instead of mm 
"""

# =======================================================================
# ========== Example of how to use the package `pyblenderSDIC` ==========
# =======================================================================

# ====================
# 0. Create the pattern
# ====================
pattern_path = os.path.join(os.path.dirname(__file__), "pattern.tiff")

# Set the parameters for the speckle image
image_shape = (3000, 6000)
grain_size = (50.0, 50.0) # in pixels
grain_size_sigma = (10.0, 10.0)
density = 0.85
seed = 42

# Generate the speckle image with specified parameters
speckle_img = create_speckle_BW_image(
    image_shape=image_shape,
    grain_size=grain_size,
    grain_size_sigma=grain_size_sigma,
    density=density,
    white_intensity=1.0, # To normalize the image between [0,1]
    seed=seed
)

plt.imsave(pattern_path, speckle_img, cmap='gray')


# ====================
# 1. CREATE THE MESH
# ====================

# Set the cylinder parameters
cylinder_center = np.array([0.0, 0.0, 0.0])
cylinder_z_axis = np.array([0.0, 0.0, 1.0])
cylinder_x_axis = np.array([1.0, 0.0, 0.0])
cylinder_y_axis = np.cross(cylinder_z_axis, cylinder_x_axis)

cylinder_frame = Frame(
    origin=cylinder_center,
    x_axis=cylinder_x_axis,
    y_axis=cylinder_y_axis,
    z_axis=cylinder_z_axis
)

cylinder_radius = 24.9 # cm
cylinder_height_min = -12.0 # cm
cylinder_height_max = 12.0 # cm
cylinder_theta_min = -np.pi
cylinder_theta_max = -np.pi + 2*np.pi*(1 - 1/100) # See the doc of the function create_axisymmetric_mesh
cylinder_Nheight = 100
cylinder_Ntheta = 100

cylinder_mesh_time_0 = create_axisymmetric_mesh(
    profile_curve=lambda z: cylinder_radius,
    frame=cylinder_frame,
    height_bounds=(cylinder_height_min, cylinder_height_max),
    theta_bounds=(cylinder_theta_min, cylinder_theta_max),
    Nheight=cylinder_Nheight,
    Ntheta=cylinder_Ntheta,
    closed=True,
    uv_layout=0, # We want the largest pattern dim along the thetas
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
    closed=True,
    uv_layout=0, # We want the largest pattern dim along the thetas
)



# ========================
# 2. Preparing the camera (part 1)
# ========================

# We select the position of the camera to allow defining the mirror
camera_position = np.array([200.0, 0.0, -200.0])
camera_target = cylinder_center + cylinder_radius*cylinder_x_axis



# ====================
# 3. CREATE THE MIRRORS
# ====================

mirror_size = 30
mirror_y_axis = np.array([0.0, 1.0, 0.0])

# Group mirrors A 
group_A_center = np.array([200.0, 0.0, 0.0])

mirror_A1_center = group_A_center + np.array([-mirror_size/2, 0.0, mirror_size/2])
mirror_A2_center = group_A_center + np.array([mirror_size/2, 0.0, -mirror_size/2])

# Create the mirror A1
input_ray = mirror_A1_center - cylinder_center
input_ray /= np.linalg.norm(input_ray)

output_ray = camera_position - mirror_A1_center
output_ray /= np.linalg.norm(output_ray)

mirror_A1_z_axis = output_ray - input_ray
mirror_A1_z_axis /= np.linalg.norm(mirror_A1_z_axis)
mirror_A1_x_axis = np.cross(mirror_y_axis, mirror_A1_z_axis)

mirror_A1_frame = Frame(
    origin=mirror_A1_center,
    x_axis=mirror_A1_x_axis,
    y_axis=mirror_y_axis,
    z_axis=mirror_A1_z_axis
)

mirror_A1_mesh = create_xy_heightmap_mesh(
    height_function=lambda x, y: 0.0,
    frame=mirror_A1_frame,
    x_bounds=(-mirror_size/2, mirror_size/2),
    y_bounds=(-mirror_size/2, mirror_size/2),
    Nx=2,
    Ny=2,
)

# Create the mirror A2
input_ray = mirror_A2_center - cylinder_center
input_ray /= np.linalg.norm(input_ray)

output_ray = camera_position - mirror_A2_center
output_ray /= np.linalg.norm(output_ray)

mirror_A2_z_axis = output_ray - input_ray
mirror_A2_z_axis /= np.linalg.norm(mirror_A2_z_axis)
mirror_A2_x_axis = np.cross(mirror_y_axis, mirror_A2_z_axis)

mirror_A2_frame = Frame(
    origin=mirror_A2_center,
    x_axis=mirror_A2_x_axis,
    y_axis=mirror_y_axis,
    z_axis=mirror_A2_z_axis
)

mirror_A2_mesh = create_xy_heightmap_mesh(
    height_function=lambda x, y: 0.0,
    frame=mirror_A2_frame,
    x_bounds=(-mirror_size/2, mirror_size/2),
    y_bounds=(-mirror_size/2, mirror_size/2),
    Nx=2,
    Ny=2,
)



# ====================
# 4. CREATE THE CAMERA (part 2)
# ====================

# The camera is observing the cylinder through the mirror
camera_direction = group_A_center - camera_position
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
    intrinsic_matrix=np.array([[17000.0, 0.0, 2000], [0.0, 17000.0, 1500], [0.0, 0.0, 1.0]]),
    resolution=(4000, 3000), # Image resolution
    pixel_size=(0.001, 0.001), # Pixel size in cm
    clip_distance=(0.1, 1500.0), # Clip distance in mm -> 15 meters
)






# ====================
# 3. CREATE THE LIGHT
# ====================
light_position = np.array([100.0, 0.0, 40.0])
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
    energy=5000000.0,
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

experiment = BlenderExperiment(Nb_frames=2) # Time 0 + Time 1 = 2 frames
experiment.set_default_background()

# Reading the mesh and adding it to the experiment
print("Adding the mesh to the experiment...")
experiment.add_mesh("Cylinder Time 0", cylinder_mesh_time_0, frames=[True, False]) # The mesh TIME 0 is only active for the first frame
experiment.add_mesh_material("Cylinder Time 0", cylinder_material)
experiment.add_mesh_pattern("Cylinder Time 0", pattern_path)
experiment.add_mesh("Cylinder Time 1", cylinder_mesh_time_1, frames=[False, True]) # The mesh TIME 1 is only active for the second frame
experiment.add_mesh_material("Cylinder Time 1", cylinder_material) # Same pattern and material for both meshes
experiment.add_mesh_pattern("Cylinder Time 1", pattern_path)

# Adding the mirrors to the experiment
print("Adding the mirrors to the experiment...")
experiment.add_mesh("Mirror A1", mirror_A1_mesh, frames=[True, True]) # The mirror is the same for both frames
experiment.add_mesh_material("Mirror A1", mirror_material) # No pattern for the mirror
experiment.add_mesh("Mirror A2", mirror_A2_mesh, frames=[True, True]) # The mirror is the same for both frames
experiment.add_mesh_material("Mirror A2", mirror_material) # No pattern for the mirror


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
    for frame in [0, 1]:
        # Selecting the frame and the camera for rendering
        experiment.set_active_camera("Camera")
        experiment.set_active_frame(frame + 1) # Blender frame start at 1
        # Rendering the scene
        print("Rendering the scene...")
        experiment.render(
            os.path.join(dat_folder, "split_screen_render_{}.tiff".format(frame)),
            N_samples=200,
        )
