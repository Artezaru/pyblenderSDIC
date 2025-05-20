try:
    import bpy
    import bmesh
    from bpy_types import Object
except ImportError:
    print("Import failed for BLENDER EXPERIMENT. Running in Blender environment.")
    Object = None

from typing import Tuple, Optional, List
from numbers import Integral
from .camera import Camera
from .materials.material_bsdf import MaterialBSDF
from .spotlight import SpotLight
import meshio
import os
import numpy

class BlenderExperiment:
    r"""
    Represents a Blender experiment with a defined scene and objects.

    The BlenderExperiment class provides methods to add and configure cameras, spot lights, and objects (mesh + material) in a Blender scene.
    The class also allows setting the properties of the scene such as resolution and pixel aspect ratio.
    Then the class provides methods to render the scene and save the rendered images.

    The number of frames in the experiment can be set using the `end_frame` parameter.
    This parameters can not be updated after the experiment is created.

    Parameters
    ----------
    Nb_frames : int, optional
        The number of frames for the experiment (default is 1).
    """
    def __init__(self, Nb_frames: int = 1) -> None:
        if not isinstance(Nb_frames, Integral):
            raise TypeError("Nb_frames must be an integer")
        if Nb_frames < 1:
            raise ValueError("Nb_frames must be greater than or equal to 1")
        self._end_frame = int(Nb_frames)
        self._camera_objects = {} # {"name" : List[Camera, List[bool]]}
        self._spotlight_objects = {}
        self._mesh_objects = {}
        self._active_camera = None
        self._active_frame = 1
        self._reset_blender()

        
    # =======================================================
    # Reset Blender
    # =======================================================
    def _reset_blender(self) -> None:
        """
        Resets the Blender environment.

        - Removes all objects in the active scene.
        - Deletes all meshes, materials, textures, armatures, cameras, lights, and collections.
        - Initializes a new scene for the experiment.
        - Resets the 3D cursor to the origin.
        - Sets the start and end frames to 1.

        """
        # Deleting all objects in the active scene
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete()
        # Deleting all meshes
        for mesh in bpy.data.meshes:
            bpy.data.meshes.remove(mesh)
        # Deleting all materials
        for material in bpy.data.materials:
            bpy.data.materials.remove(material)
        # Deleting all textures
        for texture in bpy.data.textures:
            bpy.data.textures.remove(texture)
        # Deleting all armatures
        for armature in bpy.data.armatures:
            bpy.data.armatures.remove(armature)
        # Deleting all cameras
        for camera in bpy.data.cameras:
            bpy.data.cameras.remove(camera)
        # Deleting all lights
        for light in bpy.data.lights:
            bpy.data.lights.remove(light)
        # Deleting all collections
        for collection in bpy.data.collections:
            bpy.data.collections.remove(collection)
        # Clearing all dictionaries
        self._camera_objects.clear()
        self._spotlight_objects.clear()
        self._mesh_objects.clear()
        self._active_camera = None
        self._active_frame = 1
        # Initializing of the blender scene
        self._experiment_scene = bpy.data.scenes.new(name="experiment_scene")
        bpy.context.scene.cursor.location = (0.0, 0.0, 0.0)  # Reset 3D cursor
        bpy.context.scene.frame_start = 1  # Start frame
        bpy.context.scene.frame_end = self._end_frame + 1 # End frame 


    # =======================================================
    # Blender Scene Management (CAMERA)
    # =======================================================
    def add_camera(self, name: str, camera: Camera, frames: Optional[List[bool]] = None) -> None:
        r"""
        Add a camera to the experiment.

        The frames parameter indicates which frames the camera will be active.
        If None, the camera will be active for all frames.
        Else a list of booleans must be provided, where each boolean indicates if the camera is active for that frame.

        .. code-block:: python

            # Example usage
            experiment = BlenderExperiment(Nb_frames=10)
            camera = Camera()
            # Define the camera properties here
            # ...
            experiment.add_camera(name="Camera1", camera=camera, frames=[True, False, True, False, True, False, True, False, True, False])

        .. seealso::

            - :class:`pyblenderSDIC.Camera` for more information on how to define a camera.
            - :meth:`pyblenderSDIC.BlenderExperiment.update_camera` to update the camera properties in the Blender scene.
        
        Parameters
        ----------
        name : str
            The name of the camera.
        
        camera : Camera
            The camera object to be added. 

        frames : List[bool], optional
            A list of booleans indicating which frames the camera will be active.
            If None, the camera will be active for all frames (default is None).

        Raises
        -------
        TypeError
            If name is not a string or camera is not an instance of Camera.
        ValueError
            If a camera with the same name already exists in the experiment or in Blender data.
            If the length of frames is not equal to the number of frames in the experiment (end_frame).
            If the camera is not completely defined.

        Blender Details
        ---------------
        The camera is created in the Blender scene and linked to the experiment scene.
        The camera name is set to the provided name.
        """
        if not isinstance(name, str):
            raise TypeError("name must be a string")
        if name in self._camera_objects:
            raise ValueError(f"Camera with name {name} already exists.")
        if name in bpy.data.objects:
            raise ValueError(f"Object with name {name} already exists in Blender data.")
        if not isinstance(camera, Camera):
            raise TypeError("camera must be an instance of Camera")
        if not camera.is_complete():
            raise ValueError("Camera is not completely defined.")
    
        # Ensure the experiment scene is active
        bpy.context.window.scene = self._experiment_scene

        # Create the camera
        camera_data = bpy.data.cameras.new(name=name)
        blender_camera = bpy.data.objects.new(name, camera_data)
        self._camera_objects[name] = [camera, None] # Store the camera and its frames

        # Link the camera to the scene
        self._experiment_scene.collection.objects.link(blender_camera)

        # Set the frames for which the camera will be active
        self.set_camera_frames(name, frames)

        # Update the camera properties in the Blender scene
        self.update_camera(name)


    def set_camera_frames(self, name: str, frames: Optional[List[bool]] = None) -> None:
        r"""
        Set the frames for which the camera will be active.
        If None, the camera will be active for all frames.
        Else a list of booleans must be provided, where each boolean indicates if the camera is active for that frame.

        Parameters
        ----------
        name : str
            The name of the camera.
        
        frames : List[bool], optional
            A list of booleans indicating which frames the camera will be active.
            If None, the camera will be active for all frames (default is None).

        Raises
        -------
        TypeError
            If name is not a string or frames is not a list of booleans.
        ValueError
            If a camera with the same name does not exist in the experiment or in Blender data.
            If the length of frames is not equal to the number of frames in the experiment (end_frame).
        """
        if not isinstance(name, str):
            raise TypeError("name must be a string")
        if name not in self._camera_objects:
            raise ValueError(f"Camera with name {name} does not exist.")
        if name not in bpy.data.objects:
            raise ValueError(f"No object with name {name} in Blender data.")
        if frames is not None:
            if not isinstance(frames, list):
                raise TypeError("frames must be a list of booleans")
            if len(frames) != self._end_frame:
                raise ValueError(f"Length of frames must be equal to {self._end_frame}")
            for frame in frames:
                if not isinstance(frame, bool):
                    raise TypeError("frames must be a list of booleans")
        else:
            frames = [True] * self._end_frame
        
        # Ensure the experiment scene is active
        bpy.context.window.scene = self._experiment_scene

        # Set the frames for which the camera will be active
        self._camera_objects[name][1] = frames


    def get_camera_frames(self, name: str) -> List[bool]:
        r"""
        Get the frames for which the camera is active.
        
        Parameters
        ----------
        name : str
            The name of the camera.
        
        Returns
        -------
        List[bool]
            A list of booleans indicating which frames the camera is active.
        
        Raises
        -------
        TypeError
            If name is not a string.
        ValueError
            If a camera with the same name does not exist in the experiment or in Blender data.
        """
        if not isinstance(name, str):
            raise TypeError("name must be a string")
        if name not in self._camera_objects:
            raise ValueError(f"Camera with name {name} does not exist.")
        if name not in bpy.data.objects:
            raise ValueError(f"No object with name {name} in Blender data.")
        
        # Ensure the experiment scene is active
        bpy.context.window.scene = self._experiment_scene

        # Get the frames for which the camera is active
        return self._camera_objects[name][1]


    def get_camera(self, name: str) -> Tuple[Camera, Object]:
        r"""
        Get the camera object and its Blender object.
        
        Parameters
        ----------
        name : str
            The name of the camera to retrieve.
        
        Returns
        -------
        Tuple[Camera, Object]
            A tuple containing the Camera object and its corresponding Blender object.
        """
        if not isinstance(name, str):
            raise TypeError("name must be a string")
        if name not in self._camera_objects:
            raise ValueError(f"Camera with name {name} does not exist.")
        if name not in bpy.data.objects:
            raise ValueError(f"No object with name {name} in Blender data.")
        
        # Ensure the experiment scene is active
        bpy.context.window.scene = self._experiment_scene

        # Get the camera object
        camera = self._camera_objects[name][0]
        blender_camera = bpy.data.objects[name]

        # Check the types of the objects
        if not isinstance(camera, Camera):
            raise TypeError("[ERROR] camera must be an instance of Camera")
        if not isinstance(blender_camera, Object):
            raise TypeError("[ERROR] blender_camera must be an instance of Object")
        
        return camera, blender_camera
    

    def get_camera_names(self) -> List[str]:
        r"""
        Get the names of all cameras in the experiment.

        Returns
        -------
        List[str]
            A list of camera names.
        """
        return list(self._camera_objects.keys())
    

    def remove_camera(self, name: str) -> None:
        r"""
        Remove a camera from the experiment.

        Parameters
        ----------
        name : str
            The name of the camera to remove.

        Raises
        -------
        TypeError
            If name is not a string.
        ValueError
            If a camera with the same name does not exist in the experiment or in Blender data.
        """
        if not isinstance(name, str):
            raise TypeError("name must be a string")
        if name not in self._camera_objects:
            raise ValueError(f"Camera with name {name} does not exist.")
        if name not in bpy.data.objects:
            raise ValueError(f"No object with name {name} in Blender data.")
        
        # Ensure the experiment scene is active
        bpy.context.window.scene = self._experiment_scene

        # Remove the camera from the scene
        camera, blender_camera = self.get_camera(name)
        bpy.data.objects.remove(blender_camera)
        self._camera_objects.pop(name)

        # Set active camera to None if it was removed
        if self._active_camera is not None and self._active_camera == name:
            self._active_camera = None


    def update_camera(self, name: str) -> None:
        r"""
        Update the camera properties in the Blender scene.
        This method must be called after updating the camera properties in the Camera object.

        .. code-block:: python

            # Example usage
            experiment = BlenderExperiment(Nb_frames=10)
            camera = Camera()
            # Define the camera properties here
            # ...
            experiment.add_camera(name="Camera1", camera=camera, frames=[True, False, True, False, True, False, True, False, True, False])
            # Update some properties of the camera here
            # ...
            experiment.update_camera(name="Camera1") # To set active the modifications in the Blender scene
        
        Parameters
        ----------
        name : str
            The name of the camera to update.
        """
        if not isinstance(name, str):
            raise TypeError("name must be a string")
        if name not in self._camera_objects:
            raise ValueError(f"Camera with name {name} does not exist.")
        if name not in bpy.data.objects:
            raise ValueError(f"No object with name {name} in Blender data.")

        # Ensure the experiment scene is active
        bpy.context.window.scene = self._experiment_scene

        # Get the camera object
        camera, blender_camera = self.get_camera(name)
        if not camera.is_complete():
            raise ValueError("Camera is not completely defined.")

        # Set the lens unit of the camera
        blender_camera.data.lens_unit = 'MILLIMETERS'

        # Set the position and rotation of the camera
        rotation, translation = camera.get_OpenGL_RT()
        blender_camera.location = translation
        blender_quaternion = rotation.as_quat(scalar_first=True) # blender convention [qw, qx, qy, qz]
        blender_camera.rotation_mode = 'QUATERNION'
        blender_camera.rotation_quaternion = blender_quaternion 

        # Set the clipping of the camera
        if camera.clnear is not None:
            blender_camera.data.clip_start = camera.clnear
        else:
            blender_camera.data.clip_start = 1e-6 # Lower limit for the near clipping plane
        if camera.clfar is not None:
            blender_camera.data.clip_end = camera.clfar
        else:
            blender_camera.data.clip_end = numpy.inf # Upper limit for the far clipping plane

        # Set the focal length of the camera
        if camera.lens < 1:
            raise ValueError("Focal length must be greater than 1 millimeter for Blender.")
        blender_camera.data.lens = camera.lens

        # Set size of the sensor
        blender_camera.data.sensor_width = camera.sensor_width
        blender_camera.data.sensor_height = camera.sensor_height
        blender_camera.data.sensor_fit = camera.sensor_fit

        # Set the shift of the camera
        blender_camera.data.shift_x = camera.shift_x
        blender_camera.data.shift_y = camera.shift_y

        # Update the active camera if it is the current one
        if self._active_camera is not None and self._active_camera == name:
            self.set_active_camera(name)


    def set_active_camera(self, name: str) -> None:
        r"""
        Set the active camera for the experiment.

        Parameters
        ----------
        name : str
            The name of the camera to set as active.

        Raises
        -------
        TypeError
            If name is not a string.
        ValueError
            If a camera with the same name does not exist in the experiment or in Blender data.
        """
        if not isinstance(name, str):
            raise TypeError("name must be a string")
        if name not in self._camera_objects:
            raise ValueError(f"Camera with name {name} does not exist.")
        if name not in bpy.data.objects:
            raise ValueError(f"No object with name {name} in Blender data.")

        # Ensure the experiment scene is active
        bpy.context.window.scene = self._experiment_scene

        # Get the camera object
        camera, blender_camera = self.get_camera(name)
        if not camera.is_complete():
            raise ValueError("Camera is not completely defined.")
        
        # Set the active camera
        bpy.context.scene.camera = blender_camera

        # Configure the scene properties using the camera's parameters
        render = bpy.context.scene.render

        # Set the resolution
        render.resolution_x = camera.rx
        render.resolution_y = camera.ry

        # Set the pixel aspect ratio
        render.pixel_aspect_x = camera.pixel_aspect_x
        render.pixel_aspect_y = camera.pixel_aspect_y

        # Set the active camera
        self._active_camera = name


    def get_active_camera(self) -> str:
        r"""
        Get the name of the active camera.

        Returns
        -------
        str
            The name of the active camera.
        
        Raises
        -------
        ValueError
            If no active camera is set.
        """
        return self._active_camera




    # =======================================================
    # Blender Scene Management (MESH)
    # =======================================================
    def add_mesh(self, name: str, mesh: meshio.Mesh, frames: Optional[List[bool]] = None) -> None:
        r"""
        Add a mesh to the experiment.

        The mesh must be an instance of meshio.Mesh with the following structure:

        - ``points``: A NumPy array of shape (N, 3) representing the coordinates of N mesh nodes.
        - ``cells_dict{"triangle"}.data``: A NumPy array of shape (M, 3) representing M triangular elements defined by node indices.
        - ``point_data``: A dictionary storing data associated with each point, such as "uvmap" a (N, 3) array of texture coordinates (only the first two columns are used).

        The frames parameter indicates which frames the mesh will be active.
        If None, the mesh will be active for all frames.
        Else a list of booleans must be provided, where each boolean indicates if the mesh is active for that frame.

        .. code-block:: python

            # Example usage
            experiment = BlenderExperiment(Nb_frames=10)
            # Define the mesh properties here
            # ...
            mesh = meshio.Mesh(points, cells_dict={"triangle": elements}, point_data={"uvmap": texture_coordinates})
            experiment.add_mesh(name="Mesh1", mesh=mesh, frames=[True, False, True, False, True, False, True, False, True, False])

        .. seealso::

            - :class:`pyblenderSDIC.meshio.Mesh` for more information on how to define a mesh.
            - :meth:`pyblenderSDIC.BlenderExperiment.update_mesh` to update the mesh properties in the Blender scene.
            - :meth:`pyblenderSDIC.BlenderExperiment.add_mesh_material` to set the material of the mesh.
            - :meth:`pyblenderSDIC.BlenderExperiment.add_mesh_pattern` to set the pattern image of the mesh.

        Parameters
        ----------
        name : str
            The name of the mesh.
        
        mesh : meshio.Mesh
            The mesh object to be added. 

        frames : List[bool], optional
            A list of booleans indicating which frames the mesh will be active.
            If None, the mesh will be active for all frames (default is None).

        Raises
        -------
        TypeError
            If name is not a string or mesh is not an instance of meshio.Mesh.
        ValueError
            If a mesh with the same name already exists in the experiment or in Blender data.
            If the length of frames is not equal to the number of frames in the experiment (end_frame).
            If the mesh is not completely defined.

        Blender Details
        ---------------
        The mesh is created in the Blender scene and linked to the experiment scene.
        The mesh name is set to the provided name.

        A material is created for the mesh with the name ``__[pyblenderSDIC]__{name}_material``.
        The material uses a Principled BSDF shader and is set to ``use_nodes = True``.
        The material is located at the first index of the mesh data materials.

        A MixRGB node is created at the input of the Principled BSDF ``Base Color`` node.
        The MixRGB node name is set to ``__[pyblenderSDIC]__{name}_mix_basecolor_pattern``.

        The first input of the MixRGB node is the default base color of the material (white).
        It can be setted to any color using the ``add_mesh_material`` method.

        The second input of the MixRGB node is the default pattern of the material (white).
        It can be setted to any pattern using the ``add_mesh_pattern`` method.

        The output of the MixRGB node is connected to the ``Base Color`` input of the Principled BSDF node.          
        """
        if not isinstance(name, str):
            raise TypeError("name must be a string")
        if name in self._mesh_objects:
            raise ValueError(f"Mesh with name {name} already exists.")
        if name in bpy.data.objects:
            raise ValueError(f"Object with name {name} already exists in Blender data.")
        if not isinstance(mesh, meshio.Mesh):
            raise TypeError("mesh must be an instance of meshio.Mesh")
        if not 'triangle' in mesh.cells_dict:
            raise ValueError("mesh must contain a 'triangle' cell type")
        
        # Ensure the experiment scene is active
        bpy.context.window.scene = self._experiment_scene

        # Extract the vertices and faces from the mesh
        points = mesh.points
        cells = mesh.cells_dict["triangle"]
        
        # ======================
        # MESH CREATION
        # ======================

        # Create the mesh
        mesh_data = bpy.data.meshes.new(name=name)
        blender_mesh = bpy.data.objects.new(name, mesh_data)
        self._mesh_objects[name] = [mesh, None]

        # Convert the points and cells to a format compatible with Blender
        points = [tuple(point) for point in points]
        faces = [tuple(face) for face in cells]

        # Set the vertices and faces of the Blender mesh
        blender_mesh.data.from_pydata(points, [], faces)

        # =======================
        # Material CREATION
        # =======================

        # Creating the material
        if f'__[pyblenderSDIC]__{name}_material' in bpy.data.materials:
            raise ValueError(f"Material with name __[pyblenderSDIC]__{name}_material already exists.")

        blender_material = bpy.data.materials.new(name=f'__[pyblenderSDIC]__{name}_material')
        blender_material.use_nodes = True

        # Assign the material to the object
        if len(blender_mesh.data.materials) > 0:
            blender_mesh.data.materials[0] = blender_material
        else:
            blender_mesh.data.materials.append(blender_material)

        # ========================
        # MIX RGB NODE CREATION
        # ========================
        # Here I create a MixRGB node for the base color material.
        # The idea is :
        # - The first input of the MixRGB node is the default base color of the material.
        # - The second input of the MixRGB node is the default pattern of the material (1.0, 1.0, 1.0, 1.0)
        # - The third input of the MixRGB is anything else.
        #
        # If the user add a pattern, we update the second input of the MixRGB node with the pattern.
        # If the user add a material, we update the first input of the MixRGB node with the color.
        
        # Access the nodes of the material
        nodes = blender_material.node_tree.nodes  # Access the material's node tree
        links = blender_material.node_tree.links  # Access node tree links

        # Create a MixRGB node (type 'MIX') to combine base color and pattern
        if f'__[pyblenderSDIC]__{name}_mix_basecolor_pattern' in nodes:
            raise ValueError(f"MixRGB node with name __[pyblenderSDIC]__{name}_mix_basecolor_pattern already exists.")

        mix_node = nodes.new(type='ShaderNodeMixRGB')
        mix_node.name = f'__[pyblenderSDIC]__{name}_mix_basecolor_pattern'
        mix_node.blend_type = 'MULTIPLY'
        mix_node.inputs['Fac'].default_value = 1.0  # Full multiplication

        # Set the default input values for the MixRGB node
        mix_node.inputs['Color1'].default_value = (1.0, 1.0, 1.0, 1.0)  # Default base color (white)
        mix_node.inputs['Color2'].default_value = (1.0, 1.0, 1.0, 1.0)  # Default pattern (white)

        # Ensure the Principled BSDF node exists
        if not 'Principled BSDF' in nodes:
            raise ValueError("No 'Principled BSDF' node found in the material. Ensure nodes are enabled.")

        principled = nodes['Principled BSDF']
        input_base_color = principled.inputs["Base Color"]
        existing_link = input_base_color.links[0] if input_base_color.links else None

        if existing_link:
            # disconnect the existing link
            links.remove(existing_link)
            # connect the existing link to the MixRGB node `Color1`.
            links.new(existing_link.from_node.outputs[existing_link.from_socket.name], mix_node.inputs['Color1'])

        # Connect the MixRGB node to the Principled BSDF node
        links.new(mix_node.outputs['Color'], input_base_color)

        # =========================
        # FINALIZE MESH
        # =========================
        # Update the mesh data
        mesh_data.update()

        # Link the mesh to the scene
        self._experiment_scene.collection.objects.link(blender_mesh)

        # Set the frames for which the mesh will be active
        self.set_mesh_frames(name, frames)


    def add_mesh_pattern(self, name: str, pattern_path: str) -> None:
        r"""
        Add a pattern to the mesh material.

        The pattern must be a valid image file path (e.g., PNG, JPEG).
        The pattern will be applied to the mesh material as a texture.

        .. code-block:: python

            # Example usage
            experiment = BlenderExperiment(Nb_frames=10)
            # Define the mesh properties here
            # ...
            mesh = meshio.Mesh(points, cells_dict={"triangle": elements}, point_data={"uvmap": texture_coordinates})
            experiment.add_mesh(name="Mesh1", mesh=mesh, frames=[True, False, True, False, True, False, True, False, True, False])
            experiment.add_mesh_pattern(name="Mesh1", pattern_path="path/to/pattern.png")

        .. warning::
        
            The UV coordinates of the mesh must be defined in the meshio.Mesh object.   
            as "uvmap" in the point_data dictionary.

        .. seealso::

            - :class:`pyblenderSDIC.meshes.TriMesh3D` for more information on how to define a mesh.


        Parameters
        ----------
        name : str
            The name of the mesh.
        
        pattern_path : str
            The path to the pattern image file.
        
        Raises
        -------
        TypeError
            If name is not a string or pattern_path is not a string.
        ValueError
            If a mesh with the same name does not exist in the experiment or in Blender data.
            If the pattern path is not valid or the image format is not supported.
            If the mesh does not have a UVMAP defined.

        Blender Details
        ---------------
        A ``uv_layer`` is created for the mesh and the UVMAP is set to the texture coordinates defined in the mesh.
        The uv_layer is named ``__[pyblenderSDIC]__{name}_uvmap``.

        A node is created in the material node tree to load the image texture.
        The node is named ``__[pyblenderSDIC]__{name}_image_texture``.
        This node is in `CLIP` mode to avoid stretching the image.

        The color output of the image texture node is connected to the second input of the MixRGB node.
        The MixRGB node is connected to the Base Color input of the Principled BSDF node in multiply mode.

        """
        if not isinstance(name, str):
            raise TypeError("name must be a string")
        if not isinstance(pattern_path, str):
            raise TypeError("pattern_path must be a string")
        if name not in self._mesh_objects:
            raise ValueError(f"Mesh with name {name} does not exist.")
        if name not in bpy.data.objects:
            raise ValueError(f"No object with name {name} in Blender data.")
        if not os.path.isfile(pattern_path):
            raise ValueError(f"Pattern path {pattern_path} is not valid.")
        if not pattern_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            raise ValueError(f"Pattern path {pattern_path} is not a valid image format.")
        
        # Ensure the experiment scene is active
        bpy.context.window.scene = self._experiment_scene

        # Get the mesh object
        mesh, blender_mesh = self.get_mesh(name)

        # Check if the mesh has a UVMAP defined
        if mesh.point_data is None or "uvmap" not in mesh.point_data:
            raise ValueError(f"Mesh {name} does not have a UVMAP defined.")
        
        uvmap = mesh.point_data["uvmap"] # (N, 3) array of texture coordinates

        # Setting the UVMap
        if f"__[pyblenderSDIC]__{name}_uvmap" in blender_mesh.data.uv_layers:
            raise ValueError(f"UVMap with name __[pyblenderSDIC]__{name}_uvmap already exists.")
        
        uv_layer = blender_mesh.data.uv_layers.new(name=f"__[pyblenderSDIC]__{name}_uvmap")
        for loop in blender_mesh.data.loops:
            uv_layer.data[loop.index].uv = tuple(uvmap[loop.vertex_index, :2])
            
        # Access the material
        if not blender_mesh.data.materials:
            raise ValueError(f"Mesh '{name}' has no material assigned.")
        material = blender_mesh.data.materials[0]
        if not material.use_nodes:
            raise ValueError(f"Material for '{name}' does not use nodes.")

        # Access the nodes
        nodes = material.node_tree.nodes  # Access the material's node tree
        links = material.node_tree.links  # Access node tree links

        # Add an image texture node
        if f'__[pyblenderSDIC]__{name}_image_texture' in nodes:
            raise ValueError(f"Image texture node with name __[pyblenderSDIC]__{name}_image_texture already exists.")
        
        tex_image_node = nodes.new(type="ShaderNodeTexImage")
        tex_image_node.name = f'__[pyblenderSDIC]__{name}_image_texture'
        tex_image_node.image = bpy.data.images.load(pattern_path)
        tex_image_node.extension = 'CLIP'  

        # Get the MixRGB node and connect the image texture node to it
        if not f'__[pyblenderSDIC]__{name}_mix_basecolor_pattern' in nodes:
            raise ValueError(f"MixRGB node with name __[pyblenderSDIC]__{name}_mix_basecolor_pattern does not exist.")
        
        mix_node = nodes.get(f'__[pyblenderSDIC]__{name}_mix_basecolor_pattern')
        links.new(tex_image_node.outputs['Color'], mix_node.inputs['Color2'])

        # Update the mesh
        blender_mesh.data.update()


    def add_mesh_material(self, name: str, material: MaterialBSDF) -> None:
        r"""
        Add a material to the mesh.

        The material must be an instance of MaterialBSDF.

        .. code-block:: python

            # Example usage
            experiment = BlenderExperiment(Nb_frames=10)
            # Define the mesh properties here
            # ...
            mesh = meshio.Mesh(points, cells_dict={"triangle": elements}, point_data={"uvmap": texture_coordinates})
            experiment.set_mesh(name="Mesh1", mesh=mesh, frames=[True, False, True, False, True, False, True, False, True, False])
            material = MaterialBSDF()
            # Define the material properties here
            # ...
            experiment.add_mesh_material(name="Mesh1", material=material)

        .. seealso::

            - :class:`pyblenderSDIC.materials.MaterialBSDF` for more information on how to define a material.

        .. note::

            The material is not updated when the mesh is updated.
            To update the material, you must call this method again with the new material.

        Parameters
        ----------
        name : str
            The name of the mesh.
        
        material : MaterialBSDF
            The material object to be added.
        
        Raises
        -------
        TypeError
            If name is not a string or material is not an instance of MaterialBSDF.
        ValueError
            If a mesh with the same name does not exist in the experiment or in Blender data.
        """
        if not isinstance(name, str):
            raise TypeError("name must be a string")
        if not isinstance(material, MaterialBSDF):
            raise TypeError("material must be an instance of MaterialBSDF")
        if name not in self._mesh_objects:
            raise ValueError(f"Mesh with name {name} does not exist.")
        if name not in bpy.data.objects:
            raise ValueError(f"No object with name {name} in Blender data.")
        
        # Ensure the experiment scene is active
        bpy.context.window.scene = self._experiment_scene

        # Get the mesh object
        _, blender_mesh = self.get_mesh(name)

        # Access the material
        if not blender_mesh.data.materials:
            raise ValueError(f"Mesh '{name}' has no material assigned.")
        blender_material = blender_mesh.data.materials[0]
        if not blender_material.use_nodes:
            raise ValueError(f"Material for '{name}' does not use nodes.")
        
        # Access the nodes
        nodes = blender_material.node_tree.nodes  # Access the material's node tree
        _ = blender_material.node_tree.links  # Access node tree links

        # Access the Principled BSDF node
        if not 'Principled BSDF' in nodes:
            raise ValueError("No 'Principled BSDF' node found in the material. Ensure nodes are enabled.")
        
        principled = blender_material.node_tree.nodes.get('Principled BSDF')
        
        # Access the mix node
        mix_node = blender_material.node_tree.nodes.get(f'__[pyblenderSDIC]__{name}_mix_basecolor_pattern')
        if not mix_node:
            raise ValueError(f"No MixRGB node found for mesh '{name}'. Ensure the mesh has been added with a material.")

        # Set the properties
        if material.base_color is not None:
            principled.inputs["Base Color"].default_value = material.base_color
            mix_node.inputs["Color1"].default_value = material.base_color
        if material.metallic is not None:
            principled.inputs["Metallic"].default_value = material.metallic
        if material.roughness is not None:
            principled.inputs["Roughness"].default_value = material.roughness
        if material.IOR is not None:
            principled.inputs["IOR"].default_value = material.IOR
        if material.alpha is not None:
            principled.inputs["Alpha"].default_value = material.alpha
        if material.normal is not None:
            principled.inputs["Normal"].default_value = material.normal
        if material.weight is not None:
            principled.inputs["Weight"].default_value = material.weight
        if material.subsurface_weight is not None:
            principled.inputs["Subsurface Weight"].default_value = material.subsurface_weight
        if material.subsurface_radius is not None:
            principled.inputs["Subsurface Radius"].default_value = material.subsurface_radius
        if material.subsurface_scale is not None:
            principled.inputs["Subsurface Scale"].default_value = material.subsurface_scale
        if material.subsurface_IOR is not None:
            principled.inputs["Subsurface IOR"].default_value = material.subsurface_IOR
        if material.subsurface_anisotropy is not None:
            principled.inputs["Subsurface Anisotropy"].default_value = material.subsurface_anisotropy
        if material.specular_IOR_level is not None:
            principled.inputs["Specular IOR Level"].default_value = material.specular_IOR_level
        if material.specular_tint is not None:
            principled.inputs["Specular Tint"].default_value = material.specular_tint
        if material.anisotropic is not None:
            principled.inputs["Anisotropic"].default_value = material.anisotropic
        if material.anisotropic_rotation is not None:
            principled.inputs["Anisotropic Rotation"].default_value = material.anisotropic_rotation
        if material.tangent is not None:
            principled.inputs["Tangent"].default_value = material.tangent
        if material.transmission_weight is not None:
            principled.inputs["Transmission Weight"].default_value = material.transmission_weight
        if material.coat_weight is not None:
            principled.inputs["Coat Weight"].default_value = material.coat_weight
        if material.coat_roughness is not None:
            principled.inputs["Coat Roughness"].default_value = material.coat_roughness
        if material.coat_IOR is not None:
            principled.inputs["Coat IOR"].default_value = material.coat_IOR
        if material.coat_tint is not None:
            principled.inputs["Coat Tint"].default_value = material.coat_tint
        if material.coat_normal is not None:
            principled.inputs["Coat Normal"].default_value = material.coat_normal
        if material.sheen_weight is not None:
            principled.inputs["Sheen Weight"].default_value = material.sheen_weight
        if material.sheen_roughness is not None:
            principled.inputs["Sheen Roughness"].default_value = material.sheen_roughness
        if material.sheen_tint is not None:
            principled.inputs["Sheen Tint"].default_value = material.sheen_tint
        if material.emission_color is not None:
            principled.inputs["Emission Color"].default_value = material.emission_color
        if material.emission_strength is not None:
            principled.inputs["Emission Strength"].default_value = material.emission_strength
        if material.thin_film_thickness is not None:
            principled.inputs["Thin Film Thickness"].default_value = material.thin_film_thickness
        if material.thin_film_IOR is not None:
            principled.inputs["Thin Film IOR"].default_value = material.thin_film_IOR

        # Update the mesh
        blender_mesh.data.update()


    def set_mesh_frames(self, name: str, frames: Optional[List[bool]] = None) -> None:
        r"""
        Set the frames for which the mesh will be active.
        If None, the mesh will be active for all frames.
        Else a list of booleans must be provided, where each boolean indicates if the mesh is active for that frame.

        Parameters
        ----------
        name : str
            The name of the mesh.
        
        frames : List[bool], optional
            A list of booleans indicating which frames the mesh will be active.
            If None, the mesh will be active for all frames (default is None).

        Raises
        -------
        TypeError
            If name is not a string or frames is not a list of booleans.
        ValueError
            If a mesh with the same name does not exist in the experiment or in Blender data.
            If the length of frames is not equal to the number of frames in the experiment (end_frame).
        """
        if not isinstance(name, str):
            raise TypeError("name must be a string")
        if name not in self._mesh_objects:
            raise ValueError(f"Mesh with name {name} does not exist.")
        if name not in bpy.data.objects:
            raise ValueError(f"No object with name {name} in Blender data.")
        if frames is not None:
            if not isinstance(frames, list):
                raise TypeError("frames must be a list of booleans")
            if len(frames) != self._end_frame:
                raise ValueError(f"Length of frames must be equal to {self._end_frame}")
            for frame in frames:
                if not isinstance(frame, bool):
                    raise TypeError("frames must be a list of booleans")
        else:
            frames = [True] * self._end_frame
        
        # Ensure the experiment scene is active
        bpy.context.window.scene = self._experiment_scene

        # Get the mesh object
        _, blender_mesh = self.get_mesh(name)

        # Set the visibility of the mesh for each frame
        for frame in range(1, self._end_frame + 1):
            blender_mesh.hide_render = not frames[frame - 1]
            blender_mesh.keyframe_insert(data_path="hide_render", frame=frame)

        # Apply the changes to the mesh
        blender_mesh.data.update()

        # Set the frames for which the mesh will be active
        self._mesh_objects[name][1] = frames

    
    def get_mesh_frames(self, name: str) -> List[bool]:
        r"""
        Get the frames for which the mesh is active.
        
        Parameters
        ----------
        name : str
            The name of the mesh.
        
        Returns
        -------
        List[bool]
            A list of booleans indicating which frames the mesh is active.
        
        Raises
        -------
        TypeError
            If name is not a string.
        ValueError
            If a mesh with the same name does not exist in the experiment or in Blender data.
        """
        if not isinstance(name, str):
            raise TypeError("name must be a string")
        if name not in self._mesh_objects:
            raise ValueError(f"Mesh with name {name} does not exist.")
        if name not in bpy.data.objects:
            raise ValueError(f"No object with name {name} in Blender data.")
        
        # Ensure the experiment scene is active
        bpy.context.window.scene = self._experiment_scene

        # Get the frames for which the mesh is active
        return self._mesh_objects[name][1]
    

    def get_mesh(self, name: str) -> Tuple[meshio.Mesh, Object]:
        r"""
        Get the mesh object and its Blender object.
        
        Parameters
        ----------
        name : str
            The name of the mesh to retrieve.
        
        Returns
        -------
        Tuple[meshio.Mesh, Object]
            A tuple containing the meshio.Mesh object and its corresponding Blender object.
        """
        if not isinstance(name, str):
            raise TypeError("name must be a string")
        if name not in self._mesh_objects:
            raise ValueError(f"Mesh with name {name} does not exist.")
        if name not in bpy.data.objects:
            raise ValueError(f"No object with name {name} in Blender data.")
        
        # Ensure the experiment scene is active
        bpy.context.window.scene = self._experiment_scene

        # Get the mesh object
        mesh = self._mesh_objects[name][0]
        blender_mesh = bpy.data.objects[name]

        # Check the types of the objects
        if not isinstance(mesh, meshio.Mesh):
            raise TypeError("[ERROR] mesh must be an instance of meshio.Mesh")
        if not isinstance(blender_mesh, Object):
            raise TypeError("[ERROR] blender_mesh must be an instance of Object")
        
        return mesh, blender_mesh
    

    def get_mesh_names(self) -> List[str]:
        r"""
        Get the names of all meshes in the experiment.

        Returns
        -------
        List[str]
            A list of mesh names.
        """
        return list(self._mesh_objects.keys())


    def remove_mesh(self, name: str) -> None:
        r"""
        Remove a mesh from the experiment.

        Parameters
        ----------
        name : str
            The name of the mesh to remove.

        Raises
        -------
        TypeError
            If name is not a string.
        ValueError
            If a mesh with the same name does not exist in the experiment or in Blender data.

        Blender Details
        ---------------
        When removing a mesh, the following actions are performed:

        - The mesh ``{name}`` is removed from the Blender scene.
        - The material ``__[pyblenderSDIC]__{name}_material`` is removed from the Blender data.
        - The MixRGB node ``__[pyblenderSDIC]__{name}_mix_basecolor_pattern`` is removed from the material node tree.
        - The uv_layer ``__[pyblenderSDIC]__{name}_uvmap`` is removed from the mesh data.
        - The node texture ``__[pyblenderSDIC]__{name}_image_texture`` is removed from the material node tree.

        """
        if not isinstance(name, str):
            raise TypeError("name must be a string")
        if name not in self._mesh_objects:
            raise ValueError(f"Mesh with name {name} does not exist.")
        if name not in bpy.data.objects:
            raise ValueError(f"No object with name {name} in Blender data.")
        
        # Ensure the experiment scene is active
        bpy.context.window.scene = self._experiment_scene

        # Remove the mesh from the scene
        _, blender_mesh = self.get_mesh(name)

        # Access the material
        if not blender_mesh.data.materials:
            raise ValueError(f"Mesh '{name}' has no material assigned.")
        material = blender_mesh.data.materials[0]
        if not material.use_nodes:
            raise ValueError(f"Material for '{name}' does not use nodes.")
        
        # Access the nodes of the material
        nodes = material.node_tree.nodes  # Access the material's node tree
        _ = material.node_tree.links  # Access node tree links

        # Remove the MixRGB node
        if f'__[pyblenderSDIC]__{name}_mix_basecolor_pattern' in nodes:
            mix_node = nodes[f'__[pyblenderSDIC]__{name}_mix_basecolor_pattern']
            nodes.remove(mix_node)

        # Remove the material
        blender_mesh.data.materials.clear()
        bpy.data.materials.remove(material)

        # Remove the uv_layer
        if f'__[pyblenderSDIC]__{name}_uvmap' in blender_mesh.data.uv_layers:
            uv_layer = blender_mesh.data.uv_layers[f'__[pyblenderSDIC]__{name}_uvmap']
            blender_mesh.data.uv_layers.remove(uv_layer)

        # Remove the image texture node
        if f'__[pyblenderSDIC]__{name}_image_texture' in nodes:
            tex_image_node = nodes[f'__[pyblenderSDIC]__{name}_image_texture']
            if tex_image_node.image is not None:
                bpy.data.images.remove(tex_image_node.image)
            nodes.remove(tex_image_node)

        # Remove the mesh from the scene
        bpy.data.objects.remove(blender_mesh)
        self._mesh_objects.pop(name)

    
    def update_mesh(self, name: str) -> None:
        r"""
        Update the mesh properties in the Blender scene.
        This method must be called after updating the mesh properties in the meshio.Mesh object.

        .. code-block:: python

            # Example usage
            experiment = BlenderExperiment(Nb_frames=10)
            # Define the mesh properties here
            # ...
            mesh = meshio.Mesh(points, cells_dict={"triangle": elements}, point_data={"uvmap": texture_coordinates})
            experiment.add_mesh(name="Mesh1", mesh=mesh, frames=[True, False, True, False, True, False, True, False, True, False])
            # Update some nodes of the mesh here
            # ...
            experiment.update_mesh(name="Mesh1") # To set active the modifications in the Blender scene

        .. note::

            If a pattern is setted to the mesh, the uvmap are also updated.

        .. warning::

            The number of nodes must be unchanged !!!

            Furthermore, this method does not update the material properties of the mesh and the pattern image: 
            
            - To update the material properties, use the :meth:`pyblenderSDIC.BlenderExperiment.add_mesh_material` method again.
            - To change the pattern image, use the :meth:`pyblenderSDIC.BlenderExperiment.change_mesh_pattern` method.
        
        Parameters
        ----------
        name : str
            The name of the mesh to update.
        """
        if not isinstance(name, str):
            raise TypeError("name must be a string")
        if name not in self._mesh_objects:
            raise ValueError(f"Mesh with name {name} does not exist.")
        if name not in bpy.data.objects:
            raise ValueError(f"No object with name {name} in Blender data.")
        
        # Ensure the experiment scene is active
        bpy.context.window.scene = self._experiment_scene

        # Get the mesh object
        mesh, blender_mesh = self.get_mesh(name)

        # Extract the vertices and faces from the mesh
        points = mesh.points
        cells = mesh.cells_dict["triangle"]
        uvmap = mesh.point_data["uvmap"] if mesh.point_data is not None and "uvmap" in mesh.point_data else None

        # Apply the vertices and faces to the Blender mesh
        for i, vertex in enumerate(blender_mesh.data.vertices):
            vertex.co.x = points[i, 0]
            vertex.co.y = points[i, 1]
            vertex.co.z = points[i, 2]

        for i, poly in enumerate(blender_mesh.data.polygons):
            poly.vertices = [int(vertex) for vertex in cells[i]]

        # Update the uvmap if it exists
        if f'__[pyblenderSDIC]__{name}_uvmap' in blender_mesh.data.uv_layers:
            uv_layer = blender_mesh.data.uv_layers[f'__[pyblenderSDIC]__{name}_uvmap']
            if uvmap is not None:
                for loop in blender_mesh.data.loops:
                    uv_layer.data[loop.index].uv = tuple(uvmap[loop.vertex_index, :2])
            else:
                raise ValueError(f"Mesh {name} does not have a UVMAP defined but the uv_layer exists.")

        # Update the mesh
        blender_mesh.data.update()


    def change_mesh_pattern(self, name: str, pattern_path: str) -> None:
        r"""
        Change the pattern of the mesh material.

        The pattern must be a valid image file path (e.g., PNG, JPEG).
        The pattern will be applied to the mesh material as a texture.

        .. code-block:: python

            # Example usage
            experiment = BlenderExperiment(Nb_frames=10)
            # Define the mesh properties here
            # ...
            mesh = meshio.Mesh(points, cells_dict={"triangle": elements}, point_data={"uvmap": texture_coordinates})
            experiment.add_mesh(name="Mesh1", mesh=mesh, frames=[True, False, True, False, True, False, True, False, True, False])
            experiment.add_mesh_pattern(name="Mesh1", pattern_path="path/to/pattern.png")
            experiment.change_mesh_pattern(name="Mesh1", pattern_path="path/to/new_pattern.png")

        .. seealso::

            - :class:`pyblenderSDIC.BlenderExperiment.add_mesh_pattern` for more information on how to add a pattern.

        Parameters
        ----------
        name : str
            The name of the mesh.
        
        pattern_path : str
            The path to the new pattern image file.
        
        Raises
        -------
        TypeError
            If name is not a string or pattern_path is not a string.
        ValueError
            If a mesh with the same name does not exist in the experiment or in Blender data.
            If the pattern path is not valid or the image format is not supported.
        
        """
        if not isinstance(name, str):
            raise TypeError("name must be a string")
        if not isinstance(pattern_path, str):
            raise TypeError("pattern_path must be a string")
        if name not in self._mesh_objects:
            raise ValueError(f"Mesh with name {name} does not exist.")
        if name not in bpy.data.objects:
            raise ValueError(f"No object with name {name} in Blender data.")
        if not os.path.isfile(pattern_path):
            raise ValueError(f"Pattern path {pattern_path} is not valid.")
        if not pattern_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            raise ValueError(f"Pattern path {pattern_path} is not a valid image format.")
        
        # Ensure the experiment scene is active
        bpy.context.window.scene = self._experiment_scene

        # Get the mesh object
        _, blender_mesh = self.get_mesh(name)

        # Access the material
        if not blender_mesh.data.materials:
            raise ValueError(f"Mesh '{name}' has no material assigned.")
        material = blender_mesh.data.materials[0]
        if not material.use_nodes:
            raise ValueError(f"Material for '{name}' does not use nodes.")

        # Access the nodes
        nodes = material.node_tree.nodes  # Access the material's node tree
        _ = material.node_tree.links  # Access node tree links

        # Add an image texture node
        if not f'__[pyblenderSDIC]__{name}_image_texture' in nodes:
            raise ValueError(f"Image texture node with name __[pyblenderSDIC]__{name}_image_texture does not exist.")
        
        tex_image_node = nodes[f'__[pyblenderSDIC]__{name}_image_texture']
        if tex_image_node.image is not None:
            bpy.data.images.remove(tex_image_node.image)
        tex_image_node.image = bpy.data.images.load(pattern_path)
        tex_image_node.extension = 'CLIP'  # Set the extension to CLIP to avoid stretching the image
        tex_image_node.image.reload()  # Reload the image to apply the changes

        # Update the mesh
        blender_mesh.data.update()
   
    

    # =======================================================
    # Blender Scene Management (SPOTLIGHT)
    # =======================================================
    def add_spotlight(self, name: str, spotlight: SpotLight, frames: Optional[List[bool]] = None) -> None:
        r"""
        Add a spotlight to the experiment.

        The spotlight must be an instance of SpotLight.

        .. code-block:: python

            # Example usage
            experiment = BlenderExperiment(Nb_frames=10)
            spotlight = SpotLight()
            # Define the SpotLight properties here
            # ...
            experiment.add_spotlight(name="Spotlight1", spotlight=SpotLight, frames=[True, False, True, False, True, False, True, False, True, False])

        .. seealso::

            - :class:`pyblenderSDIC.SpotLight` for more information on how to define a spotlight.     

        Parameters
        ----------
        name : str
            The name of the spotlight.
        
        spotlight : SpotLight
            The spotlight object to be added.
        
        frames : List[bool], optional
            A list of booleans indicating which frames the spotlight will be active.
            If None, the spotlight will be active for all frames (default is None).

        Raises
        -------
        TypeError
            If name is not a string or spotlight is not an instance of SpotLight.
        ValueError
            If a spotlight with the same name already exists in the experiment or in Blender data.
            If the length of frames is not equal to the number of frames in the experiment (end_frame).
        """
        if not isinstance(name, str):
            raise TypeError("name must be a string")
        if name in self._spotlight_objects:
            raise ValueError(f"Spotlight with name {name} already exists.")
        if name in bpy.data.objects:
            raise ValueError(f"Object with name {name} already exists in Blender data.")
        if not isinstance(spotlight, SpotLight):
            raise TypeError("spotlight must be an instance of Spotlight")

        # Ensure the experiment scene is active
        bpy.context.window.scene = self._experiment_scene

        # Ensure the experiment scene is active
        bpy.context.window.scene = self._experiment_scene

        # Create the spotlight
        spotlight_data = bpy.data.lights.new(name=name, type='SPOT')
        blender_spotlight = bpy.data.objects.new(name, spotlight_data)
        self._spotlight_objects[name] = [spotlight, None]

        # Set the frames for which the mesh will be active
        self.set_spotlight_frames(name, frames)

        # Link the spotlight to the scene
        self._experiment_scene.collection.objects.link(blender_spotlight)

        # Set the spotlight properties
        self.update_spotlight(name)

    
    def set_spotlight_frames(self, name: str, frames: Optional[List[bool]] = None) -> None:
        r"""
        Set the frames for which the spotlight will be active.
        If None, the spotlight will be active for all frames.
        Else a list of booleans must be provided, where each boolean indicates if the spotlight is active for that frame.

        Parameters
        ----------
        name : str
            The name of the spotlight.
        
        frames : List[bool], optional
            A list of booleans indicating which frames the spotlight will be active.
            If None, the spotlight will be active for all frames (default is None).

        Raises
        -------
        TypeError
            If name is not a string or frames is not a list of booleans.
        ValueError
            If a spotlight with the same name does not exist in the experiment or in Blender data.
            If the length of frames is not equal to the number of frames in the experiment (end_frame).
        """
        if not isinstance(name, str):
            raise TypeError("name must be a string")
        if name not in self._spotlight_objects:
            raise ValueError(f"Spotlight with name {name} does not exist.")
        if name not in bpy.data.objects:
            raise ValueError(f"No object with name {name} in Blender data.")
        if frames is not None:
            if not isinstance(frames, list):
                raise TypeError("frames must be a list of booleans")
            if len(frames) != self._end_frame:
                raise ValueError(f"Length of frames must be equal to {self._end_frame}")
            for frame in frames:
                if not isinstance(frame, bool):
                    raise TypeError("frames must be a list of booleans")
        else:
            frames = [True] * self._end_frame
        
        # Ensure the experiment scene is active
        bpy.context.window.scene = self._experiment_scene

        # Get the spotlight object
        spotlight, blender_spotlight = self.get_spotlight(name)

        # Set the visibility of the spotlight for each frame
        for frame in range(1, self._end_frame + 1):
            blender_spotlight.data.energy = spotlight.energy if frames[frame - 1] else 0
            blender_spotlight.data.keyframe_insert(data_path="energy", frame=frame)

        # Set the frames for which the mesh will be active
        self._spotlight_objects[name][1] = frames

    
    def get_spotlight_frames(self, name: str) -> List[bool]:
        r"""
        Get the frames for which the spotlight is active.
        
        Parameters
        ----------
        name : str
            The name of the spotlight.
        
        Returns
        -------
        List[bool]
            A list of booleans indicating which frames the spotlight is active.
        
        Raises
        -------
        TypeError
            If name is not a string.
        ValueError
            If a spotlight with the same name does not exist in the experiment or in Blender data.
        """
        if not isinstance(name, str):
            raise TypeError("name must be a string")
        if name not in self._spotlight_objects:
            raise ValueError(f"Spotlight with name {name} does not exist.")
        if name not in bpy.data.objects:
            raise ValueError(f"No object with name {name} in Blender data.")
        
        # Ensure the experiment scene is active
        bpy.context.window.scene = self._experiment_scene

        # Get the frames for which the spotlight is active
        return self._spotlight_objects[name][1]
    

    def get_spotlight(self, name: str) -> Tuple[SpotLight, Object]:
        r"""
        Get the spotlight object and its Blender object.
        
        Parameters
        ----------
        name : str
            The name of the spotlight to retrieve.
        
        Returns
        -------
        Tuple[SpotLight, Object]
            A tuple containing the SpotLight object and its corresponding Blender object.
        """
        if not isinstance(name, str):
            raise TypeError("name must be a string")
        if name not in self._spotlight_objects:
            raise ValueError(f"Spotlight with name {name} does not exist.")
        if name not in bpy.data.objects:
            raise ValueError(f"No object with name {name} in Blender data.")
        
        # Ensure the experiment scene is active
        bpy.context.window.scene = self._experiment_scene

        # Get the spotlight object
        spotlight = self._spotlight_objects[name][0]
        blender_spotlight = bpy.data.objects[name]

        # Check the types of the objects
        if not isinstance(spotlight, SpotLight):
            raise TypeError("[ERROR] spotlight must be an instance of SpotLight")
        if not isinstance(blender_spotlight, Object):
            raise TypeError("[ERROR] blender_spotlight must be an instance of Object")
        
        return spotlight, blender_spotlight
    

    def remove_spotlight(self, name: str) -> None:
        r"""
        Remove a spotlight from the experiment.

        Parameters
        ----------
        name : str
            The name of the spotlight to remove.

        Raises
        -------
        TypeError
            If name is not a string.
        ValueError
            If a spotlight with the same name does not exist in the experiment or in Blender data.
        """
        if not isinstance(name, str):
            raise TypeError("name must be a string")
        if name not in self._spotlight_objects:
            raise ValueError(f"Spotlight with name {name} does not exist.")
        if name not in bpy.data.objects:
            raise ValueError(f"No object with name {name} in Blender data.")
        
        # Ensure the experiment scene is active
        bpy.context.window.scene = self._experiment_scene

        # Remove the spotlight from the scene
        spotlight, blender_spotlight = self.get_spotlight(name)
        bpy.data.objects.remove(blender_spotlight)
        self._spotlight_objects.pop(name)

    
    def update_spotlight(self, name: str) -> None:
        r"""
        Update the spotlight properties in the Blender scene.
        This method must be called after updating the spotlight properties in the SpotLight object.

        .. code-block:: python

            # Example usage
            experiment = BlenderExperiment(Nb_frames=10)
            spotlight = SpotLight()
            # Define the SpotLight properties here
            # ...
            experiment.add_spotlight(name="Spotlight1", spotlight=spotlight, frames=[True, False, True, False, True, False, True, False, True, False])
            # Update some nodes of the spotlight here
            # ...
            experiment.update_spotlight(name="Spotlight1") # To set active the modifications in the Blender scene
        
        .. warning::

            This method does not update the material properties of the spotlight.
            Only the position and rotation are updated.  
            The number of nodes mustnot be changed.

        Parameters
        ----------
        name : str
            The name of the spotlight to update.
        """
        if not isinstance(name, str):
            raise TypeError("name must be a string")
        if name not in self._spotlight_objects:
            raise ValueError(f"Spotlight with name {name} does not exist.")
        if name not in bpy.data.objects:
            raise ValueError(f"No object with name {name} in Blender data.")
        
        # Ensure the experiment scene is active
        bpy.context.window.scene = self._experiment_scene

        # Get the spotlight object
        spotlight, blender_spotlight = self.get_spotlight(name)

        # Set the position and rotation of the camera
        rotation, translation = spotlight.get_OpenGL_RT()
        blender_spotlight.location = translation
        blender_quaternion = rotation.as_quat(scalar_first=True) # blender convention [qw, qx, qy, qz]
        blender_spotlight.rotation_mode = 'QUATERNION'
        blender_spotlight.rotation_quaternion = blender_quaternion

        # Set the spotlight properties
        blender_spotlight.data.energy = spotlight.energy
        blender_spotlight.data.spot_size = spotlight.spot_size
        blender_spotlight.data.spot_blend = spotlight.spot_blend

        # Reset the frames for which the spotlight will be active (in order to have energy = 0 for the frames where the spotlight is not active)
        self.set_spotlight_frames(name, self.get_spotlight_frames(name))



    # =======================================================
    # Blender Scene Management (DEFAULT)
    # =======================================================
    def set_default_background(self) -> None:
        r""" 
        
        Set the default background for the experiment scene. 
        
        """
        bpy.ops.world.new()
        bpy.context.scene.world = bpy.data.worlds["World"]
        world_light = bpy.data.worlds["World"].node_tree.nodes["Background"]
        world_light.inputs[0].default_value = (0.5, 0.5, 0.5, 1)
        world_light.inputs[1].default_value = 1.0

    def update_scene(self) -> None:
        r"""
        Update the Blender scene.
        This method must be called after adding or modifying any objects in the scene.
        """
        # Ensure the experiment scene is active
        bpy.context.window.scene = self._experiment_scene

        # Update the scene
        bpy.ops.ptcache.free_bake_all()
        bpy.ops.ptcache.bake_all()       
        bpy.context.view_layer.update()
        bpy.context.view_layer.depsgraph.update()

    def set_active_frame(self, frame: int) -> None:
        r"""
        Set the current frame of the Blender scene.

        Parameters
        ----------
        frame : int
            The frame number to set.
        
        Raises
        -------
        TypeError
            If frame is not an integer.
        ValueError
            If frame is out of range.
        """
        if not isinstance(frame, Integral):
            raise TypeError("frame must be an integer")
        if frame < 1 or frame > self._end_frame:
            raise ValueError(f"frame must be between 1 and {self._end_frame}")
        
        # Ensure the experiment scene is active
        bpy.context.window.scene = self._experiment_scene

        # Set the current frame
        bpy.context.scene.frame_set(int(frame))
        self._active_frame = int(frame)



    # =======================================================
    # Blender Scene RENDER
    # =======================================================
    def render(
        self,
        output_path: str,
        output_format: str = "TIFF",
        color_mode: str = "BW",
        color_depth: str = "8",
        N_samples: int = 200,
        ) -> None:
        r"""
        Render the experiment scene.

        Parameters
        ----------
        output_path : str
            The path to save the rendered image.

        output_format : str, optional
            The output format of the rendered image (default is "TIFF").
        
        color_mode : str, optional
            The color mode of the rendered image (default is "BW").

        color_depth : str, optional
            The color depth of the rendered image (default is "8").
        
        N_samples : int, optional
            The number of samples for rendering (default is 200).

        Raises
        -------
        TypeError
            If name or output_path is not a string.
        ValueError
            If the output format is not supported.
            If the frame or the camera is not set.
        """
        if not isinstance(output_path, str):
            raise TypeError("output_path must be a string")
        
        # Ensure the experiment scene is active
        bpy.context.window.scene = self._experiment_scene

        # Set the current frame and camera
        if self._active_frame is None:
            raise ValueError("No active frame set. Use set_active_frame() to set the current frame.")
        
        if self._active_camera is None:
            raise ValueError("No active camera set. Use set_active_camera() to set the current camera.")
        
        camera_frames = self.get_camera_frames(self._active_camera)
        if camera_frames[self._active_frame - 1] is False:
            raise ValueError(f"Camera {self._active_camera} is not active for frame {self._active_frame}.")
    
        # Set the render settings
        bpy.context.scene.render.image_settings.file_format = output_format
        bpy.context.scene.render.image_settings.tiff_codec = "NONE"
        bpy.context.scene.render.image_settings.color_mode = color_mode
        bpy.context.scene.render.image_settings.color_depth = color_depth
        bpy.context.scene.render.image_settings.compression = 0
        bpy.context.scene.render.engine = 'CYCLES'
        bpy.context.scene.cycles.device = 'GPU'
        bpy.context.scene.cycles.samples = N_samples
        bpy.context.scene.cycles.use_denoising = False
        bpy.context.scene.render.filepath = output_path

        # Perform the rendering
        bpy.ops.render.render(write_still=True)
        print(f"Rendered image saved to {output_path}")
