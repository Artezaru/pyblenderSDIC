from __future__ import annotations
import numpy
import meshio
import open3d
import json
from typing import Optional, Dict, Union, Sequence
from numbers import Integral

from .intersect_points import IntersectPoints



class TriMesh3DDisplacementAccessor:
    """
    Provides access to the displacements of a mesh at different frames.

    This accessor allows retrieving and modifying the displacements at a given frame
    using bracket notation:

    .. code-block:: python

        # Get displacements at frame 3
        disp = mesh.displacements[3]

        # Set displacements at frame 3
        mesh.displacements[3] = disp

    If the requested frame is 0, the displacements are assumed to be zero,
    representing the reference (undeformed) state. Setting displacements for frame 0
    will directly update the reference node positions (`point_data["nodes"]`).

    .. warning::

        This accessor uses ``numpy.asarray`` on the internal and provided data.
        Any modification of the array returned by ``__getitem__`` will affect
        the original mesh data, unless explicitly copied.

        .. code-block:: python

            # Modifies the mesh
            mesh.displacements[3][:, 0] = 1.0 # Modifies the displacement

            # Safe approach
            d = mesh.displacements[3].copy()
            d[:, 0] = 1.0  # Does NOT affect the mesh object

            # Similarly, setting displacements with a reference modifies the mesh
            mesh.displacements[3] = new_disp
            new_disp[:, 0] = 2.0  # Also modifies mesh

            # Safe approach
            mesh.displacements[3] = new_disp.copy()
            new_disp[:, 0] = 2.0

    Parameters
    ----------
    parent : object
        The mesh object that contains the displacement data under `point_data`.

    See Also
    --------
    - :attr:`point_data` for internal storage of per-point data.
    - :attr:`nodes` for reference node coordinates.
    """

    def __init__(self, parent):
        self._parent = parent

    def __getitem__(self, frame: int) -> numpy.ndarray:
        """
        Retrieve the displacements at the given frame.

        Parameters
        ----------
        frame : int
            The frame index. Must be a non-negative integer.

        Returns
        -------
        numpy.ndarray
            A (N, 3) array of displacements.

            - For frame 0, returns zeros.
            - For other frames, returns the stored displacements (as a view on the original data).

        Raises
        ------
        TypeError
            If `frame` is not an integer.
        ValueError
            If `frame` is negative or not defined in the mesh.
        """
        frame = self._parent._integral_frame(frame)

        if frame == 0:
            return numpy.zeros_like(self._parent.nodes)
        
        if self._parent.point_data is None:
            raise ValueError("Displacements not defined (mesh.point_data is None). Please provide data.")

        key = f"displacements_frame_{frame}"
        if key not in self._parent.point_data:
            raise ValueError(f"Displacements for frame {frame} not found. Please provide data.")
        displacements = numpy.asarray(self._parent.point_data[key], dtype=numpy.float64)

        # Check the shape of the displacements
        if not displacements.shape == (self._parent.Nnodes, 3):
            raise ValueError(f"[INTERNAL CLASS ERROR] Displacements for frame {frame} is not well formatted. Expected shape ({self._parent.Nnodes}, 3).")
        return displacements
    

    def __setitem__(self, frame: int, displacements: Optional[numpy.ndarray]) -> None:
        """
        Set the displacements at the given frame.

        If the frame is 0, the displacements are added to the reference state (nodes at frame 0).
        If the frame is greater than 0, the displacements are stored in the parent point data.
        If the displacements are None for a defined frame, the corresponding data is removed.

        Parameters
        ----------
        frame : int
            The frame index. Must be a non-negative integer.
        displacements : numpy.ndarray
            A (N, 3) array of displacements to assign.

        Raises
        ------
        TypeError
            If `frame` is not an integer.
        ValueError
            If `frame` is negative or if `displacements` does not have shape (N, 3).
        """
        frame = self._parent._integral_frame(frame)

        # Handle the case None displacements
        if displacements is None:
            key = f"displacements_frame_{frame}"
            if frame == 0:
                raise ValueError("Cannot remove reference nodes (frame=0) by setting displacements[0] to None.")
            if self._parent.point_data is None:
                return
            if key in self._parent.point_data:
                del self._parent.point_data[key]
            return

        # Handle the case of displacements
        displacements = numpy.asarray(displacements, dtype=numpy.float64)
        if displacements.shape != (self._parent.Nnodes, 3):
            raise ValueError(f"Displacements must have shape ({self._parent.Nnodes}, 3).")

        # Set the displacements if frame is 0
        if frame == 0:
            self._parent.nodes = self._parent.nodes + displacements

        # If point_data is None, create it
        if self._parent.point_data is None:
            self._parent.point_data = {}

        self._parent.point_data[f"displacements_frame_{frame}"] = displacements








class TriMesh3D(meshio.Mesh):
    """
    Represents a triangular 3D mesh with support for UV mapping, deformations across multiple frames, 
    and compatibility with the VTK format via `meshio`.

    This class is a subclass of `meshio.Mesh` and is designed to handle triangular surface meshes
    in 3D space. It includes support for texture mapping (UV coordinates), nodal displacements over time,
    and visualization tools.

    .. note::

        - Only triangular meshes are supported.
        - UV coordinates should lie between 0 and 1.
        - Displacements are relative to the reference frame (frame 0).
        - Setting displacements overwrites existing data for that frame.

    .. warning::

        The number of nodes and elements are not designed to change after the mesh is created.

    Mesh Structure
    --------------

    - ``points`` (alias ``nodes``): A NumPy array of shape (N, 3) representing the coordinates of N mesh nodes.
    - ``cells_dict{"triangle"}`` (alias ``elements``): A NumPy array of shape (M, 3) representing M triangular elements defined by node indices.
    - ``point_data``: A dictionary storing data associated with each point, such as UV coordinates and displacements.

    .. code-block:: python

        mesh.nodes         # numpy.ndarray of shape (N, 3)
        mesh.elements      # numpy.ndarray of shape (M, 3)

    UV Mapping
    ----------

    UV coordinates can be used to apply textures on the mesh surface. They are stored in the point data 
    under the key ``uvmap``, with shape (N, 3). Only the first two dimensions are used (U, V). The third 
    dimension (Z) is filled with zeros for VTK compatibility.

    - ``uvmap``: Returns the full UV map (N, 3)
    - ``get_uvmap2D()``: Returns the UV coordinates (N, 2), discarding the Z component
    - ``set_uvmap2D()``: Set the UV coordinates for the 2D map

    .. code-block:: python

        mesh.uvmap         # numpy.ndarray of shape (N, 3)
        mesh.get_uvmap2D() # numpy.ndarray of shape (N, 2)
        mesh.set_uvmap2D(uvmap2d)  # Set UV 2D map

    Deformation by Frame
    --------------------

    The mesh supports deformations over time or simulation frames. For each frame `f`, nodal displacements
    are stored under the point data key ``displacements_frame_{f}``.

    - Frame 0 is the reference state (undeformed); displacements are always zero.

    To access displacements or deformed coordinates:

    .. code-block:: python

        disp_f3 = mesh.displacements[3]        # displacement at frame 3      # shape (N, 3)
        nodes_f3 = mesh.get_nodes(frame=3)     # deformed nodes
        nodes_f3 = mesh.nodes + disp_f3       # equivalent result

    To set nodal displacements:

    .. code-block:: python

        mesh.displacements[3] = displacements  # Set displacements at frame 3
        mesh.set_nodes(mesh.nodes + displacements, frame=3)  # Or set nodes directly

    If a displacement is set to None, the corresponding frame will be removed from the mesh.

    .. warning::

        Displacements are always returned as `numpy.asarray`. Changes to the returned arrays affect the mesh directly,
        unless explicitly copied.

    Instantiation
    -------------

    It is recommended to define only the ``points`` and ``cells`` when creating the mesh, corresponding
    to the reference (undeformed) state. You can later assign UV coordinates and displacements.

    Parameters
    ----------
    points : array_like
        Node coordinates of the mesh, shape (N, 3).
    
    cells : array_like
        Triangle definitions, shape (M, 3), using zero-based node indices.
    
    point_data : dict, optional
        Point data dictionary (e.g. UV maps, displacements).

    cell_data : dict, optional
        Optional cell data dictionary.

    field_data : dict, optional
        Optional field data dictionary.

    **kwargs
        Additional keyword arguments passed to `meshio.Mesh`.

    Examples
    --------

    Create a triangular mesh with UV coordinates:

    .. code-block:: python

        import numpy as np
        from pyblenderSDIC.meshes import TriMesh3D

        nodes = numpy.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0]
        ])

        elements = numpy.array([
            [0, 2, 1],
            [0, 3, 2],
            [0, 1, 3],
            [1, 4, 3],
            [2, 3, 4],
            [2, 4, 1]
        ])

        uvmap = numpy.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [0.5, 0.5]
        ])

        cells = {"triangle": elements}
        mesh = TriMesh3D(points=nodes, cells=cells)
        mesh.uvmap = uvmap  # Automatically converts to (N, 3) with zero-padded Z

    Save the mesh to a VTK file:

    >>> mesh.save_to_vtk("test_mesh.vtk")
    """
    def __init__(self, points, cells, point_data = None, cell_data = None, field_data = None, **kwargs):
        super().__init__(points, cells, point_data, cell_data, field_data, **kwargs)
        if not 'triangle' in self.cells_dict:
            raise ValueError("Only triangular meshes are supported. Please provide a mesh with 'triangle' elements.")
        if self.cells[0].type != "triangle":
            raise ValueError("The first cell type must be 'triangle' because the class use self.cells[0] to access the elements.")
        self.displacements = TriMesh3DDisplacementAccessor(self)

    # ===========================================================
    # I/O methods
    # ===========================================================
    @classmethod
    def from_meshio(cls, mesh: meshio.Mesh) -> TriMesh3D:
        """
        Create a TriMesh3D instance from a meshio.Mesh object.

        .. note::

            The following fields are extracted:

            - mesh.points → points
            - mesh.cells → cells
            - mesh.point_data → point_data
            - mesh.cell_data → cell_data
            - mesh.field_data → field_data

        .. code-block:: python

            import meshio
            from pyblenderSDIC.mesh import TriMesh3D

            # Read the mesh from a file
            mesh = meshio.read("path/to/mesh.vtk")
            # Create a TriMesh3D instance from the meshio object
            trimesh3d = TriMesh3D.from_meshio(mesh)

        .. warning::

            If a UV mapping is expected, it must be stored in `mesh.point_data["uvmap"]` with shape (N, 3)
            and values between 0 and 1. No automatic normalization is performed.
            
            If displacements are defined, they must be stored in `mesh.point_data["displacements_frame_{k}"]`
            and must be of shape (N, 3) for each frame k. The frame index must be a positive integer.
            No automatic normalization is performed.

        Parameters
        ----------
        mesh : meshio.Mesh
            The meshio.Mesh object to convert.

        Returns
        -------
        TriMesh3D
            The created TriMesh3D instance.
        """
        return cls(points=mesh.points, cells=mesh.cells, point_data=mesh.point_data, cell_data=mesh.cell_data, field_data=mesh.field_data)


    @classmethod
    def load_from_vtk(cls, filepath: str) -> "TriMesh3D":
        """
        Load a mesh from a VTK file.

        The VTK file must contain nodes, elements, and optionally UV mapping coordinates and displacements.

        .. code-block:: python

            from pyblenderSDIC.meshes import TriMesh3D
            # Load the mesh from a VTK file
            trimesh3d = TriMesh3D.load_from_vtk("path/to/mesh.vtk")

        .. seealso::

            - :meth:`pyblenderSDIC.meshes.TriMesh3D.save_to_vtk` for saving the mesh to a VTK file.

        Parameters
        ----------
        filepath : str
            Path to the VTK file to be loaded.

        Returns
        -------
        TriMesh3D
            The loaded mesh object.

        Raises
        ------
        ValueError
            If the filepath is not a string.
        FileNotFoundError
            If the file does not exist.
        """
        if not isinstance(filepath, str):
            raise ValueError("The filepath must be a string.")

        # Read the mesh from the VTK file using meshio
        mesh = meshio.read(filepath)
        
        return cls.from_meshio(mesh)
    

    @classmethod
    def load_from_dict(cls, data: Dict) -> "TriMesh3D":
        """
        Create a TriMesh3D instance from a dictionary.

        The structure of the dictionary should be as provided by the :meth:`pyblenderSDIC.meshes.TriMesh3D.save_to_dict` method.

        .. code-block:: python

            from pyblenderSDIC.meshes import TriMesh3D

            mesh_dict = {
                "type": "TriMesh3D [pyblenderSDIC]",
                "description": "Description of the mesh",
                "nodes": [[0.1, 0.2, 0.1], [0.5, 0.6, 0.4], [0.1, 0.6, 0.2]],
                "elements": [[1], [2], [3]],
                "uvmap": [[0.1, 0.2], [0.5, 0.6], [0.1, 0.6]],
                "displacements_frame_1": [[0.2, 0.3, 0.1], [0.4, 0.5, 0.6], [0.2, 0.5, 0.3]],
                "displacements_frame_3": [[0.3, 0.4, 0.2], [0.5, 0.6, 0.7], [0.3, 0.6, 0.4]],
            }

            # Create a TriMesh3D instance from the dictionary
            trimesh3d = TriMesh3D.load_from_dict(mesh_dict)

        .. seealso::

            - :meth:`pyblenderSDIC.meshes.TriMesh3D.save_to_dict` for saving the mesh to a dictionary.
            - :meth:`pyblenderSDIC.meshes.TriMesh3D.load_from_json` for loading from a JSON file.

        Parameters
        ----------
        data : dict
            A dictionary containing the mesh's data.
        
        Returns
        -------
        TriMesh3D
            The TriMesh3D instance.

        Raises
        ------
        ValueError
            If the data is not a dictionary.
        KeyError
            If required keys are missing from the dictionary.
        """
        # Check for the input type
        if not isinstance(data, dict):
            raise ValueError("data must be a dictionary.")
        
        # Extract the nodes, elements, and uvmap (if available)
        nodes = numpy.array(data["nodes"])
        elements = numpy.array(data["elements"])
        cells = {"triangle": elements}

        # Create the TriMesh3D instance
        mesh = cls(points=nodes, cells=cells)

        # Check for the uvmap and set it if available
        uvmap = data.get("uvmap") if "uvmap" in data else None
        mesh.uvmap = uvmap

        # Load displacements for each frame
        for key, value in data.items():
            if key.startswith("displacements_frame_"):
                frame = int(key.split('_')[-1])  # Extract frame index
                mesh.set_displacements(value, frame)

        return mesh
    

    @classmethod
    def load_from_json(cls, filepath: str) -> TriMesh3D:
        """
        Create a TriMesh3D instance from a JSON file.

        The structure of the JSON file follows the :meth:`pyblenderSDIC.meshes.TriMesh3D.save_to_dict` method.

        .. code-block:: python

            from pyblenderSDIC.meshes import TriMesh3D

            # Load the mesh from a JSON file
            trimesh3d = TriMesh3D.load_from_json("path/to/mesh.json")

        .. seealso::

            - :meth:`pyblenderSDIC.meshes.TriMesh3D.save_to_json` for saving the mesh to a JSON file.
            - :meth:`pyblenderSDIC.meshes.TriMesh3D.load_from_dict` for loading from a dictionary.

        Parameters
        ----------
        filepath : str
            The path to the JSON file.
        
        Returns
        -------
        TriMesh3D
            A TriMesh3D instance.
        
        Raises
        ------
        FileNotFoundError
            If the filepath is not a valid path.
        """
        # Load the dictionary from the JSON file
        with open(filepath, "r") as file:
            data = json.load(file)
        
        # Create the Frame instance
        return cls.load_from_dict(data)
    


    # ============================================================
    # Save methods
    # ============================================================
    def save_to_vtk(self, filepath: str) -> None:
        """
        Export the mesh data to a VTK file.

        The VTK file will include:

        - Nodes and elements of the mesh.
        - UV mapping coordinates, if present, as point data. The UV mapping will be stored in 3D space with a third dimension of zeros
          to ensure compatibility with the VTK format.

        .. note::

            If the filepath does not end with ".vtk", the extension will be added automatically.

        .. seealso::

            - :meth:`pyblenderSDIC.meshes.TriMesh3D.load_from_vtk` for loading a mesh from a VTK file.    

        Parameters
        ----------
        filepath : str
            Path to the VTK file where the mesh will be saved.
        
        Raises
        ------
        ValueError
            If the filepath is not a string.
        """
        if not isinstance(filepath, str):
            raise ValueError("The filepath must be a string.")

        # Ensure the filepath ends with ".vtk"
        if not filepath.lower().endswith(".vtk"):
            filepath += ".vtk"

        self.write(filepath, file_format="vtk")

    
    def save_to_dict(self, description: str = "") -> Dict:
        """
        Export the TriMesh3D's data to a dictionary.

        The structure of the dictionary is as follows:

        .. code-block:: python

            {
                "type": "TriMesh3D [pyblenderSDIC]",
                "description": "Description of the mesh",
                "nodes": [[0.1, 0.2, 0.1], [0.5, 0.6, 0.4], [0.1, 0.6, 0.2]],
                "elements": [[1], [2], [3]],
                "uvmap": [[0.1, 0.2], [0.5, 0.6], [0.1, 0.6]],
                "displacements_frame_1": [[0.2, 0.3, 0.1], [0.4, 0.5, 0.6], [0.2, 0.5, 0.3]],
                "displacements_frame_3": [[0.3, 0.4, 0.2], [0.5, 0.6, 0.7], [0.3, 0.6, 0.4]],
            }

        Parameters
        ----------
        description : str, optional
            A description of the mesh, by default "".

        Returns
        -------
        dict
            A dictionary containing the mesh's data.

        Raises
        ------
        ValueError
            If the description is not a string.
        """
        # Check the description
        if not isinstance(description, str):
            raise ValueError("description must be a string.")

        # Initialize dictionary
        data = {
            "type": "TriMesh3D [pyblenderSDIC]",
            "nodes": self.nodes.tolist(),
            "elements": self.elements.tolist(),
            "uvmap": self.uvmap.tolist() if self.uvmap is not None else None
        }

        # Add displacements for each frame
        defined_frames = self.get_defined_frames()
        for frame in defined_frames:
            data[f"displacements_frame_{frame}"] = self.get_displacements(frame=frame).tolist()

        # Add the description if it's not empty
        if description:
            data["description"] = description
        
        return data


    def save_to_json(self, filepath: str, description: str = "") -> None:
        """
        Export the TriMesh3D's data to a JSON file.

        The structure of the JSON file follows the :meth:`pyblenderSDIC.meshes.TriMesh3D.save_to_dict` method.

        Parameters
        ----------
        filepath : str
            The path to the JSON file.
        
        description : str, optional
            A description of the mesh, by default "".

        Raises
        ------
        FileNotFoundError
            If the filepath is not a valid path.
        """
        # Create the dictionary
        data = self.save_to_dict(description=description)

        # Save the dictionary to a JSON file
        with open(filepath, "w") as file:
            json.dump(data, file, indent=4)


    
    # ===========================================================
    # Properties and setters
    # ===========================================================
    @property
    def nodes(self) -> numpy.ndarray:
        """
        Get or set the reference positions of the mesh nodes (undeformed state at frame 0).

        The node coordinates are stored in the ``points`` attribute of the mesh
        and have shape (N, 3), where N is the number of nodes.

        .. code-block:: python

            # Get the nodes at the reference state
            nodes = trimesh3d.nodes  # shape (N, 3)

            # Set the nodes at the reference state
            trimesh3d.nodes = new_nodes

        .. warning::

            This property uses ``numpy.asarray`` on the internal ``points`` array.
            As a result, any modification to the returned or setted array directly affects the mesh data.
            To avoid unintentional updates, assign a copy instead or use :

            - :meth:`pyblenderSDIC.meshes.TriMesh3D.get_nodes` to get a copy of the nodes.
            - :meth:`pyblenderSDIC.meshes.TriMesh3D.set_nodes` to set a copy of the nodes.

        .. code-block:: python

            # Extract the nodes of the mesh
            nodes = trimesh3d.nodes # shape (N, 3)

            # Update part of the nodes (modifies the original data)
            trimesh3d.nodes[5, 0] = 4.0 # The position of node 5 is modified

            # Equivalent behavior:     
            nodes = trimesh3d.nodes
            nodes[5, 0] = 4.0  # Also modifies the TriMesh3D object

            # To avoid unintentional updates, assign a copy instead:
            trimesh3d.nodes = new_nodes.copy()
            new_nodes[5, 0] = 4.0  # Does NOT modify the TriMesh3D object
            nodes = trimesh3d.nodes.copy()
            nodes[5, 0] = 4.0  # Does NOT modify the TriMesh3D object

        Returns
        -------
        numpy.ndarray
            The node coordinates as a (N, 3) array of float64.
        """
        nodes = numpy.asarray(self.points, dtype=numpy.float64)
        if not nodes.shape == (self.Nnodes, 3):
            raise ValueError(f"[INTERNAL CLASS ERROR] Nodes are not well formatted. Expected shape ({self.Nnodes}, 3).")
        return nodes

    @nodes.setter
    def nodes(self, nodes: numpy.ndarray) -> None:
        nodes = numpy.asarray(nodes, dtype=numpy.float64)
        if not nodes.shape == (self.Nnodes, 3):
            raise ValueError(f"Nodes must have shape ({self.Nnodes}, 3).")
        self.points = nodes


    @property
    def elements(self) -> numpy.ndarray:
        """
        Get or set the mesh elements (triangles).

        The elements are stored in the ``cells_dict['triangle']`` of the mesh and represent the connectivity 
        between nodes in the mesh. They are expected to be triangles, represented as 
        indices into the node array. The array has shape (M, 3), where M is the number of elements.

        .. code-block:: python

            # Get the elements (triangle connectivity)
            elements = trimesh3d.elements  # shape (M, 3)

            # Set the elements (as integer indices)
            trimesh3d.elements = new_elements

        .. warning::

            This property uses ``numpy.asarray`` on the ``cells[0].data`` attribute.
            Any modification to the returned array will directly affect the mesh data.
            To avoid unintentional updates, assign a copy instead or use :

            - :meth:`pyblenderSDIC.meshes.TriMesh3D.get_elements` to get a copy of the elements.
            - :meth:`pyblenderSDIC.meshes.TriMesh3D.set_elements` to set a copy of the elements.

        .. code-block:: python

            # Extract the elements of the mesh
            elements = trimesh3d.elements # shape (M, 3)

            # Update part of the elements (modifies the original data)
            trimesh3d.elements[0, 1] = 42 # The second node of the first element is set to the node with index 42

            # Equivalent behavior:
            elements = trimesh3d.elements
            elements[0, 1] = 42  # Also modifies the TriMesh3D object

            # To avoid unintentional updates, assign a copy instead:
            trimesh3d.elements = new_elements.copy()
            new_elements[0, 1] = 42  # Does NOT modify the TriMesh3D object
            elements = trimesh3d.elements.copy()
            elements[0, 1] = 42  # Does NOT modify the TriMesh3D object

        Returns
        -------
        numpy.ndarray
            The connectivity array of elements as integers of shape (M, 3).
        """
        elements = numpy.asarray(self.cells[0].data, dtype=int)
        if not elements.shape == (self.Nelements, 3):
            raise ValueError(f"[INTERNAL CLASS ERROR] Elements are not well formatted. Expected shape ({self.Nelements}, 3).")
        return elements

    @elements.setter
    def elements(self, elements: numpy.ndarray) -> None:
        elements = numpy.asarray(elements, dtype=int)
        if not elements.shape == (self.Nelements, 3):
            raise ValueError(f"Elements must have shape ({self.Nelements}, 3).")
        self.cells[0].data = elements

    def get_elements(self) -> numpy.ndarray:
        """
        Get the elements of the mesh as a copy.

        .. seealso::

            - :attr:`pyblenderSDIC.meshes.TriMesh3D.elements` for the original elements array.

        Returns
        -------
        numpy.ndarray
            The elements of the mesh as a (M, 3) array of integers.
        """
        return self.elements.copy()
    
    def set_elements(self, elements: numpy.ndarray) -> None:
        """
        Set the elements of the mesh as a copy.

        .. seealso::

            - :attr:`pyblenderSDIC.meshes.TriMesh3D.elements` for the original elements array.

        Parameters
        ----------
        elements : numpy.ndarray
            The new elements to set, as a (M, 3) array of integers.
        """
        self.elements = elements # No direct copy to check the input
        self.cells[0].data = self.cells[0].data.copy()


    @property
    def uvmap(self) -> Optional[numpy.ndarray]:
        """
        Get or set the UV mapping coordinates of the mesh.

        The UV coordinates are stored in the ``point_data`` dictionary under the key "uvmap".
        They must lie within the range [0, 1] and have shape (N, 3), where N is the number of nodes.
        The third column is typically filled with zeros for VTK compatibility.

        .. code-block:: python

            # Get the UV mapping coordinates
            uvmap = trimesh3d.uvmap  # shape (N, 3)

            # Set the UV mapping coordinates
            trimesh3d.uvmap = new_uvmap

        .. warning::

            This property uses ``numpy.asarray`` on the ``point_data["uvmap"]`` attribute.
            Any modification to the returned array directly affects the mesh data.
            To avoid unintentional updates, assign a copy instead or use :

            - :meth:`pyblenderSDIC.meshes.TriMesh3D.get_uvmap2D` to get a copy of the UV mapping.
            - :meth:`pyblenderSDIC.meshes.TriMesh3D.set_uvmap2D` to set a copy of the UV mapping.

        .. code-block:: python

            # Extract the UV mapping coordinates of the mesh
            uvmap = trimesh3d.uvmap # shape (N, 3)

            # Update part of the UV mapping coordinates (modifies the original data)
            trimesh3d.uvmap[2, 0] = 0.5 # The first column of the third node is set to 0.5.

            # Equivalent behavior:
            uvmap = trimesh3d.uvmap
            uvmap[2, 0] = 0.5  # Also modifies the mesh

            # To avoid unintentional updates, assign a copy instead:
            trimesh3d.uvmap = new_uvmap.copy()
            new_uvmap[2, 0] = 0.5  # Does NOT modify the TriMesh3D object
            uvmap = trimesh3d.uvmap.copy()
            uvmap[2, 0] = 0.5  # Does NOT modify the TriMesh3D object

        Returns
        -------
        Optional[numpy.ndarray]
            The UV mapping coordinates as a (N, 3) array of float64.
        """
        if self.point_data is None:
            return None
        if "uvmap" not in self.point_data:
            return None
        uvmap = numpy.asarray(self.point_data["uvmap"], dtype=numpy.float64)
        if not uvmap.shape == (self.Nnodes, 3):
            raise ValueError(f"[INTERNAL CLASS ERROR] UV mapping coordinates are not well formatted. Expected shape ({self.Nnodes}, 3).")
        return uvmap

    @uvmap.setter
    def uvmap(self, uvmap: Optional[numpy.ndarray]) -> None:
        if uvmap is None:
            if self.point_data is not None and "uvmap" in self.point_data:
                del self.point_data["uvmap"]
            return

        uvmap = numpy.asarray(uvmap, dtype=numpy.float64)
        if not ((uvmap >= 0).all() and (uvmap <= 1).all()):
            raise ValueError("UV mapping coordinates must be between 0 and 1.")
        if not uvmap.shape == (self.Nnodes, 3):
            raise ValueError(f"UV mapping coordinates must have shape ({self.Nnodes}, 3).")

        if self.point_data is None:
            self.point_data = {}

        self.point_data["uvmap"] = uvmap

    def get_uvmap(self) -> numpy.ndarray:
        """
        Get the UV mapping coordinates of the mesh as a copy.

        .. seealso::

            - :attr:`pyblenderSDIC.meshes.TriMesh3D.uvmap` for the original UV mapping coordinates.

        Returns
        -------
        numpy.ndarray
            The UV mapping coordinates as a (N, 3) array of float64.
        """
        uvmap = self.uvmap
        if uvmap is not None:
            uvmap = uvmap.copy()
        return uvmap
    
    def set_uvmap(self, uvmap: numpy.ndarray) -> None:
        """
        Set the UV mapping coordinates of the mesh as a copy.

        .. seealso::

            - :attr:`pyblenderSDIC.meshes.TriMesh3D.uvmap` for the original UV mapping coordinates.

        Parameters
        ----------
        uvmap : numpy.ndarray
            The new UV mapping coordinates to set, as a (N, 3) array of float64.
        """
        self.uvmap = uvmap # No direct copy to check the input
        if self.uvmap is not None:
            self.point_data["uvmap"] = self.point_data["uvmap"].copy()


    @property
    def Nnodes(self) -> int:
        """
        Get the number of nodes in the mesh.

        Returns
        -------
        int
            The number of nodes in the mesh.
        """
        return self.points.shape[0]
    

    @property
    def Nelements(self) -> int:
        """
        Get the number of elements in the mesh.

        Returns
        -------
        int
            The number of elements in the mesh.
        """
        return self.cells[0].data.shape[0]
    


    # ============================================================
    # frames management
    # ============================================================
    def _integral_frame(self, frame: int) -> int:
        """
        Convert a frame index to an integer.

        Parameters
        ----------
        frame : int
            Frame index to convert. Must be a non-negative integer (≥ 0).

        Returns
        -------
        int
            The converted frame index.

        Raises
        ------
        TypeError
            If ``frame`` is not an integer.
        ValueError
            If ``frame`` is negative.
        """
        if not isinstance(frame, Integral):
            raise TypeError("frame must be an integer.")
        if frame < 0:
            raise ValueError("frame must be a non-negative integer.")
        return int(frame)

    def get_defined_frames(self) -> Sequence[int]:
        """
        Return a sorted list of frame indices for which displacements are defined in the mesh.

        Displacement arrays are stored in ``point_data`` under keys of the form
        ``displacements_frame_{k}``, where ``k`` is a strictly positive integer (≥ 1).
        This method extracts all such keys and returns the corresponding frame indices.

        Frame 0, which represents the reference state, is excluded by design.

        Returns
        -------
        Sequence[int]
            Sorted list of all frame indices (≥ 1) for which displacements are available.
            Returns an empty list if no displacements are defined.
        """
        if self.point_data is None:
            return []
        return sorted([
            int(k.split("_")[-1])
            for k in self.point_data.keys()
            if k.startswith("displacements_frame_")
        ])


    def is_defined_frame(self, frame: int) -> bool:
        """
        Check whether displacements are defined for a given frame index.

        Frame 0 is always defined (reference state).

        Parameters
        ----------
        frame : int
            Frame index to check. Must be a non-negative integer (≥ 0).

        Returns
        -------
        bool
            True if displacements are defined for the given frame, False otherwise.

        Raises
        ------
        TypeError
            If ``frame`` is not an integer.
        ValueError
            If ``frame`` is negative.
        """
        frame = self._integral_frame(frame)
        if frame == 0:
            return True
        if self.point_data is None:
            return False
        return f"displacements_frame_{frame}" in self.point_data


    def last_defined_frame(self) -> int:
        """
        Return the highest frame index for which displacements are defined.

        If no displacements are defined, the reference frame (0) is returned.

        Returns
        -------
        int
            Highest defined frame index (≥ 0). Returns 0 if no frames are available.
        """
        frames = self.get_defined_frames()
        if len(frames) == 0:
            return 0
        return max(frames)


    def previous_defined_frame(self, frame: int) -> int:
        """
        Return the closest defined frame before a given frame index.

        This method returns the largest defined frame index strictly less than the given one.
        If there is no such frame, it returns 0, which corresponds to the reference state.

        .. note::
            If the input frame is itself defined, it still returns the previous one.

        Parameters
        ----------
        frame : int
            Frame index to check. Must be a non-negative integer (≥ 0).

        Returns
        -------
        int
            Closest defined frame index before the given one (≥ 0). Returns 0 if none exists.

        Raises
        ------
        TypeError
            If ``frame`` is not an integer.
        ValueError
            If ``frame`` is negative.
        """
        frame = self._integral_frame(frame)
        frames = self.get_defined_frames()
        if len(frames) == 0 or frame <= min(frames):
            return 0

        return max([f for f in frames if f < frame])

    
    def next_defined_frame(self, frame: int) -> Optional[int]:
        """
        Return the closest defined frame after a given frame index.

        This method returns the smallest defined frame index strictly greater than the input.
        If no such frame exists, returns None.

        .. note::
            If the input frame is defined, it still returns the next one.

        Parameters
        ----------
        frame : int
            Frame index to check. Must be a non-negative integer (≥ 0).

        Returns
        -------
        Optional[int]
            Closest defined frame index after the given one (≥ 0), or None if none exists.

        Raises
        ------
        TypeError
            If ``frame`` is not an integer.
        ValueError
            If ``frame`` is negative.
        """
        frame = self._integral_frame(frame)
        frames = self.get_defined_frames()
        if len(frames) == 0 or frame >= max(frames):
            return None

        return min([f for f in frames if f > frame])


    # ============================================================
    # Special get and set methods
    # ============================================================
    def get_uvmap2D(self) -> Optional[numpy.ndarray]:
        """
        Get the 2D UV mapping coordinates of the mesh.

        This returns a copy of the first two columns of the full 3D UV map.
        The values must lie within the range [0, 1], and the shape is (N, 2),
        where N is the number of nodes.

        .. code-block:: python

            # Get the 2D UV mapping coordinates
            uvmap2D = trimesh3d.get_uvmap2D()  # shape (N, 2)

        .. warning::

            This method returns a **copy** of the internal UV data.
            Any modification to the returned array will **not** affect the mesh.

            .. code-block:: python

                uvmap2D = trimesh3d.get_uvmap2D() # shape (N, 2)
                uvmap2D[:, 0] = 0.5  # Does NOT modify the mesh

        .. seealso::

            - :meth:`set_uvmap2D` to set the 2D UV mapping coordinates.
            - :attr:`uvmap` to access the full 3D UV mapping coordinates.

        Returns
        -------
        Optional[numpy.ndarray]
            A copy of the 2D UV coordinates as a (N, 2) array of float64.
        """
        if self.uvmap is None:
            return None
        return self.uvmap[:, :2].copy()
    

    def set_uvmap2D(self, uvmap2D: numpy.ndarray) -> None:
        """
        Set the 2D UV mapping coordinates of the mesh.

        The input array must have shape (N, 2), where N is the number of nodes,
        and values should lie within the range [0, 1].

        Internally, this method builds a 3D UV array with a third column of zeros
        for compatibility with the full UV map format used in VTK.

        .. code-block:: python

            # Set the 2D UV mapping coordinates
            trimesh3d.set_uvmap2D(new_uvmap2D)

        .. warning::

            A **copy** of the input array is made during the assignment.
            Therefore, any modification of the original array after setting
            will **not** affect the mesh.

            .. code-block:: python

                trimesh3d.set_uvmap2D(new_uvmap2D)
                new_uvmap2D[:, 0] = 0.2  # Does NOT modify the mesh

        .. seealso::

            - :meth:`get_uvmap2D` to retrieve the 2D UV mapping coordinates.
            - :attr:`uvmap` to access the full 3D UV mapping coordinates.

        Parameters
        ----------
        new_uvmap2D : numpy.ndarray
            A (N, 2) array of float64 representing UV mapping coordinates in [0, 1].
        """
        if uvmap2D is None:
            self.uvmap = None
            return

        uvmap2D = numpy.asarray(uvmap2D, dtype=numpy.float64)
        if not ((uvmap2D >= 0).all() and (uvmap2D <= 1).all()):
            raise ValueError("UV mapping coordinates must be between 0 and 1.")
        if not uvmap2D.shape == (self.Nnodes, 2):
            raise ValueError(f"UV mapping coordinates must have shape ({self.Nnodes}, 2).")

        # Add a third component of zeros for compatibility with VTK
        self.uvmap = numpy.hstack((uvmap2D, numpy.zeros((uvmap2D.shape[0], 1), dtype=numpy.float64)))


    def get_nodes(self, frame: int) -> numpy.ndarray:
        """
        Get the mesh nodes at a given frame.

        This method returns the deformed node positions by adding the displacements
        at the specified frame to the reference node coordinates (stored in ``nodes``).
        The returned array has shape (N, 3), where N is the number of nodes.

        .. code-block:: python

            # Get the deformed nodes at frame 3
            nodes_3 = mesh.get_nodes(frame=3)

            # Equivalent to:
            nodes_3 = mesh.nodes + mesh.displacements[3]

        .. warning::

            This method returns a **new array** created from the sum of the reference nodes
            and displacements. Modifying the returned array **does NOT affect** the internal
            state of the mesh.

            .. code-block:: python

                nodes = mesh.get_nodes(frame=0)
                nodes[:, 0] = 100.0  # Does NOT modify mesh.nodes or mesh displacements

        .. seealso::

            - :meth:`get_displacements` to access the displacements at a given frame.
            - :meth:`set_nodes` to set deformed node positions at a specific frame.
            - :meth:`interpolate_nodes` to interpolate node positions at a non-defined frame.

        Parameters
        ----------
        frame : int
            The frame index for which to retrieve deformed node positions.

        Returns
        -------
        numpy.ndarray
            A (N, 3) array of float64 representing the node coordinates at the specified frame.
        """
        return self.nodes + self.displacements[frame]


    def get_displacements(self, frame: int) -> numpy.ndarray:
        """
        Get the displacements of the mesh nodes at a given frame.

        This method returns a copy of the displacements stored for the specified frame.
        The returned array has shape (N, 3), where N is the number of nodes.

        .. code-block:: python

            # Get the displacements at frame 2
            disp = mesh.get_displacements(frame=2)

            # Equivalent to:
            disp = mesh.displacements[2].copy()

        .. warning::

            This method returns a **copy** of the displacement array.
            As a result, modifying the returned array will **not** affect the internal data.

            .. code-block:: python

                disp = mesh.get_displacements(frame=0)
                disp[:, 1] = 999.0  # Does NOT modify mesh.displacements

        .. seealso::

            - :meth:`get_nodes` to get the deformed node positions.
            - :meth:`set_displacements` to update the displacements at a given frame.
            - :meth:`interpolate_displacements` to interpolate displacements at a non-defined frame.

        Parameters
        ----------
        frame : int
            The frame index for which to retrieve displacements.

        Returns
        -------
        numpy.ndarray
            A (N, 3) array of float64 representing the displacements at the specified frame.
        """
        return self.displacements[frame].copy()


    def set_nodes(self, nodes: Optional[numpy.ndarray], frame: int = 0) -> None:
        """
        Set the node positions of the mesh at a given frame, or remove displacements.

        If ``frame == 0``, the input array replaces the reference node coordinates.
        For other frames:
        - If ``nodes`` is provided, displacements are computed and stored.
        - If ``nodes`` is None, the displacement for that frame is removed.

        The input array is copied internally. Further modification to it will **not**
        affect the internal mesh data.

        .. code-block:: python

            # Set reference geometry
            mesh.set_nodes(new_nodes, frame=0)

            # Set deformed nodes (stored as displacements)
            mesh.set_nodes(deformed_nodes, frame=3)

            # Remove displacements for frame 3
            mesh.set_nodes(None, frame=3)

        .. warning::

            - Cannot delete reference nodes (``frame=0`` and ``nodes=None`` raises an error).
            - Arrays are copied; modifying them after setting has no effect on the mesh.

        .. seealso::

            - :meth:`get_nodes` to get the deformed node positions.
            - :meth:`interpolate_nodes` to interpolate node positions at a non-defined frame.

        Parameters
        ----------
        nodes : Optional[numpy.ndarray]
            A (N, 3) array of float64 representing node coordinates, or ``None`` to delete displacements.

        frame : int, optional
            The frame index to assign (default is 0).

        Raises
        ------
        ValueError
            If ``nodes`` is None and ``frame == 0``.
            If the given nodes don't have the rigth shape
        """
        if nodes is None:
            self.displacements[frame] = None
            return
        
        nodes = numpy.asarray(nodes, dtype=numpy.float64)
        if not nodes.shape == (self.Nnodes, 3):
            raise ValueError(f"Nodes must have shape ({self.Nnodes}, 3).")

        self.displacements[frame] = nodes - self.nodes


    def set_displacements(self, displacements: Optional[numpy.ndarray], frame: int) -> None:
        """
        Set or remove the displacements of the mesh at a given frame.

        If ``displacements`` is provided, it is copied and stored for the given frame.
        If ``displacements`` is None, the displacement data for that frame is removed.

        .. code-block:: python

            # Set displacements
            mesh.set_displacements(disp, frame=2)

            # Remove displacements
            mesh.set_displacements(None, frame=2)

        .. warning::

            - Cannot delete reference nodes (``frame=0`` and ``displacements=None`` raises an error).
            - Arrays are copied; modifying them after setting has no effect on the mesh.

        .. seealso::

            - :meth:`get_displacements` to get displacements at a defined frame.
            - :meth:`set_nodes` to get the deformed node positions.
            - :meth:`interpolate_displacements` to interpolate displacements at a non-defined frame.

        Parameters
        ----------
        displacements : Optional[numpy.ndarray]
            A (N, 3) array of float64, or ``None`` to delete the displacement data.

        frame : int
            Frame index to assign or delete.

        Raises
        ------
        ValueError
            If ``displacements`` is None and ``frame == 0``.
            If the given displacements don't have the rigth shape
        """
        if displacements is None:
            self.displacements[frame] = None
            return
        
        displacements = numpy.asarray(displacements, dtype=numpy.float64)
        if not displacements.shape == (self.Nnodes, 3):
            raise ValueError(f"Displacements must have shape ({self.Nnodes}, 3).")
    
        self.displacements[frame] = displacements.copy()


    def interpolate_displacements(self, frame: int, interpolation: Optional[str] = None) -> numpy.ndarray:
        """
        Interpolate displacements at an undefined frame using a specified interpolation method.

        This method allows you to obtain displacements at any frame—even if they were not explicitly defined—
        using one of several interpolation strategies. The result is a new array and does **not** modify the mesh.

        .. code-block:: python

            disp = mesh.interpolate_displacements(frame=5, interpolation="Linear")
            nodes = mesh.nodes + disp  # Get interpolated deformed nodes

        Supported interpolation methods:

        +------------------------+--------------------------------------------------------------------------------------------------------------------------+
        | Interpolation Method   | Description                                                                                                              |
        +========================+==========================================================================================================================+
        | None                   | No interpolation: an error is raised if the frame is not explicitly defined.                                             |
        +------------------------+--------------------------------------------------------------------------------------------------------------------------+
        | "Previous"             | Use the last defined frame before the requested frame. Returns reference (0) if no frame is defined before.              |
        +------------------------+--------------------------------------------------------------------------------------------------------------------------+
        | "Next"                 | Use the first defined frame after the requested frame. If no next frame, return the last defined frame.                  |
        +------------------------+--------------------------------------------------------------------------------------------------------------------------+
        | "Linear"               | Linear interpolation between the two closest defined frames. After last frame, returns last value.                       |
        +------------------------+--------------------------------------------------------------------------------------------------------------------------+

        .. warning::

            The returned array is a **copy**. Modifying it will not alter the internal state of the mesh.

        .. seealso::

            - :meth:`get_displacements` to get displacements at a defined frame.    
            - :meth:`set_displacements` to set displacements at a specific frame.
            - :meth:`interpolate_nodes` to get interpolated node positions at a non-defined frame.

        Parameters
        ----------
        frame : int
            The target frame index at which displacements are requested.

        interpolation : {"Previous", "Next", "Linear", None}, optional
            The interpolation method to use. Default is None.

        Returns
        -------
        numpy.ndarray
            A (N, 3) array of float64 representing the interpolated displacements.

        Raises
        ------
        ValueError
            If the interpolation method is None and the frame is undefined.
            If "Linear" is selected but there are not enough frames to interpolate.
        """
        frame = self._integral_frame(frame)

        if self.is_defined_frame(frame):
            return self.get_displacements(frame).copy()

        if interpolation is None:
            raise ValueError(f"No displacements for frame {frame} and interpolation is None.")

        if interpolation == "Previous":
            prevf = self.previous_defined_frame(frame)
            return self.get_displacements(prevf).copy()

        elif interpolation == "Next":
            nextf = self.next_defined_frame(frame)
            # If no next frame, use the last defined frame
            if nextf is None:
                lastf = self.last_defined_frame()
                return self.get_displacements(lastf).copy()
            
            return self.get_displacements(nextf).copy()

        elif interpolation == "Linear":
            prevf = self.previous_defined_frame(frame)
            nextf = self.next_defined_frame(frame)
            # If no next frame, use the last defined frame
            if nextf is None:
                lastf = self.last_defined_frame()
                return self.get_displacements(lastf).copy()
            
            disp_prev = self.get_displacements(prevf)
            disp_next = self.get_displacements(nextf)
            # Linear interpolation
            alpha = (frame - prevf) / (nextf - prevf)
            return disp_prev + alpha * (disp_next - disp_prev)

        raise ValueError(f"Invalid interpolation method: {interpolation!r}")


    def interpolate_nodes(self, frame: int, interpolation: Optional[str] = None) -> numpy.ndarray:
        """
        Compute the node positions at an arbitrary frame using displacement interpolation.

        This method adds the interpolated displacements to the reference nodes to obtain
        the deformed geometry at a given frame. It allows access to intermediate frames 
        not explicitly defined in the data.

        .. code-block:: python

            deformed_nodes = mesh.interpolate_nodes(frame=5, interpolation="Linear")

        The interpolation method behaves as in :meth:`interpolate_displacements`:

        +------------------------+--------------------------------------------------------------------------------------------------------------------------+
        | Interpolation Method   | Description                                                                                                              |
        +========================+==========================================================================================================================+
        | None                   | No interpolation: raises an error if the frame is not defined.                                                           |
        +------------------------+--------------------------------------------------------------------------------------------------------------------------+
        | "Previous"             | Uses displacements from the last defined frame before the requested one.                                                 |
        +------------------------+--------------------------------------------------------------------------------------------------------------------------+
        | "Next"                 | Uses displacements from the next defined frame. If none exist, uses the last defined.                                    |
        +------------------------+--------------------------------------------------------------------------------------------------------------------------+
        | "Linear"               | Performs linear interpolation between the closest surrounding frames. Falls back to the last value if no next frame.     |
        +------------------------+--------------------------------------------------------------------------------------------------------------------------+

        .. warning::

            The returned array is a **copy**. Modifying it will not affect the mesh.

        .. seealso::

            - :meth:`get_nodes` to get the deformed node positions at a defined frame.
            - :meth:`set_nodes` to set deformed node positions at a specific frame.
            - :meth:`interpolate_displacements` to get interpolated displacements at a non-defined frame.

        Parameters
        ----------
        frame : int
            Frame index for which node positions are requested (must be ≥ 0).

        interpolation : {"Previous", "Next", "Linear", None}, optional
            Interpolation strategy to use if the frame is not explicitly defined. Default is None.

        Returns
        -------
        numpy.ndarray
            A (N, 3) array of float64 with interpolated node positions.

        Raises
        ------
        ValueError
            If the interpolation method is None and the frame is undefined.
            If "Linear" is selected but not enough data is available to interpolate.
        """
        return self.nodes + self.interpolate_displacements(frame, interpolation)


    # ============================================================
    # Validation methods
    # ============================================================
    def validate(self) -> None:
        """
        Validate the mesh.

        The method checks if the nodes, elements, UV mapping coordinates, 
        and displacements have the correct shape and consistency.
        """
        # Shape validation for nodes
        if self.points.shape != (self.Nnodes, 3):
            raise ValueError("Nodes must have shape (N, 3).")
        
        # Shape validation for elements
        if self.cells[0].data.shape != (self.Nelements, 3):
            raise ValueError("Elements must have shape (M, 3).")
        
        # Checking UV mapping if it exists
        if self.point_data is not None and "uvmap" in self.point_data:
            if self.point_data["uvmap"].shape != (self.Nnodes, 3):
                raise ValueError("UV mapping coordinates must have shape (N, 3).")
            if not numpy.all(numpy.isfinite(self.point_data["uvmap"])):
                raise ValueError("UV mapping coordinates must be finite.")
            if not ((self.point_data["uvmap"] >= 0).all() and (self.point_data["uvmap"] <= 1).all()):
                raise ValueError("UV mapping coordinates must be between 0 and 1.")

        # Checking if the nodes are finite
        if not numpy.all(numpy.isfinite(self.points)):
            raise ValueError("Nodes must be finite.")
        
        # Checking if the displacements are valid
        if self.point_data is not None:
            for key, disp in self.point_data.items():
                if key.startswith("displacements_frame_"):
                    if disp.shape != (self.Nnodes, 3):
                        raise ValueError(f"Displacements in {key} must have shape ({self.Nnodes}, 3).")
                    if not numpy.all(numpy.isfinite(disp)):
                        raise ValueError(f"Displacements in {key} must be finite.")
        
        # Checking if the elements are positive integers from 0 to Nnodes - 1
        if not numpy.all(self.cells[0].data < self.Nnodes) or not numpy.all(self.cells[0].data >= 0):
            raise ValueError("Elements must be positive integers from 0 to Nnodes - 1.")
        
        # Verifying that displacements and elements are consistent with the nodes
        for key, disp in self.point_data.items():
            if key.startswith("displacements_frame_"):
                # Ensure that the displacement corresponds to the valid node indices
                if not numpy.all(disp[:, 0] >= 0) or not numpy.all(disp[:, 0] < self.Nnodes):
                    raise ValueError(f"Displacement data in {key} must reference valid node indices between 0 and {self.Nnodes - 1}.")
                

    # ============================================================
    # Extracting sub meshes at a specified frame and visualization
    # ============================================================
    def construct_meshio_mesh(self, frame: int = 0) -> meshio.Mesh:
        """
        Construct a meshio mesh from the TriMesh3D object at a specified frame.

        This method creates a new TriMesh3D object using the nodes and elements
        of the current mesh. The mesh is deformed to the specified frame.

        .. note::

            UVMAP are included in the point data of the meshio object but not the displacements.

        .. warning::

            The nex meshio object is not a copy of the original one.
            It will be a new object with the same data but not the same reference.

        Parameters
        ----------
        frame : int, optional
            The frame index for which to construct the TriMesh3D object. Default is 0 (reference state).

        Returns
        -------
        TriMesh3D
            A new TriMesh3D object with the deformed nodes and elements, also including uvmap.
        """
        deformed_nodes = self.get_nodes(frame=frame)
        elements = self.get_elements()
        uvmap = self.get_uvmap()

        # Create a meshio mesh object
        points = deformed_nodes
        cells = {'triangle': elements}
        point_data = {'uvmap': uvmap} if uvmap is not None else None
        mesh = meshio.Mesh(points=points, cells=cells, point_data=point_data)

        return mesh


    def construct_open3d_mesh(self, frame: int = 0) -> open3d.t.geometry.TriangleMesh:
        """
        Construct an Open3D mesh from the TriMesh3D object.

        This method creates an Open3D mesh object using the nodes and elements
        of the TriMesh3D object. The mesh is deformed to the specified frame.

        Parameters
        ----------
        frame : int, optional
            The frame index for which to construct the mesh. Default is 0 (reference state).

        Returns
        -------
        open3d.t.geometry.TriangleMesh
            An Open3D triangle mesh object.
        """
        # Get the deformed nodes for the specified frame
        deformed_nodes = self.get_nodes(frame=frame)

        # Create Open3D mesh
        o3d_mesh = open3d.t.geometry.TriangleMesh()
        o3d_mesh.vertex.positions = open3d.core.Tensor(deformed_nodes, open3d.core.float32)
        o3d_mesh.triangle.indices = open3d.core.Tensor(self.elements, open3d.core.int32)

        return o3d_mesh
    

    def show(
        self, frame: int = 0, *,
        element_highlighted: Union[Integral, Sequence[Integral]] = None,
        intersect_points: Optional[IntersectPoints] = None,
        ) -> None:
        """
        Visualize the mesh using Open3D.

        This method displays the 3D mesh at the given frame using Open3D's interactive viewer.
        Optionally, it can highlight specific mesh elements (triangles) in blue, and show 3D
        intersection points in red.

        .. code-block:: python

            # Show mesh with highlighted elements and intersection points
            trimesh3d.open3d_show(
                frame=0,
                element_highlighted=[42, 77],
                intersect_points=intersect
            )

        .. note::

            - Highlighted triangles are shown in ligth blue.
            - Intersection points are visualized as **red**.
            - The mesh is shown in its deformed configuration at the given frame.

        Parameters
        ----------
        frame : int, optional
            The frame index for which to display the mesh. Default is 0.

        element_highlighted : int or sequence of int, optional
            Indices of mesh elements (triangles) to color in blue.

        intersect_points : IntersectPoints, optional
            3D intersection points to show in red. Only valid entries are displayed.
        """
        mesh = self.construct_open3d_mesh(frame)

        # Extracted the elements to be colored
        if element_highlighted is None:
            element_highlighted = []
        elif isinstance(element_highlighted, Integral):
            element_highlighted = [element_highlighted]
        element_highlighted = numpy.asarray(element_highlighted, dtype=int).reshape(-1)
        element_highlighted = numpy.unique(element_highlighted)

        if not numpy.all(0 <= element_highlighted < self.Nelements):
            raise ValueError("element_highlighted must be valid element indices.")
        
        indices = numpy.arange(self.Nelements)
        colors = numpy.full((self.Nelements, 3), [0.5, 0.5, 0.5]) # Default color for elements (gray)
        colors[numpy.isin(indices, element_highlighted)] = [0.0, 0.5, 0.5]  # Light blue for highlighted elements
        mesh.triangle.colors = open3d.core.Tensor(colors, open3d.core.float32)

        # Extract the edges of the mesh
        triangles = numpy.asarray(mesh.triangle.indices.numpy(), dtype=int)
        lines = numpy.zeros((3*self.Nelements, 2), dtype=int)
        lines[0::3, 0] = triangles[:, 0]
        lines[0::3, 1] = triangles[:, 1]
        lines[1::3, 0] = triangles[:, 1]
        lines[1::3, 1] = triangles[:, 2]
        lines[2::3, 0] = triangles[:, 2]
        lines[2::3, 1] = triangles[:, 0]

        # Create Open3D LineSet for edges
        lineset = open3d.t.geometry.LineSet()
        lineset.line.indices = open3d.core.Tensor(lines, open3d.core.int32)
        lineset.point.positions = mesh.vertex.positions
        lineset.line.colors = open3d.core.Tensor([0.2, 0.2, 0.2], open3d.core.float32)

        # Create PointCloud for intersection points
        if intersect_points is not None:
            points = self.get_intersect_points_coordinates(intersect_points, frame=frame)
            point_cloud = open3d.t.geometry.PointCloud()
            point_cloud.point.positions = open3d.core.Tensor(points, dtype=open3d.core.Dtype.Float32)
            point_cloud.point.colors = open3d.core.Tensor(numpy.tile([0.0, 0.0, 1.0], (points.shape[0], 1)), dtype=open3d.core.Dtype.Float32)  # Blue color for points

        # Create Open3D geometries
        geometries = [{
            "geometry": mesh,
            "name": "Colored Elements"
        },
        {
            "geometry": lineset,
            "name": "Mesh Edges"
        },
        {
            "geometry": point_cloud,
            "name": "Intersection Points"
        }]

        # Launch Open3D viewer
        open3d.visualization.draw(geometries, point_size=15)
        
    # ============================================================
    # Rays and intersections
    # ============================================================
    def open3d_cast_ray(self, rays: numpy.ndarray, frame: int = 0) -> Dict:
        """
        Calculate the intersection of rays with a given mesh using Open3D.

        This method uses Open3D's raycasting capabilities to find the intersection points
        of rays with the mesh.

        .. code-block:: python

            # Define ray origins and directions
            rays_origins = numpy.array([[0, 0, 0], [1, 1, 1]]) # shape (L, 3)
            rays_directions = numpy.array([[0, 0, 1], [1, 1, 0]]) # shape (L, 3)
            rays = numpy.hstack((rays_origins, rays_directions)) # shape (L, 6)

            # Perform ray-mesh intersection
            ray_intersect = trimesh3d.open3d_cast_ray(rays, frame=0)

        .. seealso::

            Documentation of Open3D's raycasting : 
            https://www.open3d.org/html/python_api/open3d.t.geometry.RaycastingScene.html#open3d.t.geometry.RaycastingScene.cast_rays

        Parameters
        ----------
        rays: numpy.ndarray
            A (..., 6) array of float32. Each component contains the position and the direction of a ray in the format [x0, y0, z0, dx, dy, dz].

        frame : int, optional
            The frame index for which to compute the intersection. Default is 0.
            The mesh will be deformed to this frame before performing the intersection.

        Returns
        -------
        ray_intersect : Dict
            The output of the raycasting operation by Open3D. 
        """
        # Extract the Open3D mesh for the specified frame
        o3d_mesh = self.construct_open3d_mesh(frame=frame)

        # Convert rays_origins and rays_directions to numpy arrays
        rays = numpy.asarray(rays, dtype=numpy.float32)
        if rays.shape[-1] != 6:
            raise ValueError("Rays must have shape (..., 6).")

        # Convert numpy arrays to Open3D point clouds (ray origins and directions)
        rays_o3d = open3d.core.Tensor(rays, open3d.core.float32)  # Shape: (..., 6)

        # Create the scene and add the mesh
        raycaster = open3d.t.geometry.RaycastingScene()
        raycaster.add_triangles(o3d_mesh)

        return raycaster.cast_rays(rays_o3d)


    def compute_intersect_points(self, rays: numpy.ndarray, frame: int = 0) -> IntersectPoints:
        """
        Compute the intersection of rays with the mesh.

        This method uses Open3D to perform ray-mesh intersection and returns the intersection points
        and the corresponding triangle indices as an `IntersectPoints` object.

        .. code-block:: python

            # Define ray origins and directions
            rays_origins = numpy.array([[0, 0, 0], [1, 1, 1]]) # shape (L, 3)
            rays_directions = numpy.array([[0, 0, 1], [1, 1, 0]]) # shape (L, 3)
            rays = numpy.hstack((rays_origins, rays_directions)) # shape (L, 6)

            # Perform ray-mesh intersection
            intersect_points = trimesh3d.compute_intersect_points(rays, frame=0)

        .. note::

            The returned :class:`IntersectPoints` contains:

            - ``uv``: A (..., 2) array of barycentric coordinates (u, v). If a ray misses the mesh, the entry is [nan, nan].
            - ``id``: A (...) array of triangle indices hit by each ray. If a ray misses, the index is set to -1.

            The barycentric coordinates are such that:

            .. code-block:: python

                A = nodes[element[0], :]
                B = nodes[element[1], :]
                C = nodes[element[2], :]

                coordinates = (1 - u - v) * A + u * B + v * C

        .. seealso::

            - :meth:`open3d_cast_ray` for the underlying Open3D implementation.
            - :class:`IntersectPoints` for the output type.

        Parameters
        ----------
        rays : numpy.ndarray
            An array of shape (..., 6) containing the ray origins and directions, in the form
            [x0, y0, z0, dx, dy, dz].

        frame : int, optional
            The frame index for which to compute the intersection. Default is 0.
            The mesh is deformed to this frame before casting rays.

        Returns
        -------
        IntersectPoints
            An object containing barycentric coordinates and triangle indices of the intersections.
        """
        # Perform ray-mesh intersection using Open3D
        results = self.open3d_cast_ray(rays, frame=frame)

        # Initialize the output arrays
        rays = numpy.asarray(rays, dtype=numpy.float64)
        barycentric_coords = numpy.full((*rays.shape[:-1], 2), numpy.nan, dtype=numpy.float64)
        element_indices = numpy.full(*rays.shape[:-1], -1, dtype=int)

        # Extract the intersection points
        intersect_true = results["t_hit"].isfinite().numpy()
        barycentric_coords[intersect_true] = results["primitive_uvs"].numpy().astype(numpy.float64)[intersect_true]
        element_indices[intersect_true] = results["primitive_ids"].numpy().astype(int)[intersect_true]

        # Construct the output
        intersect_points = IntersectPoints(barycentric_coords, element_indices)

        return intersect_points
    

    def get_intersect_points_coordinates(self, intersect_points: IntersectPoints, frame: int = 0) -> numpy.ndarray:
        """
        Compute the 3D coordinates of intersection points from barycentric data.

        This method reconstructs the 3D position of the intersection points using the barycentric
        coordinates and the triangle indices contained in the given :class:`IntersectPoints` object.

        .. code-block:: python

            intersect_points = trimesh3d.compute_intersect_points(rays)
            coords = trimesh3d.get_coordinates(intersect_points, frame=1)

        .. note::

            This method expects barycentric coordinates (u, v), with:

            - u: weight for vertex B
            - v: weight for vertex C
            - w = 1 - u - v: weight for vertex A

            The corresponding 3D coordinates are computed as:

            .. code-block:: python

                coordinates = w * A + u * B + v * C

            Where A, B, and C are the vertices of the intersected triangle in the given frame.

        Parameters
        ----------
        intersect_points : IntersectPoints
            An object containing barycentric coordinates and triangle indices.

        frame : int, optional
            The frame index to retrieve the deformed geometry. Default is 0.

        Returns
        -------
        numpy.ndarray
            An array of shape (..., 3) containing the 3D coordinates of the intersection points.
            Points with no intersection are returned as [nan, nan, nan].
        """
        # Check the shape of the intersect_points
        if not isinstance(intersect_points, IntersectPoints):
            raise ValueError("intersect_points must be an instance of IntersectPoints.")
        
        # Flatten everything to 1D
        flat_bary = intersect_points.uv.reshape(-1, 2) # Alias of intersect_points.barycentric_coordinates
        flat_idx = intersect_points.id.reshape(-1) # Alias of intersect_points.element_indices

        # Initialize output
        flat_points = numpy.full((flat_idx.shape[0], 3), numpy.nan, dtype=numpy.float64)

        # Filter valid hits
        valid = flat_idx >= 0
        valid_idx = flat_idx[valid]
        u = flat_bary[valid, 0]
        v = flat_bary[valid, 1]
        w = 1.0 - u - v

        # Get triangle vertices
        deformed_nodes = self.get_nodes(frame=frame)
        A = deformed_nodes[self.elements[valid_idx, 0]]
        B = deformed_nodes[self.elements[valid_idx, 1]]
        C = deformed_nodes[self.elements[valid_idx, 2]]

        # Compute coordinates
        flat_points[valid] = w[:, None] * A + u[:, None] * B + v[:, None] * C

        # Reshape to original shape
        output_shape = (*intersect_points.uv.shape[:-1], 3)
        return flat_points.reshape(output_shape)
    

    def get_element_normals(self, frame: int = 0) -> numpy.ndarray:
        """
        Compute the normals of the elements in the mesh.

        This method computes the normals of the elements in the mesh using the
        cross product of two edges of each triangle. The normals are normalized
        to unit length.

        .. code-block:: python

            # Get the normals of the elements at frame 0
            normals = trimesh3d.get_elements_normals(frame=0)

        .. seealso::

            :meth:`get_node_normals` for computing normals of the nodes.

        Parameters
        ----------
        frame : int, optional
            The frame index for which to compute the normals. Default is 0.
            The mesh will be deformed to this frame before computing the normals.

        Returns
        -------
        numpy.ndarray
            A (M, 3) array of float64 containing the normals of the elements.
        """
        # Get the deformed nodes for the specified frame
        deformed_nodes = self.get_nodes(frame=frame)

        # Compute the vectors for each triangle
        A = deformed_nodes[self.elements[:, 0]]
        B = deformed_nodes[self.elements[:, 1]]
        C = deformed_nodes[self.elements[:, 2]]

        # Compute two edges of each triangle
        AB = B - A
        AC = C - A

        # Compute the cross product to get the normal vector
        normals = numpy.cross(AB, AC)

        # Normalize the normals to unit length
        norms = numpy.linalg.norm(normals, axis=1)
        norms[norms == 0] = 1e-10

        normals /= norms[:, None]
        return normals
    

    def get_element_centroids(self, frame: int = 0) -> numpy.ndarray:
        """
        Compute the centroids of the elements in the mesh.

        This method computes the centroids of the elements in the mesh by averaging
        the coordinates of their vertices.

        .. code-block:: python

            # Get the centroids of the elements at frame 0
            centroids = trimesh3d.get_elements_centroids(frame=0)

        Parameters
        ----------
        frame : int, optional
            The frame index for which to compute the centroids. Default is 0.
            The mesh will be deformed to this frame before computing the centroids.

        Returns
        -------
        numpy.ndarray
            A (M, 3) array of float64 containing the centroids of the elements.
        """
        # Get the deformed nodes for the specified frame
        deformed_nodes = self.get_nodes(frame=frame)

        # Compute the centroids by averaging the vertices of each triangle
        A = deformed_nodes[self.elements[:, 0]]
        B = deformed_nodes[self.elements[:, 1]]
        C = deformed_nodes[self.elements[:, 2]]

        return (A + B + C) / 3.0
    



