from __future__ import annotations

import numpy
import meshio
import open3d
import json
from typing import Optional, Dict, Union, Sequence
from numbers import Integral

from .intersect_points import IntersectPoints


class TriMesh3D(meshio.Mesh):
    r"""
    Represents a triangular 3D mesh with support for UV mapping and compatibility with the VTK format via `meshio`.

    This class is a subclass of `meshio.Mesh` and is designed to handle triangular surface meshes
    in 3D space. It includes support for texture mapping (UV coordinates) and visualization tools.

    - Only triangular meshes are supported.
    - UV coordinates should lie between 0 and 1.

    .. warning::

        The number of nodes and elements are not designed to change after the mesh is created !

    Mesh Structure
    --------------

    - ``points`` (alias ``nodes``): A NumPy array of shape (N, 3) representing the coordinates of N mesh nodes.
    - ``cells_dict{"triangle"}`` (alias ``elements``): A NumPy array of shape (M, 3) representing M triangular elements defined by node indices.
    - ``point_data``: A dictionary storing data associated with each point, such as UV coordinates.

    .. code-block:: python

        mesh.nodes         # numpy.ndarray of shape (N, 3)
        mesh.elements      # numpy.ndarray of shape (M, 3)
    
    .. figure:: ../../../pyblenderSDIC/resources/doc/demi_cylinder_mesh.png
        :width: 400
        :align: center

        Nodes and elements of a triangular mesh.

    UV Mapping
    ----------

    UV coordinates can be used to apply textures on the mesh surface. They are stored in the point data 
    under the key ``uvmap``, with shape (N, 3). Only the first two dimensions are used (U, V). The third 
    dimension (Z) is filled with default values for VTK compatibility.

    .. code-block:: python

        mesh.uvmap         # numpy.ndarray of shape (N, 3)
        mesh.get_uvmap2D() # numpy.ndarray of shape (N, 2) removing the Z dimension
        mesh.set_uvmap2D(uvmap2d)  # Set UV 2D map

    The UV coordinates represents the position of each node in the normalized texture space, where (0, 0) is the bottom-left corner and (1, 1) is the top-right corner.
    By default, UV coordinates follow the **OpenGL convention**, which is also used by Blender and most 3D engines.

    In this convention:

    - :math:`uv = (0, 0)` corresponds to the bottom-left corner of the texture.
    - :math:`uv = (1, 0)` corresponds to the bottom-right corner of the texture.
    - :math:`uv = (0, 1)` corresponds to the top-left corner of the texture.
    - :math:`uv = (1, 1)` corresponds to the top-right corner of the texture.

    Instantiation
    -------------

    It is recommended to define only the ``points`` and ``cells`` when creating the mesh. You can later assign UV coordinates.

    Parameters
    -----------

    points : array_like
        Node coordinates of the mesh, shape (N, 3).
    
    cells : array_like
        Triangle definitions, shape (M, 3), using zero-based node indices.
    
    point_data : dict, optional
        Point data dictionary (e.g. UV maps, ...).

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

    Visualize the mesh using Open3D:

    >>> mesh.visualize()
    """
    def __init__(self, points, cells, point_data = None, cell_data = None, field_data = None, **kwargs):
        super().__init__(points, cells, point_data, cell_data, field_data, **kwargs)
        if not 'triangle' in self.cells_dict:
            raise ValueError("Only triangular meshes are supported. Please provide a mesh with 'triangle' elements.")
        if self.cells[0].type != "triangle":
            raise ValueError("The first cell type must be 'triangle' because the class use self.cells[0] to access the elements.")

    # ===========================================================
    # I/O methods
    # ===========================================================
    @classmethod
    def from_meshio(cls, mesh: meshio.Mesh) -> TriMesh3D:
        r"""
        Create a TriMesh3D instance from a meshio.Mesh object.

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

        To ensure the mesh is well formatted, run the method :meth:`pyblenderSDIC.meshes.TriMesh3D.validate` after loading.

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
    def load_from_vtk(cls, filepath: str) -> TriMesh3D:
        r"""
        Load a mesh from a VTK file.

        The VTK file must contain nodes, elements, and optionally UV mapping coordinates.

        .. code-block:: python

            from pyblenderSDIC.meshes import TriMesh3D
            # Load the mesh from a VTK file
            trimesh3d = TriMesh3D.load_from_vtk("path/to/mesh.vtk")

        To ensure the mesh is well formatted, run the method :meth:`pyblenderSDIC.meshes.TriMesh3D.validate` after loading.

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
    def load_from_dict(cls, data: Dict) -> TriMesh3D:
        r"""
        Create a TriMesh3D instance from a dictionary.

        The structure of the dictionary should be as provided by the :meth:`pyblenderSDIC.meshes.TriMesh3D.save_to_dict` method.
        The other fields of the dictionary are ignored.

        .. code-block:: python

            from pyblenderSDIC.meshes import TriMesh3D

            mesh_dict = {
                "type": "TriMesh3D [pyblenderSDIC]",
                "description": "Description of the mesh",
                "nodes": [[0.1, 0.2, 0.1], [0.5, 0.6, 0.4], [0.1, 0.6, 0.2]],
                "elements": [[1], [2], [3]],
                "uvmap": [[0.1, 0.2], [0.5, 0.6], [0.1, 0.6]],
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
        uvmap = data.get("uvmap", None)
        mesh.uvmap = uvmap

        return mesh
    

    @classmethod
    def load_from_json(cls, filepath: str) -> TriMesh3D:
        r"""
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
        r"""
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
        r"""
        Export the TriMesh3D's data to a dictionary.

        The structure of the dictionary is as follows:

        .. code-block:: python

            {
                "type": "TriMesh3D [pyblenderSDIC]",
                "description": "Description of the mesh",
                "nodes": [[0.1, 0.2, 0.1], [0.5, 0.6, 0.4], [0.1, 0.6, 0.2]],
                "elements": [[1], [2], [3]],
                "uvmap": [[0.1, 0.2], [0.5, 0.6], [0.1, 0.6]],
            }

        The other attributes of the mesh in ``point_data`` or ``cell_data`` are ignored.

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
        
        # Create the dictionary
        data = {"type": "TriMesh3D [pyblenderSDIC]",}

        # Add the description if it's not empty
        if description:
            data["description"] = description
        
        # Add the nodes and elements
        data["nodes"] = self.nodes.tolist()
        data["elements"] = self.elements.tolist()

        # Add the UV mapping if it exists
        if self.uvmap is not None:
            data["uvmap"] = self.uvmap.tolist()

        return data


    def save_to_json(self, filepath: str, description: str = "") -> None:
        r"""
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
        r"""
        Get or set the positions of the mesh nodes.

        The node coordinates are stored in the ``points`` attribute of the mesh
        and have shape (N, 3), where N is the number of nodes.

        .. code-block:: python

            # Get the nodes coordinates
            nodes = trimesh3d.nodes  # shape (N, 3)

            # Set the nodes coordinates
            trimesh3d.nodes = new_nodes
            trimesh3d.nodes[5, 0] = 42  # The 6-th node's x-coordinate is set to 42

        .. warning::

            This property uses ``numpy.asarray`` on the internal ``points`` array.
            As a result, any modification to the returned or setted array directly affects the mesh data.
            To avoid unintentional updates, assign a copy instead or use :

            - :meth:`pyblenderSDIC.meshes.TriMesh3D.get_nodes` to get a copy of the nodes.
            - :meth:`pyblenderSDIC.meshes.TriMesh3D.set_nodes` to set a copy of the nodes.

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

    def get_nodes(self) -> numpy.ndarray:
        """
        Get the nodes of the mesh as a copy.

        .. seealso::

            - :attr:`pyblenderSDIC.meshes.TriMesh3D.nodes` for the original nodes array.

        Returns
        -------
        numpy.ndarray
            The nodes of the mesh as a (N, 3) array of float64.
        """
        return self.nodes.copy()
    
    def set_nodes(self, nodes: numpy.ndarray) -> None:
        """
        Set the nodes of the mesh as a copy.

        .. seealso::

            - :attr:`pyblenderSDIC.meshes.TriMesh3D.nodes` for the original nodes array.

        Parameters
        ----------
        nodes : numpy.ndarray
            The new nodes to set, as a (N, 3) array of float64.
        """
        # First use the setter to check the input
        self.nodes = nodes
        # Then copy the data to avoid unintentional updates if the input was a view
        self.points = self.points.copy()



    @property
    def elements(self) -> numpy.ndarray:
        r"""
        Get or set the mesh elements (triangles).

        The elements are stored in the ``cells_dict['triangle']`` of the mesh and represent the connectivity 
        between nodes in the mesh. They are expected to be triangles, represented as 
        indices into the node array. The array has shape (M, 3), where M is the number of elements.

        .. code-block:: python

            # Get the elements (triangle connectivity)
            elements = trimesh3d.elements  # shape (M, 3)

            # Set the elements (as integer indices)
            trimesh3d.elements = new_elements
            trimesh3d.elements[0, 1] = 42  # The second node of the first element is set to the node with index 42

        .. warning::

            This property uses ``numpy.asarray`` on the ``cells[0].data`` attribute.
            Any modification to the returned array will directly affect the mesh data.
            To avoid unintentional updates, assign a copy instead or use :

            - :meth:`pyblenderSDIC.meshes.TriMesh3D.get_elements` to get a copy of the elements.
            - :meth:`pyblenderSDIC.meshes.TriMesh3D.set_elements` to set a copy of the elements.

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
        r"""
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
        r"""
        Set the elements of the mesh as a copy.

        .. seealso::

            - :attr:`pyblenderSDIC.meshes.TriMesh3D.elements` for the original elements array.

        Parameters
        ----------
        elements : numpy.ndarray
            The new elements to set, as a (M, 3) array of integers.
        """
        # First use the setter to check the input
        self.elements = elements
        # Then copy the data to avoid unintentional updates if the input was a view
        self.cells[0].data = self.cells[0].data.copy()


    @property
    def uvmap(self) -> Optional[numpy.ndarray]:
        r"""
        Get or set the UV mapping coordinates of the mesh.

        The UV coordinates are stored in the ``point_data`` dictionary under the key "uvmap".
        They must lie within the range [0, 1] and have shape (N, 3), where N is the number of nodes.
        The third column is typically filled with zeros for VTK compatibility.

        .. code-block:: python

            # Get the UV mapping coordinates
            uvmap = trimesh3d.uvmap  # shape (N, 3)

            # Set the UV mapping coordinates
            trimesh3d.uvmap = new_uvmap
            trimesh3d.uvmap[5, 0] = 0.5  # The 6-th node's U-coordinate is set to 0.5

        .. warning::

            This property uses ``numpy.asarray`` on the ``point_data["uvmap"]`` attribute.
            Any modification to the returned array directly affects the mesh data.
            To avoid unintentional updates, assign a copy instead or use :

            - :meth:`pyblenderSDIC.meshes.TriMesh3D.get_uvmap2D` to get a copy of the UV mapping.
            - :meth:`pyblenderSDIC.meshes.TriMesh3D.set_uvmap2D` to set a copy of the UV mapping.

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
        r"""
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
        r"""
        Set the UV mapping coordinates of the mesh as a copy.

        .. seealso::

            - :attr:`pyblenderSDIC.meshes.TriMesh3D.uvmap` for the original UV mapping coordinates.

        Parameters
        ----------
        uvmap : numpy.ndarray
            The new UV mapping coordinates to set, as a (N, 3) array of float64.
        """
        # First use the setter to check the input
        self.uvmap = uvmap
        # Then copy the data to avoid unintentional updates if the input was a view
        if self.uvmap is not None:
            self.point_data["uvmap"] = self.point_data["uvmap"].copy()


    @property
    def Nnodes(self) -> int:
        r"""
        Get the number of nodes in the mesh.

        Returns
        -------
        int
            The number of nodes in the mesh.
        """
        return self.points.shape[0]
    

    @property
    def Nelements(self) -> int:
        r"""
        Get the number of elements in the mesh.

        Returns
        -------
        int
            The number of elements in the mesh.
        """
        return self.cells[0].data.shape[0]


    # ============================================================
    # Special get and set methods
    # ============================================================
    def get_uvmap2D(self) -> Optional[numpy.ndarray]:
        r"""
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
        r"""
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
            raise ValueError(f"UV mapping coordinates 2D must have shape ({self.Nnodes}, 2).")

        # Add a third component of zeros for compatibility with VTK
        self.uvmap = numpy.hstack((uvmap2D, numpy.zeros((uvmap2D.shape[0], 1), dtype=numpy.float64)))

    # ============================================================
    # Validation methods
    # ============================================================
    def validate(self) -> None:
        r"""
        Validate the mesh.

        The method checks if the nodes, elements, UV mapping coordinates are correct shapes.
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
        
        # Checking if the elements are positive integers from 0 to Nnodes - 1
        if not numpy.all(self.cells[0].data < self.Nnodes) or not numpy.all(self.cells[0].data >= 0):
            raise ValueError("Elements must be positive integers from 0 to Nnodes - 1.")
    
    # ============================================================
    # Extracting sub meshes at a specified frame and visualization
    # ============================================================
    def construct_open3d_mesh(self) -> open3d.t.geometry.TriangleMesh:
        r"""
        Construct an Open3D mesh from the TriMesh3D object.

        This method creates an Open3D mesh object using the nodes and elements
        of the TriMesh3D object. 

        .. note::

            The open3d mesh is independent of the TriMesh3D object.
            Modifications to the Open3D mesh will not affect the TriMesh3D object and vice versa.

        Returns
        -------
        open3d.t.geometry.TriangleMesh
            An Open3D triangle mesh object.
        """
        # Get the nodes 
        nodes = self.get_nodes()

        # Create Open3D mesh
        o3d_mesh = open3d.t.geometry.TriangleMesh()
        o3d_mesh.vertex.positions = open3d.core.Tensor(nodes, open3d.core.float32)
        o3d_mesh.triangle.indices = open3d.core.Tensor(self.elements, open3d.core.int32)

        return o3d_mesh
    

    def visualize(
        self,
        element_highlighted: Union[Integral, Sequence[Integral]] = None,
        intersect_points: Optional[IntersectPoints] = None,
        ) -> None:
        r"""
        Visualize the mesh using Open3D.

        This method displays the 3D mesh using Open3D's interactive viewer.
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

        .. seealso::

            - :meth:`pyblenderSDIC.meshes.TriMesh3D.compute_intersect_points` to compute the intersection points.
            - :meth:`pyblenderSDIC.meshes.IntersectPoints` for the IntersectPoints class.

        Parameters
        ----------
        element_highlighted : int or sequence of int, optional
            Indices of mesh elements (triangles) to color in blue.

        intersect_points : IntersectPoints, optional
            3D intersection points to show in red. Only valid entries are displayed.
        """
        # Create the geometry
        geometries = []

        # Extracted the Open3D mesh
        mesh = self.construct_open3d_mesh()

        # Extracted the elements to be colored
        if element_highlighted is None:
            element_highlighted = []
        elif isinstance(element_highlighted, Integral):
            element_highlighted = [element_highlighted]
        element_highlighted = numpy.asarray(element_highlighted, dtype=int).flatten()
        element_highlighted = numpy.unique(element_highlighted)

        if element_highlighted.size !=0 and not numpy.all(0 <= element_highlighted < self.Nelements):
            raise ValueError("element_highlighted must be valid element indices.")
        
        indices = numpy.arange(self.Nelements)
        colors = numpy.full((self.Nelements, 3), [0.5, 0.5, 0.5]) # Default color for elements (gray)
        colors[numpy.isin(indices, element_highlighted)] = [0.0, 0.5, 0.5]  # Light blue for highlighted elements
        mesh.triangle.colors = open3d.core.Tensor(colors, open3d.core.float32)

        # Add the mesh to the geometries
        geometries.append({
            "geometry": mesh,
            "name": "Colored Elements"
        })

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

        # Add the lineset to the geometries
        geometries.append({
            "geometry": lineset,
            "name": "Mesh Edges"
        })

        # Create PointCloud for intersection points
        if intersect_points is not None:
            points = self.compute_intersect_points_coordinates(intersect_points)
            point_cloud = open3d.t.geometry.PointCloud()
            point_cloud.point.positions = open3d.core.Tensor(points, dtype=open3d.core.Dtype.Float32)
            point_cloud.point.colors = open3d.core.Tensor(numpy.tile([0.0, 0.0, 1.0], (points.shape[0], 1)), dtype=open3d.core.Dtype.Float32)  # Blue color for points
            
            # Add the point cloud to the geometries
            geometries.append({
                "geometry": point_cloud,
                "name": "Intersection Points"
            })
            
        # Launch Open3D viewer
        open3d.visualization.draw(geometries, point_size=15)
        
    # ============================================================
    # Rays and intersections
    # ============================================================
    def open3d_cast_ray(self, rays: numpy.ndarray) -> Dict:
        r"""
        Calculate the intersection of rays with a given mesh using Open3D.

        This method uses Open3D's raycasting capabilities to find the intersection points
        of rays with the mesh.

        .. code-block:: python

            # Define ray origins and directions
            rays_origins = numpy.array([[0, 0, 0], [1, 1, 1]]) # shape (L, 3)
            rays_directions = numpy.array([[0, 0, 1], [1, 1, 0]]) # shape (L, 3)
            rays = numpy.hstack((rays_origins, rays_directions)) # shape (L, 6)

            # Perform ray-mesh intersection
            ray_intersect = trimesh3d.open3d_cast_ray(rays)

        .. seealso::

            Documentation of Open3D's raycasting : 
            https://www.open3d.org/html/python_api/open3d.t.geometry.RaycastingScene.html#open3d.t.geometry.RaycastingScene.cast_rays

        Parameters
        ----------
        rays: numpy.ndarray
            A (..., 6) array of float32. Each component contains the position and the direction of a ray in the format [x0, y0, z0, dx, dy, dz].

        Returns
        -------
        ray_intersect : Dict
            The output of the raycasting operation by Open3D. 
        """
        # Extract the Open3D mesh for the specified frame
        o3d_mesh = self.construct_open3d_mesh()

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


    def compute_intersect_points(self, rays: numpy.ndarray) -> IntersectPoints:
        r"""
        Compute the intersection of rays with the mesh.

        This method uses Open3D to perform ray-mesh intersection and returns the intersection points
        and the corresponding triangle indices as an `IntersectPoints` object.

        .. code-block:: python

            # Define ray origins and directions
            rays_origins = numpy.array([[0, 0, 0], [1, 1, 1]]) # shape (L, 3)
            rays_directions = numpy.array([[0, 0, 1], [1, 1, 0]]) # shape (L, 3)
            rays = numpy.hstack((rays_origins, rays_directions)) # shape (L, 6)

            # Perform ray-mesh intersection
            intersect_points = trimesh3d.compute_intersect_points(rays)

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

        Returns
        -------
        IntersectPoints
            An object containing barycentric coordinates and triangle indices of the intersections.
        """
        # Perform ray-mesh intersection using Open3D
        results = self.open3d_cast_ray(rays)

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
    

    def compute_intersect_points_coordinates(self, intersect_points: IntersectPoints) -> numpy.ndarray:
        r"""
        Compute the 3D coordinates of intersection points from barycentric data.

        This method reconstructs the 3D position of the intersection points using the barycentric
        coordinates and the triangle indices contained in the given :class:`IntersectPoints` object.

        .. code-block:: python

            intersect_points = trimesh3d.compute_intersect_points(rays)
            coords = trimesh3d.compute_intersect_points_coordinates(intersect_points)

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
        nodes = self.get_nodes()
        A = nodes[self.elements[valid_idx, 0]]
        B = nodes[self.elements[valid_idx, 1]]
        C = nodes[self.elements[valid_idx, 2]]

        # Compute coordinates
        flat_points[valid] = w[:, None] * A + u[:, None] * B + v[:, None] * C

        # Reshape to original shape
        output_shape = (*intersect_points.uv.shape[:-1], 3)
        return flat_points.reshape(output_shape)
    

    def compute_element_normals(self) -> numpy.ndarray:
        r"""
        Compute the normals of the elements in the mesh.

        This method computes the normals of the elements in the mesh using the
        cross product of two edges of each triangle. The normals are normalized
        to unit length.

        .. code-block:: python

            normals = trimesh3d.compute_elements_normals()

        Returns
        -------
        numpy.ndarray
            A (M, 3) array of float64 containing the normals of the elements.
        """
        # Get the nodes
        nodes = self.get_nodes()

        # Compute the vectors for each triangle
        A = nodes[self.elements[:, 0]]
        B = nodes[self.elements[:, 1]]
        C = nodes[self.elements[:, 2]]

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
    

    def compute_element_centroids(self) -> numpy.ndarray:
        r"""
        Compute the centroids of the elements in the mesh.

        This method computes the centroids of the elements in the mesh by averaging
        the coordinates of their vertices.

        .. code-block:: python

            centroids = trimesh3d.compute_elements_centroids()

        Returns
        -------
        numpy.ndarray
            A (M, 3) array of float64 containing the centroids of the elements.
        """
        # Get the nodes
        nodes = self.get_nodes()

        # Compute the centroids by averaging the vertices of each triangle
        A = nodes[self.elements[:, 0]]
        B = nodes[self.elements[:, 1]]
        C = nodes[self.elements[:, 2]]

        return (A + B + C) / 3.0
    

    def compute_element_areas(self) -> numpy.ndarray:
        r"""
        Compute the areas of the elements in the mesh.

        This method computes the areas of the elements in the mesh using the
        cross product of two edges of each triangle.

        .. code-block:: python

            areas = trimesh3d.compute_elements_areas()

        Returns
        -------
        numpy.ndarray
            A (M,) array of float64 containing the areas of the elements.
        """
        # Get the nodes
        nodes = self.get_nodes()

        # Compute the vectors for each triangle
        A = nodes[self.elements[:, 0]]
        B = nodes[self.elements[:, 1]]
        C = nodes[self.elements[:, 2]]

        # Compute two edges of each triangle
        AB = B - A
        AC = C - A

        # Compute the cross product to get the area vector
        areas = numpy.linalg.norm(numpy.cross(AB, AC), axis=1) / 2.0

        return areas