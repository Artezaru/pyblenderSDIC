from __future__ import annotations

import os
import numpy
import meshio
import open3d
import json
from typing import Optional, Dict, Union, Any, Sequence
from numbers import Integral, Number

from .intersect_points import IntersectPoints



class TriangleMesh3D():
    r"""
    Represents a triangular 3D mesh with support for UV mapping and compatibility with: 

    - `meshio` for mesh I/O operations.
    - `open3d` for visualization and manipulation.

    This class is designed to handle triangular surface meshes in 3D space. 
    It includes support for texture mapping (UV coordinates).

    .. warning::

        The number of vertices and triangles are not designed to change after the mesh is created !

        
    Mesh Structure
    --------------

    - ``vertices``: A NumPy array of shape (N, 3) representing the coordinates of N mesh vertices.
    - ``triangles``: A NumPy array of shape (M, 3) representing M triangular triangles defined by vertex indices.

    .. code-block:: python

        import numpy
        from pyblenderSDIC.meshes import TriangleMesh3D

        # Create a simple triangle mesh with N=3 vertices and M=1 triangle.
        mesh = TriangleMesh3D(
            vertices=numpy.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]),
            triangles=numpy.array([[0, 1, 2]])
        )

        mesh.vertices # Access the vertices of the mesh (shape: (N, 3))
        mesh.triangles # Access the triangles of the mesh (shape: (M, 3))

    .. figure:: ../../../pyblenderSDIC/resources/doc/demi_cylinder_mesh.png
        :width: 400
        :align: center

        Vertices and triangles of a triangular mesh.

    Use the method ``validate`` to check the validity of the mesh structure:

    .. code-block:: python

        mesh.validate()  # Raises an exception if the mesh is invalid

    By default the method validate is automatically called when the mesh is created,

    Mesh Creation
    --------------

    The mesh can be created by giving the vertices and triangles directly, or by using gateway functions:

    - ``DICT``: Create a mesh from a dictionary containing vertices and triangles.
    - ``JSON``: Create a mesh from a JSON string or file containing vertices and triangles.
    - ``MESHIO``: Create a mesh from a meshio object or file.
    - ``OPEN3D``: Create a mesh from an open3d mesh object.
    - ``VTK``: Create a mesh from a VTK file.

    UV Mapping
    ----------

    The mesh supports UV mapping, allowing for texture coordinates to be associated with each vertex.
    UV coordinates can be set using the ``set_uvmap`` method, which takes a NumPy array of shape (M, 6) where each row contains the UV coordinates for a triangle in the format (u1, v1, u2, v2, u3, v3).
    UV coordinates can be accessed using the ``uvmap`` property, which returns a NumPy array of shape (M, 6).

    The UV coordinates represents the position of each vertex in the normalized texture space, where (0, 0) is the bottom-left corner and (1, 1) is the top-right corner.
    By default, UV coordinates follow the **OpenGL convention**, which is also used by Blender and most 3D engines.

    In this convention:

    - :math:`uv = (0, 0)` corresponds to the bottom-left corner of the texture.
    - :math:`uv = (1, 0)` corresponds to the bottom-right corner of the texture.
    - :math:`uv = (0, 1)` corresponds to the top-left corner of the texture.
    - :math:`uv = (1, 1)` corresponds to the top-right corner of the texture.

    If all vertices have the same UV coordinates in each triangle, the mesh can be considered as having a single texture applied uniformly across all triangles and the method ``set_vertices_uvmap`` can be called with a single set of UV coordinates for all triangles.

    Other-quantity
    --------------
    Other quantity of the mesh can also be computated, such as the number of vertices and triangles, the bounding box, and the volume of the mesh.
    These quantities can be accessed using the following properties:

    - ``bounding_box``: The bounding box of the mesh, represented as a NumPy array of shape (2, 3) with the minimum and maximum coordinates in each dimension.
    - ``volume``: The volume of the mesh, computed using the green therorem for polyhedra.
    - ``vertex_normals``: The normals of the vertices in the mesh, computed as the average of the normals of the triangles that share each vertex.
    - ``triangle_normals``: The normals of the triangles in the mesh, computed as the cross product of the edges of each triangle.
    - ``triangle_centroids``: The centroids of the triangles in the mesh, computed as the average position of the vertices of each triangle.
    - ``triangle_areas``: The areas of the triangles in the mesh, computed using the cross product of the edges of each triangle.

    This attributes are available only if there are already computed, otherwise they will return a None value.
    Use the ``compute_*`` methods to compute these quantities if they are not already computed.
    
    .. note::

        Any modification to the mesh vertices or triangles will invalidate these quantities, and they will need to be recomputed.

    
    Mesh-Properties
    -----------------
    According to the Open3D documentation, the mesh properties are:

    - ``is_edge_manifold``
    - ``is_edge_manifold_with_boundary``
    - ``is_vertex_manifold``
    - ``is_self_intersecting``
    - ``is_watertight``
    - ``is_orientable``

    Parameters
    ----------
    vertices : numpy.ndarray
        A NumPy array of shape (N, 3) representing the coordinates of N mesh vertices.

    triangles : numpy.ndarray
        A NumPy array of shape (M, 3) representing M triangular triangles defined by vertex indices.

    uvmap : Optional[numpy.ndarray], optional
        A NumPy array of shape (M, 6) representing the UV coordinates for each triangle in the mesh.
        Each row should contain the UV coordinates in the format (u1, v1, u2, v2, u3, v3).
        If not provided, the mesh will not have UV mapping.

    Examples
    --------

    Create a simple triangle mesh with vertices and triangles:

    .. code-block:: python

        import numpy
        from pyblenderSDIC.meshes import TriangleMesh3D

        vertices = numpy.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0]
        ])

        triangles = numpy.array([
            [0, 2, 1],
            [0, 3, 2],
            [0, 1, 3],
            [1, 4, 3],
            [2, 3, 4],
            [2, 4, 1]
        ])

        vertices_uvmap = numpy.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [0.5, 0.5]
        ])
    
        mesh = TriangleMesh3D(
            vertices=vertices,
            triangles=triangles,
            uvmap=vertices_uvmap
        )

    Save the mesh to a file in VTK format:

    .. code-block:: python

        mesh.to_vtk("mesh.vtk")

    Visualize the mesh using Open3D:

    .. code-block:: python

    """
    def __init__(self, 
                 vertices: numpy.ndarray, 
                 triangles: numpy.ndarray, 
                 uvmap: Optional[numpy.ndarray] = None,
                 ) -> None:
        # Active bypass mode for testing purposes
        self.__internal_bypass__ = True
        self.vertices = vertices
        self.triangles = triangles
        self.__internal_bypass__ = False
        self.__internal_check_vertices()
        self.__internal_check_triangles()

        # Set the UV map if provided, otherwise initialize an empty UV map
        self._uvmap: Optional[numpy.ndarray] = None
        self.uvmap = uvmap

        # Set the default values for computed quantities
        self._bounding_box: Optional[numpy.ndarray] = None
        self._volume: Optional[float] = None
        self._vertex_normals: Optional[numpy.ndarray] = None
        self._triangle_normals: Optional[numpy.ndarray] = None
        self._triangle_centroids: Optional[numpy.ndarray] = None
        self._triangle_areas: Optional[numpy.ndarray] = None

        # Set the default properties for Open3D compatibility
        self._is_edge_manifold: Optional[bool] = None
        self._is_edge_manifold_with_boundary: Optional[bool] = None
        self._is_vertex_manifold: Optional[bool] = None
        self._is_self_intersecting: Optional[bool] = None
        self._is_watertight: Optional[bool] = None
        self._is_orientable: Optional[bool] = None

    # =======================================================================
    # Internal Methods
    # =======================================================================
    @property
    def internal_bypass(self) -> bool:
        r"""
        Get and set the internal bypass mode status.
        When enabled, internal checks are skipped.

        This is useful for testing purposes, but should not be used in production code.

        Parameters
        ----------
        value : bool
            If True, internal checks are bypassed. If False, internal checks are performed.

        Raises
        --------
        TypeError
            If the value is not a boolean.
        """
        return self.__internal_bypass__
    
    @internal_bypass.setter
    def internal_bypass(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise TypeError(f"Bypass mode must be a boolean, got {type(value)}.")
        self.__internal_bypass__ = value
    
    def __internal_check_vertices(self) -> None:
        r"""
        Internal method to check the validity of the vertices array.
        """
        if self.__internal_bypass__:
            return
        
        if not isinstance(self._vertices, numpy.ndarray):
            raise TypeError(f"Points must be a numpy.ndarray, got {type(self._vertices)}.")
        if not self._vertices.dtype == numpy.float64:
            raise TypeError(f"Points must be of type float64, got {self._vertices.dtype}.")
        if not self._vertices.ndim == 2:
            raise ValueError(f"Points must be a 2D array, got {self._vertices.ndim} dimensions.")
        if not self._vertices.shape == (self.Nvertices, 3):
            raise ValueError(f"Points must have shape ({self.Nvertices}, 3), got {self._vertices.shape}.")
        if not numpy.all(numpy.isfinite(self._vertices)):
            raise ValueError("Points must contain finite values only.")
        if not self.Nvertices >= 3:
            raise ValueError("Points array must contain at least 3 vertices to form a triangle.")
        
    def __internal_check_triangles(self) -> None:
        r"""
        Internal method to check the validity of the triangles array.
        """
        if self.__internal_bypass__:
            return
        
        if not isinstance(self._triangles, numpy.ndarray):
            raise TypeError(f"Triangles must be a numpy.ndarray, got {type(self._triangles)}.")
        if not self._triangles.dtype == numpy.int64:
            raise TypeError(f"Triangles must be of type int64, got {self._triangles.dtype}.")
        if not self._triangles.ndim == 2:
            raise ValueError(f"Triangles must be a 2D array, got {self._triangles.ndim} dimensions.")
        if not self._triangles.shape == (self.Ntriangles, 3):
            raise ValueError(f"Triangles must have shape ({self.Ntriangles}, 3), got {self._triangles.shape}.")
        if not numpy.all(numpy.isfinite(self._triangles)):
            raise ValueError("Triangles must contain finite values only.")
        if not numpy.all(self._triangles >= 0):
            raise ValueError("Triangles must contain non-negative indices only.")
        if not numpy.all(self._triangles < self.Nvertices):
            raise ValueError("Triangle indices must be less than the number of vertices.")
        if not self.Ntriangles >= 1:
            raise ValueError("Triangles array must contain at least 1 triangle.")
        
    def __internal_check_uvmap(self) -> None:
        r"""
        Internal method to check the validity of the UV map.
        """
        if self.__internal_bypass__:
            return
        
        if self._uvmap is not None:
            if not isinstance(self._uvmap, numpy.ndarray):
                raise TypeError(f"UV map must be a numpy.ndarray, got {type(self._uvmap)}.")
            if not self._uvmap.dtype == numpy.float64:
                raise TypeError(f"UV map must be of type float64, got {self._uvmap.dtype}.")
            if not self._uvmap.ndim == 2:
                raise ValueError(f"UV map must be a 2D array, got {self._uvmap.ndim} dimensions.")
            if not self._uvmap.shape == (self.Ntriangles, 6):
                raise ValueError(f"UV map must have shape ({self.Ntriangles}, 6), got {self._uvmap.shape}.")
            if not numpy.all(numpy.isfinite(self._uvmap)):
                raise ValueError("UV map must contain finite values only.")
            if not numpy.all((self._uvmap >= 0) & (self._uvmap <= 1)):
                raise ValueError("UV map coordinates must be in the range [0, 1].")
            
    def __internal_check_bounding_box(self) -> None:
        r"""
        Internal method to check the validity of the bounding box.
        """
        if self.__internal_bypass__:
            return
        
        if self._bounding_box is not None:
            if not isinstance(self._bounding_box, numpy.ndarray):
                raise TypeError(f"Bounding box must be a numpy.ndarray, got {type(self._bounding_box)}.")
            if not self._bounding_box.dtype == numpy.float64:
                raise TypeError(f"Bounding box must be of type float64, got {self._bounding_box.dtype}.")
            if not self._bounding_box.ndim == 2:
                raise ValueError(f"Bounding box must be a 2D array, got {self._bounding_box.ndim} dimensions.")
            if not self._bounding_box.shape == (2, 3):
                raise ValueError(f"Bounding box must have shape (2, 3), got {self._bounding_box.shape}.")
            if not numpy.all(numpy.isfinite(self._bounding_box)):
                raise ValueError("Bounding box must contain finite values only.")
            
    def __internal_check_volume(self) -> None:
        r"""
        Internal method to check the validity of the volume.
        """
        if self.__internal_bypass__:
            return
        
        if self._volume is not None:
            if not isinstance(self._volume, Number):
                raise TypeError(f"Volume must be a number, got {type(self._volume)}.")
            if not numpy.isfinite(self._volume):
                raise ValueError("Volume must be a finite value.")
            
    def __internal_check_vertex_normals(self) -> None:
        r"""
        Internal method to check the validity of the vertices normals.
        """
        if self.__internal_bypass__:
            return
        
        if self._vertex_normals is not None:
            if not isinstance(self._vertex_normals, numpy.ndarray):
                raise TypeError(f"Points normals must be a numpy.ndarray, got {type(self._vertex_normals)}.")
            if not self._vertex_normals.dtype == numpy.float64:
                raise TypeError(f"Points normals must be of type float64, got {self._vertex_normals.dtype}.")
            if not self._vertex_normals.ndim == 2:
                raise ValueError(f"Points normals must be a 2D array, got {self._vertex_normals.ndim} dimensions.")
            if not self._vertex_normals.shape == (self.Nvertices, 3):
                raise ValueError(f"Points normals must have shape ({self.Nvertices}, 3), got {self._vertex_normals.shape}.")
            if not numpy.all(numpy.isfinite(self._vertex_normals)):
                raise ValueError("Points normals must contain finite values only.")
            
    def __internal_check_triangle_normals(self) -> None:
        r"""
        Internal method to check the validity of the triangles normals.
        """
        if self.__internal_bypass__:
            return
        
        if self._triangle_normals is not None:
            if not isinstance(self._triangle_normals, numpy.ndarray):
                raise TypeError(f"Triangles normals must be a numpy.ndarray, got {type(self._triangle_normals)}.")
            if not self._triangle_normals.dtype == numpy.float64:
                raise TypeError(f"Triangles normals must be of type float64, got {self._triangle_normals.dtype}.")
            if not self._triangle_normals.ndim == 2:
                raise ValueError(f"Triangles normals must be a 2D array, got {self._triangle_normals.ndim} dimensions.")
            if not self._triangle_normals.shape == (self.Ntriangles, 3):
                raise ValueError(f"Triangles normals must have shape ({self.Ntriangles}, 3), got {self._triangle_normals.shape}.")
            if not numpy.all(numpy.isfinite(self._triangle_normals)):
                raise ValueError("Triangles normals must contain finite values only.")
            
    def __internal_check_triangle_centroids(self) -> None:
        r"""
        Internal method to check the validity of the triangles centroids.
        """
        if self.__internal_bypass__:
            return
        
        if self._triangle_centroids is not None:
            if not isinstance(self._triangle_centroids, numpy.ndarray):
                raise TypeError(f"Triangles centroids must be a numpy.ndarray, got {type(self._triangle_centroids)}.")
            if not self._triangle_centroids.dtype == numpy.float64:
                raise TypeError(f"Triangles centroids must be of type float64, got {self._triangle_centroids.dtype}.")
            if not self._triangle_centroids.ndim == 2:
                raise ValueError(f"Triangles centroids must be a 2D array, got {self._triangle_centroids.ndim} dimensions.")
            if not self._triangle_centroids.shape == (self.Ntriangles, 3):
                raise ValueError(f"Triangles centroids must have shape ({self.Ntriangles}, 3), got {self._triangle_centroids.shape}.")
            if not numpy.all(numpy.isfinite(self._triangle_centroids)):
                raise ValueError("Triangles centroids must contain finite values only.")
            
    def __internal_check_triangle_areas(self) -> None:
        r"""
        Internal method to check the validity of the triangles areas.
        """
        if self.__internal_bypass__:
            return
        
        if self._triangle_areas is not None:
            if not isinstance(self._triangle_areas, numpy.ndarray):
                raise TypeError(f"Triangles areas must be a numpy.ndarray, got {type(self._triangle_areas)}.")
            if not self._triangle_areas.dtype == numpy.float64:
                raise TypeError(f"Triangles areas must be of type float64, got {self._triangle_areas.dtype}.")
            if not self._triangle_areas.ndim == 1:
                raise ValueError(f"Triangles areas must be a 1D array, got {self._triangle_areas.ndim} dimensions.")
            if not self._triangle_areas.shape == (self.Ntriangles,):
                raise ValueError(f"Triangles areas must have shape ({self.Ntriangles},), got {self._triangle_areas.shape}.")
            if not numpy.all(numpy.isfinite(self._triangle_areas)):
                raise ValueError("Triangles areas must contain finite values only.")
            if not numpy.all(self._triangle_areas >= 0):
                raise ValueError("Triangles areas must be non-negative values.")
            
    def validate(self) -> None:
        r"""
        Validate the mesh structure.

        This method checks the validity of the vertices, triangles, and UV map.
        If any of the checks fail, an exception is raised.
        """
        bypass_mode = self.__internal_bypass__
        self.__internal_bypass__ = False # Disable bypass mode for validation

        self.__internal_check_vertices()
        self.__internal_check_triangles()
        self.__internal_check_uvmap()
        self.__internal_check_bounding_box()
        self.__internal_check_volume()
        self.__internal_check_vertex_normals()
        self.__internal_check_triangle_normals()
        self.__internal_check_triangle_centroids()
        self.__internal_check_triangle_areas()
        
        # restore bypass mode
        self.__internal_bypass__ = bypass_mode
        

    def __remove_quantities(self) -> None:
        r"""
        Internal method to remove all computed quantities from the mesh.
        
        This method is used to reset the mesh state when vertices or triangles are modified.
        It sets all computed quantities to None, so they will be recomputed when needed.
        """
        self._bounding_box = None
        self._volume = None
        self._vertex_normals = None
        self._triangle_normals = None
        self._triangle_centroids = None
        self._triangle_areas = None


    def __remove_properties(self) -> None:
        r"""
        Internal method to remove all properties related to Open3D compatibility.

        This method is used to reset the mesh state when vertices or triangles are modified.
        It sets all Open3D properties to None, so they will be recomputed when needed.
        """
        self._is_edge_manifold = None
        self._is_edge_manifold_with_boundary = None
        self._is_vertex_manifold = None
        self._is_self_intersecting = None
        self._is_watertight = None
        self._is_orientable = None
        
    
    def __repr__(self) -> str:
        r"""
        Return a string representation of the mesh.

        This method returns a string that summarizes the mesh structure, including the number of vertices and triangles.
        """
        return f"TriangleMesh3D(Nvertices={self.Nvertices}, Ntriangles={self.Ntriangles}, UVmap={'Yes' if self._uvmap is not None else 'No'})"



    # =======================================================================
    # Properties Getters and Setters
    # =======================================================================
    @property
    def vertices(self) -> numpy.ndarray:
        r"""
        Get or set the positions of the mesh vertices.

        The vertex coordinates are stored in the ``vertices`` attribute of the mesh
        and have shape (N, 3), where N is the number of vertices.

        .. code-block:: python

            # Get the vertices coordinates
            vertices = mesh.vertices  # shape (N, 3)

            # Set the vertices coordinates
            mesh.vertices = new_vertices
            mesh.vertices[5, 0] = 42  # The 6-th vertex's x-coordinate is set to 42

        .. warning::

            This property uses ``numpy.asarray`` on the internal ``vertices`` array.
            As a result, any modification to the returned or setted array directly affects the mesh data.
            To avoid unintentional updates, assign a copy instead.

        Parameters
        ----------
        value : numpy.ndarray
            A NumPy array of shape (N, 3) representing the coordinates of N mesh vertices.
        """
        return self._vertices
    
    @vertices.setter
    def vertices(self, value: numpy.ndarray) -> None:
        self._vertices = numpy.asarray(value, dtype=numpy.float64)
        self.__internal_check_vertices()
        self.__remove_quantities()
        self.__remove_properties()


    @property
    def triangles(self) -> numpy.ndarray:
        r"""
        Get or set the indices of the mesh triangles.

        The triangle indices are stored in the ``triangles`` attribute of the mesh
        and have shape (M, 3), where M is the number of triangles.

        .. code-block:: python

            # Get the triangle indices
            triangles = mesh.triangles  # shape (M, 3)

            # Set the triangle indices
            mesh.triangles = new_triangles
            mesh.triangles[0, 0] = 42  # The first triangle's first vertex index is set to 42

        .. warning::

            This property uses ``numpy.asarray`` on the internal ``triangles`` array.
            As a result, any modification to the returned or setted array directly affects the mesh data.
            To avoid unintentional updates, assign a copy instead.

        Parameters
        ----------
        value : numpy.ndarray
            A NumPy array of shape (M, 3) representing M triangular triangles defined by vertex indices.
        """
        return self._triangles
    
    @triangles.setter
    def triangles(self, value: numpy.ndarray) -> None:
        self._triangles = numpy.asarray(value, dtype=numpy.int64)
        self.__internal_check_triangles()
        self.__remove_quantities()
        self.__remove_properties()


    @property
    def uvmap(self) -> Optional[numpy.ndarray]:
        r"""
        Get or set the UV map of the mesh.

        The UV map is stored in the ``uvmap`` attribute of the mesh
        and has shape (M, 6), where M is the number of triangles.
        Each row contains the UV coordinates for a triangle in the format (u1, v1, u2, v2, u3, v3).

        .. code-block:: python

            # Get the UV map
            uvmap = mesh.uvmap  # shape (M, 6)

            # Set the UV map
            mesh.uvmap = new_uvmap

        To apply a single set of UV coordinates to all triangles, use the method ``set_vertices_uvmap`` instead.

        Parameters
        ----------
        value : Optional[numpy.ndarray]
            A NumPy array of shape (M, 6) representing the UV coordinates for each triangle in the mesh.
            If None, the mesh will not have UV mapping.
        """
        return self._uvmap
    
    @uvmap.setter
    def uvmap(self, value: Optional[numpy.ndarray]) -> None:
        if value is not None:
            self._uvmap = numpy.asarray(value, dtype=numpy.float64)
        else:
            self._uvmap = None
        self.__internal_check_uvmap()


    @property
    def Nvertices(self) -> int:
        r"""
        Get the number of vertices in the mesh.

        Returns
        -------
        int
            The number of vertices in the mesh.
        """
        return self._vertices.shape[0]
    

    @property
    def Ntriangles(self) -> int:
        r"""
        Get the number of triangles in the mesh.

        Returns
        -------
        int
            The number of triangles in the mesh.
        """
        return self._triangles.shape[0]
    

    @property
    def bounding_box(self) -> Optional[numpy.ndarray]:
        r"""
        Get the bounding box of the mesh.

        The bounding box is represented as a NumPy array of shape (2, 3),
        where the first row contains the minimum coordinates and the second row contains the maximum coordinates in each dimension.

        .. seealso::

            The method ``compute_bounding_box`` can be used to compute the bounding box if it is not already computed.

        .. warning::

            This property uses ``numpy.asarray`` on the internal ``bounding_box`` array.
            As a result, any modification to the returned or setted array directly affects the mesh data.
            To avoid unintentional updates, assign a copy instead.

        Returns
        -------
        Optional[numpy.ndarray]
            A NumPy array of shape (2, 3) representing the bounding box of the mesh.
            If not computed, returns None.
        """
        return self._bounding_box

    @bounding_box.setter
    def bounding_box(self, value: Optional[numpy.ndarray]) -> None:
        if value is not None:
            self._bounding_box = numpy.asarray(value, dtype=numpy.float64)
        else:
            self._bounding_box = None
        self.__internal_check_bounding_box()

    
    @property
    def volume(self) -> Optional[float]:
        r"""
        Get the volume of the mesh.

        .. seealso::

            The method ``compute_volume`` can be used to compute the volume if it is not already computed.

        Returns
        -------
        Optional[float]
            The volume of the mesh.
            If not computed, returns None.
        """
        return self._volume
    
    @volume.setter
    def volume(self, value: Optional[float]) -> None:
        if value is not None:
            self._volume = float(value)
        else:
            self._volume = None
        self.__internal_check_volume()


    @property
    def vertex_normals(self) -> Optional[numpy.ndarray]:
        r"""
        Get the normals of the vertices in the mesh.

        The normals are computed as the average of the normals of the triangles that share each vertex.
        The normals are represented as a NumPy array of shape (N, 3), where N is the number of vertices.

        .. seealso::

            The method ``compute_vertex_normals`` can be used to compute the normals if they are not already computed.

        .. warning::

            This property uses ``numpy.asarray`` on the internal ``vertex_normals`` array.
            As a result, any modification to the returned or setted array directly affects the mesh data.
            To avoid unintentional updates, assign a copy instead.

        Returns
        -------
        Optional[numpy.ndarray]
            A NumPy array of shape (N, 3) representing the normals of the vertices in the mesh.
            If not computed, returns None.
        """
        return self._vertex_normals
    
    @vertex_normals.setter
    def vertex_normals(self, value: Optional[numpy.ndarray]) -> None:
        if value is not None:
            self._vertex_normals = numpy.asarray(value, dtype=numpy.float64)
        else:
            self._vertex_normals = None
        self.__internal_check_vertex_normals()


    @property
    def triangle_normals(self) -> Optional[numpy.ndarray]:
        r"""
        Get the normals of the triangles in the mesh.

        The normals are computed as the cross product of the edges of each triangle.
        The normals are represented as a NumPy array of shape (M, 3), where M is the number of triangles.

        .. seealso::

            The method ``compute_triangle_normals`` can be used to compute the normals if they are not already computed.

        .. warning::

            This property uses ``numpy.asarray`` on the internal ``triangle_normals`` array.
            As a result, any modification to the returned or setted array directly affects the mesh data.
            To avoid unintentional updates, assign a copy instead.

        Returns
        -------
        Optional[numpy.ndarray]
            A NumPy array of shape (M, 3) representing the normals of the triangles in the mesh.
            If not computed, returns None.
        """
        return self._triangle_normals
    
    @triangle_normals.setter
    def triangle_normals(self, value: Optional[numpy.ndarray]) -> None:
        if value is not None:
            self._triangle_normals = numpy.asarray(value, dtype=numpy.float64)
        else:
            self._triangle_normals = None
        self.__internal_check_triangle_normals()

    
    @property
    def triangle_centroids(self) -> Optional[numpy.ndarray]:
        r"""
        Get the centroids of the triangles in the mesh.

        The centroids are computed as the average position of the vertices of each triangle.
        The centroids are represented as a NumPy array of shape (M, 3), where M is the number of triangles.

        .. seealso::

            The method ``compute_triangle_centroids`` can be used to compute the centroids if they are not already computed.

        .. warning::

            This property uses ``numpy.asarray`` on the internal ``triangle_centroids`` array.
            As a result, any modification to the returned or setted array directly affects the mesh data.
            To avoid unintentional updates, assign a copy instead.

        Returns
        -------
        Optional[numpy.ndarray]
            A NumPy array of shape (M, 3) representing the centroids of the triangles in the mesh.
            If not computed, returns None.
        """
        return self._triangle_centroids
    
    @triangle_centroids.setter
    def triangle_centroids(self, value: Optional[numpy.ndarray]) -> None:
        if value is not None:
            self._triangle_centroids = numpy.asarray(value, dtype=numpy.float64)
        else:
            self._triangle_centroids = None
        self.__internal_check_triangle_centroids()

    
    @property
    def triangle_areas(self) -> Optional[numpy.ndarray]:
        r"""
        Get the areas of the triangles in the mesh.

        The areas are computed using the cross product of the edges of each triangle.
        The areas are represented as a NumPy array of shape (M,), where M is the number of triangles.

        .. seealso::

            The method ``compute_triangle_areas`` can be used to compute the areas if they are not already computed.

        .. warning::

            This property uses ``numpy.asarray`` on the internal ``triangle_areas`` array.
            As a result, any modification to the returned or setted array directly affects the mesh data.
            To avoid unintentional updates, assign a copy instead.

        Returns
        -------
        Optional[numpy.ndarray]
            A NumPy array of shape (M,) representing the areas of the triangles in the mesh.
            If not computed, returns None.
        """
        return self._triangle_areas
    
    @triangle_areas.setter
    def triangle_areas(self, value: Optional[numpy.ndarray]) -> None:
        if value is not None:
            self._triangle_areas = numpy.asarray(value, dtype=numpy.float64)
        else:
            self._triangle_areas = None
        self.__internal_check_triangle_areas()


    # =======================================================================
    # I/O methods
    # =======================================================================
    @classmethod
    def from_meshio(cls, mesh: meshio.Mesh) -> TriangleMesh3D:
        r"""
        Create a TriangleMesh3D instance from a meshio.Mesh object.

        The following fields are extracted:

        - mesh.points → vertices
        - mesh.cells → triangles
        - mesh.point_data → vertex_normals
        - mesh.cell_data → triangle_normals, triangle_centroids, triangle_areas, uvmap

        .. code-block:: python

            import meshio
            from pyblenderSDIC.mesh import TriangleMesh3D

            # Read the mesh from a file
            mesh = meshio.read("path/to/mesh.vtk")
            # Create a TriangleMesh3D instance from the meshio object
            mesh = TriangleMesh3D.from_meshio(mesh)

        Meshio Structure
        ----------------

        The ``points`` attribute of the meshio object is expected to be a NumPy array of shape (N, 3),
        where N is the number of vertices and each row contains the coordinates of a vertex in 3D space.

        The ``cells`` attribute of the meshio object is expected to be a list with only one element,
        This element should be a dictionary with the ``triangle`` key and a NumPy array of shape (M, 3) as value,
        where M is the number of triangles and each row contains the indices of the vertices that form a triangle.

        The ``point_data`` attribute of the meshio object can contain the normals of the vertices under the key ``"normals"``.

        The ``cell_data`` attribute of the meshio object can contain the normals of the triangles under the key ``"normals"``,
        the centroids of the triangles under the key ``"centroids"``, the areas of the triangles under the key ``"areas"``, and the UV map under the key ``"uvmap"``.
        
        Parameters
        ----------
        mesh : meshio.Mesh
            A meshio.Mesh object containing the mesh data.

        Returns
        -------
        TriangleMesh3D
            A TriangleMesh3D instance created from the meshio object.
        """
        if not isinstance(mesh, meshio.Mesh):
            raise TypeError(f"Expected a meshio.Mesh object, got {type(mesh)}.")
        if not len(mesh.cells) == 1 or not "triangle" in mesh.cells[0].type:
            raise ValueError("Mesh cells must contain a single 'triangle' cell type with shape (M, 3).")

        # Extract vertices and triangles from the meshio object
        vertices = numpy.asarray(mesh.points, dtype=numpy.float64)
        triangles = numpy.asarray(mesh.cells[0].data, dtype=numpy.int64)

        # Create the TriangleMesh3D instance
        mesh_instance = cls(vertices=vertices, triangles=triangles)
        mesh_instance.validate()  # Validate the mesh structure

        # Extract vertex normals if available
        if mesh.point_data is not None and "normals" in mesh.point_data:
            mesh_instance.vertex_normals = numpy.asarray(mesh.point_data["normals"][0], dtype=numpy.float64)

        # Extract triangle normals, centroids, areas, and UV map if available
        if mesh.cell_data is not None:
            if "normals" in mesh.cell_data:
                mesh_instance.triangle_normals = numpy.asarray(mesh.cell_data["normals"][0], dtype=numpy.float64)
            if "centroids" in mesh.cell_data:
                mesh_instance.triangle_centroids = numpy.asarray(mesh.cell_data["centroids"][0], dtype=numpy.float64)
            if "areas" in mesh.cell_data:
                mesh_instance.triangle_areas = numpy.asarray(mesh.cell_data["areas"][0], dtype=numpy.float64)
            if "uvmap" in mesh.cell_data:
                mesh_instance.uvmap = numpy.asarray(mesh.cell_data["uvmap"][0], dtype=numpy.float64)

        return mesh_instance
    
    def to_meshio(self) -> meshio.Mesh:
        r"""
        Convert the TriangleMesh3D instance to a meshio.Mesh object.

        The following fields are set in the meshio object:

        - mesh.points → vertices
        - mesh.cells → triangles
        - mesh.point_data → vertex_normals
        - mesh.cell_data → triangle_normals, triangle_centroids, triangle_areas, uvmap

        .. code-block:: python

            import meshio
            from pyblenderSDIC.mesh import TriangleMesh3D

            # Create a TriangleMesh3D instance
            mesh = TriangleMesh3D(vertices=..., triangles=...)
            # Convert the mesh to a meshio object
            meshio_mesh = mesh.to_meshio()

        .. seealso::

            :meth:`from_meshio` for creating a TriangleMesh3D instance from a meshio object and more details about the meshio structure.

        Returns
        -------
        meshio.Mesh
            A meshio.Mesh object containing the mesh data.
        """
        points = numpy.asarray(self.vertices, dtype=numpy.float64)
        cells = [("triangle", numpy.asarray(self.triangles, dtype=numpy.int64))]

        if self.vertex_normals is not None:
            point_data = {"normals": [numpy.asarray(self.vertex_normals, dtype=numpy.float64)]}
        else:
            point_data = {}

        if self.triangle_normals is not None or self.triangle_centroids is not None or self.triangle_areas is not None or self.uvmap is not None:
            cell_data = {}
            if self.triangle_normals is not None:
                cell_data["normals"] = [numpy.asarray(self.triangle_normals, dtype=numpy.float64)]
            if self.triangle_centroids is not None:
                cell_data["centroids"] = [numpy.asarray(self.triangle_centroids, dtype=numpy.float64)]
            if self.triangle_areas is not None:
                cell_data["areas"] = [numpy.asarray(self.triangle_areas, dtype=numpy.float64)]
            if self.uvmap is not None:
                cell_data["uvmap"] = [numpy.asarray(self.uvmap, dtype=numpy.float64)]
        else:
            cell_data = None
        
        return meshio.Mesh(points=points, cells=cells, point_data=point_data, cell_data=cell_data)
    

    @classmethod
    def from_open3d(cls, mesh: Union[open3d.t.geometry.TriangleMesh, open3d.geometry.TriangleMesh]) -> TriangleMesh3D:
        r"""
        Create a TriangleMesh3D instance from an Open3D TriangleMesh object.

        .. code-block:: python

            import open3d as o3d
            from pyblenderSDIC.mesh import TriangleMesh3D

            # Read the mesh from a file
            mesh = o3d.io.read_triangle_mesh("path/to/mesh.ply")
            # Create a TriangleMesh3D instance from the Open3D object
            mesh = TriangleMesh3D.from_open3d(mesh)

        .. warning::
            
            For now, the method only extracts the vertices, triangles, and UV map (if available) from the Open3D mesh.
            The other properties (normals, centroids, areas) are not extracted and must be computed separately.

        Parameters
        ----------
        mesh : Union[open3d.t.geometry.TriangleMesh, open3d.geometry.TriangleMesh]
            An Open3D TriangleMesh object containing the mesh data.

        Returns
        -------
        TriangleMesh3D
            A TriangleMesh3D instance created from the Open3D object.
        """
        if not isinstance(mesh, (open3d.t.geometry.TriangleMesh, open3d.geometry.TriangleMesh)):
            raise TypeError(f"Expected an Open3D TriangleMesh object, got {type(mesh)}.")

        if isinstance(mesh, open3d.geometry.TriangleMesh): # Legacy Open3D mesh
            vertices = numpy.asarray(mesh.vertices, dtype=numpy.float64)
            triangles = numpy.asarray(mesh.triangles, dtype=numpy.int64)
            mesh_instance = cls(vertices=vertices, triangles=triangles)
            mesh_instance.validate()  # Validate the mesh structure

            # Check if UV mapping is available
            if mesh.triangle_uvs is not None and numpy.asarray(mesh.triangle_uvs).size > 0:
                uvmap = numpy.asarray(mesh.triangle_uvs, dtype=numpy.float64)
                # Convert UV map to the format (M, 6) - u1, v1, u2, v2, u3, v3
                uvmap = uvmap.reshape(-1, 6)
                mesh_instance.uvmap = uvmap

        else: # Open3D T geometry mesh
            vertices = numpy.asarray(mesh.vertex.positions.numpy(), dtype=numpy.float64)
            triangles = numpy.asarray(mesh.triangle.indices.numpy(), dtype=numpy.int64)
            mesh_instance = cls(vertices=vertices, triangles=triangles)
            mesh_instance.validate()  # Validate the mesh structure

            # Check if UV mapping is available
            if any(key == "texture_uvs" for key, _ in mesh.triangle.items()):
                uvmap = numpy.asarray(mesh.triangle.texture_uvs.numpy(), dtype=numpy.float64)
                # Convert UV map to the format (M, 6) - u1, v1, u2, v2, u3, v3
                uvmap = uvmap.reshape(-1, 6)
                mesh_instance.uvmap = uvmap

        return mesh_instance
    

    def to_open3d(self, legacy: bool = False) -> Union[open3d.t.geometry.TriangleMesh, open3d.geometry.TriangleMesh]:
        r"""
        Convert the TriangleMesh3D instance to an Open3D TriangleMesh object.

        If `legacy` is True, the method returns a legacy Open3D TriangleMesh object.
        Otherwise, it returns a T geometry TriangleMesh object.

        .. code-block:: python

            import open3d as o3d
            from pyblenderSDIC.mesh import TriangleMesh3D

            # Create a TriangleMesh3D instance
            mesh = TriangleMesh3D(vertices=..., triangles=...)
            # Convert the mesh to an Open3D object
            open3d_mesh = mesh.to_open3d()

        .. warning::

            For now, the method only converts the vertices, triangles, and UV map (if available) to the Open3D mesh.
            The other properties (normals, centroids, areas) are not converted and must be computed separately.

        Parameters
        ----------
        legacy : bool, optional
            If True, return a legacy Open3D TriangleMesh object. Default is False.

        Returns
        -------
        Union[open3d.t.geometry.TriangleMesh, open3d.geometry.TriangleMesh]
            An Open3D TriangleMesh object containing the mesh data.
        """
        if legacy:
            o3d_mesh = open3d.geometry.TriangleMesh()
            o3d_mesh.vertices = open3d.utility.Vector3dVector(self.vertices)
            o3d_mesh.triangles = open3d.utility.Vector3iVector(self.triangles)

            # Check if UV mapping is available
            if self.uvmap is not None:
                uvmap = self.uvmap.reshape(-1, 2)
                o3d_mesh.triangle_uvs = open3d.utility.Vector2dVector(uvmap)

        else:
            o3d_mesh = open3d.t.geometry.TriangleMesh()
            o3d_mesh.vertex.positions = open3d.core.Tensor(self.vertices, dtype=open3d.core.float32)
            o3d_mesh.triangle.indices = open3d.core.Tensor(self.triangles, dtype=open3d.core.int32)

            # Check if UV mapping is available
            if self.uvmap is not None:
                uvmap = self.uvmap.reshape(self.Ntriangles, 3, 2)  # Reshape to (M, 3, 2) for Open3D T geometry
                o3d_mesh.triangle.texture_uvs = open3d.core.Tensor(uvmap, dtype=open3d.core.float32)

        return o3d_mesh
    

    @classmethod
    def from_vtk(cls, filename: str) -> TriangleMesh3D:
        r"""
        Create a TriangleMesh3D instance from a VTK file.

        This method reads the mesh data from a VTK file and creates a TriangleMesh3D instance.
        This method uses the `meshio` library to read the VTK file and extract the vertices and triangles.

        .. code-block:: python

            from pyblenderSDIC.mesh import TriangleMesh3D

            # Create a TriangleMesh3D instance from a VTK file
            mesh = TriangleMesh3D.from_vtk("path/to/mesh.vtk")

        .. seealso::

            :meth:`from_meshio` for creating a TriangleMesh3D instance from a meshio object and more details about the meshio structure.

        Parameters
        ----------
        filename : str
            The path to the VTK file.

        Returns
        -------
        TriangleMesh3D
            A TriangleMesh3D instance created from the VTK file.
        """
        if not os.path.isfile(filename):
            raise FileNotFoundError(f"VTK file '{filename}' does not exist.")
        
        mesh = meshio.read(filename)
        return cls.from_meshio(mesh)
    

    def to_vtk(self, filename: str) -> None:
        r"""
        Save the TriangleMesh3D instance to a VTK file.

        This method writes the mesh data to a VTK file.
        It uses the `meshio` library to write the mesh data, including vertices, triangles, and UV map (if available).

        .. code-block:: python

            from pyblenderSDIC.mesh import TriangleMesh3D

            # Create a TriangleMesh3D instance
            mesh = TriangleMesh3D(vertices=..., triangles=...)
            # Save the mesh to a VTK file
            mesh.to_vtk("path/to/mesh.vtk")

        .. note::

            If the filename does not end with ".vtk", the extension will be added automatically.

        .. seealso::

            :meth:`to_meshio` for converting the TriangleMesh3D instance to a meshio object and more details about the meshio structure.

        Parameters
        ----------
        filename : str
            The path to the VTK file where the mesh will be saved.
        """
        meshio_mesh = self.to_meshio()
        meshio_mesh.write(filename, file_format="vtk")

    
    @classmethod
    def from_dict(self, data: Dict[str, Any]) -> TriangleMesh3D:
        r"""
        Create a TriangleMesh3D instance from a dictionary.

        The dictionary should contain the following keys:

        - "vertices": A list of lists representing the coordinates of the mesh vertices.
        - "triangles": A list of lists representing the indices of the vertices that form each triangle.
        - "uvmap": Optional; A list of lists representing the UV coordinates for each triangle in the mesh.
        - "vertex_normals": Optional; A list of lists representing the normals of the vertices in the mesh.
        - "triangle_normals": Optional; A list of lists representing the normals of the triangles in the mesh.
        - "triangle_centroids": Optional; A list of lists representing the centroids of the triangles in the mesh.
        - "triangle_areas": Optional; A list representing the areas of the triangles in the mesh.
        - "bounding_box": Optional; A list of two lists representing the bounding box of the mesh, where the first list contains the minimum coordinates and the second list contains the maximum coordinates in each dimension.
        - "volume": Optional; A float representing the volume of the mesh.
        
        .. code-block:: python

            from pyblenderSDIC.mesh import TriangleMesh3D

            # Create a TriangleMesh3D instance from a dictionary
            data = {
                "vertices": [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
                "triangles": [[0, 1, 2]],
                "uvmap": [[0, 0, 1, 0, 0, 1]],
                "vertex_normals": [[0, 0, 1], [0, 0, 1], [0, 0, 1]],
                "triangle_normals": [[0, 0, 1]],
                "triangle_centroids": [[0.333, 0.333, 0]],
                "triangle_areas": [0.5],
                "bounding_box": [[0, 0, 0], [1, 1, 0]],
                "volume": None
            }
            mesh = TriangleMesh3D.from_dict(data)

        .. note::

            The values of the dictionary should be convertible to NumPy arrays.
            They are expected by default to be lists of lists in order to be readable from JSON or similar formats.

        Parameters
        ----------
        data : Dict[str, Any]
            A dictionary containing the mesh data.
        
        Returns
        -------
        TriangleMesh3D
            A TriangleMesh3D instance created from the dictionary.
        """
        if not isinstance(data, dict):
            raise TypeError(f"Expected a dictionary, got {type(data)}.")
        
        required_keys = ["vertices", "triangles"]
        for key in required_keys:
            if key not in data:
                raise KeyError(f"Missing required key '{key}' in the input dictionary.")
            
        vertices = numpy.asarray(data["vertices"], dtype=numpy.float64)
        triangles = numpy.asarray(data["triangles"], dtype=numpy.int64)
        mesh_instance = TriangleMesh3D(vertices=vertices, triangles=triangles)
        mesh_instance.validate()  # Validate the mesh structure

        # Set optional attributes if they exist in the dictionary
        mesh_instance.uvmap = data.get("uvmap", None)
        mesh_instance.vertex_normals = data.get("vertex_normals", None)
        mesh_instance.triangle_normals = data.get("triangle_normals", None)
        mesh_instance.triangle_centroids = data.get("triangle_centroids", None)
        mesh_instance.triangle_areas = data.get("triangle_areas", None)
        mesh_instance.bounding_box = data.get("bounding_box", None)
        mesh_instance.volume = data.get("volume", None)

        return mesh_instance
    

    def to_dict(self) -> Dict[str, Any]:
        r"""
        Convert the TriangleMesh3D instance to a dictionary.

        .. code-block:: python

            from pyblenderSDIC.mesh import TriangleMesh3D

            # Create a TriangleMesh3D instance
            mesh = TriangleMesh3D(vertices=..., triangles=...)
            # Convert the mesh to a dictionary
            data = mesh.to_dict()

        .. seealso::

            :meth:`from_dict` for more details about the return dictionary structure.

        .. note::

            The values of the dictionary are lists of lists or scalars, so they can be easily serialized to JSON or similar formats.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the mesh data.
        """
        data = {
            "vertices": self.vertices.tolist(),
            "triangles": self.triangles.tolist(),
        }
        if self.uvmap is not None:
            data["uvmap"] = self.uvmap.tolist()
        if self.vertex_normals is not None:
            data["vertex_normals"] = self.vertex_normals.tolist()
        if self.triangle_normals is not None:
            data["triangle_normals"] = self.triangle_normals.tolist()
        if self.triangle_centroids is not None:
            data["triangle_centroids"] = self.triangle_centroids.tolist()
        if self.triangle_areas is not None:
            data["triangle_areas"] = self.triangle_areas.tolist()
        if self.bounding_box is not None:
            data["bounding_box"] = self.bounding_box.tolist()
        if self.volume is not None:
            data["volume"] = float(self.volume)
        return data
    

    @classmethod
    def from_json(cls, filename: str) -> TriangleMesh3D:
        r"""
        Create a TriangleMesh3D instance from a JSON file.

        The JSON file should contain the mesh data in the same format as the dictionary expected by the `from_dict` method.

        .. code-block:: python

            from pyblenderSDIC.mesh import TriangleMesh3D

            # Create a TriangleMesh3D instance from a JSON file
            mesh = TriangleMesh3D.from_json("path/to/mesh.json")

        Parameters
        ----------
        filename : str
            The path to the JSON file.

        Returns
        -------
        TriangleMesh3D
            A TriangleMesh3D instance created from the JSON file.
        """
        if not os.path.isfile(filename):
            raise FileNotFoundError(f"JSON file '{filename}' does not exist.")
        
        with open(filename, 'r') as f:
            data = json.load(f)
        
        return cls.from_dict(data)
    

    def to_json(self, filename: str) -> None:
        r"""
        Save the TriangleMesh3D instance to a JSON file.

        The JSON file will contain the mesh data in the same format as the dictionary returned by the `to_dict` method.

        .. code-block:: python

            from pyblenderSDIC.mesh import TriangleMesh3D

            # Create a TriangleMesh3D instance
            mesh = TriangleMesh3D(vertices=..., triangles=...)
            # Save the mesh to a JSON file
            mesh.to_json("path/to/mesh.json")

        Parameters
        ----------
        filename : str
            The path to the JSON file where the mesh will be saved.
        """
        data = self.to_dict()
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)


    # =======================================================================
    # Properties with Open3D compatibility
    # =======================================================================
    @property
    def is_edge_manifold(self) -> bool:
        r"""
        Get the edge manifold property of the mesh.

        This property indicates whether the mesh is edge manifold, meaning that each edge is shared by at most two triangles.

        .. note::

            After each modification of the mesh (e.g., vertices or triangles), this property should be recomputed using Open3D (automatically done by the Open3D wrapper).
            In fact, the computation of this property is higher the first time it is accessed after a modification.

        Returns
        -------
        bool
            True if the mesh is edge manifold, False otherwise.
        """
        if self._is_edge_manifold is None:
            # Compute the edge manifold property using Open3D
            o3d_mesh = self.to_open3d(legacy=True)
            self._is_edge_manifold = o3d_mesh.is_edge_manifold(allow_boundary_edges=False)
        # Return the cached value
        return self._is_edge_manifold

    @property
    def is_edge_manifold_with_boundary(self) -> bool:
        r"""
        Get the edge manifold with boundary property of the mesh.

        This property indicates whether the mesh is edge manifold, meaning that each edge is shared by at most two triangles,
        allowing for boundary edges.

        .. note::

            After each modification of the mesh (e.g., vertices or triangles), this property should be recomputed using Open3D (automatically done by the Open3D wrapper).
            In fact, the computation of this property is higher the first time it is accessed after a modification.

        Returns
        -------
        bool
            True if the mesh is edge manifold with boundary, False otherwise.
        """
        if self._is_edge_manifold_with_boundary is None:
            # Compute the edge manifold with boundary property using Open3D
            o3d_mesh = self.to_open3d(legacy=True)
            self._is_edge_manifold_with_boundary = o3d_mesh.is_edge_manifold(allow_boundary_edges=True)
        # Return the cached value
        return self._is_edge_manifold_with_boundary
    
    @property
    def is_vertex_manifold(self) -> bool:
        r"""
        Get the vertex manifold property of the mesh.

        This property indicates whether the mesh is vertex manifold, meaning that each vertex is shared by at most two triangles.

        .. note::

            After each modification of the mesh (e.g., vertices or triangles), this property should be recomputed using Open3D (automatically done by the Open3D wrapper).
            In fact, the computation of this property is higher the first time it is accessed after a modification.

        Returns
        -------
        bool
            True if the mesh is vertex manifold, False otherwise.
        """
        if self._is_vertex_manifold is None:
            # Compute the vertex manifold property using Open3D
            o3d_mesh = self.to_open3d(legacy=True)
            self._is_vertex_manifold = o3d_mesh.is_vertex_manifold()
        # Return the cached value
        return self._is_vertex_manifold
    
    @property
    def is_self_intersecting(self) -> bool:
        r"""
        Get the self-intersecting property of the mesh.

        This property indicates whether the mesh is self-intersecting, meaning that it has overlapping triangles.

        .. note::

            After each modification of the mesh (e.g., vertices or triangles), this property should be recomputed using Open3D (automatically done by the Open3D wrapper).
            In fact, the computation of this property is higher the first time it is accessed after a modification.

        Returns
        -------
        bool
            True if the mesh is self-intersecting, False otherwise.
        """
        if self._is_self_intersecting is None:
            # Compute the self-intersecting property using Open3D
            o3d_mesh = self.to_open3d(legacy=True)
            self._is_self_intersecting = o3d_mesh.is_self_intersecting()
        # Return the cached value
        return self._is_self_intersecting
    
    @property
    def is_watertight(self) -> bool:
        r"""
        Get the watertight property of the mesh.

        This property indicates whether the mesh is watertight, meaning that it has no holes and is a closed surface.

        .. note::

            After each modification of the mesh (e.g., vertices or triangles), this property should be recomputed using Open3D (automatically done by the Open3D wrapper).
            In fact, the computation of this property is higher the first time it is accessed after a modification.

        Returns
        -------
        bool
            True if the mesh is watertight, False otherwise.
        """
        if self._is_watertight is None:
            # Compute the watertight property using Open3D
            o3d_mesh = self.to_open3d(legacy=True)
            self._is_watertight = o3d_mesh.is_watertight()
        # Return the cached value
        return self._is_watertight
    
    @property
    def is_orientable(self) -> bool:
        r"""
        Get the orientable property of the mesh.

        This property indicates whether the mesh is orientable, meaning that it has a consistent orientation across its surface.

        .. note::

            After each modification of the mesh (e.g., vertices or triangles), this property should be recomputed using Open3D (automatically done by the Open3D wrapper).
            In fact, the computation of this property is higher the first time it is accessed after a modification.

        Returns
        -------
        bool
            True if the mesh is orientable, False otherwise.
        """
        if self._is_orientable is None:
            # Compute the orientable property using Open3D
            o3d_mesh = self.to_open3d(legacy=True)
            self._is_orientable = o3d_mesh.is_orientable()
        # Return the cached value
        return self._is_orientable
    

    # =======================================================================
    # Public Methods
    # =======================================================================
    def set_vertices_uvmap(self, value: Optional[numpy.ndarray]) -> None:
        r"""
        Set the UV mapping coordinates based on the vertices.

        For each triangle in the mesh, this method set the UV mapping coordinates
        based on the vertex UV mapping.

        The UV coordinates are expected to be a NumPy array of shape (N, 2),
        where N is the number of vertices and each row contains the UV coordinates in the format (u, v).

        Parameters
        ----------
        value : Optional[numpy.ndarray]
            A NumPy array of shape (N, 2) representing the UV coordinates for each vertex in the mesh.
            Each row should contain the UV coordinates in the format (u, v).
            If None, the mesh will not have UV mapping.

        Raises
        -------
        ValueError
            If the shape of the value is not (N, 2).
        """
        if value is None:
            self.uvmap = None
            return
        
        value = numpy.asarray(value, dtype=numpy.float64)
        if not value.ndim == 2 or not value.shape == (self.Nvertices, 2):
            raise ValueError(f"UV coordinates (from vertices) must have shape ({self.Nvertices}, 2), got {value.shape}.")
        
        # Create the UV map for each triangle based on the vertices UV coordinates
        uvmap = numpy.zeros((self.Ntriangles, 6), dtype=numpy.float64)
        for index_triangle in range(self.Ntriangles):
            for index_vertex in range(3):
                index_point = self.triangles[index_triangle, index_vertex]
                uvmap[index_triangle, 2*index_vertex:2*(index_vertex+1)] = value[index_point, :] # (u, v) coordinates for each vertex of the triangle
        self.uvmap = uvmap
    

    def compute_bounding_box(self) -> None:
        r"""
        Compute the bounding box of the mesh.

        The bounding box is computed as the minimum and maximum coordinates of the vertices in each dimension.
        The result is stored in the ``bounding_box`` attribute as a NumPy array of shape (2, 3).

        The axis of the bounding box are ordered along the X, Y, Z dimensions.

        .. code-block:: python

            from pyblenderSDIC.mesh import TriangleMesh3D

            # Create a TriangleMesh3D instance
            mesh = TriangleMesh3D(vertices=..., triangles=...)

            # Compute the bounding box of the mesh
            mesh.compute_bounding_box()
        
            # The bounding box is now available in the `bounding_box` attribute
            mesh.bounding_box

        """
        min_coords = numpy.min(self.vertices, axis=0)
        max_coords = numpy.max(self.vertices, axis=0)
        self.bounding_box = numpy.array([min_coords, max_coords], dtype=numpy.float64)


    def compute_volume(self) -> None:
        r"""
        Compute the volume of the mesh.

        The volume is computed as the sum of the signed volumes of each triangle.
        The result is stored in the ``volume`` attribute as a float.

        The volume is compute by using the Green theorem for polyhedra.
        For each triangle, the signed volume is computed as:

        .. math::

            V = \frac{1}{6} P_1 \cdot (P_2 \times P_3)

        where :math:`P_1`, :math:`P_2`, and :math:`P_3` are the vertices of the triangle and :math:`\cdot` and :math:`\times` are the dot and cross products, respectively.
        The total volume is the sum of the absolute values of the signed volumes of all triangles.

        .. code-block:: python

            from pyblenderSDIC.mesh import TriangleMesh3D
            
            # Create a TriangleMesh3D instance
            mesh = TriangleMesh3D(vertices=..., triangles=...)

            # Compute the volume of the mesh
            mesh.compute_volume()

            # The volume is now available in the `volume` attribute
            mesh.volume

        .. warning::

            The volume can only be computed if the mesh ``is_watertight``.

            The volume is given as a signed value, which can be negative if the mesh is not oriented correctly.

        """
        if not self.is_watertight:
            raise ValueError("The mesh is not watertight, volume cannot be computed.")
        
        # Extract the vertices of the triangles
        p1 = self.vertices[self.triangles[:, 0]]
        p2 = self.vertices[self.triangles[:, 1]]
        p3 = self.vertices[self.triangles[:, 2]]

        # Vectorized computation of signed volumes
        cross_product = numpy.cross(p2, p3)  # Cross product p2 x p3
        signed_volumes = numpy.einsum('ij,ij->i', p1, cross_product) / 6.0  # Dot product p1 . (p2 x p3)

        # Sum of signed volumes
        self.volume = numpy.sum(signed_volumes)

    
    def compute_triangle_normals(self) -> None:
        r"""
        Compute the normals of the triangles in the mesh.

        The normals are computed as the cross product of the edges of each triangle.
        The result is stored in the ``triangle_normals`` attribute as a NumPy array of shape (M, 3),
        where M is the number of triangles and each row contains the normal vector of a triangle.

        The normals are normalized to have unit length.

        .. math::

            N = \frac{(P_2 - P_1) \times (P_3 - P_1)}{\| (P_2 - P_1) \times (P_3 - P_1) \|}

        where :math:`P_1`, :math:`P_2`, and :math:`P_3` are the vertices of the triangle, and :math:`\times` is the cross product.

        .. code-block:: python

            from pyblenderSDIC.mesh import TriangleMesh3D
            
            # Create a TriangleMesh3D instance
            mesh = TriangleMesh3D(vertices=..., triangles=...)

            # Compute the normals of the triangles
            mesh.compute_triangle_normals()

            # The normals are now available in the `triangle_normals` attribute
            mesh.triangle_normals

        """
        if self.triangles.size == 0:
            raise ValueError("The mesh has no triangles, cannot compute normals.")
        
        # Extract the vertices of the triangles
        p1 = self.vertices[self.triangles[:, 0]]
        p2 = self.vertices[self.triangles[:, 1]]
        p3 = self.vertices[self.triangles[:, 2]]

        # Vectorized computation of triangle normals
        edge1 = p2 - p1
        edge2 = p3 - p1
        normals = numpy.cross(edge1, edge2)

        # Normalize the normals
        norms = numpy.linalg.norm(normals, axis=1)
        norms[norms <= 1e-10] = 1.0
        self.triangle_normals = normals / norms[:, numpy.newaxis]


    def compute_triangle_centroids(self) -> None:
        r"""
        Compute the centroids of the triangles in the mesh.

        The centroids are computed as the average of the vertices of each triangle.
        The result is stored in the ``triangle_centroids`` attribute as a NumPy array of shape (M, 3),
        where M is the number of triangles and each row contains the centroid coordinates of a triangle.

        .. math::

            C = \frac{P_1 + P_2 + P_3}{3}

        where :math:`P_1`, :math:`P_2`, and :math:`P_3` are the vertices of the triangle.

        .. code-block:: python

            from pyblenderSDIC.mesh import TriangleMesh3D
            
            # Create a TriangleMesh3D instance
            mesh = TriangleMesh3D(vertices=..., triangles=...)

            # Compute the centroids of the triangles
            mesh.compute_triangle_centroids()

            # The centroids are now available in the `triangle_centroids` attribute
            mesh.triangle_centroids

        """
        if self.triangles.size == 0:
            raise ValueError("The mesh has no triangles, cannot compute centroids.")
        
        # Extract the vertices of the triangles
        p1 = self.vertices[self.triangles[:, 0]]
        p2 = self.vertices[self.triangles[:, 1]]
        p3 = self.vertices[self.triangles[:, 2]]

        # Vectorized computation of triangle centroids
        centroids = (p1 + p2 + p3) / 3.0
        self.triangle_centroids = centroids


    def compute_triangle_areas(self) -> None:
        r"""
        Compute the areas of the triangles in the mesh.

        The areas are computed using the cross product of the edges of each triangle.
        The result is stored in the ``triangle_areas`` attribute as a NumPy array of shape (M,),
        where M is the number of triangles and each triangle contains the area of a triangle.

        .. math::

            A = \frac{\| (P_2 - P_1) \times (P_3 - P_1) \|}{2}

        where :math:`P_1`, :math:`P_2`, and :math:`P_3` are the vertices of the triangle, :math:`\times` is the cross product, and :math:`\| \cdot \|` is the norm.

        .. code-block:: python

            from pyblenderSDIC.mesh import TriangleMesh3D
            
            # Create a TriangleMesh3D instance
            mesh = TriangleMesh3D(vertices=..., triangles=...)

            # Compute the areas of the triangles
            mesh.compute_triangle_areas()

            # The areas are now available in the `triangle_areas` attribute
            mesh.triangle_areas

        """
        if self.triangles.size == 0:
            raise ValueError("The mesh has no triangles, cannot compute areas.")
        
        # Extract the vertices of the triangles
        p1 = self.vertices[self.triangles[:, 0]]
        p2 = self.vertices[self.triangles[:, 1]]
        p3 = self.vertices[self.triangles[:, 2]]

        # Vectorized computation of triangle areas
        edge1 = p2 - p1
        edge2 = p3 - p1
        areas = numpy.linalg.norm(numpy.cross(edge1, edge2), axis=1) / 2.0
        self.triangle_areas = areas


    def compute_vertex_normals(self) -> None:
        r"""
        Compute the normals of the vertices in the mesh.

        The normals are computed as the average of the normals of the triangles that share each vertex.
        The result is stored in the ``vertex_normals`` attribute as a NumPy array of shape (N, 3),
        where N is the number of vertices and each row contains the normal vector of a vertex.

        The normals are normalized to have unit length.

        .. code-block:: python

            from pyblenderSDIC.mesh import TriangleMesh3D
            
            # Create a TriangleMesh3D instance
            mesh = TriangleMesh3D(vertices=..., triangles=...)

            # Compute the normals of the vertices
            mesh.compute_vertex_normals()

            # The normals are now available in the `vertex_normals` attribute
            mesh.vertex_normals

        .. note::

            The normals of triangles must be computed before calling this method.

        """
        if self.triangle_normals is None:
            raise ValueError("Triangle normals must be computed before computing vertex normals.")
        
        # Initialize vertex normals with zeros
        self.vertex_normals = numpy.zeros_like(self.vertices)

        # Accumulate triangle normals into vertex normals
        for i in range(3):  # Loop over each vertex of the triangle
            numpy.add.at(self.vertex_normals, self.triangles[:, i], self.triangle_normals)

        # Normalize the vertex normals
        norms = numpy.linalg.norm(self.vertex_normals, axis=1)
        norms[norms <= 1e-10] = 1.0  # Avoid division by zero
        self.vertex_normals /= norms[:, numpy.newaxis]

    
    def open3d_cast_rays(self, rays: numpy.ndarray) -> Dict:
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
            ray_intersect = trimesh3d.open3d_cast_rays(rays)

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
        o3d_mesh = self.to_open3d(legacy=False)

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


    def cast_rays(self, rays: numpy.ndarray) -> IntersectPoints:
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
            intersect_points = trimesh3d.cast_rays(rays)

        .. note::

            The returned :class:`IntersectPoints` contains:

            - ``uv``: A (..., 2) array of barycentric coordinates (u, v). If a ray misses the mesh, the entry is [nan, nan].
            - ``id``: A (...) array of triangle indices hit by each ray. If a ray misses, the index is set to -1.

            The barycentric coordinates are such that:

                coordinates = (1 - u - v) * P_1 + u * P_2 + v * P_3

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
        results = self.open3d_cast_rays(rays)

        # Initialize the output arrays
        rays = numpy.asarray(rays, dtype=numpy.float64)
        barycentric_coords = numpy.full((*rays.shape[:-1], 2), numpy.nan, dtype=numpy.float64)
        triangle_indices = numpy.full(*rays.shape[:-1], -1, dtype=int)

        # Extract the intersection points
        intersect_true = results["t_hit"].isfinite().numpy()
        barycentric_coords[intersect_true] = results["primitive_uvs"].numpy().astype(numpy.float64)[intersect_true]
        triangle_indices[intersect_true] = results["primitive_ids"].numpy().astype(int)[intersect_true]

        # Construct the output
        intersect_points = IntersectPoints(barycentric_coords, triangle_indices)

        return intersect_points
    

    def calculate_intersect_shape_functions(self, intersect_points: IntersectPoints) -> numpy.ndarray:
        r"""
        Compute the shape function values at intersection points.

        This method computes the shape function values at the intersection points
        using the barycentric coordinates and the triangle indices contained in the given
        :class:`IntersectPoints` object.

        The shape function values are computed as follows:

        .. math::

            S = \begin{bmatrix}
            1 - u - v \\
            u \\
            v
            \end{bmatrix}

        Where :math:`u` and :math:`v` are the barycentric coordinates of the intersection points.

        Parameters
        ----------
        intersect_points : IntersectPoints
            An object containing barycentric coordinates and triangle indices.

        Returns
        -------
        numpy.ndarray
            An array of shape (..., 3) containing the shape function values at the intersection points.
            Points with no intersection are returned as [nan, nan, nan].
        """
        # Check the shape of the intersect_points
        if not isinstance(intersect_points, IntersectPoints):
            raise ValueError("intersect_points must be an instance of IntersectPoints.")
        
        # Flatten everything to 1D
        flat_bary = intersect_points.uv.reshape(-1, 2) # Alias of intersect_points.barycentric_coordinates
        flat_idx = intersect_points.id.reshape(-1) # Alias of intersect_points.triangle_indices

        # Initialize output
        flat_shape_function = numpy.full((flat_idx.shape[0], 3), numpy.nan, dtype=numpy.float64)

        # Filter valid hits
        valid = flat_idx >= 0
        u = flat_bary[valid, 0]
        v = flat_bary[valid, 1]
        w = 1.0 - u - v

        # Construct the shape function values
        flat_shape_function[valid] = numpy.column_stack((w, u, v))

        # Reshape to original shape
        output_shape = (*intersect_points.uv.shape[:-1], 3)
        return flat_shape_function.reshape(output_shape)

    
    def calculate_intersect_coordinates(self, intersect_points: IntersectPoints) -> numpy.ndarray:
        r"""
        Compute the 3D coordinates of intersection points from barycentric data.

        This method reconstructs the 3D position of the intersection points using the barycentric
        coordinates and the triangle indices contained in the given :class:`IntersectPoints` object.

        .. code-block:: python

            intersect_points = trimesh3d.cast_rays(rays)
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
        flat_idx = intersect_points.id.reshape(-1) # Alias of intersect_points.triangle_indices

        # Initialize output
        flat_points = numpy.full((flat_idx.shape[0], 3), numpy.nan, dtype=numpy.float64)

        # Filter valid hits
        valid = flat_idx >= 0
        valid_idx = flat_idx[valid]
        u = flat_bary[valid, 0]
        v = flat_bary[valid, 1]
        w = 1.0 - u - v

        # Get triangle vertices
        
        A = self.vertices[self.triangles[valid_idx, 0]]
        B = self.vertices[self.triangles[valid_idx, 1]]
        C = self.vertices[self.triangles[valid_idx, 2]]

        # Compute coordinates
        flat_points[valid] = w[:, None] * A + u[:, None] * B + v[:, None] * C

        # Reshape to original shape
        output_shape = (*intersect_points.uv.shape[:-1], 3)
        return flat_points.reshape(output_shape)


    # =======================================================================
    # Visualization Methods
    # =======================================================================
    def visualize(self, 
                  pattern_path: Optional[str] = None,
                  highlighted_triangles: Optional[Union[Integral, Sequence[Integral]]] = None,
                  highlight_color: Sequence[float] = [0.5, 0.5, 0.5],
                  intersect_points: Optional[IntersectPoints] = None,
                  intersect_color: Sequence[float] = [0.0, 0.0, 1.0],
                  display_edges: bool = True,
                  edges_color: Sequence[float] = [0.2, 0.2, 0.2],
                ) -> None:
        r"""
        Visualize the mesh using Open3D.

        This method displays the 3D mesh using Open3D's interactive viewer.
        Optionally, it can highlight specific mesh triangles in and show 3D intersection points.
        Furthermore, it can visualize the mesh with a pattern if provided.

        .. figure:: ../../../../pyblenderSDIC/resources/doc/trimesh3d_visualize.png
            :width: 400
            :align: center

            Example of a mesh with highlighted triangles and intersection vertices.

        .. code-block:: python

            from pyblenderSDIC.mesh import TriangleMesh3D

            # Create a TriangleMesh3D instance
            mesh = TriangleMesh3D(vertices=..., triangles=...)

            # Visualize the mesh with highlighted triangles and intersection vertices
            mesh.visualize(pattern_path="path/to/pattern.png", highlighted_triangles=[0, 1, 2], intersect_points=intersect_points)

        Parameters
        ----------
        pattern_path : Optional[str], optional
            The path to the texture pattern image file to be applied to the mesh.
            If None, no texture is applied. Default is None.

        highlighted_triangles : Optional[Union[Integral, Sequence[Integral]]], optional
            The indices of the triangles to be highlighted in the mesh.
            If a single integer is provided, it highlights that triangle.
            If a sequence of integers is provided, it highlights all specified triangles.
            If None, no triangles are highlighted. Default is None.
        
        highlight_color : Sequence[float], optional
            The RGB color to use for highlighting the specified triangles.
            Each value should be in the range [0, 1]. Default is [0.5, 0.5, 0.5].
        
        intersect_points : Optional[IntersectPoints], optional
            An IntersectPoints object containing the 3D intersection points to be visualized.
            If None, no intersection points are visualized. Default is None.

        intersect_color : Sequence[float], optional
            The RGB color to use for the intersection vertices.
            Each value should be in the range [0, 1]. Default is [0.0, 0.0, 1.0].

        display_edges : bool, optional
            If True, the edges of the mesh will be displayed. Default is True.
        
        edges_color : Sequence[float], optional
            The RGB color to use for the edges of the mesh.
            Each value should be in the range [0, 1]. Default is [0.2, 0.2, 0.2].

        """
        # Validate the input parameters
        if pattern_path is not None and not os.path.isfile(pattern_path):
            raise FileNotFoundError(f"Pattern file '{pattern_path}' does not exist.")
        if not isinstance(highlight_color, Sequence) or len(highlight_color) != 3:
            raise ValueError("highlight_color must be a sequence of three float values representing RGB color.")
        if not isinstance(intersect_color, Sequence) or len(intersect_color) != 3:
            raise ValueError("intersect_color must be a sequence of three float values representing RGB color.")
        if not isinstance(display_edges, bool):
            raise ValueError("display_edges must be a boolean value.")
        if not isinstance(edges_color, Sequence) or len(edges_color) != 3:
            raise ValueError("edges_color must be a sequence of three float values representing RGB color.")

        # Create the geometry
        geometries = []

        # Extracted the Open3D mesh
        mesh = self.to_open3d(legacy=False)
        mesh.compute_vertex_normals()  # Compute vertex normals for better visualization

        # Check if a pattern is provided
        if pattern_path is not None:
            material = open3d.visualization.rendering.MaterialRecord()
            material.shader = 'defaultUnlit'
            material.albedo_img = open3d.io.read_image('pattern.png')

        # Extracted the triangles to be colored
        if highlighted_triangles is None:
            highlighted_triangles = []
        elif isinstance(highlighted_triangles, Integral):
            highlighted_triangles = [highlighted_triangles]
        highlighted_triangles = numpy.asarray(highlighted_triangles, dtype=numpy.int64).flatten()
        highlighted_triangles = numpy.unique(highlighted_triangles)

        if highlighted_triangles.size !=0 and (not numpy.all(0 <= highlighted_triangles) or not numpy.all(highlighted_triangles < self.Ntriangles)):
            raise ValueError("highlighted_triangles must be valid triangle indices.")
        
        indices = numpy.arange(self.Ntriangles)
        colors = numpy.full((self.Ntriangles, 3), [0.5, 0.5, 0.5]) # Default color for triangles (gray)
        colors[numpy.isin(indices, highlighted_triangles)] = list(highlight_color)  # Highlighted triangles color
        mesh.triangle.colors = open3d.core.Tensor(colors, open3d.core.float32)

        # Add the mesh to the geometries
        geometries.append({
            "name": "Colored Elements",
            "geometry": mesh,
            "material": material if pattern_path is not None else None,
        })

        # Extract the edges of the mesh
        if display_edges:
            triangles = numpy.asarray(mesh.triangle.indices.numpy(), dtype=int)
            lines = numpy.zeros((3*self.Ntriangles, 2), dtype=int)
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
            lineset.line.colors = open3d.core.Tensor(list(edges_color), open3d.core.float32)

            # Add the lineset to the geometries
            geometries.append({
                "geometry": lineset,
                "name": "Mesh Edges"
        })

        # Create PointCloud for intersection vertices
        if intersect_points is not None:
            vertices = self.calculate_intersect_coordinates(intersect_points)
            point_cloud = open3d.t.geometry.PointCloud()
            point_cloud.point.positions = open3d.core.Tensor(vertices, dtype=open3d.core.Dtype.Float32)
            point_cloud.point.colors = open3d.core.Tensor(numpy.tile(intersect_color, (vertices.shape[0], 1)), dtype=open3d.core.Dtype.Float32)  # Blue color for vertices
            
            # Add the vertex cloud to the geometries
            geometries.append({
                "geometry": point_cloud,
                "name": "Intersection Points"
            })
            
        # Launch Open3D viewer
        open3d.visualization.draw(geometries, point_size=15)

        

        