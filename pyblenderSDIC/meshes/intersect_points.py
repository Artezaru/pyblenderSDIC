from __future__ import annotations
import numpy


class IntersectPoints:
    """
    A class to represent the intersection points of rays with a 3D mesh.

    This class stores the barycentric coordinates and triangle indices of the
    intersection points. The barycentric coordinates are used to locate the
    position of the intersection within a triangle, and the triangle indices
    represent the specific triangle in the mesh that was intersected by a ray.

    For a triangle with vertices A, B, and C, the barycentric coordinates (u, v) are
    defined as follows:

    .. math::

        P = (1 - u - v) A + u B + v C

    Where P is the intersection point, and u and v are the barycentric coordinates between
    0 and 1. Their sum is always less than or equal to 1. The barycentric coordinates

    .. note::

        If no intersection occurs, the barycentric coordinates are set to NaN and
        the triangle index is set to -1.

    .. warning::

        The input arrays are wrapped with `numpy.asarray()`, meaning that the data
        is stored as a dynamic array that can be modified at any time. Therefore,
        users should be aware that the arrays can be changed after the object is created.

    Parameters
    ----------
    barycentric_coordinates : numpy.ndarray
        A (N+1)D array of shape (..., 2), where each entry represents the
        barycentric coordinates (u, v) of an intersection point within the
        corresponding triangle. If no intersection occurs, the coordinates are NaN.
    
    triangle_indices : numpy.ndarray
        A ND array of shape (...,), where each entry represents the index of the
        triangle that was intersected. If no intersection occurs, the index is -1.
    """

    def __init__(self, barycentric_coordinates: numpy.ndarray, triangle_indices: numpy.ndarray) -> None:
        # Active bypass mode for testing purposes
        self.__internal_bypass__ = True
        self.barycentric_coordinates = barycentric_coordinates
        self.triangle_indices = triangle_indices
        self.__internal_bypass__ = False
        self.__internal_check_barycentric_coordinates()
        self.__internal_check_triangle_indices()

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
    
    def __internal_check_barycentric_coordinates(self) -> None:
        r"""
        Internal method to check the validity of the barycentric coordinates array.
        """
        if self.__internal_bypass__:
            return
        
        if not isinstance(self._barycentric_coordinates, numpy.ndarray):
            raise TypeError(f"Barycentric coordinates must be a numpy.ndarray, got {type(self._barycentric_coordinates)}.")
        if not self._barycentric_coordinates.dtype == numpy.float64:
            raise TypeError(f"Barycentric coordinates must be of type float64, got {self._barycentric_coordinates.dtype}.")
        if not self._barycentric_coordinates.ndim >= 2:
            raise ValueError(f"Barycentric coordinates must have at least 2 dimensions, got {self._barycentric_coordinates.ndim} dimensions.")
        if not self._barycentric_coordinates.shape[-1] == 2:
            raise ValueError(f"Barycentric coordinates must have shape (..., 2), got {self._barycentric_coordinates.shape}.")
                
        valid_mask = self.valid_mask()
        u, v = self.uv[..., 0], self.uv[..., 1]

        if not numpy.all(numpy.isfinite(u[valid_mask]) & numpy.isfinite(v[valid_mask])):
            raise ValueError("Barycentric coordinates must contain finite values only.")
        if not numpy.all(numpy.isnan(u[~valid_mask]) & numpy.isnan(v[~valid_mask])):
            raise ValueError("Barycentric coordinates must be set to NaN if no intersection occurs.")
        if not numpy.all(u[valid_mask] >= 0) or not numpy.all(v[valid_mask] >= 0):
            raise ValueError("Barycentric coordinates must contain non-negative values only.")
        if not numpy.all(u[valid_mask] + v[valid_mask] <= 1):
            raise ValueError("Barycentric coordinates must satisfy u + v <= 1.")

        
    def __internal_check_triangle_indices(self) -> None:
        r"""
        Internal method to check the validity of the triangle indices array.
        """
        if self.__internal_bypass__:
            return
        
        if not isinstance(self._triangle_indices, numpy.ndarray):
            raise TypeError(f"Triangle indices must be a numpy.ndarray, got {type(self._triangle_indices)}.")
        if not self._triangle_indices.dtype == numpy.int64:
            raise TypeError(f"Triangle indices must be of type int, got {self._triangle_indices.dtype}.")
        if not self._triangle_indices.ndim >= 1:
            raise ValueError(f"Triangle indices must have at least 1 dimension, got {self._triangle_indices.ndim} dimensions.")
        
        valid_mask = self.valid_mask()
        
        if not numpy.all(numpy.isfinite(self._triangle_indices[valid_mask])):
            raise ValueError("Triangle indices must contain finite values only.")
        if not numpy.all(self._triangle_indices[valid_mask] >= 0):
            raise ValueError("Triangle indices must contain non-negative values only.")
        if not numpy.all(self._triangle_indices[~valid_mask] == -1):
            raise ValueError("Triangle indices must be set to -1 if no intersection occurs.")
        

    def validate(self) -> None:
        r"""
        Validate the mesh structure.

        This method checks the validity of the vertices, triangles, and UV map.
        If any of the checks fail, an exception is raised.
        """
        bypass_mode = self.__internal_bypass__
        self.__internal_bypass__ = False # Disable bypass mode for validation

        self.__internal_check_barycentric_coordinates()
        self.__internal_check_triangle_indices()
        
        # restore bypass mode
        self.__internal_bypass__ = bypass_mode

 
    # =======================================================================
    # Properties Getters and Setters
    # =======================================================================
    @property
    def barycentric_coordinates(self) -> numpy.ndarray:
        r"""
        Gets the barycentric coordinates of the intersections.

        The barycentric coordinates are represented as a 2D array of shape (..., 2),
        where each entry corresponds to the barycentric coordinates (u, v) of an intersection
        point within the corresponding triangle. The last dimension must be of size 2.

        For a triangle with vertices A, B, and C, the barycentric coordinates (u, v) are
        defined as follows:

        .. math::

            P = (1 - u - v) A + u B + v C

        .. note::

            An alias for the barycentric coordinates is ``uv``.

        Returns
        -------
        numpy.ndarray
            A 2D array of shape (..., 2) representing the barycentric coordinates of the intersections.
        """
        return self._barycentric_coordinates

    @barycentric_coordinates.setter
    def barycentric_coordinates(self, value: numpy.ndarray) -> None:
        self._barycentric_coordinates = numpy.asarray(value, dtype=numpy.float64)
        self.__internal_check_barycentric_coordinates()

    @property
    def uv(self) -> numpy.ndarray:
        """
        Alias for the barycentric coordinates.
        """
        return self.barycentric_coordinates
    
    @uv.setter
    def uv(self, value: numpy.ndarray) -> None:
        self.barycentric_coordinates = value

    @property
    def triangle_indices(self) -> numpy.ndarray:
        """
        Gets the triangle indices of the intersections.

        .. note::

            An alias for the triangle indices is ``id``.

        Returns
        -------
        numpy.ndarray
            A 1D array representing the triangle indices of the intersections.
        """
        return self._triangle_indices

    @triangle_indices.setter
    def triangle_indices(self, value: numpy.ndarray) -> None:
        self._triangle_indices = numpy.asarray(value, dtype=numpy.int64)
        self.__internal_check_triangle_indices()

    @property
    def id(self) -> numpy.ndarray:
        """
        Alias for the triangle indices.
        """
        return self.triangle_indices
    
    @id.setter
    def id(self, value: numpy.ndarray) -> None:
        self.triangle_indices = value

    # =======================================================================
    # Public Methods
    # =======================================================================
    def valid_mask(self) -> numpy.ndarray:
        """
        Returns a boolean mask indicating the validity of the barycentric coordinates.

        The mask is True if the barycentric coordinates (u, v) are no NaN values.
        The shape of the mask will be the same as the first dimensions of the barycentric coordinates array.

        Returns
        -------
        numpy.ndarray
            A boolean array of shape (...,) indicating the validity of the barycentric coordinates.
        """
        return numpy.logical_not(numpy.isnan(self.barycentric_coordinates).any(axis=-1))

    def filter_valid(self) -> IntersectPoints:
        """
        Filters out invalid intersections (where the triangle index is -1 or barycentric coordinates are NaN).

        This method modifies the shape of the arrays, changing them from the original
        (..., 2) and (...,) to (L', 2), where L' is the number of valid intersections.

        Returns
        -------
        IntersectPoints
            A new IntersectPoints object with only the valid intersections.
        """
        valid_mask = self.valid_mask()
        valid_barycentric_coords = self.barycentric_coordinates[valid_mask].reshape(-1, 2)
        valid_triangle_indices = self.triangle_indices[valid_mask].reshape(-1)

        # Reshaping to (L', 3) where L' is the number of valid intersections
        return IntersectPoints(valid_barycentric_coords, valid_triangle_indices)

