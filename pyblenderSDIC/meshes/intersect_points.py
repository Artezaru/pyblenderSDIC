from __future__ import annotations
import numpy


class IntersectPoints:
    """
    A class to represent the intersection points of rays with a 3D mesh.

    This class stores the barycentric coordinates and element indices of the
    intersection points. The barycentric coordinates are used to locate the
    position of the intersection within a triangle, and the element indices
    represent the specific triangle in the mesh that was intersected by a ray.

    For a triangle with vertices A, B, and C, the barycentric coordinates (u, v) are
    defined as follows:

    .. math::

        P = (1 - u - v) A + u B + v C

    Where P is the intersection point, and u and v are the barycentric coordinates between
    0 and 1. Their sum is always less than or equal to 1. The barycentric coordinates

    .. note::

        If no intersection occurs, the barycentric coordinates are set to NaN and
        the element index is set to -1.

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
    
    element_indices : numpy.ndarray
        A ND array of shape (...,), where each entry represents the index of the
        triangle that was intersected. If no intersection occurs, the index is -1.
    """

    def __init__(self, barycentric_coordinates: numpy.ndarray, element_indices: numpy.ndarray) -> None:
        self.barycentric_coordinates = barycentric_coordinates
        self.element_indices = element_indices

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
        barycentric_coords = numpy.asarray(self._barycentric_coordinates, dtype=numpy.float64)
        if barycentric_coords.ndim <= 1:
            raise ValueError("[INTERNAL CLASS ERROR]: Barycentric coordinates must have at least 2 dimension.")
        if barycentric_coords.shape[-1] != 2:
            raise ValueError("[INTERNAL CLASS ERROR]: Barycentric coordinates must have shape (..., 2).")
        return barycentric_coords

    @barycentric_coordinates.setter
    def barycentric_coordinates(self, value: numpy.ndarray) -> None:
        value = numpy.asarray(value, dtype=numpy.float64)
        if value.ndim <= 1:
            raise ValueError("Barycentric coordinates must have at least 2 dimension.")
        if value.shape[-1] != 2:
            raise ValueError("Barycentric coordinates must have shape (..., 2).")
        self._barycentric_coordinates = value

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
    def element_indices(self) -> numpy.ndarray:
        """
        Gets the element indices of the intersections.

        .. note::

            An alias for the element indices is ``id``.

        Returns
        -------
        numpy.ndarray
            A 1D array representing the element indices of the intersections.
        """
        element_indices = numpy.asarray(self._element_indices, dtype=int)
        if element_indices.ndim < 1:
            raise ValueError("[INTERNAL CLASS ERROR]: Element indices must have at least 1 dimension.")
        return element_indices

    @element_indices.setter
    def element_indices(self, value: numpy.ndarray) -> None:
        value = numpy.asarray(value, dtype=int)
        if value.ndim < 1:
            raise ValueError("Element indices must have at least 1 dimension.")
        self._element_indices = value

    @property
    def id(self) -> numpy.ndarray:
        """
        Alias for the element indices.
        """
        return self.element_indices
    
    @id.setter
    def id(self, value: numpy.ndarray) -> None:
        self.element_indices = value

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
        # Vérification si les barycentriques sont finis (pas NaN ni infini)
        return numpy.logical_not(numpy.isnan(self.barycentric_coordinates).any(axis=-1))

    def validate(self) -> None:
        """
        Validates the input data to ensure the barycentric coordinates and element indices
        arrays have the correct shapes and values.

        Raises
        ------
        ValueError
            If the barycentric coordinates array does not have shape (..., 2),
            if element indices are not positive integers,
            or if barycentric coordinates do not meet the conditions.
        """
        # Check the shape of barycentric coordinates and element indices
        if self.barycentric_coordinates.ndim <= 1:
            raise ValueError("Barycentric coordinates must have at least 2 dimension.")
        if self.barycentric_coordinates.shape[-1] != 2:
            raise ValueError("Barycentric coordinates must have shape (..., 2).")
        if self.element_indices.ndim < 1:
            raise ValueError("Element indices must have at least 1 dimension.")
        if self.barycentric_coordinates.shape[:-1] != self.element_indices.shape:
            raise ValueError("The shapes of barycentric coordinates and element indices must match in the first dimensions.")

        valid_mask = self.valid_mask()

        # Check the barycentric coordinates (u, v): 0 <= u, v <= 1 and u + v <= 1
        u, v = self.uv[..., 0], self.uv[..., 1]
        if not numpy.all((u[valid_mask] >= 0) & (v[valid_mask] >= 0) & (u[valid_mask] + v[valid_mask] <= 1)):
            raise ValueError("Barycentric coordinates must satisfy 0 <= u, v <= 1 and u + v <= 1.")
        if not numpy.all((numpy.isnan(u[~valid_mask]) & (numpy.isnan(v[valid_mask])))):
            raise ValueError("Barycentric coordinates must be setted to NaN if no intersection occurs.")

        # Check the element indices: must be positive integers
        if not numpy.all(self.element_indices[valid_mask] >= 0):
            raise ValueError("Element indices must be positive integers.")
        if not numpy.all(self.element_indices[~valid_mask] == -1):
            raise ValueError("Element indices must be setted to -1 if no intersection occurs.")

    def filter_valid(self) -> IntersectPoints:
        """
        Filters out invalid intersections (where the element index is -1 or barycentric coordinates are NaN).

        This method modifies the shape of the arrays, changing them from the original
        (..., 2) and (...,) to (L', 2), where L' is the number of valid intersections.

        Returns
        -------
        IntersectPoints
            A new IntersectPoints object with only the valid intersections.
        """
        valid_mask = self.valid_mask()
        valid_barycentric_coords = self.barycentric_coordinates[valid_mask].reshape(-1, 2)
        valid_element_indices = self.element_indices[valid_mask].reshape(-1)

        # Reshaping to (L', 3) where L' is the number of valid intersections
        return IntersectPoints(valid_barycentric_coords, valid_element_indices)

