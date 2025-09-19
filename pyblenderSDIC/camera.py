from __future__ import annotations
from py3dframe import Frame, Rotation
from typing import Union, Sequence, Dict, Tuple, Optional
from numbers import Number, Integral
import json
import os
import numpy

class Camera(object):
    r"""
    Represents a camera in 3D space with intrinsic parameters, orientation, and position.

    The camera orientation is defined by the frame.
    
    The frame of the camera defines the orientation of the camera in 3D space with (convention OPENCV): 

    - origin: The position of the camera in 3D space.
    - x-axis: The right direction of the camera (left to right).
    - y-axis: The up direction of the camera (up to down).
    - z-axis: The optical axis of the camera (from the camera to the scene).

    .. figure:: ../../../pyblenderSDIC/resources/doc/opencv_camera_frame.png
        :width: 500
        :align: center
        
        OpenCV camera frame convention (source: OpenCV)

    The intrinsic parameters of the camera are defined by the following parameters:

    - focal_length: The focal length of the camera in pixels along the x-axis and y-axis.
    - principal_point: The principal point of the camera in pixels.

    They are represented by the intrinsic matrix of the camera:

    .. math::

        K = \begin{bmatrix}
        f_x & 0 & c_x \\
        0 & f_y & c_y \\
        0 & 0 & 1
        \end{bmatrix}
    
    Other parameters of the camera can be stored in the camera object:

    - resolution: The resolution of the camera in numbers of pixels. A 2-element array representing the resolution of the camera in numbers of pixels along the x and y axes.
    - pixel_size: The pixel size of the camera in millimeters (distance unit). A 2-element array representing the pixel size of the camera in millimeters along the x and y axes.
    - clip_distance: The distances in millimeters to the near and far clipping planes. A 2-element array representing the distances of the near and far clipping planes of the camera in millimeters. Only points between the near and far clipping planes are visible in the camera view.

    Parameters
    ----------
    frame : Frame, optional
        The frame of the camera. (see py3dframe : https://https://artezaru.github.io/py3dframe/), default Frame().

    intrinsic_matrix : Optional[numpy.ndarray], optional
        The intrinsic matrix of the camera. (3x3 matrix), by default None.
        The intrinsic matrix is a 3x3 matrix representing the intrinsic parameters of the camera.

    resolution : Optional[Union[Sequence[Integral], numpy.ndarray]], optional
        The resolution of the camera in numbers of pixels, by default None.
        The two values are used for x and y axes respectively.

    pixel_size : Optional[Union[Sequence[Number], numpy.ndarray]], optional
        The pixel size of the camera in the millimeters of the camera, by default None.
        The two values are used for x and y axes respectively.
        The pixel size is the size of a single pixel in the millimeters of the camera.
    
    clip_distance : Optional[Union[NSequence[Number], numpy.ndarray]], optional
        The distance of the near and far clipping planes of the camera, by default None.
        The two values are used for near and far clipping planes respectively.
    """
    def __init__(
            self,
            frame: Frame = None,
            intrinsic_matrix: Optional[numpy.ndarray] = None,
            resolution: Optional[Union[Sequence[Integral], numpy.ndarray]] = None,
            pixel_size: Optional[Union[Sequence[Number], numpy.ndarray]] = None,
            clip_distance: Optional[Union[Sequence[Number], numpy.ndarray]] = None,
        ) -> None:

        # Default values
        self._fx = None # focal length in pixels in x direction
        self._fy = None # focal length in pixels in y direction
        self._cx = None # principal point in pixels in x direction
        self._cy = None # principal point in pixels in y direction
        self._px = None # pixel size in millimeters in x direction
        self._py = None # pixel size in millimeters in y direction
        self._rx = None # resolution in number of pixels in x direction
        self._ry = None # resolution in number of pixels in y direction
        self._clnear = None # near clipping plane in millimeters
        self._clfar = None # far clipping plane in millimeters

        # Set the values
        if frame is None:
            frame = Frame.canonical()
        self.frame = frame
        self.intrinsic_matrix = intrinsic_matrix
        self.resolution = resolution
        self.pixel_size = pixel_size
        self.clip_distance = clip_distance



    # ===============================================
    # Frame
    # ===============================================
    @property
    def frame(self) -> Frame:
        r"""
        Get or set the frame of the camera.
        
        The frame of the camera defines the orientation of the camera in 3D space.
        The camera observes the scene along the z-axis of the frame.

        The frame of the camera defines the orientation of the camera in 3D space with (convention OPENCV): 

        - origin: The position of the camera in 3D space.
        - x-axis: The right direction of the camera (left to right).
        - y-axis: The up direction of the camera (up to down).
        - z-axis: The optical axis of the camera (from the camera to the scene).

        .. figure:: ../../../pyblenderSDIC/resources/doc/opencv_camera_frame.png
            :width: 500
            :align: center
            
            OpenCV camera frame convention (source: OpenCV)

        .. seealso::

            - https://artezaru.github.io/py3dframe/ for more information about the frame.

        Returns
        -------
        Frame
            The frame of the camera.
        """
        return self._frame
    
    @frame.setter
    def frame(self, frame: Frame) -> None:
        if not isinstance(frame, Frame):
            raise ValueError("Frame must be a Frame object.")
        self._frame = frame



    # =============================================
    # Focal length
    # =============================================
    @property
    def focal_length_x(self) -> Optional[float]:
        r"""
        Get or set the focal length of the camera in pixels in x direction.

        The focal length is a float representing the focal length of the camera in pixels in x direction.

        This parameter is the component K[0, 0] of the intrinsic matrix K of the camera.

        .. note::

            An alias for focal_length_x is ``fx``.

        .. seealso::

            - :meth:`pyblenderSDIC.Camera.focal_length_y` or ``fy`` to set the focal length in pixels in y direction.

        Returns
        -------
        Optional[float]
            The focal length of the camera in pixels in x direction. (or None if not set)
        """
        return self._fx

    @focal_length_x.setter
    def focal_length_x(self, fx: Optional[Number]) -> None:
        if fx is None or numpy.isnan(fx):
            self._fx = None
            return
        if not isinstance(fx, Number):
            raise ValueError("Focal length in pixels in x direction must be a number.")
        if not numpy.isfinite(fx):
            raise ValueError("Focal length in pixels in x direction must be a finite number.")
        if fx <= 0:
            raise ValueError("Focal length in pixels in x direction must be greater than 0.")
        self._fx = float(fx)

    @property
    def fx(self) -> float:
        return self.focal_length_x
    
    @fx.setter
    def fx(self, fx: Optional[Number]) -> None:
        self.focal_length_x = fx


    @property
    def focal_length_y(self) -> Optional[float]:
        r"""
        Get or set the focal length of the camera in pixels in y direction.

        The focal length is a float representing the focal length of the camera in pixels in y direction.

        This parameter is the component K[1, 1] of the intrinsic matrix K of the camera.

        .. note::

            An alias for focal_length_y is ``fy``.

        .. seealso::

            - :meth:`pyblenderSDIC.Camera.focal_length_x` or ``fx`` to set the focal length in pixels in x direction.

        Returns
        -------
        Optional[float]
            The focal length of the camera in pixels in y direction. (or None if not set)
        """
        return self._fy
    
    @focal_length_y.setter
    def focal_length_y(self, fy: Optional[Number]) -> None:
        if fy is None or numpy.isnan(fy):
            self._fy = None
            return
        if not isinstance(fy, Number):
            raise ValueError("Focal length in pixels in y direction must be a number.")
        if not numpy.isfinite(fy):
            raise ValueError("Focal length in pixels in y direction must be a finite number.")
        if fy <= 0:
            raise ValueError("Focal length in pixels in y direction must be greater than 0.")
        self._fy = float(fy)
    
    @property
    def fy(self) -> float:
        return self.focal_length_y

    @fy.setter
    def fy(self, fy: Optional[Number]) -> None:
        self.focal_length_y = fy

    
    @property
    def focal_length(self) -> Tuple[Optional[float], Optional[float]]:
        r"""
        Get or set the focal length of the camera in pixels.

        The focal length is a tuple of two floats representing the focal length of the camera in pixels in x and y directions.

        Returns
        -------
        Tuple[Optional[float], Optional[float]]
            The focal length of the camera in pixels. (or None if not set)
        """
        return self._fx, self._fy
    
    @focal_length.setter
    def focal_length(self, focal_length: Optional[Sequence[Number]]) -> None:
        if focal_length is None:
            self._fx = None
            self._fy = None
            return
        if not isinstance(focal_length, (Sequence, numpy.ndarray)):
            raise ValueError("Focal length must be a sequence of two numbers.")
        focal_length = numpy.asarray(focal_length, dtype=numpy.float64).flatten()
        if len(focal_length) != 2:
            raise ValueError("Focal length must be a sequence of two numbers.")
        self.focal_length_x = focal_length[0]
        self.focal_length_y = focal_length[1]



    # =============================================
    # Principal point
    # =============================================
    @property
    def principal_point_x(self) -> Optional[float]:
        r"""
        Get or set the principal point of the camera in pixels in x direction.

        The principal point is a float representing the principal point of the camera in pixels in x direction.

        This parameter is the component K[0, 2] of the intrinsic matrix K of the camera.

        .. note::

            An alias for principal_point_x is ``cx``.

        .. seealso::

            - :meth:`pyblenderSDIC.Camera.principal_point_y` or ``cy`` to set the principal point in pixels in y direction.

        Returns
        -------
        Optional[float]
            The principal point of the camera in pixels in x direction. (or None if not set)
        """
        return self._cx
    
    @principal_point_x.setter
    def principal_point_x(self, cx: Optional[Number]) -> None:
        if cx is None or numpy.isnan(cx):
            self._cx = None
            return
        if not isinstance(cx, Number):
            raise ValueError("Principal point in pixels in x direction must be a number.")
        if not numpy.isfinite(cx):
            raise ValueError("Principal point in pixels in x direction must be a finite number.")
        if cx < 0:
            raise ValueError("Principal point in pixels in x direction must be greater than or equal to 0.")
        self._cx = float(cx)

    @property
    def cx(self) -> float:
        return self.principal_point_x
    
    @cx.setter
    def cx(self, cx: Optional[Number]) -> None:
        self.principal_point_x = cx

    @property
    def principal_point_y(self) -> Optional[float]:
        r"""
        Get or set the principal point of the camera in pixels in y direction.

        The principal point is a float representing the principal point of the camera in pixels in y direction.

        This parameter is the component K[1, 2] of the intrinsic matrix K of the camera.

        .. note::

            An alias for principal_point_y is ``cy``.

        .. seealso::

            - :meth:`pyblenderSDIC.Camera.principal_point_x` or ``cx`` to set the principal point in pixels in x direction.

        Returns
        -------
        Optional[float]
            The principal point of the camera in pixels in y direction. (or None if not set)
        """
        return self._cy
    
    @principal_point_y.setter
    def principal_point_y(self, cy: Optional[Number]) -> None:
        if cy is None or numpy.isnan(cy):
            self._cy = None
            return
        if not isinstance(cy, Number):
            raise ValueError("Principal point in pixels in y direction must be a number.")
        if not numpy.isfinite(cy):
            raise ValueError("Principal point in pixels in y direction must be a finite number.")
        if cy < 0:
            raise ValueError("Principal point in pixels in y direction must be greater than or equal to 0.")
        self._cy = float(cy)
    
    @property
    def cy(self) -> float:
        return self.principal_point_y
    
    @cy.setter
    def cy(self, cy: Optional[Number]) -> None:
        self.principal_point_y = cy

    @property
    def principal_point(self) -> Tuple[Optional[float], Optional[float]]:
        r"""
        Get or set the principal point of the camera in pixels.

        The principal point is a tuple of two floats representing the principal point of the camera in pixels in x and y directions.

        Returns
        -------
        Tuple[Optional[float], Optional[float]]
            The principal point of the camera in pixels. (or None if not set)
        """
        return self._cx, self._cy
    
    @principal_point.setter
    def principal_point(self, principal_point: Optional[Sequence[Number]]) -> None:
        if principal_point is None:
            self._cx = None
            self._cy = None
            return
        if not isinstance(principal_point, (Sequence, numpy.ndarray)):
            raise ValueError("Principal point must be a sequence of two numbers.")
        principal_point = numpy.asarray(principal_point, dtype=numpy.float64).flatten()
        if len(principal_point) != 2:
            raise ValueError("Principal point must be a sequence of two numbers.")
        self.principal_point_x = principal_point[0]
        self.principal_point_y = principal_point[1]

    
    # =============================================
    # Pixel size
    # =============================================
    @property
    def pixel_size_x(self) -> Optional[float]:
        r"""
        Get or set the pixel size of the camera in millimeters in x direction.

        The pixel size is a float representing the pixel size of the camera in millimeters in x direction.

        .. note::

            An alias for pixel_size_x is ``px``.

        .. seealso::

            - :meth:`pyblenderSDIC.Camera.pixel_size_y` or ``py`` to set the pixel size in millimeters in y direction.

        Returns
        -------
        Optional[float]
            The pixel size of the camera in millimeters in x direction. (or None if not set)
        """
        return self._px
    
    @pixel_size_x.setter
    def pixel_size_x(self, px: Optional[Number]) -> None:
        if px is None or numpy.isnan(px):
            self._px = None
            return
        if not isinstance(px, Number):
            raise ValueError("Pixel size in millimeters in x direction must be a number.")
        if not numpy.isfinite(px):
            raise ValueError("Pixel size in millimeters in x direction must be a finite number.")
        if px <= 0:
            raise ValueError("Pixel size in millimeters in x direction must be greater than 0.")
        self._px = float(px)

    @property
    def px(self) -> float:
        return self.pixel_size_x
    
    @px.setter
    def px(self, px: Optional[Number]) -> None:
        self.pixel_size_x = px

    @property
    def pixel_size_y(self) -> Optional[float]:
        r"""
        Get or set the pixel size of the camera in millimeters in y direction.

        The pixel size is a float representing the pixel size of the camera in millimeters in y direction.

        .. note::

            An alias for pixel_size_y is ``py``.

        .. seealso::

            - :meth:`pyblenderSDIC.Camera.pixel_size_x` or ``px`` to set the pixel size in millimeters in x direction.

        Returns
        -------
        Optional[float]
            The pixel size of the camera in millimeters in y direction. (or None if not set)
        """
        return self._py
    
    @pixel_size_y.setter
    def pixel_size_y(self, py: Optional[Number]) -> None:
        if py is None or numpy.isnan(py):
            self._py = None
            return
        if not isinstance(py, Number):
            raise ValueError("Pixel size in millimeters in y direction must be a number.")
        if not numpy.isfinite(py):
            raise ValueError("Pixel size in millimeters in y direction must be a finite number.")
        if py <= 0:
            raise ValueError("Pixel size in millimeters in y direction must be greater than 0.")
        self._py = float(py)

    @property
    def py(self) -> float:
        return self.pixel_size_y
    
    @py.setter
    def py(self, py: Optional[Number]) -> None:
        self.pixel_size_y = py

    @property
    def pixel_size(self) -> Tuple[Optional[float], Optional[float]]:
        r"""
        Get or set the pixel size of the camera in millimeters.

        The pixel size is a tuple of two floats representing the pixel size of the camera in millimeters in x and y directions.

        Returns
        -------
        Tuple[Optional[float], Optional[float]]
            The pixel size of the camera in millimeters. (or None if not set)
        """
        return self._px, self._py
    
    @pixel_size.setter
    def pixel_size(self, pixel_size: Optional[Sequence[Number]]) -> None:
        if pixel_size is None:
            self._px = None
            self._py = None
            return
        if not isinstance(pixel_size, (Sequence, numpy.ndarray)):
            raise ValueError("Pixel size must be a sequence of two numbers.")
        pixel_size = numpy.asarray(pixel_size, dtype=numpy.float64).flatten()
        if len(pixel_size) != 2:
            raise ValueError("Pixel size must be a sequence of two numbers.")
        self.pixel_size_x = pixel_size[0]
        self.pixel_size_y = pixel_size[1]

    # =============================================
    # Resolution
    # =============================================
    @property
    def resolution_x(self) -> Optional[float]:
        r"""
        Get or set the resolution of the camera in number of pixels in x direction.

        The resolution is a float representing the resolution of the camera in number of pixels in x direction.

        .. note::

            An alias for resolution_x is ``rx``.

        .. seealso::

            - :meth:`pyblenderSDIC.Camera.resolution_y` or ``ry`` to set the resolution in number of pixels in y direction.

        Returns
        -------
        Optional[float]
            The resolution of the camera in number of pixels in x direction. (or None if not set)
        """
        return self._rx
    
    @resolution_x.setter
    def resolution_x(self, rx: Optional[Number]) -> None:
        if rx is None or numpy.isnan(rx):
            self._rx = None
            return
        if not isinstance(rx, Number):
            raise ValueError("Resolution in number of pixels in x direction must be a number.")
        if not numpy.isfinite(rx):
            raise ValueError("Resolution in number of pixels in x direction must be a finite number.")
        if rx <= 0:
            raise ValueError("Resolution in number of pixels in x direction must be greater than 0.")
        self._rx = int(rx)

    @property
    def rx(self) -> float:
        return self.resolution_x
    
    @rx.setter
    def rx(self, rx: Optional[Number]) -> None:
        self.resolution_x = rx

    @property
    def resolution_y(self) -> Optional[float]:
        r"""
        Get or set the resolution of the camera in number of pixels in y direction.

        The resolution is a float representing the resolution of the camera in number of pixels in y direction.

        .. note::

            An alias for resolution_y is ``ry``.

        .. seealso::

            - :meth:`pyblenderSDIC.Camera.resolution_x` or ``rx`` to set the resolution in number of pixels in x direction.

        Returns
        -------
        Optional[float]
            The resolution of the camera in number of pixels in y direction. (or None if not set)
        """
        return self._ry
    
    @resolution_y.setter
    def resolution_y(self, ry: Optional[Number]) -> None:
        if ry is None or numpy.isnan(ry):
            self._ry = None
            return
        if not isinstance(ry, Number):
            raise ValueError("Resolution in number of pixels in y direction must be a number.")
        if not numpy.isfinite(ry):
            raise ValueError("Resolution in number of pixels in y direction must be a finite number.")
        if ry <= 0:
            raise ValueError("Resolution in number of pixels in y direction must be greater than 0.")
        self._ry = int(ry)

    @property
    def ry(self) -> float:
        return self.resolution_y
    
    @ry.setter
    def ry(self, ry: Optional[Number]) -> None:
        self.resolution_y = ry

    @property
    def resolution(self) -> Tuple[Optional[float], Optional[float]]:
        r"""
        Get or set the resolution of the camera in number of pixels.

        The resolution is a tuple of two floats representing the resolution of the camera in number of pixels in x and y directions.

        Returns
        -------
        Tuple[Optional[float], Optional[float]]
            The resolution of the camera in number of pixels. (or None if not set)
        """
        return self._rx, self._ry
    
    @resolution.setter
    def resolution(self, resolution: Optional[Sequence[Number]]) -> None:
        if resolution is None:
            self._rx = None
            self._ry = None
            return
        if not isinstance(resolution, (Sequence, numpy.ndarray)):
            raise ValueError("Resolution must be a sequence of two numbers.")
        resolution = numpy.asarray(resolution, dtype=numpy.float64).flatten()
        if len(resolution) != 2:
            raise ValueError("Resolution must be a sequence of two numbers.")
        self.resolution_x = resolution[0]
        self.resolution_y = resolution[1]


    # =============================================
    # Clipping distance
    # =============================================
    @property
    def clip_distance_near(self) -> Optional[float]:
        r"""
        Get or set the near clipping plane of the camera in millimeters.

        The near clipping plane is a float representing the near clipping plane of the camera in millimeters.
        Only points between the near and far clipping planes are visible in the camera view.

        .. note::

            An alias for clip_distance_near is ``clnear``.

        .. seealso::

            - :meth:`pyblenderSDIC.Camera.clip_distance_far` or ``clfar`` to set the far clipping plane in millimeters.

        Returns
        -------
        Optional[float]
            The near clipping plane of the camera in millimeters. (or None if not set)
        """
        return self._clnear
    
    @clip_distance_near.setter
    def clip_distance_near(self, clnear: Optional[Number]) -> None:
        if clnear is None or numpy.isnan(clnear):
            self._clnear = None
            return
        if not isinstance(clnear, Number):
            raise ValueError("Near clipping plane in millimeters must be a number.")
        if not numpy.isfinite(clnear):
            raise ValueError("Near clipping plane in millimeters must be a finite number.")
        if clnear <= 0:
            raise ValueError("Near clipping plane in millimeters must be greater than 0.")
        self._clnear = float(clnear)

    @property
    def clnear(self) -> float:
        return self.clip_distance_near
    
    @clnear.setter
    def clnear(self, clnear: Optional[Number]) -> None:
        self.clip_distance_near = clnear

    @property
    def clip_distance_far(self) -> Optional[float]:
        r"""
        Get or set the far clipping plane of the camera in millimeters.

        The far clipping plane is a float representing the far clipping plane of the camera in millimeters.
        Only points between the near and far clipping planes are visible in the camera view.

        .. note::

            An alias for clip_distance_far is ``clfar``.

        .. seealso::

            - :meth:`pyblenderSDIC.Camera.clip_distance_near` or ``clnear`` to set the near clipping plane in millimeters.

        Returns
        -------
        Optional[float]
            The far clipping plane of the camera in millimeters. (or None if not set)
        """
        return self._clfar
    
    @clip_distance_far.setter
    def clip_distance_far(self, clfar: Optional[Number]) -> None:
        if clfar is None or numpy.isnan(clfar):
            self._clfar = None
            return
        if not isinstance(clfar, Number):
            raise ValueError("Far clipping plane in millimeters must be a number.")
        if not numpy.isfinite(clfar):
            raise ValueError("Far clipping plane in millimeters must be a finite number.")
        if clfar <= 0:
            raise ValueError("Far clipping plane in millimeters must be greater than 0.")
        self._clfar = float(clfar)

    @property
    def clfar(self) -> float:
        return self.clip_distance_far
    
    @clfar.setter
    def clfar(self, clfar: Optional[Number]) -> None:
        self.clip_distance_far = clfar

    @property
    def clip_distance(self) -> Tuple[Optional[float], Optional[float]]:
        r"""
        Get or set the near and far clipping planes of the camera in millimeters.

        The near and far clipping planes are a tuple of two floats representing the near and far clipping planes of the camera in millimeters.
        Only points between the near and far clipping planes are visible in the camera view.

        Returns
        -------
        Tuple[Optional[float], Optional[float]]
            The near and far clipping planes of the camera in millimeters. (or None if not set)
        """
        return self._clnear, self._clfar
    
    @clip_distance.setter
    def clip_distance(self, clip_distance: Optional[Sequence[Number]]) -> None:
        if clip_distance is None:
            self._clnear = None
            self._clfar = None
            return
        if not isinstance(clip_distance, (Sequence, numpy.ndarray)):
            raise ValueError("Clip distance must be a sequence of two numbers.")
        clip_distance = numpy.asarray(clip_distance, dtype=numpy.float64).flatten()
        if len(clip_distance) != 2:
            raise ValueError("Clip distance must be a sequence of two numbers.")
        self.clip_distance_near = clip_distance[0]
        self.clip_distance_far = clip_distance[1]

    # =============================================
    # Intrinsic matrix
    # =============================================
    @property
    def intrinsic_matrix(self) -> Optional[numpy.ndarray]:
        r"""
        Get or set the intrinsic matrix of the camera.

        The intrinsic matrix is a 3x3 matrix representing the intrinsic parameters of the camera.

        .. math::

            K = \begin{bmatrix}
            f_x & 0 & c_x \\
            0 & f_y & c_y \\
            0 & 0 & 1
            \end{bmatrix}

        Returns
        -------
        Optional[numpy.ndarray]
            The intrinsic matrix of the camera. (or None if not set)
        """
        if self._fx is None or self._fy is None or self._cx is None or self._cy is None:
            return None
        return numpy.array([
            [self._fx, 0, self._cx],
            [0, self._fy, self._cy],
            [0, 0, 1]
        ], dtype=numpy.float64)
    
    @intrinsic_matrix.setter
    def intrinsic_matrix(self, intrinsic_matrix: Optional[numpy.ndarray]) -> None:
        if intrinsic_matrix is None:
            self._fx = None
            self._fy = None
            self._cx = None
            self._cy = None
            return
        intrinsic_matrix = numpy.asarray(intrinsic_matrix, dtype=numpy.float64)
        if intrinsic_matrix.shape != (3, 3):
            raise ValueError("Intrinsic matrix must be a 3x3 matrix.")
        # Check if a skew value is given
        if abs(intrinsic_matrix[0, 1]) > 1e-6:
            raise ValueError("Skew value is not supported by Blender.")
        # Set the intrinsic parameters
        self.fx = intrinsic_matrix[0, 0]
        self.fy = intrinsic_matrix[1, 1]
        self.cx = intrinsic_matrix[0, 2]
        self.cy = intrinsic_matrix[1, 2]
    
    # =============================================
    # Methods to check if each parameter is set
    # =============================================
    def is_complete(self) -> bool:
        r"""
        Check if the camera is complete.

        A camera is complete if all intrinsic parameters are set.

        Returns
        -------
        bool
            True if the camera is complete, False otherwise.
        """
        if self._fx is None or self._fy is None or self._cx is None or self._cy is None:
            return False
        if self._px is None or self._py is None:
            return False
        if self._rx is None or self._ry is None:
            return False
        if self._clnear is not None and self._clfar is not None:
            if self._clnear >= self._clfar:
                return False
        return True
    
    # =============================================
    # Extract other parameters
    # =============================================
    @property
    def sensor_size_x(self) -> Optional[float]:
        r"""
        Get the sensor size of the camera in millimeters in x direction.

        The sensor size is a float representing the sensor size of the camera in millimeters in x direction.

        The sensor size is calculated as the product of the pixel size and the resolution of the camera.

        .. note::

            An alias for sensor_size_x is ``sensor_width``.

        Returns
        -------
        Optional[float]
            The sensor size of the camera in millimeters in x direction. (or None if `px` or `rx` is not set)
        """
        if self._px is None or self._rx is None:
            return None
        return self._px * self._rx
    
    @property
    def sensor_width(self) -> Optional[float]:
        return self.sensor_size_x
    
    @property
    def sensor_size_y(self) -> Optional[float]:
        r"""
        Get the sensor size of the camera in millimeters in y direction.

        The sensor size is a float representing the sensor size of the camera in millimeters in y direction.

        The sensor size is calculated as the product of the pixel size and the resolution of the camera.

        .. note::

            An alias for sensor_size_y is ``sensor_height``.

        Returns
        -------
        Optional[float]
            The sensor size of the camera in millimeters in y direction. (or None if `py` or `ry` is not set)
        """
        if self._py is None or self._ry is None:
            return None
        return self._py * self._ry
    
    @property
    def sensor_height(self) -> Optional[float]:
        return self.sensor_size_y
    
    @property
    def pixel_aspect_x(self) -> Optional[float]:
        r"""
        Get the pixel aspect of the camera in x direction.

        The pixel aspect is a float representing the pixel aspect of the camera in x direction.

        If :math:`f_x < f_y`, the pixel aspect is calculated as :math:`\frac{f_y}{f_x}`.
        If :math:`f_x \geq f_y`, the aspect ratio is 1.

        Returns
        -------
        Optional[float]
            The pixel aspect of the camera in x direction. (or None if `fx` or `fy` is not set)
        """
        if self._fx is None or self._fy is None:
            return None
        if self._fx < self._fy:
            return self._fy / self._fx
        return 1
    
    @property
    def pixel_aspect_y(self) -> Optional[float]:
        r"""
        Get the pixel aspect of the camera in y direction.

        The pixel aspect is a float representing the pixel aspect of the camera in y direction.

        If :math:`f_y < f_x`, the pixel aspect is calculated as :math:`\frac{f_x}{f_y}`.
        If :math:`f_y \geq f_x`, the aspect ratio is 1.

        Returns
        -------
        Optional[float]
            The pixel aspect of the camera in y direction. (or None if `fx` or `fy` is not set)
        """
        if self._fx is None or self._fy is None:
            return None
        if self._fy < self._fx:
            return self._fx / self._fy
        return 1
    
    @property
    def aspect_ratio(self) -> Optional[float]:
        r"""
        Get the aspect ratio of the camera.

        The aspect ratio is a float representing the aspect ratio of the camera.

        The aspect ratio is calculated as the ratio of the pixel aspect in y direction to the pixel aspect in x direction.

        Returns
        -------
        Optional[float]
            The aspect ratio of the camera. (or None if `fx` or `fy` is not set)
        """
        if self._fx is None or self._fy is None:
            return None
        return self.pixel_aspect_y / self.pixel_aspect_x
    

    @property
    def sensor_fit(self) -> Optional[str]:
        r"""
        Get the sensor fit of the camera ("HORIZONTAL" or "VERTICAL").

        The sensor fit is a string representing the sensor fit of the camera.

        If :math:`a_x \cdot r_x \geq a_y \cdot r_y`, the sensor fit is "HORIZONTAL".
        If :math:`a_x \cdot r_x < a_y \cdot r_y`, the sensor fit is "VERTICAL".

        where :math:`a_x` and :math:`a_y` are the pixel aspect in x and y direction respectively and :math:`r_x` and :math:`r_y` are the resolution in number of pixels in x and y direction respectively.

        Returns
        -------
        str
            The sensor fit of the camera. (or None if `fx`, `fy` or `rx`, `ry` is not set)
        """
        if self._fx is None or self._fy is None:
            return None
        if self._rx is None or self._ry is None:
            return None
        if self.pixel_aspect_x * self.rx >= self.pixel_aspect_y * self.ry:
            return "HORIZONTAL"
        return "VERTICAL"
    

    @property
    def view_factor(self) -> Optional[float]:
        r"""
        Get the view factor of the camera.

        The view factor is a float representing the view factor of the camera.

        If the sensor fit is "HORIZONTAL", the view factor is directly the resolution in number of pixels in x direction.
        If the sensor fit is "VERTICAL", the view factor is calculated as the product of the aspect ratio and the resolution in number of pixels in y direction.

        .. warning::

            This view factor is only for Blender. No reel physical meaning.

        Returns
        -------
        Optional[float]
            The view factor of the camera. (or None if `fx`, `fy` or `rx`, `ry` is not set)
        """
        if self._fx is None or self._fy is None:
            return None
        if self._rx is None or self._ry is None:
            return None
        
        if self.sensor_fit == "HORIZONTAL":
            return self._rx
        if self.sensor_fit == "VERTICAL":
            return self.aspect_ratio * self._ry


    @property
    def sensor_size(self) -> Optional[float]:
        r"""
        Get the sensor size of the camera in millimeters.

        If the sensor fit is "HORIZONTAL", the sensor size is directly the sensor size in x direction.
        If the sensor fit is "VERTICAL", the sensor size is directly the sensor size in y direction.

        .. warning::

            This sensor size is only for Blender. No reel physical meaning.

        Returns
        -------
        Optional[float]
            The sensor size of the camera in millimeters. (or None if `fx`, `fy`, `px`, `py`, `rx` or `ry` is not set)
        """
        if self._px is None or self._py is None:
            return None
        if self._rx is None or self._ry is None:
            return None
        if self._fx is None or self._fy is None:
            return None
        if self.sensor_fit == "HORIZONTAL":
            return self.sensor_size_x
        if self.sensor_fit == "VERTICAL":
            return self.sensor_size_y


    @property
    def lens(self) -> Optional[float]:
        r"""
        Get the lens of the camera in millimeters.

        The lens is a float representing the focal length of the camera in millimeters.

        The lens is calculated as follows:

        .. math::

            f = \frac{f_x \cdot s}{vf}

        where :math:`f_x` is the focal length in pixels in x direction, :math:`s` is the sensor size in millimeters and :math:`vf` is the view factor of the camera.

        .. warning::

            This lens is only for Blender. No reel physical meaning.

        Returns
        -------
        Optional[float]
            The lens of the camera in millimeters. (or None if `fx`, `fy`, `px`, `py`, `rx` or `ry` is not set)
        """
        if self._fx is None or self._fy is None:
            return None
        if self._px is None or self._py is None:
            return None
        if self._rx is None or self._ry is None:
            return None
        return self.focal_length_x * self.sensor_size / self.view_factor


    @property
    def shift_x(self) -> Optional[float]:
        r"""
        Get the shift x of the camera in pixels.

        The shift x is a float representing the shift x of the camera in pixels.

        The shift x is calculated as follows:

        .. math::

            sx = - \frac{c_x - (r_x - 1) / 2}{vf}

        where :math:`c_x` is the principal point in pixels in x direction, :math:`r_x` is the resolution in number of pixels in x direction and :math:`vf` is the view factor of the camera.

        .. warning::

            This shift x is only for Blender. No reel physical meaning.

            "-" added to fix the shift direction in Blender.

        Returns
        -------
        Optional[float]
            The shift x of the camera in pixels. (or None if `fx`, `fy`, `px`, `py`, `rx` or `ry` is not set)
        """
        if self._fx is None or self._fy is None:
            return None
        if self._px is None or self._py is None:
            return None
        if self._rx is None or self._ry is None:
            return None
        return - (self._cx - (self._rx - 1) / 2) / self.view_factor
    

    @property
    def shift_y(self) -> Optional[float]:
        r"""
        Get the shift y of the camera in pixels.

        The shift y is a float representing the shift y of the camera in pixels.

        The shift y is calculated as follows:

        .. math::

            sy = \frac{(c_y - (r_y - 1) / 2) ar}{vf}

        where :math:`c_y` is the principal point in pixels in y direction, :math:`r_y` is the resolution in number of pixels in y direction, :math:`vf` is the view factor of the camera and :math:`ar` is the aspect ratio of the camera.
        
        .. warning::

            This shift y is only for Blender. No reel physical meaning.

        Returns
        -------
        Optional[float]
            The shift y of the camera in pixels. (or None if `fx`, `fy`, `px`, `py`, `rx` or `ry` is not set)
        """
        if self._fx is None or self._fy is None:
            return None
        if self._px is None or self._py is None:
            return None
        if self._rx is None or self._ry is None:
            return None
        return (self._cy - (self._ry - 1) / 2) / self.view_factor * self.aspect_ratio 
    

    # =============================================
    # OpenCV and OpenGL methods
    # =============================================
    def get_OpenCV_RT(self) -> Tuple[Rotation, numpy.ndarray]:
        r"""
        Get the rotation and translation of the camera in the OpenCV format.

        The axis of the camera frame for OpenCV are the same as the Camera frame.
        Furthermore, the convention for OpenCV is :math:`X_{cam} = R X_{world} + T`, convention=4 for py3dframe.

        Returns
        -------
        Rotation
            The rotation of the camera.
        
        numpy.ndarray
            The translation of the camera with shape (3, 1).
        """
        rotation = self.frame.get_global_rotation(convention=4)
        translation = self.frame.get_global_translation(convention=4)
        return rotation, translation
    
    def set_OpenCV_RT(self, rotation: Rotation, translation: numpy.ndarray) -> None:
        r"""
        Set the rotation and translation of the camera in the OpenCV format.

        The axis of the camera frame for OpenCV are the same as the Camera frame.
        Furthermore, the convention for OpenCV is :math:`X_{cam} = R X_{world} + T`, convention=4 for py3dframe.

        Parameters
        ----------
        rotation : Rotation
            The rotation of the camera.

        translation : numpy.ndarray
            The translation of the camera with shape (3, 1).
        """
        self.frame.set_global_rotation(rotation, convention=4)
        self.frame.set_global_translation(translation, convention=4)
    
    @property
    def OpenCV_tvec(self) -> numpy.ndarray:
        r"""
        Get or set the translation vector of the camera in the OpenCV format.

        The axis of the camera frame for OpenCV are the same as the Camera frame.
        Furthermore, the convention for OpenCV is :math:`X_{cam} = R X_{world} + T`, convention=4 for py3dframe.

        Returns
        -------
        numpy.ndarray
            The translation vector of the camera with shape (3,).
        """
        return self.frame.get_global_translation(convention=4).reshape((3,))
    
    @OpenCV_tvec.setter
    def OpenCV_tvec(self, tvec: numpy.ndarray) -> None:
        self.frame.set_global_translation(tvec, convention=4)

    @property
    def OpenCV_rvec(self) -> numpy.ndarray:
        r"""
        Get or set the rotation vector of the camera in the OpenCV format.

        The axis of the camera frame for OpenCV are the same as the Camera frame.
        Furthermore, the convention for OpenCV is :math:`X_{cam} = R X_{world} + T`, convention=4 for py3dframe.

        Returns
        -------
        numpy.ndarray
            The rotation vector of the camera with shape (3,).
        """
        return self.frame.get_global_rotation(convention=4).as_rotvec()

    @OpenCV_rvec.setter
    def OpenCV_rvec(self, rvec: numpy.ndarray) -> None:
        self.frame.set_global_rotation(Rotation.from_rotvec(rvec), convention=4)


    def get_OpenGL_RT(self) -> Tuple[Rotation, numpy.ndarray]:
        r"""
        Get the rotation and translation of the camera in the OpenGL format.

        The axis of the camera frame for OpenGL are different from the Camera frame:
        - x-axis: The same as the Camera frame : right direction of the camera (left to right).
        - y-axis: The opposite of the Camera frame : up direction of the camera (down to up).
        - z-axis: The opposite of the Camera frame : (from the scene to the camera).

        Furthermore, the convention for OpenGL is :math:`X_{world} = R X_{cam} + T`, convention=0 for py3dframe.

        Returns
        -------
        Rotation
            The rotation of the camera.
        
        numpy.ndarray
            The translation of the camera with shape (3, 1).
        """
        rotation = self.frame.get_global_rotation(convention=0)
        x_axis = rotation.as_matrix()[:, 0].reshape((3, 1))
        y_axis = - rotation.as_matrix()[:, 1].reshape((3, 1))
        z_axis = - rotation.as_matrix()[:, 2].reshape((3, 1))
        rotation = Rotation.from_matrix(numpy.column_stack((x_axis, y_axis, z_axis)))
        translation = self.frame.get_global_translation(convention=0)
        return rotation, translation
    
    def set_OpenGL_RT(self, rotation: Rotation, translation: numpy.ndarray) -> None:
        r"""
        Set the rotation and translation of the camera in the OpenGL format.

        The axis of the camera frame for OpenGL are different from the Camera frame:
        - x-axis: The same as the Camera frame : right direction of the camera (left to right).
        - y-axis: The opposite of the Camera frame : up direction of the camera (down to up).
        - z-axis: The opposite of the Camera frame : (from the scene to the camera).

        Furthermore, the convention for OpenGL is :math:`X_{world} = R X_{cam} + T`, convention=0 for py3dframe.

        Parameters
        ----------
        rotation : Rotation
            The rotation of the camera.

        translation : numpy.ndarray
            The translation of the camera with shape (3, 1).
        """
        x_axis = rotation.as_matrix()[:, 0].reshape((3, 1))
        y_axis = - rotation.as_matrix()[:, 1].reshape((3, 1))
        z_axis = - rotation.as_matrix()[:, 2].reshape((3, 1))
        rotation = Rotation.from_matrix(numpy.column_stack((x_axis, y_axis, z_axis)))
        self.frame.set_global_rotation(rotation, convention=0)
        self.frame.set_global_translation(translation, convention=0)


    # =============================================
    # Save and load methods
    # =============================================
    def to_dict(self, description: Optional[str] = None) -> Dict:
        r"""
        Export the Camera's data to a dictionary.

        The structure of the dictionary is as follows:

        .. code-block:: python

            {
                "type": "Camera [pyblenderSDIC]",
                "description": "Description of the camera",
                "frame": {
                    "translation": [float, float, float],
                    "rotvec": [float, float, float],
                    "convention": 0,
                    "parent": None
                },
                "fx": float,
                "fy": float,
                "cx": float,
                "cy": float,
                "rx": int,
                "ry": int,
                "px": float,
                "py": float,
                "clnear": float,
                "clfar": float
                }
            }

        Parameters
        ----------
        description : Optional[str]
            A description of the camera, by default None. 
            This message will be included in the dictionary under the key "description" if provided. 

        Returns
        -------
        dict
            A dictionary containing the camera's data.

        Raises
        ------
        ValueError
            If the description is not a string.
        """        
        # Create the dictionary
        data = {
            "type": "Camera [pyblenderSDIC]",
            "frame": self.frame.save_to_dict(),
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "rx": self.rx,
            "ry": self.ry,
            "px": self.px,
            "py": self.py,
            "clnear": self.clnear,
            "clfar": self.clfar
        }

        # Add the description
        if description is not None:
            if not isinstance(description, str):
                raise ValueError("Description must be a string.")
            data["description"] = description
        
        return data

    @classmethod
    def from_dict(cls, data: Dict) -> Camera:
        r"""
        Create a Camera instance from a dictionary.

        The structure of the dictionary should be as provided by the :meth:`pysdic.Camera.save_to_dict` method.

        If focal_length, resolution, pixel_size, or principal_point are not provided, the default values are used.

        The other keys are ignored.

        Parameters
        ----------
        data : dict
            A dictionary containing the camera's data.
        
        Returns
        -------
        Camera
            The Camera instance.

        Raises
        ------
        ValueError
            If the data is not a dictionary.
        """
        # Check for the input type
        if not isinstance(data, dict):
            raise ValueError("data must be a dictionary.")
        
        # Create the Camera instance
        frame = Frame.load_from_dict(data["frame"])
        camera = cls(frame=frame)

        # Set the parameters
        camera.fx = data.get("fx", None)
        camera.fy = data.get("fy", None)
        camera.cx = data.get("cx", None)
        camera.cy = data.get("cy", None)
        camera.rx = data.get("rx", None)
        camera.ry = data.get("ry", None)
        camera.px = data.get("px", None)
        camera.py = data.get("py", None)
        camera.clnear = data.get("clnear", None)
        camera.clfar = data.get("clfar", None)

        return camera
    

    def to_json(self, filename: str, description: Optional[str] = None) -> None:
        r"""
        Export the Camera's data to a JSON file.

        The structure of the JSON file follows the :meth:`pysdic.camera.Camera.to_dict` method.

        Parameters
        ----------
        filename : str
            The path to the JSON file.
        
        description : Optional[str]
            A description of the camera, by default None. 
            This message will be included in the JSON file under the key "description" if provided.

        Raises
        ------
        FileNotFoundError
            If the filename is not a valid path.
        """
        # Create the dictionary
        data = self.to_dict(description=description)

        # Save the dictionary to a JSON file
        with open(filename, "w") as file:
            json.dump(data, file, indent=4)

    @classmethod
    def from_json(cls, filename: str) -> Camera:
        r"""
        Create a Camera instance from a JSON file.

        The structure of the JSON file follows the :meth:`pysdic.camera.Camera.to_dict` method.

        Parameters
        ----------
        filename : str
            The path to the JSON file.
        
        Returns
        -------
        Camera
            A Camera instance.
        
        Raises
        ------
        FileNotFoundError
            If the filename is not a valid path.
        """
        # Load the dictionary from the JSON file
        with open(filename, "r") as file:
            data = json.load(file)
        
        # Create the Frame instance
        return cls.from_dict(data)