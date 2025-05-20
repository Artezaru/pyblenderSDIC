import numpy

from py3dframe import Frame, Transform
from typing import Callable
from numbers import Integral

from .trimesh3d import TriMesh3D


def create_axisymmetric_mesh(
    profile_curve: Callable[[float], float] = lambda z: 1.0,
    frame: Frame = Frame(),
    height_bounds: tuple[float, float] = (0.0, 1.0),
    theta_bounds: tuple[float, float] = (0.0, 2.0 * numpy.pi),
    Nheight: int = 10,
    Ntheta: int = 10,
    closed: bool = False,
    first_diagonal: bool = True,
    direct: bool = True,
    uv_layout: int = 0,
    ) -> TriMesh3D:
    r"""
    Create a 3D axisymmetric mesh using a given profile curve.

    The profile curve is a function that takes a single argument (height) and returns the radius at that height.
    The returned radius must be strictly positive for all z in the range defined by ``height_bounds``.

    The ``frame`` parameter defines the orientation and the position of the mesh in 3D space.
    The axis of symmetry is aligned with the z-axis of the frame, and z=0 corresponds to the origin of the frame.
    The x-axis of the frame defines the direction of :math:`\theta=0`, and the y-axis defines the direction of :math:`\theta=\pi/2`.
    
    The ``height_bounds`` parameter defines the vertical extent of the mesh, and ``theta_bounds`` defines the angular sweep around the axis.
    ``Nheight`` and ``Ntheta`` determine the number of nodes in the height and angular directions, respectively.
    Nodes are uniformly distributed along both directions.

    .. note::

        - ``Nheight`` and ``Ntheta`` refer to the number of **nodes**, not segments.

    For example, the following code generates a mesh of a half-cylinder whose flat face is centered on the world x-axis:

    .. code-block:: python

        from pyblenderSDIC.meshes import create_axisymmetric_mesh
        import numpy as np

        cylinder_mesh = create_axisymmetric_mesh(
            profile_curve=lambda z: 1.0,
            height_bounds=(-1.0, 1.0),
            theta_bounds=(-np.pi/4, np.pi/4),
            Nheight=10,
            Ntheta=20,
        )

        cylinder_mesh.visualize()

    .. figure:: ../../../pyblenderSDIC/resources/doc/demi_cylinder_mesh.png
        :width: 400
        :align: center

        Demi-cylinder mesh with the face centered on the world x-axis.

    Nodes are ordered first in height (indexed by ``i_H``) and then in theta (indexed by ``i_T``).
    So the node at height index ``i_H`` and angular index ``i_T`` (both starting from 0) is located at:

    .. code-block:: python

        mesh.nodes[i_T * Nheight + i_H, :]

    Each quadrilateral element is defined by the nodes:

    - :math:`(i_H, i_T)`
    - :math:`(i_H + 1, i_T)`
    - :math:`(i_H + 1, i_T + 1)`
    - :math:`(i_H, i_T + 1)`

    This quadrilateral is then split into two triangles depending on the value of ``first_diagonal``:

    - If ``first_diagonal`` is ``True``:

        - Triangle 1: :math:`(i_H, i_T)`, :math:`(i_H, i_T + 1)`, :math:`(i_H + 1, i_T + 1)`
        - Triangle 2: :math:`(i_H, i_T)`, :math:`(i_H + 1, i_T + 1)`, :math:`(i_H + 1, i_T)`

    - If ``first_diagonal`` is ``False``:

        - Triangle 1: :math:`(i_H, i_T)`, :math:`(i_H, i_T + 1)`, :math:`(i_H + 1, i_T)`
        - Triangle 2: :math:`(i_H, i_T + 1)`, :math:`(i_H + 1, i_T + 1)`, :math:`(i_H + 1, i_T)`

    These triangles are oriented in a direct (counterclockwise) order by default (for an observer outside the cylinder).
    If ``direct`` is False, the orientation is reversed by swapping the second and third vertices in each triangle.

    If ``closed`` is True, the mesh is closed in the angular direction.
    In that case, ``theta_bounds`` should be set to:

    .. math::

        (\theta_0, \theta_0 \pm 2\pi (1 - \frac{1}{Ntheta}))

    to avoid duplicating nodes at the seam.

    To generate a closed full cylinder:

    .. code-block:: python

        cylinder_mesh = create_axisymmetric_mesh(
            profile_curve=lambda z: 1.0,
            height_bounds=(-1.0, 1.0),
            theta_bounds=(0.0, 2.0 * np.pi * (1 - 1.0 / 50)),
            Nheight=10,
            Ntheta=50,
            closed=True,
        )

    The UV coordinates are generated based on the node positions in the mesh and uniformly distributed in the range [0, 1] for the OpenGL texture mapping convention.
    Several UV mapping strategies are available and synthesized in the ``uv_layout`` parameter.
    The following options are available for ``uv_layout``:

    +-----------------+-------------------------+-------------------------+--------------------------+--------------------------+
    | uv_layout       | Node lower-left corner  | Node upper-left corner  | Node lower-right corner  | Node upper-right corner  |
    +=================+=========================+=========================+==========================+==========================+   
    | 0               | (0, 0)                  | (Nheight-1, 0)          | (0, Ntheta-1)            | (Nheight-1, Ntheta-1)    |
    +-----------------+-------------------------+-------------------------+--------------------------+--------------------------+
    | 1               | (0, 0)                  | (0, Ntheta-1)           | (Nheight-1, 0)           | (Nheight-1, Ntheta-1)    |
    +-----------------+-------------------------+-------------------------+--------------------------+--------------------------+
    | 2               | (Nheight-1, 0)          | (0, 0)                  | (Nheight-1, Ntheta-1)    | (0, Ntheta-1)            |
    +-----------------+-------------------------+-------------------------+--------------------------+--------------------------+
    | 3               | (0, Ntheta-1)           | (0, 0)                  | (Nheight-1, Ntheta-1)    | (Nheight-1, 0)           |
    +-----------------+-------------------------+-------------------------+--------------------------+--------------------------+
    | 4               | (0, Ntheta-1)           | (Nheight-1, Ntheta-1)   | (0, 0)                   | (Nheight-1, 0)           |
    +-----------------+-------------------------+-------------------------+--------------------------+--------------------------+
    | 5               | (Nheight-1, 0)          | (Nheight-1, Ntheta-1)   | (0, 0)                   | (0, Ntheta-1)            |
    +-----------------+-------------------------+-------------------------+--------------------------+--------------------------+
    | 6               | (Nheight-1, Ntheta-1)   | (0, Ntheta-1)           | (Nheight-1, 0)           | (0, 0)                   |
    +-----------------+-------------------------+-------------------------+--------------------------+--------------------------+
    | 7               | (Nheight-1, Ntheta-1)   | (Nheight-1, 0)          | (0, Ntheta-1)            | (0, 0)                   |
    +-----------------+-------------------------+-------------------------+--------------------------+--------------------------+

    The table above gives for the 4 corners of a image the corresponding node in the mesh.

    .. seealso:: 
    
        - :class:`pyblenderSDIC.meshes.TriMesh3D` for more information on how to visualize and manipulate the mesh.
        - https://github.com/Artezaru/py3dframe for details on the ``Frame`` class.

    Parameters
    ----------
    profile_curve : Callable[[float], float], optional
        A function that takes a single height coordinate z and returns a strictly positive radius.
        The default is a function that returns 1.0 for all z.
    
    frame : Frame, optional
        The reference frame for the mesh. Defaults to the identity frame.
    
    height_bounds : tuple[float, float], optional
        The lower and upper bounds for the height coordinate. Defaults to (0.0, 1.0).
        The order determines the direction of node placement.
    
    theta_bounds : tuple[float, float], optional
        The angular sweep in radians. Defaults to (-numpy.pi, numpy.pi).
        The order determines the angular direction of node placement.
    
    Nheight : int, optional
        Number of nodes along the height direction. Must be more than 1. Default is 10.
    
    Ntheta : int, optional
        Number of nodes along the angular direction. Must be more than 1. Default is 10.
    
    closed : bool, optional
        If True, the mesh is closed in the angular direction. Default is False.

    first_diagonal : bool, optional
        If True, the quad is split along the first diagonal (bottom-left to top-right). Default is True.
    
    direct : bool, optional
        If True, triangle vertices are ordered counterclockwise. Default is True.

    uv_layout : int, optional
        The UV mapping strategy. Default is 0.

    Returns
    -------
    TriMesh3D
        The generated axisymmetric mesh as a TriMesh3D object.
    """
    # Check the input parameters
    if not isinstance(frame, Frame):
        raise TypeError("frames must be a Frame object")
    
    if not isinstance(profile_curve, Callable):
        raise TypeError("profile_curve must be a callable function")
    
    height_bounds = numpy.array(height_bounds, dtype=numpy.float64).flatten()
    if height_bounds.shape != (2,):
        raise ValueError("height_bounds must be a 2D array of shape (2,)")
    if height_bounds[0] == height_bounds[1]:
        raise ValueError("height_bounds must be different")
    
    theta_bounds = numpy.array(theta_bounds, dtype=numpy.float64).flatten()
    if theta_bounds.shape != (2,):
        raise ValueError("theta_bounds must be a 2D array of shape (2,)")
    if theta_bounds[0] == theta_bounds[1]:
        raise ValueError("theta_bounds must be different")
    
    if not isinstance(Nheight, Integral) or Nheight < 2:
        raise ValueError("Nheight must be an integer greater than 1")
    Nheight = int(Nheight)

    if not isinstance(Ntheta, Integral) or Ntheta < 2:
        raise ValueError("Ntheta must be an integer greater than 1")
    Ntheta = int(Ntheta)

    if not isinstance(closed, bool):
        raise TypeError("closed must be a boolean")
    if closed and (abs(theta_bounds[0] - theta_bounds[1]) - 2.0*numpy.pi*(1 - 1/Ntheta)) > 1e-6:
        print("Warning: The theta bounds are not set to the closed condition (theta_max = theta_min + 2*pi*(1 - 1/Ntheta)). The mesh will be closed in the theta direction but the output can be unexpected.")

    if not isinstance(first_diagonal, bool):
        raise TypeError("first_diagonal must be a boolean")
    
    if not isinstance(direct, bool):
        raise TypeError("direct must be a boolean")
    
    if not isinstance(uv_layout, Integral) or uv_layout < 0 or uv_layout > 7:
        raise ValueError("uv_layout must be an integer between 0 and 7")
    uv_layout = int(uv_layout)
    
    # Generate the transform
    transform = Transform(input_frame=frame, output_frame=Frame())

    # Extract the parameters
    height_min = height_bounds[0]
    height_max = height_bounds[1]
    theta_min = theta_bounds[0]
    theta_max = theta_bounds[1]

    # Get the indices of the nodes in the array
    index = lambda ih, it: it*Nheight + ih

    # Set the UV mapping strategy (list of 3D points -> [(0,0) ; (0,Nt) ; (Nh,0) ; (Nh,Nt)])
    lower_left = numpy.array([0.0, 0.0, 0.0])
    lower_right = numpy.array([1.0, 0.0, 0.0])
    upper_left = numpy.array([0.0, 1.0, 0.0])
    upper_right = numpy.array([1.0, 1.0, 0.0])
    if uv_layout == 0:
        uv_mapping = [lower_left, lower_right, upper_left, upper_right]
    elif uv_layout == 1:
        uv_mapping = [lower_left, upper_left, lower_right, upper_right]
    elif uv_layout == 2:
        uv_mapping = [upper_left, upper_right, lower_left, lower_right]
    elif uv_layout == 3:
        uv_mapping = [upper_left, lower_left, upper_right, lower_right]
    elif uv_layout == 4:
        uv_mapping = [lower_right, lower_left, upper_right, upper_left]
    elif uv_layout == 5:
        uv_mapping = [lower_right, upper_right, lower_left, upper_left]
    elif uv_layout == 6:
        uv_mapping = [upper_right, upper_left, lower_right, lower_left]
    elif uv_layout == 7:
        uv_mapping = [upper_right, lower_right, upper_left, lower_left]

    # Generate the nodes
    uvmap = numpy.zeros((Nheight*Ntheta, 3))
    nodes = numpy.zeros((Nheight*Ntheta, 3))

    for it in range(Ntheta):
        for ih in range(Nheight):
            # Compute the coordinates of the node in the local frame.
            theta = theta_min + (theta_max - theta_min)*it/(Ntheta-1)
            height = height_min + (height_max - height_min)*ih/(Nheight-1)
            rho = profile_curve(height)
            x = rho*numpy.cos(theta)
            y = rho*numpy.sin(theta)
            z = height

            # Convert the local point to the global frame
            local_point = numpy.array([x, y, z]).reshape((3,1))
            nodes[index(ih, it), :] = transform.transform(point=local_point).flatten()

            # Compute the uvmap
            uvmap[index(ih, it), :] = uv_mapping[0] + ih/(Nheight-1)*(uv_mapping[2] - uv_mapping[0]) + it/(Ntheta-1)*(uv_mapping[1] - uv_mapping[0])


    # Generate the mesh
    elements = []

    for it in range(Ntheta-1):
        for ih in range(Nheight-1):
            if first_diagonal and direct:
                elements.append([index(ih, it), index(ih, it+1), index(ih+1, it+1)])
                elements.append([index(ih, it), index(ih+1, it+1), index(ih+1, it)])

            elif first_diagonal and not direct:
                elements.append([index(ih, it), index(ih+1, it+1), index(ih, it+1)])
                elements.append([index(ih, it), index(ih+1, it), index(ih+1, it+1)])

            elif not first_diagonal and direct:
                elements.append([index(ih, it), index(ih, it+1), index(ih+1, it)])
                elements.append([index(ih, it+1), index(ih+1, it+1), index(ih+1, it)])

            elif not first_diagonal and not direct:
                elements.append([index(ih, it), index(ih+1, it), index(ih, it+1)])
                elements.append([index(ih, it+1), index(ih+1, it), index(ih+1, it+1)])

    if closed:
        for ih in range(Nheight-1):
            if first_diagonal and direct:
                elements.append([index(ih, Ntheta-1), index(ih, 0), index(ih+1, 0)])
                elements.append([index(ih, Ntheta-1), index(ih+1, 0), index(ih+1, Ntheta-1)])

            elif first_diagonal and not direct:
                elements.append([index(ih, Ntheta-1), index(ih+1, 0), index(ih, 0)])
                elements.append([index(ih, Ntheta-1), index(ih+1, Ntheta-1), index(ih+1, 0)])

            elif not first_diagonal and direct:
                elements.append([index(ih, Ntheta-1), index(ih, 0), index(ih+1, Ntheta-1)])
                elements.append([index(ih, 0), index(ih+1, 0), index(ih+1, Ntheta-1)])

            elif not first_diagonal and not direct:
                elements.append([index(ih, Ntheta-1), index(ih+1, Ntheta-1), index(ih, 0)])
                elements.append([index(ih, 0), index(ih+1, Ntheta-1), index(ih+1, 0)])

    elements = numpy.array(elements)

    # Prepare the elements for the mesh
    points = nodes
    cells = [("triangle", elements)]
    point_data = {"uvmap": uvmap}

    # Create the mesh
    mesh = TriMesh3D(points=points, cells=cells, point_data=point_data)
    return mesh
