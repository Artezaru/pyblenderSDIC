import numpy

from py3dframe import Frame, Transform
from typing import Callable
from numbers import Integral

from .trimesh3d import TriMesh3D

def create_xy_heightmap_mesh(
    height_function: Callable[[float, float], float] = lambda x, y: 0.0,
    frame: Frame = Frame(),
    x_bounds: tuple[float, float] = (-1.0, 1.0),
    y_bounds: tuple[float, float] = (-1.0, 1.0),
    Nx: int = 10,
    Ny: int = 10,
    first_diagonal: bool = True,
    direct: bool = True,
    uv_layout: int = 0,
) -> TriMesh3D:
    r"""
    Create a 3D XY-plane mesh with variable height defined by a surface function.

    The surface is defined by a function that takes two arguments (x and y) and returns a scalar height z.
    The returned value is interpreted as the vertical position of the surface at that point.

    The ``frame`` parameter defines the orientation and the position of the mesh in 3D space.
    The (x, y) grid is centered on the frame origin, lying in the local XY plane.
    The height (z) is applied along the local Z-axis of the frame.

    The ``x_bounds`` and ``y_bounds`` parameters define the rectangular domain over which the mesh is generated.
    ``Nx`` and ``Ny`` determine the number of nodes along the x and y directions, respectively.
    Nodes are uniformly distributed across both directions.

    .. note::

        - ``Nx`` and ``Ny`` refer to the number of **nodes**, not segments.
        - The height function must return a finite scalar value for every (x, y) in the specified domain.

    For example, the following code generates a sinusoidal surface mesh centered on the world origin:

    .. code-block:: python

        from pyblenderSDIC.meshes import create_xy_heightmap_mesh
        import numpy as np

        surface_mesh = create_xy_heightmap_mesh(
            height_function=lambda x, y: 0.5 * np.sin(np.pi * x) * np.cos(np.pi * y),
            x_bounds=(-1.0, 1.0),
            y_bounds=(-1.0, 1.0),
            Nx=50,
            Ny=50,
        )

        surface_mesh.visualize()

    .. figure:: ../../../pyblenderSDIC/resources/doc/sinus_surface_mesh.png
        :width: 400
        :align: center

        Sinusoidal height map over a square domain centered at the origin.

    Nodes are ordered first in y (indexed by ``i_Y``), then in x (indexed by ``i_X``).
    So the node at y index ``i_Y`` and x index ``i_X`` (both starting from 0) is located at:

    .. code-block:: python

        mesh.nodes[i_Y * Nx + i_X, :]

    Each quadrilateral element is defined by the nodes:

    - :math:`(i_X, i_Y)`
    - :math:`(i_X + 1, i_Y)`
    - :math:`(i_X + 1, i_Y + 1)`
    - :math:`(i_X, i_Y + 1)`

    This quadrilateral is split into two triangles depending on the value of ``first_diagonal``:

    - If ``first_diagonal`` is ``True``:

        - Triangle 1: :math:`(i_X, i_Y)`, :math:`(i_X, i_Y + 1)`, :math:`(i_X + 1, i_Y + 1)`
        - Triangle 2: :math:`(i_X, i_Y)`, :math:`(i_X + 1, i_Y + 1)`, :math:`(i_X + 1, i_Y)`

    - If ``first_diagonal`` is ``False``:

        - Triangle 1: :math:`(i_X, i_Y)`, :math:`(i_X, i_Y + 1)`, :math:`(i_X + 1, i_Y)`
        - Triangle 2: :math:`(i_X, i_Y + 1)`, :math:`(i_X + 1, i_Y + 1)`, :math:`(i_X + 1, i_Y)`

    By default, the triangles are oriented counterclockwise (direct orientation) for an observer looking from above.
    Set ``direct=False`` to reverse the orientation.

    The UV coordinates are generated based on the node positions in the mesh and uniformly distributed in the range [0, 1] for the OpenGL texture mapping convention.
    Several UV mapping strategies are available and synthesized in the ``uv_layout`` parameter.
    The following options are available for ``uv_layout``:

    +-----------------+-------------------------+-------------------------+--------------------------+--------------------------+
    | uv_layout       | Node lower-left corner  | Node upper-left corner  | Node lower-right corner  | Node upper-right corner  |
    +=================+=========================+=========================+==========================+==========================+   
    | 0               | (0, 0)                  | (Nx-1, 0)               | (0, Ny-1)                | (Nx-1, Ny-1)             |
    +-----------------+-------------------------+-------------------------+--------------------------+--------------------------+
    | 1               | (0, 0)                  | (0, Ny-1)               | (Nx-1, 0)                | (Nx-1, Ny-1)             |
    +-----------------+-------------------------+-------------------------+--------------------------+--------------------------+
    | 2               | (Nx-1, 0)               | (0, 0)                  | (Nx-1, Ny-1)             | (0, Ny-1)                |
    +-----------------+-------------------------+-------------------------+--------------------------+--------------------------+
    | 3               | (0, Ny-1)               | (0, 0)                  | (Nx-1, Ny-1)             | (Nx-1, 0)                |
    +-----------------+-------------------------+-------------------------+--------------------------+--------------------------+
    | 4               | (0, Ny-1)               | (Nx-1, Ny-1)            | (0, 0)                   | (Nx-1, 0)                |
    +-----------------+-------------------------+-------------------------+--------------------------+--------------------------+
    | 5               | (Nx-1, 0)               | (Nx-1, Ny-1)            | (0, 0)                   | (0, Ny-1)                |
    +-----------------+-------------------------+-------------------------+--------------------------+--------------------------+
    | 6               | (Nx-1, Ny-1)            | (0, Ny-1)               | (Nx-1, 0)                | (0, 0)                   |
    +-----------------+-------------------------+-------------------------+--------------------------+--------------------------+
    | 7               | (Nx-1, Ny-1)            | (Nx-1, 0)               | (0, Ny-1)                | (0, 0)                   |
    +-----------------+-------------------------+-------------------------+--------------------------+--------------------------+

    The table above gives for the 4 corners of a image the corresponding node in the mesh.

    .. seealso:: 
        
        - :class:`pyblenderSDIC.meshes.TriMesh3D` for more information on how to visualize and manipulate the mesh.
        - https://github.com/Artezaru/py3dframe for details on the ``Frame`` class.

    Parameters
    ----------
    height_function : Callable[[float, float], float]
        A function that takes x and y coordinates and returns the corresponding height (z).
        
    frame : Frame, optional
        The reference frame for the mesh. Defaults to the identity frame.
        
    x_bounds : tuple[float, float], optional
        The lower and upper bounds for the x coordinate. Default is (-1.0, 1.0).
        
    y_bounds : tuple[float, float], optional
        The lower and upper bounds for the y coordinate. Default is (-1.0, 1.0).

    Nx : int, optional
        Number of nodes along the x direction. Must be more than 1. Default is 10.

    Ny : int, optional
        Number of nodes along the y direction. Must be more than 1. Default is 10.

    first_diagonal : bool, optional
        If True, the quad is split along the first diagonal (bottom-left to top-right). Default is True.

    direct : bool, optional
        If True, triangle vertices are ordered counterclockwise. Default is True.

    uv_layout : int, optional
        The UV mapping strategy. Default is 0.

    Returns
    -------
    TriMesh3D
        The generated surface mesh as a TriMesh3D object.
    """
    # Check the input parameters
    if not isinstance(frame, Frame):
        raise TypeError("frames must be a Frame object")
    
    if not isinstance(height_function, Callable):
        raise TypeError("height_function must be a callable function")
    
    x_bounds = numpy.array(x_bounds, dtype=numpy.float64).flatten()
    if x_bounds.shape != (2,):
        raise ValueError("x_bounds must be a 2D array of shape (2,)")
    if x_bounds[0] == x_bounds[1]:
        raise ValueError("x_bounds must be different")
    
    y_bounds = numpy.array(y_bounds, dtype=numpy.float64).flatten()
    if y_bounds.shape != (2,):
        raise ValueError("y_bounds must be a 2D array of shape (2,)")
    if y_bounds[0] == y_bounds[1]:
        raise ValueError("y_bounds must be different")
    
    if not isinstance(Nx, Integral) or Nx < 2:
        raise ValueError("Nx must be an integer greater than 1")
    Nx = int(Nx)

    if not isinstance(Ny, Integral) or Ny < 2:
        raise ValueError("Ny must be an integer greater than 1")
    Ny = int(Ny)

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
    x_min = x_bounds[0]
    x_max = x_bounds[1]
    y_min = y_bounds[0]
    y_max = y_bounds[1]

    # Get the indices of the nodes in the array
    index = lambda ih, it: it*Nx + ih

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
    uvmap = numpy.zeros((Nx*Ny, 3))
    nodes = numpy.zeros((Nx*Ny, 3))

    for it in range(Ny):
        for ih in range(Nx):
            # Compute the coordinates of the node in the local frame.
            y = y_min + (y_max - y_min)*it/(Ny-1)
            x = x_min + (x_max - x_min)*ih/(Nx-1)
            z = height_function(x, y)

            # Convert the local point to the global frame
            local_point = numpy.array([x, y, z]).reshape((3,1))
            nodes[index(ih, it), :] = transform.transform(point=local_point).flatten()

            # Compute the uvmap
            uvmap[index(ih, it), :] = uv_mapping[0] + ih/(Nx-1)*(uv_mapping[2] - uv_mapping[0]) + it/(Ny-1)*(uv_mapping[1] - uv_mapping[0])


    # Generate the mesh
    elements = []

    for it in range(Ny-1):
        for ih in range(Nx-1):
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

    elements = numpy.array(elements)

    # Prepare the elements for the mesh
    points = nodes
    cells = [("triangle", elements)]
    point_data = {"uvmap": uvmap}

    # Create the mesh
    mesh = TriMesh3D(points=points, cells=cells, point_data=point_data)
    return mesh
