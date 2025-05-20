API Reference
==============

The package ``pyblenderSDIC`` is composed of the following functions, classes, and modules:

- The class ``pyblenderSDIC.BlenderExperiment`` is the main class of the package. It contains all the functions and methods to create a Blender experiment in order to generate virtual images.
- The class ``pyblenderSDIC.Camera`` is used to manipulate cameras.
- The class ``pyblenderSDIC.SpotLight`` is used to manipulate spot lights.

- The submodule ``meshes`` contains class and functions for creating and manipulating 3D triangles meshes. 
- The submodule ``materials`` contains classes and functions for creating and manipulating Principled BSDF materials. It also contains a set of default materials that can be used in the package.
- The submodule ``patterns`` contains a set of default patterns (texture) that can be used in the package.

.. toctree:: 
    :maxdepth: 1
    :caption: BlenderExperiment Module

    ./api_doc/blender_experiment.rst
    ./api_doc/camera.rst
    ./api_doc/spotlight.rst

.. toctree:: 
    :maxdepth: 1
    :caption: meshes Module

    ./api_doc/trimesh3d.rst
    ./api_doc/intersect_points.rst
    ./api_doc/create_axisymmetric_mesh.rst
    ./api_doc/create_xy_heightmap_mesh.rst


.. toctree:: 
    :maxdepth: 1
    :caption: materials Module

    ./api_doc/material_bsdf.rst
    ./api_doc/default_materials.rst


.. toctree::
    :maxdepth: 1
    :caption: patterns Module

    ./api_doc/create_speckle_BW_image.rst
    ./api_doc/default_patterns.rst

To learn how to use the package effectively, refer to the documentation :doc:`../usage`.