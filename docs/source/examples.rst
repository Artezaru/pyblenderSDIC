Examples
================

Several examples are provided in the ``examples`` folder of the package.
These examples demonstrate how to use the package to generate stereo-digital images for correlation using Blender.

.. toctree::
   :maxdepth: 1
   :caption: Examples

   ./usage_doc/create_meshes
   ./usage_doc/simple_example
   ./usage_doc/SDIC_example
   ./usage_doc/mirror_example

In this section, we will provide a brief overview of how to create a simple Blender experiment using the package.
The example we want to create is the following:

- A Iron cylinder with a diameter of 100 mm and a height of 200 mm is deformed 

1. Create a new Blender experiment:

First, you need to create a new Blender experiment. You can do this by creating an instance of the `:class:pyblenderSDIC.BlenderExperiment` class.

.. code-block:: python

    from pyblenderSDIC import BlenderExperiment

    experiment = BlenderExperiment()


