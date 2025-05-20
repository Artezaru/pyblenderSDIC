.. currentmodule:: pyblenderSDIC.meshes

pyblenderSDIC.meshes.TriMesh3D
===============================

.. autoclass:: TriMesh3D


Create, Save and Load Meshes
--------------------------------

.. autosummary::
   :toctree: trimesh3d_generated/

   TriMesh3D.from_meshio
   TriMesh3D.load_from_vtk
   TriMesh3D.save_to_vtk
   TriMesh3D.load_from_dict
   TriMesh3D.save_to_dict
   TriMesh3D.load_from_json
   TriMesh3D.save_to_json


Access TriMesh3D Data
--------------------------------

.. autosummary::
   :toctree: trimesh3d_generated/

   TriMesh3D.nodes
   TriMesh3D.elements
   TriMesh3D.uvmap
   TriMesh3D.get_nodes
   TriMesh3D.set_nodes
   TriMesh3D.get_elements
   TriMesh3D.set_elements
   TriMesh3D.get_uvmap
   TriMesh3D.set_uvmap
   TriMesh3D.get_uvmap2D
   TriMesh3D.set_uvmap2D
   TriMesh3D.Nnodes
   TriMesh3D.Nelements
   TriMesh3D.validate


Other Methods
--------------------------------

.. autosummary::
   :toctree: trimesh3d_generated/

   TriMesh3D.construct_open3d_mesh
   TriMesh3D.visualize
   TriMesh3D.open3d_cast_ray
   TriMesh3D.compute_intersect_points
   TriMesh3D.compute_intersect_points_coordinates
   TriMesh3D.compute_element_normals
   TriMesh3D.compute_element_centroids
   TriMesh3D.compute_element_areas

   
   
