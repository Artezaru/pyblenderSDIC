.. currentmodule:: pyblenderSDIC.meshes

pyblenderSDIC.meshes.TriangleMesh3D
====================================

.. autoclass:: TriangleMesh3D


Create, Save and Load Meshes
--------------------------------

.. autosummary::
   :toctree: trimesh3d_generated/

   TriangleMesh3D.from_meshio
   TriangleMesh3D.from_vtk
   TriangleMesh3D.from_open3d
   TriangleMesh3D.from_dict
   TriangleMesh3D.from_json
   TriangleMesh3D.to_meshio
   TriangleMesh3D.to_vtk
   TriangleMesh3D.to_open3d
   TriangleMesh3D.to_dict
   TriangleMesh3D.to_json

Access and set TriangleMesh3D Data
-----------------------------------

.. autosummary::
   :toctree: trimesh3d_generated/

   TriangleMesh3D.vertices
   TriangleMesh3D.triangles
   TriangleMesh3D.uvmap
   TriangleMesh3D.set_vertices_uvmap
   TriangleMesh3D.Nvertices
   TriangleMesh3D.Ntriangles
   TriangleMesh3D.validate

Compute and access TriangleMesh3D Quantities
----------------------------------------------

.. autosummary::
   :toctree: trimesh3d_generated/

   TriangleMesh3D.compute_bounding_box
   TriangleMesh3D.bounding_box
   TriangleMesh3D.compute_volume
   TriangleMesh3D.volume
   TriangleMesh3D.compute_vertex_normals
   TriangleMesh3D.vertex_normals
   TriangleMesh3D.compute_triangle_normals
   TriangleMesh3D.triangle_normals
   TriangleMesh3D.compute_triangle_areas
   TriangleMesh3D.triangle_areas
   TriangleMesh3D.compute_triangle_centroids
   TriangleMesh3D.triangle_centroids

Calculate the properties of the mesh
---------------------------------------

.. autosummary::
   :toctree: trimesh3d_generated/

   TriangleMesh3D.is_edge_manifold
   TriangleMesh3D.is_edge_manifold_with_boundary
   TriangleMesh3D.is_vertex_manifold
   TriangleMesh3D.is_self_intersecting
   TriangleMesh3D.is_watertight
   TriangleMesh3D.is_orientable

Other Methods
--------------------------------

.. autosummary::
   :toctree: trimesh3d_generated/

   TriangleMesh3D.open3d_cast_rays
   TriangleMesh3D.cast_rays
   TriangleMesh3D.calculate_intersect_shape_functions
   TriangleMesh3D.calculate_intersect_coordinates
   TriangleMesh3D.visualize

   
   
