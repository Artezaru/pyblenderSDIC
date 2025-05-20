Usage
==============

Other Documentation will be added later.
See the folder ``examples`` of the package for examples of usage.

Before First Use
=================

All the scripts made with ``pyblenderSDIC`` are made to be run in the Blender Python interpreter.
First, you need to install the package in the Blender Python interpreter.

To do this, create a default ``install_pyblenderSDIC.py`` with the following content:

.. code-block:: python

   import sys
   import subprocess

   subprocess.check_call([sys.executable, "-m", "pip", "install", "git+https://github.com/Artezaru/pyblenderSDIC.git"])


Then, run the script in the Blender Python interpreter. You can do this by opening a terminal and running the following command:

.. code-block:: console

   ~/blender/blender --background --python install_pyblenderSDIC.py


Where ``~/blender/blender`` is the path to the Blender executable. Replace it with the path to your Blender installation.
If the installation of ``pyblenderSDIC`` don't install all the dependencies, you can add the following line to the script:

.. code-block:: python

   import sys
   import subprocess

   subprocess.check_call([sys.executable, "-m", "pip", "install", "git+https://github.com/Artezaru/pyblenderSDIC.git"])

   subprocess.check_call([sys.executable, "-m", "pip", "install", "git+https://github.com/Artezaru/py3dframe.git"])
   subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy"])
   subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib"])
   subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python-headless."])
   subprocess.check_call([sys.executable, "-m", "pip", "install", "open3d"])
   subprocess.check_call([sys.executable, "-m", "pip", "install", "meshio"])

   import numpy
   print("numpy is well installed in Blender Python interpreter")
   import matplotlib
   print("matplotlib is well installed in Blender Python interpreter")
   import cv2
   print("opencv-python-headless is well installed in Blender Python interpreter")
   import open3d
   print("open3d is well installed in Blender Python interpreter")
   import meshio
   print("meshio is well installed in Blender Python interpreter")
   import py3dframe
   print("py3dframe is well installed in Blender Python interpreter")
   import pyblenderSDIC
   print("pyblenderSDIC is well installed in Blender Python interpreter")


How to run a script with the Blender Python interpreter
=======================================================

To run a script using the Blender Python interpreter, you need to use the Blender executable and pass the script as an argument.

Assume the Blender executable is located at:

.. code-block:: console

    ~/blender/blender


And your project is structured as follows:

.. code-block:: console

    .
    ├── blender
    │   └── blender
    ├── Documents
    │   └── My_project
    │       └── scripts
    │           └── script1.py

You are currently in the ``~/Documents/My_project`` directory and want to run the script ``script1.py``, located at:

- Absolute path: ``~/Documents/My_project/scripts/script1.py``
- Relative path: ``scripts/script1.py``

To run the script using the Blender Python interpreter and launch the Blender GUI, use:

.. code-block:: console

    ~/blender/blender --python scripts/script1.py

If you want to run the script without opening the GUI (in background mode), use:

.. code-block:: console

    ~/blender/blender --background --python scripts/script1.py

This will execute the script using Blender's Python interpreter without showing the Blender interface.
