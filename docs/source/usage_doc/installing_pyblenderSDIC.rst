Installing pyblenderSDIC in Blender
=================================================

When using the Blender Python Interpreter
--------------------------------------------------

The scripts made with ``pyblenderSDIC`` are made to be run in the Blender Python interpreter.
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

.. seealso::

    :doc:`running_scripts` for more information on how to run scripts in Blender.

If the installation of ``pyblenderSDIC`` don't install all the dependencies, you can add the following lines to the script:

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

Now the package and its dependencies should be installed in the Blender Python interpreter, and you can use it to create stereo-digital images for correlation using Blender.
Simply import the package in your scripts:

.. code-block:: python

   from pyblenderSDIC import BlenderExperiment

   """ Some code using BlenderExperiment Here """

And run your scripts in the Blender Python interpreter as described in the :doc:`running_scripts` section.


When using the Blender GUI Scripting Console
--------------------------------------------------

You can also run scripts directly in the Blender GUI using the scripting console.

To install the package in the Blender Python interpreter, you can copy this script in the Blender GUI Scripting Console:

.. code-block:: python

    import sys
    import subprocess

    # Name of the package and GitHub repo URL
    paquet = "pyblenderSDIC"
    repo_url = f"git+https://github.com/Artezaru/{paquet}.git"

    # Get the user site-packages path
    user_site = subprocess.check_output([sys.executable, "-m", "site", "--user-site"], text=True).strip()

    # Add this path to sys.path if necessary
    if user_site not in sys.path:
        sys.path.append(user_site)
    
    # Install the package via pip in subprocess (with auto confirmation)
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", repo_url])
        print(f"{paquet} installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error installing {paquet}: {e}")
    
    # Test the installation by importing the module
    try:
        import pyblenderSDIC
        print("pyblenderSDIC is well installed in Blender Python interpreter")
    except ImportError as e:
        print(f"Error importing pyblenderSDIC: {e}")


By default, you cannot directly install packages in the Blender Python interpreter using pip package from the Blender GUI Scripting Console.
The package will be installed in the user site-packages directory, which is not the same as the Blender Python interpreter site-packages directory.

Then you need to add the following lines at the beggining of all your scripts to work properly:

.. code-block:: python

    import sys
    import subprocess

    # Get the user site-packages path
    user_site = subprocess.check_output([sys.executable, "-m", "site", "--user-site"], text=True).strip()

    # Add this path to sys.path if necessary
    if user_site not in sys.path:
        sys.path.append(user_site)