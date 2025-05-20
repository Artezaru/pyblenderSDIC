import sys
import subprocess

def install_packages() -> None:
    # Use Blender pip to install packages
    subprocess.check_call([sys.executable, "-m", "pip", "install", "git+https://github.com/Artezaru/py3dframe.git"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "meshio"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "open3d"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python-headless"])
    print("Packages installed successfully.")

    # Test imports
    import numpy
    print("numpy imported successfully.")
    import meshio
    print("meshio imported successfully.")
    import open3d
    print("open3d imported successfully.")
    import py3dframe
    print("py3dframe imported successfully.")
    import cv2
    print("opencv-python-headless imported successfully.")
    import matplotlib
    print("matplotlib imported successfully.")



if __name__ == "__main__":
    install_packages()