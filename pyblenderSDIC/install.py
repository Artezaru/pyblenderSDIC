import sys
import subprocess

def install_packages() -> None:
    # Use Blender pip to install packages
    subprocess.check_call([sys.executable, "-m", "pip", "install", "git+https://github.com/Artezaru/py3dframe.git"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "meshio"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "open3d"])
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



if __name__ == "__main__":
    install_packages()