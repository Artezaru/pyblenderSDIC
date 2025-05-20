import sys
import subprocess
import os

print(os.getcwd())
subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."])

from pyblenderSDIC import install_packages, uninstall_packages

install_packages()