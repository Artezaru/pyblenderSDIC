import sys

try:
    __import__('pyblenderSDIC')
    installed = True
except ImportError as e:
    installed = False

sys.exit(0 if installed else 1)