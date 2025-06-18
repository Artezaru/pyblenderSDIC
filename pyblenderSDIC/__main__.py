import argparse
import sys
from .user_setup import UserSetup, WrongBlenderPath, PackageNotInstalled

def __main__() -> None:
    r"""
    Main entry point of the package.

    This method contains the script to run if the user enter the name of the package on the command line. 

    .. code-block:: console
        pyblenderSDIC
        
    """
    parser = argparse.ArgumentParser(
        description="Main entry point for the pyblenderSDIC package."
    )

    parser.add_argument(
        '--blender',
        type=str,
        help="Set the path to the Blender executable."
    )

    parser.add_argument(
        '--install',
        action='store_true',
        help="Install the package in Blender."
    )

    parser.add_argument(
        '--uninstall',
        action='store_true',
        help="Uninstall the package from Blender."
    )

    parser.add_argument(
        'script',
        nargs='?',
        help="Path to the Python script to run in Blender (default action)."
    )

    args = parser.parse_args()

    user_setup = UserSetup()

    if args.blender is not None:
        user_setup.blender_path = args.blender
        user_setup.check_blender_path()

    if args.install:
        user_setup.check_blender_path()
        user_setup.install_pyblenderSDIC()

    elif args.uninstall:
        user_setup.check_blender_path()
        user_setup.uninstall_pyblenderSDIC()

    elif args.script:
        user_setup.check_blender_path()
        user_setup.check_package_installed()
        user_setup.run_script_in_blender(args.script)

    else:
        parser.print_help()
        sys.exit(1)


def __main_gui__() -> None:
    r"""
    Graphical user interface entry point of the package.

    This method contains the script to run if the user enter the name of the package on the command line with the ``gui`` extension.

    .. code-block:: console
        pyblenderSDIC-gui
        
    """
    raise NotImplementedError("The graphical user interface entry point is not implemented yet.")

