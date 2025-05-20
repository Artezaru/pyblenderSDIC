from importlib.resources import files

def get_mouchtichu_path(format: str = "TIFF") -> str:
    """
    Returns the path to the mouchtichu pattern.

    .. figure:: ../../../pyblenderSDIC/resources/patterns/mouchtichu.png
        :width: 500
        :align: center
        
        Mouchtichu pattern

    Parameters
    ----------
    format : str
        The format of the pattern. Currently, only "TIFF" and "PNG" are supported.

    Returns
    -------
    str
        The path to the mouchtichu pattern.
    """
    if format == "TIFF":
        return str(files("pyblenderSDIC.resources") / "patterns" / "mouchtichu.tiff")
    if format == "PNG":
        return str(files("pyblenderSDIC.resources") / "patterns" / "mouchtichu.png")
    else:
        raise ValueError("Format not supported. Use 'TIFF' or 'PNG'.")
    

def get_speckle_path(format: str = "TIFF") -> str:
    """
    Returns the path to the speckle pattern.

    .. figure:: ../../../pyblenderSDIC/resources/patterns/speckle.png
        :width: 500
        :align: center
        
        Speckle pattern

    Parameters
    ----------
    format : str
        The format of the pattern. Currently, only "TIFF" and "PNG" are supported.

    Returns
    -------
    str
        The path to the speckle pattern.
    """
    if format == "TIFF":
        return str(files("pyblenderSDIC.resources") / "patterns" / "speckle.tiff")
    if format == "PNG":
        return str(files("pyblenderSDIC.resources") / "patterns" / "speckle.png")
    else:
        raise ValueError("Format not supported. Use 'TIFF' or 'PNG'.")


