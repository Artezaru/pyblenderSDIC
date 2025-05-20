from .material_bsdf import MaterialBSDF

def get_mirror_material() -> MaterialBSDF:
    r"""
    Returns a mirror material for use in Blender.

    This material is a perfect mirror with no roughness and a refractive index of 1.50.

    .. code-block:: python

        {
            "base_color": (1.0, 1.0, 1.0, 1.0), # White color
            "roughness": 0.0, # No roughness
            "metallic": 1.0, # Fully metallic
            "IOR": 1.5, # Refractive index between 1.45 and 1.55
            "transmission_weight": 0.0, # No transmission
            "alpha": 1.0, # Fully opaque
        }

    Returns
    -------
    MaterialBSDF
        A MaterialBSDF object representing a mirror material.
    """
    return MaterialBSDF(
        base_color=(1.0, 1.0, 1.0, 1.0), # White color
        roughness=0.0, # No roughness
        metallic=1.0, # Fully metallic
        IOR=1.5, # Refractive index between 1.45 and 1.55
        transmission_weight=0.0, # No transmission
        alpha=1.0, # Fully opaque
    )


def get_steel_material() -> MaterialBSDF:
    r"""
    Returns a steel-like material.

    Polished steel is metallic, slightly rough, and has a neutral gray tint.

    .. code-block:: python

        {
            "base_color": (0.55, 0.55, 0.55, 1.0), # Neutral gray
            "roughness": 0.2, # Slightly rough
            "metallic": 1.0, # Fully metallic
            "IOR": 2.5, # Approximate for steel
            "transmission_weight": 0.0, # No transmission
            "alpha": 1.0, # Fully opaque
        }

    Returns
    -------
    MaterialBSDF
        A MaterialBSDF object representing polished steel.
    """
    return MaterialBSDF(
        base_color=(0.55, 0.55, 0.55, 1.0),  # Neutral gray
        roughness=0.2,
        metallic=1.0,
        IOR=2.5,  # Approximate for steel
        transmission_weight=0.0,
        alpha=1.0,
    )


def get_titanium_material() -> MaterialBSDF:
    r"""
    Returns a titanium-like material.

    Titanium has a slightly bluish-gray hue and is less reflective than polished steel.

    .. code-block:: python

        {
            "base_color": (0.5, 0.52, 0.6, 1.0), # Slight bluish-gray
            "roughness": 0.25, # Slightly rough
            "metallic": 1.0, # Fully metallic
            "IOR": 2.5, # Close approximation
            "transmission_weight": 0.0, # No transmission
            "alpha": 1.0, # Fully opaque
        }

    Returns
    -------
    MaterialBSDF
        A MaterialBSDF object representing titanium.
    """
    return MaterialBSDF(
        base_color=(0.5, 0.52, 0.6, 1.0),  # Slight bluish-gray
        roughness=0.25,
        metallic=1.0,
        IOR=2.5,  # Close approximation
        transmission_weight=0.0,
        alpha=1.0,
    )


def get_iron_material() -> MaterialBSDF:
    r"""
    Returns an iron-like material.

    Iron is dark gray and typically more rough than polished steel.

    .. code-block:: python

        {
            "base_color": (0.3, 0.3, 0.3, 1.0), # Dark gray
            "roughness": 0.35, # More rough than polished steel
            "metallic": 1.0, # Fully metallic
            "IOR": 2.9, # Iron has a higher refractive index
            "transmission_weight": 0.0, # No transmission
            "alpha": 1.0, # Fully opaque
        }

    Returns
    -------
    MaterialBSDF
        A MaterialBSDF object representing iron.
    """
    return MaterialBSDF(
        base_color=(0.3, 0.3, 0.3, 1.0),  # Dark gray
        roughness=0.35,
        metallic=1.0,
        IOR=2.9,  # Iron has a higher refractive index
        transmission_weight=0.0,
        alpha=1.0,
    )


def get_copper_material() -> MaterialBSDF:
    r"""
    Returns a copper-like material.

    Copper has a distinctive reddish-orange color, is highly reflective, and fully metallic.

    .. code-block:: python

        {
            "base_color": (0.955, 0.637, 0.538, 1.0), # Reddish-orange typical of copper
            "roughness": 0.2, # Slight surface roughness
            "metallic": 1.0, # Fully metallic
            "IOR": 2.7, # Approximate value for copper
            "transmission_weight": 0.0, # No transmission
            "alpha": 1.0, # Fully opaque
        }

    Returns
    -------
    MaterialBSDF
        A MaterialBSDF object representing copper.
    """
    return MaterialBSDF(
        base_color=(0.955, 0.637, 0.538, 1.0),  # Reddish-orange typical of copper
        roughness=0.2,
        metallic=1.0,
        IOR=2.7,  # Approximate value for copper
        transmission_weight=0.0,
        alpha=1.0,
    )

