from typing import Optional, Tuple, Sequence
import numpy
from scipy.ndimage import gaussian_filter

def create_speckle_BW_image(
    image_shape: Tuple[int, int],
    grain_size: Tuple[float, float] = (5.0, 5.0),
    grain_size_sigma: Optional[Tuple[float, float]] = None,
    density: float = 0.5,
    white_intensity: float = 255.0,
    ratio_BW: float = 1.0,
    ratio_BW_sigma: Optional[float] = None,
    invert: bool = False,
    seed: Optional[int] = None
    ) -> numpy.ndarray:
    r"""
    Generate a black-and-white speckle pattern with Gaussian variation in grain size and coverage ratio.

    This function creates a 2D (grayscale) or 3D (RGB) speckle image composed of elliptical grains 
    randomly distributed over the image area. Grain sizes and coverage ratios can be fixed or vary 
    according to Gaussian distributions defined by provided standard deviations. The image can be 
    inverted to produce white grains on a black background.

    Generate a default 512x512 grayscale speckle image with black grains on white background:

    .. code-block:: python

        import matplotlib.pyplot as plt

        # Set the parameters for the speckle image
        image_shape = (512, 512)
        grain_size = (5.0, 5.0)
        grain_size_sigma = (1.0, 1.0)
        density = 0.8
        seed = 42

        # Generate the speckle image with specified parameters
        speckle_img = create_speckle_BW_image(
            image_shape=image_shape,
            grain_size=grain_size,
            grain_size_sigma=grain_size_sigma,
            density=density,
            seed=seed
        )

    .. figure:: ../../../pyblenderSDIC/resources/doc/speckle_image.png
        :width: 400
        :align: center

        Example of a speckle image generated with default parameters.

    .. note::

        The size of the grains are clipped to a minimum of 1 pixel.

    Parameters
    ----------
    image_shape : tuple of int
        Dimensions of the output image.
        For grayscale: (height, width).
        For RGB: (height, width, 3).

    grain_size : tuple of float, optional
        Mean grain size (in pixels) along the (x, y) axes.
        Default is (5.0, 5.0).

    grain_size_sigma : tuple of float, optional
        Standard deviation of grain size along the (x, y) axes.
        If None, grain size is fixed at the mean value.
        Default is None.

    density : float, optional
        Relative density of grains in the image, expressed as a strictly positive float.
        This parameter influences the number of grains drawn.
        Default is 0.5.

    white_intensity : float, optional
        Intensity value for white color in the image.
        Must be strictly positive.
        Default is 255.0.

    ratio_BW : float, optional
        Mean proportion of the image covered by grains.
        Interpreted as black grain coverage if `invert=False`, white grain coverage otherwise.
        Must be in the range [0, 1].
        Default is 1.0.

    ratio_BW_sigma : float, optional
        Standard deviation of the coverage ratio.
        If None, the coverage ratio is fixed.
        Default is None.

    invert : bool, optional
        If True, grains are white on a black background.
        If False, grains are black on a white background.
        Default is False.

    seed : int or None, optional
        Seed for the random number generator to ensure reproducibility.
        If None, the RNG is not seeded.
        Must be a non-negative integer less than 2^32.
        Default is None.

    Returns
    -------
    numpy.ndarray
        Generated speckle image as a 2D array (grayscale) or 3D array (RGB) with float values in [0, white_intensity].
    """
    # Check the input parameters
    if not isinstance(image_shape, Sequence):
        raise ValueError("image_shape must be a sequence (tuple or list).")
    if len(image_shape) not in (2, 3):
        raise ValueError("image_shape must be a 2D or 3D shape.")
    if len(image_shape) == 3 and image_shape[2] != 3:
        raise ValueError("For 3D image_shape, the last dimension must be 3 (RGB).")
    
    if not isinstance(grain_size, (Sequence, numpy.ndarray)) or len(grain_size) != 2:
        raise ValueError("grain_size must be a sequence of two floats (mean_x, mean_y).")
    grain_size = numpy.asarray(grain_size, dtype=numpy.float64).flatten()
    if not numpy.all(grain_size > 0) or not numpy.all(numpy.isfinite(grain_size)):
        raise ValueError("grain_size must be positive finite values.")
    
    if grain_size_sigma is not None:
        if not isinstance(grain_size_sigma, (Sequence, numpy.ndarray)) or len(grain_size_sigma) != 2:
            raise ValueError("grain_size_sigma must be a sequence of two floats (sigma_x, sigma_y).")
        grain_size_sigma = numpy.asarray(grain_size_sigma, dtype=numpy.float64).flatten()
        if not numpy.all(grain_size_sigma >= 0) or not numpy.all(numpy.isfinite(grain_size_sigma)):
            raise ValueError("grain_size_sigma must be non-negative finite values.")

    if not isinstance(density, float) or density <= 0:
        raise ValueError("density must be a strictly positive float.")

    if not isinstance(white_intensity, float) or white_intensity <= 0:
        raise ValueError("white_intensity must be a strictly positive float.")

    if not isinstance(ratio_BW, float) or not (0 <= ratio_BW <= 1):
        raise ValueError("ratio_BW must be a float in [0, 1].")

    if ratio_BW_sigma is not None:
        if not isinstance(ratio_BW_sigma, float) or ratio_BW_sigma < 0:
            raise ValueError("ratio_BW_sigma must be a non-negative float.")

    if not isinstance(invert, bool):
        raise ValueError("invert must be a boolean.")
    
    if seed is not None and not isinstance(seed, int):
        raise ValueError("seed must be an integer or None.")
    if seed is not None and (seed < 0 or seed > 2**32 - 1):
        raise ValueError("seed must be a non-negative integer less than 2^32.")
    
    # Select the random seed
    if seed is not None:
        numpy.random.seed(seed)

    # Check if the image is RGB
    is_rgb = len(image_shape) == 3
    height, width = image_shape[:2]
    if not invert:
        image = white_intensity * numpy.ones((height, width), dtype=float)
    else:
        image = numpy.zeros((height, width), dtype=float)
    total_pixels = height * width
    average_area = numpy.pi * grain_size[0] * grain_size[1]
    num_grains = int((density * total_pixels) / average_area)

    for _ in range(num_grains):
        # Determine individual grain size
        rx = max(1, numpy.random.normal(grain_size[0], grain_size_sigma[0]) if grain_size_sigma is not None else grain_size[0])
        ry = max(1, numpy.random.normal(grain_size[1], grain_size_sigma[1]) if grain_size_sigma is not None else grain_size[1])

        # Determine the color of the grain
        ratio = max(0, min(1, numpy.random.normal(ratio_BW, ratio_BW_sigma) if ratio_BW_sigma is not None else ratio_BW))
        if not invert:
            intensity = (1 - ratio) * white_intensity
        else:
            intensity = ratio * white_intensity

        # Generate an elliptical mask
        x_rad, y_rad = int(numpy.ceil(rx)), int(numpy.ceil(ry))
        y, x = numpy.ogrid[-y_rad:y_rad, -x_rad:x_rad]
        mask = (x / rx) ** 2 + (y / ry) ** 2 <= 1

        # Random position
        cx = numpy.random.randint(x_rad, width - x_rad)
        cy = numpy.random.randint(y_rad, height - y_rad)

        # Apply grain
        region = image[cy - y_rad:cy + y_rad, cx - x_rad:cx + x_rad]
        mask = mask[:region.shape[0], :region.shape[1]]
        region[mask] = intensity

    # Convert to RGB if requested
    if is_rgb:
        image = numpy.stack([image]*3, axis=-1)

    return image
