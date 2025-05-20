import numpy as np
import pytest
from pyblenderSDIC.patterns import create_speckle_BW_image

def test_output_shape_and_type():
    img = create_speckle_BW_image(image_shape=(512, 512))
    assert isinstance(img, np.ndarray)
    assert img.ndim == 2
    assert img.shape == (512, 512)
    assert img.dtype == float or img.dtype == np.float64
    assert img.min() >= 0 and img.max() <= 255

def test_output_shape_rgb():
    img = create_speckle_BW_image(image_shape=(128, 128, 3))
    assert isinstance(img, np.ndarray)
    assert img.ndim == 3
    assert img.shape == (128, 128, 3)
    assert img.min() >= 0 and img.max() <= 255

@pytest.mark.parametrize("missing_or_invalid_shape", [
    None,        # missing
    (512,),      # too short
    (512, 512, 1), # wrong 3rd dim
    "512,512",   # wrong type
    512         # wrong type
])
def test_invalid_image_shape(missing_or_invalid_shape):
    if missing_or_invalid_shape is None:
        with pytest.raises(TypeError):
            create_speckle_BW_image()
    else:
        with pytest.raises(ValueError):
            create_speckle_BW_image(image_shape=missing_or_invalid_shape)

@pytest.mark.parametrize("invalid_grain_size", [
    (5,), (5, 5, 5), "5,5", -1, (0, 0)
])
def test_invalid_grain_size(invalid_grain_size):
    with pytest.raises(ValueError):
        create_speckle_BW_image(image_shape=(512,512), grain_size=invalid_grain_size)

@pytest.mark.parametrize("invalid_density", [-1, 0, "high"])
def test_invalid_density(invalid_density):
    with pytest.raises(ValueError):
        create_speckle_BW_image(image_shape=(512,512), density=invalid_density)

def test_invert_option_changes_image():
    img_normal = create_speckle_BW_image(image_shape=(128,128), invert=False, seed=42)
    img_invert = create_speckle_BW_image(image_shape=(128,128), invert=True, seed=42)
    assert not np.allclose(img_normal, img_invert)

def test_seed_reproducibility():
    img1 = create_speckle_BW_image(image_shape=(256,256), seed=999)
    img2 = create_speckle_BW_image(image_shape=(256,256), seed=999)
    assert np.allclose(img1, img2)

def test_ratio_BW_and_ratio_BW_sigma_effect():
    img1 = create_speckle_BW_image(image_shape=(256,256), ratio_BW=0.1, seed=1)
    img2 = create_speckle_BW_image(image_shape=(256,256), ratio_BW=0.9, seed=1)
    assert not np.allclose(img1, img2)
    img3 = create_speckle_BW_image(image_shape=(256,256), ratio_BW=0.5, ratio_BW_sigma=0.2, seed=1)
    assert img3.shape == (256, 256)

