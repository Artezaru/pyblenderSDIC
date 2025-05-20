import numpy as np
import pytest
from pyblenderSDIC import Camera


@pytest.fixture
def camera():
    """Fixture to create a fresh Camera with some data for each test."""
    cam = Camera()
    cam.frame.origin = np.array([1.0, 2.0, 3.0]).reshape(3, 1)
    cam.frame.quaternion = np.array([0.707, 0.0, 0.707, 0.0]) / np.sqrt(2*0.707**2)
    cam.pixel_size = np.array([0.01, 0.01])
    cam.resolution = np.array([1920, 1080])
    cam.principal_point = np.array([960.0, 540.0])
    cam.focal_length = np.array([50.0, 50.0])
    cam.clip_distance = np.array([0.1, 1000.0])

    return cam

def test_camera_init(camera):
    """Test the initialization of a Camera object."""
    assert isinstance(camera, Camera)
    assert camera.pixel_size == (0.01, 0.01)
    assert camera.resolution == (1920, 1080)
    assert camera.principal_point == (960.0, 540.0)
    assert camera.focal_length == (50.0, 50.0)
    assert camera.clip_distance == (0.1, 1000.0)

def test_save_load_to_dict(camera):
    """Test saving and loading a Camera object to/from a dictionary."""
    data = camera.save_to_dict(description="Test Camera")
    print(data)
    assert data["type"] == "Camera [pysdic]"
    assert data["description"] == "Test Camera"
    assert data["frame"]["translation"] == pytest.approx(camera.frame.origin.flatten(), rel=1e-5)
    assert data["frame"]["rotation_vector"] == pytest.approx(camera.frame.rotation_vector, rel=1e-5)
    assert data["fx"] == camera.fx
    assert data["fy"] == camera.fy
    assert data["cx"] == camera.cx
    assert data["cy"] == camera.cy
    assert data["rx"] == camera.rx
    assert data["ry"] == camera.ry
    assert data["px"] == camera.px
    assert data["py"] == camera.py
    assert data["clnear"] == camera.clnear
    assert data["clfar"] == camera.clfar

    loaded_camera = Camera.load_from_dict(data)
    assert loaded_camera.focal_length == (50.0, 50.0)
    assert loaded_camera.pixel_size == (0.01, 0.01)
    assert loaded_camera.resolution == (1920, 1080)
    assert loaded_camera.principal_point == (960.0, 540.0)
    assert loaded_camera.clip_distance == (0.1, 1000.0)

def test_cv2_RT(camera):
    """Test the OpenCV rotation and translation of a Camera object."""
    rotation, translation = camera.get_OpenCV_RT()
    assert rotation.as_rotvec() == pytest.approx(camera.frame.get_global_rotation_vector(convention=4), rel=1e-5)
    assert translation.flatten() == pytest.approx(camera.frame.get_global_translation(convention=4).flatten(), rel=1e-5)
    
