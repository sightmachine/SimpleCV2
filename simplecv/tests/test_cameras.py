import os
import tempfile

import numpy as np
import mock
from nose.tools import assert_is_instance, assert_equals

from simplecv.core.camera.camera import Camera
from simplecv.core.camera.frame_source import FrameSource
from simplecv.core.camera.virtual_camera import VirtualCamera
from simplecv.image import Image
from simplecv.tests.utils import perform_diff

testoutput = os.path.join(tempfile.gettempdir(), 'cam.jpg')


def test_virtual_camera_constructor():
    mycam = VirtualCamera(testoutput, 'image')

    props = mycam.get_all_properties()

    for i in props.keys():
        print str(i) + ": " + str(props[i]) + "\n"


@mock.patch('simplecv.core.camera.camera.cv2.VideoCapture')
def test_camera_image(video_capture_mock):
    img = Image((10, 10))
    video_capture_mock.return_value.retrieve.return_value = (True, img)
    video_capture_mock.return_value.isOpened.return_value = True

    mycam = Camera(0)
    camera_image = mycam.get_image()
    assert_is_instance(camera_image, Image)
    assert_equals(camera_image.size_tuple, (10, 10))

    video_capture_mock.assert_called_with(0)
    video_capture_mock.return_value.grab.assert_called_with()


@mock.patch('simplecv.core.camera.camera.cv2.VideoCapture')
def test_camera_multiple_instances(video_capture_mock):
    img = Image((10, 10))
    video_capture_mock.return_value.retrieve.return_value = (True, img)

    cam1 = Camera()
    assert_is_instance(cam1.get_image(), Image)
    cam2 = Camera()
    assert_is_instance(cam2.get_image(), Image)
    cam3 = Camera(0)  # assuming the default camera index is 0
    assert_is_instance(cam3.get_image(), Image)

    video_capture_mock.assert_called_with(0)


def test_camera_undistort():
    fake_camera = FrameSource()
    fake_camera.load_calibration("../data/test/StereoVision/Default")
    img = Image("../data/sampleimages/CalibImage0.png")
    img2 = fake_camera.undistort(img)
    assert img2 is not None

    results = [img2]
    name_stem = "test_camera_undistort"
    perform_diff(results, name_stem)


def test_camera_calibration():
    fake_camera = FrameSource()
    path = "../data/sampleimages/CalibImage"
    ext = ".png"
    imgs = []
    for i in range(0, 10):
        fname = path + str(i) + ext
        img = Image(fname)
        imgs.append(img)

    fake_camera.calibrate(imgs)
    #we're just going to check that the function doesn't puke
    assert isinstance(fake_camera.camera_matrix, np.ndarray)

    #we're also going to test load in save in the same pass
    matname = os.path.join(tempfile.gettempdir(), "TestCalibration")
    fake_camera.save_calibration(matname)
    intrinsic = matname + "Intrinsic.bin"
    distortion = matname + "Distortion.bin"
    assert os.path.exists(intrinsic)
    assert os.path.exists(distortion)
    result =  fake_camera.load_calibration(matname)
    os.remove(intrinsic)
    os.remove(distortion)

    assert result