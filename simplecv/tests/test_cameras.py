import os
import tempfile

import numpy as np

from simplecv.camera import Camera, VirtualCamera, FrameSource
from simplecv.image import Image
from simplecv.tests.utils import perform_diff

testoutput = os.path.join(tempfile.gettempdir(), 'cam.jpg')


def test_virtual_camera_constructor():
    mycam = VirtualCamera(testoutput, 'image')

    props = mycam.get_all_properties()

    for i in props.keys():
        print str(i) + ": " + str(props[i]) + "\n"


def test_camera_image():
    mycam = Camera(0)

    img = mycam.get_image()
    img.save(testoutput)


def test_camera_multiple_instances():
    cam1 = Camera()
    img1 = cam1.get_image()
    cam2 = Camera()
    img2 = cam2.get_image()

    if not cam1 or not cam2 or not img1 or not img2:
        assert False

    cam3 = Camera(0)  # assuming the default camera index is 0
    img3 = cam3.get_image()

    if not cam3 or not img3:
        assert False


def test_camera_undistort():
    fake_camera = FrameSource()
    fake_camera.load_calibration("../data/test/StereoVision/Default")
    img = Image("../data/sampleimages/CalibImage0.png")
    img2 = fake_camera.undistort(img)
    assert img2

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
    mat = fake_camera.get_camera_matrix()
    assert isinstance(mat, np.ndarray)

    #we're also going to test load in save in the same pass
    matname = "TestCalibration"
    assert fake_camera.save_calibration(matname)
    assert fake_camera.load_calibration(matname)
