import os

import cv2
from nose.tools import *
from simplecv.base import ScvException

from simplecv.color import Color
from simplecv.core.camera.stereo_camera import StereoImage, StereoCamera
from simplecv.image import Image
from simplecv.tests.utils import perform_diff, sampleimage_path
from simplecv import DATA_DIR


# Colors
black = Color.BLACK
white = Color.WHITE
red = Color.RED
green = Color.GREEN
blue = Color.BLUE

# Images
pair1 = (sampleimage_path("stereo1_left.png"),
         sampleimage_path("stereo1_right.png"))
pair2 = (sampleimage_path("stereo2_left.png"),
         sampleimage_path("stereo2_right.png"))
pair3 = (sampleimage_path("stereo1_real_left.png"),
         sampleimage_path("stereo1_real_right.png"))
pair4 = (sampleimage_path("stereo2_real_left.png"),
         sampleimage_path("stereo2_real_right.png"))
pair5 = (sampleimage_path("stereo3_real_left.png"),
         sampleimage_path("stereo3_real_right.png"))

pair6 = (sampleimage_path("simplecv.png"),
         sampleimage_path("lenna.png"))

correct_pairs = [pair1, pair2, pair3, pair4, pair5]
incorrect_pairs = [pair6]

STEREO_TEST_DIR = os.path.join(DATA_DIR, 'test/StereoVision')


def test_find_fundamental_mat():
    for pairs in correct_pairs:
        img1 = Image(pairs[0])
        img2 = Image(pairs[1])
        stereo_img = StereoImage(img1, img2)
        assert stereo_img.find_fundamental_mat()


def test_find_homography():
    for pairs in correct_pairs:
        img1 = Image(pairs[0])
        img2 = Image(pairs[1])
        stereo_img = StereoImage(img1, img2)
        assert stereo_img.find_homography()


def test_find_disparity_map_bm():
    dips = []
    for pairs in correct_pairs:
        img1 = Image(pairs[0])
        img2 = Image(pairs[1])
        stereo_img = StereoImage(img1, img2)
        dips.append(stereo_img.find_disparity_map(method="BM"))
    name_stem = "test_disparitymap_bm"
    perform_diff(dips, name_stem)


def test_find_disparity_map_sgbm():
    dips = []
    for pairs in correct_pairs:
        img1 = Image(pairs[0])
        img2 = Image(pairs[1])
        stereo_img = StereoImage(img1, img2)
        dips.append(stereo_img.find_disparity_map(method="SGBM"))
    name_stem = "test_disparitymap_sgbm"
    perform_diff(dips, name_stem, tolerance=8.0)


def test_eline():
    for pairs in correct_pairs:
        img1 = Image(pairs[0])
        img2 = Image(pairs[1])
        stereo_img = StereoImage(img1, img2)
        f, pts_left, pts_right = stereo_img.find_fundamental_mat()
        for pts in pts_left:
            assert stereo_img.eline(pts, f, 2) is not None


def test_project_point():
    for pairs in correct_pairs:
        img1 = Image(pairs[0])
        img2 = Image(pairs[1])
        stereo_img = StereoImage(img1, img2)
        h, pts_left, pts_right = stereo_img.find_homography()
        for pts in pts_left:
            assert stereo_img.project_point(pts, h, 2) is not None


def test_stereo_calibration():
    # This test requires two cameras
    c0 = cv2.VideoCapture()
    if not c0.open(0):  # check first camera
        return
    c0.release()
    c1 = cv2.VideoCapture()
    if not c1.open(1):  # check second camera
        return
    c1.release()

    cam = StereoCamera()
    calibration = cam.stereo_calibration(0, 1, nboards=1)
    assert_is_not_none(calibration)


def test_load_calibration():
    cam = StereoCamera()
    calibration = cam.load_calibration("Stereo", STEREO_TEST_DIR)
    assert_is_not_none(calibration)


def test_stereo_rectify():
    cam = StereoCamera()
    calib = cam.load_calibration("Stereo", STEREO_TEST_DIR)
    sr = cam.stereo_rectify(calib)
    assert_is_not_none(sr)


def test_get_images_undistort():
    img1 = Image(correct_pairs[0][0]).resize(352, 288)
    img2 = Image(correct_pairs[0][1]).resize(352, 288)
    cam = StereoCamera()
    calib = cam.load_calibration("Stereo", STEREO_TEST_DIR)
    rectify = cam.stereo_rectify(calib)
    rect_left, rect_right = cam.get_images_undistort(img1, img2,
                                                     calib, rectify)

    assert_is_not_none(rect_left)
    assert_is_not_none(rect_right)


def test_incorrect_pairs():
    img1 = Image(incorrect_pairs[0][0])
    img2 = Image(incorrect_pairs[0][1])

    assert_raises(ScvException, StereoImage, img1, img2)
