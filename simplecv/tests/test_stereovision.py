# To run this test you need python nose tools installed
# Run test just use:
#   nosetest test_stereovision.py

import numpy as np

from simplecv.color import Color
from simplecv.core.camera.camera import Camera
from simplecv.core.camera.stereo_camera import StereoImage, StereoCamera
from simplecv.image import Image
from simplecv.tests.utils import perform_diff

# Colors
black = Color.BLACK
white = Color.WHITE
red = Color.RED
green = Color.GREEN
blue = Color.BLUE

###############
# TODO -
# Examples of how to do profiling
# Examples of how to do a single test -
# UPDATE THE VISUAL TESTS WITH EXAMPLES.
# Fix exif data
# Turn off test warnings using decorators.
# Write a use the tests doc.

# Images
pair1 = ("../data/sampleimages/stereo1_left.png",
         "../data/sampleimages/stereo1_right.png")
pair2 = ("../data/sampleimages/stereo2_left.png",
         "../data/sampleimages/stereo2_right.png")
pair3 = ("../data/sampleimages/stereo1_real_left.png",
         "../data/sampleimages/stereo1_real_right.png")
pair4 = ("../data/sampleimages/stereo2_real_left.png",
         "../data/sampleimages/stereo2_real_right.png")
pair5 = ("../data/sampleimages/stereo3_real_left.png",
         "../data/sampleimages/stereo3_real_right.png")

correct_pairs = [pair1, pair2, pair3, pair4, pair5]


def test_find_fundamental_mat():
    for pairs in correct_pairs:
        img1 = Image(pairs[0])
        img2 = Image(pairs[1])
        stereo_img = StereoImage(img1, img2)
        if not stereo_img.find_fundamental_mat():
            assert False


def test_find_homography():
    for pairs in correct_pairs:
        img1 = Image(pairs[0])
        img2 = Image(pairs[1])
        stereo_img = StereoImage(img1, img2)
        if not stereo_img.find_homography():
            assert False


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
    perform_diff(dips, name_stem)


def test_eline():
    for pairs in correct_pairs:
        img1 = Image(pairs[0])
        img2 = Image(pairs[1])
        stereo_img = StereoImage(img1, img2)
        f, pts_left, pts_right = stereo_img.find_fundamental_mat()
        for pts in pts_left:
            line = stereo_img.eline(pts, f, 2)
            if line is None:
                assert False


def test_project_point():
    for pairs in correct_pairs:
        img1 = Image(pairs[0])
        img2 = Image(pairs[1])
        stereo_img = StereoImage(img1, img2)
        h, pts_left, pts_right = stereo_img.find_homography()
        for pts in pts_left:
            line = stereo_img.project_point(pts, h, 2)
            if line is None:
                assert False


def test_stereo_calibration():
    cam = StereoCamera()

    cam1 = Camera(0)
    cam2 = Camera(1)
    cam1.get_image()
    cam2.get_image()

    cam = StereoCamera()
    assert cam.stereo_calibration(0, 1, nboards=1)


def test_load_calibration():
    cam = StereoCamera()
    assert cam.load_calibration("Stereo", "../data/test/StereoVision/")


def test_stereo_rectify():
    cam = StereoCamera()
    calib = cam.load_calibration("Stereo", "../data/test/StereoVision/")
    assert cam.stereo_rectify(calib)


def test_get_images_undistort():
    img1 = Image(correct_pairs[0][0]).resize(352, 288)
    img2 = Image(correct_pairs[0][1]).resize(352, 288)
    cam = StereoCamera()
    calib = cam.load_calibration("Stereo", "../data/test/StereoVision/")
    rectify = cam.stereo_rectify(calib)
    rect_left, rect_right = cam.get_images_undistort(img1, img2,
                                                     calib, rectify)
    assert rect_left and rect_right
