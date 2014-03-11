# /usr/bin/python
# To run this test you need python nose tools installed
# Run test just use:
#   nosetest test_stereovision.py
#

import os, sys, pickle
from simplecv import *
from nose.tools import with_setup, nottest

VISUAL_TEST = True  # if TRUE we save the images - otherwise we DIFF against them - the default is False
SHOW_WARNING_TESTS = False  # show that warnings are working - tests will pass but warnings are generated.

#colors
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

#images
pair1 = ("../data/sampleimages/stereo1_left.png" , "../data/sampleimages/stereo1_right.png")
pair2 = ("../data/sampleimages/stereo2_left.png" , "../data/sampleimages/stereo2_right.png")
pair3 = ("../data/sampleimages/stereo1_real_left.png" , "../data/sampleimages/stereo1_real_right.png")
pair4 = ("../data/sampleimages/stereo2_real_left.png" , "../data/sampleimages/stereo2_real_right.png")
pair5 = ("../data/sampleimages/stereo3_real_left.png" , "../data/sampleimages/stereo3_real_right.png")

correct_pairs = [pair1,pair2,pair3,pair4,pair5]

#standards path
standard_path = "../data/test/standard/"


#Given a set of images, a path, and a tolerance do the image diff.
def imgDiffs(test_imgs,name_stem,tolerance,path):
    count = len(test_imgs)
    for idx in range(0,count):
        lhs = test_imgs[idx].apply_layers() # this catches drawing methods
        fname = standard_path+name_stem+str(idx)+".jpg"
        rhs = Image(fname)
        if( lhs.width == rhs.width and lhs.height == rhs.height ):
            diff = (lhs-rhs)
            val = np.average(diff.get_numpy())
            if( val > tolerance ):
                print val
                return True
    return False

#Save a list of images to a standard path.
def imgSaves(test_imgs, name_stem, path=standard_path):
    count = len(test_imgs)
    for idx in range(0,count):
        fname = standard_path+name_stem+str(idx)+".jpg"
        test_imgs[idx].save(fname)#,quality=95)

#perform the actual image save and image diffs.
def perform_diff(result,name_stem,tolerance=2.0,path=standard_path):
    if(VISUAL_TEST): # save the correct images for a visual test
        imgSaves(result,name_stem,path)
    else: # otherwise we test our output against the visual test
        if( imgDiffs(result,name_stem,tolerance,path) ):
            assert False
        else:
            pass

#These function names are required by nose test, please leave them as is
def setup_context():
    img = Image(pair1[0])

def destroy_context():
    img = ""

@with_setup(setup_context, destroy_context)
def test_findFundamentalMat():
    for pairs in correct_pairs :
        img1 = Image(pairs[0])
        img2 = Image(pairs[1])
        StereoImg = StereoImage(img1,img2)
        if ( not StereoImg.find_fundamental_mat()):
            assert False

def test_findHomography():
    for pairs in correct_pairs :
        img1 = Image(pairs[0])
        img2 = Image(pairs[1])
        StereoImg = StereoImage(img1,img2)
        if (not StereoImg.find_homography()):
            assert False

def test_findDisparityMap():
    dips = []
    for pairs in correct_pairs :
        img1 = Image(pairs[0])
        img2 = Image(pairs[1])
        StereoImg = StereoImage(img1,img2)
        dips.append(StereoImg.find_disparity_map(method="BM"))
    name_stem = "test_disparitymapBM"
    perform_diff(dips,name_stem)

    dips = []
    for pairs in correct_pairs :
        img1 = Image(pairs[0])
        img2 = Image(pairs[1])
        StereoImg = StereoImage(img1,img2)
        dips.append(StereoImg.find_disparity_map(method="SGBM"))
    name_stem = "test_disparitymapSGBM"
    perform_diff(dips,name_stem)

def test_eline():
    for pairs in correct_pairs :
        img1 = Image(pairs[0])
        img2 = Image(pairs[1])
        StereoImg = StereoImage(img1,img2)
        F,ptsLeft,ptsRight = StereoImg.find_fundamental_mat()
        for pts in ptsLeft :
            line = StereoImg.eline(pts,F,2)
            if (line == None):
                assert False


def test_projectPoint():
    for pairs in correct_pairs :
        img1 = Image(pairs[0])
        img2 = Image(pairs[1])
        StereoImg = StereoImage(img1,img2)
        H,ptsLeft,ptsRight = StereoImg.find_homography()
        for pts in ptsLeft :
            line = StereoImg.project_point(pts,H,2)
            if (line == None):
                assert False


def test_StereoCalibration():
    cam = StereoCamera()
    try :
        cam1 = Camera(0)
        cam2 = Camera(1)
        cam1.get_image()
        cam2.get_image()
        try :
            cam = StereoCamera()
            calib = cam.stereo_calibration(0,1,nboards=1)
            if (calib):
                assert True
            else :
                assert False
        except:
            assert False
    except :
        assert True

def test_loadCalibration():
    cam = StereoCamera()
    calbib =  cam.load_calibration("Stereo","../data/test/StereoVision/")
    if (calbib) :
        assert True
    else :
        assert False

def test_StereoRectify():
    cam = StereoCamera()
    calib = cam.load_calibration("Stereo","../data/test/StereoVision/")
    rectify = cam.stereo_rectify(calib)
    if rectify :
        assert True
    else :
        assert False

def test_getImagesUndistort():
    img1 = Image(correct_pairs[0][0]).resize(352,288)
    img2 = Image(correct_pairs[0][1]).resize(352,288)
    cam = StereoCamera()
    calib = cam.load_calibration("Stereo","../data/test/StereoVision/")
    rectify = cam.stereo_rectify(calib)
    rectLeft,rectRight = cam.get_images_undistort(img1,img2,calib,rectify)
    if rectLeft and rectRight :
        assert True
    else :
        assert False
