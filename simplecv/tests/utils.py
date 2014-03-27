import os

from nose.tools import nottest
import cv2
import numpy as np

from simplecv.image_class import Image

VISUAL_TEST = False  # if TRUE we save the images - otherwise we DIFF against
                     # them - the default is False

#standards path
standard_path = "../data/test/standard/"


#Given a set of images, a path, and a tolerance do the image diff.
@nottest
def img_diffs(test_imgs, name_stem, tolerance, path):
    count = len(test_imgs)
    ret_val = False
    for idx in range(0, count):
        lhs = test_imgs[idx].apply_layers()  # this catches drawing methods
        if lhs.is_gray():
            lhs = lhs.to_bgr()
        fname = standard_path + name_stem + str(idx)
        fname_jpg = fname + ".jpg"
        fname_png = fname + ".png"
        if os.path.exists(fname_png):
            rhs = Image(fname_png)
        else:
            rhs = Image(fname_jpg)
        if lhs.size() == rhs.size():
            num_img_pixels = lhs.width * lhs.height * 3
            diff = cv2.absdiff(lhs.get_ndarray(), rhs.get_ndarray())
            diff_pixels = (diff > 0).astype(np.uint8)
            diff_pixels_sum = diff_pixels.sum()
            if diff_pixels_sum > 0:
                percent_diff_pixels = diff_pixels_sum / num_img_pixels
                print "{0:.2f}% difference".format(percent_diff_pixels * 100)
                lhs.save(fname_png)  # lhs.save(fname + "_RESULT.png")
                lhs = Image((diff_pixels * (0, 0, 255)).astype(np.uint8))
                lhs.save(fname + "_DIFF.png")
                ret_val = True
    return ret_val


#Save a list of images to a standard path.
@nottest
def img_saves(test_imgs, name_stem, path=standard_path):
    count = len(test_imgs)
    for idx in range(0, count):
        fname = standard_path + name_stem + str(idx) + ".png"
        test_imgs[idx].save(fname)


#perform the actual image save and image diffs.
@nottest
def perform_diff(result, name_stem, tolerance=0.03, path=standard_path):
    if VISUAL_TEST:  # save the correct images for a visual test
        img_saves(result, name_stem, path)
    else:  # otherwise we test our output against the visual test
        assert not img_diffs(result, name_stem, tolerance, path)
