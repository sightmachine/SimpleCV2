import os

from nose.tools import nottest
import cv2
import numpy as np

from simplecv.image import Image

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
        lhs = lhs.to_bgr().get_ndarray()
        fname = standard_path + name_stem + str(idx)
        fname_png = fname + ".png"
        if os.path.exists(fname_png):
            rhs = cv2.imread(fname_png)
        else:
            raise Exception('Cannot load standard image')
        if lhs.shape == rhs.shape:
            diff = cv2.absdiff(lhs, rhs)
            diff_pixels = (diff > 0).astype(np.uint8)
            diff_pixels_sum = diff_pixels.sum()
            if diff_pixels_sum > 0:
                num_img_pixels = lhs.size
                percent_diff_pixels = diff_pixels_sum / num_img_pixels
                print "{0:.2f}% difference".format(percent_diff_pixels * 100)
                # Uncomment this to save result and diff images
                # cv2.imwrite(fname + "_RESULT.png", lhs)
                # cv2.imwrite(fname + "_DIFF.png",
                #             (diff_pixels * (0, 0, 255)).astype(np.uint8))
                ret_val = True
        else:
            print "images have different size {} and {}".format(lhs.shape,
                                                                rhs.shape)
            # Uncomment this to save result image with wrong size
            # cv2.imwrite(fname + "_WRONG_SIZE.png", lhs)
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


@nottest
def create_test_array():
    """ Returns array 2 x 2 pixels, 8 bit and BGR color space
        pixels are colored so:
        RED, GREEN
        BLUE, WHITE
    """
    return np.array([[[0, 0, 255], [0, 255, 0]],       # RED,  GREEN
                     [[255, 0, 0], [255, 255, 255]]],  # BLUE, WHITE
                    dtype=np.uint8)


@nottest
def create_test_image():
    bgr_array = create_test_array()
    return Image(array=bgr_array, color_space=Image.BGR)
