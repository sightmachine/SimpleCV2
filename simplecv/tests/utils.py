import os

from nose.tools import nottest, assert_equals
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
                if (percent_diff_pixels * 100 < tolerance):
                    ret_val = False
                else:
                    ret_val = True
                # Uncomment this to save result and diff images
                # cv2.imwrite(fname + "_RESULT.png", lhs)
                # cv2.imwrite(fname + "_DIFF.png",
                #             (diff_pixels * (0, 0, 255)).astype(np.uint8))
                
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


@nottest
def perform_diff_blobs(blob1, blob2):
    assert_equals(blob1.m00, blob2.m00)
    assert_equals(blob1.m01, blob2.m01)
    assert_equals(blob1.m10, blob2.m10)
    assert_equals(blob1.m20, blob2.m20)
    assert_equals(blob1.m02, blob2.m02)
    assert_equals(blob1.m21, blob2.m21)
    assert_equals(blob1.m12, blob2.m12)
    assert_equals(blob1.label, blob2.label)
    assert_equals(blob1.label_color, blob2.label_color)
    assert_equals(blob1.avg_color, blob2.avg_color)
    assert_equals(blob1.hu.data, blob2.hu.data)
    assert_equals(blob1.perimeter, blob2.perimeter)
    assert_equals(blob1.min_rectangle, blob2.min_rectangle)
    assert_equals(blob1._scdescriptors, blob2._scdescriptors)
    assert_equals(blob1._complete_contour, blob2._complete_contour)
    assert_equals(blob1.contour, blob2.contour)
    assert_equals(blob1.convex_hull, blob2.convex_hull)
    assert_equals(blob1.contour_appx, blob2.contour_appx)
    assert_equals(blob1.image.get_ndarray().data,
                  blob2.image.get_ndarray().data)
    assert_equals(blob1.points, blob2.points)
    assert_equals(blob1.hole_contour, blob2.hole_contour)
