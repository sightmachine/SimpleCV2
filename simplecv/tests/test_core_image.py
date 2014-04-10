import os
import tempfile

import cv2
from nose.tools import assert_equals, nottest, raises
import numpy as np

from simplecv.core.image import Image as CoreImage


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
    return CoreImage(array=bgr_array, color_space=CoreImage.BGR)


def test_init_core_image():
    img = create_test_image()
    assert_equals((2, 2), img.size)
    assert_equals(np.uint8, img.dtype)
    assert_equals(np.uint8, img.dtype)
    assert_equals(CoreImage.BGR, img.color_space)
    assert img.is_color_space(CoreImage.BGR)
    assert False == img.is_color_space(CoreImage.GRAY)


def test_init_core_image_grayscale():
    gray_array = np.array([[76, 150], [29, 255]], dtype=np.uint8)
    img = CoreImage(array=gray_array)
    assert_equals((2, 2), img.size)
    assert_equals(np.uint8, img.dtype)
    assert_equals(np.uint8, img.dtype)
    assert_equals(CoreImage.GRAY, img.color_space)
    assert img.is_color_space(CoreImage.GRAY)
    assert False == img.is_color_space(CoreImage.BGR)


@raises(AttributeError)
def test_core_image_wrong_array():
    CoreImage(array=None, color_space=CoreImage.BGR)


@raises(AttributeError)
def test_core_image_wrong_color_space():
    CoreImage(array=create_test_array(), color_space=999)
