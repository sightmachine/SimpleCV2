import cv2
from nose.tools import assert_equals
import numpy as np

from simplecv.image_class import Image, ColorSpace

LENNA_PATH = '../data/sampleimages/lenna.png'


def test_image_init_path_to_png():
    img1 = Image("lenna")
    assert_equals(512, img1.width)
    assert_equals(512, img1.height)
    assert_equals(ColorSpace.BGR, img1.get_color_space())
    assert img1.is_bgr()

    img_ndarray = img1.get_ndarray()

    assert isinstance(img_ndarray, np.ndarray)
    assert_equals(3, len(img_ndarray.shape))


def test_image_init_ndarray_color():
    color_ndarray = cv2.imread(LENNA_PATH)
    img1 = Image(color_ndarray, color_space=ColorSpace.BGR)

    assert_equals(512, img1.width)
    assert_equals(512, img1.height)
    assert_equals(ColorSpace.BGR, img1.get_color_space())
    assert img1.is_bgr()

    img_ndarray = img1.get_ndarray()

    assert isinstance(img_ndarray, np.ndarray)
    assert_equals(3, len(img_ndarray.shape))


def test_image_init_ndarray_grayscale():
    ndarray = cv2.imread(LENNA_PATH)
    gray_ndarray = cv2.cvtColor(ndarray, cv2.COLOR_BGR2GRAY)

    img1 = Image(gray_ndarray)
    assert_equals(512, img1.width)
    assert_equals(512, img1.height)
    assert_equals(ColorSpace.GRAY, img1.get_color_space())
    assert img1.is_gray()

    img_ndarray = img1.get_gray_ndarray()

    assert isinstance(img_ndarray, np.ndarray)
    assert_equals(2, len(img_ndarray.shape))
