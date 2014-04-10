import numpy as np
import cv2
from nose.tools import assert_equals, assert_list_equal

from simplecv.tests.utils import perform_diff
from simplecv.image import Image


def test_color_meancolor():
    a = np.arange(0, 256)
    b = a[::-1]
    c = np.copy(a) / 2
    a = a.reshape(16, 16)
    b = b.reshape(16, 16)
    c = c.reshape(16, 16)
    imgarr = np.dstack((a, b, c)).astype(np.uint8)
    img = Image(array=imgarr, color_space=Image.RGB)

    b, g, r = img.mean_color('BGR')
    print b, g, r
    if not (127 < r < 128 and 127 < g < 128 and 63 < b < 64):
        assert False

    r, g, b = img.mean_color('RGB')
    if not (127 < r < 128 and 127 < g < 128 and 63 < b < 64):
        assert False

    h, s, v = img.mean_color('HSV')
    if not (83 < h < 84 and 191 < s < 192 and 191 < v < 192):
        assert False

    x, y, z = img.mean_color('XYZ')
    if not (109 < x < 110 and 122 < y < 123 and 77 < z < 79):
        assert False

    gray = img.mean_color('Gray')
    if not (120 < gray < 121):
        assert False

    y, cr, cb = img.mean_color('YCrCb')
    if not (120 < y < 121 and 133 < cr < 134 and 96 < cb < 97):
        assert False

    h, l, s = img.mean_color('HLS')
    if not (84 < h < 85 and 117 < l < 118 and 160 < s < 161):
        assert False


def test_image_edgemap():
    img_a = Image(source='simplecv')
    array = img_a.get_edge_map()
    results = [Image(array=array)]
    name_stem = "test_image_edgemap"
    perform_diff(results, name_stem)


def test_image_horz_scanline():
    img = Image(source='simplecv')
    sl = img.get_horz_scanline(10)
    assert len(sl.shape) == 2
    assert sl.shape[0] == img.width
    assert sl.shape[1] == 3


def test_image_vert_scanline():
    img = Image(source='simplecv')
    sl = img.get_vert_scanline(10)
    assert len(sl.shape) == 2
    assert sl.shape[0] == img.height
    assert sl.shape[1] == 3


def test_image_horz_scanline_gray():
    img = Image(source='simplecv')
    sl = img.get_horz_scanline_gray(10)
    assert len(sl.shape) == 1
    assert sl.shape[0] == img.width


def test_image_vert_scanline_gray():
    img = Image(source='simplecv')
    sl = img.get_vert_scanline_gray(10)
    assert len(sl.shape) == 1
    assert sl.shape[0] == img.width


def test_image_get_pixel():
    img = Image(source='simplecv')
    assert_list_equal([0, 0, 0], img.get_pixel(0, 0))


def test_image_get_gray_pixel():
    img = Image(source='simplecv')
    assert_equals(0, img.get_gray_pixel(0, 0))


def test_image_intergralimage():
    img = Image(source='simplecv')
    array = img.integral_image()
    assert isinstance(array, np.ndarray)
    assert_equals(np.int32, array.dtype)
    assert_equals(img.get_ndarray().shape[0] + 1, array.shape[0])
    assert_equals(img.get_ndarray().shape[1] + 1, array.shape[1])


def test_image_intergralimage_tilted():
    img = Image(source='simplecv')
    array = img.integral_image(tilted=True)
    assert isinstance(array, np.ndarray)
    assert_equals(np.int32, array.dtype)
    assert_equals(img.get_ndarray().shape[0] + 1, array.shape[0])
    assert_equals(img.get_ndarray().shape[1] + 1, array.shape[1])
