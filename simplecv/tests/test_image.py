import cv2
from nose.tools import assert_equals, nottest
import numpy as np

from simplecv.image_class import Image, ColorSpace

LENNA_PATH = '../data/sampleimages/lenna.png'


@nottest
def create_test_array():
    ''' Returns array 2 x 2 pixels, 8 bit and BGR color space
        pixels are colored so:
        RED, GREEN
        BLUE, WHITE
    '''
    return np.array([[[0, 0, 255], [0, 255, 0]],       # RED,  GREEN
                     [[255, 0, 0], [255, 255, 255]]],  # BLUE, WHITE
                    dtype=np.uint8)


@nottest
def create_test_image():
    bgr_array = create_test_array()
    return Image(bgr_array, color_space=ColorSpace.BGR)


def test_image_init_path_to_png():
    img1 = Image("lenna")
    assert_equals((512, 512), img1.size())
    assert_equals(ColorSpace.BGR, img1.get_color_space())
    assert img1.is_bgr()

    img_ndarray = img1.get_ndarray()

    assert isinstance(img_ndarray, np.ndarray)
    assert_equals(3, len(img_ndarray.shape))

    assert not img1.is_rgb()
    assert not img1.is_hsv()
    assert not img1.is_hls()
    assert not img1.is_ycrcb()
    assert not img1.is_xyz()
    assert not img1.is_gray()


def test_image_init_ndarray_color():
    color_ndarray = cv2.imread(LENNA_PATH)
    img1 = Image(color_ndarray, color_space=ColorSpace.BGR)

    assert_equals((512, 512), img1.size())
    assert_equals(ColorSpace.BGR, img1.get_color_space())
    assert img1.is_bgr()

    img_ndarray = img1.get_ndarray()

    assert isinstance(img_ndarray, np.ndarray)
    assert_equals(3, len(img_ndarray.shape))


def test_image_init_ndarray_grayscale():
    ndarray = cv2.imread(LENNA_PATH)
    gray_ndarray = cv2.cvtColor(ndarray, cv2.COLOR_BGR2GRAY)

    img1 = Image(gray_ndarray)
    assert_equals((512, 512), img1.size())
    assert_equals(ColorSpace.GRAY, img1.get_color_space())
    assert img1.is_gray()

    img_ndarray = img1.get_ndarray()

    assert isinstance(img_ndarray, np.ndarray)
    assert_equals(2, len(img_ndarray.shape))


def test_image_convert_bgr_to_bgr():
    bgr_array = create_test_array()
    result_bgr_array = Image.convert(bgr_array, ColorSpace.BGR, ColorSpace.BGR)
    assert_equals(bgr_array.data, result_bgr_array.data)


def test_image_bgr_to_rbg_to_bgr():
    bgr_img = create_test_image()
    rgb_img = bgr_img.to_rgb()
    rgb_array = np.array([[[255, 0, 0], [0, 255, 0]],
                          [[0, 0, 255], [255, 255, 255]]], dtype=np.uint8)
    assert_equals(rgb_array.data, rgb_img.get_ndarray().data)
    assert rgb_img.is_rgb()

    new_bgr_img = rgb_img.to_bgr()

    assert_equals(bgr_img.get_ndarray().data, new_bgr_img.get_ndarray().data)
    assert new_bgr_img.is_bgr()


def test_image_bgr_to_gray():
    bgr_img = create_test_image()
    gray_img = bgr_img.to_gray()
    gray_array = np.array([[76, 150],
                           [29, 255]], dtype=np.uint8)
    assert_equals(gray_array.data, gray_img.get_ndarray().data)
    assert gray_img.is_gray()


def test_image_bgr_to_hsv():
    bgr_img = create_test_image()
    hsv_img = bgr_img.to_hsv()
    hsv_array = np.array([[[0, 255, 255], [60, 255, 255]],
                          [[120, 255, 255], [0, 0, 255]]], dtype=np.uint8)
    assert_equals(hsv_array.data, hsv_img.get_ndarray().data)
    assert hsv_img.is_hsv()


def test_image_bgr_to_ycrcb():
    bgr_img = create_test_image()
    ycrcb_img = bgr_img.to_ycrcb()
    ycrcb_array = np.array([[[76, 255, 85], [150, 21, 43]],
                            [[29, 107, 255], [255, 128, 128]]], dtype=np.uint8)
    assert_equals(ycrcb_array.data, ycrcb_img.get_ndarray().data)
    assert ycrcb_img.is_ycrcb()


def test_image_bgr_to_zyx():
    bgr_img = create_test_image()
    xyz_img = bgr_img.to_xyz()
    xyz_array = np.array([[[105, 54, 5], [91, 182, 30]],
                          [[46, 18, 242], [242, 255, 255]]], dtype=np.uint8)
    assert_equals(xyz_array.data, xyz_img.get_ndarray().data)
    assert xyz_img.is_xyz()


def test_image_bgr_to_hls():
    bgr_img = create_test_image()
    hls_img = bgr_img.to_hls()
    hls_array = np.array([[[0, 128, 255], [60, 128, 255]],
                          [[120, 128, 255], [0, 255, 0]]], dtype=np.uint8)
    assert_equals(hls_array.data, hls_img.get_ndarray().data)
    assert hls_img.is_hls()


def test_image_hsv_to_gray():
    hsv_array = np.array([[[0, 255, 255], [60, 255, 255]],
                          [[120, 255, 255], [0, 0, 255]]], dtype=np.uint8)
    hsv_img = Image(hsv_array, color_space=ColorSpace.HSV)
    gray_array = np.array([[76, 150],
                           [29, 255]], dtype=np.uint8)
    gray_img = hsv_img.to_gray()

    assert_equals(gray_array.data, gray_img.get_ndarray().data)
    assert gray_img.is_gray()
