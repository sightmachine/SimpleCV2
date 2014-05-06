import os
import tempfile

import cv2
from nose.tools import assert_equals, nottest, raises
import numpy as np

from simplecv.core.image import Image as CoreImage
from simplecv.tests.utils import create_test_array, create_test_image


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

def test_core_sub():
    np_array = np.array([[255, 255, 255], [255, 255, 255], [255, 255, 255]],
                        dtype=np.uint8)
    np_array1 = np.array([[255, 255, 255], [255, 255, 255]],
                        dtype=np.uint8)
    np_array2 = np.array([[255, 0, 255], [255, 0, 255], [255, 0, 255]],
                        dtype=np.uint8)

    img = CoreImage(array=np_array)
    img1 = CoreImage(array=np_array1)
    img2 = CoreImage(array=np_array2)

    np_res1 = np.array([[0, 255, 0],[0, 255, 0], [0, 255, 0]], dtype=np.uint8)
    np_res2 = np.array([[127, 127, 127],[127, 127, 127], [127, 127, 127]], dtype=np.uint8)

    assert_equals((img - img1), None)
    assert_equals((img - img2).get_ndarray().data, np_res1.data)
    assert_equals((img - 128).get_ndarray().data, np_res2.data)

def test_core_add():
    np_array = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                        dtype=np.uint8)
    np_array1 = np.array([[255, 255, 255], [255, 255, 255]],
                        dtype=np.uint8)
    np_array2 = np.array([[255, 0, 255], [255, 0, 255], [255, 0, 255]],
                        dtype=np.uint8)

    img = CoreImage(array=np_array)
    img1 = CoreImage(array=np_array1)
    img2 = CoreImage(array=np_array2)

    np_res1 = np.array([[255, 0, 255],[255, 0, 255], [255, 0, 255]], dtype=np.uint8)
    np_res2 = np.array([[127, 127, 127],[127, 127, 127], [127, 127, 127]], dtype=np.uint8)

    assert_equals((img + img1), None)
    assert_equals((img + img2).get_ndarray().data, np_res1.data)
    assert_equals((img + 127).get_ndarray().data, np_res2.data)

def test_core_and():
    np_array = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                        dtype=np.uint8)
    np_array1 = np.array([[255, 255, 255], [255, 255, 255]],
                        dtype=np.uint8)
    np_array2 = np.array([[255, 0, 255], [255, 0, 255], [255, 0, 255]],
                        dtype=np.uint8)

    img = CoreImage(array=np_array)
    img1 = CoreImage(array=np_array1)
    img2 = CoreImage(array=np_array2)

    np_res1 = np.array([[1, 0, 1],[1, 0, 1], [1, 0, 1]], dtype=np.uint8)
    np_res2 = np.array([[1, 1, 1],[1, 1, 1], [1, 1, 1]], dtype=np.uint8)
    
    assert_equals((img & img1), None)
    assert_equals((img & img2).get_ndarray().data, np_res1.data)
    assert_equals((img & 127).get_ndarray().data, np_res2.data)

def test_core_or():
    np_array = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                        dtype=np.uint8)
    np_array1 = np.array([[255, 255, 255], [255, 255, 255]],
                        dtype=np.uint8)
    np_array2 = np.array([[255, 0, 255], [255, 0, 255], [255, 0, 255]],
                        dtype=np.uint8)

    img = CoreImage(array=np_array)
    img1 = CoreImage(array=np_array1)
    img2 = CoreImage(array=np_array2)

    np_res1 = np.array([[255, 1, 255],[255, 1, 255], [255, 1, 255]], dtype=np.uint8)
    np_res2 = np.array([[127, 127, 127],[127, 127, 127], [127, 127, 127]], dtype=np.uint8)

    assert_equals((img | img1), None)
    assert_equals((img | img2).get_ndarray().data, np_res1.data)
    assert_equals((img | 127).get_ndarray().data, np_res2.data)

def test_core_div():
    np_array = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                        dtype=np.uint8)
    np_array1 = np.array([[255, 255, 255], [255, 255, 255]],
                        dtype=np.uint8)
    np_array2 = np.array([[255, 0, 255], [255, 0, 255], [255, 0, 255]],
                        dtype=np.uint8)

    img = CoreImage(array=np_array)
    img1 = CoreImage(array=np_array1)
    img2 = CoreImage(array=np_array2)

    np_res1 = np.array([[255, 0, 255],[255, 0, 255], [255, 0, 255]], dtype=np.uint8)
    np_res2 = np.array([[127, 0, 127],[127, 0, 127], [127, 0, 127]], dtype=np.uint8)

    assert_equals((img/img1), None)
    assert_equals((img2/img).get_ndarray().data, np_res1.data)
    assert_equals((img2/2).get_ndarray().data, np_res2.data)

def test_core_mul():
    np_array = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                        dtype=np.uint8)
    np_array1 = np.array([[255, 255, 255], [255, 255, 255]],
                        dtype=np.uint8)
    np_array2 = np.array([[255, 0, 255], [255, 0, 255], [255, 0, 255]],
                        dtype=np.uint8)

    img = CoreImage(array=np_array)
    img1 = CoreImage(array=np_array1)
    img2 = CoreImage(array=np_array2)

    np_res1 = np.array([[255, 0, 255],[255, 0, 255], [255, 0, 255]], dtype=np.uint8)
    np_res2 = np.array([[127, 127, 127],[127, 127, 127], [127, 127, 127]], dtype=np.uint8)

    assert_equals((img*img1), None)
    assert_equals((img2*img).get_ndarray().data, np_res1.data)
    assert_equals((img*127).get_ndarray().data, np_res2.data)

def test_core_pow():
    np_array = np.array([[2, 3, 2], [3, 2, 3], [7, 2, 1]],
                        dtype=np.uint8)
    img = CoreImage(array=np_array)

    np_res1 = np.array([[16, 81, 16],[81, 16, 81], [255, 16, 1]], dtype=np.uint8)

    assert_equals((img**4).get_ndarray().data, np_res1.data)
    assert_equals((img**2.3), None)

def test_core_neg():
    np_array = np.array([[2, 127, 128], [50, 100, 150], [7, 2, 0]],
                        dtype=np.uint8)
    img = CoreImage(array=np_array)
    np_res1 = np.array([[253, 128, 127],[205, 155, 105], [248, 253, 255]], dtype=np.uint8)

    assert_equals((~img).get_ndarray().data, np_res1.data)

def test_core_get_ndarray():
    np_array = np.array([[2, 127, 128], [50, 100, 150], [7, 2, 0]],
                        dtype=np.uint8)
    img = CoreImage(array=np_array)
    assert_equals(img.get_ndarray().data, np_array.data)

def test_core_get_gray_ndarray():
    np_array = np.array([[[2, 127, 128], [50, 100, 150], [7, 2, 0]],
                        [[253, 128, 127],[205, 155, 105], [248, 253, 255]],
                        [[16, 81, 16],[81, 16, 81], [255, 16, 1]]],
                        dtype=np.uint8)
    img = CoreImage(array=np_array)
    gray = img.to_gray()

    assert_equals(img.get_gray_ndarray().data, gray.get_ndarray().data)

def test_core_get_fp_ndarray():
    np_array = np.array([[2, 127, 128], [50, 100, 150], [7, 2, 0]],
                        dtype=np.uint8)
    img = CoreImage(array=np_array)
    fp_array = np_array.astype(np.float32)
    assert_equals(img.get_fp_ndarray().data, fp_array.data)

def test_core_get_empty():
    np_array = np.ones((10, 10, 3), np.uint8)
    img = CoreImage(array=np_array)
    empty1 = np.zeros((img.height, img.width, 1), dtype=np.uint8)
    empty2 = np.zeros((img.height, img.width, 2), dtype=np.uint8)
    empty3 = np.zeros((img.height, img.width, 3), dtype=np.uint8)

    assert_equals(img.get_empty(1).data, empty1.data)
    assert_equals(img.get_empty(2).data, empty2.data)
    assert_equals(img.get_empty(3).data, empty3.data)

def test_core_convert():
    np_array = np.array([[[2, 127, 128], [50, 100, 150], [7, 2, 0]],
                        [[253, 128, 127],[205, 155, 105], [248, 253, 255]],
                        [[16, 81, 16],[81, 16, 81], [255, 16, 1]]],
                        dtype=np.uint8)
    np_array2 = np.array([[[128, 127, 2], [150, 100, 50], [0, 2, 7]],
                        [[127, 128, 253],[105, 155, 205], [255, 253, 248]],
                        [[16, 81, 16],[81, 16, 81], [1, 16, 255]]],
                        dtype=np.uint8)
    img = CoreImage(array=np_array)
    
    res1 = CoreImage.convert(np_array, img._color_space, img._color_space)
    res2 = CoreImage.convert(np_array, img._color_space, CoreImage.RGB)

    assert_equals(res1.data, np_array.data)
    assert_equals(res2.data, np_array2.data)

def test_core_to_color_space():
    np_array = np.array([[[2, 127, 128], [50, 100, 150], [7, 2, 0]],
                        [[253, 128, 127],[205, 155, 105], [248, 253, 255]],
                        [[16, 81, 16],[81, 16, 81], [255, 16, 1]]],
                        dtype=np.uint8)
    np_array2 = np.array([[[128, 127, 2], [150, 100, 50], [0, 2, 7]],
                        [[127, 128, 253],[105, 155, 205], [255, 253, 248]],
                        [[16, 81, 16],[81, 16, 81], [1, 16, 255]]],
                        dtype=np.uint8)
    img = CoreImage(array=np_array)
    
    res1 = img.to_color_space(img._color_space)
    res2 = img.to_color_space(img.RGB)

    assert_equals(res1.get_ndarray().data, np_array.data)
    assert_equals(res2.get_ndarray().data, np_array2.data)
    assert_equals(res2.is_rgb(), True)

def test_core_to_string():
    np_array = np.array([[[2, 127, 128], [50, 100, 150], [7, 2, 0]],
                        [[253, 128, 127],[205, 155, 105], [248, 253, 255]],
                        [[16, 81, 16],[81, 16, 81], [255, 16, 1]]],
                        dtype=np.uint8)
    img = CoreImage(array=np_array)

    assert_equals(img.to_string(), np_array.tostring())

def test_core_clear():
    np_array = np.array([[[2, 127, 128], [50, 100, 150]],
                        [[253, 128, 127],[205, 155, 105]],
                        [[16, 81, 16],[81, 16, 81]]],
                        dtype=np.uint8)
    np_array_gray = np.array([[2, 127, 128], [50, 100, 150]],
                        dtype=np.uint8)

    img = CoreImage(array=np_array)
    gray_img = CoreImage(array=np_array_gray)
    zero_array = np.zeros(np_array.shape, dtype=np.uint8)
    zero_gray_array = np.zeros(np_array_gray.shape, dtype=np.uint8)

    img.clear()
    gray_img.clear()

    assert_equals(img.get_ndarray().data, zero_array.data)
    assert_equals(gray_img.get_ndarray().data, zero_gray_array.data)

def test_core_is_empty():
    img = CoreImage(array=np.zeros((0,0), dtype=np.uint8))
    img1 = CoreImage(array=np.zeros((10,10), dtype=np.uint8))

    assert_equals(img.is_empty(), True)
    assert_equals(img1.is_empty(), False)

def test_core_get_area():
    img1 = CoreImage(array=np.zeros((10,10), dtype=np.uint8))
    assert_equals(img1.get_area(), 100)

def test_core_split_channels():
    np_array = np.array([[[2, 127, 128], [50, 100, 150]],
                        [[253, 128, 127],[205, 155, 105]],
                        [[16, 81, 16],[81, 16, 81]]],
                        dtype=np.uint8)
    img = CoreImage(array=np_array)

    c1, c2, c3 = img.split_channels()
    assert_equals(c1.get_ndarray().data, np_array[:, :, 0].copy().data)
    assert_equals(c2.get_ndarray().data, np_array[:, :, 1].copy().data)
    assert_equals(c3.get_ndarray().data, np_array[:, :, 2].copy().data)

def test_core_merge_channels():
    np_array = np.array([[[2, 127, 128], [50, 100, 150]],
                        [[253, 128, 127],[205, 155, 105]],
                        [[16, 81, 16],[81, 16, 81]]],
                        dtype=np.uint8)
    img = CoreImage(array=np_array)

    c1, c2, c3 = img.split_channels()

    c1 = c1 + c2
    c3 = c1 - c2

    img_new = img.merge_channels(c1, c2, c3)
    np_array_new = np.dstack((c1.get_ndarray(), c2.get_ndarray(), c3.get_ndarray()))

    assert_equals(img_new.get_ndarray().data, np_array_new.data)
