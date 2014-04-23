from nose.tools import assert_equals, assert_almost_equals
import numpy as np

from simplecv.image import Image
from simplecv.segmentation.diff_segmentation import DiffSegmentation
from simplecv.tests.utils import perform_diff, perform_diff_blobs

def test_diff_segmentation_add_image():
    d = DiffSegmentation()
    retval = d.add_image(None)

    assert_equals(retval, None)

    d1 = DiffSegmentation()
    d1.grayonly_mode = True
    img1 = Image(source="lenna")
    img1_gray = img1.to_gray()

    d1.add_image(img1)

    assert_equals(d1.last_img.get_ndarray().data, img1_gray.get_ndarray().data)
    assert_equals(d1.curr_img, None)

    d2 = DiffSegmentation()
    d2.grayonly_mode = True
    img2 = Image(source="lenna")
    img2_gray = img2.to_gray()

    curr_img = Image((512, 512))
    nparray = curr_img.get_ndarray()
    nparray[0:512, 0:512] = (255, 255, 255)

    d2.last_img = Image((512, 512))
    d2.curr_img = curr_img

    # d2.add_image(img2)    # In this case, diff_img is unaffected. 
    #                       # Hence, cv2.absdiff is giving an error. Is this right?

    # assert_equals(d2.last_img.get_ndarray().data, curr_img.get_ndarray().data)
    # assert_equals(d2.color_img.get_ndarray().data, img2.get_ndarray().data)
    # assert_equals(d2.curr_img.get_ndarray().data, img2_gray.get_ndarray().data)

def test_diff_segmentation_is_ready():
    d = DiffSegmentation()
    assert not d.is_ready()

    img = Image(source="lenna")
    d.add_image(img)
    assert d.is_ready()

def test_diff_segmentation_is_error():
    d = DiffSegmentation()
    assert not d.is_error()

def test_diff_segmentation_reset_error():
    d = DiffSegmentation()
    d.error = True
    assert not d.reset_error() 

def test_diff_segmentation_reset():
    d = DiffSegmentation()
    img = Image(source="lenna")
    d.add_image(img)
    d.reset()

    assert_equals(d.last_img, None)
    assert_equals(d.diff_img, None)
    assert_equals(d.curr_img, None)

def test_diff_segmentation_get_raw_image():
    d = DiffSegmentation()
    img = Image(source="lenna")
    d.add_image(img)

    result = d.get_raw_image()   # work on line 52-54 of diff_segmentation

    diff_img = Image(img.get_empty(3))
    assert_equals(result.get_ndarray().data, diff_img.get_ndarray().data)

    # If the cv2.absdiff() is corrected in add_image, 
    # this test can be modified so that we test line 56-68 instead

def test_diff_segmentation_get_segmented_image():
    d = DiffSegmentation()
    img = Image(source="lenna")
    d.add_image(img)

    result = [d.get_segmented_image()]
    name_stem = "test_diff_segmentation_get_segmented_image"

    perform_diff(result, name_stem, 0.0)

    # If the cv2.absdiff() is corrected in add_image, 
    # this test can be modified so that we test line 56-68 instead
