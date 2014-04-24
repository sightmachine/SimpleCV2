from nose.tools import assert_equals, assert_almost_equals
import numpy as np

from simplecv.image import Image
from simplecv.segmentation.diff_segmentation import DiffSegmentation
from simplecv.tests.utils import perform_diff, perform_diff_blobs
from simplecv.factory import Factory

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

    curr_img = Factory.Image(img2.get_empty(3))
    nparray = curr_img.get_ndarray()
    nparray[0:512, 0:512] = (255, 255, 255)

    d2.last_img = Factory.Image(img2.get_empty(1))
    d2.curr_img = curr_img
    d2.add_image(img2)    # In this case, diff_img is unaffected. 

    assert_equals(d2.last_img.get_ndarray().data, img2.to_gray().get_ndarray().data)
    assert_equals(d2.color_img.get_ndarray().data, img2.get_ndarray().data)
    assert_equals(d2.curr_img.get_ndarray().data, img2_gray.get_ndarray().data)

    d3 = DiffSegmentation()
    img3 = Image(source="lenna")

    curr_img = Factory.Image(img3.get_empty(3))
    nparray = curr_img.get_ndarray()
    nparray[0:512, 0:512] = (255, 255, 255)

    d3.last_img = Factory.Image(img3.get_empty(3))
    d3.curr_img = curr_img
    d3.add_image(img3)    # In this case, diff_img is unaffected. 

    assert_equals(d3.last_img.get_ndarray().data, img3.get_ndarray().data)
    assert_equals(d3.color_img.get_ndarray().data, img3.get_ndarray().data)
    assert_equals(d3.curr_img.get_ndarray().data, img3.get_ndarray().data)

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

    result = d.get_raw_image()

    diff_img = Image(img.get_empty(3))
    assert_equals(result.get_ndarray().data, diff_img.get_ndarray().data)

def test_diff_segmentation_get_segmented_image():
    d = DiffSegmentation(threshold=(50, 80, 100))
    img = Image(source="lenna")
    d.add_image(img)
    d.add_image(img.rotate90())
    result = [d.get_segmented_image(), d.get_segmented_image(False)]
    name_stem = "test_diff_segmentation_get_segmented_image"

    perform_diff(result, name_stem, 0.0)

def test_diff_segmentation_state():
    d = DiffSegmentation(threshold=(30, 50, 20))
    img = Image(source="lenna")
    d.add_image(img)

    mydict = d.__getstate__()

    assert_equals(mydict['threshold'], (30, 50, 20))
    assert_equals(mydict['grayonly_mode'], False)
    assert_equals(mydict['last_img'].get_ndarray().data,
                  img.get_ndarray().data)

    last_img = img.to_bgr()
    diff_img = img.to_hsv()

    mydict['diff_img'] = diff_img
    mydict['last_img'] = last_img
    mydict['threshold'] = (20, 50, 60)
    mydict['grayonly_mode'] = True

    d.__setstate__(mydict)

    assert_equals(d.threshold, (20, 50, 60))
    assert_equals(d.grayonly_mode, True)
    assert_equals(d.last_img.get_ndarray().data, last_img.get_ndarray().data)
    assert_equals(d.diff_img.get_ndarray().data, diff_img.get_ndarray().data)
