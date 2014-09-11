from nose.tools import assert_equals, assert_almost_equals
import numpy as np

from simplecv.image import Image
from simplecv.segmentation.running_segmentation import RunningSegmentation
from simplecv.tests.utils import perform_diff, perform_diff_blobs

def test_running_segmentation_add_image(): # add this case to existing test
    r = RunningSegmentation()
    img = r.add_image(None)

    assert_equals(img, None)

def test_running_segmentation_is_ready():
    r = RunningSegmentation()
    assert not r.is_ready()

    model_img = Image((512, 512))
    img = Image(source="lenna")
    r.model_img = model_img
    r.add_image(img)

    assert r.is_ready()

def test_running_segmentation_is_error():
    r = RunningSegmentation()
    assert not r.is_error()

def test_running_segmentation_reset_error():
    r = RunningSegmentation()
    r.reset_error()
    assert not r.error

def test_running_segmentation_reset():
    r = RunningSegmentation()
    r.model_img = Image("../data/sampleimages/black.png")    # random image, snice it doesn't matter - they're going to beset to None anyway
    r.diff_img = Image("../data/sampleimages/black.png")

    r.reset()

    assert_equals(r.model_img, None)
    assert_equals(r.diff_img, None)

def test_running_segmentation_get_raw_image():
    r = RunningSegmentation()

    img_ref = Image((512, 512))
    model_img = Image(img_ref.get_empty(3).astype(np.float32))

    img = Image(source="lenna")
    r.model_img = model_img
    r.add_image(img)

    result = r.get_raw_image()

    assert_equals(result.data, img.data)

def test_running_segmentation_get_segmented_image():
    r = RunningSegmentation()

    img_ref = Image((512, 512))
    model_img = Image(img_ref.get_empty(3).astype(np.float32))

    img = Image(source="lenna")
    r.model_img = model_img
    r.add_image(img)

    final = r.get_segmented_image()
    final1 = r.get_segmented_image(False)
    result = [final, final1]
    name_stem = "test_running_segmentation_get_segmented_image"

    perform_diff(result, name_stem, 0.0)

def test_running_segmentation_state():
    r = RunningSegmentation(alpha=0.5, thresh=(40, 20, 30))
    img_ref = Image((512, 512))
    model_img = Image(img_ref.get_empty(3).astype(np.float32))

    img = Image(source="lenna")
    r.model_img = model_img
    r.add_image(img)

    final = r.get_segmented_image()

    mydict = r.__getstate__()
    assert_equals(mydict['color_img'].data,
                  img.data)
    assert_equals(mydict['alpha'], 0.5)
    assert_equals(mydict['thresh'], (40, 20, 30))

    mydict['alpha'] = 0.3
    mydict['thresh'] = (30, 20, 50)

    newimg = Image(source="lenna")
    mydict['color_img'] = newimg

    r.__setstate__(mydict)

    assert_equals(r.alpha, 0.3)
    assert_equals(r.thresh, (30, 20, 50))
    assert_equals(r.color_img.data, newimg.data)
