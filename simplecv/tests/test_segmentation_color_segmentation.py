from nose.tools import assert_equals

from simplecv.image import Image
from simplecv.segmentation.color_segmentation import ColorSegmentation


def test_color_segmentation_add_image():
    c = ColorSegmentation()
    filename = "../data/sampleimages/04000.jpg"
    img = Image("../data/sampleimages/04000.jpg")

    c.add_image(filename)

    assert_equals(c.truth_img.data, img.data)


def test_color_segmentation_is_ready():
    c = ColorSegmentation()
    assert c.is_ready()


def test_color_segmentation_is_error():     # beta function (?)
    c = ColorSegmentation()
    assert not c.is_error()


def test_color_segmentation_reset_error():     # beta function (?)
    c = ColorSegmentation()
    assert not c.reset_error()


def test_color_segmentation_get_raw_image():
    c = ColorSegmentation()
    img = c.get_raw_image()

    assert_equals(img, None)

    c.cur_img = Image("../data/sampleimages/04000.jpg")
    img1 = c.get_raw_image()
    assert_equals(c.cur_img.data, img1.data)


def test_color_segmentation_get_segmented_image():
    c = ColorSegmentation()
    c.cur_img = Image("../data/sampleimages/RatMask.png")

    img = c.get_segmented_image()
    assert_equals(c.cur_img.data, img.data)


def test_color_segmentation_reset():
    c = ColorSegmentation()
    c.add_to_model((255, 0, 0))
    c.add_to_model((0, 255, 0))
    assert_equals(len(c.color_model.data), 2)
    c.reset()
    assert_equals(c.color_model.data, {})


def test_color_segmentation_add_to_model():
    c = ColorSegmentation()
    c.add_to_model((255, 0, 0))
    c.add_to_model((0, 255, 0))
    assert_equals(len(c.color_model.data), 2)


def test_color_segmentation_subtract_model():
    c = ColorSegmentation()
    c.add_to_model((255, 0, 0))
    c.add_to_model((0, 255, 0))
    assert_equals(len(c.color_model.data), 2)
    c.subtract_model((255, 0, 0))
    assert_equals(len(c.color_model.data), 1)

# FIXME: add test for pickle
def test_color_segmentation_state():
    c = ColorSegmentation()
    c.cur_img = Image("../data/sampleimages/RatMask.png")
    c.add_to_model((255, 0, 0))
    c.add_to_model((0, 255, 0))
    img = c.get_segmented_image()
