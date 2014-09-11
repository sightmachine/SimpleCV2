from nose.tools import assert_equals

from simplecv.color import Color, ColorCurve, ColorMap
from simplecv.features.blob import Blob
from simplecv.image import Image
from simplecv.tests.utils import perform_diff


def test_hue_from_rgb():
    assert_equals(90.0, Color.get_hue_from_rgb((100, 200, 150)))


def test_hue_from_bgr():
    assert_equals(90.0, Color.get_hue_from_bgr((150, 200, 100)))


def test_hue_to_rgb():
    assert_equals((255, 0, 0), Color.hue_to_rgb(0))
    assert_equals((255, 128, 0), Color.hue_to_rgb(15))
    assert_equals((255, 255, 0), Color.hue_to_rgb(30))
    assert_equals((128, 255, 0), Color.hue_to_rgb(45))
    assert_equals((0, 255, 0), Color.hue_to_rgb(60))
    assert_equals((0, 255, 128), Color.hue_to_rgb(75))
    assert_equals((0, 255, 255), Color.hue_to_rgb(90))
    assert_equals((0, 128, 255), Color.hue_to_rgb(105))
    assert_equals((0, 0, 255), Color.hue_to_rgb(120))
    assert_equals((128, 0, 255), Color.hue_to_rgb(135))
    assert_equals((255, 0, 255), Color.hue_to_rgb(150))
    assert_equals((255, 0, 128), Color.hue_to_rgb(165))


def test_hue_to_bgr():
    assert_equals((0, 0, 255), Color.hue_to_bgr(0))
    assert_equals((0, 128, 255), Color.hue_to_bgr(15))
    assert_equals((0, 255, 255), Color.hue_to_bgr(30))
    assert_equals((0, 255, 128), Color.hue_to_bgr(45))
    assert_equals((0, 255, 0), Color.hue_to_bgr(60))
    assert_equals((128, 255, 0), Color.hue_to_bgr(75))
    assert_equals((255, 255, 0), Color.hue_to_bgr(90))
    assert_equals((255, 128, 0), Color.hue_to_bgr(105))
    assert_equals((255, 0, 0), Color.hue_to_bgr(120))
    assert_equals((255, 0, 128), Color.hue_to_bgr(135))
    assert_equals((255, 0, 255), Color.hue_to_bgr(150))
    assert_equals((128, 0, 255), Color.hue_to_bgr(165))

# def test_color_get_random():
    

def test_color_get_lightness():
    # y = Color.YELLOW()
    retval = Color.get_lightness((255, 255, 0))
    assert_equals(retval, 127)


def test_color_get_luminosity():
    retval = Color.get_luminosity((255, 255, 0))
    assert_equals(retval, 234)


def test_color_color_map():
    import cv2
    img = Image("lenna")
    blobs = img.find(Blob)
    cm = ColorMap(Color.YELLOW, min(blobs.get_area()), max(blobs.get_area()))
    for b in blobs:
       b.draw(cm[b.area])
    result = [img]
    name_stem = "test_color_ColorMap"

    perform_diff(result, name_stem, 0.0)

    retval1 = cm.__getitem__(150000)
    assert_equals(retval1, (255, 255, 255))

    retval2 = cm.__getitem__(5)
    assert_equals(retval2, (255, 255, 0))
