from nose.tools import assert_equals

from simplecv.color import Color


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
