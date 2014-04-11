from nose.tools import nottest, assert_equals

from simplecv.factory import Factory
from simplecv.core.image import image_method, cached_method

img_method_called = False


@nottest
@image_method
@cached_method
def img_method(img, arg1=2, arg2=4, arg3=8, arg4=16):
    global img_method_called
    if img_method_called:
        raise Exception

    img_method_called = True
    return arg1 + arg2 + arg3 + arg4


def test_cached_method():
    global img_method_called
    img = Factory.Image((10, 10))

    r1 = img.img_method()
    r2 = img.img_method()
    assert_equals(r1, r2)
    img_method_called = False

    r1 = img.img_method(5)
    r2 = img.img_method(5)
    assert_equals(r1, r2)
    img_method_called = False

    r1 = img.img_method(5, 10)
    r2 = img.img_method(5, 10)
    assert_equals(r1, r2)
    img_method_called = False

    r1 = img.img_method(0, arg3=5, arg4=20)
    r2 = img.img_method(0, arg3=5, arg4=20)
    assert_equals(r1, r2)
    img_method_called = False

    r1 = img.img_method(arg2=5, arg4=10)
    r2 = img.img_method(arg2=5, arg4=10)
    assert_equals(r1, r2)
    img_method_called = False


def test_cached_method_update():
    global img_method_called
    img = Factory.Image((10, 10))

    assert_equals(30, img.img_method())
    img_method_called = False
    assert_equals(31, img.img_method(3))
    img_method_called = False
    assert_equals(32, img.img_method(4))
    img_method_called = False
    assert_equals(33, img.img_method(5))
    img_method_called = False
