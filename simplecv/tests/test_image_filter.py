import math

import numpy as np
from nose.tools import assert_equals

from simplecv.features.detection import Corner
from simplecv.image import Image
from simplecv.tests.utils import perform_diff
from simplecv.color import Color, ColorCurve

barcode = "../data/sampleimages/barcode.png"
greyscaleimage = "../data/sampleimages/greyscale.jpg"
testimage = "../data/sampleimages/9dots4lines.png"
testimage2 = "../data/sampleimages/aerospace.jpg"
blackimage = "../data/sampleimages/black.png"
testimageclr = "../data/sampleimages/statue_liberty.jpg"

topimg = "../data/sampleimages/RatTop.png"
bottomimg = "../data/sampleimages/RatBottom.png"
maskimg = "../data/sampleimages/RatMask.png"
alphamaskimg = "../data/sampleimages/RatAlphaMask.png"
alphasrcimg = "../data/sampleimages/GreenMaskSource.png"


def test_image_max_int():
    array1 = np.ones((2, 2, 3), dtype=np.uint8) * 5
    img1 = Image(array=array1)
    array = np.ones((2, 2, 3), dtype=np.uint8) * 20

    img = img1.maximum(20)
    assert_equals(array.data, img.get_ndarray().data)


def test_image_max_image():
    array1 = np.ones((2, 2, 3), dtype=np.uint8) * 2
    img1 = Image(array=array1)
    array2 = np.ones((2, 2, 3), dtype=np.uint8) * 3
    img2 = Image(array=array2)
    array = np.ones((2, 2, 3), dtype=np.uint8) * 3

    img = img1.maximum(img2)
    assert_equals(array.data, img.get_ndarray().data)


def test_image_min_int():
    array1 = np.ones((2, 2, 3), dtype=np.uint8) * 5
    img1 = Image(array=array1)
    array = np.ones((2, 2, 3), dtype=np.uint8) * 5

    img = img1.minimum(20)
    assert_equals(array.data, img.get_ndarray().data)


def test_image_min_image():
    array1 = np.ones((2, 2, 3), dtype=np.uint8) * 2
    img1 = Image(array=array1)
    array2 = np.ones((2, 2, 3), dtype=np.uint8) * 3
    img2 = Image(array=array2)
    array = np.ones((2, 2, 3), dtype=np.uint8) * 2

    img = img1.minimum(img2)
    assert_equals(array.data, img.get_ndarray().data)


def test_image_stretch():
    img = Image(source=greyscaleimage)
    stretched = img.stretch(100, 200)
    if stretched is None:
        assert False

    result = [stretched]
    name_stem = "test_stretch"
    perform_diff(result, name_stem)


def test_image_smooth():
    img = Image(source=testimage2)
    result = []
    result.append(img.smooth())
    result.append(img.smooth('bilateral', (3, 3), 4, 1))
    result.append(img.smooth('blur', (3, 3)))
    result.append(img.smooth('median', (3, 3)))
    result.append(img.smooth('gaussian', (5, 5), 0))
    result.append(img.smooth('bilateral', (3, 3), 4, 1, grayscale=False))
    result.append(img.smooth('blur', (3, 3), grayscale=True))
    result.append(img.smooth('median', (3, 3), grayscale=True))
    result.append(img.smooth('gaussian', (5, 5), 0, grayscale=True))
    name_stem = "test_image_smooth"
    perform_diff(result, name_stem)


def test_image_gamma_correct():
    img = Image(source=topimg)
    img2 = img.gamma_correct(1)
    img3 = img.gamma_correct(0.5)
    img4 = img.gamma_correct(2)
    result = []
    result.append(img3)
    result.append(img4)

    assert img3.mean_color() >= img2.mean_color()
    assert img4.mean_color() <= img2.mean_color()

    name_stem = "test_image_gamma_correct"
    perform_diff(result, name_stem)


def test_image_binarize():
    img = Image(source=testimage2)
    binary = img.binarize()
    binary2 = img.binarize((60, 100, 200))

    result = [binary, binary2]
    name_stem = "test_image_binarize"
    perform_diff(result, name_stem)


def test_image_binarize_adaptive():
    img = Image(source=testimage2)
    binary = img.binarize()
    result = [binary]
    name_stem = "test_image_binarize_adaptive"
    perform_diff(result, name_stem)


def test_color_colordistance():
    img = Image(source=blackimage)
    c1 = Corner(img, 1, 1)
    c2 = Corner(img, 1, 2)
    assert c1.color_distance(c2.mean_color()) == 0
    assert c1.color_distance((0, 0, 0)) == 0
    assert c1.color_distance((0, 0, 255)) == 255
    assert c1.color_distance((255, 255, 255)) == math.sqrt(255 ** 2 * 3)


def test_color_curve_hls():
    # These are the weights
    y = np.array([[0, 0], [64, 128], [192, 128], [255, 255]])
    curve = ColorCurve(y)
    img = Image(source=testimage)
    img2 = img.apply_hls_curve(curve, curve, curve)
    img3 = img - img2

    result = [img2, img3]
    name_stem = "test_color_curve_hls"
    perform_diff(result, name_stem)


def test_color_curve_rgb():
    # These are the weights
    y = np.array([[0, 0], [64, 128], [192, 128], [255, 255]])
    curve = ColorCurve(y)
    img = Image(source=testimage)
    img2 = img.apply_rgb_curve(curve, curve, curve)
    img3 = img - img2

    result = [img2, img3]
    name_stem = "test_color_curve_rgb"
    perform_diff(result, name_stem)


def test_color_curve_gray():
    # These are the weights
    y = np.array([[0, 0], [64, 128], [192, 128], [255, 255]])
    curve = ColorCurve(y)
    img = Image(source=testimage)
    img2 = img.apply_intensity_curve(curve)

    result = [img2]
    name_stem = "test_color_curve_gray"
    perform_diff(result, name_stem)


def test_image_dilate():
    img = Image(source=barcode)
    img2 = img.dilate(20)

    result = [img2]
    name_stem = "test_image_dilate"
    perform_diff(result, name_stem)


def test_image_erode():
    img = Image(source=barcode)
    img2 = img.erode(100)

    result = [img2]
    name_stem = "test_image_erode"
    perform_diff(result, name_stem)


def test_image_morph_open():
    img = Image(source=barcode)
    erode = img.erode()
    dilate = erode.dilate()
    result = img.morph_open()
    test = result - dilate

    results = [result]
    name_stem = "test_image_morph_open"
    perform_diff(results, name_stem)


def test_image_morph_close():
    img = Image(source=barcode)
    dilate = img.dilate()
    erode = dilate.erode()
    result = img.morph_close()
    test = result - erode

    results = [result]
    name_stem = "test_image_morph_close"
    perform_diff(results, name_stem)


def test_image_morph_grad():
    img = Image(source=barcode)
    dilate = img.dilate()
    erode = img.erode()
    dif = dilate - erode
    result = img.morph_gradient()
    test = result - dif

    results = [result]
    name_stem = "test_image_morph_grad"
    perform_diff(results, name_stem)


def test_image_convolve():
    img = Image(source=testimageclr)
    kernel = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    img2 = img.convolve(kernel, center=(2, 2))

    results = [img2]
    name_stem = "test_image_convolve"
    perform_diff(results, name_stem)


def test_create_binary_mask():
    img2 = Image(source='simplecv')
    results = []
    results.append(
        img2.create_binary_mask(color1=(0, 100, 100), color2=(255, 200, 200)))
    results.append(
        img2.create_binary_mask(color1=(0, 0, 0), color2=(128, 128, 128)))
    results.append(
        img2.create_binary_mask(color1=(0, 0, 128), color2=(255, 255, 255)))
    name_stem = "test_create_binary_mask"
    perform_diff(results, name_stem)


def test_apply_binary_mask():
    img = Image(source='simplecv')
    mask = img.create_binary_mask(color1=(0, 128, 128), color2=(255, 255, 255))
    results = []
    results.append(img.apply_binary_mask(mask))
    results.append(img.apply_binary_mask(mask, bg_color=Color.RED))
    name_stem = "test_apply_binary_mask"
    perform_diff(results, name_stem, tolerance=3.0)


def test_apply_pixel_func():
    img = Image(source='simplecv')

    def myfunc(pixels):
        b, g, r = pixels
        return r, g, b

    img = img.apply_pixel_function(myfunc)
    name_stem = "test_apply_pixel_func"
    results = [img]
    perform_diff(results, name_stem)


def test_create_alpha_mask():
    alpha_mask = Image(source=alphasrcimg)
    mask = alpha_mask.create_alpha_mask(hue=60)
    mask2 = alpha_mask.create_alpha_mask(hue_lb=59, hue_ub=61)
    top = Image(source=topimg)
    bottom = Image(source=bottomimg)
    bottom = bottom.blit(top, alpha_mask=mask2)
    results = [mask, mask2, bottom]
    name_stem = "test_create_alpha_mask"
    perform_diff(results, name_stem)
