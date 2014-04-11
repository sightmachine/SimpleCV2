import math

from nose.tools import assert_equals
import numpy as np

from simplecv.color import Color, ColorCurve
from simplecv.dft import DFT
from simplecv.features.detection import Corner
from simplecv.image import Image
from simplecv.tests.utils import perform_diff

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


def test_normalize():
    img = Image("lenna")
    img1 = img.normalize()
    img2 = img.normalize(min_cut=0, max_cut=0)
    result = [img1, img2]
    name_stem = "test_image_normalize"
    perform_diff(result, name_stem, 5)


def test_get_lightness():
    img = Image('lenna')
    i = img.get_lightness()
    if int(i[27, 42]) == int((max(img[27, 42]) + min(img[27, 42])) / 2):
        pass
    else:
        assert False


def test_get_luminosity():
    img = Image('lenna')
    i = img.get_luminosity()
    assert_equals(150, i[27, 42])


def test_get_average():
    img = Image('lenna')
    i = img.get_average()
    if int(i[0, 0]) == int((img[0, 0][0]
                            + img[0, 0][1]
                            + img[0, 0][2]) / 3):
        pass
    else:
        assert False


# FIXME: the following tests should be merged
def test_motion_blur():
    i = Image('lenna')
    d = ('n', 's', 'e', 'w', 'ne', 'nw', 'se', 'sw')
    i0 = i.motion_blur(intensity=20, direction=d[0])
    i1 = i.motion_blur(intensity=20, direction=d[1])
    i2 = i.motion_blur(intensity=20, direction=d[2])
    i3 = i.motion_blur(intensity=20, direction=d[3])
    i4 = i.motion_blur(intensity=10, direction=d[4])
    i5 = i.motion_blur(intensity=10, direction=d[5])
    i6 = i.motion_blur(intensity=10, direction=d[6])
    i7 = i.motion_blur(intensity=10, direction=d[7])
    a = i.motion_blur(intensity=0)
    c = 0
    img = (i0, i1, i2, i3, i4, i5, i6, i7)
    for im in img:
        if im is not i:
            c += 1

    if c == 8 and a is i:
        pass
    else:
        assert False


def test_motion_blur2():
    image = Image('lenna')
    d = (-70, -45, -30, -10, 100, 150, 235, 420)
    p = (10, 20, 30, 40, 50, 60, 70, 80)
    img = []

    a = image.motion_blur2(0)
    for i in range(8):
        img += [image.motion_blur2(p[i], d[i])]
    c = 0
    for im in img:
        if im is not i:
            c += 1

    if c == 8 and a is image:
        pass
    else:
        assert False


def test_watershed():
    img = Image('../data/sampleimages/wshed.jpg')
    img1 = img.watershed()
    img2 = img.watershed(dilate=3, erode=2)
    img3 = img.watershed(mask=img.threshold(128), erode=1, dilate=1)
    my_mask = Image((img.width, img.height))
    my_mask = my_mask.flood_fill((0, 0), color=Color.WATERSHED_BG)
    mask = img.threshold(128)
    my_mask = my_mask - mask.dilate(2).to_bgr()
    my_mask = my_mask + mask.erode(2).to_bgr()
    img4 = img.watershed(mask=my_mask, use_my_mask=True)
    blobs = img.find_blobs_from_watershed(dilate=3, erode=2)
    blobs = img.find_blobs_from_watershed()
    blobs = img.find_blobs_from_watershed(mask=img.threshold(128), erode=1,
                                          dilate=1)
    blobs = img.find_blobs_from_watershed(mask=img.threshold(128), erode=1,
                                          dilate=1, invert=True)
    blobs = img.find_blobs_from_watershed(mask=my_mask, use_my_mask=True)
    result = [img1, img2, img3, img4]
    name_stem = "test_watershed"
    perform_diff(result, name_stem)


def test_pixelize():
    img = Image("../data/sampleimages/The1970s.png")
    img1 = img.pixelize(4)
    img2 = img.pixelize((5, 13))
    img3 = img.pixelize((img.width / 10, img.height))
    img4 = img.pixelize((img.width, img.height / 10))
    img5 = img.pixelize((12, 12), (200, 180, 250, 250))
    img6 = img.pixelize((12, 12), (600, 80, 250, 250))
    img7 = img.pixelize((12, 12), (600, 80, 250, 250), levels=4)
    img8 = img.pixelize((12, 12), levels=6)
    #img9 = img.pixelize(4, )
    #img10 = img.pixelize((5,13))
    #img11 = img.pixelize((img.width/10,img.height), mode=True)
    #img12 = img.pixelize((img.width,img.height/10), mode=True)
    #img13 = img.pixelize((12,12),(200,180,250,250), mode=True)
    #img14 = img.pixelize((12,12),(600,80,250,250), mode=True)
    #img15 = img.pixelize((12,12),(600,80,250,250),levels=4, mode=True)
    #img16 = img.pixelize((12,12),levels=6, mode=True)

    results = [img1, img2, img3, img4, img5, img6, img7, img8]
              # img9,img10,img11,img12,img13,img14,img15,img16]
    name_stem = "test_pixelize"
    perform_diff(results, name_stem)


def test_apply_dft_filter():
    img = Image("../data/sampleimages/RedDog2.jpg")
    flt = Image("../data/sampleimages/RedDogFlt.png")
    f1 = img.apply_dft_filter(flt)
    f2 = img.apply_dft_filter(flt, grayscale=True)
    results = [f1, f2]
    name_stem = "test_apply_dft_filter"
    perform_diff(results, name_stem)


def test_high_pass_filter():
    img = Image("../data/sampleimages/RedDog2.jpg")
    a = img.high_pass_filter(0.5)
    b = img.high_pass_filter(0.5, grayscale=True)
    c = img.high_pass_filter(0.5, y_cutoff=0.4)
    d = img.high_pass_filter(0.5, y_cutoff=0.4, grayscale=True)
    e = img.high_pass_filter([0.5, 0.4, 0.3])
    f = img.high_pass_filter([0.5, 0.4, 0.3], y_cutoff=[0.5, 0.4, 0.3])

    results = [a, b, c, d, e, f]
    name_stem = "test_high_pass_filter"
    perform_diff(results, name_stem)


def test_low_pass_filter():
    img = Image("../data/sampleimages/RedDog2.jpg")
    a = img.low_pass_filter(0.5)
    b = img.low_pass_filter(0.5, grayscale=True)
    c = img.low_pass_filter(0.5, y_cutoff=0.4)
    d = img.low_pass_filter(0.5, y_cutoff=0.4, grayscale=True)
    e = img.low_pass_filter([0.5, 0.4, 0.3])
    f = img.low_pass_filter([0.5, 0.4, 0.3], y_cutoff=[0.5, 0.4, 0.3])

    results = [a, b, c, d, e, f]
    name_stem = "test_low_pass_filter"
    perform_diff(results, name_stem)


def test_dft_gaussian():
    img = Image("../data/sampleimages/RedDog2.jpg")
    flt = DFT.create_gaussian_filter(dia=300, size=(300, 300), highpass=False)
    fltimg = img.filter(flt)
    fltimggray = img.filter(flt, grayscale=True)
    flt = DFT.create_gaussian_filter(dia=300, size=(300, 300), highpass=True)
    fltimg1 = img.filter(flt)
    fltimggray1 = img.filter(flt, grayscale=True)
    results = [fltimg, fltimggray, fltimg1, fltimggray1]
    name_stem = "test_dft_gaussian"
    perform_diff(results, name_stem)


def test_dft_butterworth():
    img = Image("../data/sampleimages/RedDog2.jpg")
    flt = DFT.create_butterworth_filter(dia=300, size=(300, 300), order=3,
                                        highpass=False)
    fltimg = img.filter(flt)
    fltimggray = img.filter(flt, grayscale=True)
    flt = DFT.create_butterworth_filter(dia=100, size=(300, 300), order=3,
                                        highpass=True)
    fltimg1 = img.filter(flt)
    fltimggray1 = img.filter(flt, grayscale=True)
    results = [fltimg, fltimggray, fltimg1, fltimggray1]
    name_stem = "test_dft_butterworth"
    perform_diff(results, name_stem)


def test_dft_lowpass():
    img = Image("../data/sampleimages/RedDog2.jpg")
    flt = DFT.create_lowpass_filter(x_cutoff=150, size=(600, 600))
    fltimg = img.filter(flt)
    fltimggray = img.filter(flt, grayscale=True)
    results = [fltimg, fltimggray]
    name_stem = "test_dft_lowpass"
    perform_diff(results, name_stem)


def test_dft_highpass():
    img = Image("../data/sampleimages/RedDog2.jpg")
    flt = DFT.create_lowpass_filter(x_cutoff=10, size=(600, 600))
    fltimg = img.filter(flt)
    fltimggray = img.filter(flt, grayscale=True)
    results = [fltimg, fltimggray]
    name_stem = "test_dft_highpass"
    perform_diff(results, name_stem)


def test_dft_notch():
    img = Image("../data/sampleimages/RedDog2.jpg")
    flt = DFT.create_notch_filter(dia1=500, size=(512, 512), ftype="lowpass")
    fltimg = img.filter(flt)
    fltimggray = img.filter(flt, grayscale=True)
    flt = DFT.create_notch_filter(dia1=300, size=(512, 512), ftype="highpass")
    fltimg1 = img.filter(flt)
    fltimggray1 = img.filter(flt, grayscale=True)
    results = [fltimg, fltimggray, fltimg1, fltimggray1]
    name_stem = "test_dft_notch"
    perform_diff(results, name_stem)


def test_band_pass_filter():
    img = Image("../data/sampleimages/RedDog2.jpg")
    a = img.band_pass_filter(0.1, 0.3)
    b = img.band_pass_filter(0.1, 0.3, grayscale=True)
    c = img.band_pass_filter(0.1, 0.3, y_cutoff_low=0.1, y_cutoff_high=0.3)
    d = img.band_pass_filter(0.1, 0.3, y_cutoff_low=0.1, y_cutoff_high=0.3,
                             grayscale=True)
    e = img.band_pass_filter([0.1, 0.2, 0.3], [0.5, 0.5, 0.5])
    f = img.band_pass_filter([0.1, 0.2, 0.3], [0.5, 0.5, 0.5],
                             y_cutoff_low=[0.1, 0.2, 0.3],
                             y_cutoff_high=[0.6, 0.6, 0.6])
    results = [a, b, c, d, e, f]
    name_stem = "test_band_pass_filter"
    perform_diff(results, name_stem)


def test_skeletonize():
    img = Image('simplecv')
    s = img.skeletonize()
    s2 = img.skeletonize(10)

    results = [s, s2]
    name_stem = "test_skeletonize"
    perform_diff(results, name_stem)


def test_threshold():
    img = Image('simplecv')
    results = []
    for t in range(0, 255):
        timg = img.threshold(t)
        if t % 64 == 0:
            results.append(timg)

    name_stem = "test_threshold"
    perform_diff(results, name_stem)


def test_smart_threshold():
    img = Image("../data/sampleimages/RatTop.png")
    mask = Image((img.width, img.height))
    mask.dl().circle((100, 100), 80, color=Color.MAYBE_BACKGROUND, filled=True)
    mask.dl().circle((100, 100), 60, color=Color.MAYBE_FOREGROUND, filled=True)
    mask.dl().circle((100, 100), 40, color=Color.FOREGROUND, filled=True)
    mask = mask.apply_layers()
    new_mask1 = img.smart_threshold(mask=mask)
    new_mask2 = img.smart_threshold(rect=(30, 30, 150, 185))

    results = [new_mask1, new_mask2]
    name_stem = "test_smart_threshold"
    perform_diff(results, name_stem)
