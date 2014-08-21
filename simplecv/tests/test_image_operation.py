import numpy as np
import cv2
import math

from nose.tools import assert_equals, assert_list_equal, assert_is_none

from simplecv.tests.utils import perform_diff
from simplecv.image import Image

testimageclr = "../data/sampleimages/statue_liberty.jpg"
bottomImg = "../data/sampleimages/RatBottom.png"

def test_color_meancolor():
    a = np.arange(0, 256)
    b = a[::-1]
    c = np.copy(a) / 2
    a = a.reshape(16, 16)
    b = b.reshape(16, 16)
    c = c.reshape(16, 16)
    imgarr = np.dstack((a, b, c)).astype(np.uint8)
    img = Image(array=imgarr, color_space=Image.RGB)

    b, g, r = img.mean_color('BGR')
    print b, g, r
    if not (127 < r < 128 and 127 < g < 128 and 63 < b < 64):
        assert False

    r, g, b = img.mean_color('RGB')
    if not (127 < r < 128 and 127 < g < 128 and 63 < b < 64):
        assert False

    h, s, v = img.mean_color('HSV')
    if not (83 < h < 84 and 191 < s < 192 and 191 < v < 192):
        assert False

    x, y, z = img.mean_color('XYZ')
    if not (109 < x < 110 and 122 < y < 123 and 77 < z < 79):
        assert False

    gray = img.mean_color('Gray')
    if not (120 < gray < 121):
        assert False

    y, cr, cb = img.mean_color('YCrCb')
    if not (120 < y < 121 and 133 < cr < 134 and 96 < cb < 97):
        assert False

    h, l, s = img.mean_color('HLS')
    if not (84 < h < 85 and 117 < l < 118 and 160 < s < 161):
        assert False

    # incorrect color space
    assert_is_none(img.mean_color('UNKOWN'))


def test_image_edgemap():
    img_a = Image(source='simplecv')
    array = img_a.get_edge_map()
    results = [Image(array=array)]
    name_stem = "test_image_edgemap"
    perform_diff(results, name_stem)


def test_image_horz_scanline():
    img = Image(source='simplecv')
    sl = img.get_horz_scanline(10)
    assert len(sl.shape) == 2
    assert sl.shape[0] == img.width
    assert sl.shape[1] == 3

    # incorrect row value
    assert_is_none(img.get_horz_scanline(-10))
    assert_is_none(img.get_horz_scanline(img.height+10))


def test_image_vert_scanline():
    img = Image(source='simplecv')
    sl = img.get_vert_scanline(10)
    assert len(sl.shape) == 2
    assert sl.shape[0] == img.height
    assert sl.shape[1] == 3

    # incorrect column value
    assert_is_none(img.get_vert_scanline(-10))
    assert_is_none(img.get_vert_scanline(img.width+10))


def test_image_horz_scanline_gray():
    img = Image(source='simplecv')
    sl = img.get_horz_scanline_gray(10)
    assert len(sl.shape) == 1
    assert sl.shape[0] == img.width

    # incorrect row value
    assert_is_none(img.get_horz_scanline_gray(-10))
    assert_is_none(img.get_horz_scanline_gray(img.height+10))


def test_image_vert_scanline_gray():
    img = Image(source='simplecv')
    sl = img.get_vert_scanline_gray(10)
    assert len(sl.shape) == 1
    assert sl.shape[0] == img.width

    # incorrect column value
    assert_is_none(img.get_vert_scanline_gray(-10))
    assert_is_none(img.get_vert_scanline_gray(img.height+10))


def test_image_get_pixel():
    img = Image(source='simplecv')
    assert_list_equal([0, 0, 0], img.get_pixel(0, 0))

    # incorrect x, y values
    assert_is_none(img.get_pixel(-1, 50))
    assert_is_none(img.get_pixel(50, -1))
    assert_is_none(img.get_pixel(50, img.height+10))
    assert_is_none(img.get_pixel(img.width+10, 10))


def test_image_get_gray_pixel():
    img = Image(source='simplecv')
    assert_equals(0, img.get_gray_pixel(0, 0))
    # incorrect x, y values
    assert_is_none(img.get_gray_pixel(-1, 50))
    assert_is_none(img.get_gray_pixel(50, -1))
    assert_is_none(img.get_gray_pixel(50, img.height+10))
    assert_is_none(img.get_gray_pixel(img.width+10, 10))


def test_image_intergralimage():
    img = Image(source='simplecv')
    array = img.integral_image()
    assert isinstance(array, np.ndarray)
    assert_equals(np.int32, array.dtype)
    assert_equals(img.ndarray.shape[0] + 1, array.shape[0])
    assert_equals(img.ndarray.shape[1] + 1, array.shape[1])


def test_image_intergralimage_tilted():
    img = Image(source='simplecv')
    array = img.integral_image(tilted=True)
    assert isinstance(array, np.ndarray)
    assert_equals(np.int32, array.dtype)
    assert_equals(img.ndarray.shape[0] + 1, array.shape[0])
    assert_equals(img.ndarray.shape[1] + 1, array.shape[1])

def test_image_hue_histogram():
    array = np.arange(0,179,dtype=np.uint8)
    hsv_array = np.dstack((array, array, array))

    hsv_img = Image(array=hsv_array, color_space=Image.HSV)
    hist1 = hsv_img.hue_histogram()
    assert_equals(hist1.data, np.ones((1, 179)).astype(np.uint64).data)

    array1 = np.arange(0, 359, dtype=np.int64)
    hsv_array1 = np.dstack((array1, array1, array1))
    hsv_img1 = Image(array=hsv_array1, color_space=Image.HSV)
    hist2 = hsv_img1.hue_histogram(359,dynamic_range=False)

def test_image_hue_peaks():
    array = np.arange(0,179,dtype=np.uint8)
    for i in range(array.shape[0]):
        if i%8 == 0:
            array[i] = 80
    hsv_array = np.dstack((array, array, array))
    hsv_img = Image(array=hsv_array, color_space=Image.HSV)
    hist1 = hsv_img.hue_peaks()
    assert_equals(math.ceil(hist1[0][0]), 80)

def test_image_re_palette():
    img = Image(testimageclr)
    img = img.scale(0.1)  # scale down the image to reduce test time
    img2 = Image(bottomImg)
    img2 = img2.scale(0.1)  # scale down the image to reduce test time
    p = img.get_palette()
    img3 = img2.to_hsv().re_palette(p)
    p = img.get_palette(hue=True)
    img4 = img2.re_palette(p, hue=True)
    assert all(map(lambda a: isinstance(a, Image), [img3, img4]))

def test_image_get_threshold_crossing():
    np_array = np.arange(0, 256, dtype=np.uint8).reshape(16,16)
    img = Image(array=np_array)
    pt1 = (1, 5)
    pt2 = (13, 13)
    threshold = 140

    retval = img.get_threshold_crossing(pt1, pt2, threshold)
    assert_equals(retval, (6, 8))

    retval = img.get_threshold_crossing(pt1, pt2, threshold,
                                       departurethreshold=5)
    assert_equals(retval, (6, 8))

    retval = img.get_threshold_crossing(pt1, pt2, threshold, False)
    assert_equals(retval, (-1, -1))

    retval = img.get_threshold_crossing(pt2, pt1, threshold, False)
    assert_equals(retval, (7, 9))

def test_image_get_diagonal_scanline_grey():
    np_array = np.arange(0, 256, dtype=np.uint8).reshape(16,16)
    img = Image(array=np_array)
    pt1 = (5, 5)
    pt2 = (8, 8)

    scanline = img.get_diagonal_scanline_grey(pt1, pt2)
    expected_scanline = np.array([85, 85, 102, 119]).astype(img.dtype)
    assert_equals(scanline.data, expected_scanline.data)

def test_image_get_line_scan():
    def lsstuff(ls):
        def a_line(x, m, b):
            return m * x + b

        ls2 = ls.smooth(degree=4)
        ls2 = ls2.normalize()
        ls2 = ls2.scale(value_range=[-1, 1])
        ls2 = ls2.derivative()
        ls2 = ls2.resample(100)
        ls2 = ls2.convolve([.25, 0.25, 0.25, 0.25])
        ls2.minima()
        ls2.maxima()
        ls2.local_minima()
        ls2.local_maxima()
        fft, f = ls2.fft()
        ls3 = ls2.ifft(fft)
        ls4 = ls3.fit_to_model(a_line)
        ls4.get_model_parameters(a_line)

    img = Image("lenna")
    ls = img.get_line_scan(x=128, channel=1)
    lsstuff(ls)
    ls = img.get_line_scan(y=128)
    lsstuff(ls)
    ls = img.get_line_scan(pt1=(0, 0), pt2=(128, 128), channel=2)
    lsstuff(ls)

    # incorrect scanline params
    assert_is_none(img.get_line_scan(x=-10))
    assert_is_none(img.get_line_scan(x=img.width))
    assert_is_none(img.get_line_scan(x=img.width+10))

    assert_is_none(img.get_line_scan(y=-10))
    assert_is_none(img.get_line_scan(y=img.height))
    assert_is_none(img.get_line_scan(y=img.height+10))

    assert_is_none(img.get_line_scan())

def test_image_set_line_scan():

    # grey linescan
    np_array = np.arange(0, 256, dtype=np.uint8).reshape(16,16)
    img = Image(array=np_array)

    # x linescan
    x_linescan = img.get_line_scan(x=10)
    x_linescan.reverse()
    new_img = img.set_line_scan(x_linescan, x=8)
    new_linescan = new_img.get_line_scan(x=8)
    assert_equals(x_linescan, new_linescan)

    #incorrect x value
    assert_is_none(img.set_line_scan(x_linescan, x=-1))
    assert_is_none(img.set_line_scan(x_linescan, x=img.width))
    assert_is_none(img.set_line_scan(x_linescan, x=img.width+5))

    #provide no params
    new_img = img.set_line_scan(x_linescan)
    new_linescan = new_img.get_line_scan(x=10)
    assert_equals(x_linescan, new_linescan)

    # y linescan
    y_linescan = img.get_line_scan(y=5)
    y_linescan.reverse()
    new_img = img.set_line_scan(y_linescan, y=10)
    new_linescan = new_img.get_line_scan(y=10)
    assert_equals(y_linescan, new_linescan)

    #incorrect y value
    assert_is_none(img.set_line_scan(y_linescan, y=-1))
    assert_is_none(img.set_line_scan(y_linescan, y=img.height))
    assert_is_none(img.set_line_scan(y_linescan, y=img.height+5))

    #provide no params
    new_img = img.set_line_scan(y_linescan)
    new_linescan = new_img.get_line_scan(y=5)
    assert_equals(y_linescan, new_linescan)

    # provide points
    linescan = img.get_line_scan(pt1=(5, 5), pt2=(8, 8))
    linescan.reverse()
    new_img = img.set_line_scan(linescan, pt1=(8, 8), pt2=(11, 5)) # no resampling
    new_linescan = new_img.get_line_scan(pt1=(8, 8), pt2=(11, 5))
    assert_equals(linescan, new_linescan)

    resampled_linecan = linescan.resample(img.height)
    new_img = img.set_line_scan(linescan, x=5)
    new_linescan = new_img.get_line_scan(x=5)
    resampled_linecan = [int(x) for x in resampled_linecan]
    assert_equals(resampled_linecan, new_linescan)

    # provide no params
    new_img = img.set_line_scan(linescan)
    new_linescan = new_img.get_line_scan(pt1=linescan.pt1, pt2=linescan.pt2)
    assert_equals(linescan, new_linescan)

    # make linescan points None
    linescan.pt1 = None
    linescan.pt2 = None
    assert_is_none(new_img.set_line_scan(linescan))

    # For color image
    np_array = np.arange(0, 256, dtype=np.uint8).reshape(16,16)
    np_arr = np.arange(255, -1, -1, dtype=np.uint8).reshape(16, 16)
    np_array = np.dstack((np_array, np_arr, np_array))
    img = Image(array=np_array)

    x_linescan = img.get_line_scan(x=10, channel=0)
    x_linescan.reverse()
    new_img = img.set_line_scan(x_linescan, x=8, channel=1)
    new_linescan = new_img.get_line_scan(x=8, channel=1)
    assert_equals(x_linescan, new_linescan)

def test_image_replace_line_scan():
    np_array = np.arange(0, 256, dtype=np.uint8).reshape(16,16)
    img = Image(array=np_array)

    # x linescan
    x_linescan = img.get_line_scan(x=10)
    x_linescan.reverse()
    new_img = img.replace_line_scan(x_linescan)

    new_linescan = new_img.get_line_scan(x=10)
    assert_equals(x_linescan, new_linescan)

    # size mismatch
    r_linescan = x_linescan.resample(40)
    assert_is_none(img.replace_line_scan(r_linescan))

    # y linescan
    y_linescan = img.get_line_scan(y=5)
    y_linescan.reverse()
    new_img = img.replace_line_scan(y_linescan)

    new_linescan = new_img.get_line_scan(y=5)
    assert_equals(y_linescan, new_linescan)

    # size mismatch
    r_linescan = y_linescan.resample(5)
    assert_is_none(img.replace_line_scan(r_linescan))

    #provide points
    linescan = img.get_line_scan(pt1=(5, 5), pt2=(8, 2))
    new_img = img.replace_line_scan(linescan)
    new_linescan = img.get_line_scan(pt1=(5, 5), pt2=(8, 2))
    assert_equals(linescan, new_linescan)

    # resampling
    r_linescan = linescan.resample(15)
    rr_linescan = r_linescan.resample(4) # lenght of points

    new_img = img.replace_line_scan(r_linescan)
    new_linescan = new_img.get_line_scan(pt1=(5, 5), pt2=(8, 2))
    rr_linescan = [int(x) for x in rr_linescan]
    assert_equals(rr_linescan, new_linescan)

    # provide params
    new_img = img.replace_line_scan(x_linescan, x=5)
    new_linescan = new_img.get_line_scan(x=5)
    assert_equals(x_linescan, new_linescan)

    new_img = img.replace_line_scan(y_linescan, y=9)
    new_linescan = new_img.get_line_scan(y=9)
    assert_equals(y_linescan, new_linescan)

    new_img = img.replace_line_scan(linescan, pt1=(11, 11), pt2=(8, 14))
    new_linescan = new_img.get_line_scan(pt1=(11, 11), pt2=(8, 14))
    assert_equals(linescan, new_linescan)

    # provide channels in color images
    np_array = np.arange(0, 256, dtype=np.uint8).reshape(16,16)
    np_arr = np.arange(255, -1, -1, dtype=np.uint8).reshape(16, 16)
    np_array = np.dstack((np_array, np_arr, np_array))
    img = Image(array=np_array)

    x_linescan = img.get_line_scan(x=10, channel=0)
    x_linescan.reverse()
    new_img = img.replace_line_scan(x_linescan)
    new_linescan = new_img.get_line_scan(x=10, channel=0)
    assert_equals(x_linescan, new_linescan)

    new_img = img.replace_line_scan(x_linescan, channel=2)
    new_linescan = new_img.get_line_scan(x=10, channel=2)
    assert_equals(x_linescan, new_linescan)

    #invalid channel
    x_linescan.channel = 5
    assert_is_none(img.replace_line_scan(x_linescan))

def test_image_get_pixels_online():
    np_array = np.arange(0, 256, dtype=np.uint8).reshape(16,16)
    img = Image(array=np_array)
    pts = img.get_pixels_online(pt1=(5, 5), pt2=(2, 8))
    assert_equals(len(pts), 4)
    assert_equals(pts, [85, 100, 115, 130])

    # invlaid points
    assert_is_none(img.get_pixels_online(5, 6))
    assert_is_none(img.get_pixels_online((5, 5, 1), (6, 6, 0)))

def test_image_logical_and():
    img = Image("lenna")
    img1 = img.logical_and(img.invert())
    assert not img1.ndarray.all()

    img2 = img.logical_and(img.invert(), grayscale=False)
    assert not img2.ndarray.all()

    # size mismatch
    assert_is_none(img.logical_and(img.resize(img.width/2, img.height/2)))

def test_image_logical_or():
    img = Image("lenna")
    img1 = img.logical_or(img.invert())
    assert img1.ndarray.all()

    img2 = img.logical_or(img.invert(), grayscale=False)
    assert img2.ndarray.all()

    # size mismatch
    assert_is_none(img.logical_or(img.resize(img.width/2, img.height/2)))

def test_image_logical_nand():
    img = Image("lenna")
    img1 = img.logical_nand(img.invert())
    assert img1.ndarray.all()

    img2 = img.logical_nand(img.invert(), grayscale=False)
    assert img2.ndarray.all()

    # size mismatch
    assert_is_none(img.logical_nand(img.resize(img.width/2, img.height/2)))

def test_image_logical_xor():
    img = Image("lenna")
    img1 = img.logical_xor(img.invert())
    assert img1.ndarray.all()

    img2 = img.logical_xor(img.invert(), grayscale=False)
    assert img2.ndarray.all()

    # size mismatch
    assert_is_none(img.logical_xor(img.resize(img.width/2, img.height/2)))

def test_image_histograms():
    img = Image('lenna')
    h = img.vertical_histogram()
    assert_list_equal([14529, 5660,  4727,  7878,  12915,
                       21381, 17274, 15408, 16442, 15384], h.tolist())
    h = img.horizontal_histogram()
    assert_list_equal([14115, 12981, 15231, 15310, 12256,
                       13430, 13668, 11019, 12914, 10674], h.tolist())

    h = img.vertical_histogram(bins=3)
    assert_list_equal([27401, 51905, 52292], h.tolist())
    h = img.horizontal_histogram(bins=3)
    assert_list_equal([48418, 44229, 38951], h.tolist())

    h = img.vertical_histogram(threshold=10)
    assert_list_equal([26624, 26112, 26112, 26112, 26112,
                       26624, 26112, 26112, 26112, 26112], h.tolist())
    h = img.horizontal_histogram(threshold=255)
    assert_list_equal([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], h.tolist())

    h = img.vertical_histogram(normalize=True)
    assert_list_equal([0.0021563362000182377, 0.0008400346129880394,
                       0.0007015624762534386, 0.0011692213217526103,
                       0.0019167927609082206, 0.0031732826961655956,
                       0.0025637381457165004, 0.0022867938722472988,
                       0.002440256025927446,  0.0022832318880226144],
                      h.tolist())
    h = img.horizontal_histogram(normalize=True)
    assert_list_equal([0.002094891972142434,  0.0019265882175261023,
                       0.0022605242385902525, 0.002272249103329838,
                       0.0018189866107387652, 0.0019932270057295707,
                       0.0020285500159576897, 0.0016353960071581635,
                       0.0019166443448988587, 0.0015841924839283272],
                      h.tolist())

    h = img.vertical_histogram(for_plot=True, normalize=True)
    assert_equals(3, len(h))
    assert_list_equal([0.0, 51.2, 102.4, 153.60000000000002, 204.8,
                       256.0, 307.20000000000005, 358.40000000000003,
                       409.6, 460.8], h[0].tolist())
    assert_list_equal([0.0021563362000182377, 0.0008400346129880394,
                       0.0007015624762534386, 0.0011692213217526103,
                       0.0019167927609082206, 0.0031732826961655956,
                       0.0025637381457165004, 0.0022867938722472988,
                       0.002440256025927446,  0.0022832318880226144],
                      h[1].tolist())
    assert_equals(51, h[2])
    h = img.horizontal_histogram(for_plot=True, normalize=True)
    assert_list_equal([0.0, 51.2, 102.4, 153.60000000000002, 204.8,
                       256.0, 307.20000000000005, 358.40000000000003,
                       409.6, 460.8], h[0].tolist())
    assert_list_equal([0.002094891972142434,  0.0019265882175261023,
                       0.0022605242385902525, 0.002272249103329838,
                       0.0018189866107387652, 0.0019932270057295707,
                       0.0020285500159576897, 0.0016353960071581635,
                       0.0019166443448988587, 0.0015841924839283272],
                      h[1].tolist())
    assert_equals(51, h[2])

    # incorrect bin values
    assert_is_none(img.vertical_histogram(bins=0))
    assert_is_none(img.vertical_histogram(bins=-3))
    assert_is_none(img.horizontal_histogram(bins=0))
    assert_is_none(img.horizontal_histogram(bins=-3))

def test_image_gray_peaks():
    np_array = np.arange(0, 256, dtype=np.uint8)
    for i in range(np_array.shape[0]):
        if i%5==0:
            np_array[i] = 127

    np_array = np_array.reshape(16,16)
    img = Image(array=np_array)
    peak = img.gray_peaks()[0][0]
    assert_equals(peak, 127)

def test_image_back_project_hue_histogram():
    img = Image('lenna')
    img2 = Image('lyle')
    a = img2.get_normalized_hue_histogram()
    img_a = img.back_project_hue_histogram(a)
    img_b = img.back_project_hue_histogram((10, 10, 50, 50), smooth=False,
                                           full_color=True)
    img_c = img.back_project_hue_histogram(img2, threshold=1)
    result = [img_a, img_b, img_c]
    name_stem = "test_image_hist_back_proj"
    perform_diff(result, name_stem, 5)

    # invalid params
    assert_is_none(img.back_project_hue_histogram(model=None))
    assert_is_none(img.back_project_hue_histogram(model=1.0))
    assert_is_none(img.back_project_hue_histogram(model=img.ndarray))

