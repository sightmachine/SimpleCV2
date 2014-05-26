import numpy as np
from simplecv.image import Image
from simplecv.linescan import LineScan
from nose.tools import assert_equals, assert_is_none, nottest

def test_linescan():
    np_array = np.arange(0, 256, dtype=np.uint8).reshape(16,16)
    img = Image(array=np_array)
    ls1 = LineScan(np_array[:, 10], image=img, x=10)

    assert_equals(ls1, np_array[:, 10].tolist())

def test_linescan_sub():
    np_array = np.arange(0, 256, dtype=np.uint8).reshape(16,16)
    np_array2 = np.ones((16, 8), np.uint8)

    ls1 = LineScan(np_array[:, 4])
    ls2 = LineScan(np_array2[:, 5])
    ls3 = LineScan(np_array2[5, :])

    assert_is_none(ls1 - ls3)
    assert_equals(ls1 - ls2, (np_array[:, 4] - np_array2[:, 5]).tolist())

def test_linescan_add():
    np_array = np.arange(0, 256, dtype=np.uint8).reshape(16,16)
    np_array2 = np.ones((16, 8), np.uint8)

    ls1 = LineScan(np_array[:, 4])
    ls2 = LineScan(np_array2[:, 5])
    ls3 = LineScan(np_array2[5, :])

    assert_is_none(ls1 + ls3)
    assert_equals(ls1 + ls2, (np_array[:, 4] + np_array2[:, 5]).tolist())

def test_linescan_mul():
    np_array = np.arange(0, 256, dtype=np.uint8).reshape(16,16)
    np_array2 = np.ones((16, 8), np.uint8)

    ls1 = LineScan(np_array[:, 4])
    ls2 = LineScan(np_array2[:, 5])
    ls3 = LineScan(np_array2[5, :])

    assert_is_none(ls1 * ls3)
    assert_equals(ls1 * ls2, (np_array[:, 4] * np_array2[:, 5]).tolist())

def test_linescan_div():
    np_array = np.arange(0, 256, dtype=np.uint8).reshape(16,16)
    np_array2 = np.ones((16, 8), np.uint8)
    np_array3 = np.zeros((16, 8), np.uint8)

    ls1 = LineScan(np_array[:, 4])
    ls2 = LineScan(np_array2[:, 5])
    ls3 = LineScan(np_array2[5, :])
    ls4 = LineScan(np_array3[:, 3])

    assert_is_none(ls1 / ls3)
    assert_equals(ls1 / ls2, (np_array[:, 4] / np_array2[:, 5]).tolist())
    assert_is_none(ls1/ls4)

def test_linescan_running_average():
    img = Image('lenna')
    ls = img.get_line_scan(y=120)
    ra = ls.running_average(5)
    assert_equals(sum(ls[48:53]) / 5, ra[50])
    val = ls.running_average(5, "gaussian")
    assert_is_none(ls.running_average(6))


@nottest
def line_scan_perform_diff(o_linescan, p_linescan, func, **kwargs):
    n_linescan = func(o_linescan, **kwargs)
    diff = sum([(i - j) for i, j in zip(p_linescan, n_linescan)])
    if diff > 10 or diff < -10:
        return False
    return True


def test_linescan_smooth():
    img = Image("lenna")
    l1 = img.get_line_scan(x=60)
    l2 = l1.smooth(degree=7)
    assert line_scan_perform_diff(l1, l2, LineScan.smooth, degree=7)


def test_linescan_normalize():
    img = Image("lenna")
    l1 = img.get_line_scan(x=90)
    l2 = l1.normalize()
    assert line_scan_perform_diff(l1, l2, LineScan.normalize)


def test_linescan_scale():
    img = Image("lenna")
    l1 = img.get_line_scan(y=90)
    l2 = l1.scale()
    assert line_scan_perform_diff(l1, l2, LineScan.scale)


def test_linescan_derivative():
    img = Image("lenna")
    l1 = img.get_line_scan(y=140)
    l2 = l1.derivative()
    assert line_scan_perform_diff(l1, l2, LineScan.derivative)


def test_linescan_resample():
    img = Image("lenna")
    l1 = img.get_line_scan(pt1=(300, 300), pt2=(450, 500))
    l2 = l1.resample(n=50)
    assert line_scan_perform_diff(l1, l2, LineScan.resample, n=50)


def test_linescan_fit_to_model():
    def a_line(x, m, b):
        return x * m + b

    img = Image("lenna")
    l1 = img.get_line_scan(y=200)
    l2 = l1.fit_to_model(a_line)
    assert line_scan_perform_diff(l1, l2, LineScan.fit_to_model, f=a_line)


def test_linescan_convolve():
    kernel = [0, 2, 0, 4, 0, 2, 0]
    img = Image("lenna")
    l1 = img.get_line_scan(x=400)
    l2 = l1.convolve(kernel)
    assert line_scan_perform_diff(l1, l2, LineScan.convolve, kernel=kernel)


def test_linescan_threshold():
    img = Image("lenna")
    l1 = img.get_line_scan(x=350)
    l2 = l1.threshold(threshold=200, invert=True)
    assert line_scan_perform_diff(l1, l2, LineScan.threshold, threshold=200,
                                  invert=True)


def test_linescan_invert():
    img = Image("lenna")
    l1 = img.get_line_scan(y=200)
    l2 = l1.invert(max=40)
    assert line_scan_perform_diff(l1, l2, LineScan.invert, max=40)


def test_linescan_median():
    img = Image("lenna")
    l1 = img.get_line_scan(x=120)
    l2 = l1.median(sz=9)
    assert line_scan_perform_diff(l1, l2, LineScan.median, sz=9)


def test_linescan_median_filter():
    img = Image("lenna")
    l1 = img.get_line_scan(y=250)
    l2 = l1.median_filter(kernel_size=7)
    l3 = l1.median_filter(kernel_size=8)

    assert line_scan_perform_diff(l1, l2, LineScan.median_filter,
                                  kernel_size=7)
    assert line_scan_perform_diff(l1, l3, LineScan.median_filter,
                                  kernel_size=7)

def test_linescan_detrend():
    img = Image("lenna")
    l1 = img.get_line_scan(y=90)
    l2 = l1.detrend()
    assert line_scan_perform_diff(l1, l2, LineScan.detrend)

def test_linescan_create_empty_lut():
    lut1 = LineScan.create_empty_lut(0)
    lut2 = LineScan.create_empty_lut(-1)
    lut3 = LineScan.create_empty_lut(3)
    lut4 = LineScan.create_empty_lut((10, 200))

    l4 = np.around(np.linspace(10, 200, 256), 0).astype(np.uint8).tolist()

    assert_equals(lut1, np.zeros([1, 256], np.uint8).tolist()[0])
    assert_equals(lut2, np.arange(0, 256, dtype=np.uint8).tolist())
    assert_equals(lut3, (3*np.ones([1, 256], np.uint8)).tolist()[0])
    assert_equals(lut4, l4)

def test_linescan_fill_lut():
    lut = LineScan.create_empty_lut(0)
    np_array = np.arange(0, 256, dtype=np.uint8).reshape(16,16)
    img = Image(array=np_array)
    swatch = img.crop(0, 0, 10, 12)
    flut = LineScan.fill_lut(lut, swatch, 255)

    rlut = np.zeros((16, 16), np.uint8)
    for i in xrange(10):
        for j in xrange(12):
            rlut[j, i] = 255

    rlut = rlut.reshape(1, 256).tolist()[0]
    
    assert_equals(flut, rlut)

def test_linescan_mean():
    np_array = np.arange(0, 256, dtype=np.uint8).reshape(16,16)
    ls = LineScan(np_array[:, 4])
    assert_equals(ls.mean(), np.mean(np_array[:, 4]))

def test_linescan_variance():
    np_array = np.arange(0, 256, dtype=np.uint8).reshape(16,16)
    ls = LineScan(np_array[:, 4])
    assert_equals(ls.variance(), np.var(np_array[:, 4]))

def test_linescan_std():
    np_array = np.arange(0, 256, dtype=np.uint8).reshape(16,16)
    ls = LineScan(np_array[:, 4])
    assert_equals(ls.std(), np.std(np_array[:, 4]))

def test_linescan_find_first_idx_equal_to():
    np_array = np.arange(0, 256, dtype=np.uint8)
    np_array = np.append(np_array, np_array)
    np_array = np_array.reshape(32, 16)
    ls = LineScan(np_array[:, 4])
    assert_equals(ls.find_first_idx_equal_to(116), 7)
    assert_is_none(ls.find_first_idx_equal_to(50))

def test_linescan_find_last_idx_equal_to():
    np_array = np.arange(0, 256, dtype=np.uint8)
    np_array = np.append(np_array, np_array)
    np_array = np_array.reshape(32, 16)
    ls = LineScan(np_array[:, 4])
    assert_equals(ls.find_last_idx_equal_to(116), 23)
    assert_is_none(ls.find_last_idx_equal_to(50))

def test_linescan_find_first_idx_greater_than():
    np_array = np.arange(0, 256, dtype=np.uint8)
    np_array = np.append(np_array, np_array)
    np_array = np_array.reshape(32, 16)
    ls = LineScan(np_array[:, 4])
    assert_equals(ls.find_first_idx_greater_than(116), 8)
    assert_is_none(ls.find_first_idx_greater_than(244))

def test_linescan_apply_lut():
    np_array = np.arange(0, 256, dtype=np.uint8).reshape(16,16)
    img = Image(array=np_array)

    ls = LineScan(np_array[:, 5])

    lut = ls.create_empty_lut()
    lut = ls.fill_lut(lut, img.crop(2, 2, 6, 8))
    alut = ls.apply_lut(lut)

    rlut = np.arange(0, 256, dtype=np.uint8).reshape(16, 16)
    for i in xrange(2, 8):
        for j in xrange(2, 10):
            rlut[j, i] = 255

    retval = rlut[:, 5].tolist()

    assert_equals(alut, retval)

def test_linescan_find_valleys():
    np_array = np.arange(0, 256, dtype=np.uint8)
    np_array = np.append(np_array, np_array)
    np_array = np_array.reshape(32, 16)
    ls = LineScan(np_array[:, 4])
    valleys = ls.find_valleys(window=15)
    assert_equals(valleys, [(0, 4), (16, 4)])

def test_linescan_fit_spline():
    np_array = np.arange(0, 256, dtype=np.uint8)
    np_array = np.append(np_array, np_array)
    np_array = np_array.reshape(32, 16)
    ls = LineScan(np_array[:, 5])

    spl = ls.fit_spline(degree=5)
    assert_is_none(ls.fit_spline(degree=0))
