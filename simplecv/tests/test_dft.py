import numpy as np
from simplecv.image import Image
from simplecv.dft import DFT
from nose.tools import assert_equals, assert_is_none

def test_dft():
    dft = DFT()
    assert_equals(type(dft), DFT)

    dft = DFT(size=(200, 300), channels=2, dia=100, type="gaussian",
              y_cutoff_high=200, x_cutoff_low=100)
    assert_equals(dft.width, 200)
    assert_equals(dft.height, 300)
    assert_equals(dft.channels, 2)
    assert_equals(dft._dia, 100)
    assert_equals(dft._type, "gaussian")
    assert_equals(dft._numpy, None)
    assert_equals(dft._image, None)
    assert_equals(dft._order, 0)
    assert_equals(dft._freqpass, "")
    assert_equals(dft._x_cutoff_low, 100)
    assert_equals(dft._y_cutoff_low, 0)
    assert_equals(dft._x_cutoff_high, 0)
    assert_equals(dft._y_cutoff_high, 200)

def test_dft_add():
    size = (200, 100)
    dia = 100

    sz_x, sz_y = size
    x0 = sz_x / 2
    y0 = sz_y / 2
    x, y = np.meshgrid(np.arange(sz_x), np.arange(sz_y))
    d = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)
    flt = 255 * np.exp(-0.5 * (d / dia) ** 2)

    dia1 = 80
    flt1 = 255 * np.exp(-0.5 * (d / dia1) ** 2)
    flt1 = 255 - flt1

    dft = DFT(size=size, numpyarray=flt)
    dft1 = DFT(size=size, numpyarray=flt1)
    dft3 = DFT(width=sz_x*2, height=sz_y/2, numpyarray=flt1)

    dft2 = dft + dft1
    assert_equals(dft2._numpy.data, (flt+flt1).data)

    # invalid params
    dft2 = dft + flt1
    assert_is_none(dft2)

    dft4 = dft3 + dft
    assert_is_none(dft4)

def test_dft_invert():
    arr = np.arange(0, 256).reshape(16,16)
    arr1 = 255 - arr

    dft = DFT(numpyarray=arr)
    dft1 = ~dft

    assert_equals(dft1._numpy.data, arr1.data)

def test_dft_update():
    arr = np.arange(0, 256).reshape(16,16)
    dft = DFT(numpyarray=arr)

    dft1 = DFT(channels=3, dia=200, type="gaussian",
                y_cutoff_low=200, x_cutoff_high=100)

    dft._update_params(dft1)
    assert_equals(dft.channels, dft1.channels)
    assert_equals(dft._dia, dft1._dia)
    assert_equals(dft._type, dft1._type)
    assert_equals(dft._order, dft1._order)
    assert_equals(dft._x_cutoff_high, dft1._x_cutoff_high)
    assert_equals(dft._x_cutoff_low, dft1._x_cutoff_low)
    assert_equals(dft._y_cutoff_high, dft1._y_cutoff_high)
    assert_equals(dft._y_cutoff_low, dft1._y_cutoff_low)

def test_dft_create_gaussian_filter():
    gauss_filter1 = DFT.create_gaussian_filter([100])
    gauss_filter2 = DFT.create_gaussian_filter(100)
    gauss_filter3 = DFT.create_gaussian_filter([100, 80, 75], size=(100, 80),
                                              highpass=True)

    assert_equals(gauss_filter1._numpy.data, gauss_filter2._numpy.data)
    assert_equals(gauss_filter3._numpy.shape, (80, 100, 3))
    assert_equals(gauss_filter3._type, "Gaussian")

    # invalid params
    assert_is_none(DFT.create_gaussian_filter([100, 80]))

def test_dft_create_butterworth_filter():
    filter1 = DFT.create_butterworth_filter([100])
    filter2 = DFT.create_butterworth_filter(100)
    filter3 = DFT.create_butterworth_filter([100, 80, 75], size=(100, 80),
                                            highpass=True)

    assert_equals(filter1._numpy.data, filter2._numpy.data)
    assert_equals(filter3._numpy.shape, (80, 100, 3))
    assert_equals(filter3._type, "Butterworth")

    # invalid params
    assert_is_none(DFT.create_butterworth_filter([100, 80]))

def test_dft_create_lowpass_filter():
    filter1 = DFT.create_lowpass_filter(x_cutoff=75, size=(320, 280))
    filter2 = DFT.create_lowpass_filter(x_cutoff=[75], size=(320, 280))
    filter3 = DFT.create_lowpass_filter(x_cutoff=[75, 125, 25],
                                        size=(320, 280))
    filter4 = DFT.create_lowpass_filter(x_cutoff=75, y_cutoff=25,
                                         size=(280, 320))
    filter5 = DFT.create_lowpass_filter(x_cutoff=75, y_cutoff=[25],
                                         size=(280, 320))
    filter6 = DFT.create_lowpass_filter(x_cutoff=[75, 125, 80], y_cutoff=[25],
                                         size=(280, 320))
    filter7 = DFT.create_lowpass_filter(x_cutoff=[75, 125, 80], y_cutoff=[25, 25, 25],
                                         size=(280, 320))

    assert_equals(filter1._numpy.data, filter2._numpy.data)
    assert_equals(filter1._numpy.data, filter3._numpy[:, :, 0].copy().data)
    assert_equals(filter4._numpy.data, filter5._numpy.data)
    assert_equals(filter6._numpy.data, filter7._numpy.data)

    # invalid params
    assert_is_none(DFT.create_lowpass_filter([100, 200]))
    assert_is_none(DFT.create_lowpass_filter([100], [200, 300]))
    assert_is_none(DFT.create_lowpass_filter([100, 80, 30], [200, 300]))

def test_dft_create_highpass_filter():
    filter1 = DFT.create_highpass_filter(x_cutoff=75, size=(320, 280))
    filter2 = DFT.create_highpass_filter(x_cutoff=[75], size=(320, 280))
    filter3 = DFT.create_highpass_filter(x_cutoff=[75, 125, 25],
                                        size=(320, 280))
    filter4 = DFT.create_highpass_filter(x_cutoff=75, y_cutoff=25,
                                         size=(280, 320))
    filter5 = DFT.create_highpass_filter(x_cutoff=75, y_cutoff=[25],
                                         size=(280, 320))
    filter6 = DFT.create_highpass_filter(x_cutoff=[75, 125, 80], y_cutoff=[25],
                                         size=(280, 320))
    filter7 = DFT.create_highpass_filter(x_cutoff=[75, 125, 80], y_cutoff=[25, 25, 25],
                                         size=(280, 320))

    assert_equals(filter1._numpy.data, filter2._numpy.data)
    assert_equals(filter1._numpy.data, filter3._numpy[:, :, 0].copy().data)
    assert_equals(filter4._numpy.data, filter5._numpy.data)
    assert_equals(filter6._numpy.data, filter7._numpy.data)

    # invalid params
    assert_is_none(DFT.create_highpass_filter([100, 200]))
    assert_is_none(DFT.create_highpass_filter([100], [200, 300]))
    assert_is_none(DFT.create_highpass_filter([100, 80, 30], [200, 300]))

def test_dft_create_bandpass_filter():
    filter1 = DFT.create_bandpass_filter(x_cutoff_low=20, x_cutoff_high=50)
    filter2 = DFT.create_bandpass_filter(x_cutoff_low=20, x_cutoff_high=50,
                                         y_cutoff_low=20, y_cutoff_high=50)

    filter3 = DFT.create_bandpass_filter(x_cutoff_low=[20, 30, 25],
                                         x_cutoff_high=[100, 120, 125],
                                         y_cutoff_low=[10, 5, 20],
                                         y_cutoff_high=[100, 90, 80],
                                         size=(250, 300))
    filter4 = DFT.create_lowpass_filter(x_cutoff=[20, 30, 25],
                                        y_cutoff=[10, 5, 20],
                                        size=(250, 300))
    filter5 = DFT.create_highpass_filter(x_cutoff=[100, 120, 125],
                                         y_cutoff=[100, 90, 80],
                                         size=(250, 300))

    assert_equals(filter1._numpy.data, filter2._numpy.data)
    assert_equals(filter3._numpy.data, (filter4._numpy + filter5._numpy).data)