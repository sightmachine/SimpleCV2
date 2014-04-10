# SimpleCV DFT Library
#
# This library is used to crate DFT filters
import warnings

from numpy import arange, clip, dstack, exp, meshgrid, ogrid, ones, sqrt, zeros

from simplecv.image import Image


class DFT(object):
    """
    **SUMMARY**

    The DFT class is the refactored class to crate DFT filters which can
    be used to filter images by applying Digital Fourier Transform. This
    is a factory class to create various DFT filters.

    **PARAMETERS**

    Any of the following parameters can be supplied to create
    a simple DFT object.

    * *width*        - width of the filter
    * *height*       - height of the filter
    * *channels*     - number of channels of the filter
    * *size*         - size of the filter (width, height)
    * *_numpy*       - numpy array of the filter
    * *_image*       - SimpleCV.Image of the filter
    * *_dia*         - diameter of the filter
                      (applicable for gaussian, butterworth, notch)
    * *_type*        - Type of the filter
    * *_order*       - order of the butterworth filter
    * *_freqpass*    - frequency of the filter (lowpass, highpass, bandpass)
    * *_x_cutoff_low*  - Lower horizontal cut off frequency for lowpassfilter
    * *_y_cutoff_low*  - Lower vertical cut off frequency for lowpassfilter
    * *_x_cutoff_high* - Upper horizontal cut off frequency for highpassfilter
    * *_y_cutoff_high* - Upper vertical cut off frequency for highassfilter



    **EXAMPLE**

    >>> gauss = DFT.create_gaussian_filter(dia=40, size=(512, 512))
    >>> dft = DFT()
    >>> btw = dft.create_butterworth_filter(dia=300, order=2, size=(300, 300))

    """
    width = 0
    height = 0
    channels = 1
    _numpy = None
    _image = None
    _dia = 0
    _type = ""
    _order = 0
    _freqpass = ""
    _x_cutoff_low = 0
    _y_cutoff_low = 0
    _x_cutoff_high = 0
    _y_cutoff_high = 0

    def __init__(self, **kwargs):
        for key in kwargs:
            if key == 'width':
                self.width = kwargs[key]
            elif key == 'height':
                self.height = kwargs[key]
            elif key == 'channels':
                self.channels = kwargs[key]
            elif key == 'size':
                self.width, self.height = kwargs[key]
            elif key == 'numpyarray':
                self._numpy = kwargs[key]
            elif key == 'image':
                self._image = kwargs[key]
            elif key == 'dia':
                self._dia = kwargs[key]
            elif key == 'type':
                self._type = kwargs[key]
            elif key == 'order':
                self._order = kwargs[key]
            elif key == 'frequency':
                self._freqpass = kwargs[key]
            elif key == 'x_cutoff_low':
                self._x_cutoff_low = kwargs[key]
            elif key == 'y_cutoff_low':
                self._y_cutoff_low = kwargs[key]
            elif key == 'x_cutoff_high':
                self._x_cutoff_high = kwargs[key]
            elif key == 'y_cutoff_high':
                self._y_cutoff_high = kwargs[key]

    def __repr__(self):
        return "<SimpleCV.DFT Object: %s %s filter of size:(%d, %d) \
                and channels: %d>" % (self._type, self._freqpass, self.width,
                                      self.height, self.channels)

    def __add__(self, flt):
        if not isinstance(flt, type(self)):
            warnings.warn("Provide SimpleCV.DFT object")
            return None
        if self.size() != flt.size():
            warnings.warn("Both SimpleCV.DFT object must have the same size")
            return None
        flt_numpy = self._numpy + flt._numpy
        flt_image = Image(flt_numpy)
        ret_value = DFT(numpyarray=flt_numpy, image=flt_image,
                        size=flt_image.size())
        return ret_value

    def __invert__(self):
        return self.invert()

    def _update_params(self, flt):
        self.channels = flt.channels
        self._dia = flt._dia
        self._type = flt._type
        self._order = flt._order
        self._freqpass = flt._freqpass
        self._x_cutoff_low = flt._x_cutoff_low
        self._y_cutoff_low = flt._y_cutoff_low
        self._x_cutoff_high = flt._x_cutoff_high
        self._y_cutoff_high = flt._y_cutoff_high

    def invert(self):
        """
        **SUMMARY**

        Invert the filter. All values will be subtracted from 255.

        **RETURNS**

        Inverted Filter

        **EXAMPLE**

        >>> flt = DFT.create_gaussian_filter()
        >>> invertflt = flt.invert()
        """

        flt = self._numpy
        flt = 255 - flt
        img = Image(flt)
        inverted_filter = DFT(numpyarray=flt, image=img,
                              size=self.size(), type=self._type)
        inverted_filter._update_params(self)
        return inverted_filter

    @classmethod
    def create_gaussian_filter(self, dia=400, size=(64, 64), highpass=False):
        """
        **SUMMARY**

        Creates a gaussian filter of given size.

        **PARAMETERS**

        * *dia*       -  int - diameter of Gaussian filter
                      - list - provide a list of three diameters to create
                               a 3 channel filter
        * *size*      - size of the filter (width, height)
        * *highpass*: -  bool
                         True: highpass filter
                         False: lowpass filter

        **RETURNS**

        DFT filter.

        **EXAMPLE**

        >>> gauss = DFT.create_gaussian_filter(200, (512, 512),\
                                               highpass=True)

        >>> gauss = DFT.create_gaussian_filter([100, 120, 140], (512, 512), \
                                               highpass=False)
        >>> img = Image('lenna')
        >>> gauss.apply_filter(img).show()
        """
        if isinstance(dia, list):
            if len(dia) != 3 and len(dia) != 1:
                warnings.warn("diameter list must be of size 1 or 3")
                return None
            stacked_filter = DFT()
            for d in dia:
                stacked_filter = stacked_filter._stack_filters(
                    self.create_gaussian_filter(d, size, highpass))
            image = Image(stacked_filter._numpy)
            ret_value = DFT(numpyarray=stacked_filter._numpy, image=image,
                            dia=dia, channels=len(dia), size=size,
                            type="Gaussian",
                            frequency=stacked_filter._freqpass)
            return ret_value

        freqpass = "lowpass"
        sz_x, sz_y = size
        x0 = sz_x/2
        y0 = sz_y/2
        x, y = meshgrid(arange(sz_x), arange(sz_y))
        d = sqrt((x-x0)**2+(y-y0)**2)
        flt = 255*exp(-0.5*(d/dia)**2)
        if highpass:
            flt = 255 - flt
            freqpass = "highpass"
        img = Image(flt)
        ret_value = DFT(size=size, numpyarray=flt, image=img, dia=dia,
                        type="Gaussian", frequency=freqpass)
        return ret_value

    @classmethod
    def create_butterworth_filter(self, dia=400, size=(64, 64), order=2,
                                  highpass=False):
        """
        **SUMMARY**

        Creates a butterworth filter of given size and order.

        **PARAMETERS**

        * *dia*       - int - diameter of Gaussian filter
                      - list - provide a list of three diameters to create
                               a 3 channel filter
        * *size*      - size of the filter (width, height)
        * *order*     - order of the filter
        * *highpass*: -  bool
                         True: highpass filter
                         False: lowpass filter

        **RETURNS**

        DFT filter.

        **EXAMPLE**

        >>> flt = DFT.create_butterworth_filter(100, (512, 512), order=3,\
                                                highpass=True)
        >>> flt = DFT.create_butterworth_filter([100, 120, 140], (512, 512),\
                                                order=3, highpass=False)
        >>> img = Image('lenna')
        >>> flt.apply_filter(img).show()
        """
        if isinstance(dia, list):
            if len(dia) != 3 and len(dia) != 1:
                warnings.warn("diameter list must be of size 1 or 3")
                return None
            stackedfilter = DFT()
            for d in dia:
                stackedfilter = stackedfilter._stack_filters(
                    self.create_butterworth_filter(d, size, order, highpass))
            image = Image(stackedfilter._numpy)
            ret_value = DFT(numpyarray=stackedfilter._numpy, image=image,
                            dia=dia, channels=len(dia), size=size,
                            type=stackedfilter._type, order=order,
                            frequency=stackedfilter._freqpass)
            return ret_value
        freqpass = "lowpass"
        sz_x, sz_y = size
        x0 = sz_x/2
        y0 = sz_y/2
        x, y = meshgrid(arange(sz_x), arange(sz_y))
        d = sqrt((x-x0)**2+(y-y0)**2)
        flt = 255/(1.0 + (d/dia)**(order*2))
        if highpass:
            freqpass = "highpass"
            flt = 255 - flt
        img = Image(flt)
        ret_value = DFT(size=size, numpyarray=flt, image=img, dia=dia,
                        type="Butterworth", frequency=freqpass)
        return ret_value

    @classmethod
    def create_lowpass_filter(self, x_cutoff, y_cutoff=None, size=(64, 64)):
        """
        **SUMMARY**

        Creates a lowpass filter of given size and order.

        **PARAMETERS**

        * *x_cutoff*       - int - horizontal cut off frequency
                           - list - provide a list of three cut off frequencies
                                    to create a 3 channel filter
        * *y_cutoff*       - int - vertical cut off frequency
                           - list - provide a list of three cut off frequencies
                                    to create a 3 channel filter
        * *size*           - size of the filter (width, height)

        **RETURNS**

        DFT filter.

        **EXAMPLE**

        >>> flt = DFT.create_lowpass_filter(x_cutoff=75, size=(320, 280))

        >>> flt = DFT.create_lowpass_filter(x_cutoff=[75], size=(320, 280))

        >>> flt = DFT.create_lowpass_filter(x_cutoff=[75, 100, 120],\
                                            size=(320, 280))

        >>> flt = DFT.create_lowpass_filter(x_cutoff=75, y_cutoff=35,\
                                            size=(320, 280))

        >>> flt = DFT.create_lowpass_filter(x_cutoff=[75], y_cutoff=[35],\
                                            size=(320, 280))

        >>> flt = DFT.create_lowpass_filter(x_cutoff=[75, 100, 125], \
                                            y_cutoff=35,\
                                            size=(320, 280))
        >>> # y_cutoff will be [35, 35, 35]

        >>> flt = DFT.create_lowpass_filter(x_cutoff=[75, 113, 124],\
                                            y_cutoff=[35, 45, 90],\
                                            size=(320, 280))

        >>> img = Image('lenna')
        >>> flt.apply_filter(img).show()
        """
        if isinstance(x_cutoff, list):
            if len(x_cutoff) != 3 and len(x_cutoff) != 1:
                warnings.warn("x_cutoff list must be of size 3 or 1")
                return None
            if isinstance(y_cutoff, list):
                if len(y_cutoff) != 3 and len(y_cutoff) != 1:
                    warnings.warn("y_cutoff list must be of size 3 or 1")
                    return None
                if len(y_cutoff) == 1:
                    y_cutoff = [y_cutoff[0]]*len(x_cutoff)
            else:
                y_cutoff = [y_cutoff]*len(x_cutoff)
            stacked_filter = DFT()
            for xfreq, yfreq in zip(x_cutoff, y_cutoff):
                stacked_filter = stacked_filter._stack_filters(
                    self.create_lowpass_filter(xfreq, yfreq, size))
            image = Image(stacked_filter._numpy)
            ret_value = DFT(numpyarray=stacked_filter._numpy, image=image,
                            x_cutoff_low=x_cutoff, y_cutoff_low=y_cutoff,
                            channels=len(x_cutoff), size=size,
                            type=stacked_filter._type, order=self._order,
                            frequency=stacked_filter._freqpass)
            return ret_value

        w, h = size
        x_cutoff = clip(int(x_cutoff), 0, w/2)
        if y_cutoff is None:
            y_cutoff = x_cutoff
        y_cutoff = clip(int(y_cutoff), 0, h/2)
        flt = zeros((w, h))
        flt[0:x_cutoff, 0:y_cutoff] = 255
        flt[0:x_cutoff, h-y_cutoff:h] = 255
        flt[w-x_cutoff:w, 0:y_cutoff] = 255
        flt[w-x_cutoff:w, h-y_cutoff:h] = 255
        img = Image(flt)
        lowpass_filter = DFT(size=size, numpyarray=flt, image=img,
                             type="Lowpass", x_cutoff_low=x_cutoff,
                             y_cutoff_low=y_cutoff, frequency="lowpass")
        return lowpass_filter

    @classmethod
    def create_highpass_filter(self, x_cutoff, y_cutoff=None, size=(64, 64)):
        """
        **SUMMARY**

        Creates a highpass filter of given size and order.

        **PARAMETERS**

        * *x_cutoff*      -  int - horizontal cut off frequency
                          - list - provide a list of three cut off frequencies
                                   to create a 3 channel filter
        * *y_cutoff*      -  int - vertical cut off frequency
                          - list - provide a list of three cut off frequencies
                                   to create a 3 channel filter
        * *size*          - size of the filter (width, height)

        **RETURNS**

        DFT filter.

        **EXAMPLE**

        >>> flt = DFT.create_highpass_filter(x_cutoff=75, size=(320, 280))

        >>> flt = DFT.create_highpass_filter(x_cutoff=[75], size=(320, 280))

        >>> flt = DFT.create_highpass_filter(x_cutoff=[75, 100, 120],\
                                             size=(320, 280))

        >>> flt = DFT.create_highpass_filter(x_cutoff=75, y_cutoff=35, \
                                             size=(320, 280))

        >>> flt = DFT.create_highpass_filter(x_cutoff=[75], y_cutoff=[35],\
                                             size=(320, 280))

        >>> flt = DFT.create_highpass_filter(x_cutoff=[75, 100, 125], \
                                             y_cutoff=35,\
                                             size=(320, 280))
        >>> # y_cutoff will be [35, 35, 35]

        >>> flt = DFT.create_highpass_filter(x_cutoff=[75, 113, 124],\
                                             y_cutoff=[35, 45, 90],\
                                             size=(320, 280))

        >>> img = Image('lenna')
        >>> flt.apply_filter(img).show()
        """
        if isinstance(x_cutoff, list):
            if len(x_cutoff) != 3 and len(x_cutoff) != 1:
                warnings.warn("x_cutoff list must be of size 3 or 1")
                return None
            if isinstance(y_cutoff, list):
                if len(y_cutoff) != 3 and len(y_cutoff) != 1:
                    warnings.warn("y_cutoff list must be of size 3 or 1")
                    return None
                if len(y_cutoff) == 1:
                    y_cutoff = [y_cutoff[0]]*len(x_cutoff)
            else:
                y_cutoff = [y_cutoff]*len(x_cutoff)
            stacked_filter = DFT()
            for xfreq, yfreq in zip(x_cutoff, y_cutoff):
                stacked_filter = stacked_filter._stack_filters(
                    self.create_highpass_filter(xfreq, yfreq, size))
            image = Image(stacked_filter._numpy)
            ret_value = DFT(numpyarray=stacked_filter._numpy, image=image,
                            x_cutoff_high=x_cutoff, y_cutoff_high=y_cutoff,
                            channels=len(x_cutoff), size=size,
                            type=stacked_filter._type, order=self._order,
                            frequency=stacked_filter._freqpass)
            return ret_value

        lowpass = self.create_lowpass_filter(x_cutoff, y_cutoff, size)
        #w, h = lowpass.size()
        flt = lowpass._numpy
        flt = 255 - flt
        img = Image(flt)
        highpass_filter = DFT(size=size, numpyarray=flt, image=img,
                              type="Highpass", x_cutoff_high=x_cutoff,
                              y_cutoff_high=y_cutoff, frequency="highpass")
        return highpass_filter

    @classmethod
    def create_bandpass_filter(self, x_cutoff_low, x_cutoff_high,
                               y_cutoff_low=None, y_cutoff_high=None,
                               size=(64, 64)):
        """
        **SUMMARY**

        Creates a banf filter of given size and order.

        **PARAMETERS**

        * *x_cutoff_low*    -  int - horizontal lower cut off frequency
                            - list - provide a list of three cut off
                                     frequencies
        * *x_cutoff_high*   -  int - horizontal higher cut off frequency
                            - list - provide a list of three cut off
                                     frequencies
        * *y_cutoff_low*    -  int - vertical lower cut off frequency
                            - list - provide a list of three cut off
                                     frequencies
        * *y_cutoff_high*   -  int - verical higher cut off frequency
                            - list - provide a list of three cut off
                                     frequencies to create a 3 channel filter
        * *size*            - size of the filter (width, height)

        **RETURNS**

        DFT filter.

        **EXAMPLE**

        >>> flt = DFT.create_bandpass_filter(x_cutoff_low=75,\
                                             x_cutoff_high=190, \
                                             size=(320, 280))

        >>> flt = DFT.create_bandpass_filter(x_cutoff_low=[75],\
                                             x_cutoff_high=[190], \
                                             size=(320, 280))

        >>> flt = DFT.create_bandpass_filter(x_cutoff_low=[75, 120, 132],\
                                             x_cutoff_high=[190, 210, 234],\
                                             size=(320, 280))

        >>> flt = DFT.create_bandpass_filter(x_cutoff_low=75, \
                                             x_cutoff_high=190, \
                                             y_cutoff_low=60, \
                                             y_cutoff_high=210, \
                                             size=(320, 280))

        >>> flt = DFT.create_bandpass_filter(x_cutoff_low=[75], \
                                             x_cutoff_high=[190],\
                                             y_cutoff_low=[60], \
                                             y_cutoff_high=[210],\
                                             size=(320, 280))

        >>> flt = DFT.create_bandpass_filter(x_cutoff_low=[75, 120, 132],\
                                             x_cutoff_high=[190, 210, 234],\
                                             y_cutoff_low=[70, 110, 112],\
                                             y_cutoff_high=[180, 220, 220],\
                                             size=(320, 280))

        >>> img = Image('lenna')
        >>> flt.apply_filter(img).show()
        """
        lowpass = self.create_lowpass_filter(x_cutoff_low, y_cutoff_low, size)
        highpass = self.create_highpass_filter(x_cutoff_high, y_cutoff_high,
                                               size)
        lowpassnumpy = lowpass._numpy
        highpassnumpy = highpass._numpy
        bandpassnumpy = lowpassnumpy + highpassnumpy
        bandpassnumpy = clip(bandpassnumpy, 0, 255)
        img = Image(bandpassnumpy)
        bandpass_filter = DFT(size=size, image=img,
                              numpyarray=bandpassnumpy, type="bandpass",
                              x_cutoff_low=x_cutoff_low,
                              y_cutoff_low=y_cutoff_low,
                              x_cutoff_high=x_cutoff_high,
                              y_cutoff_high=y_cutoff_high,
                              frequency="bandpass", channels=lowpass.channels)
        return bandpass_filter

    @classmethod
    def create_notch_filter(self, dia1, dia2=None, cen=None, size=(64, 64),
                            ftype="lowpass"):
        """
        **SUMMARY**

        Creates a disk shaped notch filter of given diameter at given center.

        **PARAMETERS**

        * *dia1*       -  int - diameter of the disk shaped notch
                       - list - provide a list of three diameters to create
                               a 3 channel filter
        * *dia2*       -  int - outer diameter of the disk shaped notch
                                used for bandpass filter
                       - list - provide a list of three diameters to create
                               a 3 channel filter
        * *cen*        - tuple (x, y) center of the disk shaped notch
                         if not provided, it will be at the center of the
                         filter
        * *size*       - size of the filter (width, height)
        * *ftype*:     - lowpass or highpass filter

        **RETURNS**
        DFT notch filter

        **EXAMPLE**

        >>> notch = DFT.create_notch_filter(dia1=200, cen=(200, 200),\
                                            size=(512, 512), type="highpass")
        >>> notch = DFT.create_notch_filter(dia1=200, dia2=300, \
                                            cen=(200, 200), \
                                            size=(512, 512))
        >>> img = Image('lenna')
        >>> notch.apply_filter(img).show()
        """

        if isinstance(dia1, list):
            if len(dia1) != 3 and len(dia1) != 1:
                warnings.warn("diameter list must be of size 1 or 3")
                return None

            if isinstance(dia2, list):
                if len(dia2) != 3 and len(dia2) != 1:
                    warnings.warn("diameter list must be of size 3 or 1")
                    return None
                if len(dia2) == 1:
                    dia2 = [dia2[0]]*len(dia1)
            else:
                dia2 = [dia2]*len(dia1)

            if isinstance(cen, list):
                if len(cen) != 3 and len(cen) != 1:
                    warnings.warn("center list must be of size 3 or 1")
                    return None
                if len(cen) == 1:
                    cen = [cen[0]]*len(dia1)
            else:
                cen = [cen]*len(dia1)

            stacked_filter = DFT()
            for d1, d2, c in zip(dia1, dia2, cen):
                stacked_filter = stacked_filter._stack_filters(
                    self.create_notch_filter(d1, d2, c, size, ftype))
            image = Image(stacked_filter._numpy)
            ret_value = DFT(numpyarray=stacked_filter._numpy, image=image,
                            dia=dia1+dia2, channels=len(dia1), size=size,
                            type=stacked_filter._type,
                            frequency=stacked_filter._freqpass)
            return ret_value

        w, h = size
        if cen is None:
            cen = (w/2, h/2)
        a, b = cen
        y, x = ogrid[-a:w-a, -b:h-b]
        r = dia1/2
        mask = x*x + y*y <= r*r
        flt = ones((w, h))
        flt[mask] = 255
        if ftype == "highpass":
            flt = 255-flt
        if dia2 is not None:
            a, b = cen
            y, x = ogrid[-a:w-a, -b:h-b]
            r = dia2/2
            mask = x*x + y*y <= r*r
            flt1 = ones((w, h))
            flt1[mask] = 255
            flt1 = 255 - flt1
            flt += flt1
            clip(flt, 0, 255)
            ftype = "bandpass"
        img = Image(flt)
        notch_filter = DFT(size=size, numpyarray=flt, image=img, dia=dia1,
                           type="Notch", frequency=ftype)
        return notch_filter

    def apply_filter(self, image, grayscale=False):
        """
        **SUMMARY**

        Apply the DFT filter to given image.

        **PARAMETERS**

        * *image*     - SimpleCV.Image image
        * *grayscale* - if this value is True we perfrom the operation on the
                        DFT of the gray version of the image and the result is
                        gray image. If grayscale is true we perform the
                        operation on each channel and the recombine them to
                        create the result.

        **RETURNS**

        Filtered Image.

        **EXAMPLE**

        >>> notch = DFT.create_notch_filter(dia1=200, cen=(200, 200),\
                                            size=(512, 512), type="highpass")
        >>> img = Image('lenna')
        >>> notch.apply_filter(img).show()
        """

        if self.width == 0 or self.height == 0:
            warnings.warn("Empty Filter. Returning the image.")
            return image
        image_size = image.size
        if grayscale:
            image = image.to_gray()
        flt_img = self._image
        if flt_img.size != image.size:
            flt_img = flt_img.resize(*image_size)
        filtered_image = image.apply_dft_filter(flt_img)
        return filtered_image

    def get_image(self):
        """
        **SUMMARY**

        Get the SimpleCV Image of the filter

        **RETURNS**

        Image of the filter.

        **EXAMPLE**

        >>> notch = DFT.create_notch_filter(dia1=200, cen=(200, 200),\
                                            size=(512, 512), type="highpass")
        >>> notch.get_image().show()
        """
        if isinstance(self._image, type(None)):
            if isinstance(self._numpy, type(None)):
                warnings.warn("Filter doesn't contain any image")
            self._image = Image(self._numpy)
        return self._image

    def get_numpy(self):
        """
        **SUMMARY**

        Get the numpy array of the filter

        **RETURNS**

        numpy array of the filter.

        **EXAMPLE**

        >>> notch = DFT.create_notch_filter(dia1=200, cen=(200, 200),\
                                            size=(512, 512), type="highpass")
        >>> notch.get_numpy()
        """
        if isinstance(self._numpy, type(None)):
            if isinstance(self._image, type(None)):
                warnings.warn("Filter doesn't contain any image")
            self._numpy = self._image.get_numpy()
        return self._numpy

    def get_order(self):
        """
        **SUMMARY**

        Get order of the butterworth filter

        **RETURNS**

        order of the butterworth filter

        **EXAMPLE**

        >>> flt = DFT.create_butterworth_filter(order=4)
        >>> print flt.get_order()
        """
        return self._order

    def size(self):
        """
        **SUMMARY**

        Get size of the filter

        **RETURNS**

        tuple of (width, height)

        **EXAMPLE**

        >>> flt = DFT.create_gaussian_filter(size=(380, 240))
        >>> print flt.size()
        """
        return self.width, self.height

    def get_dia(self):
        """
        **SUMMARY**

        Get diameter of the filter

        **RETURNS**

        diameter of the filter

        **EXAMPLE**

        >>> flt = DFT.create_gaussian_filter(dia=200, size=(380, 240))
        >>> print flt.get_dia()
        """
        return self._dia

    def get_type(self):
        """
        **SUMMARY**

        Get type of the filter

        **RETURNS**

        type of the filter

        **EXAMPLE**

        >>> flt = DFT.create_gaussian_filter(dia=200, size=(380, 240))
        >>> print flt.get_type() # Gaussian
        """
        return self._type

    def stack_filters(self, flt1, flt2):
        """
        **SUMMARY**

        Stack three signle channel filters of the same size to create
        a 3 channel filter.

        **PARAMETERS**

        * *flt1* - second filter to be stacked
        * *flt2* - thrid filter to be stacked

        **RETURNS**

        DFT filter

        **EXAMPLE**

        >>> flt1 = DFT.create_gaussian_filter(dia=200, size=(380, 240))
        >>> flt2 = DFT.create_gaussian_filter(dia=100, size=(380, 240))
        >>> flt3 = DFT.create_gaussian_filter(dia=70, size=(380, 240))
        >>> flt = flt1.stack_filters(flt2, flt3) # 3 channel filter
        """
        if not(self.channels == 1 and flt1.channels == 1 and
               flt2.channels == 1):
            warnings.warn("Filters must have only 1 channel")
            return None
        if not (self.size() == flt1.size() and self.size() == flt2.size()):
            warnings.warn("All the filters must be of same size")
            return None
        numpyflt = self._numpy
        numpyflt1 = flt1._numpy
        numpyflt2 = flt2._numpy
        flt = dstack((numpyflt, numpyflt1, numpyflt2))
        img = Image(flt)
        stackedfilter = DFT(size=self.size(), numpyarray=flt,
                            image=img, channels=3)
        return stackedfilter

    def _stack_filters(self, flt1):
        """
        **SUMMARY**

        stack two filters of same size. channels don't matter.

        **PARAMETERS**

        * *flt1* - second filter to be stacked

        **RETURNS**

        DFT filter

        """
        if isinstance(self._numpy, type(None)):
            return flt1
        if not self.size() == flt1.size():
            warnings.warn("All the filters must be of same size")
            return None
        numpyflt = self._numpy
        numpyflt1 = flt1._numpy
        flt = dstack((numpyflt, numpyflt1))
        stacked_filter = DFT(size=self.size(), numpyarray=flt,
                             channels=self.channels+flt1.channels,
                             type=self._type, frequency=self._freqpass)
        return stacked_filter
