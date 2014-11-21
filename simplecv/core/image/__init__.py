import functools

import cv2
import numpy as np

from simplecv.base import logger, ScvException, PicklabeNdarray
from simplecv.core.pluginsystem import plugin_method


image_method = functools.partial(plugin_method, 'Image', False)
static_image_method = functools.partial(plugin_method, 'Image', True)


def cached_method(func):
    """ Decorator for image methods. Saves the method result and arguments
        to the cache and returns cached result if the arguments are not
        changed.
    """
    def wrapper(*args, **kwargs):
        img = args[0]  # first arg is image
        img_args = args[1:]
        cache_item = img._cache.get(func.__name__)
        if cache_item \
                and cache_item['args'] == img_args \
                and cache_item['kwargs'] == kwargs:
            return cache_item['result']

        result = func(*args, **kwargs)
        img._cache[func.__name__] = {
            'result': result,
            'args': img_args,
            'kwargs': kwargs
        }
        return result
    wrapper.__name__ = func.__name__
    return wrapper


class ColorSpace(int):

    # Available color spaces
    BGR = 1
    GRAY = 2
    RGB = 3
    HLS = 4
    HSV = 5
    XYZ = 6
    YCR_CB = 7
    BGRA = 8
    RGBA = 9

    color_space_to_string = {
        BGR: 'BGR',
        GRAY: 'GRAY',
        RGB: 'RGB',
        HLS: 'HLS',
        HSV: 'HSV',
        XYZ: 'XYZ',
        YCR_CB: 'YCR_CB',
        BGRA: 'BGRA',
        RGBA: 'RGBA',
    }

    color_spaces = color_space_to_string.keys()

    def __str__(self):
        return self.to_string()

    def __repr__(self):
        return 'ColorSpace({0})'.format(self.to_string())

    def to_string(self):
        return self.color_space_to_string.get(self, 'UNKNOWN')


class Image(PicklabeNdarray):
    """
    Core Image class

    Responsible to:
    * Store image data as numpy.ndarray (cv2 format)
    * Store color space information
    * Conversion between color spaces
    """

    __survive_pickling__ = ['_color_space']

    BGR = ColorSpace(ColorSpace.BGR)
    BGRA = ColorSpace(ColorSpace.BGRA)
    GRAY = ColorSpace(ColorSpace.GRAY)
    RGB = ColorSpace(ColorSpace.RGB)
    RGBA = ColorSpace(ColorSpace.RGBA)
    HLS = ColorSpace(ColorSpace.HLS)
    HSV = ColorSpace(ColorSpace.HSV)
    XYZ = ColorSpace(ColorSpace.XYZ)
    YCR_CB = ColorSpace(ColorSpace.YCR_CB)

    def __new__(cls, array, color_space=None):
        if not isinstance(array, np.ndarray):
            raise ValueError('array is not a numpy.ndarray')
        self = np.asarray(array).view(cls)

        if len(self.shape) == 2:
            self._color_space = Image.GRAY
        else:
            if color_space is None:
                if self.shape[2] == 4:
                    self._color_space = Image.BGRA
                else:
                    self._color_space = Image.BGR
            elif color_space not in ColorSpace.color_spaces:
                raise ValueError('Unknown color space')
            else:
                self._color_space = ColorSpace(color_space)
        return self

    def __array_finalize__(self, obj):
        if len(self.shape) == 2:
            self._color_space = Image.GRAY

        if obj is not None:
            self._color_space = getattr(obj, 'color_space', Image.BGR)
        else:
            self._color_space = Image.BGR
        self._cache = {}

    @property
    def color_space(self):
        return self._color_space

    @property
    def height(self):
        return self.shape[0]

    @property
    def width(self):
        return self.shape[1]

    @property
    def size_tuple(self):
        return self.width, self.height

    def __sub__(self, other):
        if isinstance(other, Image):
            if self.shape != other.shape:
                raise ScvException("Both images should have same dimensions.")
            array = cv2.subtract(self, other)
            return self.__class__(array=array, color_space=self._color_space)
        else:
            array = super(Image, self).__sub__(other).astype(self.dtype)
            return self.__class__(array=array, color_space=self._color_space)

    def __add__(self, other):
        if isinstance(other, Image):
            if self.shape != other.shape:
                raise ScvException("Both images should have same dimensions.")
            array = cv2.add(self, other)
            return self.__class__(array=array, color_space=self._color_space)
        else:
            array = super(Image, self).__add__(other)
            return self.__class__(array=array, color_space=self._color_space)

    def __div__(self, other):
        if isinstance(other, Image):
            if self.shape != other.shape:
                raise ScvException("Both images should have same dimensions.")
            array = cv2.divide(self, other)
            return self.__class__(array=array, color_space=self._color_space)
        else:
            array = super(Image, self).__div__(other).astype(self.dtype)
            return self.__class__(array=array, color_space=self._color_space)

    def __mul__(self, other):
        if isinstance(other, Image):
            if self.shape != other.shape:
                raise ScvException("Both images should have same dimensions.")
            array = cv2.multiply(self, other)
            return self.__class__(array=array, color_space=self._color_space)
        else:
            array = super(Image, self).__mul__(other).astype(self.dtype)
            return self.__class__(array=array, color_space=self._color_space)

    def __pow__(self, power):
        if isinstance(power, int):
            array = cv2.pow(self, power)
            return self.__class__(array=array, color_space=self._color_space)
        else:
            array = super(Image, self).__pow__(power).astype(self.dtype)
            return self.__class__(array=array, color_space=self._color_space)

    def __invert__(self):
        return 255 - self

    def get_empty(self, channels=3):
        """
        Create a new, empty SimpleCV Image with the specified number of channels
        (default 3). This method basically Images an empty copy of the image.

        :param channels: The number of channels in the returned OpenCV image.
        :type channels: int

        :returns: instance of Image.
        """
        shape = [self.height, self.width]
        if channels > 1:
            shape.append(channels)
        return self.__class__(array=np.zeros(shape, dtype=self.dtype),
                              color_space=self._color_space)

    @staticmethod
    def convert(ndarray, from_color_space, to_color_space):
        """
        Converts a numpy array from one color space to another

        :param ndarray: array to convert
        :type ndarray: instance of numpy.ndarray.
        :param from_color_space: color space to convert from.
        :type from_color_space: int.
        :param from_color_space: color space to convert to.
        :type from_color_space: int.
        :returns: instance of numpy.ndarray.
        """
        if from_color_space == to_color_space:
            return ndarray.copy()

        if from_color_space not in ColorSpace.color_spaces \
                or to_color_space not in ColorSpace.color_spaces:
            raise AttributeError('Unknown color space')

        converter_str = 'COLOR_{}2{}'.format(from_color_space,
                                             to_color_space)
        try:
            converter = getattr(cv2, converter_str)
        except AttributeError:
            # convert to BGR first
            converter_bgr_str = 'COLOR_{}2BGR'.format(from_color_space)
            converter_str = 'COLOR_BGR2{}'.format(to_color_space)

            converter_bgr = getattr(cv2, converter_bgr_str)
            converter = getattr(cv2, converter_str)

            new_ndarray = cv2.cvtColor(ndarray, converter_bgr)
            return cv2.cvtColor(new_ndarray, code=converter)
        else:
            return cv2.cvtColor(ndarray, code=converter)

    def to_color_space(self, color_space):
        """
        This method converts the image to the given color space.

        Available color spaces:
        Image.BGR
        Image.GRAY
        Image.RGB
        Image.HLS
        Image.HSV
        Image.XYZ
        Image.YCR_CB
        """
        array = Image.convert(self, self._color_space, color_space)
        return self.__class__(array=array, color_space=color_space)

    def to_bgr(self):
        return self.to_color_space(Image.BGR)

    def to_bgra(self):
        return self.to_color_space(Image.BGRA)

    def to_rgb(self):
        return self.to_color_space(Image.RGB)

    def to_rgba(self):
        return self.to_color_space(Image.RGBA)

    def to_hsv(self):
        return self.to_color_space(Image.HSV)

    def to_hls(self):
        return self.to_color_space(Image.HLS)

    def to_ycrcb(self):
        return self.to_color_space(Image.YCR_CB)

    def to_xyz(self):
        return self.to_color_space(Image.XYZ)

    def to_gray(self):
        return self.to_color_space(Image.GRAY)

    def is_color_space(self, color_space):
        """
        Returns True if this image uses the given color space.

        :returns: bool
        """
        return self._color_space == color_space

    def is_bgr(self):
        return self.is_color_space(Image.BGR)

    def is_bgra(self):
        return self.is_color_space(Image.BGRA)

    def is_rgb(self):
        return self.is_color_space(Image.RGB)

    def is_rgba(self):
        return self.is_color_space(Image.RGBA)

    def is_hsv(self):
        return self.is_color_space(Image.HSV)

    def is_hls(self):
        return self.is_color_space(Image.HLS)

    def is_ycrcb(self):
        return self.is_color_space(Image.YCR_CB)

    def is_xyz(self):
        return self.is_color_space(Image.XYZ)

    def is_gray(self):
        return self.is_color_space(Image.GRAY)

    def to_string(self):
        """
        Returns the image as a string, useful for moving data around.

        :returns: string
        """
        return self.tostring()

    def clear(self):
        """
        This is a slightly unsafe method that clears out the entire image state
        it is usually used in conjunction with the drawing blobs to fill in
        draw a single large blob in the image.

        .. Warning:
          Do not use this method unless you have a particularly compelling
          reason.
        """
        self.fill(0)
        self.clear_cache()

    def clear_cache(self):
        """ Removes all cached results from methods that was decorated with
            cached_method
        """
        self._cache.clear()

    def is_empty(self):
        """
        Checks if the image is empty by checking its width and height.

        :returns: True if the image's size is (0, 0),
                  False for any other size.
        """
        return self.size_tuple == (0, 0)

    def get_area(self):
        '''
        Returns the area of the Image.
        '''
        return self.width * self.height

    def split_channels(self):
        """
        **SUMMARY**

        Split the channels of an image.

        **RETURNS**

        A tuple of 3 image objects.

        **EXAMPLE**

        >>> img = Image("lenna")
        >>> data = img.split_channels()
        >>> for d in data:
        >>>    d.show()
        >>>    time.sleep(1)

        **SEE ALSO**

        :py:meth:`merge_channels`
        """
        chanel_0 = self[:, :, 0].copy()
        chanel_1 = self[:, :, 1].copy()
        chanel_2 = self[:, :, 2].copy()

        return (self.__class__(array=chanel_0, color_space=Image.GRAY),
                self.__class__(array=chanel_1, color_space=Image.GRAY),
                self.__class__(array=chanel_2, color_space=Image.GRAY))

    def merge_channels(self, c1=None, c2=None, c3=None,
                       color_space=None):
        """
        **SUMMARY**

        Merge channels is the oposite of split_channels. The image takes one
        image for each of the R,G,B channels and then recombines them into a
        single image. Optionally any of these channels can be None.

        **PARAMETERS**

        * *r* - The r or last channel  of the result SimpleCV Image.
        * *g* - The g or center channel of the result SimpleCV Image.
        * *b* - The b or first channel of the result SimpleCV Image.


        **RETURNS**

        A SimpleCV Image.

        **EXAMPLE**

        >>> img = Image("lenna")
        >>> r, g, b = img.split_channels()
        >>> r = r.binarize()
        >>> g = g.binarize()
        >>> b = b.binarize()
        >>> result = img.merge_channels(r, g, b)
        >>> result.show()


        **SEE ALSO**
        :py:meth:`split_channels`

        """
        if color_space is None:
            color_space = ColorSpace.RGB
        if c1 is None and c2 is None and c3 is None:
            logger.warning("Image.merge_channels - we need at least "
                           "one valid channel")
            return None

        if c1 is None:
            c1 = self.get_empty(1)
        if c2 is None:
            c2 = self.get_empty(1)
        if c3 is None:
            c3 = self.get_empty(1)

        array = np.dstack((c1, c2, c3))
        return self.__class__(array=array, color_space=color_space)

    @staticmethod
    def roi_to_slice(roi):
        x, y, w, h = roi
        return slice(y, y + h), slice(x, x + w)

    @staticmethod
    def rect_overlap_rois(top, bottom, pos):
        """
        top is a rectangle (w,h)
        bottom is a rectangle (w,h)
        pos is the top left corner of the top rectangle with respect to the
        bottom rectangle's top left corner method returns none if the two
        rectangles do not overlap. Otherwise returns the top rectangle's
        ROI (x,y,w,h) and the bottom rectangle's ROI (x,y,w,h)
        """
        # the position of the top rect coordinates
        # give bottom top right = (0,0)
        tr = (pos[0] + top[0], pos[1])
        tl = pos
        br = (pos[0] + top[0], pos[1] + top[1])
        bl = (pos[0], pos[1] + top[1])

        # do an overlap test to weed out corner cases and errors
        def in_bounds((w, h), (x, y)):
            ret_val = True
            if x < 0 or y < 0 or x > w or y > h:
                ret_val = False
            return ret_val

        trc = in_bounds(bottom, tr)
        tlc = in_bounds(bottom, tl)
        brc = in_bounds(bottom, br)
        blc = in_bounds(bottom, bl)
        if not trc and not tlc and not brc and not blc:  # no overlap
            return None, None
        # easy case top is fully inside bottom
        elif trc and tlc and brc and blc:
            t_ret = (0, 0, top[0], top[1])
            b_ret = (pos[0], pos[1], top[0], top[1])
            return t_ret, b_ret
        # let's figure out where the top rectangle sits on the bottom
        # we clamp the corners of the top rectangle to live inside
        # the bottom rectangle and from that get the x,y,w,h
        tl = (np.clip(tl[0], 0, bottom[0]), np.clip(tl[1], 0, bottom[1]))
        br = (np.clip(br[0], 0, bottom[0]), np.clip(br[1], 0, bottom[1]))

        bx = tl[0]
        by = tl[1]
        bw = abs(tl[0] - br[0])
        bh = abs(tl[1] - br[1])
        # now let's figure where the bottom rectangle is in the top rectangle
        # we do the same thing with different coordinates
        pos = (-1 * pos[0], -1 * pos[1])
        #recalculate the bottoms's corners with respect to the top.
        tr = (pos[0] + bottom[0], pos[1])
        tl = pos
        br = (pos[0] + bottom[0], pos[1] + bottom[1])
        bl = (pos[0], pos[1] + bottom[1])
        tl = (np.clip(tl[0], 0, top[0]), np.clip(tl[1], 0, top[1]))
        br = (np.clip(br[0], 0, top[0]), np.clip(br[1], 0, top[1]))
        tx = tl[0]
        ty = tl[1]
        tw = abs(br[0] - tl[0])
        th = abs(br[1] - tl[1])
        return (tx, ty, tw, th), (bx, by, bw, bh)
