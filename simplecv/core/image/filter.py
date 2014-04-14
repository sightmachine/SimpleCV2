import math

import cv2
import numpy as np
import scipy.ndimage as ndimage
import scipy.spatial.distance as spsd
import scipy.stats.stats as sss  # for auto white balance

from simplecv.base import is_tuple, is_number, logger
from simplecv.color import Color, ColorCurve
from simplecv.core.image import image_method
from simplecv.factory import Factory


@image_method
def equalize(img):
    """
    **SUMMARY**

    Perform a histogram equalization on the image.

    **RETURNS**

    Returns a grayscale simplecv Image.

    **EXAMPLE**

    >>> img = Image("lenna")
    >>> img = equalize(img)

    """
    equalized_array = cv2.equalizeHist(img.get_gray_ndarray())
    return Factory.Image(equalized_array)


@image_method
def smooth(img, algorithm_name='gaussian', aperture=(3, 3), sigma=0,
           spatial_sigma=0, grayscale=False):
    """
    **SUMMARY**

    Smooth the image, by default with the Gaussian blur.  If desired,
    additional algorithms and apertures can be specified.  Optional
    parameters are passed directly to OpenCV's functions.

    If grayscale is true the smoothing operation is only performed on a
    single channel otherwise the operation is performed on each channel
    of the image.

    for OpenCV versions >= 2.3.0 it is advisible to take a look at
           - :py:meth:`bilateral_filter`
           - :py:meth:`median_filter`
           - :py:meth:`blur`
           - :py:meth:`gaussian_blur`

    **PARAMETERS**

    * *algorithm_name* - valid options are 'blur' or gaussian, 'bilateral',
     and 'median'.

      * `Median Filter <http://en.wikipedia.org/wiki/Median_filter>`_

      * `Gaussian Blur <http://en.wikipedia.org/wiki/Gaussian_blur>`_

      * `Bilateral Filter <http://en.wikipedia.org/wiki/Bilateral_filter>`_

    * *aperture* - A tuple for the aperture of the gaussian blur as an
                   (x,y) tuple.

    .. Warning::
      These must be odd numbers.

    * *sigma* -

    * *spatial_sigma* -

    * *grayscale* - Return just the grayscale image.



    **RETURNS**

    The smoothed image.

    **EXAMPLE**

    >>> img = Image("Lenna")
    >>> img2 = smooth(img)
    >>> img3 = smooth(img, 'median')

    **SEE ALSO**

    :py:meth:`bilateral_filter`
    :py:meth:`median_filter`
    :py:meth:`blur`

    """
    if is_tuple(aperture):
        win_x, win_y = aperture
        if win_x <= 0 or win_y <= 0 or win_x % 2 == 0 or win_y % 2 == 0:
            logger.warning("The aperture (x,y) must be odd number and "
                           "greater than 0.")
            return None
    else:
        raise ValueError("Please provide a tuple to aperture, "
                         "got: %s" % type(aperture))

    window = (win_x, win_y)
    if algorithm_name == "blur":
        return img.blur(window=window, grayscale=grayscale)
    elif algorithm_name == "bilateral":
        return img.bilateral_filter(diameter=win_x, grayscale=grayscale)
    elif algorithm_name == "median":
        return img.median_filter(window=window, grayscale=grayscale)
    else:
        return img.gaussian_blur(window=window, sigma_x=sigma,
                                 sigma_y=spatial_sigma, grayscale=grayscale)


@image_method
def median_filter(img, window=None, grayscale=False):
    """
    **SUMMARY**

    Smooths the image, with the median filter. Performs a median filtering
    operation to denoise/despeckle the image.
    The optional parameter is the window size.
    see : http://en.wikipedia.org/wiki/Median_filter

    **Parameters**

    * *window* - should be in the form a tuple (win_x,win_y). Where win_x
     should be equal to win_y. By default it is set to 3x3,
     i.e window = (3x3).

    **Note**

    win_x and win_y should be greater than zero, a odd number and equal.

    For OpenCV versions >= 2.3.0
    cv2.medianBlur function is called.

    """
    if is_tuple(window):
        win_x, win_y = window
        if win_x >= 0 and win_y >= 0 and win_x % 2 == 1 and win_y % 2 == 1:
            if win_x != win_y:
                win_x = win_y
        else:
            logger.warning("The aperture (win_x, win_y) must be odd "
                           "number and greater than 0.")
            return None

    elif is_number(window):
        win_x = window
    else:
        win_x = 3  # set the default aperture window size (3x3)

    if grayscale:
        img_medianblur = cv2.medianBlur(img.get_gray_ndarray(), ksize=win_x)
        return Factory.Image(img_medianblur)
    else:
        img_medianblur = cv2.medianBlur(img.get_ndarray(), ksize=win_x)
        return Factory.Image(img_medianblur, color_space=img.color_space)


@image_method
def bilateral_filter(img, diameter=5, sigma_color=10, sigma_space=10,
                     grayscale=False):
    """
    **SUMMARY**

    Smooths the image, using bilateral filtering. Potential of bilateral
    filtering is for the removal of texture.
    The optional parameter are diameter, sigma_color, sigma_space.

    Bilateral Filter
    see : http://en.wikipedia.org/wiki/Bilateral_filter
    see : http://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/
    MANDUCHI1/Bilateral_Filtering.html

    **Parameters**

    * *diameter* - A tuple for the window of the form (diameter,diameter).
                   By default window = (3x3).
                   (for OpenCV versions <= 2.3.0)
                 - Diameter of each pixel neighborhood that is used during
                   filtering. ( for OpenCV versions >= 2.3.0)


    * *sigma_color* - Filter the specified value in the color space. A
     larger value of the parameter means that farther colors within the
     pixel neighborhood (see sigma_space ) will be mixed together,
     resulting in larger areas of semi-equal color.

    * *sigma_space* - Filter the specified value in the coordinate space.
     A larger value of the parameter means that farther pixels will
     influence each other as long as their colors are close enough

    **NOTE**
    For OpenCV versions <= 2.3.0
    -- this acts as Convience function derived from the :py:meth:`smooth`
       method.
    -- where aperture(window) is (diameter,diameter)
    -- sigma_color and sigmanSpace become obsolete

    For OpenCV versions higher than 2.3.0. i.e >= 2.3.0
    -- cv2.bilateralFilter function is called
    -- If the sigma_color and sigma_space values are small (< 10),
       the filter will not have much effect, whereas if they are large
       (> 150), they will have a very strong effect, making the image look
       'cartoonish'
    -- It is recommended to use diamter=5 for real time applications, and
       perhaps diameter=9 for offile applications that needs heavy noise
       filtering.
    """
    if is_tuple(diameter):
        win_x, win_y = diameter
        if win_x >= 0 and win_y >= 0 and win_x % 2 == 1 and win_y % 2 == 1:
            if win_x != win_y:
                diameter = (win_x, win_y)
        else:
            logger.warning("The aperture (win_x, win_y) must be odd number "
                           "and greater than 0.")
            return None

    elif is_number(diameter):
        pass
    else:
        win_x = 3  # set the default aperture window size (3x3)
        diameter = (win_x, win_x)

    if grayscale:
        img_bilateral = cv2.bilateralFilter(img.get_gray_ndarray(),
                                            d=diameter, sigmaColor=sigma_color,
                                            sigmaSpace=sigma_space)
        return Factory.Image(img_bilateral)
    else:
        img_bilateral = cv2.bilateralFilter(img.get_ndarray(), d=diameter,
                                            sigmaColor=sigma_color,
                                            sigmaSpace=sigma_space)
        return Factory.Image(img_bilateral, color_space=img.color_space)


@image_method
def blur(img, window=None, grayscale=False):
    """
    **SUMMARY**

    Smoothes an image using the normalized box filter.
    The optional parameter is window.

    see : http://en.wikipedia.org/wiki/Blur

    **Parameters**

    * *window* - should be in the form a tuple (win_x,win_y).
               - By default it is set to 3x3, i.e window = (3x3).

    **NOTE**
    For OpenCV versions <= 2.3.0
    -- this acts as Convience function derived from the :py:meth:`smooth`
       method.

    For OpenCV versions higher than 2.3.0. i.e >= 2.3.0
    -- cv2.blur function is called
    """
    if is_tuple(window):
        win_x, win_y = window
        if win_x <= 0 or win_y <= 0:
            logger.warning("win_x and win_y should be greater than 0.")
            return None
    elif is_number(window):
        window = (window, window)
    else:
        window = (3, 3)

    if grayscale:
        img_blur = cv2.blur(img.get_gray_ndarray(), ksize=window)
        return Factory.Image(img_blur)
    else:
        img_blur = cv2.blur(img.get_ndarray(), ksize=window)
        return Factory.Image(img_blur, color_space=img.color_space)


@image_method
def gaussian_blur(img, window=None, sigma_x=0, sigma_y=0,
                  grayscale=False):
    """
    **SUMMARY**

    Smoothes an image, typically used to reduce image noise and reduce
    detail.
    The optional parameter is window.

    see : http://en.wikipedia.org/wiki/Gaussian_blur

    **Parameters**

    * *window* - should be in the form a tuple (win_x,win_y). Where win_x
                 and win_y should be positive and odd.
               - By default it is set to 3x3, i.e window = (3x3).

    * *sigma_x* - Gaussian kernel standard deviation in X direction.

    * *sigma_y* - Gaussian kernel standard deviation in Y direction.

    * *grayscale* - If true, the effect is applied on grayscale images.

    **NOTE**
    For OpenCV versions <= 2.3.0
    -- this acts as Convience function derived from the :py:meth:`smooth`
       method.

    For OpenCV versions higher than 2.3.0. i.e >= 2.3.0
    -- cv2.GaussianBlur function is called
    """
    if is_tuple(window):
        win_x, win_y = window
        if win_x >= 0 and win_y >= 0 and win_x % 2 == 1 and win_y % 2 == 1:
            pass
        else:
            logger.warning("The aperture (win_x, win_y) must be odd number "
                           "and greater than 0.")
            return None

    elif is_number(window):
        window = (window, window)
    else:
        window = (3, 3)  # set the default aperture window size (3x3)
    if grayscale:
        image_gauss = cv2.GaussianBlur(img.get_gray_ndarray(), window,
                                       sigma_x, None, sigma_y)
        return Factory.Image(image_gauss)
    else:
        image_gauss = cv2.GaussianBlur(img.get_ndarray(), window, sigma_x,
                                       None, sigma_y)
        return Factory.Image(image_gauss, color_space=img.color_space)


@image_method
def invert(img):
    """
    **SUMMARY**

    Invert (negative) the image note that this can also be done with the
    unary minus (-) operator. For binary image this turns black into white
    and white into black (i.e. white is the new black).

    **RETURNS**

    The opposite of the current image.

    **EXAMPLE**

    >>> img = Image("polar_bear_in_the_snow.png")
    >>> invert(img).save("black_bear_at_night.png")

    **SEE ALSO**

    :py:meth:`binarize`

    """
    return -img


@image_method
def stretch(img, thresh_low=0, thresh_high=255):
    """
    **SUMMARY**

    The stretch filter works on a greyscale image, if the image
    is color, it returns a greyscale image.  The filter works by
    taking in a lower and upper threshold.  Anything below the lower
    threshold is pushed to black (0) and anything above the upper
    threshold is pushed to white (255)

    **PARAMETERS**

    * *thresh_low* - The lower threshold for the stretch operation.
      This should be a value between 0 and 255.

    * *thresh_high* - The upper threshold for the stretch operation.
      This should be a value between 0 and 255.

    **RETURNS**

    A gray scale version of the image with the appropriate histogram
    stretching.


    **EXAMPLE**

    >>> img = Image("orson_welles.jpg")
    >>> img2 = stretch(img, 56.200)
    >>> img2.show()

    **NOTES**

    TODO - make this work on RGB images with thresholds for each channel.

    **SEE ALSO**

    :py:meth:`binarize`
    :py:meth:`equalize`

    """
    threshold, array = cv2.threshold(img.get_gray_ndarray(), thresh=thresh_low,
                                     maxval=255, type=cv2.THRESH_TOZERO)
    array = cv2.bitwise_not(array)
    threshold, array = cv2.threshold(array, thresh=255 - thresh_high,
                                     maxval=255, type=cv2.THRESH_TOZERO)
    array = cv2.bitwise_not(array)
    return Factory.Image(array)


@image_method
def gamma_correct(img, gamma=1):

    """
    **DESCRIPTION**

    Transforms an image according to Gamma Correction also known as
    Power Law Transform.

    **PARAMETERS**

    * *gamma* - A non-negative real number.

    **RETURNS**

    A Gamma corrected image.

    **EXAMPLE**

    >>> img = Image('simplecv')
    >>> img.show()
    >>> gamma_correct(img, 1.5).show()
    >>> gamma_correct(img, 0.7).show()

    """
    if gamma < 0:
        return "Gamma should be a non-negative real number"
    scale = 255.0
    dst = (((1.0 / scale) * img.get_ndarray()) ** gamma) * scale
    return Factory.Image(dst.astype(img.dtype), color_space=img.color_space)


@image_method
def binarize(img, thresh=None, maxv=255, blocksize=0, p=5):
    """
    **SUMMARY**

    Do a binary threshold the image, changing all values below thresh to
    maxv and all above to black.  If a color tuple is provided, each color
    channel is thresholded separately.


    If threshold is -1 (default), an adaptive method (OTSU's method) is
    used.
    If then a blocksize is specified, a moving average over each region of
    block*block pixels a threshold is applied where threshold =
    local_mean - p.

    **PARAMETERS**

    * *thresh* - the threshold as an integer or an (r,g,b) tuple , where
      pixels below (darker) than thresh are set to to max value,
      and all values above this value are set to black. If this parameter
      is -1 we use Otsu's method.

    * *maxv* - The maximum value for pixels below the threshold. Ordinarily
     this should be 255 (white)

    * *blocksize* - the size of the block used in the adaptive binarize
      operation.

    .. Warning::
      This parameter must be an odd number.

    * *p* - The difference from the local mean to use for thresholding
     in Otsu's method.

    **RETURNS**

    A binary (two colors, usually black and white) SimpleCV image. This
    works great for the find_blobs family of functions.

    **EXAMPLE**

    Example of a vanila threshold versus an adaptive threshold:

    >>> img = Image("orson_welles.jpg")
    >>> b1 = binarize(img, 128)
    >>> b2 = binarize(img, blocksize=11, p=7)
    >>> b3 = b1.side_by_side(b2)
    >>> b3.show()


    **NOTES**

    `Otsu's Method Description<http://en.wikipedia.org/wiki/Otsu's_method>`

    **SEE ALSO**

    :py:meth:`threshold`
    :py:meth:`find_blobs`
    :py:meth:`invert`
    :py:meth:`dilate`
    :py:meth:`erode`

    """
    if is_tuple(thresh):
        b = img.get_ndarray()[:, :, 0].copy()
        g = img.get_ndarray()[:, :, 1].copy()
        r = img.get_ndarray()[:, :, 2].copy()

        r = cv2.threshold(r, thresh=thresh[0], maxval=maxv,
                          type=cv2.THRESH_BINARY_INV)[1]
        g = cv2.threshold(g, thresh=thresh[1], maxval=maxv,
                          type=cv2.THRESH_BINARY_INV)[1]
        b = cv2.threshold(b, thresh=thresh[2], maxval=maxv,
                          type=cv2.THRESH_BINARY_INV)[1]
        array = r + g + b
        return Factory.Image(array)

    elif thresh is None:
        if blocksize:
            array = cv2.adaptiveThreshold(
                img.get_gray_ndarray(), maxValue=maxv,
                adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                thresholdType=cv2.THRESH_BINARY_INV,
                blockSize=blocksize, C=p)
        else:
            array = cv2.threshold(
                img.get_gray_ndarray(), -1, float(maxv),
                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        return Factory.Image(array)
    else:
        # desaturate the image, and apply the new threshold
        array = cv2.threshold(img.get_gray_ndarray(), thresh=thresh,
                              maxval=maxv, type=cv2.THRESH_BINARY_INV)[1]
        return Factory.Image(array)


@image_method
def get_skintone_mask(img, dilate_iter=0):
    """
    **SUMMARY**

    Find Skintone mask will look for continuous
    regions of Skintone in a color image and return a binary mask where the
    white pixels denote Skintone region.

    **PARAMETERS**

    * *dilate_iter* - the number of times to run the dilation operation.


    **RETURNS**

    Returns a binary mask.

    **EXAMPLE**

    >>> img = Image("lenna")
    >>> mask = img.findSkintoneMask()
    >>> mask.show()

    """
    if img.is_ycrcb():
        ycrcb = img.get_ndarray()
    else:
        ycrcb = img.to_ycrcb().get_ndarray()

    y = np.zeros((256, 1), dtype=np.uint8)
    y[5:] = 255
    cr = np.zeros((256, 1), dtype=np.uint8)
    cr[140:180] = 255
    cb = np.zeros((256, 1), dtype=np.uint8)
    cb[77:135] = 255

    y_array = ycrcb[:, :, 0]
    cr_array = ycrcb[:, :, 1]
    cb_array = ycrcb[:, :, 2]

    y_array = cv2.LUT(y_array, lut=y)
    cr_array = cv2.LUT(cr_array, lut=cr)
    cb_array = cv2.LUT(cb_array, lut=cb)

    array = np.dstack((y_array, cr_array, cb_array))

    mask = Factory.Image(array, color_space=Factory.Image.YCR_CB)
    mask = mask.binarize(thresh=(128, 128, 128))
    mask = mask.to_rgb().binarize()
    return mask.dilate(iterations=dilate_iter)


@image_method
def apply_hls_curve(img, hcurve, lcurve, scurve):
    """
    **SUMMARY**

    Apply a color correction curve in HSL space. This method can be used
    to change values for each channel. The curves are
    :py:class:`ColorCurve` class objects.

    **PARAMETERS**

    * *hcurve* - the hue ColorCurve object.
    * *lcurve* - the lightnes / value ColorCurve object.
    * *scurve* - the saturation ColorCurve object

    **RETURNS**

    A SimpleCV Image

    **EXAMPLE**

    >>> img = Image("lenna")
    >>> hc = ColorCurve([[0,0], [100, 120], [180, 230], [255, 255]])
    >>> lc = ColorCurve([[0,0], [90, 120], [180, 230], [255, 255]])
    >>> sc = ColorCurve([[0,0], [70, 110], [180, 230], [240, 255]])
    >>> img2 = img.apply_hls_curve(hc,lc,sc)

    **SEE ALSO**

    :py:class:`ColorCurve`
    :py:meth:`apply_rgb_curve`
    """
    #TODO CHECK ROI
    #TODO CHECK CURVE SIZE
    #TODO CHECK CURVE SIZE

    # Move to HLS space
    array = img.to_hls().get_ndarray()

    # now apply the color curve correction
    array[:, :, 0] = np.take(hcurve.curve, array[:, :, 0])
    array[:, :, 1] = np.take(lcurve.curve, array[:, :, 1])
    array[:, :, 2] = np.take(scurve.curve, array[:, :, 2])

    # Move back to original color space
    array = Factory.Image.convert(array, Factory.Image.HLS, img.color_space)
    return Factory.Image(array, color_space=img.color_space)


@image_method
def apply_rgb_curve(img, rcurve, gcurve, bcurve):
    """
    **SUMMARY**

    Apply a color correction curve in RGB space. This method can be used
    to change values for each channel. The curves are
    :py:class:`ColorCurve` class objects.

    **PARAMETERS**

    * *rcurve* - the red ColorCurve object, or appropriately formatted
     list
    * *gcurve* - the green ColorCurve object, or appropriately formatted
     list
    * *bcurve* - the blue ColorCurve object, or appropriately formatted
     list

    **RETURNS**

    A SimpleCV Image

    **EXAMPLE**

    >>> img = Image("lenna")
    >>> rc = ColorCurve([[0,0], [100, 120], [180, 230], [255, 255]])
    >>> gc = ColorCurve([[0,0], [90, 120], [180, 230], [255, 255]])
    >>> bc = ColorCurve([[0,0], [70, 110], [180, 230], [240, 255]])
    >>> img2 = img.apply_rgb_curve(rc,gc,bc)

    **SEE ALSO**

    :py:class:`ColorCurve`
    :py:meth:`apply_hls_curve`

    """
    if isinstance(bcurve, list):
        bcurve = ColorCurve(bcurve)
    if isinstance(gcurve, list):
        gcurve = ColorCurve(gcurve)
    if isinstance(rcurve, list):
        rcurve = ColorCurve(rcurve)

    array = img.get_ndarray().copy()
    array[:, :, 0] = np.take(bcurve.curve, array[:, :, 0])
    array[:, :, 1] = np.take(gcurve.curve, array[:, :, 1])
    array[:, :, 2] = np.take(rcurve.curve, array[:, :, 2])
    return Factory.Image(array, color_space=img.color_space)


@image_method
def apply_intensity_curve(img, curve):
    """
    **SUMMARY**

    Intensity applied to all three color channels

    **PARAMETERS**

    * *curve* - a ColorCurve object, or 2d list that can be conditioned
     into one

    **RETURNS**

    A SimpleCV Image

    **EXAMPLE**

    >>> img = Image("lenna")
    >>> cc = ColorCurve([[0,0], [100, 120], [180, 230], [255, 255]])
    >>> img2 = img.apply_rgb_curve(cc)

    **SEE ALSO**

    :py:class:`ColorCurve`
    :py:meth:`apply_hls_curve`

    """
    return img.apply_rgb_curve(rcurve=curve, gcurve=curve, bcurve=curve)


@image_method
def color_distance(img, color=Color.BLACK):
        """
        **SUMMARY**

        Returns an image representing the distance of each pixel from a given
        color tuple, scaled between 0 (the given color) and 255. Pixels distant
        from the given tuple will appear as brighter and pixels closest to the
        target color will be darker.


        By default this will give image intensity (distance from pure black)

        **PARAMETERS**

        * *color*  - Color object or Color Tuple

        **RETURNS**

        A SimpleCV Image.

        **EXAMPLE**

        >>> img = Image("logo")
        >>> img2 = img.color_distance(color=Color.BLACK)
        >>> img2.show()


        **SEE ALSO**

        :py:meth:`binarize`
        :py:meth:`hue_distance`
        :py:meth:`find_blobs_from_mask`
        """
        # reshape our matrix to 1xN
        pixels = img._ndarray.copy().reshape(-1, 3)

        # calculate the distance each pixel is
        distances = spsd.cdist(pixels, [color])
        distances *= (255.0 / distances.max())  # normalize to 0 - 255
        array = distances.reshape(img.width, img.height)
        return Factory.Image(array)


@image_method
def hue_distance(img, color=Color.BLACK, minsaturation=20, minvalue=20,
                 maxvalue=255):
    """
    **SUMMARY**

    Returns an image representing the distance of each pixel from the given
    hue of a specific color.  The hue is "wrapped" at 180, so we have to
    take the shorter of the distances between them -- this gives a hue
    distance of max 90, which we'll scale into a 0-255 grayscale image.

    The minsaturation and minvalue are optional parameters to weed out very
    weak hue signals in the picture, they will be pushed to max distance
    [255]


    **PARAMETERS**

    * *color* - Color object or Color Tuple.
    * *minsaturation*  - the minimum saturation value for color
     (from 0 to 255).
    * *minvalue*  - the minimum hue value for the color
     (from 0 to 255).

    **RETURNS**

    A simpleCV image.

    **EXAMPLE**

    >>> img = Image("logo")
    >>> img2 = img.hue_distance(color=Color.BLACK)
    >>> img2.show()

    **SEE ALSO**

    :py:meth:`binarize`
    :py:meth:`hue_distance`
    :py:meth:`morph_open`
    :py:meth:`morph_close`
    :py:meth:`morph_gradient`
    :py:meth:`find_blobs_from_mask`

    """
    if isinstance(color, (float, int, long, complex)):
        color_hue = color
    else:
        color_hue = Color.hsv(color)[0]

    # again, gets transposed to vsh
    vsh_matrix = img.to_hsv().get_ndarray().reshape(-1, 3)
    hue_channel = np.cast['int'](vsh_matrix[:, 2])

    if color_hue < 90:
        hue_loop = 180
    else:
        hue_loop = -180
    #set whether we need to move back or forward on the hue circle

    distances = np.minimum(np.abs(hue_channel - color_hue),
                           np.abs(hue_channel - (color_hue + hue_loop)))
    #take the minimum distance for each pixel

    distances = np.where(
        np.logical_and(
            vsh_matrix[:, 0] > minvalue,
            vsh_matrix[:, 1] > minsaturation),
        distances * (255.0 / 90.0),  # normalize 0 - 90 -> 0 - 255
        # use the maxvalue if it false outside
        # of our value/saturation tolerances
        255.0)

    return Factory.Image(distances.reshape(img.width, img.height))


@image_method
def erode(img, iterations=1, kernelsize=3):
    """
    **SUMMARY**

    Apply a morphological erosion. An erosion has the effect of removing
    small bits of noise and smothing blobs.

    This implementation uses the default openCV 3X3 square kernel

    Erosion is effectively a local minima detector, the kernel moves over
    the image and takes the minimum value inside the kernel.
    iterations - this parameters is the number of times to apply/reapply
    the operation

    * See: http://en.wikipedia.org/wiki/Erosion_(morphology).

    * See: http://opencv.willowgarage.com/documentation/cpp/
     image_filtering.html#cv-erode

    * Example Use: A threshold/blob image has 'salt and pepper' noise.

    * Example Code: /examples/MorphologyExample.py

    **PARAMETERS**

    * *iterations* - the number of times to run the erosion operation.

    **RETURNS**

    A SimpleCV image.

    **EXAMPLE**

    >>> img = Image("lenna")
    >>> derp = img.binarize()
    >>> derp.erode(3).show()

    **SEE ALSO**
    :py:meth:`dilate`
    :py:meth:`binarize`
    :py:meth:`morph_open`
    :py:meth:`morph_close`
    :py:meth:`morph_gradient`
    :py:meth:`find_blobs_from_mask`

    """
    kern = cv2.getStructuringElement(shape=cv2.MORPH_RECT,
                                     ksize=(kernelsize, kernelsize),
                                     anchor=(1, 1))
    array = cv2.erode(img.get_ndarray(), kernel=kern, iterations=iterations)
    return Factory.Image(array, color_space=img.color_space)


@image_method
def dilate(img, iterations=1):
    """
    **SUMMARY**

    Apply a morphological dilation. An dilation has the effect of smoothing
    blobs while intensifying the amount of noise blobs.
    This implementation uses the default openCV 3X3 square kernel
    Erosion is effectively a local maxima detector, the kernel moves over
    the image and takes the maxima value inside the kernel.

    * See: http://en.wikipedia.org/wiki/Dilation_(morphology)

    * See: http://opencv.willowgarage.com/documentation/cpp/
     image_filtering.html#cv-dilate

    * Example Use: A part's blob needs to be smoother

    * Example Code: ./examples/MorphologyExample.py

    **PARAMETERS**

    * *iterations* - the number of times to run the dilation operation.

    **RETURNS**

    A SimpleCV image.

    **EXAMPLE**

    >>> img = Image("lenna")
    >>> derp = img.binarize()
    >>> derp.dilate(3).show()

    **SEE ALSO**

    :py:meth:`erode`
    :py:meth:`binarize`
    :py:meth:`morph_open`
    :py:meth:`morph_close`
    :py:meth:`morph_gradient`
    :py:meth:`find_blobs_from_mask`

    """
    kern = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(3, 3),
                                     anchor=(1, 1))
    array = cv2.dilate(img.get_ndarray(), kernel=kern, iterations=iterations)
    return Factory.Image(array, color_space=img.color_space)


@image_method
def morph_open(img):
    """
    **SUMMARY**

    morphologyOpen applies a morphological open operation which is
    effectively an erosion operation followed by a morphological dilation.
    This operation helps to 'break apart' or 'open' binary regions which
    are close together.


    * `Morphological opening on Wikipedia <http://en.wikipedia.org/wiki/
     Opening_(morphology)>`_

    * `OpenCV documentation <http://opencv.willowgarage.com/documentation/
     cpp/image_filtering.html#cv-morphologyex>`_

    * Example Use: two part blobs are 'sticking' together.

    * Example Code: ./examples/MorphologyExample.py

    **RETURNS**

    A SimpleCV image.

    **EXAMPLE**

    >>> img = Image("lenna")
    >>> derp = img.binarize()
    >>> derp.morph_open.show()

    **SEE ALSO**

    :py:meth:`erode`
    :py:meth:`dilate`
    :py:meth:`binarize`
    :py:meth:`morph_close`
    :py:meth:`morph_gradient`
    :py:meth:`find_blobs_from_mask`

    """
    kern = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(3, 3),
                                     anchor=(1, 1))
    array = cv2.morphologyEx(src=img.get_ndarray(), op=cv2.MORPH_OPEN,
                             kernel=kern, anchor=(1, 1), iterations=1)
    return Factory.Image(array, color_space=img.color_space)


@image_method
def morph_close(img):
    """
    **SUMMARY**

    morphologyClose applies a morphological close operation which is
    effectively a dilation operation followed by a morphological erosion.
    This operation helps to 'bring together' or 'close' binary regions
    which are close together.


    * See: `Closing <http://en.wikipedia.org/wiki/Closing_(morphology)>`_

    * See: `Morphology from OpenCV <http://opencv.willowgarage.com/
     documentation/cpp/image_filtering.html#cv-morphologyex>`_

    * Example Use: Use when a part, which should be one blob is really two
     blobs.

    * Example Code: ./examples/MorphologyExample.py

    **RETURNS**

    A SimpleCV image.

    **EXAMPLE**

    >>> img = Image("lenna")
    >>> derp = img.binarize()
    >>> derp.morph_close.show()

    **SEE ALSO**

    :py:meth:`erode`
    :py:meth:`dilate`
    :py:meth:`binarize`
    :py:meth:`morph_open`
    :py:meth:`morph_gradient`
    :py:meth:`find_blobs_from_mask`

    """
    kern = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(3, 3),
                                     anchor=(1, 1))
    array = cv2.morphologyEx(src=img.get_ndarray(), op=cv2.MORPH_CLOSE,
                             kernel=kern, anchor=(1, 1), iterations=1)
    return Factory.Image(array, color_space=img.color_space)


@image_method
def morph_gradient(img):
    """
    **SUMMARY**

    The morphological gradient is the difference betwen the morphological
    dilation and the morphological gradient. This operation extracts the
    edges of a blobs in the image.


    * `See Morph Gradient of Wikipedia <http://en.wikipedia.org/wiki/
    Morphological_Gradient>`_

    * `OpenCV documentation <http://opencv.willowgarage.com/documentation/
     cpp/image_filtering.html#cv-morphologyex>`_

    * Example Use: Use when you have blobs but you really just want to know
     the blob edges.

    * Example Code: ./examples/MorphologyExample.py


    **RETURNS**

    A SimpleCV image.

    **EXAMPLE**

    >>> img = Image("lenna")
    >>> derp = img.binarize()
    >>> derp.morph_gradient.show()

    **SEE ALSO**

    :py:meth:`erode`
    :py:meth:`dilate`
    :py:meth:`binarize`
    :py:meth:`morph_open`
    :py:meth:`morph_close`
    :py:meth:`find_blobs_from_mask`

    """
    kern = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(3, 3),
                                     anchor=(1, 1))
    array = cv2.morphologyEx(img.get_ndarray(), op=cv2.MORPH_GRADIENT,
                             kernel=kern)
    return Factory.Image(array, color_space=img.color_space)


@image_method
def maximum(img, other):
    """
    **SUMMARY**

    The maximum value of my image, and the other image, in each channel
    If other is a number, returns the maximum of that and the number

    **PARAMETERS**

    * *other* - Image of the same size or a number.

    **RETURNS**

    A SimpelCV image.

    """
    if isinstance(other, Factory.Image):
        if img.size != other.size:
            logger.warn("Both images should have same dimensions. "
                        "Returning None.")
            return None
        array = cv2.max(img.get_ndarray(), other.get_ndarray())
        return Factory.Image(array, color_space=img.color_space)
    else:
        array = np.maximum(img.get_ndarray(), other)
        return Factory.Image(array, color_space=img.color_space)


@image_method
def minimum(img, other):
    """
    **SUMMARY**

    The minimum value of my image, and the other image, in each channel
    If other is a number, returns the minimum of that and the number

    **Parameter**

    * *other* - Image of the same size or number

    **Returns**

    IMAGE
    """

    if isinstance(other, Factory.Image):
        if img.size != other.size:
            logger.warn("Both images should have same dimensions. "
                        "Returning None.")
            return None
        array = cv2.min(img.get_ndarray(), other.get_ndarray())
        return Factory.Image(array, color_space=img.color_space)
    else:
        array = np.minimum(img.get_ndarray(), other)
        return Factory.Image(array, color_space=img.color_space)


@image_method
def edges(img, t1=50, t2=100):
    """
    **SUMMARY**

    Finds an edge map Image using the Canny edge detection method. Edges
    will be brighter than the surrounding area.

    The t1 parameter is roughly the "strength" of the edge required, and
    the value between t1 and t2 is used for edge linking.

    For more information:

    * http://opencv.willowgarage.com/documentation/python/
    imgproc_feature_detection.html

    * http://en.wikipedia.org/wiki/Canny_edge_detector

    **PARAMETERS**

    * *t1* - Int - the lower Canny threshold.
    * *t2* - Int - the upper Canny threshold.

    **RETURNS**

    A SimpleCV image where the edges are white on a black background.

    **EXAMPLE**

    >>> cam = Camera()
    >>> while True:
    >>>    cam.getImage().edges().show()


    **SEE ALSO**

    :py:meth:`find_lines`

    """
    return Factory.Image(Factory.Image.get_edge_map(img, t1, t2),
                         color_space=img.color_space)


@image_method
def create_binary_mask(img, color1=(0, 0, 0), color2=(255, 255, 255)):
    """
    **SUMMARY**

    Generate a binary mask of the image based on a range of rgb values.
    A binary mask is a black and white image where the white area is kept
    and the black area is removed.

    This method is used by specifying two colors as the range between the
    minimum and maximum values that will be masked white.

    **PARAMETERS**

    * *color1* - The starting color range for the mask..
    * *color2* - The end of the color range for the mask.

    **RETURNS**

    A binary (black/white) image mask as a SimpleCV Image.

    **EXAMPLE**

    >>> img = Image("lenna")
    >>> mask = img.create_binary_mask(color1=(0, 128, 128),
    ...                               color2=(255, 255, 255))
    >>> mask.show()

    **SEE ALSO**

    :py:meth:`create_binary_mask`
    :py:meth:`create_alpha_mask`
    :py:meth:`blit`
    :py:meth:`threshold`

    """
    if not img.is_bgr():
        logger.warning("create_binary_mask works only with BGR image")
        return None
    if color1[0] - color2[0] == 0 \
            or color1[1] - color2[1] == 0 \
            or color1[2] - color2[2] == 0:
        logger.warning("No color range selected, the result will be "
                       "black, returning None instead.")
        return None
    if color1[0] > 255 or color1[0] < 0 \
            or color1[1] > 255 or color1[1] < 0 \
            or color1[2] > 255 or color1[2] < 0 \
            or color2[0] > 255 or color2[0] < 0 \
            or color2[1] > 255 or color2[1] < 0 \
            or color2[2] > 255 or color2[2] < 0:
        logger.warning("One of the tuple values falls "
                       "outside of the range of 0 to 255")
        return None
    # converting to BGR
    color1 = tuple(reversed(color1))
    color2 = tuple(reversed(color2))

    results = []
    for index, color in enumerate(zip(color1, color2)):
        chanel = cv2.inRange(img.get_ndarray()[:, :, index],
                             lowerb=np.array(min(color)),
                             upperb=np.array(max(color)))
        results.append(chanel)
    array = cv2.bitwise_and(results[0], results[1])
    array = cv2.bitwise_and(array, results[2]).astype(img.dtype)
    return Factory.Image(array, color_space=Factory.Image.GRAY)


@image_method
def apply_binary_mask(img, mask, bg_color=Color.BLACK):
    """
    **SUMMARY**

    Apply a binary mask to the image. The white areas of the mask will be
    kept, and the black areas removed. The removed areas will be set to the
    color of bg_color.

    **PARAMETERS**

    * *mask* - the binary mask image. White areas are kept, black areas are
     removed.
    * *bg_color* - the color of the background on the mask.

    **RETURNS**

    A binary (black/white) image mask as a SimpleCV Image.

    **EXAMPLE**

    >>> img = Image("lenna")
    >>> mask = img.create_binary_mask(color1=(0, 128, 128),
    ...                               color2=(255, 255, 255))
    >>> result = img.apply_binary_mask(mask)
    >>> result.show()

    **SEE ALSO**

    :py:meth:`create_binary_mask`
    :py:meth:`create_alpha_mask`
    :py:meth:`apply_binary_mask`
    :py:meth:`blit`
    :py:meth:`threshold`

    """
    if img.size != mask.size:
        logger.warning("Image.apply_binary_mask: your mask and image "
                       "don't match sizes, if the mask doesn't fit, you "
                       "can't apply it! Try using the scale function. ")
        return None

    array = np.zeros((img.height, img.width, 3), img.dtype)
    array = cv2.add(array, np.array(tuple(reversed(bg_color)),
                                    dtype=img.dtype))
    binary_mask = mask.get_gray_ndarray() != 0

    array[binary_mask] = img.get_ndarray()[binary_mask]
    return Factory.Image(array, color_space=img.color_space)


@image_method
def create_alpha_mask(img, hue=60, hue_lb=None, hue_ub=None):
    """
    **SUMMARY**

    Generate a grayscale or binary mask image based either on a hue or an
    RGB triplet that can be used like an alpha channel. In the resulting
    mask, the hue/rgb_color will be treated as transparent (black).

    When a hue is used the mask is treated like an 8bit alpha channel.
    When an RGB triplet is used the result is a binary mask.
    rgb_thresh is a distance measure between a given a pixel and the mask
    value that we will add to the mask. For example, if rgb_color=(0,255,0)
    and rgb_thresh=5 then any pixel within five color values of the
    rgb_color will be added to the mask (e.g. (0,250,0),(5,255,0)....)

    Invert flips the mask values.


    **PARAMETERS**

    * *hue* - a hue used to generate the alpha mask.
    * *hue_lb* - the upper value  of a range of hue values to use.
    * *hue_ub* - the lower value  of a range of hue values to use.

    **RETURNS**

    A grayscale alpha mask as a SimpleCV Image.

    >>> img = Image("lenna")
    >>> mask = img.create_alpha_mask(hue_lb=50, hue_ub=70)
    >>> mask.show()

    **SEE ALSO**

    :py:meth:`create_binary_mask`
    :py:meth:`create_alpha_mask`
    :py:meth:`apply_binary_mask`
    :py:meth:`blit`
    :py:meth:`threshold`

    """

    if hue < 0 or hue > 180:
        logger.warning("Invalid hue color, valid hue range is 0 to 180.")
        return None

    if not img.is_hsv():
        hsv = img.to_hsv()
    else:
        hsv = img.copy()
    h = hsv.get_ndarray()[:, :, 0]
    v = hsv.get_ndarray()[:, :, 2]
    hlut = np.zeros(256, dtype=np.uint8)
    if hue_lb is not None and hue_ub is not None:
        hlut[hue_lb:hue_ub] = 255
    else:
        hlut[hue] = 255
    mask = cv2.LUT(h, lut=hlut)[:, :, 0]
    array = hsv.get_empty(1)
    array = np.where(mask, v, array)
    return Factory.Image(array)


@image_method
def apply_pixel_function(img, func):
    """
    **SUMMARY**

    apply a function to every pixel and return the result
    The function must be of the form int (r,g,b)=func((r,g,b))

    **PARAMETERS**

    * *func* - a function pointer to a function of the form
     (r,g.b) = func((r,g,b))

    **RETURNS**

    A simpleCV image after mapping the function to the image.

    **EXAMPLE**

    >>> def derp(pixels):
    >>>     b, g, r = pixels
    >>>     return int(b * .2), int(r * .3), int(g * .5)
    >>>
    >>> img = Image("lenna")
    >>> img2 = img.apply_pixel_function(derp)

    """
    # there should be a way to do this faster using numpy vectorize
    # but I can get vectorize to work with the three channels together...
    # have to split them
    #TODO: benchmark this against vectorize
    pixels = np.array(img.get_ndarray()).reshape(-1, 3).tolist()
    result = np.array(map(func, pixels), dtype=np.uint8).reshape(
        (img.width, img.height, 3))
    return Factory.Image(result)


@image_method
def convolve(img, kernel=None, center=None):
    """
    **SUMMARY**

    Convolution performs a shape change on an image.  It is similiar to
    something like a dilate.  You pass it a kernel in the form of a list,
    np.array, or cvMat

    **PARAMETERS**

    * *kernel* - The convolution kernel. As list, set or Numpy Array.
    * *center* - If true we use the center of the kernel.

    **RETURNS**

    The image after we apply the convolution.

    **EXAMPLE**

    >>> img = Image("data/sampleimages/simplecv.png")
    >>> kernel = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    >>> conv = img.convolve()

    **SEE ALSO**

    http://en.wikipedia.org/wiki/Convolution

    """
    if kernel is None:
        kernel = np.array(((1, 0, 0), (0, 1, 0), (0, 0, 1)))
    elif isinstance(kernel, (list, set)):
        kernel = np.array(kernel)
    elif isinstance(kernel, np.ndarray):
        pass
    else:
        logger.warning("Image.convolve: kernel should be numpy array.")
        return None
    kernel = kernel.astype(np.float32)
    array = cv2.filter2D(img.get_ndarray(), ddepth=-1,
                         kernel=kernel, anchor=center)
    return Factory.Image(array, color_space=img.color_space)


@image_method
def white_balance(img, method="Simple"):
    """
    **SUMMARY**

    Attempts to perform automatic white balancing.
    Gray World see:
    http://scien.stanford.edu/pages/labsite/2000/psych221/
    projects/00/trek/GWimages.html

    Robust AWB:
    http://scien.stanford.edu/pages/labsite/2010/psych221/
    projects/2010/JasonSu/robustawb.html

    http://scien.stanford.edu/pages/labsite/2010/psych221/
    projects/2010/JasonSu/Papers/
    Robust%20Automatic%20White%20Balance%20Algorithm%20using
    %20Gray%20Color%20Points%20in%20Images.pdf

    Simple AWB:
    http://www.ipol.im/pub/algo/lmps_simplest_color_balance/
    http://scien.stanford.edu/pages/labsite/2010/psych221/
    projects/2010/JasonSu/simplestcb.html



    **PARAMETERS**

    * *method* - The method to use for white balancing. Can be one of the
     following:

      * `Gray World <http://scien.stanford.edu/pages/labsite/2000/psych221/
        projects/00/trek/GWimages.html>`_

      * `Robust AWB <http://scien.stanford.edu/pages/labsite/2010/psych221/
        projects/2010/JasonSu/robustawb.html>`_

      * `Simple AWB <http://www.ipol.im/pub/algo/
        lmps_simplest_color_balance/>`_


    **RETURNS**

    A SimpleCV Image.

    **EXAMPLE**

    >>> img = Image("lenna")
    >>> img2 = img.white_balance()

    """
    if not img.is_bgr():
        logger.warning("Image.white_balance: works only with BGR image")
        return None

    if method == "GrayWorld":
        avg = cv2.mean(img.get_ndarray())
        bf = float(avg[0])
        gf = float(avg[1])
        rf = float(avg[2])
        af = (bf + gf + rf) / 3.0
        if bf == 0.00:
            b_factor = 1.00
        else:
            b_factor = af / bf

        if gf == 0.00:
            g_factor = 1.00
        else:
            g_factor = af / gf

        if rf == 0.00:
            r_factor = 1.00
        else:
            r_factor = af / rf

        b = img.get_ndarray()[:, :, 0]
        g = img.get_ndarray()[:, :, 1]
        r = img.get_ndarray()[:, :, 2]

        bfloat = cv2.convertScaleAbs(b.astype(np.float32), alpha=b_factor)
        gfloat = cv2.convertScaleAbs(g.astype(np.float32), alpha=g_factor)
        rfloat = cv2.convertScaleAbs(r.astype(np.float32), alpha=r_factor)

        (min_b, max_b, min_b_loc, max_b_loc) = cv2.minMaxLoc(bfloat)
        (min_g, max_g, min_g_loc, max_g_loc) = cv2.minMaxLoc(gfloat)
        (min_r, max_r, min_r_loc, max_r_loc) = cv2.minMaxLoc(rfloat)
        scale = max([max_r, max_g, max_b])
        sfactor = 1.00
        if scale > 255:
            sfactor = 255.00 / float(scale)

        b = cv2.convertScaleAbs(bfloat, alpha=sfactor)
        g = cv2.convertScaleAbs(gfloat, alpha=sfactor)
        r = cv2.convertScaleAbs(rfloat, alpha=sfactor)

        array = np.dstack((b, g, r)).astype(img.dtype)
        return Factory.Image(array, color_space=Factory.Image.BGR)
    elif method == "Simple":
        thresh = 0.003
        sz = img.width * img.height
        bcf = sss.cumfreq(img.get_ndarray()[:, :, 0], numbins=256)
        # get our cumulative histogram of values for this color
        bcf = bcf[0]

        blb = -1  # our upper bound
        bub = 256  # our lower bound
        lower_thresh = 0.00
        upper_thresh = 0.00
        #now find the upper and lower thresh% of our values live
        while lower_thresh < thresh:
            blb = blb + 1
            lower_thresh = bcf[blb] / sz
        while upper_thresh < thresh:
            bub = bub - 1
            upper_thresh = (sz - bcf[bub]) / sz

        gcf = sss.cumfreq(img._ndarray[:, :, 1], numbins=256)
        gcf = gcf[0]
        glb = -1  # our upper bound
        gub = 256  # our lower bound
        lower_thresh = 0.00
        upper_thresh = 0.00
        #now find the upper and lower thresh% of our values live
        while lower_thresh < thresh:
            glb = glb + 1
            lower_thresh = gcf[glb] / sz
        while upper_thresh < thresh:
            gub = gub - 1
            upper_thresh = (sz - gcf[gub]) / sz

        rcf = sss.cumfreq(img.get_ndarray()[:, :, 2], numbins=256)
        rcf = rcf[0]
        rlb = -1  # our upper bound
        rub = 256  # our lower bound
        lower_thresh = 0.00
        upper_thresh = 0.00
        #now find the upper and lower thresh% of our values live
        while lower_thresh < thresh:
            rlb = rlb + 1
            lower_thresh = rcf[rlb] / sz
        while upper_thresh < thresh:
            rub = rub - 1
            upper_thresh = (sz - rcf[rub]) / sz
        #now we create the scale factors for the remaining pixels
        rlbf = float(rlb)
        rubf = float(rub)
        glbf = float(glb)
        gubf = float(gub)
        blbf = float(blb)
        bubf = float(bub)

        r_lut = np.ones((256, 1), dtype=np.uint8)
        g_lut = np.ones((256, 1), dtype=np.uint8)
        b_lut = np.ones((256, 1), dtype=np.uint8)
        for i in range(256):
            if i <= rlb:
                r_lut[i][0] = 0
            elif i >= rub:
                r_lut[i][0] = 255
            else:
                rf = ((float(i) - rlbf) * 255.00 / (rubf - rlbf))
                r_lut[i][0] = int(rf)
            if i <= glb:
                g_lut[i][0] = 0
            elif i >= gub:
                g_lut[i][0] = 255
            else:
                gf = ((float(i) - glbf) * 255.00 / (gubf - glbf))
                g_lut[i][0] = int(gf)
            if i <= blb:
                b_lut[i][0] = 0
            elif i >= bub:
                b_lut[i][0] = 255
            else:
                bf = ((float(i) - blbf) * 255.00 / (bubf - blbf))
                b_lut[i][0] = int(bf)
        return img.apply_lut(r_lut, g_lut, b_lut)


@image_method
def apply_lut(img, r_lut=None, b_lut=None, g_lut=None):
    """
    **SUMMARY**

    Apply LUT allows you to apply a LUT (look up table) to the pixels in a
    image. Each LUT is just an array where each index in the array points
    to its value in the result image. For example r_lut[0]=255 would change
    all pixels where the red channel is zero to the value 255.

    **PARAMETERS**

    * *r_lut* - np.array of size (256x1) with dtype=uint8.
    * *g_lut* - np.array of size (256x1) with dtype=uint8.
    * *b_lut* - np.array of size (256x1) with dtype=uint8.

    .. warning::
      The dtype is very important. Will throw the following error without
      it: error: dst.size() == src.size() &&
      dst.type() == CV_MAKETYPE(lut.depth(), src.channels())


    **RETURNS**

    The SimpleCV image remapped using the LUT.

    **EXAMPLE**

    This example saturates the red channel:

    >>> rlut = np.ones((256, 1), dtype=np.uint8) * 255
    >>> img=img.apply_lut(r_lut=rlut)


    NOTE:

    -==== BUG NOTE ====-
    This method seems to error on the LUT map for some versions of OpenCV.
    I am trying to figure out why. -KAS
    """
    if not img.is_bgr():
        logger.warning("Image.apply_lut: works only with BGR image")
        return None
    b = img.get_ndarray()[:, :, 0]
    g = img.get_ndarray()[:, :, 1]
    r = img.get_ndarray()[:, :, 2]
    if r_lut is not None:
        r = cv2.LUT(r, lut=r_lut)
    if g_lut is not None:
        g = cv2.LUT(g, lut=g_lut)
    if b_lut is not None:
        b = cv2.LUT(b, lut=b_lut)
    array = np.dstack((b, g, r))
    return Factory.Image(array, color_space=img.color_space)


@image_method
def palettize(img, bins=10, hue=False, centroids=None):
    """
    **SUMMARY**

    This method analyzes an image and determines the most common colors
    using a k-means algorithm. The method then goes through and replaces
    each pixel with the centroid of the clutsters found by k-means. This
    reduces the number of colors in an image to the number of bins. This
    can be particularly handy for doing segementation based on color.

    **PARAMETERS**

    * *bins* - an integer number of bins into which to divide the colors
      in the image.
    * *hue* - if hue is true we do only cluster on the image hue values.


    **RETURNS**

    An image matching the original where each color is replaced with its
    palette value.

    **EXAMPLE**

    >>> img2 = img1.palettize()
    >>> img2.show()

    **NOTES**

    The hue calculations should be siginificantly faster than the generic
    RGB calculation as it works in a one dimensional space. Sometimes the
    underlying scipy method freaks out about k-means initialization with
    the following warning:

    .. Warning::
      UserWarning: One of the clusters is empty. Re-run kmean with a
      different initialization. This shouldn't be a real problem.

    **SEE ALSO**

    :py:meth:`re_palette`
    :py:meth:`draw_palette_colors`
    :py:meth:`palettize`
    :py:meth:`get_palette`
    :py:meth:`binarize_from_palette`
    :py:meth:`find_blobs_from_palette`

    """
    ret_val = None
    img.generate_palette(bins, hue, centroids)
    derp = img._palette[img._palette_members]
    if hue:
        ret_val = Factory.Image(derp.reshape(img.height, img.width))
    else:
        ret_val = Factory.Image(derp.reshape(img.height, img.width, 3))
    return ret_val


@image_method
def binarize_from_palette(img, palette_selection):
    """
    **SUMMARY**

    This method uses the color palette to generate a binary (black and
    white) image. Palaette selection is a list of color tuples retrieved
    from img.get_palette(). The provided values will be drawn white
    while other values will be black.

    **PARAMETERS**

    palette_selection - color triplets selected from our palette that will
    serve turned into blobs. These values can either be a 3xN numpy array,
    or a list of RGB triplets.

    **RETURNS**

    This method returns a black and white images, where colors that are
    close to the colors in palette_selection are set to white

    **EXAMPLE**

    >>> img = Image("lenna")
    >>> p = img.get_palette()
    >>> b = img.binarize_from_palette((p[0], p[1], [6]))
    >>> b.show()

    **SEE ALSO**

    :py:meth:`re_palette`
    :py:meth:`draw_palette_colors`
    :py:meth:`palettize`
    :py:meth:`get_palette`
    :py:meth:`binarize_from_palette`
    :py:meth:`find_blobs_from_palette`

    """

    #we get the palette from find palete
    #ASSUME: GET PALLETE WAS CALLED!
    if img._palette is None:
        logger.warning("Image.binarize_from_palette: No palette exists, "
                       "call get_palette())")
        return None
    ret_val = None
    palettized_img = img.palettize(bins=img._palette_bins,
                                   hue=img._do_hue_palette)
    if not img._do_hue_palette:
        npimg = palettized_img.get_ndarray()
        white = np.array([255, 255, 255], dtype=np.uint8)
        black = np.array([0, 0, 0], dtype=np.uint8)

        for p in palette_selection:
            npimg = np.where(npimg != p, npimg, white)

        npimg = np.where(npimg != white, black, white)
        ret_val = Factory.Image(npimg)
    else:
        npimg = palettized_img.get_ndarray()
        white = np.array([255], dtype=np.uint8)
        black = np.array([0], dtype=np.uint8)

        for p in palette_selection:
            npimg = np.where(npimg != p, npimg, white)

        npimg = np.where(npimg != white, black, white)
        ret_val = Factory.Image(npimg.astype(np.uint8))
    return ret_val


@image_method
def skeletonize(img, radius=5):
    """
    **SUMMARY**

    Skeletonization is the process of taking in a set of blobs (here blobs
    are white on a black background) and finding a squigly line that would
    be the back bone of the blobs were they some sort of vertebrate animal.
    Another way of thinking about skeletonization is that it finds a series
    of lines that approximates a blob's shape.

    A good summary can be found here:

    http://www.inf.u-szeged.hu/~palagyi/skel/skel.html

    **PARAMETERS**

    * *radius* - an intenger that defines how roughly how wide a blob must
      be to be added to the skeleton, lower values give more skeleton
      lines, higher values give fewer skeleton lines.

    **EXAMPLE**

    >>> cam = Camera()
    >>> while True:
    ...     img = cam.getImage()
    ...     b = img.binarize().invert()
    ...     s = img.skeletonize()
    ...     r = b - s
    ...     r.show()


    **NOTES**

    This code was a suggested improvement by Alex Wiltchko, check out his
    awesome blog here:

    http://alexbw.posterous.com/

    """
    img_array = img.get_gray_ndarray()
    distance_img = ndimage.distance_transform_edt(img_array)
    morph_laplace_img = ndimage.morphological_laplace(distance_img,
                                                      size=(radius, radius))
    skeleton = morph_laplace_img < morph_laplace_img.min() / 2
    ret_val = np.zeros([img.width, img.height])
    ret_val[skeleton] = 255
    ret_val = ret_val.astype(img.dtype)
    return Factory.Image(ret_val)


@image_method
def smart_threshold(img, mask=None, rect=None):
    """
    **SUMMARY**

    smart_threshold uses a method called grabCut, also called graph cut, to
    automagically generate a grayscale mask image. The dumb version of
    threshold just uses color, smart_threshold looks at both color and
    edges to find a blob. To work smart_threshold needs either a rectangle
    that bounds the object you want to find, or a mask. If you use
    a rectangle make sure it holds the complete object. In the case of
    a mask, it need not be a normal binary mask, it can have the normal
    white foreground and black background, but also a light and dark gray
    values that correspond to areas that are more likely to be foreground
    and more likely to be background. These values can be found in the
    color class as Color.BACKGROUND, Color.FOREGROUND,
    Color.MAYBE_BACKGROUND, and Color.MAYBE_FOREGROUND.

    **PARAMETERS**

    * *mask* - A grayscale mask the same size as the image using the 4 mask
     color values
    * *rect* - A rectangle tuple of the form (x_position, y_position,
      width, height)

    **RETURNS**

    A grayscale image with the foreground / background values assigned to:

    * BACKGROUND = (0,0,0)

    * MAYBE_BACKGROUND = (64,64,64)

    * MAYBE_FOREGROUND =  (192,192,192)

    * FOREGROUND = (255,255,255)

    **EXAMPLE**

    >>> img = Image("RatTop.png")
    >>> mask = Image((img.width,img.height))
    >>> mask.dl().circle((100, 100), 80, color=Color.MAYBE_BACKGROUND,
    ...                  filled=True)
    >>> mask.dl().circle((100, 100), 60, color=Color.MAYBE_FOREGROUND,
    ...                  filled=True)
    >>> mask.dl().circle((100 ,100), 40, color=Color.FOREGROUND,
    ...                  filled=True)
    >>> mask = mask.apply_layers()
    >>> new_mask = img.smart_threshold(mask=mask)
    >>> new_mask.show()

    **NOTES**

    http://en.wikipedia.org/wiki/Graph_cuts_in_computer_vision

    **SEE ALSO**

    :py:meth:`smart_find_blobs`

    """
    ret_val = None
    if mask is not None:
        gray_array = mask.get_gray_ndarray()
        # translate the human readable images to something
        # opencv wants using a lut
        lut = np.zeros((256, 1), dtype=np.uint8)
        lut[255] = 1
        lut[64] = 2
        lut[192] = 3
        gray_array = cv2.LUT(gray_array, lut)
        mask_in = gray_array.copy()
        # get our image in a flavor grab cut likes
        npimg = img.get_ndarray()
        # require by opencv
        tmp1 = np.zeros((1, 13 * 5))
        tmp2 = np.zeros((1, 13 * 5))
        # do the algorithm
        cv2.grabCut(npimg, mask_in, None, tmp1, tmp2, 10,
                    mode=cv2.GC_INIT_WITH_MASK)
        # remap the color space
        lut = np.zeros((256, 1), dtype=np.uint8)
        lut[1] = 255
        lut[2] = 64
        lut[3] = 192
        output = cv2.LUT(mask_in, lut=lut)
        ret_val = Factory.Image(output)

    elif rect is not None:
        npimg = img.get_ndarray()
        tmp1 = np.zeros((1, 13 * 5))
        tmp2 = np.zeros((1, 13 * 5))
        mask = np.zeros((img.height, img.width), dtype=np.uint8)
        cv2.grabCut(npimg, mask=mask, rect=rect, bgdModel=tmp1,
                    fgdModel=tmp2, iterCount=10, mode=cv2.GC_INIT_WITH_RECT)
        lut = np.zeros((256, 1), dtype=np.uint8)
        lut[1] = 255
        lut[2] = 64
        lut[3] = 192
        array = cv2.LUT(mask, lut=lut)
        ret_val = Factory.Image(array)
    else:
        logger.warning("ImageClass.findBlobsSmart requires either a mask "
                       "or a selection rectangle. Failure to provide one "
                       "of these causes your bytes to splinter and bit "
                       "shrapnel to hit your pipeline making it asplode "
                       "in a ball of fire. Okay... not really")
    return ret_val


@image_method
def threshold(img, value):
    """
    **SUMMARY**

    We roll old school with this vanilla threshold function. It takes your
    image converts it to grayscale, and applies a threshold. Values above
    the threshold are white, values below the threshold are black
    (note this is in contrast to binarize... which is a stupid function
    that drives me up a wall). The resulting black and white image is
    returned.

    **PARAMETERS**

    * *value* - the threshold, goes between 0 and 255.

    **RETURNS**

    A black and white SimpleCV image.

    **EXAMPLE**

    >>> img = Image("purplemonkeydishwasher.png")
    >>> result = img.threshold(42)

    **NOTES**

    THRESHOLD RULES BINARIZE DROOLS!

    **SEE ALSO**

    :py:meth:`binarize`

    """
    gray = img.get_gray_ndarray()
    _, array = cv2.threshold(gray, thresh=value, maxval=255,
                             type=cv2.THRESH_BINARY)
    return Factory.Image(array)


@image_method
def flood_fill(img, points, tolerance=None, color=Color.WHITE, lower=None,
               upper=None, fixed_range=True):
    """
    **SUMMARY**

    FloodFill works just like ye olde paint bucket tool in your favorite
    image manipulation program. You select a point (or a list of points),
    a color, and a tolerance, and flood_fill will start at that point,
    looking for pixels within the tolerance from your intial pixel. If the
    pixel is in tolerance, we will convert it to your color, otherwise the
    method will leave the pixel alone. The method accepts both single
    values, and triplet tuples for the tolerance values. If you require
    more control over your tolerance you can use the upper and lower
    values. The fixed range parameter let's you toggle between setting the
    tolerance with repect to the seed pixel, and using a tolerance that is
    relative to the adjacent pixels. If fixed_range is true the method will
    set its tolerance with respect to the seed pixel, otherwise the
    tolerance will be with repsect to adjacent pixels.

    **PARAMETERS**

    * *points* - A tuple, list of tuples, or np.array of seed points for
     flood fill
    * *tolerance* - The color tolerance as a single value or a triplet.
    * *color* - The color to replace the flood_fill pixels with
    * *lower* - If tolerance does not provide enough control you can
      optionally set the upper and lower values
      around the seed pixel. This value can be a single value or a triplet.
      This will override the tolerance variable.
    * *upper* - If tolerance does not provide enough control you can
      optionally set the upper and lower values around the seed pixel. This
      value can be a single value or a triplet. This will override the
      tolerance variable.
    * *fixed_range* - If fixed_range is true we use the seed_pixel +/-
      tolerance. If fixed_range is false, the tolerance is +/- tolerance of
      the values of the adjacent pixels to the pixel under test.

    **RETURNS**

    An Image where the values similar to the seed pixel have been replaced
    by the input color.

    **EXAMPLE**

    >>> img = Image("lenna")
    >>> img2 = img.flood_fill(((10, 10), (54, 32)), tolerance=(10, 10, 10),
    ...                       color=Color.RED)
    >>> img2.show()

    **SEE ALSO**

    :py:meth:`flood_fill_to_mask`
    :py:meth:`find_flood_fill_blobs`

    """
    if isinstance(color, np.ndarray):
        color = color.tolist()
    elif isinstance(color, dict):
        color = (color['R'], color['G'], color['B'])

    if isinstance(points, tuple):
        points = np.array(points)
    # first we guess what the user wants to do
    # if we get and int/float convert it to a tuple
    if upper is None and lower is None and tolerance is None:
        upper = (0, 0, 0)
        lower = (0, 0, 0)

    if tolerance is not None and isinstance(tolerance, (float, int)):
        tolerance = (int(tolerance), int(tolerance), int(tolerance))

    if lower is not None and isinstance(lower, (float, int)):
        lower = (int(lower), int(lower), int(lower))
    elif lower is None:
        lower = tolerance

    if upper is not None and isinstance(upper, (float, int)):
        upper = (int(upper), int(upper), int(upper))
    elif upper is None:
        upper = tolerance

    if isinstance(points, tuple):
        points = np.array(points)

    flags = 8
    if fixed_range:
        flags |= cv2.FLOODFILL_FIXED_RANGE

    mask = np.zeros((img.height + 2, img.width + 2), dtype=np.uint8)
    array = img.get_ndarray().copy()

    if len(points.shape) != 1:
        for p in points:
            cv2.floodFill(array, mask=mask, seedPoint=tuple(p),
                          newVal=color, loDiff=lower, upDiff=upper,
                          flags=flags)
    else:
        cv2.floodFill(array, mask=mask, seedPoint=tuple(points),
                      newVal=color, loDiff=lower, upDiff=upper, flags=flags)
    return Factory.Image(array)


@image_method
def flood_fill_to_mask(img, points, tolerance=None, color=Color.WHITE,
                       lower=None, upper=None, fixed_range=True,
                       mask=None):
    """
    **SUMMARY**

    flood_fill_to_mask works sorta paint bucket tool in your favorite image
    manipulation program. You select a point (or a list of points), a
    color, and a tolerance, and flood_fill will start at that point,
    looking for pixels within the tolerance from your intial pixel. If the
    pixel is in tolerance, we will convert it to your color, otherwise the
    method will leave the pixel alone. Unlike regular flood_fill,
    flood_fill_to_mask, will return a binary mask of your flood fill
    operation. This is handy if you want to extract blobs from an area, or
    create a selection from a region. The method takes in an optional mask.
    Non-zero values of the mask act to block the flood fill operations.
    This is handy if you want to use an edge image to "stop" the flood fill
    operation within a particular region.

    The method accepts both single values, and triplet tuples for the
    tolerance values. If you require more control over your tolerance you
    can use the upper and lower values. The fixed range parameter let's you
    toggle between setting the tolerance with repect to the seed pixel, and
    using a tolerance that is relative to the adjacent pixels. If
    fixed_range is true the method will set its tolerance with respect to
    the seed pixel, otherwise the tolerance will be with repsect to
    adjacent pixels.

    **PARAMETERS**

    * *points* - A tuple, list of tuples, or np.array of seed points for
      flood fill
    * *tolerance* - The color tolerance as a single value or a triplet.
    * *color* - The color to replace the flood_fill pixels with
    * *lower* - If tolerance does not provide enough control you can
      optionally set the upper and lower values around the seed pixel. This
      value can be a single value or a triplet. This will override
      the tolerance variable.
    * *upper* - If tolerance does not provide enough control you can
      optionally set the upper and lower values around the seed pixel. This
      value can be a single value or a triplet. This will override
      the tolerance variable.
    * *fixed_range* - If fixed_range is true we use the seed_pixel +/-
      tolerance. If fixed_range is false, the tolerance is +/- tolerance of
      the values of the adjacent pixels to the pixel under test.
    * *mask* - An optional mask image that can be used to control the flood
      fill operation. the output of this function will include the mask
      data in the input mask.

    **RETURNS**

    An Image where the values similar to the seed pixel have been replaced
    by the input color.

    **EXAMPLE**

    >>> img = Image("lenna")
    >>> mask = img.edges()
    >>> mask= img.flood_fill_to_mask(((10, 10), (54, 32)),
    ...                              tolerance=(10, 10, 10), mask=mask)
    >>> mask.show

    **SEE ALSO**

    :py:meth:`flood_fill`
    :py:meth:`find_flood_fill_blobs`

    """
    if isinstance(color, np.ndarray):
        color = color.tolist()
    elif isinstance(color, dict):
        color = (color['R'], color['G'], color['B'])

    if isinstance(points, tuple):
        points = np.array(points)

    # first we guess what the user wants to do
    # if we get and int/float convert it to a tuple
    if upper is None and lower is None and tolerance is None:
        upper = (0, 0, 0)
        lower = (0, 0, 0)

    if tolerance is not None and isinstance(tolerance, (float, int)):
        tolerance = (int(tolerance), int(tolerance), int(tolerance))

    if lower is not None and isinstance(lower, (float, int)):
        lower = (int(lower), int(lower), int(lower))
    elif lower is None:
        lower = tolerance

    if upper is not None and isinstance(upper, (float, int)):
        upper = (int(upper), int(upper), int(upper))
    elif upper is None:
        upper = tolerance

    if isinstance(points, tuple):
        points = np.array(points)

    flags = (255 << 8) + 8
    if fixed_range:
        flags |= cv2.FLOODFILL_FIXED_RANGE

    #opencv wants a mask that is slightly larger
    if mask is None:
        local_mask = np.zeros((img.height + 2, img.width + 2),
                              dtype=np.uint8)
    else:
        local_mask = mask.embiggen(size=(img.width + 2, img.height + 2))
        local_mask = local_mask.get_gray_ndarray()

    temp = img.get_ndarray().copy()
    if len(points.shape) != 1:
        for p in points:
            cv2.floodFill(temp, mask=local_mask, seedPoint=tuple(p),
                          newVal=color, loDiff=lower, upDiff=upper,
                          flags=flags)
    else:
        cv2.floodFill(temp, mask=local_mask, seedPoint=tuple(points),
                      newVal=color, loDiff=lower, upDiff=upper, flags=flags)

    ret_val = Factory.Image(local_mask)
    ret_val = ret_val.crop(x=1, y=1, w=img.width, h=img.height)
    return ret_val


@image_method
def get_lightness(img):
    """
    **SUMMARY**

    This method converts the given RGB image to grayscale using the
    Lightness method.

    **Parameters**

    None

    **RETURNS**

    A GrayScale image with values according to the Lightness method

    **EXAMPLE**
    >>> img = Image ('lenna')
    >>> out = img.get_lightness()
    >>> out.show()

    **NOTES**

    Algorithm used: value = (MAX(R,G,B) + MIN(R,G,B))/2

    """
    if not img.is_bgr():
        logger.warnings('Input a BGR image')
        return None
    img_mat = np.array(img.get_ndarray(), dtype=np.int)
    ret_val = np.array((np.max(img_mat, 2) + np.min(img_mat, 2)) / 2,
                       dtype=np.uint8)
    return Factory.Image(ret_val)


@image_method
def get_luminosity(img):
    """
    **SUMMARY**

    This method converts the given RGB image to grayscale using the
    Luminosity method.

    **Parameters**

    None

    **RETURNS**

    A GrayScale image with values according to the Luminosity method

    **EXAMPLE**
    >>> img = Image ('lenna')
    >>> out = img.get_luminosity()
    >>> out.show()

    **NOTES**

    Algorithm used: value =  0.21 R + 0.71 G + 0.07 B

    """
    if not img.is_bgr():
        logger.warnings('Input a BGR image')
        return None
    img_mat = np.array(img.get_ndarray(), dtype=np.int)
    ret_val = np.array(np.average(img_mat, 2, (0.07, 0.71, 0.21)),
                       dtype=np.uint8)
    return Factory.Image(ret_val)


@image_method
def get_average(img):
    """
    **SUMMARY**

    This method converts the given RGB image to grayscale by averaging out
    the R,G,B values.

    **Parameters**

    None

    **RETURNS**

    A GrayScale image with values according to the Average method

    **EXAMPLE**
    >>> img = Image ('lenna')
    >>> out = img.get_average()
    >>> out.show()

    **NOTES**

    Algorithm used: value =  (R+G+B)/3

    """
    if not img.is_bgr():
        logger.warnings('Input a BGR image')
        return None
    img_mat = np.array(img.get_ndarray(), dtype=np.int)
    ret_val = np.array(img_mat.mean(2), dtype=np.uint8)
    return Factory.Image(ret_val)


@image_method
def pixelize(img, block_size=10, region=None, levels=None, do_hue=False):
    """
    **SUMMARY**

    Pixelation blur, like the kind used to hide naughty bits on your
    favorite tv show.

    **PARAMETERS**

    * *block_size* - the blur block size in pixels, an integer is an square
       blur, a tuple is rectangular.
    * *region* - do the blur in a region in format (x_position, y_position,
      width, height)
    * *levels* - the number of levels per color channel. This makes the
      image look like an 8-bit video game.
    * *do_hue* - If this value is true we calculate the peak hue for the
      area, not the average color for the area.

    **RETURNS**

    Returns the image with the pixelation blur applied.

    **EXAMPLE**

    >>> img = Image("lenna")
    >>> result = img.pixelize(16, (200, 180, 250, 250), levels=4)
    >>> img.show()

    """

    if isinstance(block_size, int):
        block_size = (block_size, block_size)

    ret_val = img.get_empty()

    levels_f = 0.00
    if levels is not None:
        levels = 255 / int(levels)
        if levels <= 1:
            levels = 2
        levels_f = float(levels)

    if region is not None:
        xs = region[0]
        ys = region[1]
        w = region[2]
        h = region[3]
        ret_val = img.get_ndarray().copy()
        ret_val[ys:ys + w, xs:xs + h] = 0
    else:
        xs = 0
        ys = 0
        w = img.width
        h = img.height

    #if( region is None ):
    hc = w / block_size[0]  # number of horizontal blocks
    vc = h / block_size[1]  # number of vertical blocks
    #when we fit in the blocks, we're going to spread the round off
    #over the edges 0->x_0, 0->y_0  and x_0+hc*block_size
    x_lhs = int(np.ceil(
        float(w % block_size[0]) / 2.0))  # this is the starting point
    y_lhs = int(np.ceil(float(h % block_size[1]) / 2.0))
    x_rhs = int(np.floor(
        float(w % block_size[0]) / 2.0))  # this is the starting point
    y_rhs = int(np.floor(float(h % block_size[1]) / 2.0))
    x_0 = xs + x_lhs
    y_0 = ys + y_lhs
    x_f = (x_0 + (block_size[0] * hc))  # this would be the end point
    y_f = (y_0 + (block_size[1] * vc))

    for i in range(0, hc):
        for j in range(0, vc):
            xt = x_0 + (block_size[0] * i)
            yt = y_0 + (block_size[1] * j)
            roi = (xt, yt, block_size[0], block_size[1])
            img._copy_avg(img, ret_val, roi, levels, levels_f, do_hue)

    if x_lhs > 0:  # add a left strip
        xt = xs
        wt = x_lhs
        ht = block_size[1]
        for j in range(0, vc):
            yt = y_0 + (j * block_size[1])
            roi = (xt, yt, wt, ht)
            img._copy_avg(img, ret_val, roi, levels, levels_f, do_hue)

    if x_rhs > 0:  # add a right strip
        xt = (x_0 + (block_size[0] * hc))
        wt = x_rhs
        ht = block_size[1]
        for j in range(0, vc):
            yt = y_0 + (j * block_size[1])
            roi = (xt, yt, wt, ht)
            img._copy_avg(img, ret_val, roi, levels, levels_f, do_hue)

    if y_lhs > 0:  # add a left strip
        yt = ys
        ht = y_lhs
        wt = block_size[0]
        for i in range(0, hc):
            xt = x_0 + (i * block_size[0])
            roi = (xt, yt, wt, ht)
            img._copy_avg(img, ret_val, roi, levels, levels_f, do_hue)

    if y_rhs > 0:  # add a right strip
        yt = (y_0 + (block_size[1] * vc))
        ht = y_rhs
        wt = block_size[0]
        for i in range(0, hc):
            xt = x_0 + (i * block_size[0])
            roi = (xt, yt, wt, ht)
            img._copy_avg(img, ret_val, roi, levels, levels_f, do_hue)

    #now the corner cases
    if x_lhs > 0 and y_lhs > 0:
        roi = (xs, ys, x_lhs, y_lhs)
        img._copy_avg(img, ret_val, roi, levels, levels_f, do_hue)

    if x_rhs > 0 and y_rhs > 0:
        roi = (x_f, y_f, x_rhs, y_rhs)
        img._copy_avg(img, ret_val, roi, levels, levels_f, do_hue)

    if x_lhs > 0 and y_rhs > 0:
        roi = (xs, y_f, x_lhs, y_rhs)
        img._copy_avg(img, ret_val, roi, levels, levels_f, do_hue)

    if x_rhs > 0 and y_lhs > 0:
        roi = (x_f, ys, x_rhs, y_lhs)
        img._copy_avg(img, ret_val, roi, levels, levels_f, do_hue)

    if do_hue:
        ret_val = cv2.cvtColor(ret_val, code=cv2.COLOR_HSV2BGR)

    return Factory.Image(ret_val)


@image_method
def normalize(img, new_min=0, new_max=255, min_cut=2, max_cut=98):
    """
    **SUMMARY**

    Performs image normalization and yeilds a linearly normalized gray
    image. Also known as contrast strestching.

    see : http://en.wikipedia.org/wiki/Normalization_(image_processing)

    **Parameters**

    * *new_min* - The minimum of the new range over which the image is
    normalized

    * *new_max* - The maximum of the new range over which the image is
    normalized

    * *min_cut* - A number between 0 to 100. The threshold percentage
    for the current minimum value selection. This helps us to avoid the
    effect of outlying pixel with either very low value

    * *max_cut* - A number between 0 to 100. The threshold percentage for
    the current minimum value selection. This helps us to avoid the effect
    of outlying pixel with either very low value

    **RETURNS**

    A normalized grayscale image.

    **EXAMPLE**
    >>> img = Image('lenna')
    >>> norm = img.normalize()
    >>> norm.show()

    """
    if new_min < 0 or new_max > 255:
        logger.warn("new_min and new_max can vary from 0-255")
        return None
    if new_max < new_min:
        logger.warn("new_min should be less than new_max")
        return None
    if min_cut > 100 or max_cut > 100:
        logger.warn("min_cut and max_cut")
        return None

    # avoiding the effect of odd pixels
    try:
        hist = img.get_gray_histogram_counts()
        freq, val = zip(*hist)
        maxfreq = (freq[0] - freq[-1]) * max_cut / 100.0
        minfreq = (freq[0] - freq[-1]) * min_cut / 100.0
        closest_match = lambda a, l: min(l, key=lambda x: abs(x - a))
        maxval = closest_match(maxfreq, val)
        minval = closest_match(minfreq, val)
        array = img.get_gray_ndarray()
        array = cv2.subtract(array,
                             minval * np.ones(array.shape, np.uint8))
        n = ((new_max - new_min) / float(maxval - minval)) \
            * np.ones(array.shape, np.float64)
        array = cv2.multiply(array, n, dtype=cv2.CV_8U)
        array = cv2.add(array, new_min * np.ones(array.shape, np.uint8))
    #catching zero division in case there are very less intensities present
    #Normalizing based on absolute max and min intensities present
    except ZeroDivisionError:
        maxval = img.max_value()
        minval = img.min_value()
        array = img.get_gray_ndarray()
        array = cv2.subtract(array,
                             minval * np.ones(array.shape, np.uint8))
        n = ((new_max - new_min) / float(maxval - minval)) \
            * np.ones(array.shape, np.float64)
        array = cv2.multiply(array, n, dtype=cv2.CV_8U)
        array = cv2.add(array, new_min * np.ones(array.shape, np.uint8))
    return Factory.Image(array.astype(np.uint8))


@image_method
def sobel(img, xorder=1, yorder=1, do_gray=True, aperture=5):
    """
    **DESCRIPTION**

    Sobel operator for edge detection

    **PARAMETERS**

    * *xorder* - int - Order of the derivative x.
    * *yorder* - int - Order of the derivative y.
    * *do_gray* - Bool - grayscale or not.
    * *aperture* - int - Size of the extended Sobel kernel. It must be 1,
      3, 5, or 7.

    **RETURNS**

    Image with sobel opeartor applied on it

    **EXAMPLE**

    >>> img = Image("lenna")
    >>> s = img.sobel()
    >>> s.show()
    """
    if aperture not in [1, 3, 5, 7]:
        logger.warning("Bad Sobel Aperture, values are [1, 3, 5, 7].")
        return None

    if do_gray:
        dst = cv2.Sobel(img.get_gray_ndarray(), cv2.CV_32F, xorder,
                        yorder, ksize=aperture)
        minv = np.min(dst)
        maxv = np.max(dst)
        cscale = 255 / (maxv - minv)
        shift = -1 * minv

        t = cv2.convertScaleAbs(dst, alpha=cscale, beta=shift / 255.0)
        ret_val = Factory.Image(t)

    else:
        layers = img.split_channels(grayscale=False)
        sobel_layers = []
        for layer in layers:
            dst = cv2.Sobel(layer.get_gray_numpy(), cv2.CV_32F, xorder,
                            yorder, ksize=aperture)

            minv = np.min(dst)
            maxv = np.max(dst)
            cscale = 255 / (maxv - minv)
            shift = -1 * minv

            t = cv2.convertScaleAbs(dst, alpha=cscale, beta=shift / 255.0)
            sobel_layers.append(Factory.Image(t))
        b, g, r = sobel_layers

        ret_val = img.merge_channels(c1=b, c2=g, c3=r)
    return ret_val


@image_method
def watershed(img, mask=None, erode=2, dilate=2, use_my_mask=False):
    """
    **SUMMARY**

    Implements the Watershed algorithm on the input image.

    Read more:

    Watershed: "http://en.wikipedia.org/wiki/Watershed_(image_processing)"

    **PARAMETERS**

    * *mask* - an optional binary mask. If none is provided we do a
      binarize and invert.
    * *erode* - the number of times to erode the mask to find the
      foreground.
    * *dilate* - the number of times to dilate the mask to find possible
      background.
    * *use_my_mask* - if this is true we do not modify the mask.

    **RETURNS**

    The Watershed image

    **EXAMPLE**

    >>> img = Image("/data/sampleimages/wshed.jpg")
    >>> img1 = img.watershed()
    >>> img1.show()

    # here is an example of how to create your own mask

    >>> img = Image('lenna')
    >>> myMask = Image((img.width, img.height))
    >>> myMask = myMask.flood_fill((0, 0), color=Color.WATERSHED_BG)
    >>> mask = img.threshold(128)
    >>> myMask = (myMask - mask.dilate(2) + mask.erode(2))
    >>> result = img.watershed(mask=myMask, use_my_mask=True)

    **SEE ALSO**
    Color.WATERSHED_FG - The watershed foreground color
    Color.WATERSHED_BG - The watershed background color
    Color.WATERSHED_UNSURE - The watershed not sure if fg or bg color.

    TODO: Allow the user to pass in a function that defines the watershed
    mask.
    """
    if mask is None:
        mask = img.binarize().invert()
    if not use_my_mask:
        newmask = Factory.Image((img.width, img.height))
        newmask = newmask.flood_fill((0, 0), color=Color.WATERSHED_BG)
        newmask = newmask - mask.dilate(iterations=dilate).to_bgr()
        newmask = newmask + mask.erode(iterations=erode).to_bgr()
    else:
        newmask = mask
    m = np.int32(newmask.get_gray_ndarray())
    cv2.watershed(img.get_ndarray(), m)
    m = cv2.convertScaleAbs(m)
    ret, thresh = cv2.threshold(m, 0, 255, cv2.THRESH_OTSU)
    return Factory.Image(thresh)


# FIXME: following functions should be merged
@image_method
def motion_blur(img, intensity=15, direction='NW'):
    """
    **SUMMARY**

    Performs the motion blur of an Image. Uses different filters to find
    out the motion blur in different directions.

    see : https://en.wikipedia.org/wiki/Motion_blur

    **Parameters**

    * *intensity* - The intensity of the motion blur effect. Basically
       defines the size of the filter used in the process. It has to be an
       integer. 0 intensity implies no blurring.

    * *direction* - The direction of the motion. It is a string taking
        values left, right, up, down as well as N, S, E, W for north,
        south, east, west and NW, NE, SW, SE for northwest and so on.
        default is NW

    **RETURNS**

    An image with the specified motion blur filter applied.

    **EXAMPLE**
    >>> i = Image ('lenna')
    >>> mb = i.motion_blur()
    >>> mb.show()

    """
    mid = int(intensity / 2)
    tmp = np.identity(intensity)

    if intensity == 0:
        logger.warn("0 intensity means no blurring")
        return img

    elif intensity % 2 is 0:
        div = mid
        for i in range(mid, intensity - 1):
            tmp[i][i] = 0
    else:
        div = mid + 1
        for i in range(mid + 1, intensity - 1):
            tmp[i][i] = 0

    if direction == 'right' or direction.upper() == 'E':
        kernel = np.concatenate(
            (np.zeros((1, mid)), np.ones((1, mid + 1))), axis=1)
    elif direction == 'left' or direction.upper() == 'W':
        kernel = np.concatenate(
            (np.ones((1, mid + 1)), np.zeros((1, mid))), axis=1)
    elif direction == 'up' or direction.upper() == 'N':
        kernel = np.concatenate(
            (np.ones((1 + mid, 1)), np.zeros((mid, 1))), axis=0)
    elif direction == 'down' or direction.upper() == 'S':
        kernel = np.concatenate(
            (np.zeros((mid, 1)), np.ones((mid + 1, 1))), axis=0)
    elif direction.upper() == 'NW':
        kernel = tmp
    elif direction.upper() == 'NE':
        kernel = np.fliplr(tmp)
    elif direction.upper() == 'SW':
        kernel = np.flipud(tmp)
    elif direction.upper() == 'SE':
        kernel = np.flipud(np.fliplr(tmp))
    else:
        logger.warn("Please enter a proper direction")
        return None

    retval = img.convolve(kernel=kernel / div)
    return retval


@image_method
def motion_blur2(img, intensity=15, angle=0):
    """
    **SUMMARY**

    Performs the motion blur of an Image given the intensity and angle

    see : https://en.wikipedia.org/wiki/Motion_blur

    **Parameters**

    * *intensity* - The intensity of the motion blur effect. Governs the
        size of the kernel used in convolution

    * *angle* - Angle in degrees at which motion blur will occur. Positive
        is Clockwise and negative is Anti-Clockwise. 0 blurs from left to
        right


    **RETURNS**

    An image with the specified motion blur applied.

    **EXAMPLE**
    >>> img = Image ('lenna')
    >>> blur = img.motion_blur(40, 45)
    >>> blur.show()

    """

    intensity = int(intensity)

    if intensity <= 1:
        logger.warning('power less than 1 will result in no change')
        return img

    kernel = np.zeros((intensity, intensity))

    rad = math.radians(angle)
    x1, y1 = intensity / 2, intensity / 2

    x2 = int(x1 - (intensity - 1) / 2 * math.sin(rad))
    y2 = int(y1 - (intensity - 1) / 2 * math.cos(rad))

    line = img.bresenham_line((x1, y1), (x2, y2))

    x = [p[0] for p in line]
    y = [p[1] for p in line]

    kernel[x, y] = 1
    kernel = kernel / len(line)
    return img.convolve(kernel=kernel)


@image_method
def channel_mixer(img, channel='r', weight=(100, 100, 100)):
    """
    **SUMMARY**

    Mixes channel of an RGB image based on the weights provided. The output
    is given at the channel provided in the parameters. Basically alters
    the value of one channelg of an RGB image based in the values of other
    channels and itself. If the image is not RGB then first converts the
    image to RGB and then mixes channel

    **PARAMETERS**

    * *channel* - The output channel in which the values are to be
    replaced. It can have either 'r' or 'g' or 'b'

    * *weight* - The weight of each channel in calculation of the mixed
    channel. It is a tuple having 3 values mentioning the percentage of the
    value of the channels, from -200% to 200%

    **RETURNS**

    A SimpleCV RGB Image with the provided channel replaced with the mixed
    channel.

    **EXAMPLE**

    >>> img = Image("lenna")
    >>> img2 = img.channel_mixer()
    >>> Img3 = img.channel_mixer(channel='g', weights=(3, 2, 1))

    **NOTE**

    Read more at http://docs.gimp.org/en/plug-in-colors-channel-mixer.html

    """
    r, g, b = img.split_channels()
    if weight[0] > 200 or weight[1] > 200 or weight[2] >= 200:
        if weight[0] < -200 or weight[1] < -200 or weight[2] < -200:
            logger.warn('Value of weights can be from -200 to 200%')
            return None

    weight = map(float, weight)
    channel = channel.lower()
    if channel == 'r':
        r = r * (weight[0] / 100.0) + \
            g * (weight[1] / 100.0) + \
            b * (weight[2] / 100.0)
    elif channel == 'g':
        g = r * (weight[0] / 100.0) + \
            g * (weight[1] / 100.0) + \
            b * (weight[2] / 100.0)
    elif channel == 'b':
        b = r * (weight[0] / 100.0) + \
            g * (weight[1] / 100.0) + \
            b * (weight[2] / 100.0)
    else:
        logger.warn('Please enter a valid channel(r/g/b)')
        return None

    ret_val = img.merge_channels(c1=r, c2=g, c3=b)
    return ret_val


@image_method
def prewitt(img):
    """
    **SUMMARY**

    Prewitt operator for edge detection

    **PARAMETERS**

    None

    **RETURNS**

    Image with prewitt opeartor applied on it

    **EXAMPLE**

    >>> img = Image("lenna")
    >>> p = img.prewitt()
    >>> p.show()

    **NOTES**

    Read more at: http://en.wikipedia.org/wiki/Prewitt_operator

    """
    grayimg = img.to_gray()
    gx = [[1, 1, 1], [0, 0, 0], [-1, -1, -1]]
    gy = [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]
    grayx = grayimg.convolve(gx)
    grayy = grayimg.convolve(gy)
    grayxnp = np.uint64(grayx.get_gray_ndarray())
    grayynp = np.uint64(grayy.get_gray_ndarray())
    ret_val = Factory.Image(np.sqrt(grayxnp ** 2 + grayynp ** 2))
    return ret_val
