import cv2
import numpy as np

from simplecv.base import logger
from simplecv.core.image import (image_method, cached_method,
                                 static_image_method)
from simplecv.factory import Factory


@image_method
@cached_method
def do_dft(img, grayscale=False):
    """
    **SUMMARY**

    This private method peforms the discrete Fourier transform on an input
    image. The transform can be applied to a single channel gray image or
    to each channel of the image. Each channel generates a 64F 2 channel
    IPL image corresponding to the real and imaginary components of the
    DFT. A list of these IPL images are then cached in the private member
    variable _dft.


    **PARAMETERS**

    * *grayscale* - If grayscale is True we first covert the image to
      grayscale, otherwise we perform the operation on each channel.

    **RETURNS**

    nothing - but creates a locally cached list of IPL imgaes corresponding
    to the real and imaginary components of each channel.

    **EXAMPLE**

    >>> img = Image('logo.png')
    >>> dft = img._do_dft()
    >>> dft[0] # get the b channel Re/Im components

    **NOTES**

    http://en.wikipedia.org/wiki/Discrete_Fourier_transform
    http://math.stackexchange.com/questions/1002/
    fourier-transform-for-dummies

    **TO DO**

    This method really needs to convert the image to an optimal DFT size.
    http://opencv.itseez.com/modules/core/doc/
    operations_on_arrays.html#getoptimaldftsize

    """
    width, height = img.size_tuple
    dft = []
    if grayscale:
        img_array = img.to_gray()
        data = img_array.astype(np.float64)
        blank = np.zeros((height, width))
        src = np.dstack((data, blank))
        dst = cv2.dft(src)
        dft.append(dst)
    else:
        img_array = img.copy()
        if len(img_array.shape) == 3:
            b = img_array[:, :, 0]
            g = img_array[:, :, 1]
            r = img_array[:, :, 2]
            chanels = [b, g, r]
        else:
            chanels = [img_array, img_array, img_array]
        for c in chanels:
            data = c.astype(np.float64)
            blank = np.zeros((height, width))
            src = np.dstack((data, blank))
            dst = cv2.dft(src)
            dft.append(dst)
    return dft


@image_method
def _get_dft_clone(img, grayscale=False):
    """
    **SUMMARY**

    This method works just like _do_dft but returns a deep copy
    of the resulting array which can be used in destructive operations.

    **PARAMETERS**

    * *grayscale* - If grayscale is True we first covert the image to
      grayscale, otherwise we perform the operation on each channel.

    **RETURNS**

    A deep copy of the cached DFT real/imaginary image list.

    **EXAMPLE**

    >>> img = Image('logo.png')
    >>> myDFT = img._get_dft_clone()
    >>> SomeCVFunc(myDFT[0])

    **NOTES**

    http://en.wikipedia.org/wiki/Discrete_Fourier_transform
    http://math.stackexchange.com/questions/1002/
    fourier-transform-for-dummies

    **SEE ALSO**

    ImageClass._do_dft()

    """
    # this is needs to be switched to the optimal
    # DFT size for faster processing.
    dft = img.do_dft(grayscale)
    return [dft_img.copy() for dft_img in dft]


@image_method
def raw_dft_image(img, grayscale=False):
    """
    **SUMMARY**

    This method returns the **RAW** DFT transform of an image as a list of
    IPL Images. Each result image is a two channel 64f image where the
    irst channel is the real component and the second channel is the
    imaginary component. If the operation is performed on an RGB image and
    grayscale is False the result is a list of these images of the form
    [b, g, r].

    **PARAMETERS**

    * *grayscale* - If grayscale is True we first covert the image to
      grayscale, otherwise we perform the operation on each channel.

    **RETURNS**

    A list of the DFT images (see above). Note that this is a shallow copy
    operation.

    **EXAMPLE**

    >>> img = Image('logo.png')
    >>> myDFT = img.raw_dft_image()
    >>> for c in myDFT:
    >>>    #do some operation on the DFT

    **NOTES**

    http://en.wikipedia.org/wiki/Discrete_Fourier_transform
    http://math.stackexchange.com/questions/1002/
    fourier-transform-for-dummies

    **SEE ALSO**

    :py:meth:`raw_dft_image`
    :py:meth:`get_dft_log_magnitude`
    :py:meth:`apply_dft_filter`
    :py:meth:`high_pass_filter`
    :py:meth:`low_pass_filter`
    :py:meth:`band_pass_filter`
    :py:meth:`inverse_dft`
    :py:meth:`apply_butterworth_filter`
    :py:meth:`inverse_dft`
    :py:meth:`apply_gaussian_filter`
    :py:meth:`apply_unsharp_mask`

    """
    return img.do_dft(grayscale)


@image_method
def get_dft_log_magnitude(img, grayscale=False):
    """
    **SUMMARY**

    This method returns the log value of the magnitude image of the DFT
    transform. This method is helpful for examining and comparing the
    results of DFT transforms. The log component helps to "squish" the
    large floating point values into an image that can be rendered easily.

    In the image the low frequency components are in the corners of the
    image and the high frequency components are in the center of the image.

    **PARAMETERS**

    * *grayscale* - if grayscale is True we perform the magnitude operation
       of the grayscale image otherwise we perform the operation on each
       channel.

    **RETURNS**

    Returns a SimpleCV image corresponding to the log magnitude of the
    input image.

    **EXAMPLE**

    >>> img = Image("RedDog2.jpg")
    >>> img.get_dft_log_magnitude().show()
    >>> lpf = img.low_pass_filter(img.width/10.img.height/10)
    >>> lpf.get_dft_log_magnitude().show()

    **NOTES**

    * http://en.wikipedia.org/wiki/Discrete_Fourier_transform
    * http://math.stackexchange.com/questions/1002/
      fourier-transform-for-dummies

    **SEE ALSO**

    :py:meth:`raw_dft_image`
    :py:meth:`get_dft_log_magnitude`
    :py:meth:`apply_dft_filter`

    :py:meth:`high_pass_filter`
    :py:meth:`low_pass_filter`
    :py:meth:`band_pass_filter`
    :py:meth:`inverse_dft`
    :py:meth:`apply_butterworth_filter`
    :py:meth:`inverse_dft`
    :py:meth:`apply_gaussian_filter`
    :py:meth:`apply_unsharp_mask`


    """
    dft = img._get_dft_clone(grayscale)
    if grayscale:
        chans = [img.get_empty(1)]
    else:
        chans = [img.get_empty(1), img.get_empty(1), img.get_empty(1)]

    for i in range(0, len(chans)):
        data = dft[i][:, :, 0]
        blank = dft[i][:, :, 1]
        data = cv2.pow(data, 2.0)
        blank = cv2.pow(blank, 2.0)
        data += blank
        data = cv2.pow(data, 0.5)
        data += 1  # 1 + Mag
        data = cv2.log(data)  # log(1 + Mag)
        min_val, max_val, pt1, pt2 = cv2.minMaxLoc(data)
        denom = max_val - min_val
        if denom == 0:
            denom = 1
        data = data / denom - min_val / denom  # scale
        data = cv2.multiply(data, data, scale=255.0)
        chans[i] = np.copy(data).astype(img.dtype)
    if grayscale:
        ret_val = Factory.Image(chans[0])
    else:
        ret_val = Factory.Image(np.dstack(tuple(chans)))
    return ret_val


@image_method
def apply_dft_filter(img, flt, grayscale=False):
    """
    **SUMMARY**

    This function allows you to apply an arbitrary filter to the DFT of an
    image. This filter takes in a gray scale image, whiter values are kept
    and black values are rejected. In the DFT image, the lower frequency
    values are in the corners of the image, while the higher frequency
    components are in the center. For example, a low pass filter has white
    squares in the corners and is black everywhere else.

    **PARAMETERS**

    * *grayscale* - if this value is True we perfrom the operation on the
      DFT of the gray version of the image and the result is gray image.
      If grayscale is true we perform the operation on each channel and the
      recombine them to create the result.

    * *flt* - A grayscale filter image. The size of the filter must match
      the size of the image.

    **RETURNS**

    A SimpleCV image after applying the filter.

    **EXAMPLE**

    >>>  filter = Image("MyFilter.png")
    >>>  myImage = Image("MyImage.png")
    >>>  result = myImage.apply_dft_filter(filter)
    >>>  result.show()

    **SEE ALSO**

    :py:meth:`raw_dft_image`
    :py:meth:`get_dft_log_magnitude`
    :py:meth:`apply_dft_filter`
    :py:meth:`high_pass_filter`
    :py:meth:`low_pass_filter`
    :py:meth:`band_pass_filter`
    :py:meth:`inverse_dft`
    :py:meth:`apply_butterworth_filter`
    :py:meth:`inverse_dft`
    :py:meth:`apply_gaussian_filter`
    :py:meth:`apply_unsharp_mask`

    **TODO**

    Make this function support a separate filter image for each channel.
    """
    if isinstance(flt, Factory.DFT):
        filteredimage = flt.apply_filter(img, grayscale)
        return filteredimage

    if flt.size_tuple != img.size_tuple:
        logger.warning("Image.apply_dft_filter - Your filter must match "
                       "the size of the image")
        return None
    dft = img._get_dft_clone(grayscale)
    if grayscale:
        flt64f = flt.to_gray().astype(np.float64)
        final_filt = np.dstack((flt64f, flt64f))
        for i in range(len(dft)):
            dft[i] = cv2.mulSpectrums(dft[i], final_filt, flags=0)
    else:  # break down the filter and then do each channel
        if len(flt.shape) == 3:
            b = flt[:, :, 0]
            g = flt[:, :, 1]
            r = flt[:, :, 2]
            chans = [b, g, r]
        else:
            chans = [flt, flt, flt]
        for i in range(0, len(chans)):
            flt64f = np.copy(chans[i])
            final_filt = np.dstack((flt64f, flt64f))
            if dft[i].dtype != final_filt.dtype:
                final_filt = final_filt.astype(dft[i].dtype)
            dft[i] = cv2.mulSpectrums(dft[i], final_filt, flags=0)
    return img._inverse_dft(dft)


@image_method
def high_pass_filter(img, x_cutoff, y_cutoff=None, grayscale=False):
    """
    **SUMMARY**

    This method applies a high pass DFT filter. This filter enhances
    the high frequencies and removes the low frequency signals. This has
    the effect of enhancing edges. The frequencies are defined as going
    between 0.00 and 1.00 and where 0 is the lowest frequency in the image
    and 1.0 is the highest possible frequencies. Each of the frequencies
    are defined with respect to the horizontal and vertical signal. This
    filter isn't perfect and has a harsh cutoff that causes ringing
    artifacts.

    **PARAMETERS**

    * *x_cutoff* - The horizontal frequency at which we perform the cutoff.
      A separate frequency can be used for the b,g, and r signals by
      providing list of values. The frequency is defined between zero to
      one, where zero is constant component and 1 is the highest possible
      frequency in the image.

    * *y_cutoff* - The cutoff frequencies in the y direction. If none are
      provided we use the same values as provided for x.

    * *grayscale* - if this value is True we perfrom the operation on the
      DFT of the gray version of the image and the result is gray image.
      If grayscale is true we perform the operation on each channel and
      the recombine them to create the result.

    **RETURNS**

    A SimpleCV Image after applying the filter.

    **EXAMPLE**

    >>> img = Image("SimpleCV/data/sampleimages/RedDog2.jpg")
    >>> img.get_dft_log_magnitude().show()
    >>> hpf = img.high_pass_filter([0.2, 0.1, 0.2])
    >>> hpf.show()
    >>> hpf.get_dft_log_magnitude().show()

    **NOTES**

    This filter is far from perfect and will generate a lot of ringing
    artifacts.

    * See: http://en.wikipedia.org/wiki/Ringing_(signal)
    * See: http://en.wikipedia.org/wiki/High-pass_filter#Image

    **SEE ALSO**

    :py:meth:`raw_dft_image`
    :py:meth:`get_dft_log_magnitude`
    :py:meth:`apply_dft_filter`
    :py:meth:`high_pass_filter`
    :py:meth:`low_pass_filter`
    :py:meth:`band_pass_filter`
    :py:meth:`inverse_dft`
    :py:meth:`apply_butterworth_filter`
    :py:meth:`inverse_dft`
    :py:meth:`apply_gaussian_filter`
    :py:meth:`apply_unsharp_mask`

    """
    if isinstance(x_cutoff, float):
        x_cutoff = [x_cutoff, x_cutoff, x_cutoff]
    if isinstance(y_cutoff, float):
        y_cutoff = [y_cutoff, y_cutoff, y_cutoff]
    if y_cutoff is None:
        y_cutoff = [x_cutoff[0], x_cutoff[1], x_cutoff[2]]

    for i in range(0, len(x_cutoff)):
        x_cutoff[i] = img._bounds_from_percentage(x_cutoff[i], img.width)
        y_cutoff[i] = img._bounds_from_percentage(y_cutoff[i],
                                                  img.height)

    filter = None
    h = img.height
    w = img.width

    if grayscale:
        filter = img.get_empty(1)
        filter += 255  # make everything white

        # now make all of the corners black
        cv2.rectangle(filter, pt1=(0, 0), pt2=(int(x_cutoff[0]),
                                               int(y_cutoff[0])),
                      color=0, thickness=-1)  # TL
        cv2.rectangle(filter, pt1=(0, int(h - y_cutoff[0])),
                      pt2=(int(x_cutoff[0]), int(h)),
                      color=0, thickness=-1)  # BL
        cv2.rectangle(filter, pt1=(int(w - x_cutoff[0]), 0),
                      pt2=(int(w), int(y_cutoff[0])),
                      color=0, thickness=-1)  # TR
        cv2.rectangle(filter, pt1=(int(w - x_cutoff[0]),
                                   int(h - y_cutoff[0])),
                      pt2=(int(w), int(h)), color=0, thickness=-1)  # BR

        scv_filt = Factory.Image(filter)
    else:
        # I need to looking into CVMERGE/SPLIT... I would really
        # need to know how much memory we're allocating here
        filter_b = img.get_empty(1) + 255  # make everything white
        filter_g = img.get_empty(1) + 255  # make everything white
        filter_r = img.get_empty(1) + 255  # make everything white

        # now make all of the corners black
        temp = [filter_b, filter_g, filter_r]
        i = 0
        for f in temp:
            cv2.rectangle(f, pt1=(0, 0), pt2=(int(x_cutoff[i]),
                                              int(y_cutoff[i])),
                          color=0, thickness=-1)
            cv2.rectangle(f, pt1=(0, int(h - y_cutoff[i])),
                          pt2=(int(x_cutoff[i]), int(h)),
                          color=0, thickness=-1)
            cv2.rectangle(f, pt1=(int(w - x_cutoff[i]), 0),
                          pt2=(int(w), int(y_cutoff[i])),
                          color=0, thickness=-1)
            cv2.rectangle(f, pt1=(int(w - x_cutoff[i]),
                                  int(h - y_cutoff[i])),
                          pt2=(int(w), int(h)), color=0, thickness=-1)
            i = i + 1

        filter = np.dstack(tuple(temp))
        scv_filt = Factory.Image(filter, color_space=Factory.Image.BGR)

    return img.apply_dft_filter(scv_filt, grayscale)


@image_method
def low_pass_filter(img, x_cutoff, y_cutoff=None, grayscale=False):
    """
    **SUMMARY**

    This method applies a low pass DFT filter. This filter enhances
    the low frequencies and removes the high frequency signals. This has
    the effect of reducing noise. The frequencies are defined as going
    between 0.00 and 1.00 and where 0 is the lowest frequency in the image
    and 1.0 is the highest possible frequencies. Each of the frequencies
    are defined with respect to the horizontal and vertical signal. This
    filter isn't perfect and has a harsh cutoff that causes ringing
    artifacts.

    **PARAMETERS**

    * *x_cutoff* - The horizontal frequency at which we perform the cutoff.
      A separate frequency can be used for the b,g, and r signals by
      providing a list of values. The frequency is defined between zero to
      one, where zero is constant component and 1 is the highest possible
      frequency in the image.

    * *y_cutoff* - The cutoff frequencies in the y direction. If none are
      provided we use the same values as provided for x.

    * *grayscale* - if this value is True we perfrom the operation on the
      DFT of the gray version of the image and the result is gray image.
      If grayscale is true we perform the operation on each channel and the
      recombine them to create the result.

    **RETURNS**

    A SimpleCV Image after applying the filter.

    **EXAMPLE**

    >>> img = Image("SimpleCV/data/sampleimages/RedDog2.jpg")
    >>> img.get_dft_log_magnitude().show()
    >>> lpf = img.low_pass_filter([0.2, 0.2, 0.05])
    >>> lpf.show()
    >>> lpf.get_dft_log_magnitude().show()

    **NOTES**

    This filter is far from perfect and will generate a lot of ringing
    artifacts.

    See: http://en.wikipedia.org/wiki/Ringing_(signal)
    See: http://en.wikipedia.org/wiki/Low-pass_filter

    **SEE ALSO**

    :py:meth:`raw_dft_image`
    :py:meth:`get_dft_log_magnitude`
    :py:meth:`apply_dft_filter`
    :py:meth:`high_pass_filter`
    :py:meth:`low_pass_filter`
    :py:meth:`band_pass_filter`
    :py:meth:`inverse_dft`
    :py:meth:`apply_butterworth_filter`
    :py:meth:`inverse_dft`
    :py:meth:`apply_gaussian_filter`
    :py:meth:`apply_unsharp_mask`

    """
    if isinstance(x_cutoff, float):
        x_cutoff = [x_cutoff, x_cutoff, x_cutoff]
    if isinstance(y_cutoff, float):
        y_cutoff = [y_cutoff, y_cutoff, y_cutoff]
    if y_cutoff is None:
        y_cutoff = [x_cutoff[0], x_cutoff[1], x_cutoff[2]]

    for i in range(0, len(x_cutoff)):
        x_cutoff[i] = img._bounds_from_percentage(x_cutoff[i], img.width)
        y_cutoff[i] = img._bounds_from_percentage(y_cutoff[i], img.height)

    filter = None
    h = img.height
    w = img.width

    if grayscale:
        filter = img.get_empty(1)

        #now make all of the corners white
        cv2.rectangle(filter, pt1=(0, 0), pt2=(x_cutoff[0], y_cutoff[0]),
                      color=255, thickness=-1)  # TL
        cv2.rectangle(filter, pt1=(0, h - y_cutoff[0]), pt2=(x_cutoff[0], h),
                      color=255, thickness=-1)  # BL
        cv2.rectangle(filter, pt1=(w - x_cutoff[0], 0), pt2=(w, y_cutoff[0]),
                      color=255, thickness=-1)  # TR
        cv2.rectangle(filter, pt1=(w - x_cutoff[0], h - y_cutoff[0]),
                      pt2= (w, h), color=255, thickness=-1)  # BR
        scv_filt = Factory.Image(filter)

    else:
        # I need to looking into CVMERGE/SPLIT... I would really need
        # to know how much memory we're allocating here
        filter_b = img.get_empty(1)
        filter_g = img.get_empty(1)
        filter_r = img.get_empty(1)

        # now make all of the corners white
        temp = [filter_b, filter_g, filter_r]
        i = 0
        for f in temp:
            cv2.rectangle(f, (0, 0), (x_cutoff[i], y_cutoff[i]),
                          color=255, thickness=-1)
            cv2.rectangle(f, (0, h - y_cutoff[i]), (x_cutoff[i], h),
                          color=255, thickness=-1)
            cv2.rectangle(f, (w - x_cutoff[i], 0), (w, y_cutoff[i]),
                          color=255, thickness=-1)
            cv2.rectangle(f, (w - x_cutoff[i], h - y_cutoff[i]), (w, h),
                          color=255, thickness=-1)
            i = i + 1

        filter = np.dstack(tuple(temp))
        scv_filt = Factory.Image(filter)

    return img.apply_dft_filter(scv_filt, grayscale)


#FIXME: need to decide BGR or RGB
# ((rx_begin,ry_begin)(gx_begin,gy_begin)(bx_begin,by_begin))
# or (x,y)
@image_method
def band_pass_filter(img, x_cutoff_low, x_cutoff_high, y_cutoff_low=None,
                     y_cutoff_high=None, grayscale=False):
    """
    **SUMMARY**

    This method applies a simple band pass DFT filter. This filter enhances
    the a range of frequencies and removes all of the other frequencies.
    This allows a user to precisely select a set of signals to display.
    The frequencies are defined as going between 0.00 and 1.00 and where 0
    is the lowest frequency in the image and 1.0 is the highest possible
    frequencies. Each of the frequencies are defined with respect to the
    horizontal and vertical signal. This filter isn't perfect and has a
    harsh cutoff that causes ringing artifacts.

    **PARAMETERS**

    * *x_cutoff_low*  - The horizontal frequency at which we perform the
      cutoff of the low frequency signals. A separate frequency can be used
      for the b,g, and r signals by providing a list of values. The
      frequency is defined between zero to one, where zero is constant
      component and 1 is the highest possible frequency in the image.

    * *x_cutoff_high* - The horizontal frequency at which we perform the
      cutoff of the high frequency signals. Our filter passes signals
      between x_cutoff_low and x_cutoff_high. A separate frequency can be
      used for the b, g, and r channels by providing a list of values. The
      frequency is defined between zero to one, where zero is constant
      component and 1 is the highest possible frequency in the image.

    * *y_cutoff_low* - The low frequency cutoff in the y direction. If none
      are provided we use the same values as provided for x.

    * *y_cutoff_high* - The high frequency cutoff in the y direction. If
      none are provided we use the same values as provided for x.

    * *grayscale* - if this value is True we perfrom the operation on the
      DFT of the gray version of the image and the result is gray image.
      If grayscale is true we perform the operation on each channel and
      the recombine them to create the result.

    **RETURNS**

    A SimpleCV Image after applying the filter.

    **EXAMPLE**

    >>> img = Image("SimpleCV/data/sampleimages/RedDog2.jpg")
    >>> img.get_dft_log_magnitude().show()
    >>> lpf = img.band_pass_filter([0.2, 0.2, 0.05],[0.3, 0.3, 0.2])
    >>> lpf.show()
    >>> lpf.get_dft_log_magnitude().show()

    **NOTES**

    This filter is far from perfect and will generate a lot of ringing
    artifacts.

    See: http://en.wikipedia.org/wiki/Ringing_(signal)

    **SEE ALSO**

    :py:meth:`raw_dft_image`
    :py:meth:`get_dft_log_magnitude`
    :py:meth:`apply_dft_filter`
    :py:meth:`high_pass_filter`
    :py:meth:`low_pass_filter`
    :py:meth:`band_pass_filter`
    :py:meth:`inverse_dft`
    :py:meth:`apply_butterworth_filter`
    :py:meth:`inverse_dft`
    :py:meth:`apply_gaussian_filter`
    :py:meth:`apply_unsharp_mask`

    """

    if isinstance(x_cutoff_low, float):
        x_cutoff_low = [x_cutoff_low, x_cutoff_low, x_cutoff_low]
    if isinstance(y_cutoff_low, float):
        y_cutoff_low = [y_cutoff_low, y_cutoff_low, y_cutoff_low]
    if isinstance(x_cutoff_high, float):
        x_cutoff_high = [x_cutoff_high, x_cutoff_high, x_cutoff_high]
    if isinstance(y_cutoff_high, float):
        y_cutoff_high = [y_cutoff_high, y_cutoff_high, y_cutoff_high]

    if y_cutoff_low is None:
        y_cutoff_low = [x_cutoff_low[0], x_cutoff_low[1], x_cutoff_low[2]]
    if y_cutoff_high is None:
        y_cutoff_high = [x_cutoff_high[0], x_cutoff_high[1],
                         x_cutoff_high[2]]

    for i in range(0, len(x_cutoff_low)):
        x_cutoff_low[i] = img._bounds_from_percentage(x_cutoff_low[i],
                                                      img.width)
        x_cutoff_high[i] = img._bounds_from_percentage(x_cutoff_high[i],
                                                       img.width)
        y_cutoff_high[i] = img._bounds_from_percentage(y_cutoff_high[i],
                                                       img.height)
        y_cutoff_low[i] = img._bounds_from_percentage(y_cutoff_low[i],
                                                      img.height)

    filter = None
    h = img.height
    w = img.width
    if grayscale:
        filter = img.get_empty(1)

        # now make all of the corners white
        cv2.rectangle(filter, pt1=(0, 0),
                      pt2=(int(x_cutoff_high[0]), int(y_cutoff_high[0])),
                      color=255, thickness=-1)  # TL
        cv2.rectangle(filter, pt1=(0, int(h - y_cutoff_high[0])),
                      pt2=(int(x_cutoff_high[0]), int(h)),
                      color=255, thickness=-1)  # BL
        cv2.rectangle(filter, pt1=(int(w - x_cutoff_high[0]), 0),
                      pt2=(int(w), int(y_cutoff_high[0])),
                      color=255, thickness=-1)  # TR
        cv2.rectangle(filter, pt1=(int(w - x_cutoff_high[0]),
                                   int(h - y_cutoff_high[0])),
                      pt2=(int(w), int(h)), color=255, thickness=-1)  # BR
        cv2.rectangle(filter, pt1=(0, 0),
                      pt2=(int(x_cutoff_low[0]), int(y_cutoff_low[0])),
                      color=0, thickness=-1)  # TL
        cv2.rectangle(filter, pt1=(0, int(h - y_cutoff_low[0])),
                      pt2=(int(x_cutoff_low[0]), int(h)),
                      color=0, thickness=-1)  # BL
        cv2.rectangle(filter, pt1=(int(w - x_cutoff_low[0]), 0),
                      pt2=(int(w), int(y_cutoff_low[0])),
                      color=0, thickness=-1)  # TR
        cv2.rectangle(filter, pt1=(int(w - x_cutoff_low[0]),
                                   int(h - y_cutoff_low[0])),
                      pt2=(int(w), int(h)),
                      color=0, thickness=-1)  # BR
        scv_filt = Factory.Image(filter)

    else:
        # I need to looking into CVMERGE/SPLIT... I would really need
        # to know how much memory we're allocating here
        filter_b = img.get_empty(1)
        filter_g = img.get_empty(1)
        filter_r = img.get_empty(1)

        #now make all of the corners black
        temp = [filter_b, filter_g, filter_r]
        i = 0
        for f in temp:
            cv2.rectangle(f, pt1=(0, 0),
                          pt2=(int(x_cutoff_high[i]), int(y_cutoff_high[i])),
                          color=255, thickness=-1)  # TL
            cv2.rectangle(f, pt1=(0, int(h - y_cutoff_high[i])),
                          pt2=(int(x_cutoff_high[i]), int(h)),
                          color=255, thickness=-1)  # BL
            cv2.rectangle(f, pt1=(int(w - x_cutoff_high[i]), 0),
                          pt2=(int(w), int(y_cutoff_high[i])),
                          color=255, thickness=-1)  # TR
            cv2.rectangle(f, pt1=(int(w - x_cutoff_high[i]),
                                  int(h - y_cutoff_high[i])),
                          pt2=(int(w), int(h)), color=255, thickness=-1)  # BR
            cv2.rectangle(f, pt1=(0, 0),
                          pt2=(int(x_cutoff_low[i]), int(y_cutoff_low[i])),
                          color=0, thickness=-1)  # TL
            cv2.rectangle(f, pt1=(0, int(h - y_cutoff_low[i])),
                          pt2=(int(x_cutoff_low[i]), int(h)),
                          color=0, thickness=-1)  # BL
            cv2.rectangle(f, pt1=(int(w - x_cutoff_low[i]), 0),
                          pt2=(int(w), int(y_cutoff_low[i])),
                          color=0, thickness=-1)  # TR
            cv2.rectangle(f, pt1=(int(w - x_cutoff_low[i]), int(h - y_cutoff_low[i])),
                          pt2=(int(w), int(h)), color=0, thickness=-1)  # BR
            i = i + 1

        filter = np.dstack(tuple(temp))
        scv_filt = Factory.Image(filter, color_space=Factory.Image.BGR)

    return img.apply_dft_filter(scv_filt, grayscale)


@static_image_method
def _inverse_dft(input):
    """
    **SUMMARY**
    **PARAMETERS**
    **RETURNS**
    **EXAMPLE**
    NOTES:
    SEE ALSO:
    """
    # a destructive IDFT operation for internal calls
    if len(input) == 1:
        dftimg = cv2.dft(input[0], flags=cv2.DFT_INVERSE)
        data = dftimg[:, :, 0].copy()
        min, max, pt1, pt2 = cv2.minMaxLoc(data)
        denom = max - min
        if denom == 0:
            denom = 1
        data = data / denom - min / denom
        data = cv2.multiply(data, data, scale=255.0)
        result = np.copy(data).astype(np.uint8)  # convert
        ret_val = Factory.Image(result)
    else:  # DO RGB separately
        results = []
        for i in range(0, len(input)):
            dftimg = cv2.dft(input[i], flags=cv2.DFT_INVERSE)
            data = dftimg[:, :, 0].copy()
            min, max, pt1, pt2 = cv2.minMaxLoc(data)
            denom = max - min
            if denom == 0:
                denom = 1
            data = data / denom - min / denom  # scale
            data = cv2.multiply(data, data, scale=255.0)
            result = np.copy(data).astype(np.uint8)
            results.append(result)
        ret_val = Factory.Image(np.dstack((results[0],
                                           results[1],
                                           results[2])))
    return ret_val


@static_image_method
def inverse_dft(raw_dft_image):
    """
    **SUMMARY**

    This method provides a way of performing an inverse discrete Fourier
    transform on a real/imaginary image pair and obtaining the result as
    a simplecv image. This method is helpful if you wish to perform custom
    filter development.

    **PARAMETERS**

    * *raw_dft_image* - A list object with either one or three IPL images.
      Each image should have a 64f depth and contain two channels (the real
      and the imaginary).

    **RETURNS**

    A simpleCV image.

    **EXAMPLE**

    Note that this is an example, I don't recommend doing this unless you
    know what you are doing.

    >>> raw = img.raw_dft_image()
    >>> cv2.SomeOperation(raw)
    >>> result = img.inverse_dft(raw)
    >>> result.show()

    **SEE ALSO**

    :py:meth:`raw_dft_image`
    :py:meth:`get_dft_log_magnitude`
    :py:meth:`apply_dft_filter`
    :py:meth:`high_pass_filter`
    :py:meth:`low_pass_filter`
    :py:meth:`band_pass_filter`
    :py:meth:`inverse_dft`
    :py:meth:`apply_butterworth_filter`
    :py:meth:`inverse_dft`
    :py:meth:`apply_gaussian_filter`
    :py:meth:`apply_unsharp_mask`

    """
    input_dft = [raw_img.copy() for raw_img in raw_dft_image]
    return Factory.Image._inverse_dft(input_dft)


@image_method
def apply_butterworth_filter(img, dia=400, order=2, highpass=False,
                             grayscale=False):
    """
    **SUMMARY**

    Creates a butterworth filter of 64x64 pixels, resizes it to fit
    image, applies DFT on image using the filter.
    Returns image with DFT applied on it

    **PARAMETERS**

    * *dia* - int Diameter of Butterworth low pass filter
    * *order* - int Order of butterworth lowpass filter
    * *highpass*: BOOL True: highpass filterm False: lowpass filter
    * *grayscale*: BOOL

    **EXAMPLE**

    >>> im = Image("lenna")
    >>> img = im.apply_butterworth_filter(dia=400, order=2,
    ...                                   highpass=True, grayscale=False)

    Output image: http://i.imgur.com/5LS3e.png

    >>> img = im.apply_butterworth_filter(dia=400, order=2,
    ...                                   highpass=False, grayscale=False)

    Output img: http://i.imgur.com/QlCAY.png

    >>> # take image from here: http://i.imgur.com/O0gZn.png
    >>> im = Image("grayscale_lenn.png")
    >>> img = im.apply_butterworth_filter(dia=400, order=2,
    ...                                   highpass=True, grayscale=True)

    Output img: http://i.imgur.com/BYYnp.png

    >>> img = im.apply_butterworth_filter(dia=400, order=2,
    ...                           highpass=False, grayscale=True)

    Output img: http://i.imgur.com/BYYnp.png

    **SEE ALSO**

    :py:meth:`raw_dft_image`
    :py:meth:`get_dft_log_magnitude`
    :py:meth:`apply_dft_filter`
    :py:meth:`high_pass_filter`
    :py:meth:`low_pass_filter`
    :py:meth:`band_pass_filter`
    :py:meth:`inverse_dft`
    :py:meth:`apply_butterworth_filter`
    :py:meth:`inverse_dft`
    :py:meth:`apply_gaussian_filter`
    :py:meth:`apply_unsharp_mask`

    """
    #reimplemented with faster, vectorized filter kernel creation
    w, h = img.size_tuple
    intensity_scale = 2 ** 8 - 1  # for now 8-bit
    sz_x = 64  # for now constant, symmetric
    sz_y = 64  # for now constant, symmetric
    x0 = sz_x / 2.0  # for now, on center
    y0 = sz_y / 2.0  # for now, on center
    # efficient "vectorized" computation
    x, y = np.meshgrid(np.arange(sz_x), np.arange(sz_y))
    d = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)
    flt = intensity_scale / (1.0 + (d / dia) ** (order * 2))
    if highpass:  # then invert the filter
        flt = intensity_scale - flt

    # numpy arrays are in row-major form...
    # doesn't matter for symmetric filter
    flt = Factory.Image(flt)
    flt_re = flt.resize(w, h)
    return img.apply_dft_filter(flt_re, grayscale)


@image_method
def apply_gaussian_filter(img, dia=400, highpass=False, grayscale=False):
    """
    **SUMMARY**

    Creates a gaussian filter of 64x64 pixels, resizes it to fit
    image, applies DFT on image using the filter.
    Returns image with DFT applied on it

    **PARAMETERS**

    * *dia* -  int - diameter of Gaussian filter
    * *highpass*: BOOL True: highpass filter False: lowpass filter
    * *grayscale*: BOOL

    **EXAMPLE**

    >>> im = Image("lenna")
    >>> img = im.apply_gaussian_filter(dia=400, highpass=True,
    ...                                grayscale=False)

    Output image: http://i.imgur.com/DttJv.png

    >>> img = im.apply_gaussian_filter(dia=400, highpass=False,
    ...                                grayscale=False)

    Output img: http://i.imgur.com/PWn4o.png

    >>> # take image from here: http://i.imgur.com/O0gZn.png
    >>> im = Image("grayscale_lenn.png")
    >>> img = im.apply_gaussian_filter(dia=400, highpass=True,
    ...                                grayscale=True)

    Output img: http://i.imgur.com/9hX5J.png

    >>> img = im.apply_gaussian_filter(dia=400, highpass=False,
    ...                                grayscale=True)

    Output img: http://i.imgur.com/MXI5T.png

    **SEE ALSO**

    :py:meth:`raw_dft_image`
    :py:meth:`get_dft_log_magnitude`
    :py:meth:`apply_dft_filter`
    :py:meth:`high_pass_filter`
    :py:meth:`low_pass_filter`
    :py:meth:`band_pass_filter`
    :py:meth:`inverse_dft`
    :py:meth:`apply_butterworth_filter`
    :py:meth:`inverse_dft`
    :py:meth:`apply_gaussian_filter`
    :py:meth:`apply_unsharp_mask`

    """
    #reimplemented with faster, vectorized filter kernel creation
    w, h = img.size_tuple
    intensity_scale = 2 ** 8 - 1  # for now 8-bit
    sz_x = 64  # for now constant, symmetric
    sz_y = 64  # for now constant, symmetric
    x0 = sz_x / 2.0  # for now, on center
    y0 = sz_y / 2.0  # for now, on center
    # efficient "vectorized" computation
    x, y = np.meshgrid(np.arange(sz_x), np.arange(sz_y))
    d = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)
    flt = intensity_scale * np.exp(-0.5 * (d / dia) ** 2)
    if highpass:  # then invert the filter
        flt = intensity_scale - flt
    # numpy arrays are in row-major form...
    # doesn't matter for symmetric filter
    flt = Factory.Image(flt)
    flt_re = flt.resize(w, h)
    return img.apply_dft_filter(flt_re, grayscale)


@image_method
def apply_unsharp_mask(img, boost=1, dia=400, grayscale=False):
    """
    **SUMMARY**

    This method applies unsharp mask or highboost filtering
    on image depending upon the boost value provided.
    DFT is applied on image using gaussian lowpass filter.
    A mask is created subtracting the DFT image from the original
    iamge. And then mask is added in the image to sharpen it.
    unsharp masking => image + mask
    highboost filtering => image + (boost)*mask

    **PARAMETERS**

    * *boost* - int  boost = 1 => unsharp masking, boost > 1 => highboost
      filtering
    * *dia* - int Diameter of Gaussian low pass filter
    * *grayscale* - BOOL

    **EXAMPLE**

    Gaussian Filters:

    >>> im = Image("lenna")
    >>> # highboost filtering
    >>> img = im.apply_unsharp_mask(2, grayscale=False)


    output image: http://i.imgur.com/A1pZf.png

    >>> img = im.apply_unsharp_mask(1, grayscale=False) # unsharp masking

    output image: http://i.imgur.com/smCdL.png

    >>> # take image from here: http://i.imgur.com/O0gZn.png
    >>> im = Image("grayscale_lenn.png")
    >>> # highboost filtering
    >>> img = im.apply_unsharp_mask(2, grayscale=True)

    output image: http://i.imgur.com/VtGzl.png

    >>> img = im.apply_unsharp_mask(1,grayscale=True) #unsharp masking

    output image: http://i.imgur.com/bywny.png

    **SEE ALSO**

    :py:meth:`raw_dft_image`
    :py:meth:`get_dft_log_magnitude`
    :py:meth:`apply_dft_filter`
    :py:meth:`high_pass_filter`
    :py:meth:`low_pass_filter`
    :py:meth:`band_pass_filter`
    :py:meth:`inverse_dft`
    :py:meth:`apply_butterworth_filter`
    :py:meth:`inverse_dft`
    :py:meth:`apply_gaussian_filter`
    :py:meth:`apply_unsharp_mask`

    """
    if boost < 0:
        print "boost >= 1"
        return None

    lp_im = img.apply_gaussian_filter(dia=dia, grayscale=grayscale,
                                      highpass=False)
    im = img.copy()
    mask = im - lp_im
    img = im
    for i in range(boost):
        img = img + mask
    return img


@image_method
def filter(img, flt, grayscale=False):
    """
    **SUMMARY**

    This function allows you to apply an arbitrary filter to the DFT of an
    image. This filter takes in a gray scale image, whiter values are kept
    and black values are rejected. In the DFT image, the lower frequency
    values are in the corners of the image, while the higher frequency
    components are in the center. For example, a low pass filter has white
    squares in the corners and is black everywhere else.

    **PARAMETERS**

    * *flt* - A DFT filter

    * *grayscale* - if this value is True we perfrom the operation on the
    DFT of the gray version of the image and the result is gray image. If
    grayscale is true we perform the operation on each channel and the
    recombine them to create the result.

    **RETURNS**

    A SimpleCV image after applying the filter.

    **EXAMPLE**

    >>>  filter = DFT.create_gaussian_filter()
    >>>  myImage = Image("MyImage.png")
    >>>  result = myImage.filter(filter)
    >>>  result.show()
    """
    return flt.apply_filter(img, grayscale)
