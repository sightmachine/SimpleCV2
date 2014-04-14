import math

import cv2
import numpy as np
import scipy.cluster.vq as scv
import scipy.spatial.distance as spsd

from simplecv.base import logger
from simplecv.core.image import (image_method, static_image_method,
                                 cached_method)
from simplecv.factory import Factory
from simplecv.linescan import LineScan


@image_method
def mean_color(img, color_space=None):
    """
    **SUMMARY**

    This method finds the average color of all the pixels in the image and
    displays tuple in the colorspace specfied by the user.
    If no colorspace is specified , (B,G,R) colorspace is taken as default.

    **RETURNS**

    A tuple of the average image values. Tuples are in the channel order.
    *For most images this means the results are (B,G,R).*

    **EXAMPLE**

    >>> img = Image('lenna')
    >>> # returns tuple in Image's colorspace format.
    >>> colors = img.mean_color()
    >>> colors1 = img.mean_color('BGR')   # returns tuple in (B,G,R) format
    >>> colors2 = img.mean_color('RGB')   # returns tuple in (R,G,B) format
    >>> colors3 = img.mean_color('HSV')   # returns tuple in (H,S,V) format
    >>> colors4 = img.mean_color('XYZ')   # returns tuple in (X,Y,Z) format
    >>> colors5 = img.mean_color('Gray')  # returns float of mean intensity
    >>> colors6 = img.mean_color('YCrCb') # returns tuple in Y,Cr,Cb format
    >>> colors7 = img.mean_color('HLS')   # returns tuple in (H,L,S) format

    """
    if color_space is None:
        array = img.get_ndarray()
        if len(array.shape) == 2:
            return np.average(array)
    elif color_space == 'BGR':
        array = img.to_bgr().get_ndarray()
    elif color_space == 'RGB':
        array = img.to_rgb().get_ndarray()
    elif color_space == 'HSV':
        array = img.to_hsv().get_ndarray()
    elif color_space == 'XYZ':
        array = img.to_xyz().get_ndarray()
    elif color_space == 'Gray':
        array = img.get_gray_ndarray()
        return np.average(array)
    elif color_space == 'YCrCb':
        array = img.to_ycrcb().get_ndarray()
    elif color_space == 'HLS':
        array = img.to_hls().get_ndarray()
    else:
        logger.warning("Image.meanColor: There is no supported conversion "
                       "to the specified colorspace. Use one of these as "
                       "argument: 'BGR' , 'RGB' , 'HSV' , 'Gray' , 'XYZ' "
                       ", 'YCrCb' , 'HLS' .")
        return None
    return (np.average(array[:, :, 0]),
            np.average(array[:, :, 1]),
            np.average(array[:, :, 2]))


@image_method
def histogram(img, numbins=50):
    """
    **SUMMARY**

    Return a numpy array of the 1D histogram of intensity for pixels in
    the image
    Single parameter is how many "bins" to have.


    **PARAMETERS**

    * *numbins* - An interger number of bins in a histogram.

    **RETURNS**

    A list of histogram bin values.

    **EXAMPLE**

    >>> img = Image('lenna')
    >>> hist = img.histogram()

    **SEE ALSO**

    :py:meth:`hue_histogram`

    """
    hist, bin_edges = np.histogram(img.get_gray_ndarray(), bins=numbins)
    return hist.tolist()


@image_method
def hue_histogram(img, bins=179, dynamic_range=True):
    """
    **SUMMARY**

    Returns the histogram of the hue channel for the image


    **PARAMETERS**

    * *numbins* - An interger number of bins in a histogram.

    **RETURNS**

    A list of histogram bin values.

    **SEE ALSO**

    :py:meth:`histogram`

    """
    if dynamic_range:
        return np.histogram(img.to_hsv().get_ndarray()[:, :, 2],
                            bins=bins)[0]
    else:
        return np.histogram(img.to_hsv().get_ndarray()[:, :, 2],
                            bins=bins, range=(0.0, 360.0))[0]


@image_method
def hue_peaks(img, bins=179):
    """
    **SUMMARY**

    Takes the histogram of hues, and returns the peak hue values, which
    can be useful for determining what the "main colors" in a picture.

    The bins parameter can be used to lump hues together, by default it
    is 179 (the full resolution in OpenCV's HSV format)

    Peak detection code taken from https://gist.github.com/1178136
    Converted from/based on a MATLAB script at
    http://billauer.co.il/peakdet.html

    Returns a list of tuples, each tuple contains the hue, and the fraction
    of the image that has it.

    **PARAMETERS**

    * *bins* - the integer number of bins, between 0 and 179.

    **RETURNS**

    A list of (hue,fraction) tuples.

    """
    # keyword arguments:
    # y_axis -- A list containg the signal over which to find peaks
    # x_axis -- A x-axis whose values correspond to the
    # 'y_axis' list and is used
    #     in the return to specify the postion of the peaks.
    #     If omitted the index
    #     of the y_axis is used. (default: None)
    # lookahead -- (optional) distance to look ahead from a peak
    # candidate to
    #     determine if it is the actual peak (default: 500)
    #     '(sample / period) / f' where '4 >= f >= 1.25' might be a good
    #      value
    # delta -- (optional) this specifies a minimum difference between
    #     a peak and the following points, before a peak may be considered
    #     a peak. Useful to hinder the algorithm from picking up false
    #     peaks towards to end of the signal. To work well delta should
    #     be set to 'delta >= RMSnoise * 5'.
    #     (default: 0)
    #         Delta function causes a 20% decrease in speed, when omitted
    #         Correctly used it can double the speed of the algorithm
    # return --  Each cell of the lists contains a tupple of:
    #     (position, peak_value)
    #     to get the average peak value
    #     do 'np.mean(maxtab, 0)[1]' on the results

    y_axis, x_axis = np.histogram(img.to_hsv().get_ndarray()[:, :, 2],
                                  bins=bins)
    x_axis = x_axis[0:bins]
    lookahead = int(bins / 17)
    delta = 0

    maxtab = []
    mintab = []
    dump = []  # Used to pop the first hit which always if false

    length = len(y_axis)
    if x_axis is None:
        x_axis = range(length)

    #perform some checks
    if length != len(x_axis):
        raise ValueError("Input vectors y_axis and "
                         "x_axis must have same length")
    if lookahead < 1:
        raise ValueError("Lookahead must be above '1' in value")
    if not (np.isscalar(delta) and delta >= 0):
        raise ValueError("delta must be a positive number")

    #needs to be a numpy array
    y_axis = np.asarray(y_axis)

    #maxima and minima candidates are temporarily stored in
    #mx and mn respectively
    mn, mx = np.Inf, -np.Inf

    #Only detect peak if there is 'lookahead' amount of points after it
    for index, (x, y) in enumerate(
            zip(x_axis[:-lookahead], y_axis[:-lookahead])):
        if y > mx:
            mx = y
            mxpos = x
        if y < mn:
            mn = y
            mnpos = x

        ####look for max####
        if y < mx - delta and mx != np.Inf:
            # Maxima peak candidate found
            # look ahead in signal to ensure that
            # this is a peak and not jitter
            if y_axis[index:index + lookahead].max() < mx:
                maxtab.append((mxpos, mx))
                dump.append(True)
                #set algorithm to only find minima now
                mx = np.Inf
                mn = np.Inf

        ####look for min####
        if y > mn + delta and mn != -np.Inf:
            # Minima peak candidate found
            # look ahead in signal to ensure that
            # this is a peak and not jitter
            if y_axis[index:index + lookahead].min() > mn:
                mintab.append((mnpos, mn))
                dump.append(False)
                #set algorithm to only find maxima now
                mn = -np.Inf
                mx = -np.Inf

    #Remove the false hit on the first value of the y_axis
    try:
        if dump[0]:
            maxtab.pop(0)
            #print "pop max"
        else:
            mintab.pop(0)
            #print "pop min"
        del dump
    except IndexError:
        #no peaks were found, should the function return empty lists?
        pass

    huetab = []
    for hue, pixelcount in maxtab:
        huetab.append((hue, pixelcount / float(img.width * img.height)))
    return huetab


@image_method
@cached_method
def get_edge_map(img, t1=50, t2=100):
    """
    Return the binary bitmap which shows where edges are in the image.
    The two parameters determine how much change in the image determines
    an edge, and how edges are linked together.  For more information
    refer to:

    http://en.wikipedia.org/wiki/Canny_edge_detector
    http://opencv.willowgarage.com/documentation/python/
    imgproc_feature_detection.html?highlight=canny#Canny
    """
    return cv2.Canny(img.get_gray_ndarray(), threshold1=t1, threshold2=t2)


@image_method
def get_pixel(img, x, y):
    """
    **SUMMARY**

    This function returns the RGB value for a particular image pixel given
    a specific row and column.

    .. Warning::
      this function will always return pixels in RGB format even if the
      image is BGR format.

    **PARAMETERS**

        * *x* - Int the x pixel coordinate.
        * *y* - Int the y pixel coordinate.

    **RETURNS**

    A color value that is a three element integer tuple.

    **EXAMPLE**

    >>> img = Image(logo)
    >>> color = img.get_pixel(10,10)


    .. Warning::
      We suggest that this method be used sparingly. For repeated pixel
      access use python array notation. I.e. img[x][y].

    """
    ret_val = None
    if x < 0 or x >= img.width:
        logger.warning("get_pixel: X value is not valid.")
    elif y < 0 or y >= img.height:
        logger.warning("get_pixel: Y value is not valid.")
    else:
        ret_val = img[y, x]
    return ret_val


@image_method
def get_gray_pixel(img, x, y):
    """
    **SUMMARY**

    This function returns the gray value for a particular image pixel given
     a specific row and column.

    .. Warning::
      This function will always return pixels in RGB format even if the
      image is BGR format.

    **PARAMETERS**

    * *x* - Int the x pixel coordinate.
    * *y* - Int the y pixel coordinate.

    **RETURNS**

    A gray value integer between 0 and 255.

    **EXAMPLE**

    >>> img = Image(logo)
    >>> color = img.get_gray_pixel(10,10)


    .. Warning::
      We suggest that this method be used sparingly. For repeated pixel
      access use python array notation. I.e. img[x][y].

    """
    ret_val = None
    if x < 0 or x >= img.width:
        logger.warning("get_gray_pixel: X value is not valid.")
    elif y < 0 or y >= img.height:
        logger.warning("get_gray_pixel: Y value is not valid.")
    else:
        ret_val = img.get_gray_ndarray()[y, x].tolist()
    return ret_val


@image_method
def get_vert_scanline(img, column):
    """
    **SUMMARY**

    This function returns a single column of RGB values from the image as
    a numpy array. This is handy if you want to crawl the image looking
    for an edge.

    **PARAMETERS**

    * *column* - the column number working from left=0 to right=img.width.

    **RETURNS**

    A numpy array of the pixel values. Ususally this is in BGR format.

    **EXAMPLE**

    >>> img = Image("lenna")
    >>> myColor = [0, 0, 0]
    >>> sl = img.get_vert_scanline(423)
    >>> sll = sl.tolist()
    >>> for p in sll:
    >>>    if p == myColor:
    >>>        # do something

    **SEE ALSO**

    :py:meth:`get_horz_scanline_gray`
    :py:meth:`get_horz_scanline`
    :py:meth:`get_vert_scanline_gray`
    :py:meth:`get_vert_scanline`

    """
    ret_val = None
    if column < 0 or column >= img.width:
        logger.warning("get_vert_scanline: column value is not valid.")
    else:
        ret_val = img.get_ndarray()[column, :]
    return ret_val


@image_method
def get_horz_scanline(img, row):
    """
    **SUMMARY**

    This function returns a single row of RGB values from the image.
    This is handy if you want to crawl the image looking for an edge.

    **PARAMETERS**

    * *row* - the row number working from top=0 to bottom=img.height.

    **RETURNS**

    A a lumpy numpy array of the pixel values. Ususally this is in BGR
    format.

    **EXAMPLE**

    >>> img = Image("lenna")
    >>> myColor = [0,0,0]
    >>> sl = img.get_horz_scanline(422)
    >>> sll = sl.tolist()
    >>> for p in sll:
    >>>    if p == myColor:
    >>>        # do something

    **SEE ALSO**

    :py:meth:`get_horz_scanline_gray`
    :py:meth:`get_vert_scanline_gray`
    :py:meth:`get_vert_scanline`

    """
    ret_val = None
    if row < 0 or row >= img.height:
        logger.warning("get_horz_scanline: row value is not valid.")
    else:
        ret_val = img.get_ndarray()[:, row]
    return ret_val


@image_method
def get_vert_scanline_gray(img, column):
    """
    **SUMMARY**

    This function returns a single column of gray values from the image as
    a numpy array. This is handy if you want to crawl the image looking
    for an edge.

    **PARAMETERS**

    * *column* - the column number working from left=0 to right=img.width.

    **RETURNS**

    A a lumpy numpy array of the pixel values.

    **EXAMPLE**

    >>> img = Image("lenna")
    >>> myColor = [255]
    >>> sl = img.get_vert_scanline_gray(421)
    >>> sll = sl.tolist()
    >>> for p in sll:
    >>>    if p == myColor:
    >>>        # do something

    **SEE ALSO**

    :py:meth:`get_horz_scanline_gray`
    :py:meth:`get_horz_scanline`
    :py:meth:`get_vert_scanline`

    """
    ret_val = None
    if column < 0 or column >= img.width:
        logger.warning("getHorzRGBScanline: row value is not valid.")
    else:
        ret_val = img.get_gray_ndarray()[column, :]
    return ret_val


@image_method
def get_horz_scanline_gray(img, row):
    """
    **SUMMARY**

    This function returns a single row of gray values from the image as
    a numpy array. This is handy if you want to crawl the image looking
    for an edge.

    **PARAMETERS**

    * *row* - the row number working from top=0 to bottom=img.height.

    **RETURNS**

    A a lumpy numpy array of the pixel values.

    **EXAMPLE**

    >>> img = Image("lenna")
    >>> myColor = [255]
    >>> sl = img.get_horz_scanline_gray(420)
    >>> sll = sl.tolist()
    >>> for p in sll:
    >>>    if p == myColor:
    >>>        # do something

    **SEE ALSO**

    :py:meth:`get_horz_scanline_gray`
    :py:meth:`get_horz_scanline`
    :py:meth:`get_vert_scanline_gray`
    :py:meth:`get_vert_scanline`

    """
    ret_val = None
    if row < 0 or row >= img.height:
        logger.warning("get_horz_scanline_gray: row value is not valid.")
    else:
        ret_val = img.get_gray_ndarray()[:, row]
    return ret_val


@image_method
def integral_image(img, tilted=False):
    """
    **SUMMARY**

    Calculate the integral image and return it as a numpy array.
    The integral image gives the sum of all of the pixels above and to the
    right of a given pixel location. It is useful for computing Haar
    cascades. The return type is a numpy array the same size of the image.
    The integral image requires 32Bit values which are not easily supported
    by the simplecv Image class.

    **PARAMETERS**

    * *tilted*  - if tilted is true we tilt the image 45 degrees and then
     calculate the results.

    **RETURNS**

    A numpy array of the values.

    **EXAMPLE**

    >>> img = Image("logo")
    >>> derp = img.integral_image()

    **SEE ALSO**

    http://en.wikipedia.org/wiki/Summed_area_table
    """
    if tilted:
        array = cv2.integral3(img.get_gray_ndarray())[2]
    else:
        array = cv2.integral(img.get_gray_ndarray())
    return array


@image_method
def _get_raw_keypoints(img, thresh=500.00, flavor="SURF", highquality=1,
                       force_reset=False):
    """
    .. _get_raw_keypoints:
    This method finds keypoints in an image and returns them as the raw
    keypoints and keypoint descriptors. When this method is called it
    caches a the features and keypoints locally for quick and easy access.

    Parameters:
    min_quality - The minimum quality metric for SURF descriptors. Good
                  values range between about 300.00 and 600.00

    flavor - a string indicating the method to use to extract features.
             A good primer on how feature/keypoint extractiors can be found
             here:

             http://en.wikipedia.org/wiki/
             Feature_detection_(computer_vision)

             http://www.cg.tu-berlin.de/fileadmin/fg144/
             Courses/07WS/compPhoto/Feature_Detection.pdf


             "SURF" - extract the SURF features and descriptors. If you
             don't know what to use, use this.
             See: http://en.wikipedia.org/wiki/SURF

             "STAR" - The STAR feature extraction algorithm
             See: http://pr.willowgarage.com/wiki/Star_Detector

             "FAST" - The FAST keypoint extraction algorithm
             See: http://en.wikipedia.org/wiki/
             Corner_detection#AST_based_feature_detectors

             All the flavour specified below are for
             OpenCV versions >= 2.4.0:

             "MSER" - Maximally Stable Extremal Regions algorithm

             See: http://en.wikipedia.org/
             wiki/Maximally_stable_extremal_regions

             "Dense" - Dense Scale Invariant Feature Transform.

             See: http://www.vlfeat.org/api/dsift.html

             "ORB" - The Oriented FAST and Rotated BRIEF

             See: http://www.willowgarage.com/sites/default/
             files/orb_final.pdf

             "SIFT" - Scale-invariant feature transform

             See: http://en.wikipedia.org/wiki/
             Scale-invariant_feature_transform

             "BRISK" - Binary Robust Invariant Scalable Keypoints

              See: http://www.asl.ethz.ch/people/lestefan/personal/BRISK

             "FREAK" - Fast Retina Keypoints

              See: http://www.ivpe.com/freak.htm
              Note: It's a keypoint descriptor and not a KeyPoint detector.
              SIFT KeyPoints are detected and FERAK is used to extract
              keypoint descriptor.

    highquality - The SURF descriptor comes in two forms, a vector of 64
                  descriptor values and a vector of 128 descriptor values.
                  The latter are "high" quality descriptors.

    force_reset - If keypoints have already been calculated for this image
                 those keypoints are returned veresus recalculating the
                 values. If force reset is True we always recalculate the
                 values, otherwise we will used the cached copies.

    Returns:
    A tuple of keypoint objects and optionally a numpy array of the
    descriptors.

    Example:
    >>> img = Image("aerospace.jpg")
    >>> kp,d = img._get_raw_keypoints()

    Notes:
    If you would prefer to work with the raw keypoints and descriptors each
    image keeps a local cache of the raw values. These are named:

    self._key_points # A tuple of keypoint objects
    See: http://opencv.itseez.com/modules/features2d/doc/
    common_interfaces_of_feature_detectors.html#keypoint-keypoint
    self._kp_descriptors # The descriptor as a floating point numpy array
    self._kp_flavor = "NONE" # The flavor of the keypoints as a string.

    See Also:
     ImageClass._get_raw_keypoints(self, thresh=500.00,
                                 force_reset=False,
                                 flavor="SURF", highquality=1)
     ImageClass._get_flann_matches(self,sd,td)
     ImageClass.find_keypoint_match(self, template, quality=500.00,
                                  minDist=0.2, minMatch=0.4)
     ImageClass.draw_keypoint_matches(self, template, thresh=500.00,
                                    minDist=0.15, width=1)

    """
    if force_reset:
        img._key_points = None
        img._kp_descriptors = None

    _detectors = ["SIFT", "SURF", "FAST", "STAR", "FREAK", "ORB", "BRISK",
                  "MSER", "Dense"]
    _descriptors = ["SIFT", "SURF", "ORB", "FREAK", "BRISK"]
    if flavor not in _detectors:
        logger.warn("Invalid choice of keypoint detector.")
        return None, None

    if img._key_points is not None and img._kp_flavor == flavor:
        return img._key_points, img._kp_descriptors

    if hasattr(cv2, flavor):

        if flavor == "SURF":
            # cv2.SURF(hessianThreshold, nOctaves,
            #          nOctaveLayers, extended, upright)
            detector = cv2.SURF(thresh, 4, 2, highquality, 1)
            img._key_points, img._kp_descriptors = \
                detector.detect(img.get_gray_ndarray(), None, False)
            if len(img._key_points) == 0:
                return None, None
            if highquality == 1:
                img._kp_descriptors = img._kp_descriptors.reshape(
                    (-1, 128))
            else:
                img._kp_descriptors = img._kp_descriptors.reshape(
                    (-1, 64))

        elif flavor in _descriptors:
            detector = getattr(cv2, flavor)()
            img._key_points, img._kp_descriptors = \
                detector.detectAndCompute(img.get_gray_ndarray(), None,
                                          False)
        elif flavor == "MSER":
            if hasattr(cv2, "FeatureDetector_create"):
                detector = cv2.FeatureDetector_create("MSER")
                img._key_points = detector.detect(img.get_gray_ndarray())
    elif flavor == "STAR":
        detector = cv2.StarDetector()
        img._key_points = detector.detect(img.get_gray_ndarray())
    elif flavor == "FAST":
        if not hasattr(cv2, "FastFeatureDetector"):
            logger.warn("You need OpenCV >= 2.4.0 to support FAST")
            return None, None
        detector = cv2.FastFeatureDetector(int(thresh), True)
        img._key_points = detector.detect(img.get_gray_ndarray(), None)
    elif hasattr(cv2, "FeatureDetector_create"):
        if flavor in _descriptors:
            extractor = cv2.DescriptorExtractor_create(flavor)
            if flavor == "FREAK":
                flavor = "SIFT"
            detector = cv2.FeatureDetector_create(flavor)
            img._key_points = detector.detect(img.get_gray_ndarray())
            img._key_points, img._kp_descriptors = extractor.compute(
                img.get_gray_ndarray(), img._key_points)
        else:
            detector = cv2.FeatureDetector_create(flavor)
            img._key_points = detector.detect(img.get_gray_ndarray())
    else:
        logger.warn("simplecv can't seem to find appropriate function "
                    "with your OpenCV version.")
        return None, None
    return img._key_points, img._kp_descriptors


@static_image_method
def _get_flann_matches(sd, td):
    """
    Summary:
    This method does a fast local approximate nearest neighbors (FLANN)
    calculation between two sets of feature vectors. The result are two
    numpy arrays the first one is a list of indexes of the matches and the
    second one is the match distance value. For the match indices or idx,
    the index values correspond to the values of td, and the value in the
    array is the index in td. I. I.e. j = idx[i] is where td[i] matches
    sd[j]. The second numpy array, at the index i is the match distance
    between td[i] and sd[j]. Lower distances mean better matches.

    Parameters:
    sd - A numpy array of feature vectors of any size.
    td - A numpy array of feature vectors of any size, this vector is used
         for indexing and the result arrays will have a length matching
         this vector.

    Returns:
    Two numpy arrays, the first one, idx, is the idx of the matches of the
    vector td with sd. The second one, dist, is the distance value for the
    closest match.

    Example:
    >>> kpt,td = img1._get_raw_keypoints()  # t is template
    >>> kps,sd = img2._get_raw_keypoints()  # s is source
    >>> idx,dist = img1._get_flann_matches(sd, td)
    >>> j = idx[42]
    >>> print kps[j] # matches kp 42
    >>> print dist[i] # the match quality.

    Notes:
    If you would prefer to work with the raw keypoints and descriptors each
    image keeps a local cache of the raw values. These are named:

    self._key_points # A tuple of keypoint objects
    See: http://opencv.itseez.com/modules/features2d/doc/
    common_interfaces_of_feature_detectors.html#keypoint-keypoint
    self._kp_descriptors # The descriptor as a floating point numpy array
    self._kp_flavor = "NONE" # The flavor of the keypoints as a string.

    See:
     ImageClass._get_raw_keypoints(self, thresh=500.00, forceReset=False,
                                 flavor="SURF", highQuality=1)
     ImageClass._get_flann_matches(self, sd, td)
     ImageClass.draw_keypoint_matches(self, template, thresh=500.00,
                                    minDist=0.15, width=1)
     ImageClass.find_keypoints(self, min_quality=300.00,
                              flavor="SURF", highQuality=False)
     ImageClass.find_keypoint_match(self, template, quality=500.00,
                                  minDist=0.2, minMatch=0.4)
    """
    flann_index_kdtree = 1  # bug: flann enums are missing
    flann_params = dict(algorithm=flann_index_kdtree, trees=4)
    flann = cv2.flann_Index(features=sd, params=flann_params)
    # FIXME: need to provide empty dict
    idx, dist = flann.knnSearch(td, 1, params={})
    del flann
    return idx, dist


@image_method
def generate_palette(img, bins, hue, centroids=None):
    """
    **SUMMARY**

    This is the main entry point for palette generation. A palette, for our
    purposes, is a list of the main colors in an image. Creating a palette
    with 10 bins, tries to cluster the colors in rgb space into ten
    distinct groups. In hue space we only look at the hue channel. All of
    the relevant palette data is cached in the image
    class.

    **PARAMETERS**

    * *bins* - an integer number of bins into which to divide the colors in
     the image.
    * *hue* - if hue is true we do only cluster on the image hue values.
    * *centroids* - A list of tuples that are the initial k-means
    estimates. This is handy if you want consisten results from the
    palettize.

    **RETURNS**

    Nothing, but creates the image's cached values for:

    self._do_hue_palette
    self._palette_bins
    self._palette
    self._palette_members
    self._palette_percentages


    **EXAMPLE**

    >>> img._generate_palette(bins=42)

    **NOTES**

    The hue calculations should be siginificantly faster than the generic
    RGB calculation as it works in a one dimensional space. Sometimes the
    underlying scipy method freaks out about k-means initialization with
    the following warning:

    UserWarning: One of the clusters is empty. Re-run kmean with
    a different initialization.

    This shouldn't be a real problem.

    **SEE ALSO**

    ImageClass.get_palette(self, bins=10, hue=False)
    ImageClass.re_palette(self, palette, hue=False)
    ImageClass.draw_palette_colors(self, size=(-1, -1), horizontal=True,
                                 bins=10, hue=False)
    ImageClass.palettize(self, bins=10 ,hue=False)
    ImageClass.binarize_from_palette(self, palette_selection)
    ImageClass.find_blobs_from_palette(self, palette_selection, dilate = 0,
                                    minsize=5, maxsize=0)
    """
    # FIXME: There is a performance issue

    if img._palette_bins != bins or img._do_hue_palette != hue:
        total = float(img.width * img.height)
        percentages = []
        result = None
        if not hue:
            # reshape our matrix to 1xN
            pixels = np.array(img.get_ndarray()).reshape(-1, 3)
            if centroids is None:
                result = scv.kmeans(pixels, bins)
            else:
                if isinstance(centroids, list):
                    centroids = np.array(centroids, dtype=np.uint8)
                result = scv.kmeans(pixels, centroids)

            img._palette_members = scv.vq(pixels, result[0])[0]

        else:
            hsv = img
            if not img.is_hsv():
                hsv = img.to_hsv()

            h = hsv._ndarray[:, :, 0]
            pixels = h.reshape(-1, 1)

            if centroids is None:
                result = scv.kmeans(pixels, bins)
            else:
                if isinstance(centroids, list):
                    centroids = np.array(centroids, dtype=np.uint8)
                    centroids = centroids.reshape(centroids.shape[0], 1)
                result = scv.kmeans(pixels, centroids)

            img._palette_members = scv.vq(pixels, result[0])[0]

        for i in range(0, bins):
            count = np.where(img._palette_members == i)
            v = float(count[0].shape[0]) / total
            percentages.append(v)

        img._do_hue_palette = hue
        img._palette_bins = bins
        img._palette = np.array(result[0], dtype=np.uint8)
        img._palette_percentages = percentages


@image_method
def get_palette(img, bins=10, hue=False, centroids=None):
    """
    **SUMMARY**

    This method returns the colors in the palette of the image. A palette
    is the set of the most common colors in an image. This method is
    helpful for segmentation.

    **PARAMETERS**

    * *bins* - an integer number of bins into which to divide the colors in
     the image.
    * *hue*  - if hue is true we do only cluster on the image hue values.
    * *centroids* - A list of tuples that are the initial k-means
     estimates. This is handy if you want consisten results from the
     palettize.

    **RETURNS**

    A numpy array of the BGR color tuples.

    **EXAMPLE**

    >>> p = img.get_palette(bins=42)
    >>> print p[2]

    **NOTES**

    The hue calculations should be siginificantly faster than the generic
    RGB calculation as it works in a one dimensional space. Sometimes the
    underlying scipy method freaks out about k-means initialization with
    the following warning:

    .. Warning::
      One of the clusters is empty. Re-run kmean with a different
      initialization. This shouldn't be a real problem.

    **SEE ALSO**

    :py:meth:`re_palette`
    :py:meth:`draw_palette_colors`
    :py:meth:`palettize`
    :py:meth:`get_palette`
    :py:meth:`binarize_from_palette`
    :py:meth:`find_blobs_from_palette`

    """
    img.generate_palette(bins, hue, centroids)
    return img._palette


@image_method
def re_palette(img, palette, hue=False):
    """
    **SUMMARY**

    re_palette takes in the palette from another image and attempts to
    apply it to this image. This is helpful if you want to speed up the
    palette computation for a series of images (like those in a video
    stream).

    **PARAMETERS**

    * *palette* - The pre-computed palette from another image.
    * *hue* - Boolean Hue - if hue is True we use a hue palette, otherwise
     we use a BGR palette.

    **RETURNS**

    A SimpleCV Image.

    **EXAMPLE**

    >>> img = Image("lenna")
    >>> img2 = Image("logo")
    >>> p = img.get_palette()
    >>> result = img2.re_palette(p)
    >>> result.show()

    **SEE ALSO**

    :py:meth:`re_palette`
    :py:meth:`draw_palette_colors`
    :py:meth:`palettize`
    :py:meth:`get_palette`
    :py:meth:`binarize_from_palette`
    :py:meth:`find_blobs_from_palette`

    """
    ret_val = None
    if hue:
        if not img.is_hsv():
            hsv = img.to_hsv()
        else:
            hsv = img.copy()

        h = hsv.get_ndarray()[:, :, 0]
        pixels = h.reshape(-1, 1)
        result = scv.vq(pixels, palette)
        derp = palette[result[0]]
        ret_val = Factory.Image(derp.reshape(img.height, img.width))
        ret_val = ret_val.rotate(-90, fixed=False)
        ret_val._do_hue_palette = True
        ret_val._palette_bins = len(palette)
        ret_val._palette = palette
        ret_val._palette_members = result[0]

    else:
        result = scv.vq(img.get_ndarray().reshape(-1, 3), palette)
        ret_val = Factory.Image(
            palette[result[0]].reshape(img.width, img.height, 3))
        ret_val._do_hue_palette = False
        ret_val._palette_bins = len(palette)
        ret_val._palette = palette
        pixels = np.array(img.get_ndarray()).reshape(-1, 3)
        ret_val._palette_members = scv.vq(pixels, palette)[0]

    percentages = []
    total = img.width * img.height
    for i in range(0, len(palette)):
        count = np.where(img._palette_members == i)
        v = float(count[0].shape[0]) / total
        percentages.append(v)
    img._palette_percentages = percentages
    return ret_val


@image_method
def max_value(img, locations=False):
    """
    **SUMMARY**
    Returns the brightest/maximum pixel value in the
    grayscale image. This method can also return the
    locations of pixels with this value.

    **PARAMETERS**

    * *locations* - If true return the location of pixels
       that have this value.

    **RETURNS**

    The maximum value and optionally the list of points as
    a list of (x,y) tuples.

    **EXAMPLE**

    >>> img = Image("lenna")
    >>> max = img.max_value()
    >>> min, pts = img.min_value(locations=True)
    >>> img2 = img.stretch(min,max)

    """
    val = np.max(img.get_gray_ndarray())
    if locations:
        y, x = np.where(img.get_gray_ndarray() == val)
        locs = zip(y.tolist(), x.tolist())
        return int(val), locs
    else:
        return int(val)


@image_method
def min_value(img, locations=False):
    """
    **SUMMARY**
    Returns the darkest/minimum pixel value in the
    grayscale image. This method can also return the
    locations of pixels with this value.

    **PARAMETERS**

    * *locations* - If true return the location of pixels
       that have this value.

    **RETURNS**

    The minimum value and optionally the list of points as
    a list of (x,y) tuples.

    **EXAMPLE**

    >>> img = Image("lenna")
    >>> max = img.max_value()
    >>> min, pts = img.min_value(locations=True)
    >>> img2 = img.stretch(min,max)

    """
    val = np.min(img.get_gray_ndarray())
    if locations:
        y, x = np.where(img.get_gray_ndarray() == val)
        locs = zip(y.tolist(), x.tolist())
        return int(val), locs
    else:
        return int(val)


@image_method
def _copy_avg(img, src, dst, roi, levels, levels_f, mode):
    '''
    Take the value in an ROI, calculate the average / peak hue
    and then set the output image roi to the value.
    '''
    src_roi = src.get_ndarray()[src.roi_to_slice(roi)]
    dst_roi = dst[src.roi_to_slice(roi)]
    if mode:  # get the peak hue for an area
        h = Factory.Image(src_roi).hue_histogram()
        my_hue = np.argmax(h)
        c = (float(my_hue), float(255), float(255), float(0))
        dst_roi += c
    else:  # get the average value for an area optionally set levels
        avg = cv2.mean(src_roi)
        avg = (float(avg[0]), float(avg[1]), float(avg[2]))
        if levels is not None:
            avg = (int(avg[0] / levels) * levels_f,
                   int(avg[1] / levels) * levels_f,
                   int(avg[2] / levels) * levels_f)
        dst_roi += avg

    dst[src.roi_to_slice(roi)] = dst_roi


@image_method
def edge_intersections(img, pt0, pt1, width=1, canny1=0, canny2=100):
    """
    **SUMMARY**

    Find the outermost intersection of a line segment and the edge image
    and return a list of the intersection points. If no intersections are
    found the method returns an empty list.

    **PARAMETERS**

    * *pt0* - an (x,y) tuple of one point on the intersection line.
    * *pt1* - an (x,y) tuple of the second point on the intersection line.
    * *width* - the width of the line to use. This approach works better
                when for cases where the edges on an object are not always
                closed and may have holes.
    * *canny1* - the lower bound of the Canny edge detector parameters.
    * *canny2* - the upper bound of the Canny edge detector parameters.

    **RETURNS**

    A list of two (x,y) tuples or an empty list.

    **EXAMPLE**

    >>> img = Image("SimpleCV")
    >>> a = (25, 100)
    >>> b = (225, 110)
    >>> pts = img.edge_intersections(a, b, width=3)
    >>> e = img.edges(0, 100)
    >>> e.draw_line(a, b, color=Color.RED)
    >>> e.draw_circle(pts[0], 10, color=Color.GREEN)
    >>> e.draw_circle(pts[1], 10, color=Color.GREEN)
    >>> e.show()

    """
    w = abs(pt0[0] - pt1[0])
    h = abs(pt0[1] - pt1[1])
    x = np.min([pt0[0], pt1[0]])
    y = np.min([pt0[1], pt1[1]])
    if w <= 0:
        w = width
        x = np.clip(x - (width / 2), 0, x - (width / 2))
    if h <= 0:
        h = width
        y = np.clip(y - (width / 2), 0, y - (width / 2))
    #got some corner cases to catch here
    p0p = np.array([(pt0[0] - x, pt0[1] - y)])
    p1p = np.array([(pt1[0] - x, pt1[1] - y)])
    edges = img.crop(x, y, w, h).get_edge_map(canny1, canny2)
    line = np.zeros((h, w), np.uint8)
    cv2.line(line, pt1=((pt0[0] - x), (pt0[1] - y)),
             pt2=((pt1[0] - x), (pt1[1] - y)), color=255.00,
             thickness=width, lineType=8)
    line = cv2.multiply(line, edges)
    intersections = line.transpose()
    (xs, ys) = np.where(intersections == 255)
    points = zip(xs, ys)
    if len(points) == 0:
        return [None, None]
    a = np.argmin(spsd.cdist(p0p, points, 'cityblock'))
    b = np.argmin(spsd.cdist(p1p, points, 'cityblock'))
    pt_a = (int(xs[a] + x), int(ys[a] + y))
    pt_b = (int(xs[b] + x), int(ys[b] + y))
    # we might actually want this to be list of all the points
    return [pt_a, pt_b]


@image_method
def fit_contour(img, initial_curve, window=(11, 11),
                params=(0.1, 0.1, 0.1), do_appx=True, appx_level=1):
    """

    **SUMMARY**

    This method tries to fit a list of points to lines in the image. The
    list of points is a list of (x,y) tuples that are near (i.e. within the
    window size) of the line you want to fit in the image. This method uses
    a binary such as the result of calling edges.

    This method is based on active contours. Please see this reference:
    http://en.wikipedia.org/wiki/Active_contour_model

    **PARAMETERS**

    * *initial_curve* - region of the form [(x0,y0),(x1,y1)...] that are
      the initial conditions to fit.
    * *window* - thesearch region around each initial point to look for
      a solution.
    * *params* - The alpha, beta, and gamma parameters for the active
      contours algorithm as a list [alpha, beta, gamma].
    * *do_appx* - post process the snake into a polynomial approximation.
      Basically this flag will clean up the output of the get_contour
      algorithm.
    * *appx_level* - how much to approximate the snake, higher numbers mean
      more approximation.

    **DISCUSSION**

    THIS SECTION IS QUOTED FROM: http://users.ecs.soton.ac.uk/msn/
    book/new_demo/Snakes/
    There are three components to the Energy Function:

    * Continuity
    * Curvature
    * Image (Gradient)

    Each Weighted by Specified Parameter:

    Total Energy = Alpha*Continuity + Beta*Curvature + Gamma*Image

    Choose different values dependent on Feature to extract:

    * Set alpha high if there is a deceptive Image Gradient
    * Set beta  high if smooth edged Feature, low if sharp edges
    * Set gamma high if contrast between Background and Feature is low


    **RETURNS**

    A list of (x,y) tuples that approximate the curve. If you do not use
    approximation the list should be the same length as the input list
    length.

    **EXAMPLE**

    >>> img = Image("lenna")
    >>> edges = img.edges(t1=120, t2=155)
    >>> guess = [(311, 284), (313, 270),
        ...      (320, 259), (330, 253), (347, 245)]
    >>> result = edges.fit_contour(guess)
    >>> img.draw_points(guess, color=Color.RED)
    >>> img.draw_points(result, color=Color.GREEN)
    >>> img.show()

    """
    raise Exception('deprecated. cv2 has no SnakeImage')
    # alpha = [params[0]]
    # beta = [params[1]]
    # gamma = [params[2]]
    # if window[0] % 2 == 0:
    #     window = (window[0] + 1, window[1])
    #     logger.warn("Yo dawg, just a heads up, snakeFitPoints wants an "
    #                 "odd window size. I fixed it for you, but you may "
    #                 "want to take a look at your code.")
    # if window[1] % 2 == 0:
    #     window = (window[0], window[1] + 1)
    #     logger.warn("Yo dawg, just a heads up, snakeFitPoints wants an "
    #                 "odd window size. I fixed it for you, but you may "
    #                 "want to take a look at your code.")
    # raw = cv.SnakeImage(self._get_grayscale_bitmap(), initial_curve,
    #                     alpha, beta, gamma, window,
    #                     (cv.CV_TERMCRIT_ITER, 10, 0.01))
    # if do_appx:
    #     appx = cv2.approxPolyDP(np.array([raw], 'float32'), appx_level,
    #                             True)
    #     ret_val = []
    #     for p in appx:
    #         ret_val.append((int(p[0][0]), int(p[0][1])))
    # else:
    #     ret_val = raw
    #
    # return ret_val


@image_method
def get_threshold_crossing(self, pt1, pt2, threshold=128, darktolight=True,
                           lighttodark=True, departurethreshold=1):
    """
    **SUMMARY**

    This function takes in an image and two points, calculates the
    intensity profile between the points, and returns the single point at
    which the profile crosses an intensity

    **PARAMETERS**

    * *p1, p2* - the starting and ending points in tuple form e.g. (1,2)
    * *threshold* pixel value of desired threshold crossing
    * *departurethreshold* - noise reduction technique.  requires this
     many points to be above the threshold to trigger crossing

    **RETURNS**

    A a lumpy numpy array of the pixel values. Ususally this is in BGR
    format.

    **EXAMPLE**

    >>> img = Image("lenna")
    >>> myColor = [0,0,0]
    >>> sl = img.get_horz_scanline(422)
    >>> sll = sl.tolist()
    >>> for p in sll:
    >>>    if p == myColor:
    >>>        # do something

    **SEE ALSO**

    :py:meth:`get_horz_scanline_gray`
    :py:meth:`get_vert_scanline_gray`
    :py:meth:`get_vert_scanline`

    """
    linearr = self.get_diagonal_scanline_grey(pt1, pt2)
    ind = 0
    crossing = -1
    if departurethreshold == 1:
        while ind < linearr.size - 1:
            if darktolight:
                if linearr[ind] <= threshold \
                        and linearr[ind + 1] > threshold:
                    crossing = ind
                    break
            if lighttodark:
                if linearr[ind] >= threshold \
                        and linearr[ind + 1] < threshold:
                    crossing = ind
                    break
            ind = ind + 1
        if crossing != -1:
            xind = pt1[0] + int(
                round((pt2[0] - pt1[0]) * crossing / linearr.size))
            yind = pt1[1] + int(
                round((pt2[1] - pt1[1]) * crossing / linearr.size))
            ret_val = (xind, yind)
        else:
            ret_val = (-1, -1)
            logger.warning('Edgepoint not found.')
    else:
        while ind < linearr.size - (departurethreshold + 1):
            if darktolight:
                if linearr[ind] <= threshold and \
                        (linearr[ind + 1:ind + 1 + departurethreshold]
                         > threshold).all():
                    crossing = ind
                    break
            if lighttodark:
                if linearr[ind] >= threshold and \
                        (linearr[ind + 1:ind + 1 + departurethreshold]
                         < threshold).all():
                    crossing = ind
                    break
            ind = ind + 1
        if crossing != -1:
            xind = pt1[0] + int(
                round((pt2[0] - pt1[0]) * crossing / linearr.size))
            yind = pt1[1] + int(
                round((pt2[1] - pt1[1]) * crossing / linearr.size))
            ret_val = (xind, yind)
        else:
            ret_val = (-1, -1)
            logger.warning('Edgepoint not found.')
    return ret_val


@image_method
def get_diagonal_scanline_grey(img, pt1, pt2):
    """
    **SUMMARY**

    This function returns a single line of greyscale values from the image.
    TODO: speed inprovements and RGB tolerance

    **PARAMETERS**

    * *pt1, pt2* - the starting and ending points in tuple form e.g. (1,2)

    **RETURNS**

    An array of the pixel values.

    **EXAMPLE**

    >>> img = Image("lenna")
    >>> sl = img.get_diagonal_scanline_grey((100, 200), (300, 400))


    **SEE ALSO**

    :py:meth:`get_horz_scanline_gray`
    :py:meth:`get_vert_scanline_gray`
    :py:meth:`get_vert_scanline`

    """
    if not img.is_gray():
        img = img.to_gray()

    width = round(math.sqrt(
        math.pow(pt2[0] - pt1[0], 2) + math.pow(pt2[1] - pt1[1], 2)))
    ret_val = np.zeros(width)

    for x in range(0, ret_val.size):
        xind = pt1[0] + int(round((pt2[0] - pt1[0]) * x / ret_val.size))
        yind = pt1[1] + int(round((pt2[1] - pt1[1]) * x / ret_val.size))
        current_pixel = img.get_pixel(xind, yind)
        ret_val[x] = current_pixel[0]
    return ret_val


@image_method
def get_line_scan(img, x=None, y=None, pt1=None, pt2=None, channel=-1):
    """
    **SUMMARY**

    This function takes in a channel of an image or grayscale by default
    and then pulls out a series of pixel values as a linescan object
    than can be manipulated further.

    **PARAMETERS**

    * *x* - Take a vertical line scan at the column x.
    * *y* - Take a horizontal line scan at the row y.
    * *pt1* - Take a line scan between two points on the line the line
      scan values always go in the +x direction
    * *pt2* - Second parameter for a non-vertical or horizontal line scan.
    * *channel* - To select a channel. eg: selecting a channel RED, GREEN
      or BLUE. If set to -1 it operates with gray scale values


    **RETURNS**

    A SimpleCV.LineScan object or None if the method fails.

    **EXAMPLE**

    >>>> import matplotlib.pyplot as plt
    >>>> img = Image('lenna')
    >>>> a = img.get_line_scan(x=10)
    >>>> b = img.get_line_scan(y=10)
    >>>> c = img.get_line_scan(pt1=(10,10), pt2 = (500,500))
    >>>> plt.plot(a)
    >>>> plt.plot(b)
    >>>> plt.plot(c)
    >>>> plt.show()

    """

    if channel == -1:
        img_array = img.get_gray_ndarray()
    else:
        try:
            img_array = img.get_ndarray()[:, :, channel]
        except IndexError:
            logger.warning('Channel missing!')
            return None

    ret_val = None
    if x is not None and y is None and pt1 is None and pt2 is None:
        if x >= 0 and x < img.width:
            ret_val = LineScan(img_array[:, x])
            ret_val.image = img
            ret_val.pt1 = (x, 0)
            ret_val.pt2 = (x, img.height)
            ret_val.col = x
            x = np.ones((1, img.height))[0] * x
            y = range(0, img.height, 1)
            pts = zip(x, y)
            ret_val.point_loc = pts
        else:
            logger.warn("ImageClass.get_line_scan - "
                        "that is not valid scanline.")
            return None

    elif x is None and y is not None and pt1 is None and pt2 is None:
        if y >= 0 and y < img.height:
            ret_val = LineScan(img_array[y, :])
            ret_val.image = img
            ret_val.pt1 = (0, y)
            ret_val.pt2 = (img.width, y)
            ret_val.row = y
            y = np.ones((1, img.width))[0] * y
            x = range(0, img.width, 1)
            pts = zip(x, y)
            ret_val.point_loc = pts

        else:
            logger.warn("ImageClass.get_line_scan - "
                        "that is not valid scanline.")
            return None

    elif isinstance(pt1, (tuple, list)) and isinstance(pt2, (tuple, list))\
            and len(pt1) == 2 and len(pt2) == 2 \
            and x is None and y is None:

        pts = img.bresenham_line(pt1, pt2)
        ret_val = LineScan([img_array[p[1], p[0]] for p in pts])
        ret_val.point_loc = pts
        ret_val.image = img
        ret_val.pt1 = pt1
        ret_val.pt2 = pt2

    else:
        # an invalid combination - warn
        logger.warn("ImageClass.get_line_scan - that is not valid "
                    "scanline.")
        return None
    ret_val.channel = channel
    return ret_val


@image_method
def set_line_scan(img, linescan, x=None, y=None, pt1=None, pt2=None,
                  channel=-1):
    """
    **SUMMARY**

    This function helps you put back the linescan in the image.

    **PARAMETERS**

    * *linescan* - LineScan object
    * *x* - put  line scan at the column x.
    * *y* - put line scan at the row y.
    * *pt1* - put line scan between two points on the line the line scan
      values always go in the +x direction
    * *pt2* - Second parameter for a non-vertical or horizontal line scan.
    * *channel* - To select a channel. eg: selecting a channel RED,GREEN
      or BLUE. If set to -1 it operates with gray scale values


    **RETURNS**

    A SimpleCV.Image

    **EXAMPLE**

    >>> img = Image('lenna')
    >>> a = img.get_line_scan(x=10)
    >>> for index in range(len(a)):
        ... a[index] = 0
    >>> newimg = img.set_line_scan(a, x=50)
    >>> newimg.show()
    # This will show you a black line in column 50.

    """
    #retVal = self.to_gray()
    if channel == -1:
        img_array = np.copy(img.get_gray_ndarray())
    else:
        try:
            img_array = np.copy(img.get_ndarray()[:, :, channel])
        except IndexError:
            logger.warn('Channel missing!')
            return None

    if x is None and y is None and pt1 is None and pt2 is None:
        if linescan.pt1 is None or linescan.pt2 is None:
            logger.warn("ImageClass.set_line_scan: No coordinates to "
                        "re-insert linescan.")
            return None
        else:
            pt1 = linescan.pt1
            pt2 = linescan.pt2
            if pt1[0] == pt2[0] and np.abs(pt1[1] - pt2[1]) == img.height:
                x = pt1[0]  # vertical line
                pt1 = None
                pt2 = None

            elif pt1[1] == pt2[1] \
                    and np.abs(pt1[0] - pt2[0]) == img.width:
                y = pt1[1]  # horizontal line
                pt1 = None
                pt2 = None

    ret_val = None
    if x is not None and y is None and pt1 is None and pt2 is None:
        if x >= 0 and x < img.width:
            if len(linescan) != img.height:
                linescan = linescan.resample(img.height)
            #check for number of points
            #linescan = np.array(linescan)
            img_array[x, :] = np.clip(linescan[:], 0, 255)
        else:
            logger.warn("ImageClass.set_line_scan: No coordinates to "
                        "re-insert linescan.")
            return None
    elif x is None and y is not None and pt1 is None and pt2 is None:
        if y >= 0 and y < img.height:
            if len(linescan) != img.width:
                linescan = linescan.resample(img.width)
            #check for number of points
            #linescan = np.array(linescan)
            img_array[:, y] = np.clip(linescan[:], 0, 255)
        else:
            logger.warn("ImageClass.set_line_scan: No coordinates to "
                        "re-insert linescan.")
            return None
    elif isinstance(pt1, (tuple, list)) and isinstance(pt2, (tuple, list))\
            and len(pt1) == 2 and len(pt2) == 2 \
            and x is None and y is None:

        pts = img.bresenham_line(pt1, pt2)
        if len(linescan) != len(pts):
            linescan = linescan.resample(len(pts))
        #linescan = np.array(linescan)
        linescan = np.clip(linescan[:], 0, 255)
        idx = 0
        for pt in pts:
            img_array[pt[0], pt[1]] = linescan[idx]
            idx = idx + 1
    else:
        logger.warn("ImageClass.set_line_scan: No coordinates to "
                    "re-insert linescan.")
        return None
    if channel == -1:
        ret_val = Factory.Image(img_array)
    else:
        temp = np.copy(img.get_ndarray())
        temp[:, :, channel] = img_array
        ret_val = Factory.Image(temp)
    return ret_val


@image_method
def replace_line_scan(img, linescan, x=None, y=None, pt1=None, pt2=None,
                      channel=None):
    """

    **SUMMARY**

    This function easily lets you replace the linescan in the image.
    Once you get the LineScan object, you might want to edit it. Perform
    some task, apply some filter etc and now you want to put it back where
    you took it from. By using this function, it is not necessary to
    specify where to put the data. It will automatically replace where you
    took the LineScan from.

    **PARAMETERS**

    * *linescan* - LineScan object
    * *x* - put  line scan at the column x.
    * *y* - put line scan at the row y.
    * *pt1* - put line scan between two points on the line the line scan
      values always go in the +x direction
    * *pt2* - Second parameter for a non-vertical or horizontal line scan.
    * *channel* - To select a channel. eg: selecting a channel RED,GREEN
      or BLUE. If set to -1 it operates with gray scale values


    **RETURNS**

    A SimpleCV.Image

    **EXAMPLE**

    >>> img = Image('lenna')
    >>> a = img.get_line_scan(x=10)
    >>> for index in range(len(a)):
        ... a[index] = 0
    >>> newimg = img.replace_line_scan(a)
    >>> newimg.show()
    # This will show you a black line in column 10.

    """

    if x is None and y is None \
            and pt1 is None and pt2 is None and channel is None:

        if linescan.channel == -1:
            img_array = img.get_gray_ndarray().copy()
        else:
            try:
                img_array = np.copy(img.get_ndarray()[:, :, linescan.channel])
            except IndexError:
                logger.warn('Channel missing!')
                return None

        if linescan.row is not None:
            if len(linescan) == img.width:
                ls = np.clip(linescan, 0, 255)
                img_array[linescan.row, :] = ls[:]
            else:
                logger.warn("LineScan Size and Image size do not match")
                return None

        elif linescan.col is not None:
            if len(linescan) == img.height:
                ls = np.clip(linescan, 0, 255)
                img_array[:, linescan.col] = ls[:]
            else:
                logger.warn("LineScan Size and Image size do not match")
                return None
        elif linescan.pt1 and linescan.pt2:
            pts = img.bresenham_line(linescan.pt1, linescan.pt2)
            if len(linescan) != len(pts):
                linescan = linescan.resample(len(pts))
            ls = np.clip(linescan[:], 0, 255)
            idx = 0
            for pt in pts:
                img_array[pt[0], pt[1]] = ls[idx]
                idx = idx + 1

        if linescan.channel == -1:
            ret_val = Factory.Image(img_array)
        else:
            temp = np.copy(img.get_ndarray())
            temp[:, :, linescan.channel] = img_array
            ret_val = Factory.Image(temp)

    else:
        if channel is None:
            ret_val = img.set_line_scan(linescan, x, y, pt1, pt2,
                                        linescan.channel)
        else:
            ret_val = img.set_line_scan(linescan, x, y, pt1, pt2, channel)
    return ret_val


@image_method
def get_pixels_online(img, pt1, pt2):
    """
    **SUMMARY**

    Return all of the pixels on an arbitrary line.

    **PARAMETERS**

    * *pt1* - The first pixel coordinate as an (x,y) tuple or list.
    * *pt2* - The second pixel coordinate as an (x,y) tuple or list.

    **RETURNS**

    Returns a list of RGB pixels values.

    **EXAMPLE**

    >>>> img = Image('something.png')
    >>>> img.get_pixels_online( (0,0), (img.width/2,img.height/2) )
    """
    ret_val = None
    if isinstance(pt1, (tuple, list)) and isinstance(pt2, (tuple, list)) \
            and len(pt1) == 2 and len(pt2) == 2:
        pts = img.bresenham_line(pt1, pt2)
        ret_val = [img.get_pixel(p[0], p[1]) for p in pts]
    else:
        logger.warn("ImageClass.get_pixels_online - The line you "
                    "provided is not valid")
    return ret_val


@image_method
def bresenham_line(img, (x, y), (x2, y2)):
    """
    Brensenham line algorithm

    cribbed from: http://snipplr.com/view.php?codeview&id=22482

    This is just a helper method
    """
    if not 0 <= x <= img.width - 1 \
            or not 0 <= y <= img.height - 1 \
            or not 0 <= x2 <= img.width - 1 \
            or not 0 <= y2 <= img.height - 1:
        from simplecv.features.detection import Line
        l = Line(img, ((x, y), (x2, y2))).crop_to_image_edges()
        if l:
            ep = list(l.end_points)
            ep.sort()
            x, y = ep[0]
            x2, y2 = ep[1]
        else:
            return []

    steep = 0
    coords = []
    dx = abs(x2 - x)
    if (x2 - x) > 0:
        sx = 1
    else:
        sx = -1
    dy = abs(y2 - y)
    if (y2 - y) > 0:
        sy = 1
    else:
        sy = -1
    if dy > dx:
        steep = 1
        x, y = y, x
        dx, dy = dy, dx
        sx, sy = sy, sx
    d = (2 * dy) - dx
    for i in range(0, dx):
        if steep:
            coords.append((y, x))
        else:
            coords.append((x, y))
        while d >= 0:
            y = y + sy
            d = d - (2 * dx)
        x = x + sx
        d = d + (2 * dy)
    coords.append((x2, y2))
    return coords


@image_method
def uncrop(img, list_of_pts):  # (x,y),(x2,y2)):
    """
    **SUMMARY**

    This function allows us to translate a set of points from the crop
    window back to the coordinate of the source window.

    **PARAMETERS**

    * *list_of_pts* - set of points from cropped image.

    **RETURNS**

    Returns a list of coordinates in the source image.

    **EXAMPLE**

    >> img = Image('lenna')
    >> croppedImg = img.crop(10,20,250,500)
    >> sourcePts = croppedImg.uncrop([(2,3),(56,23),(24,87)])
    """
    return [(i[0] + img.uncropped_x, i[1] + img.uncropped_y) for i in
            list_of_pts]


@image_method
def logical_and(self, img, grayscale=True):
    """
    **SUMMARY**

    Perform bitwise AND operation on images

    **PARAMETERS**

    img - the bitwise operation to be performed with
    grayscale

    **RETURNS**

    SimpleCV.ImageClass.Image

    **EXAMPLE**

    >>> img = Image("something.png")
    >>> img1 = Image("something_else.png")
    >>> img.logical_and(img1, grayscale=False)
    >>> img.logical_and(img1)

    """
    if self.size != img.size:
        logger.warning("Both images must have same sizes")
        return None
    if grayscale:
        retval = cv2.bitwise_and(self.get_gray_ndarray(),
                                 img.get_gray_ndarray())
    else:
        retval = cv2.bitwise_and(self.get_ndarray(), img.get_ndarray())
    return Factory.Image(retval)


@image_method
def logical_nand(self, img, grayscale=True):
    """
    **SUMMARY**

    Perform bitwise NAND operation on images

    **PARAMETERS**

    img - the bitwise operation to be performed with
    grayscale

    **RETURNS**

    SimpleCV.ImageClass.Image

    **EXAMPLE**

    >>> img = Image("something.png")
    >>> img1 = Image("something_else.png")
    >>> img.logical_nand(img1, grayscale=False)
    >>> img.logical_nand(img1)

    """
    if self.size != img.size:
        logger.warning("Both images must have same sizes")
        return None
    if grayscale:
        retval = cv2.bitwise_and(self.get_gray_ndarray(),
                                 img.get_gray_ndarray())
    else:
        retval = cv2.bitwise_and(self.get_ndarray(), img.get_ndarray())
    retval = cv2.bitwise_not(retval)
    return Factory.Image(retval)


@image_method
def logical_or(self, img, grayscale=True):
    """
    **SUMMARY**

    Perform bitwise OR operation on images

    **PARAMETERS**

    img - the bitwise operation to be performed with
    grayscale

    **RETURNS**

    SimpleCV.ImageClass.Image

    **EXAMPLE**

    >>> img = Image("something.png")
    >>> img1 = Image("something_else.png")
    >>> img.logical_or(img1, grayscale=False)
    >>> img.logical_or(img1)

    """
    if self.size != img.size:
        logger.warning("Both images must have same sizes")
        return None
    if grayscale:
        retval = cv2.bitwise_or(self.get_gray_ndarray(),
                                img.get_gray_ndarray())
    else:
        retval = cv2.bitwise_or(self.get_ndarray(), img.get_ndarray())
    return Factory.Image(retval)


@image_method
def logical_xor(self, img, grayscale=True):
    """
    **SUMMARY**

    Perform bitwise XOR operation on images

    **PARAMETERS**

    img - the bitwise operation to be performed with
    grayscale

    **RETURNS**

    SimpleCV.ImageClass.Image

    **EXAMPLE**

    >>> img = Image("something.png")
    >>> img1 = Image("something_else.png")
    >>> img.logical_xor(img1, grayscale=False)
    >>> img.logical_xor(img1)

    """
    if self.size != img.size:
        logger.warning("Both images must have same sizes")
        return None
    if grayscale:
        retval = cv2.bitwise_xor(self.get_gray_ndarray(),
                                 img.get_gray_ndarray())
    else:
        retval = cv2.bitwise_xor(self.get_ndarray(), img.get_ndarray())
    return Factory.Image(retval)


@image_method
def vertical_histogram(img, bins=10, threshold=128, normalize=False,
                       for_plot=False):
    """

    **DESCRIPTION**

    This method generates histogram of the number of grayscale pixels
    greater than the provided threshold. The method divides the image
    into a number evenly spaced vertical bins and then counts the number
    of pixels where the pixel is greater than the threshold. This method
    is helpful for doing basic morphological analysis.

    **PARAMETERS**

    * *bins* - The number of bins to use.
    * *threshold* - The grayscale threshold. We count pixels greater than
      this value.
    * *normalize* - If normalize is true we normalize the bin countsto sum
      to one. Otherwise we return the number of pixels.
    * *for_plot* - If this is true we return the bin indicies, the bin
      counts, and the bin widths as a tuple. We can use these values in
      pyplot.bar to quickly plot the histogram.


    **RETURNS**

    The default settings return the raw bin counts moving from left to
    right on the image. If for_plot is true we return a tuple that
    contains a list of bin labels, the bin counts, and the bin widths.
    This tuple can be used to plot the histogram using
    matplotlib.pyplot.bar function.


    **EXAMPLE**

      >>> import matplotlib.pyplot as plt
      >>> img = Image('lenna')
      >>> plt.bar(*img.vertical_histogram(threshold=128, bins=10,
          ...                            normalize=False, for_plot=True),
          ...     color='y')
      >>> plt.show()


    **NOTES**

    See: http://docs.scipy.org/doc/numpy/reference/generated/
    numpy.histogram.html
    See: http://matplotlib.org/api/
    pyplot_api.html?highlight=hist#matplotlib.pyplot.hist

    """
    if bins <= 0:
        raise Exception("Not enough bins")

    img_array = img.get_gray_ndarray()
    pts = np.where(img_array > threshold)
    y = pts[1]
    hist = np.histogram(y, bins=bins, range=(0, img.height),
                        normed=normalize)
    ret_val = None
    if for_plot:
        # for using matplotlib bar command
        # bin labels, bin values, bin width
        ret_val = (hist[1][0:-1], hist[0], img.height / bins)
    else:
        ret_val = hist[0]
    return ret_val


@image_method
def horizontal_histogram(img, bins=10, threshold=128, normalize=False,
                         for_plot=False):
    """

    **DESCRIPTION**

    This method generates histogram of the number of grayscale pixels
    greater than the provided threshold. The method divides the image
    into a number evenly spaced horizontal bins and then counts the number
    of pixels where the pixel is greater than the threshold. This method
    is helpful for doing basic morphological analysis.

    **PARAMETERS**

    * *bins* - The number of bins to use.
    * *threshold* - The grayscale threshold. We count pixels greater than
      this value.
    * *normalize* - If normalize is true we normalize the bin counts to sum
      to one. Otherwise we return the number of pixels.
    * *for_plot* - If this is true we return the bin indicies, the bin
      counts, and the bin widths as a tuple. We can use these values in
      pyplot.bar to quickly plot the histogram.


    **RETURNS**

    The default settings return the raw bin counts moving from top to
    bottom on the image. If for_plot is true we return a tuple that
    contains a list of bin labels, the bin counts, and the bin widths.
    This tuple can be used to plot the histogram using
    matplotlib.pyplot.bar function.

    **EXAMPLE**

    >>>> import matplotlib.pyplot as plt
    >>>> img = Image('lenna')
    >>>> plt.bar(img.horizontal_histogram(threshold=128, bins=10,
    ...                                   normalize=False, for_plot=True),
    ...          color='y')
    >>>> plt.show())

    **NOTES**

    See: http://docs.scipy.org/doc/numpy/reference/generated/
    numpy.histogram.html
    See: http://matplotlib.org/api/
    pyplot_api.html?highlight=hist#matplotlib.pyplot.hist

    """
    if bins <= 0:
        raise Exception("Not enough bins")

    img_array = img.get_gray_ndarray()
    pts = np.where(img_array > threshold)
    x = pts[0]
    hist = np.histogram(x, bins=bins, range=(0, img.width),
                        normed=normalize)
    ret_val = None
    if for_plot:
        # for using matplotlib bar command
        # bin labels, bin values, bin width
        ret_val = (hist[1][0:-1], hist[0], img.width / bins)
    else:
        ret_val = hist[0]
    return ret_val


@image_method
def get_gray_histogram_counts(img, bins=255, limit=-1):
    '''
    This function returns a list of tuples of greyscale pixel counts
    by frequency.  This would be useful in determining the dominate
    pixels (peaks) of the greyscale image.

    **PARAMETERS**

    * *bins* - The number of bins for the hisogram, defaults to 255
      (greyscale)
    * *limit* - The number of counts to return, default is all

    **RETURNS**

    * List * - A list of tuples of (frequency, value)

    **EXAMPLE**

    >>> img = Image("lenna")
    >>> counts = img.get_gray_histogram_counts()
    >>> # the most dominate pixel color tuple of frequency and value
    >>> counts[0]
    >>> counts[1][1] # the second most dominate pixel color value
    '''

    hist = img.histogram(bins)
    vals = [(e, h) for h, e in enumerate(hist)]
    vals.sort()
    vals.reverse()

    if limit == -1:
        limit = bins

    return vals[:limit]


@image_method
def gray_peaks(img, bins=255, delta=0, lookahead=15):
    """
    **SUMMARY**

    Takes the histogram of a grayscale image, and returns the peak
    grayscale intensity values.

    The bins parameter can be used to lump grays together, by default it is
    set to 255

    Returns a list of tuples, each tuple contains the grayscale intensity,
    and the fraction of the image that has it.

    **PARAMETERS**

    * *bins* - the integer number of bins, between 1 and 255.

    * *delta* - the minimum difference betweena peak and the following
                points, before a peak may be considered a peak.Useful to
                hinder the algorithm from picking up false peaks towards
                to end of the signal.

    * *lookahead* - the distance to lookahead from a peakto determine if it
                    is an actual peak, should be an integer greater than 0.

    **RETURNS**

    A list of (grays,fraction) tuples.

    **NOTE**

    Implemented using the techniques used in huetab()

    """

    # The bins are the no of edges bounding an histogram.
    # Thus bins= Number of bars in histogram+1
    # As range() function is exclusive,
    # hence bins+2 is passed as parameter.

    y_axis, x_axis = np.histogram(img.get_gray_ndarray(),
                                  bins=range(bins + 2))
    x_axis = x_axis[0:bins + 1]
    maxtab = []
    mintab = []
    length = len(y_axis)
    if x_axis is None:
        x_axis = range(length)

    #perform some checks
    if length != len(x_axis):
        raise ValueError("Input vectors y_axis and x_axis must have "
                         "same length")
    if lookahead < 1:
        raise ValueError("Lookahead must be above '1' in value")
    if not (np.isscalar(delta) and delta >= 0):
        raise ValueError("delta must be a positive number")

    #needs to be a numpy array
    y_axis = np.asarray(y_axis)

    #maxima and minima candidates are temporarily stored in
    #mx and mn respectively
    mn, mx = np.Inf, -np.Inf

    #Only detect peak if there is 'lookahead' amount of points after it
    for index, (x, y) in enumerate(zip(x_axis[:-lookahead],
                                       y_axis[:-lookahead])):
        if y > mx:
            mx = y
            mxpos = x
        if y < mn:
            mn = y
            mnpos = x

        ####look for max####
        if y < mx - delta and mx != np.Inf:
            # Maxima peak candidate found
            # look ahead in signal to ensure that this
            # is a peak and not jitter
            if y_axis[index:index + lookahead].max() < mx:
                maxtab.append((mxpos, mx))
                #set algorithm to only find minima now
                mx = np.Inf
                mn = np.Inf

        if y > mn + delta and mn != -np.Inf:
            # Minima peak candidate found
            # look ahead in signal to ensure that
            # this is a peak and not jitter
            if y_axis[index:index + lookahead].min() > mn:
                mintab.append((mnpos, mn))
                #set algorithm to only find maxima now
                mn = -np.Inf
                mx = -np.Inf

    ret_val = []
    for intensity, pixelcount in maxtab:
        ret_val.append(
            (intensity, pixelcount / float(img.width * img.height)))
    return ret_val


@static_image_method
def _bounds_from_percentage2(float_val, bound):
    return np.clip(int(float_val * bound), 0, bound)


@static_image_method
def _bounds_from_percentage(float_val, bound):
    return np.clip(int(float_val * (bound / 2.00)), 0, (bound / 2))


@image_method
def get_normalized_hue_histogram(img, roi=None):
    """
    **SUMMARY**

    This method generates a normalized hue histogram for the image
    or the ROI within the image. The hue histogram is a 2D hue/saturation
    numpy array histogram with a shape of 180x256. This histogram can
    be used for histogram back projection.

    **PARAMETERS**

    * *roi* - Anything that can be cajoled into being an ROI feature
      including a tuple of (x,y,w,h), a list of points, or another feature.

    **RETURNS**

    A normalized 180x256 numpy array that is the hue histogram.

    **EXAMPLE**

    >>> img = Image('lenna')
    >>> roi = (0, 0, 100, 100)
    >>> hist = img.get_normalized_hue_histogram(roi)

    **SEE ALSO**

    ImageClass.back_project_hue_histogram()
    ImageClass.find_blobs_from_hue_histogram()

    """
    from simplecv.features.detection import ROI

    if roi:  # roi is anything that can be taken to be an roi
        roi = ROI(roi, img)
        hsv = roi.crop().to_hsv().get_ndarray()
    else:
        hsv = img.to_hsv().get_ndarray()
    hist = cv2.calcHist(images=[hsv], channels=[0, 1], mask=None,
                        histSize=[180, 256], ranges=[0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
    return hist


@image_method
def back_project_hue_histogram(img, model, smooth=True, full_color=False,
                               threshold=None):
    """
    **SUMMARY**

    This method performs hue histogram back projection on the image. This
    is a very quick and easy way of matching objects based on color. Given
    a hue histogram taken from another image or an roi within the image we
    attempt to find all pixels that are similar to the colors inside the
    histogram. The result can either be a grayscale image that shows the
    matches or a color image.


    **PARAMETERS**

    * *model* - The histogram to use for pack projection. This can either
    be a histogram, anything that can be converted into an ROI for the
    image (like an x,y,w,h tuple or a feature, or another image.

    * *smooth* - A bool, True means apply a smoothing operation after doing
    the back project to improve the results.

    * *full_color* - return the results as a color image where pixels
    included in the back projection are rendered as their source colro.

    * *threshold* - If this value is not None, we apply a threshold to the
    result of back projection to yield a binary image. Valid values are
    from 1 to 255.

    **RETURNS**

    A SimpleCV Image rendered according to the parameters provided.

    **EXAMPLE**

    >>>> img = Image('lenna')

    Generate a hist

    >>>> hist = img.get_normalized_hue_histogram((0, 0, 50, 50))
    >>>> a = img.back_project_hue_histogram(hist)
    >>>> b = img.back_project_hue_histogram((0, 0, 50, 50))  # same result
    >>>> c = img.back_project_hue_histogram(Image('lyle'))

    **SEE ALSO**
    ImageClass.get_normalized_hue_histogram()
    ImageClass.find_blobs_from_hue_histogram()

    """
    if model is None:
        logger.warn('Backproject requires a model')
        return None
    # this is the easier test, try to cajole model into ROI
    if isinstance(model, Factory.Image):
        model = model.get_normalized_hue_histogram()
    if not isinstance(model, np.ndarray) or model.shape != (180, 256):
        model = img.get_normalized_hue_histogram(model)
    if isinstance(model, np.ndarray) and model.shape == (180, 256):
        hsv = img.to_hsv().get_ndarray()
        dst = cv2.calcBackProject(images=[hsv], channels=[0, 1],
                                  hist=model, ranges=[0, 180, 0, 256], scale=1)
        if smooth:
            disc = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE,
                                             ksize=(5, 5))
            dst = cv2.filter2D(dst, ddepth=-1, kernel=disc)
        result = Factory.Image(dst)
        result = result.to_bgr()
        if threshold:
            result = result.threshold(threshold)
        if full_color:
            temp = Factory.Image((img.width, img.height))
            result = temp.blit(img, alpha_mask=result)
        return result
    else:
        logger.warn('Backproject model does not appear to be valid')
        return None
