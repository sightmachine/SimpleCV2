import itertools
import math
import os

import cv2
import numpy as np
import scipy.cluster.vq as scv
import scipy.linalg as nla  # for linear algebra / least squares

from simplecv import DATA_DIR
from simplecv.base import logger
from simplecv.color import Color
from simplecv.core.image import image_method
from simplecv.factory import Factory
from simplecv.features.blobmaker import BlobMaker
from simplecv.features.facerecognizer import FaceRecognizer
from simplecv.features.features import FeatureSet, Feature
from simplecv.features.haar_cascade import HaarCascade


@image_method
def find_corners(img, maxnum=50, minquality=0.04, mindistance=1.0):
    """
    **SUMMARY**

    This will find corner Feature objects and return them as a FeatureSet
    strongest corners first.  The parameters give the number of corners to
    look for, the minimum quality of the corner feature, and the minimum
    distance between corners.

    **PARAMETERS**

    * *maxnum* - The maximum number of corners to return.

    * *minquality* - The minimum quality metric. This shoudl be a number
     between zero and one.

    * *mindistance* - The minimum distance, in pixels, between successive
     corners.

    **RETURNS**

    A featureset of :py:class:`Corner` features or None if no corners are
     found.


    **EXAMPLE**

    Standard Test:

    >>> img = Image("data/sampleimages/simplecv.png")
    >>> corners = img.find_corners()
    >>> if corners: True

    True

    Validation Test:

    >>> img = Image("data/sampleimages/black.png")
    >>> corners = img.find_corners()
    >>> if not corners: True

    True

    **SEE ALSO**

    :py:class:`Corner`
    :py:meth:`find_keypoints`

    """
    corner_coordinates = cv2.goodFeaturesToTrack(img.get_gray_ndarray(),
                                                 maxCorners=maxnum,
                                                 qualityLevel=minquality,
                                                 minDistance=mindistance)
    corner_features = []
    for x, y in corner_coordinates[:, 0, :]:
        corner_features.append(Factory.Corner(img, x, y))

    return FeatureSet(corner_features)


@image_method
def find_blobs(img, threshval=None, minsize=10, maxsize=0,
               threshblocksize=0, threshconstant=5, appx_level=3):
    """

    **SUMMARY**

    Find blobs  will look for continuous
    light regions and return them as Blob features in a FeatureSet.
    Parameters specify the binarize filter threshold value, and minimum and
    maximum size for blobs. If a threshold value is -1, it will use an
    adaptive threshold.  See binarize() for more information about
    thresholding.  The threshblocksize and threshconstant parameters are
    only used for adaptive threshold.


    **PARAMETERS**

    * *threshval* - the threshold as an integer or an (r,g,b) tuple , where
      pixels below (darker) than thresh are set to to max value,
      and all values above this value are set to black. If this parameter
      is -1 we use Otsu's method.

    * *minsize* - the minimum size of the blobs, in pixels, of the returned
     blobs. This helps to filter out noise.

    * *maxsize* - the maximim size of the blobs, in pixels, of the returned
     blobs.

    * *threshblocksize* - the size of the block used in the adaptive
      binarize operation. *TODO - make this match binarize*

    * *appx_level* - The blob approximation level - an integer for the
      maximum distance between the true edge and the
      approximation edge - lower numbers yield better approximation.

      .. warning::
        This parameter must be an odd number.

    * *threshconstant* - The difference from the local mean to use for
     thresholding in Otsu's method. *TODO - make this match binarize*


    **RETURNS**

    Returns a featureset (basically a list) of :py:class:`blob` features.
    If no blobs are found this method returns None.

    **EXAMPLE**

    >>> img = Image("lenna")
    >>> fs = img.find_blobs()
    >>> if fs is not None:
    >>>     fs.draw()

    **NOTES**

    .. Warning::
      For blobs that live right on the edge of the image OpenCV reports the
      position and width height as being one over for the true position.
      E.g. if a blob is at (0,0) OpenCV reports its position as (1,1).
      Likewise the width and height for the other corners is reported as
      being one less than the width and height. This is a known bug.

    **SEE ALSO**
    :py:meth:`threshold`
    :py:meth:`binarize`
    :py:meth:`invert`
    :py:meth:`dilate`
    :py:meth:`erode`
    :py:meth:`find_blobs_from_palette`
    :py:meth:`smart_find_blobs`
    """
    if maxsize == 0:
        maxsize = img.width * img.height
    #create a single channel image, thresholded to parameters

    blobmaker = BlobMaker()
    blobs = blobmaker.extract_from_binary(
        img.binarize(thresh=threshval, maxv=255, blocksize=threshblocksize,
                     p=threshconstant).invert(),
        img, minsize=minsize, maxsize=maxsize, appx_level=appx_level)

    if not len(blobs):
        return None

    return FeatureSet(blobs).sort_area()


@image_method
def find_skintone_blobs(img, minsize=10, maxsize=0, dilate_iter=1):
    """
    **SUMMARY**

    Find Skintone blobs will look for continuous
    regions of Skintone in a color image and return them as Blob features
    in a FeatureSet. Parameters specify the binarize filter threshold
    value, and minimum and maximum size for blobs. If a threshold value is
    -1, it will use an adaptive threshold.  See binarize() for more
    information about thresholding.  The threshblocksize and threshconstant
    parameters are only used for adaptive threshold.


    **PARAMETERS**

    * *minsize* - the minimum size of the blobs, in pixels, of the returned
     blobs. This helps to filter out noise.

    * *maxsize* - the maximim size of the blobs, in pixels, of the returned
     blobs.

    * *dilate_iter* - the number of times to run the dilation operation.

    **RETURNS**

    Returns a featureset (basically a list) of :py:class:`blob` features.
    If no blobs are found this method returns None.

    **EXAMPLE**

    >>> img = Image("lenna")
    >>> fs = img.find_skintone_blobs()
    >>> if fs is not None:
    >>>     fs.draw()

    **NOTES**
    It will be really awesome for making UI type stuff, where you want to
    track a hand or a face.

    **SEE ALSO**
    :py:meth:`threshold`
    :py:meth:`binarize`
    :py:meth:`invert`
    :py:meth:`dilate`
    :py:meth:`erode`
    :py:meth:`find_blobs_from_palette`
    :py:meth:`smart_find_blobs`
    """
    if maxsize == 0:
        maxsize = img.width * img.height
    mask = img.get_skintone_mask(dilate_iter)
    blobmaker = BlobMaker()
    blobs = blobmaker.extract_from_binary(mask, img, minsize=minsize,
                                          maxsize=maxsize)
    if not len(blobs):
        return None
    return FeatureSet(blobs).sort_area()


# this code is based on code that's based on code from
# http://blog.jozilla.net/2008/06/27/
# fun-with-python-opencv-and-face-detection/
@image_method
def find_haar_features(img, cascade, scale_factor=1.2, min_neighbors=2,
                       use_canny=cv2.cv.CV_HAAR_DO_CANNY_PRUNING,
                       min_size=(20, 20), max_size=(1000, 1000)):
    """
    **SUMMARY**

    A Haar like feature cascase is a really robust way of finding the
    location of a known object. This technique works really well for a few
    specific applications like face, pedestrian, and vehicle detection. It
    is worth noting that this approach **IS NOT A MAGIC BULLET** . Creating
    a cascade file requires a large number of images that have been sorted
    by a human.vIf you want to find Haar Features (useful for face
    detection among other purposes) this will return Haar feature objects
    in a FeatureSet.

    For more information, consult the cv2.CascadeClassifier documentation.

    To see what features are available run img.list_haar_features() or you
    can provide your own haarcascade file if you have one available.

    Note that the cascade parameter can be either a filename, or a
    HaarCascade loaded with cv2.CascadeClassifier(),
    or a SimpleCV HaarCascade object.

    **PARAMETERS**

    * *cascade* - The Haar Cascade file, this can be either the path to a
      cascade file or a HaarCascased SimpleCV object that has already been
      loaded.

    * *scale_factor* - The scaling factor for subsequent rounds of the
      Haar cascade (default 1.2) in terms of a percentage
      (i.e. 1.2 = 20% increase in size)

    * *min_neighbors* - The minimum number of rectangles that makes up an
      object. Ususally detected faces are clustered around the face, this
      is the number of detections in a cluster that we need for detection.
      Higher values here should reduce false positives and decrease false
      negatives.

    * *use-canny* - Whether or not to use Canny pruning to reject areas
     with too many edges (default yes, set to 0 to disable)

    * *min_size* - Minimum window size. By default, it is set to the size
      of samples the classifier has been trained on ((20,20) for face
      detection)

    * *max_size* - Maximum window size. By default, it is set to the size
      of samples the classifier has been trained on ((1000,1000) for face
      detection)

    **RETURNS**

    A feature set of HaarFeatures

    **EXAMPLE**

    >>> faces = HaarCascade(
        ...         "./SimpleCV/data/Features/HaarCascades/face.xml",
        ...         "myFaces")
    >>> cam = Camera()
    >>> while True:
    >>>     f = cam.get_image().find_haar_features(faces)
    >>>     if f is not None:
    >>>          f.show()

    **NOTES**

    OpenCV Docs:
    - http://opencv.willowgarage.com/documentation/python/
      objdetect_cascade_classification.html

    Wikipedia:
    - http://en.wikipedia.org/wiki/Viola-Jones_object_detection_framework
    - http://en.wikipedia.org/wiki/Haar-like_features

    The video on this pages shows how Haar features and cascades work to
    located faces:
    - http://dismagazine.com/dystopia/evolved-lifestyles/8115/
    anti-surveillance-how-to-hide-from-machines/

    """
    if isinstance(cascade, basestring):
        cascade = HaarCascade(cascade)
        if not cascade.get_cascade():
            return None
    elif isinstance(cascade, HaarCascade):
        pass
    else:
        logger.warning('Could not initialize HaarCascade. '
                       'Enter Valid cascade value.')
        return None

    haar_classify = cv2.CascadeClassifier(cascade.get_fhandle())
    objects = haar_classify.detectMultiScale(
        img.get_gray_ndarray(), scaleFactor=scale_factor,
        minNeighbors=min_neighbors, minSize=min_size,
        flags=use_canny)

    if objects is not None and len(objects) != 0:
        return FeatureSet(
            [Factory.HaarFeature(img, o, cascade, True) for o in objects])

    return None


 #this function contains two functions -- the basic edge detection algorithm
#and then a function to break the lines down given a threshold parameter
@image_method
def find_lines(img, threshold=80, minlinelength=30, maxlinegap=10,
               cannyth1=50, cannyth2=100, use_standard=False, nlines=-1,
               maxpixelgap=1):
    """
    **SUMMARY**

    find_lines will find line segments in your image and returns line
    feature objects in a FeatureSet. This method uses the Hough
    (pronounced "HUFF") transform.

    See http://en.wikipedia.org/wiki/Hough_transform

    **PARAMETERS**

    * *threshold* - which determines the minimum "strength" of the line.
    * *minlinelength* - how many pixels long the line must be to be
     returned.
    * *maxlinegap* - how much gap is allowed between line segments to
     consider them the same line .
    * *cannyth1* - thresholds used in the edge detection step, refer to
     :py:meth:`_get_edge_map` for details.
    * *cannyth2* - thresholds used in the edge detection step, refer to
     :py:meth:`_get_edge_map` for details.
    * *use_standard* - use standard or probabilistic Hough transform.
    * *nlines* - maximum number of lines for return.
    * *maxpixelgap* - how much distance between pixels is allowed to
     consider them the same line.

    **RETURNS**

    Returns a :py:class:`FeatureSet` of :py:class:`Line` objects. If no
     lines are found the method returns None.

    **EXAMPLE**

    >>> img = Image("lenna")
    >>> lines = img.find_lines()
    >>> lines.draw()
    >>> img.show()

    **SEE ALSO**
    :py:class:`FeatureSet`
    :py:class:`Line`
    :py:meth:`edges`

    """
    em = Factory.Image.get_edge_map(img, cannyth1, cannyth2)

    lines_fs = FeatureSet()
    if use_standard:
        lines = cv2.HoughLines(em, rho=1.0, theta=math.pi/180.0,
                               threshold=threshold, srn=minlinelength,
                               stn=maxlinegap)[0]
        if nlines == -1:
            nlines = lines.shape[0]
        # All white points (edges) in Canny edge image
        y, x = np.where(em > 128)  #
        # Put points in dictionary for fast checkout if point is white
        pts = dict((p, 1) for p in zip(x, y))

        w, h = img.width - 1, img.height - 1
        for rho, theta in lines[:nlines]:
            ep = []
            ls = []
            a = math.cos(theta)
            b = math.sin(theta)
            # Find endpoints of line on the image's edges
            if round(b, 4) == 0:  # slope of the line is infinity
                ep.append((int(round(abs(rho))), 0))
                ep.append((int(round(abs(rho))), h))
            elif round(a, 4) == 0:  # slope of the line is zero
                ep.append((0, int(round(abs(rho)))))
                ep.append((w, int(round(abs(rho)))))
            else:
                # top edge
                x = rho / float(a)
                if 0 <= x <= w:
                    ep.append((int(round(x)), 0))
                # bottom edge
                x = (rho - h * b) / float(a)
                if 0 <= x <= w:
                    ep.append((int(round(x)), h))
                # left edge
                y = rho / float(b)
                if 0 <= y <= h:
                    ep.append((0, int(round(y))))
                # right edge
                y = (rho - w * a) / float(b)
                if 0 <= y <= h:
                    ep.append((w, int(round(y))))
            # remove duplicates if line crosses the image at corners
            ep = list(set(ep))
            ep.sort()
            brl = img.bresenham_line(ep[0], ep[1])

            # Follow the points on Bresenham's line. Look for white points.
            # If the distance between two adjacent white points (dist) is
            # less than or equal maxpixelgap then consider them the same
            # line. If dist is bigger maxpixelgap then check if length of
            # the line is bigger than minlinelength. If so then add line.

            # distance between two adjacent white points
            dist = float('inf')
            len_l = float('-inf')  # length of the line
            for p in brl:
                if p in pts:
                    # found the end of the previous line and
                    # the start of the new line
                    if dist > maxpixelgap:
                        if len_l >= minlinelength:
                            if ls:
                                # If the gap between current line and
                                # previous is less than maxlinegap then
                                # merge this lines
                                l = ls[-1]
                                gap = round(math.sqrt(
                                    (start_p[0] - l[1][0]) ** 2 +
                                    (start_p[1] - l[1][1]) ** 2))
                                if gap <= maxlinegap:
                                    ls.pop()
                                    start_p = l[0]
                            ls.append((start_p, last_p))
                        # First white point of the new line found
                        dist = 1
                        len_l = 1
                        start_p = p  # first endpoint of the line
                    else:
                        # dist is less than or equal maxpixelgap,
                        # so line doesn't end yet
                        len_l += dist
                        dist = 1
                    last_p = p  # last white point
                else:
                    dist += 1

            for l in ls:
                lines_fs.append(Factory.Line(img, l))
        lines_fs = lines_fs[:nlines]
    else:
        lines = cv2.HoughLinesP(em, rho=1.0, theta=math.pi/180.0,
                                threshold=threshold,
                                minLineLength=minlinelength,
                                maxLineGap=maxlinegap)[0]
        if nlines == -1:
            nlines = lines.shape[0]

        for l in lines[:nlines]:
            lines_fs.append(Factory.Line(img, ((l[0], l[1]), (l[2], l[3]))))

    return lines_fs


@image_method
def find_chessboard(img, dimensions=(8, 5), subpixel=True):
    """
    **SUMMARY**

    Given an image, finds a chessboard within that image.  Returns the
    Chessboard featureset.
    The Chessboard is typically used for calibration because of its evenly
    spaced corners.


    The single parameter is the dimensions of the chessboard, typical one
    can be found in \SimpleCV\tools\CalibGrid.png

    **PARAMETERS**

    * *dimensions* - A tuple of the size of the chessboard in width and
     height in grid objects.
    * *subpixel* - Boolean if True use sub-pixel accuracy, otherwise use
     regular pixel accuracy.

    **RETURNS**

    A :py:class:`FeatureSet` of :py:class:`Chessboard` objects. If no
     chessboards are found None is returned.

    **EXAMPLE**

    >>> img = cam.getImage()
    >>> cb = img.find_chessboard()
    >>> cb.draw()

    **SEE ALSO**

    :py:class:`FeatureSet`
    :py:class:`Chessboard`

    """
    gray_array = img.get_gray_ndarray()
    equalized_grayscale_array = cv2.equalizeHist(gray_array)
    found, corners = cv2.findChessboardCorners(
        equalized_grayscale_array, patternSize=dimensions,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)

    if not found:
        return None

    if corners is not None and len(corners) == dimensions[0] * dimensions[1]:
        if subpixel:
            sp_corners = cv2.cornerSubPix(
                gray_array, corners=corners[1], winSize=(11, 11),
                zeroZone=(-1, -1),
                criteria=(cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS,
                          10, 0.01))
            if sp_corners is None:
                logger.warning("subpixel corners not found. Returning None.")
                return None
        else:
            sp_corners = corners
        return FeatureSet([Factory.Chessboard(img, dimensions, sp_corners)])
    else:
        return None


@image_method
def find_template(img, template_image=None, threshold=5,
                  method="SQR_DIFF_NORM", grayscale=True,
                  rawmatches=False):
    """
    **SUMMARY**

    This function searches an image for a template image.  The template
    image is a smaller image that is searched for in the bigger image.
    This is a basic pattern finder in an image.  This uses the standard
    OpenCV template (pattern) matching and cannot handle scaling or
    rotation

    Template matching returns a match score for every pixel in the image.
    Often pixels that are near to each other and a close match to the
    template are returned as a match. If the threshold is set too low
    expect to get a huge number of values. The threshold parameter is in
    terms of the number of standard deviations from the mean match value
    you are looking

    For example, matches that are above three standard deviations will
    return 0.1% of the pixels. In a 800x600 image this means there will be
    800*600*0.001 = 480 matches.

    This method returns the locations of wherever it finds a match above a
    threshold. Because of how template matching works, very often multiple
    instances of the template overlap significantly. The best approach is
    to find the centroid of all of these values. We suggest using an
    iterative k-means approach to find the centroids.


    **PARAMETERS**

    * *template_image* - The template image.
    * *threshold* - Int
    * *method* -

      * SQR_DIFF_NORM - Normalized square difference
      * SQR_DIFF      - Square difference
      * CCOEFF        -
      * CCOEFF_NORM   -
      * CCORR         - Cross correlation
      * CCORR_NORM    - Normalize cross correlation
    * *grayscale* - Boolean - If false, template Match is found using BGR
     image.

    **EXAMPLE**

    >>> image = Image("/path/to/img.png")
    >>> pattern_image = image.crop(100, 100, 100, 100)
    >>> found_patterns = image.find_template(pattern_image)
    >>> found_patterns.draw()
    >>> image.show()

    **RETURNS**

    This method returns a FeatureSet of TemplateMatch objects.

    """
    if template_image is None:
        logger.info("Need image for matching")
        return
    if template_image.width > img.width:
        logger.info("Image too wide")
        return
    if template_image.height > img.height:
        logger.info("Image too tall")
        return

    check = 0  # if check = 0 we want maximal value, otherwise minimal
    # minimal
    if method is None or method == "" or method == "SQR_DIFF_NORM":
        method = cv2.TM_SQDIFF_NORMED
        check = 1
    elif method == "SQR_DIFF":  # minimal
        method = cv2.TM_SQDIFF
        check = 1
    elif method == "CCOEFF":  # maximal
        method = cv2.TM_CCOEFF
    elif method == "CCOEFF_NORM":  # maximal
        method = cv2.TM_CCOEFF_NORMED
    elif method == "CCORR":  # maximal
        method = cv2.TM_CCORR
    elif method == "CCORR_NORM":  # maximal
        method = cv2.TM_CCORR_NORMED
    else:
        logger.warning("ooops.. I don't know what template matching "
                       "method you are looking for.")
        return None

    #choose template matching method to be used
    if grayscale:
        img_array = img.get_gray_ndarray()
        template_array = template_image.get_gray_ndarray()
    else:
        img_array = img.get_ndarray()
        template_array = template_image.get_ndarray()

    matches = cv2.matchTemplate(img_array, templ=template_array, method=method)
    mean = np.mean(matches)
    sd = np.std(matches)
    if check > 0:
        compute = np.where((matches < mean - threshold * sd))
    else:
        compute = np.where((matches > mean + threshold * sd))

    mapped = map(tuple, np.column_stack(compute))
    fs = FeatureSet()
    for location in mapped:
        fs.append(
            Factory.TemplateMatch(img, template_image,
                                  (location[1], location[0]),
                                  matches[location[0], location[1]]))

    if rawmatches:
        return fs
    # cluster overlapping template matches
    finalfs = FeatureSet()
    if len(fs) > 0:
        finalfs.append(fs[0])
        for f in fs:
            match = False
            for f2 in finalfs:
                if f2._template_overlaps(f):  # if they overlap
                    f2.consume(f)  # merge them
                    match = True
                    break

            if not match:
                finalfs.append(f)

        # rescale the resulting clusters to fit the template size
        for f in finalfs:
            f.rescale(template_image.width, template_image.height)
        fs = finalfs
    return fs


@image_method
def find_template_once(img, template_image=None, threshold=0.2,
                       method="SQR_DIFF_NORM", grayscale=True):
    """
    **SUMMARY**

    This function searches an image for a single template image match.The
    template image is a smaller image that is searched for in the bigger
    image. This is a basic pattern finder in an image.  This uses the
    standard OpenCV template (pattern) matching and cannot handle scaling
    or rotation

    This method returns the single best match if and only if that
    match less than the threshold (greater than in the case of
    some methods).

    **PARAMETERS**

    * *template_image* - The template image.
    * *threshold* - Int
    * *method* -

      * SQR_DIFF_NORM - Normalized square difference
      * SQR_DIFF      - Square difference
      * CCOEFF        -
      * CCOEFF_NORM   -
      * CCORR         - Cross correlation
      * CCORR_NORM    - Normalize cross correlation
    * *grayscale* - Boolean - If false, template Match is found using BGR
     image.

    **EXAMPLE**

    >>> image = Image("/path/to/img.png")
    >>> pattern_image = image.crop(100, 100, 100, 100)
    >>> found_patterns = image.find_template_once(pattern_image)
    >>> found_patterns.draw()
    >>> image.show()

    **RETURNS**

    This method returns a FeatureSet of TemplateMatch objects.

    """
    if template_image is None:
        logger.info("Need image for template matching.")
        return
    if template_image.width > img.width:
        logger.info("Template image is too wide for the given image.")
        return
    if template_image.height > img.height:
        logger.info("Template image too tall for the given image.")
        return

    check = 0  # if check = 0 we want maximal value, otherwise minimal
    # minimal
    if method is None or method == "" or method == "SQR_DIFF_NORM":
        method = cv2.TM_SQDIFF_NORMED
        check = 1
    elif method == "SQR_DIFF":  # minimal
        method = cv2.TM_SQDIFF
        check = 1
    elif method == "CCOEFF":  # maximal
        method = cv2.TM_CCOEFF
    elif method == "CCOEFF_NORM":  # maximal
        method = cv2.TM_CCOEFF_NORMED
    elif method == "CCORR":  # maximal
        method = cv2.TM_CCORR
    elif method == "CCORR_NORM":  # maximal
        method = cv2.TM_CCORR_NORMED
    else:
        logger.warning("ooops.. I don't know what template matching "
                       "method you are looking for.")
        return None
    #choose template matching method to be used
    if grayscale:
        img_array = img.get_gray_ndarray()
        template_array = template_image.get_gray_ndarray()
    else:
        img_array = img.get_ndarray()
        template_array = template_image.get_ndarray()

    matches = cv2.matchTemplate(img_array, templ=template_array, method=method)
    if check > 0:
        if np.min(matches) <= threshold:
            compute = np.where(matches == np.min(matches))
        else:
            return []
    else:
        if np.max(matches) >= threshold:
            compute = np.where(matches == np.max(matches))
        else:
            return []
    mapped = map(tuple, np.column_stack(compute))
    fs = FeatureSet()
    for location in mapped:
        fs.append(
            Factory.TemplateMatch(img, template_image, (location[1],
                                                        location[0]),
                                  matches[location[0], location[1]]))
    return fs


@image_method
def find_circle(img, canny=100, thresh=350, distance=-1):
    """
    **SUMMARY**

    Perform the Hough Circle transform to extract _perfect_ circles from
    the image canny - the upper bound on a canny edge detector used to find
    circle edges.

    **PARAMETERS**

    * *thresh* - the threshold at which to count a circle. Small parts of
      a circle get added to the accumulator array used internally to the
      array. This value is the minimum threshold. Lower thresholds give
      more circles, higher thresholds give fewer circles.

    .. ::Warning:
      If this threshold is too high, and no circles are found the
      underlying OpenCV routine fails and causes a segfault.

    * *distance* - the minimum distance between each successive circle in
      pixels. 10 is a good starting value.

    **RETURNS**

    A feature set of Circle objects.

    **EXAMPLE**

    >>> img = Image("lenna")
    >>> circs = img.find_circle()
    >>> for c in circs:
    >>>    print c
    """
    # a distnace metric for how apart our circles should be
    # this is sa good bench mark
    if distance < 0:
        distance = 1 + max(img.width, img.height) / 50

    circs = cv2.HoughCircles(img.get_gray_ndarray(),
                             method=cv2.cv.CV_HOUGH_GRADIENT,
                             dp=2, minDist=distance,
                             param1=canny, param2=thresh)
    if circs is None:
        return None
    circle_fs = FeatureSet()
    for circ in circs[0]:
        circle_fs.append(Factory.Circle(img, int(circ[0]), int(circ[1]),
                                        int(circ[2])))
    return circle_fs


@image_method
def find_keypoint_match(img, template, quality=500.00, min_dist=0.2,
                        min_match=0.4):
    """
    **SUMMARY**

    find_keypoint_match allows you to match a template image with another
    image using SURF keypoints. The method extracts keypoints from each
    image, uses the Fast Local Approximate Nearest Neighbors algorithm to
    find correspondences between the feature points, filters the
    correspondences based on quality, and then, attempts to calculate
    a homography between the two images. This homography allows us to draw
    a matching bounding box in the source image that corresponds to the
    template. This method allows you to perform matchs the ordinarily fail
    when using the find_template method. This method should be able to
    handle a reasonable changes in camera orientation and illumination.
    Using a template that is close to the target image will yield much
    better results.

    .. Warning::
      This method is only capable of finding one instance of the template
      in an image. If more than one instance is visible the homography
      calculation and the method will fail.

    **PARAMETERS**

    * *template* - A template image.
    * *quality* - The feature quality metric. This can be any value between
      about 300 and 500. Higher values should return fewer, but higher
      quality features.
    * *min_dist* - The value below which the feature correspondence is
       considered a match. This is the distance between two feature
       vectors. Good values are between 0.05 and 0.3
    * *min_match* - The percentage of features which must have matches to
      proceed with homography calculation. A value of 0.4 means 40% of
      features must match. Higher values mean better matches are used.
      Good values are between about 0.3 and 0.7


    **RETURNS**

    If a homography (match) is found this method returns a feature set with
    a single KeypointMatch feature. If no match is found None is returned.

    **EXAMPLE**

    >>> template = Image("template.png")
    >>> img = camera.getImage()
    >>> fs = img.find_keypoint_match(template)
    >>> if fs is not None:
    >>>      fs.draw()
    >>>      img.show()

    **NOTES**

    If you would prefer to work with the raw keypoints and descriptors each
    image keeps a local cache of the raw values. These are named:

    | self._key_points # A Tuple of keypoint objects
    | self._kp_descriptors # The descriptor as a floating point numpy array
    | self._kp_flavor = "NONE" # The flavor of the keypoints as a string.
    | `See Documentation <http://opencv.itseez.com/modules/features2d/doc/
    | common_interfaces_of_feature_detectors.html#keypoint-keypoint>`_

    **SEE ALSO**

    :py:meth:`_get_raw_keypoints`
    :py:meth:`_get_flann_matches`
    :py:meth:`draw_keypoint_matches`
    :py:meth:`find_keypoints`

    """
    if template is None:
        return None

    skp, sd = img._get_raw_keypoints(quality)
    tkp, td = template._get_raw_keypoints(quality)
    if skp is None or tkp is None:
        logger.warn("I didn't get any keypoints. Image might be too "
                    "uniform or blurry.")
        return None

    template_points = float(td.shape[0])
    sample_points = float(sd.shape[0])
    magic_ratio = 1.00
    if sample_points > template_points:
        magic_ratio = float(sd.shape[0]) / float(td.shape[0])

    # match our keypoint descriptors
    idx, dist = img._get_flann_matches(sd, td)
    p = dist[:, 0]
    result = p * magic_ratio < min_dist
    pr = result.shape[0] / float(dist.shape[0])

    # if more than min_match % matches we go ahead and get the data
    if pr > min_match and len(result) > 4:
        lhs = []
        rhs = []
        for i in range(0, len(idx)):
            if result[i]:
                lhs.append((tkp[i].pt[1], tkp[i].pt[0]))
                rhs.append((skp[idx[i]].pt[0], skp[idx[i]].pt[1]))

        rhs_pt = np.array(rhs)
        lhs_pt = np.array(lhs)
        if len(rhs_pt) < 16 or len(lhs_pt) < 16:
            return None
        (homography, mask) = cv2.findHomography(srcPoints=lhs_pt,
                                                dstPoints=rhs_pt,
                                                method=cv2.RANSAC,
                                                ransacReprojThreshold=1.0)
        w, h = template.size

        pts = np.array([[0, 0], [0, h], [w, h], [w, 0]], dtype=np.float32)

        ppts = cv2.perspectiveTransform(np.array([pts]), m=homography)

        pt0i = (ppts[0][0][0], ppts[0][0][1])
        pt1i = (ppts[0][1][0], ppts[0][1][1])
        pt2i = (ppts[0][2][0], ppts[0][2][1])
        pt3i = (ppts[0][3][0], ppts[0][3][1])

        #construct the feature set and return it.
        fs = FeatureSet()
        fs.append(Factory.KeypointMatch(img, template,
                                        (pt0i, pt1i, pt2i, pt3i),
                                        homography))
        # the homography matrix is necessary for many purposes like image
        # stitching.
        # No need to add homography as it is already being
        # fs.append(homography)
        # added in KeyPointMatch class.
        return fs
    else:
        return None


@image_method
def find_keypoints(img, min_quality=300.00, flavor="SURF",
                   highquality=False):
    """
    **SUMMARY**

    This method finds keypoints in an image and returns them as a feature
    set. Keypoints are unique regions in an image that demonstrate some
    degree of invariance to changes in camera pose and illumination. They
    are helpful for calculating homographies between camera views, object
    rotations, and multiple view overlaps.

    We support four keypoint detectors and only one form of keypoint
    descriptors. Only the surf flavor of keypoint returns feature and
    descriptors at this time.

    **PARAMETERS**

    * *min_quality* - The minimum quality metric for SURF descriptors.
      Good values range between about 300.00 and 600.00

    * *flavor* - a string indicating the method to use to extract features.
      A good primer on how feature/keypoint extractiors can be found in
      `feature detection on wikipedia <http://en.wikipedia.org/wiki/
      Feature_detection_(computer_vision)>`_
      and
      `this tutorial. <http://www.cg.tu-berlin.de/fileadmin/fg144/
      Courses/07WS/compPhoto/Feature_Detection.pdf>`_


      * "SURF" - extract the SURF features and descriptors. If you don't
       know what to use, use this.

        See: http://en.wikipedia.org/wiki/SURF

      * "STAR" - The STAR feature extraction algorithm

        See: http://pr.willowgarage.com/wiki/Star_Detector

      * "FAST" - The FAST keypoint extraction algorithm

        See: http://en.wikipedia.org/wiki/
        Corner_detection#AST_based_feature_detectors

      All the flavour specified below are for OpenCV versions >= 2.4.0 :

      * "MSER" - Maximally Stable Extremal Regions algorithm

        See: http://en.wikipedia.org/wiki/Maximally_stable_extremal_regions

      * "Dense" -

      * "ORB" - The Oriented FAST and Rotated BRIEF

        See: http://www.willowgarage.com/sites/default/files/orb_final.pdf

      * "SIFT" - Scale-invariant feature transform

        See: http://en.wikipedia.org/wiki/Scale-invariant_feature_transform

      * "BRISK" - Binary Robust Invariant Scalable Keypoints

        See: http://www.asl.ethz.ch/people/lestefan/personal/BRISK

       * "FREAK" - Fast Retina Keypoints

         See: http://www.ivpe.com/freak.htm
         Note: It's a keypoint descriptor and not a KeyPoint detector.
         SIFT KeyPoints are detected and FERAK is used to extract
         keypoint descriptor.

    * *highquality* - The SURF descriptor comes in two forms, a vector of
      64 descriptor values and a vector of 128 descriptor values. The
      latter are "high" quality descriptors.

    **RETURNS**

    A feature set of KeypointFeatures. These KeypointFeatures let's you
    draw each feature, crop the features, get the feature descriptors, etc.

    **EXAMPLE**

    >>> img = Image("aerospace.jpg")
    >>> fs = img.find_keypoints(flavor="SURF", min_quality=500,
        ...                    highquality=True)
    >>> fs = fs.sort_area()
    >>> fs[-1].draw()
    >>> img.draw()

    **NOTES**

    If you would prefer to work with the raw keypoints and descriptors each
    image keeps a local cache of the raw values. These are named:

    :py:meth:`_get_raw_keypoints`
    :py:meth:`_get_flann_matches`
    :py:meth:`draw_keypoint_matches`
    :py:meth:`find_keypoints`

    """

    fs = FeatureSet()
    kp, d = img._get_raw_keypoints(thresh=min_quality,
                                   force_reset=True,
                                   flavor=flavor,
                                   highquality=int(highquality))

    if flavor in ["ORB", "SIFT", "SURF", "BRISK", "FREAK"] \
            and kp is not None and d is not None:
        for i in range(0, len(kp)):
            fs.append(Factory.KeyPoint(img, kp[i], d[i], flavor))
    elif flavor in ["FAST", "STAR", "MSER", "Dense"] and kp is not None:
        for i in range(0, len(kp)):
            fs.append(Factory.KeyPoint(img, kp[i], None, flavor))
    else:
        logger.warning("ImageClass.Keypoints: I don't know the method "
                       "you want to use")
        return None

    return fs


@image_method
def find_motion(img, previous_frame, window=11, aggregate=True):
    """
    **SUMMARY**

    find_motion performs an optical flow calculation. This method attempts
    to find motion between two subsequent frames of an image. You provide
    it with the previous frame image and it returns a feature set of motion
    fetures that are vectors in the direction of motion.

    **PARAMETERS**

    * *previous_frame* - The last frame as an Image.
    * *window* - The block size for the algorithm. For the the HS and LK
      methods this is the regular sample grid at which we return motion
      samples. For the block matching method this is the matching window
      size.
    * *method* - The algorithm to use as a string.
      Your choices are:

      * 'BM' - default block matching robust but slow - if you are unsure
       use this.

      * 'LK' - `Lucas-Kanade method <http://en.wikipedia.org/
      wiki/Lucas%E2%80%93Kanade_method>`_

      * 'HS' - `Horn-Schunck method <http://en.wikipedia.org/
      wiki/Horn%E2%80%93Schunck_method>`_

    * *aggregate* - If aggregate is true, each of our motion features is
      the average of motion around the sample grid defined by window. If
      aggregate is false we just return the the value as sampled at the
      window grid interval. For block matching this flag is ignored.

    **RETURNS**

    A featureset of motion objects.

    **EXAMPLES**

    >>> cam = Camera()
    >>> img1 = cam.getImage()
    >>> img2 = cam.getImage()
    >>> motion = img2.find_motion(img1)
    >>> motion.draw()
    >>> img2.show()

    **SEE ALSO**

    :py:class:`Motion`
    :py:class:`FeatureSet`

    """
    if img.size != previous_frame.size:
        logger.warning("Image.find_motion: To find motion the current "
                       "and previous frames must match")
        return None

    flow = cv2.calcOpticalFlowFarneback(prev=previous_frame.get_gray_ndarray(),
                                        next=img.get_gray_ndarray(),
                                        pyr_scale=0.5, levels=1,
                                        winsize=window, iterations=1,
                                        poly_n=7, poly_sigma=1.5, flags=0,
                                        flow=None)
    fs = FeatureSet()
    max_mag = 0.00
    w = math.floor(float(window) / 2.0)
    cx = ((img.width - window) / window) + 1  # our sample rate
    cy = ((img.height - window) / window) + 1
    xf = flow[:, :, 0]
    yf = flow[:, :, 1]
    for x in range(0, int(cx)):  # go through our sample grid
        for y in range(0, int(cy)):
            xi = (x * window) + w  # calculate the sample point
            yi = (y * window) + w
            if aggregate:
                lowx = int(xi - w)
                highx = int(xi + w)
                lowy = int(yi - w)
                highy = int(yi + w)
                # get the average x/y components in the output
                xderp = xf[lowy:highy, lowx:highx]
                yderp = yf[lowy:highy, lowx:highx]
                vx = np.average(xderp)
                vy = np.average(yderp)
            else:  # other wise just sample
                vx = xf[yi, xi]
                vy = yf[yi, xi]

            mag = (vx * vx) + (vy * vy)
            # calculate the max magnitude for normalizing our vectors
            if mag > max_mag:
                max_mag = mag
            # add the sample to the feature set
            fs.append(Factory.Motion(img, xi, yi, vx, vy, window))
    return fs


@image_method
def find_blobs_from_palette(img, palette_selection, dilate=0, minsize=5,
                            maxsize=0, appx_level=3):
    """
    **SUMMARY**

    This method attempts to use palettization to do segmentation and
    behaves similar to the find_blobs blob in that it returs a feature set
    of blob objects. Once a palette has been extracted using get_palette()
    we can then select colors from that palette to be labeled white within
    our blobs.

    **PARAMETERS**

    * *palette_selection* - color triplets selected from our palette that
      will serve turned into blobs. These values can either be a 3xN numpy
      array, or a list of RGB triplets.
    * *dilate* - the optional number of dilation operations to perform on
      the binary image prior to performing blob extraction.
    * *minsize* - the minimum blob size in pixels
    * *maxsize* - the maximim blob size in pixels.
    * *appx_level* - The blob approximation level - an integer for the
      maximum distance between the true edge and the approximation edge -
      lower numbers yield better approximation.

    **RETURNS**

    If the method executes successfully a FeatureSet of Blobs is returned
    from the image. If the method fails a value of None is returned.

   **EXAMPLE**

    >>> img = Image("lenna")
    >>> p = img.get_palette()
    >>> blobs = img.find_blobs_from_palette((p[0], p[1], p[6]))
    >>> blobs.draw()
    >>> img.show()

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
    bwimg = img.binarize_from_palette(palette_selection)
    if dilate > 0:
        bwimg = bwimg.dilate(dilate)

    if maxsize == 0:
        maxsize = img.width * img.height
    #create a single channel image, thresholded to parameters

    blobmaker = BlobMaker()
    blobs = blobmaker.extract_from_binary(bwimg,
                                          img, minsize=minsize,
                                          maxsize=maxsize,
                                          appx_level=appx_level)
    if not len(blobs):
        return None
    return blobs


@image_method
def smart_find_blobs(img, mask=None, rect=None, thresh_level=2,
                     appx_level=3):
    """
    **SUMMARY**

    smart_find_blobs uses a method called grabCut, also called graph cut,
    to  automagically determine the boundary of a blob in the image. The
    dumb find blobs just uses color threshold to find the boundary,
    smart_find_blobs looks at both color and edges to find a blob. To work
    smart_find_blobs needs either a rectangle that bounds the object you
    want to find, or a mask. If you use a rectangle make sure it holds the
    complete object. In the case of a mask, it need not be a normal binary
    mask, it can have the normal white foreground and black background, but
    also a light and dark gray values that correspond to areas that are
    more likely to be foreground and more likely to be background. These
    values can be found in the color class as Color.BACKGROUND,
    Color.FOREGROUND, Color.MAYBE_BACKGROUND, and Color.MAYBE_FOREGROUND.

    **PARAMETERS**

    * *mask* - A grayscale mask the same size as the image using the 4 mask
     color values
    * *rect* - A rectangle tuple of the form (x_position, y_position,
     width, height)
    * *thresh_level* - This represents what grab cut values to use in the
     mask after the graph cut algorithm is run,

      * 1  - means use the foreground, maybe_foreground, and
        maybe_background values
      * 2  - means use the foreground and maybe_foreground values.
      * 3+ - means use just the foreground

    * *appx_level* - The blob approximation level - an integer for the
      maximum distance between the true edge and the approximation edge -
      lower numbers yield better approximation.


    **RETURNS**

    A featureset of blobs. If everything went smoothly only a couple of
    blobs should be present.

    **EXAMPLE**

    >>> img = Image("RatTop.png")
    >>> mask = Image((img.width,img.height))
    >>> mask.dl().circle((100, 100), 80, color=Color.MAYBE_BACKGROUND,
        ...              filled=True
    >>> mask.dl().circle((100, 100), 60, color=Color.MAYBE_FOREGROUND,
        ...              filled=True)
    >>> mask.dl().circle((100, 100), 40, color=Color.FOREGROUND,
        ...              filled=True)
    >>> mask = mask.apply_layers()
    >>> blobs = img.smart_find_blobs(mask=mask)
    >>> blobs.draw()
    >>> blobs.show()

    **NOTES**

    http://en.wikipedia.org/wiki/Graph_cuts_in_computer_vision

    **SEE ALSO**

    :py:meth:`smart_threshold`

    """
    result = img.smart_threshold(mask, rect)
    binary = None
    ret_val = None

    if result:
        if thresh_level == 1:
            result = result.threshold(192)
        elif thresh_level == 2:
            result = result.threshold(128)
        elif thresh_level > 2:
            result = result.threshold(1)
        bm = BlobMaker()
        ret_val = bm.extract_from_binary(result, img, appx_level)

    return ret_val


@image_method
def find_blobs_from_mask(img, mask, threshold=128, minsize=10, maxsize=0,
                         appx_level=3):
    """
    **SUMMARY**

    This method acts like find_blobs, but it lets you specifiy blobs
    directly by providing a mask image. The mask image must match the size
    of this image, and the mask should have values > threshold where you
    want the blobs selected. This method can be used with binarize, dialte,
    erode, flood_fill, edges etc to get really nice segmentation.

    **PARAMETERS**

    * *mask* - The mask image, areas lighter than threshold will be counted
      as blobs. Mask should be the same size as this image.
    * *threshold* - A single threshold value used when we binarize the
      mask.
    * *minsize* - The minimum size of the returned blobs.
    * *maxsize*  - The maximum size of the returned blobs, if none is
      specified we peg this to the image size.
    * *appx_level* - The blob approximation level - an integer for the
      maximum distance between the true edge and the approximation edge -
      lower numbers yield better approximation.


    **RETURNS**

    A featureset of blobs. If no blobs are found None is returned.

    **EXAMPLE**

    >>> img = Image("Foo.png")
    >>> mask = img.binarize().dilate(2)
    >>> blobs = img.find_blobs_from_mask(mask)
    >>> blobs.show()

    **SEE ALSO**

    :py:meth:`find_blobs`
    :py:meth:`binarize`
    :py:meth:`threshold`
    :py:meth:`dilate`
    :py:meth:`erode`
    """
    if maxsize == 0:
        maxsize = img.width * img.height
    #create a single channel image, thresholded to parameters
    if mask.size != img.size:
        logger.warning("Image.find_blobs_from_mask - your mask does "
                       "not match the size of your image")
        return None

    blobmaker = BlobMaker()
    gray = mask.get_gray_ndarray()
    val, result = cv2.threshold(gray, thresh=threshold, maxval=255,
                                type=cv2.THRESH_BINARY)
    blobs = blobmaker.extract_from_binary(
        Factory.Image(result), img,
        minsize=minsize, maxsize=maxsize, appx_level=appx_level)
    if not len(blobs):
        return None
    return FeatureSet(blobs).sort_area()


@image_method
def find_flood_fill_blobs(img, points, tolerance=None, lower=None,
                          upper=None,
                          fixed_range=True, minsize=30, maxsize=-1):
    """

    **SUMMARY**

    This method lets you use a flood fill operation and pipe the results to
    find_blobs. You provide the points to seed flood_fill and the rest is
    taken care of.

    flood_fill works just like ye olde paint bucket tool in your favorite
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
      flood fill.
    * *tolerance* - The color tolerance as a single value or a triplet.
    * *color* - The color to replace the flood_fill pixels with
    * *lower* - If tolerance does not provide enough control you can
      optionally set the upper and lower values around the seed pixel.
      This value can be a single value or a triplet. This will override
      the tolerance variable.
    * *upper* - If tolerance does not provide enough control you can
      optionally set the upper and lower values around the seed pixel.
       This value can be a single value or a triplet. This will override
      the tolerance variable.
    * *fixed_range* - If fixed_range is true we use the seed_pixel +/-
      tolerance. If fixed_range is false, the tolerance is +/- tolerance
      of the values of the adjacent pixels to the pixel under test.
    * *minsize* - The minimum size of the returned blobs.
    * *maxsize* - The maximum size of the returned blobs, if none is
      specified we peg this to the image size.

    **RETURNS**

    A featureset of blobs. If no blobs are found None is returned.

    An Image where the values similar to the seed pixel have been replaced
    by the input color.

    **EXAMPLE**

    >>> img = Image("lenna")
    >>> blerbs = img.find_flood_fill_blobs(((10, 10), (20, 20), (30, 30)),
        ...                             tolerance=30)
    >>> blerbs.show()

    **SEE ALSO**

    :py:meth:`find_blobs`
    :py:meth:`flood_fill`

    """
    mask = img.flood_fill_to_mask(points, tolerance, color=Color.WHITE,
                                  lower=lower, upper=upper,
                                  fixed_range=fixed_range)
    return img.find_blobs_from_mask(mask, minsize, maxsize)

@image_method
def list_haar_features(img):
    '''
    This is used to list the built in features available for HaarCascade
    feature detection.  Just run this function as:

    >>> img.list_haar_features()

    Then use one of the file names returned as the input to the
    findHaarFeature() function. So you should get a list, more than likely
    you will see face.xml, to use it then just

    >>> img.find_haar_features('face.xml')
    '''

    features_directory = os.path.join(DATA_DIR, 'Features/HaarCascades')
    features = os.listdir(features_directory)
    print features
    return features


@image_method
def anonymize(img, block_size=10, features=None, transform=None):
    """
    **SUMMARY**

    Anonymize, for additional privacy to images.

    **PARAMETERS**

    * *features* - A list with the Haar like feature cascades that should
       be matched.
    * *block_size* - The size of the blocks for the pixelize function.
    * *transform* - A function, to be applied to the regions matched
      instead of pixelize.
    * This function must take two arguments: the image and the region
      it'll be applied to,
    * as in region = (x, y, width, height).

    **RETURNS**

    Returns the image with matching regions pixelated.

    **EXAMPLE**

    >>> img = Image("lenna")
    >>> anonymous = img.anonymize()
    >>> anonymous.show()

    >>> def my_function(img, region):
    >>>     x, y, width, height = region
    >>>     img = img.crop(x, y, width, height)
    >>>     return img
    >>>
    >>>img = Image("lenna")
    >>>transformed = img.anonymize(transform = my_function)

    """

    regions = []

    if features is None:
        regions.append(img.find_haar_features("face.xml"))
        regions.append(img.find_haar_features("profile.xml"))
    else:
        for feature in features:
            regions.append(img.find_haar_features(feature))

    print regions
    found = [f for f in regions if f is not None]

    img = img.copy()

    if found:
        for feature_set in found:
            for region in feature_set:
                rect = (region.top_left_corner()[0],
                        region.top_left_corner()[1],
                        region.get_width(), region.get_height())
                if transform is None:
                    img = img.pixelize(block_size=block_size, region=rect)
                else:
                    img = transform(img, rect)
    return img


@image_method
def fit_edge(img, guess, window=10, threshold=128, measurements=5,
             darktolight=True, lighttodark=True, departurethreshold=1):
    """
    **SUMMARY**

    Fit edge in a binary/gray image using an initial guess and the least
    squares method. The functions returns a single line

    **PARAMETERS**

    * *guess* - A tuples of the form ((x0,y0),(x1,y1)) which is an
      approximate guess
    * *window* - A window around the guess to search.
    * *threshold* - the threshold above which we count a pixel as a line
    * *measurements* -the number of line projections to use for fitting
    the line

    TODO: Constrict a line to black to white or white to black
    Right vs. Left orientation.

    **RETURNS**

    A a line object
    **EXAMPLE**
    """
    search_lines = FeatureSet()
    fit_points = FeatureSet()
    x1 = guess[0][0]
    x2 = guess[1][0]
    y1 = guess[0][1]
    y2 = guess[1][1]
    dx = float((x2 - x1)) / (measurements - 1)
    dy = float((y2 - y1)) / (measurements - 1)
    s = np.zeros((measurements, 2))
    lpstartx = np.zeros(measurements)
    lpstarty = np.zeros(measurements)
    lpendx = np.zeros(measurements)
    lpendy = np.zeros(measurements)
    linefitpts = np.zeros((measurements, 2))

    # obtain equation for initial guess line
    # vertical line must be handled as special
    # case since slope isn't defined
    if x1 == x2:
        m = 0
        mo = 0
        b = x1
        for i in xrange(0, measurements):
            s[i][0] = x1
            s[i][1] = y1 + i * dy
            lpstartx[i] = s[i][0] + window
            lpstarty[i] = s[i][1]
            lpendx[i] = s[i][0] - window
            lpendy[i] = s[i][1]
            cur_line = Factory.Line(img, ((lpstartx[i], lpstarty[i]),
                                          (lpendx[i], lpendy[i])))
            search_lines.append(cur_line)
            tmp = img.get_threshold_crossing(
                (int(lpstartx[i]), int(lpstarty[i])),
                (int(lpendx[i]), int(lpendy[i])), threshold=threshold,
                lighttodark=lighttodark, darktolight=darktolight,
                departurethreshold=departurethreshold)
            fit_points.append(Factory.Circle(img, tmp[0], tmp[1], 3))
            linefitpts[i] = tmp

    else:
        m = float((y2 - y1)) / (x2 - x1)
        b = y1 - m * x1
        mo = -1 / m  # slope of orthogonal line segments

        # obtain points for measurement along the initial guess line
        for i in xrange(0, measurements):
            s[i][0] = x1 + i * dx
            s[i][1] = y1 + i * dy
            fx = (math.sqrt(math.pow(window, 2)) / (1 + mo)) / 2
            fy = fx * mo
            lpstartx[i] = s[i][0] + fx
            lpstarty[i] = s[i][1] + fy
            lpendx[i] = s[i][0] - fx
            lpendy[i] = s[i][1] - fy
            cur_line = Factory.Line(img, ((lpstartx[i], lpstarty[i]),
                                          (lpendx[i], lpendy[i])))
            search_lines.append(cur_line)
            tmp = img.get_threshold_crossing(
                (int(lpstartx[i]), int(lpstarty[i])),
                (int(lpendx[i]), int(lpendy[i])), threshold=threshold,
                lighttodark=lighttodark, darktolight=darktolight,
                departurethreshold=departurethreshold)
            fit_points.append((tmp[0], tmp[1]))
            linefitpts[i] = tmp

    x = linefitpts[:, 0]
    y = linefitpts[:, 1]
    ymin = np.min(y)
    ymax = np.max(y)
    xmax = np.max(x)
    xmin = np.min(x)

    if (xmax - xmin) > (ymax - ymin):
        # do the least squares
        a = np.vstack([x, np.ones(len(x))]).T
        m, c = nla.lstsq(a, y)[0]
        y0 = int(m * xmin + c)
        y1 = int(m * xmax + c)
        final_line = Factory.Line(img, ((xmin, y0), (xmax, y1)))
    else:
        # do the least squares
        a = np.vstack([y, np.ones(len(y))]).T
        m, c = nla.lstsq(a, x)[0]
        x0 = int(ymin * m + c)
        x1 = int(ymax * m + c)
        final_line = Factory.Line(img, ((x0, ymin), (x1, ymax)))

    return final_line, search_lines, fit_points


@image_method
def fit_lines(img, guesses, window=10, threshold=128):
    """
    **SUMMARY**

    Fit lines in a binary/gray image using an initial guess and the least
    squares method. The lines are returned as a line feature set.

    **PARAMETERS**

    * *guesses* - A list of tuples of the form ((x0,y0),(x1,y1)) where each
      of the lines is an approximate guess.
    * *window* - A window around the guess to search.
    * *threshold* - the threshold above which we count a pixel as a line

    **RETURNS**

    A feature set of line features, one per guess.

    **EXAMPLE**


    >>> img = Image("lsq.png")
    >>> guesses = [((313, 150), (312, 332)), ((62, 172), (252, 52)),
    ...            ((102, 372), (182, 182)), ((372, 62), (572, 162)),
    ...            ((542, 362), (462, 182)), ((232, 412), (462, 423))]
    >>> l = img.fit_lines(guesses, window=10)
    >>> l.draw(color=Color.RED, width=3)
    >>> for g in guesses:
    >>>    img.draw_line(g[0], g[1], color=Color.YELLOW)

    >>> img.show()
    """

    ret_val = FeatureSet()
    i = 0
    for g in guesses:
        # Guess the size of the crop region from the line
        # guess and the window.
        ymin = np.min([g[0][1], g[1][1]])
        ymax = np.max([g[0][1], g[1][1]])
        xmin = np.min([g[0][0], g[1][0]])
        xmax = np.max([g[0][0], g[1][0]])

        xmin_w = np.clip(xmin - window, 0, img.width)
        xmax_w = np.clip(xmax + window, 0, img.width)
        ymin_w = np.clip(ymin - window, 0, img.height)
        ymax_w = np.clip(ymax + window, 0, img.height)
        temp = img.crop(xmin_w, ymin_w, xmax_w - xmin_w, ymax_w - ymin_w)
        temp = temp.get_gray_ndarray()

        # pick the lines above our threshold
        x, y = np.where(temp > threshold)
        pts = zip(x, y)
        gpv = np.array([float(g[0][0] - xmin_w), float(g[0][1] - ymin_w)])
        gpw = np.array([float(g[1][0] - xmin_w), float(g[1][1] - ymin_w)])

        def line_segment_to_point(p):
            w = gpw
            v = gpv
            #print w,v
            p = np.array([float(p[0]), float(p[1])])
            l2 = np.sum((w - v) ** 2)
            t = float(np.dot((p - v), (w - v))) / float(l2)
            if t < 0.00:
                return np.sqrt(np.sum((p - v) ** 2))
            elif t > 1.0:
                return np.sqrt(np.sum((p - w) ** 2))
            else:
                project = v + (t * (w - v))
                return np.sqrt(np.sum((p - project) ** 2))

        # http://stackoverflow.com/questions/849211/
        # shortest-distance-between-a-point-and-a-line-segment

        distances = np.array(map(line_segment_to_point, pts))
        closepoints = np.where(distances < window)[0]

        pts = np.array(pts)

        if len(closepoints) < 3:
            continue

        good_pts = pts[closepoints]
        good_pts = good_pts.astype(float)

        x = good_pts[:, 0]
        y = good_pts[:, 1]
        # do the shift from our crop
        # generate the line values
        x = x + xmin_w
        y = y + ymin_w

        ymin = np.min(y)
        ymax = np.max(y)
        xmax = np.max(x)
        xmin = np.min(x)

        if (xmax - xmin) > (ymax - ymin):
            # do the least squares
            a = np.vstack([x, np.ones(len(x))]).T
            m, c = nla.lstsq(a, y)[0]
            y0 = int(m * xmin + c)
            y1 = int(m * xmax + c)
            ret_val.append(Factory.Line(img, ((xmin, y0), (xmax, y1))))
        else:
            # do the least squares
            a = np.vstack([y, np.ones(len(y))]).T
            m, c = nla.lstsq(a, x)[0]
            x0 = int(ymin * m + c)
            x1 = int(ymax * m + c)
            ret_val.append(Factory.Line(img, ((x0, ymin), (x1, ymax))))

    return ret_val


@image_method
def fit_line_points(img, guesses, window=(11, 11), samples=20,
                    params=(0.1, 0.1, 0.1)):
    """
    **DESCRIPTION**

    This method uses the snakes / active get_contour approach in an attempt
    to fit a series of points to a line that may or may not be exactly
    linear.

    **PARAMETERS**

    * *guesses* - A set of lines that we wish to fit to. The lines are
      specified as a list of tuples of (x,y) tuples.
      E.g. [((x0,y0),(x1,y1))....]
    * *window* - The search window in pixels for the active contours
      approach.
    * *samples* - The number of points to sample along the input line,
      these are the initial conditions for active contours method.
    * *params* - the alpha, beta, and gamma values for the active
      contours routine.

    **RETURNS**

    A list of fitted get_contour points. Each get_contour is a list of
    (x,y) tuples.

    **EXAMPLE**

    >>> img = Image("lsq.png")
    >>> guesses = [((313, 150), (312, 332)), ((62, 172), (252, 52)),
    ...            ((102, 372), (182, 182)), ((372, 62), (572, 162)),
    ...            ((542, 362), (462, 182)), ((232, 412), (462, 423))]
    >>> r = img.fit_line_points(guesses)
    >>> for rr in r:
    >>>    img.draw_line(rr[0], rr[1], color=Color.RED, width=3)
    >>> for g in guesses:
    >>>    img.draw_line(g[0], g[1], color=Color.YELLOW)

    >>> img.show()

    """
    pts = []
    for g in guesses:
        #generate the approximation
        best_guess = []
        dx = float(g[1][0] - g[0][0])
        dy = float(g[1][1] - g[0][1])
        l = np.sqrt((dx * dx) + (dy * dy))
        if l <= 0:
            logger.warning("Can't Do snakeFitPoints without "
                           "OpenCV >= 2.3.0")
            return

        dx = dx / l
        dy = dy / l
        for i in range(-1, samples + 1):
            t = i * (l / samples)
            best_guess.append(
                (int(g[0][0] + (t * dx)), int(g[0][1] + (t * dy))))
        # do the snake fitting
        appx = img.fit_contour(best_guess, window=window, params=params,
                               do_appx=False)
        pts.append(appx)

    return pts


@image_method
def find_grid_lines(img):

    """
    **SUMMARY**

    Return Grid Lines as a Line Feature Set

    **PARAMETERS**

    None

    **RETURNS**

    Grid Lines as a Feature Set

    **EXAMPLE**

    >>>> img = Image('something.png')
    >>>> img.grid([20,20],(255,0,0))
    >>>> lines = img.find_grid_lines()

    """
    print img._grid_layer
    if img._grid_layer[0] is None:
        print "Cannot find grid on the image, Try adding a grid first"
        return None

    grid_index = img.get_drawing_layer(img._grid_layer[0])
    
    line_fs = FeatureSet()
    try:
        step_row = img.size[1] / img._grid_layer[1][0]
        step_col = img.size[0] / img._grid_layer[1][1]
    except ZeroDivisionError:
        return None

    i = 1
    j = 1

    while i < img._grid_layer[1][0]:
        line_fs.append(Factory.Line(img, ((0, step_row * i),
                                          (img.size[0], step_row * i))))
        i = i + 1
    while j < img._grid_layer[1][1]:
        line_fs.append(Factory.Line(img, ((step_col * j, 0),
                                          (step_col * j, img.size[1]))))
        j = j + 1

    return line_fs


@image_method
def match_sift_key_points(img, template, quality=200):
    """
    **SUMMARY**

    matchSIFTKeypoint allows you to match a template image with another
    image using SIFT keypoints. The method extracts keypoints from each
    image, uses the Fast Local Approximate Nearest Neighbors algorithm to
    find correspondences between the feature points, filters the
    correspondences based on quality. This method should be able to handle
    a reasonable changes in camera orientation and illumination. Using a
    template that is close to the target image will yield much better
    results.

    **PARAMETERS**

    * *template* - A template image.
    * *quality* - The feature quality metric. This can be any value
      between about 100 and 500. Lower values should return fewer, but
      higher quality features.

    **RETURNS**

    A Tuple of lists consisting of matched KeyPoints found on the image
    and matched keypoints found on the template. keypoints are sorted
    according to lowest distance.

    **EXAMPLE**

    >>> camera = Camera()
    >>> template = Image("template.png")
    >>> img = camera.get_image()
    >>> fs = img.match_sift_key_points(template)

    **SEE ALSO**

    :py:meth:`_get_raw_keypoints`
    :py:meth:`_get_flann_matches`
    :py:meth:`draw_keypoint_matches`
    :py:meth:`find_keypoints`

    """
    if not hasattr(cv2, "FeatureDetector_create"):
        logger.warn("OpenCV >= 2.4.3 required")
        return None
    if template is None:
        return None
    detector = cv2.FeatureDetector_create("SIFT")
    descriptor = cv2.DescriptorExtractor_create("SIFT")
    img_array = img.get_ndarray()
    template_img = template.get_ndarray()

    skp = detector.detect(img_array)
    skp, sd = descriptor.compute(img_array, skp)

    tkp = detector.detect(template_img)
    tkp, td = descriptor.compute(template_img, tkp)

    idx, dist = img._get_flann_matches(sd, td)
    dist = dist[:, 0] / 2500.0
    dist = dist.reshape(-1, ).tolist()
    idx = idx.reshape(-1).tolist()
    indices = range(len(dist))
    indices.sort(key=lambda i: dist[i])
    dist = [dist[i] for i in indices]
    idx = [idx[i] for i in indices]
    sfs = []
    for i, dis in itertools.izip(idx, dist):
        if dis < quality:
            sfs.append(Factory.KeyPoint(template, skp[i], sd, "SIFT"))
        else:
            break  # since sorted

    idx, dist = img._get_flann_matches(td, sd)
    dist = dist[:, 0] / 2500.0
    dist = dist.reshape(-1, ).tolist()
    idx = idx.reshape(-1).tolist()
    indices = range(len(dist))
    indices.sort(key=lambda i: dist[i])
    dist = [dist[i] for i in indices]
    idx = [idx[i] for i in indices]
    tfs = []
    for i, dis in itertools.izip(idx, dist):
        if dis < quality:
            tfs.append(Factory.KeyPoint(template, tkp[i], td, "SIFT"))
        else:
            break

    return sfs, tfs


@image_method
def find_features(img, method="szeliski", threshold=1000):
    """
    **SUMMARY**

    Find szeilski or Harris features in the image.
    Harris features correspond to Harris corner detection in the image.

    Read more:

    Harris Features: http://en.wikipedia.org/wiki/Corner_detection
    szeliski Features: http://research.microsoft.com/en-us/um/people/
    szeliski/publications.htm

    **PARAMETERS**

    * *method* - Features type
    * *threshold* - threshold val

    **RETURNS**

    A list of Feature objects corrseponding to the feature points.

    **EXAMPLE**

    >>> img = Image("corner_sample.png")
    >>> fpoints = img.find_features("harris", 2000)
    >>> for f in fpoints:
        ... f.draw()
    >>> img.show()

    **SEE ALSO**

    :py:meth:`draw_keypoint_matches`
    :py:meth:`find_keypoints`
    :py:meth:`find_keypoint_match`

    """
    if method not in ["harris", "szeliski"]:
        logger.warning("Invalid method.")
        return None

    img_array = img.get_gray_ndarray()
    blur = cv2.GaussianBlur(img_array, ksize=(3, 3), sigmaX=0)

    ix = cv2.Sobel(blur, ddepth=cv2.CV_32F, dx=1, dy=0)
    iy = cv2.Sobel(blur, ddepth=cv2.CV_32F, dx=0, dy=1)

    ix_ix = np.multiply(ix, ix)
    iy_iy = np.multiply(iy, iy)
    ix_iy = np.multiply(ix, iy)

    ix_ix_blur = cv2.GaussianBlur(ix_ix, ksize=(5, 5), sigmaX=0)
    iy_iy_blur = cv2.GaussianBlur(iy_iy, ksize=(5, 5), sigmaX=0)
    ix_iy_blur = cv2.GaussianBlur(ix_iy, ksize=(5, 5), sigmaX=0)

    harris_thresh = threshold * 5000
    alpha = 0.06
    det_a = ix_ix_blur * iy_iy_blur - ix_iy_blur ** 2
    trace_a = ix_ix_blur + iy_iy_blur
    feature_list = []
    if method == "szeliski":
        harmonic_mean = det_a / trace_a
        for j, i in np.argwhere(harmonic_mean > threshold):
            feature_list.append(
                Feature(img, i, j, ((i, j), (i, j), (i, j), (i, j))))

    elif method == "harris":
        harris_function = det_a - (alpha * trace_a * trace_a)
        for j, i in np.argwhere(harris_function > harris_thresh):
            feature_list.append(
                Feature(img, i, j, ((i, j), (i, j), (i, j), (i, j))))
        
    return feature_list


@image_method
def find_blobs_from_watershed(img, mask=None, erode=2, dilate=2,
                              use_my_mask=False, invert=False, minsize=20,
                              maxsize=None):
    """
    **SUMMARY**

    Implements the watershed algorithm on the input image with an optional
    mask and then uses the mask to find blobs.

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
    * *invert* - invert the resulting mask before finding blobs.
    * *minsize* - minimum blob size in pixels.
    * *maxsize* - the maximum blob size in pixels.

    **RETURNS**

    A feature set of blob features.

    **EXAMPLE**

    >>> img = Image("/data/sampleimages/wshed.jpg")
    >>> mask = img.threshold(100).dilate(3)
    >>> blobs = img.find_blobs_from_watershed(mask)
    >>> blobs.show()

    **SEE ALSO**
    Color.WATERSHED_FG - The watershed foreground color
    Color.WATERSHED_BG - The watershed background color
    Color.WATERSHED_UNSURE - The watershed not sure if fg or bg color.

    """
    newmask = img.watershed(mask, erode, dilate, use_my_mask)
    if invert:
        newmask = mask.invert()
    return img.find_blobs_from_mask(newmask, minsize=minsize,
                                    maxsize=maxsize)


@image_method
def find_keypoint_clusters(img, num_of_clusters=5, order='dsc',
                           flavor='surf'):
    '''
    This function is meant to try and find interesting areas of an
    image. It does this by finding keypoint clusters in an image.
    It uses keypoint (ORB) detection to locate points of interest
    and then uses kmeans clustering to get the X,Y coordinates of
    those clusters of keypoints. You provide the expected number
    of clusters and you will get back a list of the X,Y coordinates
    and rank order of the number of Keypoints around those clusters

    **PARAMETERS**
    * num_of_clusters - The number of clusters you are looking for
      (default: 5)
    * order - The rank order you would like the points returned in, dsc or
      asc, (default: dsc)
    * flavor - The keypoint type, or 'corner' for just corners


    **EXAMPLE**

    >>> img = Image('simplecv')
    >>> clusters = img.find_keypoint_clusters()
    >>> clusters.draw()
    >>> img.show()

    **RETURNS**

    FeatureSet
    '''
    if flavor.lower() == 'corner':
        keypoints = img.find_corners()  # fallback to corners
    else:
        keypoints = img.find_keypoints(
            flavor=flavor.upper())  # find the keypoints
    if keypoints is None or keypoints <= 0:
        return None

    xypoints = np.array([(f.x, f.y) for f in keypoints])
    # find the clusters of keypoints
    xycentroids, xylabels = scv.kmeans2(xypoints, num_of_clusters)
    xycounts = np.array([])

    # count the frequency of occurences for sorting
    for i in range(num_of_clusters):
        xycounts = np.append(xycounts, len(np.where(xylabels == i)[-1]))

    # sort based on occurence
    merged = np.msort(np.hstack((np.vstack(xycounts), xycentroids)))
    clusters = [c[1:] for c in
                merged]  # strip out just the values ascending
    if order.lower() == 'dsc':
        clusters = clusters[::-1]  # reverse if descending

    fs = FeatureSet()
    for x, y in clusters:  # map the values to a feature set
        f = Factory.Corner(img, x, y)
        fs.append(f)

    return fs


@image_method
def get_freak_descriptor(img, flavor="SURF"):
    """
    **SUMMARY**

    Compute FREAK Descriptor of given keypoints.
    FREAK - Fast Retina Keypoints.
    Read more: http://www.ivpe.com/freak.htm

    Keypoints can be extracted using following detectors.

    - SURF
    - SIFT
    - BRISK
    - ORB
    - STAR
    - MSER
    - FAST
    - Dense

    **PARAMETERS**

    * *flavor* - Detector (see above list of detectors) - string

    **RETURNS**

    * FeatureSet* - A feature set of KeyPoint Features.
    * Descriptor* - FREAK Descriptor

    **EXAMPLE**

    >>> img = Image("lenna")
    >>> fs, des = img.get_freak_descriptor("ORB")

    """
    if cv2.__version__.startswith('$Rev:'):
        logger.warn("OpenCV version >= 2.4.2 requierd")
        return None

    if int(cv2.__version__.replace('.', '0')) < 20402:
        logger.warn("OpenCV version >= 2.4.2 requierd")
        return None

    flavors = ["SIFT", "SURF", "BRISK", "ORB", "STAR", "MSER", "FAST",
               "Dense"]
    if flavor not in flavors:
        logger.warn("Unkown Keypoints detector. Returning None.")
        return None
    detector = cv2.FeatureDetector_create(flavor)
    extractor = cv2.DescriptorExtractor_create("FREAK")
    img._key_points = detector.detect(img.get_gray_ndarray())
    img._key_points, img._kp_descriptors = extractor.compute(
        img.get_gray_ndarray(),
        img._key_points)
    fs = FeatureSet()
    for i in range(len(img._key_points)):
        fs.append(Factory.KeyPoint(img, img._key_points[i],
                                   img._kp_descriptors[i], flavor))

    return fs, img._kp_descriptors


@image_method
def recognize_face(img, recognizer=None):
    """
    **SUMMARY**

    Find faces in the image using FaceRecognizer and predict their class.

    **PARAMETERS**

    * *recognizer*   - Trained FaceRecognizer object

    **EXAMPLES**

    >>> cam = Camera()
    >>> img = cam.get_image()
    >>> recognizer = FaceRecognizer()
    >>> recognizer.load("training.xml")
    >>> print img.recognize_face(recognizer)
    """
    if not hasattr(cv2, "createFisherFaceRecognizer"):
        logger.warn("OpenCV >= 2.4.4 required to use this.")
        return None

    if not isinstance(recognizer, FaceRecognizer):
        logger.warn("SimpleCV.Features.FaceRecognizer object required.")
        return None

    w, h = recognizer.image_size
    label = recognizer.predict(img.resize(w, h))
    return label


@image_method
def find_and_recognize_faces(img, recognizer, cascade=None):
    """
    **SUMMARY**

    Predict the class of the face in the image using FaceRecognizer.

    **PARAMETERS**

    * *recognizer*  - Trained FaceRecognizer object

    * *cascade*     -haarcascade which would identify the face
                     in the image.

    **EXAMPLES**

    >>> cam = Camera()
    >>> img = cam.get_image()
    >>> recognizer = FaceRecognizer()
    >>> recognizer.load("training.xml")
    >>> feat = img.find_and_recognize_faces(recognizer, "face.xml")
    >>> for feature, label, confidence in feat:
        ... i = feature.crop()
        ... i.draw_text(str(label))
        ... i.show()
    """
    if not hasattr(cv2, "createFisherFaceRecognizer"):
        logger.warn("OpenCV >= 2.4.4 required to use this.")
        return None

    if not isinstance(recognizer, FaceRecognizer):
        logger.warn("SimpleCV.Features.FaceRecognizer object required.")
        return None

    if not cascade:
        cascade = os.path.join(DATA_DIR, 'Features/HaarCascades/face.xml')

    faces = img.find_haar_features(cascade)
    if not faces:
        logger.warn("Faces not found in the image.")
        return None

    ret_val = []
    for face in faces:
        label, confidence = face.crop().recognize_face(recognizer)
        ret_val.append([face, label, confidence])
    return ret_val


@image_method
def edge_snap(img, point_list, step=1):
    """
    **SUMMARY**

    Given a List of points finds edges closet to the line joining two
    successive points, edges are returned as a FeatureSet of
    Lines.

    Note : Image must be binary, it is assumed that prior conversion is
    done

    **Parameters**

   * *point_list* - List of points to be checked for nearby edges.

    * *step* - Number of points to skip if no edge is found in vicinity.
               Keep this small if you want to sharply follow a curve

    **RETURNS**

    * FeatureSet * - A FeatureSet of Lines

    **EXAMPLE**

    >>> image = Image("logo").edges()
    >>> edgeLines = image.edge_snap([(50, 50), (230, 200)])
    >>> edgeLines.draw(color=Color.YELLOW, width=3)
    """
    img_array = img.get_gray_ndarray().transpose()
    c1 = np.count_nonzero(img_array)
    c2 = np.count_nonzero(img_array - 255)

    #checking that all values are 0 and 255
    if c1 + c2 != img_array.size:
        raise ValueError("Image must be binary")

    if len(point_list) < 2:
        return None

    final_list = [point_list[0]]
    feature_set = FeatureSet()
    last = point_list[0]
    for point in point_list[1:None]:
        final_list += img._edge_snap2(last, point, step)
        last = point

    last = final_list[0]
    for point in final_list:
        feature_set.append(Factory.Line(img, (last, point)))
        last = point
    return feature_set


@image_method
def _edge_snap2(img, start, end, step):
    """
    **SUMMARY**

    Given a two points returns a list of edge points closet to the line
    joining the points. Point is a tuple of two numbers

    Note : Image must be binary

    **Parameters**

    * *start* - First Point

    * *end* - Second Point

    * *step* - Number of points to skip if no edge is found in vicinity
               Keep this low to detect sharp curves

    **RETURNS**

    * List * - A list of tuples , each tuple contains (x,y) values

    """

    edge_map = np.copy(img.get_gray_ndarray().transpose())

    #Size of the box around a point which is checked for edges.
    box = step * 4

    xmin = min(start[0], end[0])
    xmax = max(start[0], end[0])
    ymin = min(start[1], end[1])
    ymax = max(start[1], end[1])

    line = img.bresenham_line(start, end)

    #List of Edge Points.
    final_list = []
    i = 0

    #Closest any point has ever come to the end point
    overall_min_dist = None

    while i < len(line):

        x, y = line[i]

        #Get the matrix of points fromx around current point.
        region = edge_map[x - box:x + box, y - box:y + box]

        #Condition at the boundary of the image
        if region.shape[0] == 0 or region.shape[1] == 0:
            i += step
            continue

        #Index of all Edge points
        index_list = np.argwhere(region > 0)
        if index_list.size > 0:

            #Center the coordinates around the point
            index_list -= box
            min_dist = None

            # Incase multiple edge points exist, choose the one closest
            # to the end point
            for ix, iy in index_list:
                dist = math.hypot(x + ix - end[0], iy + y - end[1])
                if min_dist is None or dist < min_dist:
                    dx, dy = ix, iy
                    min_dist = dist

            # The distance of the new point is compared with the least
            # distance computed till now, the point is rejected if it's
            # comparitively more. This is done so that edge points don't
            # wrap around a curve instead of heading towards the end point
            if overall_min_dist is not None \
                    and min_dist > overall_min_dist * 1.1:
                i += step
                continue

            if overall_min_dist is None or min_dist < overall_min_dist:
                overall_min_dist = min_dist

            # Reset the points in the box so that they are not detected
            # during the next iteration.
            edge_map[x - box:x + box, y - box:y + box] = 0

            # Keep all the points in the bounding box
            if xmin <= x + dx <= xmax and ymin <= y + dx <= ymax:
                #Add the point to list and redefine the line
                line = [(x + dx, y + dy)] \
                    + img.bresenham_line((x + dx, y + dy), end)
                final_list += [(x + dx, y + dy)]

                i = 0

        i += step
    final_list += [end]
    return final_list


@image_method
def smart_rotate(img, bins=18, point=[-1, -1], auto=True, threshold=80,
                 min_length=30, max_gap=10, t1=150, t2=200, fixed=True):
    """
    **SUMMARY**

    Attempts to rotate the image so that the most significant lines are
    approximately parellel to horizontal or vertical edges.

    **Parameters**


    * *bins* - The number of bins the lines will be grouped into.

    * *point* - the point about which to rotate, refer :py:meth:`rotate`

    * *auto* - If true point will be computed to the mean of centers of all
        the lines in the selected bin. If auto is True, value of point is
        ignored

    * *threshold* - which determines the minimum "strength" of the line
        refer :py:meth:`find_lines` for details.

    * *min_length* - how many pixels long the line must be to be returned,
        refer :py:meth:`find_lines` for details.

    * *max_gap* - how much gap is allowed between line segments to consider
        them the same line .refer to :py:meth:`find_lines` for details.

    * *t1* - thresholds used in the edge detection step,
        refer to :py:meth:`_get_edge_map` for details.

    * *t2* - thresholds used in the edge detection step,
        refer to :py:meth:`_get_edge_map` for details.

    * *fixed* - if fixed is true,keep the original image dimensions,
        otherwise scale the image to fit the rotation , refer to
        :py:meth:`rotate`

    **RETURNS**

    A rotated image

    **EXAMPLE**
    >>> i = Image ('image.jpg')
    >>> i.smart_rotate().show()

    """
    lines = img.find_lines(threshold, min_length, max_gap, t1, t2)

    if len(lines) == 0:
        logger.warning("No lines found in the image")
        return img

    # Initialize empty bins
    binn = [[] for i in range(bins)]

    #Convert angle to bin number
    conv = lambda x: int(x + 90) / bins

    #Adding lines to bins
    [binn[conv(line.get_angle())].append(line) for line in lines]

    #computing histogram, value of each column is total length of all lines
    #in the bin
    hist = [sum([line.length() for line in lines]) for lines in binn]

    #The maximum histogram
    index = np.argmax(np.array(hist))

    #Good ol weighted mean, for the selected bin
    avg = sum([line.get_angle() * line.length() for line in binn[index]]) \
        / sum([line.length() for line in binn[index]])

    #Mean of centers of all lines in selected bin
    if auto:
        x = sum([line.end_points[0][0] + line.end_points[1][0]
                 for line in binn[index]]) / 2 / len(binn[index])
        y = sum([line.end_points[0][1] + line.end_points[1][1]
                 for line in binn[index]]) / 2 / len(binn[index])
        point = [x, y]

    #Determine whether to rotate the lines to vertical or horizontal
    if -45 <= avg <= 45:
        return img.rotate(avg, fixed=fixed, point=point)
    elif avg > 45:
        return img.rotate(avg - 90, fixed=fixed, point=point)
    else:
        return img.rotate(avg + 90, fixed=fixed, point=point)
        #Congratulations !! You did a smart thing


@image_method
def find_blobs_from_hue_histogram(img, model, threshold=1, smooth=True,
                                  minsize=10, maxsize=None):
    """
    **SUMMARY**

    This method performs hue histogram back projection on the image and
    uses the results to generate a FeatureSet of blob objects. This is a
    very quick and easy way of matching objects based on color. Given a hue
    histogram taken from another image or an roi within the image we
    attempt to find all pixels that are similar to the colors inside the
    histogram.

    **PARAMETERS**

    * *model* - The histogram to use for pack projection. This can either
    be a histogram, anything that can be converted into an ROI for the
    image (like an x,y,w,h tuple or a feature, or another image.

    * *smooth* - A bool, True means apply a smoothing operation after doing
    the back project to improve the results.

    * *threshold* - If this value is not None, we apply a threshold to the
    result of back projection to yield a binary image. Valid values are
    from 1 to 255.

    * *minsize* - the minimum blob size in pixels.

    * *maxsize* - the maximum blob size in pixels.

    **RETURNS**

    A FeatureSet of blob objects or None if no blobs are found.

    **EXAMPLE**

    >>>> img = Image('lenna')

    Generate a hist

    >>>> hist = img.get_normalized_hue_histogram((0, 0, 50, 50))
    >>>> blobs = img.find_blobs_from_hue_histogram(hist)
    >>>> blobs.show()

    **SEE ALSO**

    ImageClass.get_normalized_hue_histogram()
    ImageClass.back_project_hue_histogram()

    """
    new_mask = img.back_project_hue_histogram(model=model, smooth=smooth,
                                              full_color=False,
                                              threshold=threshold)
    return img.find_blobs_from_mask(new_mask, minsize=minsize,
                                    maxsize=maxsize)
