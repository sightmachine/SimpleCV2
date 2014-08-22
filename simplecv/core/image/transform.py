"""
Transformation methods
"""
import cv2
import numpy as np

from simplecv.base import logger
from simplecv.color import Color
from simplecv.core.image import image_method
from simplecv.factory import Factory
from simplecv.features.features import Feature

# maximum image size -
# about twice the size of a full 35mm images
# if you hit this, you got a lot data.
MAX_DIMENSION = 2 * 6000


@image_method
def scale(img, ratio, interpolation=cv2.INTER_LINEAR):
    """
    Scale the image by given ratio.

    :param ratio: target width and height in pixels
    :type ratio: list or tuple of int
    :param interpolation: how to generate new pixels that don't match the
    original pixels. Argument goes direction to cv2.resize.
    See http://docs.opencv.org/modules/imgproc/doc/
    geometric_transformations.html?highlight=resize#cv2.resize
    for more details
    :type interpolation: cv2 constant

    :returns: instance of :class:simplecv.core.image.Image.

    :py:meth:`resize`
    """
    size = tuple(map(lambda a: int(ratio * a), img.size))
    if max(size) > MAX_DIMENSION or min(size) < 1:
            logger.warning("Holy Heck! You tried to make an image really "
                           "big or impossibly small. I can't scale that")
            return img
    scaled_array = cv2.resize(img.ndarray, dsize=size,
                              interpolation=interpolation)
    return Factory.Image(scaled_array, color_space=img.color_space)


@image_method
def resize(img, w=None, h=None, interpolation=cv2.INTER_LINEAR):
    """
    This method resizes an image based on a width, a height, or both.
    If either width or height is not provided the value is inferred by
    keeping the aspect ratio.
    If both values are provided then the image is resized accordingly.

    :param w: The width of the output image in pixels.
    :type w: int
    :param h: The height of the output image in pixels.
    :type h: int

    :returns: instance of :class:simplecv.core.image.Image or if the size
    is invalid a warning is issued and None is returned.

    **EXAMPLE**

    >>> img = Image("lenna")
    >>> img2 = img.resize(w=1024)  # h is guessed from w
    >>> img3 = img.resize(h=1024)  # w is guessed from h
    >>> img4 = img.resize(w=200, h=100)

    :py:meth:`scale`
    """
    if w is None and h is None:
        logger.warning("Image.resize has no parameters. "
                       "No operation is performed")
        return None
    elif w is not None and h is None:
        sfactor = float(w) / float(img.width)
        h = int(sfactor * float(img.height))
    elif w is None and h is not None:
        sfactor = float(h) / float(img.height)
        w = int(sfactor * float(img.width))
    if max(w, h) > MAX_DIMENSION:
        logger.warning("Image.resize Holy Heck! You tried to make an "
                       "image really big or impossibly small. "
                       "I can't scale that")
        return None
    saceld_array = cv2.resize(img.ndarray, dsize=(w, h),
                              interpolation=interpolation)
    return Factory.Image(saceld_array, color_space=img.color_space)


@image_method
def flip_horizontal(img):
    """
    Horizontally mirror an image.

    .. Warning::
      Note that flip does not mean rotate 180 degrees! The two are
      different.

    **RETURNS**

    The flipped SimpleCV image.

    **EXAMPLE**

    >>> img = Image("lenna")
    >>> upsidedown = img.flip_horizontal()

    **SEE ALSO**

    :py:meth:`flip_vertical`
    :py:meth:`rotate`
    """
    flip_array = cv2.flip(img.ndarray, flipCode=1)
    return Factory.Image(flip_array, color_space=img.color_space)


@image_method
def flip_vertical(img):
    """
    Vertically mirror an image.

    .. Warning::
      Note that flip does not mean rotate 180 degrees! The two are
      different.

    **RETURNS**

    The flipped SimpleCV image.

    **EXAMPLE**

    >>> img = Image("lenna")
    >>> img = img.flip_vertical()

    **SEE ALSO**

    :py:meth:`rotate`
    :py:meth:`flip_horizontal`
    """
    flip_array = cv2.flip(img.ndarray, flipCode=0)
    return Factory.Image(flip_array, color_space=img.color_space)


@image_method
def rotate(img, angle, fixed=True, point=None, scale=1.0):
    """
    **SUMMARY***

    This function rotates an image around a specific point by the given
    angle. By default in "fixed" mode, the returned Image is the same
    dimensions as the original Image, and the contents will be scaled to
    fit. In "full" mode the contents retain the original size, and the
    Image object will scale by default, the point is the center of the
    image. You can also specify a scaling parameter

    .. Note:
      that when fixed is set to false selecting a rotation point has no
      effect since the image is move to fit on the screen.

    **PARAMETERS**

    * *angle* - angle in degrees positive is clockwise, negative is counter
     clockwise
    * *fixed* - if fixed is true,keep the original image dimensions,
     otherwise scale the image to fit the rotation
    * *point* - the point about which we want to rotate, if none is
     defined we use the center.
    * *scale* - and optional floating point scale parameter.

    **RETURNS**

    The rotated SimpleCV image.

    **EXAMPLE**

    >>> img = Image('logo')
    >>> img2 = img.rotate(73.00, point=(img.width / 2, img.height / 2))
    >>> img3 = img.rotate(73.00,
        ...               fixed=False,
        ...               point=(img.width / 2, img.height / 2))
    >>> img4 = img2.side_by_side(img3)
    >>> img4.show()

    **SEE ALSO**

    :py:meth:`rotate90`

    """
    if point is None:
        point = [-1, -1]
    if point[0] == -1 or point[1] == -1:
        point[0] = (img.width - 1) / 2
        point[1] = (img.height - 1) / 2

    # first we create what we thing the rotation matrix should be
    rot_mat = cv2.getRotationMatrix2D(center=(float(point[0]),
                                              float(point[1])),
                                      angle=float(angle),
                                      scale=float(scale))
    if fixed:
        array = cv2.warpAffine(img.ndarray, M=rot_mat, dsize=img.size)
        return Factory.Image(array, color_space=img.color_space)

    # otherwise, we're expanding the matrix to
    # fit the image at original size
    a1 = np.array([0, 0, 1])
    b1 = np.array([img.width, 0, 1])
    c1 = np.array([img.width, img.height, 1])
    d1 = np.array([0, img.height, 1])
    # So we have defined our image ABC in homogenous coordinates
    # and apply the rotation so we can figure out the image size
    a = np.dot(rot_mat, a1)
    b = np.dot(rot_mat, b1)
    c = np.dot(rot_mat, c1)
    d = np.dot(rot_mat, d1)
    # I am not sure about this but I think the a/b/c/d are transposed
    # now we calculate the extents of the rotated components.
    min_y = min(a[1], b[1], c[1], d[1])
    min_x = min(a[0], b[0], c[0], d[0])
    max_y = max(a[1], b[1], c[1], d[1])
    max_x = max(a[0], b[0], c[0], d[0])
    # from the extents we calculate the new size
    new_width = np.ceil(max_x - min_x)
    new_height = np.ceil(max_y - min_y)
    # now we calculate a new translation
    tx = 0
    ty = 0
    # calculate the translation that will get us centered in the new image
    if min_x < 0:
        tx = -1.0 * min_x
    elif max_x > new_width - 1:
        tx = -1.0 * (max_x - new_width)

    if min_y < 0:
        ty = -1.0 * min_y
    elif max_y > new_height - 1:
        ty = -1.0 * (max_y - new_height)

    # now we construct an affine map that will the rotation and scaling
    # we want with the the corners all lined up nicely
    # with the output image.
    src = ((a1[0], a1[1]), (b1[0], b1[1]), (c1[0], c1[1]))
    dst = ((a[0] + tx, a[1] + ty),
           (b[0] + tx, b[1] + ty),
           (c[0] + tx, c[1] + ty))

    # calculate the translation of the corners to center the image
    # use these new corner positions as the input to cvGetAffineTransform
    rot_mat = cv2.getAffineTransform(
        src=np.array(src).astype(np.float32),
        dst=np.array(dst).astype(np.float32))
    array = cv2.warpAffine(img.ndarray, M=rot_mat,
                           dsize=(int(new_width), int(new_height)))
    return Factory.Image(array, color_space=img.color_space)


@image_method
def transpose(img):
    """
    **SUMMARY**

    Does a fast 90 degree rotation to the right with a flip.

    .. Warning::
      Subsequent calls to this function *WILL NOT* keep rotating it to the
      right!!!
      This function just does a matrix transpose so following one transpose
      by another will just yield the original image.

    **RETURNS**

    The rotated SimpleCV Image.

    **EXAMPLE**

    >>> img = Image("logo")
    >>> img2 = img.transpose()
    >>> img2.show()

    **SEE ALSO**

    :py:meth:`rotate`
    """
    array = cv2.transpose(img.ndarray)
    return Factory.Image(array, color_space=img.color_space)


@image_method
def shear(img, cornerpoints):
    """
    **SUMMARY**

    Given a set of new corner points in clockwise order, return a shear-ed
    image that transforms the image contents.  The returned image is the
    same dimensions.

    **PARAMETERS**

    * *cornerpoints* - a 2x4 tuple of points. The order is
     (top_left, top_right, bottom_left, bottom_right)

    **RETURNS**

    A simpleCV image.

    **EXAMPLE**

    >>> img = Image("lenna")
    >>> points = ((50, 0), (img.width + 50, 0),
        ...       (img.width, img.height), (0, img.height))
    >>> img.shear(points).show()

    **SEE ALSO**

    :py:meth:`transform_affine`
    :py:meth:`warp`
    :py:meth:`rotate`

    http://en.wikipedia.org/wiki/Transformation_matrix

    """
    src = ((0, 0), (img.width - 1, 0), (img.width - 1, img.height - 1))
    rot_matrix = cv2.getAffineTransform(
        src=np.array(src).astype(np.float32),
        dst=np.array(cornerpoints).astype(np.float32))
    return img.transform_affine(rot_matrix=rot_matrix)


@image_method
def transform_affine(img, rot_matrix):
    """
    **SUMMARY**

    This helper function for shear performs an affine rotation using the
    supplied matrix. The matrix can be a either an openCV mat or an
    np.ndarray type. The matrix should be a 2x3

    **PARAMETERS**

    * *rot_matrix* - A 2x3 numpy array or CvMat of the affine transform.

    **RETURNS**

    The rotated image. Note that the rotation is done in place, i.e.
    the image is not enlarged to fit the transofmation.

    **EXAMPLE**

    >>> img = Image("lenna")
    >>> points = ((50, 0), (img.width + 50, 0),
    ...           (img.width, img.height), (0, img.height))
    >>> src = ((0, 0), (img.width - 1, 0),
    ...        (img.width - 1, img.height - 1))
    >>> rot_matrix = cv2.getAffineTransform(src, points)
    >>> img.transform_affine(rot_matrix).show()

    **SEE ALSO**

    :py:meth:`shear`
    :py:meth`warp`
    :py:meth:`transform_perspective`
    :py:meth:`rotate`

    http://en.wikipedia.org/wiki/Transformation_matrix

    """
    array = cv2.warpAffine(img.ndarray, M=rot_matrix, dsize=img.size)
    return Factory.Image(array, color_space=img.color_space)


@image_method
def warp(img, cornerpoints):
    """
    **SUMMARY**

    This method performs and arbitrary perspective transform.
    Given a new set of corner points in clockwise order frin top left,
    return an Image with the images contents warped to the new coordinates.
    The returned image will be the same size as the original image


    **PARAMETERS**

    * *cornerpoints* - A list of four tuples corresponding to the
     destination corners in the order of
     (top_left,top_right,bottom_left,bottom_right)

    **RETURNS**

    A simpleCV Image with the warp applied. Note that this operation does
    not enlarge the image.

    **EXAMPLE**

    >>> img = Image("lenna")
    >>> points = ((30, 30), (img.width - 10, 70),
        ...       (img.width - 1 - 40, img.height - 1 + 30),
        ...       (20, img.height + 10))
    >>> img.warp(points).show()

    **SEE ALSO**

    :py:meth:`shear`
    :py:meth:`transform_affine`
    :py:meth:`transform_perspective`
    :py:meth:`rotate`

    http://en.wikipedia.org/wiki/Transformation_matrix

    """
    #original coordinates
    src = np.array(((0, 0), (img.width - 1, 0),
                    (img.width - 1, img.height - 1),
                    (0, img.height - 1))).astype(np.float32)
    # figure out the warp matrix
    p_wrap = cv2.getPerspectiveTransform(
        src=src, dst=np.array(cornerpoints).astype(np.float32))
    return img.transform_perspective(rot_matrix=p_wrap)


@image_method
def transform_perspective(img, rot_matrix):
    """
    **SUMMARY**

    This helper function for warp performs an affine rotation using the
    supplied matrix.
    The matrix can be a either an openCV mat or an np.ndarray type.
    The matrix should be a 3x3

   **PARAMETERS**
        * *rot_matrix* - Numpy Array or CvMat

    **RETURNS**

    The rotated image. Note that the rotation is done in place, i.e. the
    image is not enlarged to fit the transofmation.

    **EXAMPLE**

    >>> img = Image("lenna")
    >>> points = ((50,0), (img.width + 50, 0),
        ...       (img.width, img.height), (0, img.height))
    >>> src = ((30, 30), (img.width - 10, 70),
        ...    (img.width - 1 - 40, img.height - 1 + 30),
        ...    (20, img.height + 10))
    >>> result = cv2.getPerspectiveTransform(
    ...     np.array(src).astype(np.float32),
    ...     np.array(points).astype(np.float32))
    >>> img.transform_perspective(result).show()


    **SEE ALSO**

    :py:meth:`shear`
    :py:meth:`warp`
    :py:meth:`transform_perspective`
    :py:meth:`rotate`

    http://en.wikipedia.org/wiki/Transformation_matrix
    """
    array = cv2.warpPerspective(src=img.ndarray, dsize=img.size,
                                M=rot_matrix, flags=cv2.INTER_CUBIC)
    return Factory.Image(array, color_space=img.color_space)


@image_method
def split(img, cols, rows):
    """
    **SUMMARY**

    This method can be used to brak and image into a series of image
    chunks. Given number of cols and rows, splits the image into a cols x
    rows 2d array of cropped images

    **PARAMETERS**

    * *rows* - an integer number of rows.
    * *cols* - an integer number of cols.

    **RETURNS**

    A list of SimpleCV images.

    **EXAMPLE**

    >>> img = Image("lenna")
    >>> quadrant =img.split(2,2)
    >>> for f in quadrant:
    >>>    f.show()
    >>>    time.sleep(1)


    **NOTES**

    TODO: This should return and ImageList

    """
    crops = []

    wratio = img.width / cols
    hratio = img.height / rows

    for i in range(rows):
        row = []
        for j in range(cols):
            row.append(img.crop(j * wratio, i * hratio, wratio, hratio))
        crops.append(row)

    return crops


@image_method
def crop(img, x, y=None, w=None, h=None, centered=False, smart=False):
    """
    **SUMMARY**

    Consider you want to crop a image with the following dimension::

        (x,y)
        +--------------+
        |              |
        |              |h
        |              |
        +--------------+
              w      (x1,y1)


    Crop attempts to use the x and y position variables and the w and h
    width and height variables to crop the image. When centered is false,
    x and y define the top and left of the cropped rectangle. When centered
    is true the function uses x and y as the centroid of the cropped
    region.

    You can also pass a feature into crop and have it automatically return
    the cropped image within the bounding outside area of that feature

    Or parameters can be in the form of a
     - tuple or list : (x,y,w,h) or [x,y,w,h]
     - two points : (x,y),(x1,y1) or [(x,y),(x1,y1)]

    **PARAMETERS**

    * *x* - An integer or feature.
          - If it is a feature we crop to the features dimensions.
          - This can be either the top left corner of the image or the
            center cooridnate of the the crop region.
          - or in the form of tuple/list. i,e (x,y,w,h) or [x,y,w,h]
          - Otherwise in two point form. i,e [(x,y),(x1,y1)] or (x,y)
    * *y* - The y coordinate of the center, or top left corner  of the
            crop region.
          - Otherwise in two point form. i,e (x1,y1)
    * *w* - Int - the width of the cropped region in pixels.
    * *h* - Int - the height of the cropped region in pixels.
    * *centered*  - Boolean - if True we treat the crop region as being
      the center coordinate and a width and height. If false we treat it as
      the top left corner of the crop region.
    * *smart* - Will make sure you don't try and crop outside the image
     size, so if your image is 100x100 and you tried a crop like
     img.crop(50,50,100,100), it will autoscale the crop to the max width.


    **RETURNS**

    A SimpleCV Image cropped to the specified width and height.

    **EXAMPLE**

    >>> img = Image('lenna')
    >>> img.crop(50, 40, 128, 128).show()
    >>> img.crop((50, 40, 128, 128)).show() #roi
    >>> img.crop([50, 40, 128, 128]) #roi
    >>> img.crop((50, 40), (178, 168)) # two point form
    >>> img.crop([(50, 40),(178, 168)]) # two point form
    >>> # list of x's and y's
    >>> img.crop([x1, x2, x3, x4, x5], [y1, y1, y3, y4, y5])
    >>> img.crop([(x, y), (x, y), (x, y), (x, y), (x, y)] # list of (x,y)
    >>> img.crop(x, y, 100, 100, smart=True)
    **SEE ALSO**

    :py:meth:`embiggen`
    :py:meth:`region_select`
    """

    if smart:
        if x > img.width:
            x = img.width
        elif x < 0:
            x = 0
        elif y > img.height:
            y = img.height
        elif y < 0:
            y = 0
        elif (x + w) > img.width:
            w = img.width - x
        elif (y + h) > img.height:
            h = img.height - y

    if isinstance(x, np.ndarray):
        x = x.tolist()
    if isinstance(y, np.ndarray):
        y = y.tolist()

    #If it's a feature extract what we need
    if isinstance(x, Feature):
        feature = x
        x = feature.points[0][0]
        y = feature.points[0][1]
        w = feature.width
        h = feature.height

    elif isinstance(x, (tuple, list)) and len(x) == 4 \
            and isinstance(x[0], (int, long, float)) \
            and y is None and w is None and h is None:
        x, y, w, h = x
    # x of the form [(x,y),(x1,y1),(x2,y2),(x3,y3)]
    # x of the form [[x,y],[x1,y1],[x2,y2],[x3,y3]]
    # x of the form ([x,y],[x1,y1],[x2,y2],[x3,y3])
    # x of the form ((x,y),(x1,y1),(x2,y2),(x3,y3))
    # x of the form (x,y,x1,y2) or [x,y,x1,y2]
    elif isinstance(x, (list, tuple)) \
            and isinstance(x[0], (list, tuple)) \
            and (len(x) == 4 and len(x[0]) == 2) \
            and y is None and w is None and h is None:
        if len(x[0]) == 2 and len(x[1]) == 2 \
                and len(x[2]) == 2 and len(x[3]) == 2:
            xmax = np.max([x[0][0], x[1][0], x[2][0], x[3][0]])
            ymax = np.max([x[0][1], x[1][1], x[2][1], x[3][1]])
            xmin = np.min([x[0][0], x[1][0], x[2][0], x[3][0]])
            ymin = np.min([x[0][1], x[1][1], x[2][1], x[3][1]])
            x = xmin
            y = ymin
            w = xmax - xmin
            h = ymax - ymin
        else:
            logger.warning("x should be in the form  "
                           "((x,y),(x1,y1),(x2,y2),(x3,y3))")
            return None

    # x,y of the form [x1,x2,x3,x4,x5....] and y similar
    elif isinstance(x, (tuple, list)) \
            and isinstance(y, (tuple, list)) \
            and len(x) > 4 and len(y) > 4:
        if isinstance(x[0], (int, long, float)) \
                and isinstance(y[0], (int, long, float)):
            xmax = np.max(x)
            ymax = np.max(y)
            xmin = np.min(x)
            ymin = np.min(y)
            x = xmin
            y = ymin
            w = xmax - xmin
            h = ymax - ymin
        else:
            logger.warning("x should be in the form "
                           "x = [1, 2, 3, 4, 5] y = [0, 2, 4, 6, 8]")
            return None

    # x of the form [(x,y),(x,y),(x,y),(x,y),(x,y),(x,y)]
    elif isinstance(x, (list, tuple)) and len(x) > 4 \
            and len(x[0]) == 2 and y is None and w is None and h is None:
        if isinstance(x[0][0], (int, long, float)):
            xs = [pt[0] for pt in x]
            ys = [pt[1] for pt in x]
            xmax = np.max(xs)
            ymax = np.max(ys)
            xmin = np.min(xs)
            ymin = np.min(ys)
            x = xmin
            y = ymin
            w = xmax - xmin
            h = ymax - ymin
        else:
            logger.warning("x should be in the form "
                           "[(x,y),(x,y),(x,y),(x,y),(x,y),(x,y)]")
            return None

    # x of the form [(x,y),(x1,y1)]
    elif isinstance(x, (list, tuple)) and len(x) == 2 \
            and isinstance(x[0], (list, tuple)) \
            and isinstance(x[1], (list, tuple)) \
            and y is None and w is None and h is None:
        if len(x[0]) == 2 and len(x[1]) == 2:
            xt = np.min([x[0][0], x[1][0]])
            yt = np.min([x[0][1], x[1][1]])
            w = np.abs(x[0][0] - x[1][0])
            h = np.abs(x[0][1] - x[1][1])
            x = xt
            y = yt
        else:
            logger.warning("x should be in the form [(x1,y1),(x2,y2)]")
            return None

    # x and y of the form (x,y),(x1,y2)
    elif isinstance(x, (tuple, list)) and isinstance(y, (tuple, list)) \
            and w is None and h is None:
        if len(x) == 2 and len(y) == 2:
            xt = np.min([x[0], y[0]])
            yt = np.min([x[1], y[1]])
            w = np.abs(y[0] - x[0])
            h = np.abs(y[1] - x[1])
            x = xt
            y = yt

        else:
            logger.warning("if x and y are tuple it should be in the form "
                           "(x1,y1) and (x2,y2)")
            return None

    if y is None or w is None or h is None:
        print "Please provide an x, y, width, height to function"

    if w <= 0 or h <= 0:
        logger.warning("Can't do a negative crop!")
        return None

    if x < 0 or y < 0:
        logger.warning("Crop will try to help you, but you have a "
                       "negative crop position, your width and height "
                       "may not be what you want them to be.")

    if centered:
        rectangle = (int(x - (w / 2)), int(y - (h / 2)), int(w), int(h))
    else:
        rectangle = (int(x), int(y), int(w), int(h))

    (top_roi, bottom_roi) = img.rect_overlap_rois(
        (rectangle[2], rectangle[3]), (img.width, img.height),
        (rectangle[0], rectangle[1]))

    if bottom_roi is None:
        logger.warning("Hi, your crop rectangle doesn't even overlap your "
                       "image. I have no choice but to return None.")
        return None

    array = img.ndarray[img.roi_to_slice(bottom_roi)].copy()

    result_img = Factory.Image(array, color_space=img.color_space)

    #Buffering the top left point (x, y) in a image.
    result_img.uncropped_x = result_img.uncropped_x + int(x)
    result_img.uncropped_y = result_img.uncropped_y + int(y)
    return result_img


@image_method
def region_select(img, x1, y1, x2, y2):
    """
    **SUMMARY**

    Region select is similar to crop, but instead of taking a position and
    width and height values it simply takes two points on the image and
    returns the selected region. This is very helpful for creating
    interactive scripts that require the user to select a region.

    **PARAMETERS**

    * *x1* - Int - Point one x coordinate.
    * *y1* - Int  - Point one y coordinate.
    * *x2* - Int  - Point two x coordinate.
    * *y2* - Int  - Point two y coordinate.

    **RETURNS**

    A cropped SimpleCV Image.

    **EXAMPLE**

    >>> img = Image("lenna")
    >>> # often this comes from a mouse click
    >>> subreg = img.region_select(10, 10, 100, 100)
    >>> subreg.show()

    **SEE ALSO**

    :py:meth:`crop`

    """
    w = abs(x1 - x2)
    h = abs(y1 - y2)

    ret_val = None
    if w <= 0 or h <= 0 or w > img.width or h > img.height:
        logger.warning("region_select: the given values will not fit in "
                       "the image or are too small.")
    else:
        xf = x2
        if x1 < x2:
            xf = x1
        yf = y2
        if y1 < y2:
            yf = y1
        ret_val = img.crop(xf, yf, w, h)

    return ret_val


@image_method
def adaptive_scale(img, resolution, fit=True):
    """
    **SUMMARY**

    Adapative Scale is used in the Display to automatically
    adjust image size to match the display size. This method attempts to
    scale an image to the desired resolution while keeping the aspect ratio
    the same. If fit is False we simply crop and center the image to the
    resolution. In general this method should look a lot better than
    arbitrary cropping and scaling.

    **PARAMETERS**

    * *resolution* - The size of the returned image as a (width,height)
     tuple.
    * *fit* - If fit is true we try to fit the image while maintaining the
     aspect ratio. If fit is False we crop and center the image to fit the
     resolution.

    **RETURNS**

    A SimpleCV Image.

    **EXAMPLE**

    This is typically used in this instance:

    >>> d = Display((800, 600))
    >>> i = Image((640, 480))
    >>> i.save(d)

    Where this would scale the image to match the display size of 800x600

    """

    wndw_ar = float(resolution[0]) / float(resolution[1])
    img_ar = float(img.width) / float(img.height)
    targetx = 0
    targety = 0
    targetw = resolution[0]
    targeth = resolution[1]
    if img.size == resolution:  # we have to resize
        ret_val = img
        return ret_val
    elif img_ar == wndw_ar and fit:
        ret_val = img.resize(w=resolution[0], h=resolution[1])
        return ret_val
    elif fit:
        #scale factors
        ret_val = np.zeros((resolution[1], resolution[0], 3),
                           dtype='uint8')
        wscale = (float(img.width) / float(resolution[0]))
        hscale = (float(img.height) / float(resolution[1]))
        if wscale > 1:  # we're shrinking what is the percent reduction
            wscale = 1 - (1.0 / wscale)
        else:  # we need to grow the image by a percentage
            wscale = 1.0 - wscale
        if hscale > 1:
            hscale = 1 - (1.0 / hscale)
        else:
            hscale = 1.0 - hscale
        if wscale == 0:  # if we can get away with not scaling do that
            targetx = 0
            targety = (resolution[1] - img.height) / 2
            targetw = img.width
            targeth = img.height
        elif hscale == 0:  # if we can get away with not scaling do that
            targetx = (resolution[0] - img.width) / 2
            targety = 0
            targetw = img.width
            targeth = img.height
        elif wscale < hscale:  # the width has less distortion
            sfactor = float(resolution[0]) / float(img.width)
            targetw = int(float(img.width) * sfactor)
            targeth = int(float(img.height) * sfactor)
            if targetw > resolution[0] or targeth > resolution[1]:
                #aw shucks that still didn't work do the other way instead
                sfactor = float(resolution[1]) / float(img.height)
                targetw = int(float(img.width) * sfactor)
                targeth = int(float(img.height) * sfactor)
                targetx = (resolution[0] - targetw) / 2
                targety = 0
            else:
                targetx = 0
                targety = (resolution[1] - targeth) / 2
            img = img.resize(w=targetw, h=targeth)
        else:  # the height has more distortion
            sfactor = float(resolution[1]) / float(img.height)
            targetw = int(float(img.width) * sfactor)
            targeth = int(float(img.height) * sfactor)
            if targetw > resolution[0] or targeth > resolution[1]:
                # aw shucks that still didn't work do the other way instead
                sfactor = float(resolution[0]) / float(img.width)
                targetw = int(float(img.width) * sfactor)
                targeth = int(float(img.height) * sfactor)
                targetx = 0
                targety = (resolution[1] - targeth) / 2
            else:
                targetx = (resolution[0] - targetw) / 2
                targety = 0
            img = img.resize(w=targetw, h=targeth)

    else:  # we're going to crop instead
        # center a too small image
        if img.width <= resolution[0] and img.height <= resolution[1]:
            #we're too small just center the thing
            ret_val = np.zeros((resolution[1], resolution[0], 3),
                               dtype=np.uint8)
            targetx = (resolution[0] / 2) - (img.width / 2)
            targety = (resolution[1] / 2) - (img.height / 2)
            targeth = img.height
            targetw = img.width
        # crop too big on both axes
        elif img.width > resolution[0] and img.height > resolution[1]:
            targetw = resolution[0]
            targeth = resolution[1]
            targetx = 0
            targety = 0
            x = (img.width - resolution[0]) / 2
            y = (img.height - resolution[1]) / 2
            img = img.crop(x, y, targetw, targeth)
            return img
        # height too big
        elif img.width <= resolution[0] and img.height > resolution[1]:
            # crop along the y dimension and center along the x dimension
            ret_val = np.zeros((resolution[1], resolution[0], 3),
                               dtype=np.uint8)
            targetw = img.width
            targeth = resolution[1]
            targetx = (resolution[0] - img.width) / 2
            targety = 0
            x = 0
            y = (img.height - resolution[1]) / 2
            img = img.crop(x, y, targetw, targeth)

        # width too big
        elif img.width > resolution[0] and img.height <= resolution[1]:
            # crop along the y dimension and center along the x dimension
            ret_val = np.zeros((resolution[1], resolution[0], 3),
                               dtype=np.uint8)
            targetw = resolution[0]
            targeth = img.height
            targetx = 0
            targety = (resolution[1] - img.height) / 2
            x = (img.width - resolution[0]) / 2
            y = 0
            img = img.crop(x, y, targetw, targeth)

    ret_val[targety:targety + targeth,
            targetx:targetx + targetw] = img.ndarray
    ret_val = Factory.Image(array=ret_val, color_space=img.color_space)
    return ret_val


@image_method
def blit(img, top_img, pos=None, alpha=None, mask=None,
         alpha_mask=None):
    """
    **SUMMARY**

    Blit aka bit blit - which in ye olden days was an acronym for bit-block
    transfer. In other words blit is when you want to smash two images
    together, or add one image to another. This method takes in a second
    simplecv image, and then allows you to add to some point on the calling
    image. A general blit command will just copy all of the image. You can
    also copy the image with an alpha value to the source image is
    semi-transparent. A binary mask can be used to blit non-rectangular
    image onto the souce image. An alpha mask can be used to do and
    arbitrarily transparent image to this image. Both the mask and alpha
    masks are SimpleCV Images.

    **PARAMETERS**

    * *img* - an image to place ontop of this image.
    * *pos* - an (x,y) position tuple of the top left corner of img on this
     image. Note that these values can be negative.
    * *alpha* - a single floating point alpha value
     (0=see the bottom image, 1=see just img, 0.5 blend the two 50/50).
    * *mask* - a binary mask the same size as the input image.
     White areas are blitted, black areas are not blitted.
    * *alpha_mask* - an alpha mask where each grayscale value maps how much
    of each image is shown.

    **RETURNS**

    A SimpleCV Image. The size will remain the same.

    **EXAMPLE**

    >>> topImg = Image("top.png")
    >>> bottomImg = Image("bottom.png")
    >>> mask = Image("mask.png")
    >>> aMask = Image("alpphaMask.png")
    >>> bottomImg.blit(top, pos=(100, 100)).show()
    >>> bottomImg.blit(top, alpha=0.5).show()
    >>> bottomImg.blit(top, pos=(100, 100), mask=mask).show()
    >>> bottomImg.blit(top, pos=(-10, -10), alpha_mask=aMask).show()

    **SEE ALSO**

    :py:meth:`create_binary_mask`
    :py:meth:`create_alpha_mask`

    """
    if pos is None:
        pos = (0, 0)

    (top_roi, bottom_roi) = img.rect_overlap_rois(top_img.size,
                                                  img.size, pos)

    if alpha:
        top_img = top_img.copy().crop(*top_roi)
        bottom_img = img.copy().crop(*bottom_roi)
        alpha = float(alpha)
        beta = float(1.00 - alpha)
        gamma = float(0.00)
        blit_array = cv2.addWeighted(top_img.ndarray, alpha,
                                     bottom_img.ndarray, beta, gamma)
        array = img._ndarray.copy()
        array[bottom_roi[1]:bottom_roi[1] + bottom_roi[3],
              bottom_roi[0]:bottom_roi[0] + bottom_roi[2]] = blit_array
        return Factory.Image(array, color_space=img.color_space)
    elif alpha_mask:
        if alpha_mask.size != top_img.size:
            logger.warning("Image.blit: your mask and image don't match "
                           "sizes, if the mask doesn't fit, you can not "
                           "blit! Try using the scale function.")
            return None
        top_img = top_img.copy().crop(*top_roi)
        bottom_img = img.copy().crop(*bottom_roi)
        mask_img = alpha_mask.crop(*top_roi)
        # Apply mask to img
        top_array = top_img.ndarray.astype(np.float32)
        gray_mask_array = mask_img.gray_ndarray
        gray_mask_array = gray_mask_array.astype(np.float32) / 255.0
        gray_mask_array = np.dstack((gray_mask_array, gray_mask_array,
                                     gray_mask_array))
        masked_top_array = cv2.multiply(top_array, gray_mask_array)
        # Apply inverted mask to img
        bottom_array = bottom_img.ndarray.astype(np.float32)
        inv_graymask_array = mask_img.invert().gray_ndarray
        inv_graymask_array = inv_graymask_array.astype(np.float32) / 255.0
        inv_graymask_array = np.dstack((inv_graymask_array,
                                        inv_graymask_array,
                                        inv_graymask_array))
        masked_bottom_array = cv2.multiply(bottom_array,
                                           inv_graymask_array)

        blit_array = cv2.add(masked_top_array, masked_bottom_array)
        blit_array = blit_array.astype(img.dtype)

        array = img._ndarray.copy()
        array[bottom_roi[1]:bottom_roi[1] + bottom_roi[3],
              bottom_roi[0]:bottom_roi[0] + bottom_roi[2]] = blit_array
        return Factory.Image(array, color_space=img.color_space)

    elif mask:
        if mask.size != top_img.size:
            logger.warning("Image.blit: your mask and image don't match "
                           "sizes, if the mask doesn't fit, you can not "
                           "blit! Try using the scale function. ")
            return None
        top_img = top_img.copy().crop(*top_roi)
        mask_img = mask.crop(*top_roi)
        # Apply mask to img
        top_array = top_img.ndarray
        gray_mask_array = mask_img.gray_ndarray
        binary_mask = gray_mask_array != 0
        array = img._ndarray.copy()
        array[img.roi_to_slice(bottom_roi)][binary_mask] = \
            top_array[binary_mask]
        return Factory.Image(array, color_space=img.color_space)

    else:  # vanilla blit
        top_img = top_img.copy().crop(*top_roi)
        array = img._ndarray.copy()
        array[bottom_roi[1]:bottom_roi[1] + bottom_roi[3],
              bottom_roi[0]:bottom_roi[0] + bottom_roi[2]] = \
            top_img.ndarray
        return Factory.Image(array, color_space=img.color_space)


@image_method
def side_by_side(image1, image2, side="right", scale=True):
    """
    **SUMMARY**

    Combine two images as a side by side images. Great for before and after
    images.

    **PARAMETERS**

    * *side* - what side of this image to place the other image on.
      choices are ('left'/'right'/'top'/'bottom').

    * *scale* - if true scale the smaller of the two sides to match the
      edge touching the other image. If false we center the smaller
      of the two images on the edge touching the larger image.

    **RETURNS**

    A new image that is a combination of the two images.

    **EXAMPLE**

    >>> img = Image("lenna")
    >>> img2 = Image("orson_welles.jpg")
    >>> img3 = img.side_by_side(img2)

    **TODO**

    Make this accept a list of images.

    """
    # there is probably a cleaner way to do this, but I know I hit every
    # case when they are enumerated
    if side == "top":
        return image2.side_by_side(image1, "bottom", scale)
    elif side == "bottom":
        if image1.width > image2.width:
            # scale the other image width to fit
            resized = image2.resize(w=image1.width) \
                if scale else image2
            topimage = image1
            w = image1.width
        else:  # our width is smaller than the other image
            # scale the other image width to fit
            topimage = image1.resize(w=image2.width) \
                if scale else image1
            resized = image2
            w = image2.width
        h = topimage.height + resized.height
        xc = (topimage.width - resized.width) / 2
        array = np.zeros((h, w, 3), dtype=image1.dtype)
        if xc > 0:
            array[:topimage.height, :topimage.width] = \
                topimage.ndarray
            array[h - resized.height:, xc:xc + resized.width] = \
                resized.ndarray
        else:
            array[:topimage.height, abs(xc):abs(xc) + topimage.width] = \
                topimage.ndarray
            array[h - resized.height:, :resized.width] = \
                resized.ndarray
        return Factory.Image(array, color_space=image1.color_space)
    elif side == "right":
        return image2.side_by_side(image1, "left", scale)
    else:  # default to left
        if image1.height > image2.height:
            # scale the other image height to fit
            resized = image2.resize(h=image1.height) \
                if scale else image2
            rightimage = image1
            h = rightimage.height
        else:  # our height is smaller than the other image
            #scale our height to fit
            rightimage = image1.resize(h=image2.height) \
                if scale else image1
            h = image2.height
            resized = image2
        w = rightimage.width + resized.width
        yc = (rightimage.height - resized.height) / 2
        array = np.zeros((h, w, 3), dtype=image1.dtype)
        if yc > 0:
            array[:rightimage.height, w - rightimage.width:] = \
                rightimage.ndarray
            array[yc:yc + resized.height, :resized.width] = \
                resized.ndarray
        else:
            array[abs(yc):abs(yc) + rightimage.height,
                  w - rightimage.width:] = rightimage.ndarray
            array[:resized.height, :resized.width] = resized.ndarray
        return Factory.Image(array, color_space=image1.color_space)


@image_method
def embiggen(img, size=None, color=Color.BLACK, pos=None):
    """
    **SUMMARY**

    Make the canvas larger but keep the image the same size.

    **PARAMETERS**

    * *size* - width and heigt tuple of the new canvas or give a single
     vaule in which to scale the image size, for instance size=2 would make
     the image canvas twice the size

    * *color* - the color of the canvas

    * *pos* - the position of the top left corner of image on the new
     canvas, if none the image is centered.

    **RETURNS**

    The enlarged SimpleCV Image.

    **EXAMPLE**

    >>> img = Image("lenna")
    >>> img = img.embiggen((1024, 1024), color=Color.BLUE)
    >>> img.show()

    """
    if not (img.is_bgr() or img.is_gray()):
        logger.warning("Image.embiggen works only with "
                       "bgr and gray images")
        return None

    if not isinstance(size, tuple) and size > 1:
        size = (img.width * size, img.height * size)
    elif size < 1:
        logger.warning("embiggen size must be greater than 1")
        return None

    if size is None or size[0] < img.width or size[1] < img.height:
        logger.warning("Image.embiggen: the size provided is invalid")
        return None
    if img.is_gray():
        array = np.zeros((size[1], size[0]), dtype=img.dtype)
        array[:, :] = Color.get_average_rgb(color)
    else:
        array = np.zeros((size[1], size[0], 3), dtype=img.dtype)
        array[:, :, :] = color[::-1]  # RBG to BGR
    if pos is None:
        pos = (((size[0] - img.width) / 2), ((size[1] - img.height) / 2))
    (top_roi, bottom_roi) = img.rect_overlap_rois(img.size, size, pos)
    if top_roi is None or bottom_roi is None:
        logger.warning("Image.embiggen: the position of the old image "
                       "doesn't make sense, there is no overlap")
        return None
    blit_array = img._ndarray[img.roi_to_slice(top_roi)]
    array[img.roi_to_slice(bottom_roi)] = blit_array
    return Factory.Image(array, color_space=img.color_space)


@image_method
def rotate270(img):
    """
    **DESCRIPTION**

    Rotate the image 270 degrees to the left, the same as 90 degrees to
    the right. This is the same as rotate_right()

    **RETURNS**

    A SimpleCV image.

    **EXAMPLE**

    >>>> img = Image('lenna')
    >>>> img.rotate270().show()

    """
    array = cv2.flip(img.ndarray, flipCode=0)  # vertical
    array = cv2.transpose(array)
    return Factory.Image(array, color_space=img.color_space)


@image_method
def rotate90(img):
    """
    **DESCRIPTION**

    Rotate the image 90 degrees to the left, the same as 270 degrees to the
    right. This is the same as rotate_right()

    **RETURNS**

    A SimpleCV image.

    **EXAMPLE**

    >>>> img = Image('lenna')
    >>>> img.rotate90().show()

    """
    array = cv2.transpose(img.ndarray)
    array = cv2.flip(array, flipCode=0)  # vertical
    return Factory.Image(array, color_space=img.color_space)


@image_method
def rotate_left(img):  # same as 90
    """
    **DESCRIPTION**

    Rotate the image 90 degrees to the left.
    This is the same as rotate 90.

    **RETURNS**

    A SimpleCV image.

    **EXAMPLE**

    >>>> img = Image('lenna')
    >>>> img.rotate_left().show()

    """
    return img.rotate90()


@image_method
def rotate_right(img):  # same as 270
    """
    **DESCRIPTION**

    Rotate the image 90 degrees to the right.
    This is the same as rotate 270.

    **RETURNS**

    A SimpleCV image.

    **EXAMPLE**

    >>>> img = Image('lenna')
    >>>> img.rotate_right().show()

    """
    return img.rotate270()


@image_method
def rotate180(img):
    """
    **DESCRIPTION**

    Rotate the image 180 degrees to the left/right.
    This is the same as rotate 90.

    **RETURNS**

    A SimpleCV image.

    **EXAMPLE**

    >>>> img = Image('lenna')
    >>>> img.rotate180().show()
    """
    array = cv2.flip(img.ndarray, flipCode=0)  # vertical
    array = cv2.flip(array, flipCode=1)  # horizontal
    return Factory.Image(array, color_space=img.color_space)
