import itertools
import math

import cv2
import numpy as np
import scipy.cluster.vq as scv
import scipy.linalg as nla  # for linear algebra / least squares

from simplecv.base import logger, ScvException
from simplecv.color import Color
from simplecv.core.image import image_method
from simplecv.factory import Factory
from simplecv.features.detection import Corner, KeyPoint, Line
from simplecv.features.features import FeatureSet


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
        temp = temp.to_gray()

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
            raise ScvException("Can't Do snakeFitPoints without "
                               "OpenCV >= 2.3.0")

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
        raise ScvException("Cannot find grid on the image, Try adding a grid first")

    grid_index = img.get_drawing_layer(img._grid_layer[0])

    try:
        step_row = img.size_tuple[1] / img._grid_layer[1][0]
        step_col = img.size_tuple[0] / img._grid_layer[1][1]
    except ZeroDivisionError:
        return FeatureSet()

    i = 1
    j = 1

    line_fs = FeatureSet()
    while i < img._grid_layer[1][0]:
        line_fs.append(Factory.Line(img, ((0, step_row * i),
                                          (img.size_tuple[0], step_row * i))))
        i = i + 1
    while j < img._grid_layer[1][1]:
        line_fs.append(Factory.Line(img, ((step_col * j, 0),
                                          (step_col * j, img.size_tuple[1]))))
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
        raise ScvException("OpenCV >= 2.4.3 required")

    if template is None:
        raise ScvException('template should not be None')

    detector = cv2.FeatureDetector_create("SIFT")
    descriptor = cv2.DescriptorExtractor_create("SIFT")
    img_array = img
    template_img = template

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
        keypoints = img.find(Corner)  # fallback to corners
    else:
        keypoints = img.find(KeyPoint,
                             flavor=flavor.upper())  # find the keypoints
    if keypoints is None or keypoints <= 0:
        return FeatureSet()

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
        raise ScvException("OpenCV version >= 2.4.2 requierd")

    if int(cv2.__version__.replace('.', '0')) < 20402:
        raise ScvException("OpenCV version >= 2.4.2 requierd")

    flavors = ["SIFT", "SURF", "BRISK", "ORB", "STAR", "MSER", "FAST",
               "Dense"]
    if flavor not in flavors:
        raise ScvException("Unkown Keypoints detector. Returning None.")

    detector = cv2.FeatureDetector_create(flavor)
    extractor = cv2.DescriptorExtractor_create("FREAK")
    img._key_points = detector.detect(img.to_gray())
    img._key_points, img._kp_descriptors = extractor.compute(
        img.to_gray(),
        img._key_points)
    fs = FeatureSet()
    for i in range(len(img._key_points)):
        fs.append(Factory.KeyPoint(img, img._key_points[i],
                                   img._kp_descriptors[i], flavor))

    return fs, img._kp_descriptors


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
    img_array = img.to_gray().transpose()
    c1 = np.count_nonzero(img_array)
    c2 = np.count_nonzero(img_array - 255)

    #checking that all values are 0 and 255
    if c1 + c2 != img_array.size:
        raise ValueError("Image must be binary")

    if len(point_list) < 2:
        return FeatureSet()

    final_list = [point_list[0]]
    last = point_list[0]
    for point in point_list[1:]:
        final_list += img._edge_snap2(last, point, step)
        last = point

    last = final_list[0]
    feature_set = FeatureSet()
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

    edge_map = np.copy(img.to_gray().transpose())

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
    lines = img.find(Line, threshold, min_length, max_gap, t1, t2)

    if len(lines) == 0:
        logger.warning("No lines found in the image")
        return img

    # Initialize empty bins
    binn = [[] for i in range(bins)]

    #Convert angle to bin number
    conv = lambda x: int(x + 90) / bins

    #Adding lines to bins
    [binn[conv(line.angle)].append(line) for line in lines]

    #computing histogram, value of each column is total length of all lines
    #in the bin
    hist = [sum([line.length for line in lines]) for lines in binn]

    #The maximum histogram
    index = np.argmax(np.array(hist))

    #Good ol weighted mean, for the selected bin
    avg = sum([line.angle * line.length for line in binn[index]]) \
        / sum([line.length for line in binn[index]])

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
