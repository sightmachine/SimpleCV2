#SimpleCV Detection Library
#
#This library includes classes for finding things in images

#FYI -
#All angles shalt be described in degrees with zero pointing east in the
#plane of the image with all positive rotations going counter-clockwise.
#Therefore a rotation from the x-axis to to the y-axis is positive and follows
#the right hand rule.

from copy import copy
from math import atan2, sqrt, pi, sin, cos, radians

import cv2
import numpy as np
import pickle
import scipy.spatial.distance as spsd
import math

from simplecv.base import logger, lazyproperty, force_update_lazyproperties
from simplecv.color import Color
from simplecv.core.pluginsystem import apply_plugins
from simplecv.factory import Factory
from simplecv.features.features import Feature, FeatureSet


@apply_plugins
class Corner(Feature):
    """
    **SUMMARY**

    The Corner feature is a point returned by the FindCorners function
    Corners are used in machine vision as a very computationally efficient way
    to find unique features in an image.  These corners can be used in
    conjunction with many other algorithms.

    **SEE ALSO**

    :py:meth:`find_corners`
    """

    def __init__(self, i, at_x, at_y):
        points = [(at_x - 1, at_y - 1), (at_x - 1, at_y + 1),
                  (at_x + 1, at_y + 1), (at_x + 1, at_y - 1)]
        super(Corner, self).__init__(i, at_x, at_y, points)
        #can we look at the eigenbuffer and find direction?

    def draw(self, color=(255, 0, 0), width=1):
        """
        **SUMMARY**

        Draw a small circle around the corner.  Color tuple is single
        parameter, default is Red.

        **PARAMETERS**

        * *color* - An RGB color triplet.
        * *width* - if width is less than zero we draw the feature filled in,
         otherwise we draw the get_contour using the specified width.


        **RETURNS**

        Nothing - this is an inplace operation that modifies the source images
        drawing layer.
        """
        self.image.draw_circle((self.x, self.y), 4, color, width)

    @classmethod
    def find(cls, img, maxnum=50, minquality=0.04, mindistance=1.0):
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
        corner_coordinates = cv2.goodFeaturesToTrack(img.gray_ndarray,
                                                     maxCorners=maxnum,
                                                     qualityLevel=minquality,
                                                     minDistance=mindistance)
        corner_features = []
        for x, y in corner_coordinates[:, 0, :]:
            corner_features.append(Factory.Corner(img, x, y))

        return FeatureSet(corner_features)


######################################################################
@apply_plugins
class Line(Feature):
    """
    **SUMMARY**

    The Line class is returned by the find_lines function, but can also be
    initialized with any two points.

    >>> l = Line(Image, (point1, point2))

    Where point1 and point2 are (x,y) coordinate tuples.

    >>> l.points

    Returns a tuple of the two points


    """
    #TODO - A nice feature would be to calculate the endpoints of the line.

    def __init__(self, i, line):
        self.image = i
        self.end_points = list(copy(line))

        if self.end_points[1][0] - self.end_points[0][0] == 0:
            self.slope = float("inf")
        else:
            self.slope = float(self.end_points[1][1] - self.end_points[0][1]) \
                / float(self.end_points[1][0] - self.end_points[0][0])
        #coordinate of the line object is the midpoint
        at_x = (line[0][0] + line[1][0]) / 2
        at_y = (line[0][1] + line[1][1]) / 2
        xmin = int(np.min([line[0][0], line[1][0]]))
        xmax = int(np.max([line[0][0], line[1][0]]))
        ymax = int(np.min([line[0][1], line[1][1]]))
        ymin = int(np.max([line[0][1], line[1][1]]))
        points = [(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin)]
        super(Line, self).__init__(i, at_x, at_y, points)

    def draw(self, color=(0, 0, 255), width=1):
        """
        Draw the line, default color is blue

        **SUMMARY**

        Draw a small circle around the corner.  Color tuple is single
        parameter, default is Red.

        **PARAMETERS**

        * *color* - An RGB color triplet.
        * *width* - Draw the line using the specified width.

        **RETURNS**

        Nothing - this is an inplace operation that modifies the source images
        drawing layer.
        """
        self.image.draw_line(self.end_points[0], self.end_points[1], color,
                             width)

    @property
    def length(self):
        """

        **SUMMARY**

        This property returns the length of the line.

        **RETURNS**

        A floating point length value.

        **EXAMPLE**

        >>> img = Image("OWS.jpg")
        >>> lines = img.find_lines
        >>> for l in lines:
        >>>    if l.length > 100:
        >>>       print "OH MY! - WHAT A BIG LINE YOU HAVE!"
        >>>       print "---I bet you say that to all the lines."

        """
        return float(spsd.euclidean(self.end_points[0], self.end_points[1]))

    def crop(self):
        """
        **SUMMARY**

        This function crops the source image to the location of the feature and
        returns a new simplecv image.

        **RETURNS**

        A SimpleCV image that is cropped to the feature position and size.

        **EXAMPLE**

        >>> img = Image("../sampleimages/EdgeTest2.png")
        >>> l = img.find_lines()
        >>> myLine = l[0].crop()

        """
        tlc = self.top_left_corner
        return self.image.crop(tlc[0], tlc[1], self.width,
                               self.height)

    @property
    def mean_color(self):
        """
        **SUMMARY**

        Returns the mean color of pixels under the line.  Note that when the
        line falls "between" pixels, each pixels color contributes to the
        weighted average.


        **RETURNS**

        Returns an RGB triplet corresponding to the mean color of the feature.

        **EXAMPLE**

        >>> img = Image("lenna")
        >>> l = img.find_lines()
        >>> c = l[0].mean_color

        """
        (pt1, _) = self.end_points
        #we're going to walk the line, and take the mean color from all the px
        #points -- there's probably a much more optimal way to do this
        (maxx, minx, maxy, miny) = self.extents

        d_x = maxx - minx
        d_y = maxy - miny
        #orient the line so it is going in the positive direction

        #if it's a straight one, we can just get mean color on the slice
        if d_x == 0.0:
            return self.image[miny:maxy, pt1[0]:pt1[0]+1].mean_color()
            # return self.image[pt1[0]:pt1[0] + 1, miny:maxy].mean_color()
        if d_y == 0.0:
            return self.image[pt1[1]:pt1[1]+1, minx:maxx].mean_color()
            # return self.image[minx:maxx, pt1[1]:pt1[1] + 1].mean_color()

        error = 0.0
        # this is how much our "error" will increase in every step
        d_err = d_y / d_x
        px = []
        weights = []

        if d_err < 1:
            y = miny
            #iterate over X
            for x in range(minx, maxx):
                #this is the pixel we would draw on, check the color at that px
                #weight is reduced from 1.0 by the abs amount of error
                px.append(self.image[x, y])
                weights.append(1.0 - abs(error))

                # if we have error in either direction, we're going to use the
                # px above or below
                if error > 0:
                    px.append(self.image[x, y + 1])
                    weights.append(error)

                if error < 0:
                    px.append(self.image[x, y - 1])
                    weights.append(abs(error))

                error = error + d_err
                if error >= 0.5:
                    y += 1
                    error -= 1.0
        else:
            #this is a "steep" line, so we iterate over X
            #copy and paste.  Ugh, sorry.
            x = minx
            for y in range(miny, maxy):
                #this is the pixel we would draw on, check the color at that px
                #weight is reduced from 1.0 by the abs amount of error
                px.append(self.image[x, y])
                weights.append(1.0 - abs(error))

                # if we have error in either direction, we're going to use the
                # px above or below
                if error > 0:  #
                    px.append(self.image[x + 1, y])
                    weights.append(error)

                if error < 0:
                    px.append(self.image[x - 1, y])
                    weights.append(abs(error))

                error += 1.0 / d_err  # we use the reciprocal of error
                if error >= 0.5:
                    x += 1
                    error -= 1.0

        #once we have iterated over every pixel in the line, we avg the weights
        clr_arr = np.array(px)
        weight_arr = np.array(weights)

        weighted_clrs = np.transpose(np.transpose(clr_arr) * weight_arr)
        #multiply each color tuple by its weight

        temp = sum(weighted_clrs) / sum(weight_arr)  # return the weighted avg
        return float(temp[0]), float(temp[1]), float(temp[2])

    def find_intersection(self, line):
        """
        **SUMMARY**

        Returns the interesction point of two lines.

        **RETURNS**

        A point tuple.

        **EXAMPLE**

        >>> img = Image("lenna")
        >>> l = img.find_lines()
        >>> c = l[0].find_intersection[1]


        TODO: THIS NEEDS TO RETURN A TUPLE OF FLOATS
        """
        if self.slope == float("inf"):
            x = self.end_points[0][0]
            y = line.slope * (x - line.end_points[1][0]) + line.end_points[1][
                1]
            return x, y

        if line.slope == float("inf"):
            x = line.end_points[0][0]
            y = self.slope * (x - self.end_points[1][0]) + self.end_points[1][
                1]
            return x, y

        m1 = self.slope
        x12, y12 = self.end_points[1]
        m2 = line.slope
        x22, y22 = line.end_points[1]

        x = (m1 * x12 - m2 * x22 + y22 - y12) / float(m1 - m2)
        y = (m1 * m2 * (x12 - x22) - m2 * y12 + m1 * y22) / float(m1 - m2)

        return x, y

    def is_parallel(self, line):
        """
        **SUMMARY**

        Checks whether two lines are parallel or not.

        **RETURNS**

        Bool. True or False

        **EXAMPLE**

        >>> img = Image("lenna")
        >>> l = img.find_lines()
        >>> c = l[0].is_parallel(l[1])

        """
        if self.slope == line.slope:
            return True
        return False

    def is_perpendicular(self, line):
        """
        **SUMMARY**

        Checks whether two lines are perpendicular or not.

        **RETURNS**

        Bool. True or False

        **EXAMPLE**

        >>> img = Image("lenna")
        >>> l = img.find_lines()
        >>> c = l[0].is_perpendicular(l[1])

        """
        if self.slope == float("inf"):
            if line.slope == 0:
                return True
            return False

        if line.slope == float("inf"):
            if self.slope == 0:
                return True
            return False

        if self.slope * line.slope == -1:
            return True
        return False

    def img_intersections(self, img):
        """
        **SUMMARY**

        Returns a set of pixels where the line intersects with the binary
        image.

        **RETURNS**

        list of points.

        **EXAMPLE**

        >>> img = Image("lenna")
        >>> l = img.find_lines()
        >>> c = l[0].img_intersections(img.binarize())

        """
        pixels = []
        if self.slope == float("inf"):
            for y in range(self.end_points[0][1], self.end_points[1][1] + 1):
                pixels.append((self.end_points[0][0], y))
        elif self.slope == 0.0:
            for x in range(self.end_points[0][0], self.end_points[1][0] + 1):
                pixels.append((x, self.end_points[0][1]))
        else:
            for x in range(self.end_points[0][0], self.end_points[1][0] + 1):
                pixels.append((x, int(self.end_points[1][1] + self.slope *
                                      (x - self.end_points[1][0]))))
            for y in range(self.end_points[0][1], self.end_points[1][1] + 1):
                pixels.append((int(((y - self.end_points[1][1]) / self.slope) +
                                   self.end_points[1][0]), y))
        pixels = list(set(pixels))
        matched_pixels = []
        for pixel in pixels:
            if img[pixel[1], pixel[0]] == [255.0, 255.0, 255.0]:
                matched_pixels.append(pixel)
        matched_pixels.sort()

        return matched_pixels

    @lazyproperty
    def angle(self):
        """
        **SUMMARY**

        This is the angle of the line, from the leftmost point to the rightmost
        point
        Returns angle (theta) in radians, with 0 = horizontal,
        -pi/2 = vertical positive slope, pi/2 = vertical negative slope

        **RETURNS**

        An angle value in degrees.

        **EXAMPLE**

        >>> img = Image("OWS.jpg")
        >>> ls = img.find_lines
        >>> for l in ls:
        >>>    if l.get_angle() == 0:
        >>>       print "I AM HORIZONTAL."


        """
        #first find the leftmost point
        a = 0
        b = 1
        if self.end_points[a][0] > self.end_points[b][0]:
            a, b = b, a

        d_x = self.end_points[b][0] - self.end_points[a][0]
        d_y = self.end_points[b][1] - self.end_points[a][1]
        #our internal standard is degrees
        return float(
            360.00 * (atan2(d_y, d_x) / (2 * np.pi)))  # formerly 0 was west

    def crop_to_image_edges(self):
        """
        **SUMMARY**

        Returns the line with endpoints on edges of image. If some endpoints
        lies inside image then those points remain the same without extension
        to the edges.

        **RETURNS**

        Returns a :py:class:`Line` object. If line does not cross the image's
        edges or cross at one point returns None.

        **EXAMPLE**

        >>> img = Image("lenna")
        >>> l = Line(img, ((-100, -50), (1000, 25))
        >>> cr_l = l.crop_to_image_edges()

        """
        pt1, pt2 = self.end_points
        pt1, pt2 = min(pt1, pt2), max(pt1, pt2)
        x1, y1 = pt1
        x2, y2 = pt2
        w, h = self.image.width - 1, self.image.height - 1
        slope = self.slope

        ep = []
        if slope == float('inf'):
            if 0 <= x1 <= w and 0 <= x2 <= w:
                ep.append((x1, 0))
                ep.append((x2, h))
        elif slope == 0:
            if 0 <= y1 <= w and 0 <= y2 <= w:
                ep.append((0, y1))
                ep.append((w, y2))
        else:
            x = (slope * x1 - y1) / slope  # top edge y = 0
            if 0 <= x <= w:
                ep.append((int(round(x)), 0))

            x = (slope * x1 + h - y1) / slope  # bottom edge y = h
            if 0 <= x <= w:
                ep.append((int(round(x)), h))

            y = -slope * x1 + y1  # left edge x = 0
            if 0 <= y <= h:
                ep.append((0, (int(round(y)))))

            y = slope * (w - x1) + y1  # right edge x = w
            if 0 <= y <= h:
                ep.append((w, (int(round(y)))))

        # remove duplicates of points if line cross image at corners
        ep = list(set(ep))
        ep.sort()
        if len(ep) == 2:
            # if points lies outside image then change them
            if not (0 < x1 < w and 0 < y1 < h):
                pt1 = ep[0]
            if not (0 < x2 < w and 0 < y2 < h):
                pt2 = ep[1]
        elif len(ep) == 1:
            logger.warning("Line cross the image only at one point")
            return None
        else:
            logger.warning("Line does not cross the image")
            return None

        return Line(self.image, (pt1, pt2))

    @lazyproperty
    def vector(self):
        return [float(self.end_points[1][0] - self.end_points[0][0]),
                float(self.end_points[1][1] - self.end_points[0][1])]

    def dot(self, other):
        return np.dot(self.vector, other.vector)

    def cross(self, other):
        return np.cross(self.vector, other.vector)

    @lazyproperty
    def y_intercept(self):
        """
        **SUMMARY**

        Returns the y intercept based on the lines equation.  Note that this
        point is potentially not contained in the image itself

        **RETURNS**

        Returns a floating point intersection value

        **EXAMPLE**

        >>> img = Image("lenna")
        >>> l = Line(img, ((50, 150), (2, 225))
        >>> b = l.y_intercept
        """
        pt1, _ = self.end_points
        #y = mx + b | b = y-mx
        return pt1[1] - self.slope * pt1[0]

    def extend_to_image_edges(self):
        """
        **SUMMARY**

        Returns the line with endpoints on edges of image.

        **RETURNS**

        Returns a :py:class:`Line` object. If line does not lies entirely
        inside image then returns None.

        **EXAMPLE**

        >>> img = Image("lenna")
        >>> l = Line(img, ((50, 150), (2, 225))
        >>> cr_l = l.extend_to_image_edges()

        """
        pt1, pt2 = self.end_points
        pt1, pt2 = min(pt1, pt2), max(pt1, pt2)
        x1, y1 = pt1
        x2, y2 = pt2
        w, h = self.image.width - 1, self.image.height - 1
        slope = self.slope

        if not 0 <= x1 <= w or not 0 <= x2 <= w or not 0 <= y1 <= w \
                or not 0 <= y2 <= w:
            logger.warning("At first the line should be cropped")
            return None

        ep = []
        if slope == float('inf'):
            if 0 <= x1 <= w and 0 <= x2 <= w:
                return Line(self.image, ((x1, 0), (x2, h)))
        elif slope == 0:
            if 0 <= y1 <= w and 0 <= y2 <= w:
                return Line(self.image, ((0, y1), (w, y2)))
        else:
            x = (slope * x1 - y1) / slope  # top edge y = 0
            if 0 <= x <= w:
                ep.append((int(round(x)), 0))

            x = (slope * x1 + h - y1) / slope  # bottom edge y = h
            if 0 <= x <= w:
                ep.append((int(round(x)), h))

            y = -slope * x1 + y1  # left edge x = 0
            if 0 <= y <= h:
                ep.append((0, (int(round(y)))))

            y = slope * (w - x1) + y1  # right edge x = w
            if 0 <= y <= h:
                ep.append((w, (int(round(y)))))

        # remove duplicates of points if line cross image at corners
        ep = list(set(ep))
        ep.sort()
        print type(ep), "typeof"
        return Line(self.image, ep)

    # this function contains two functions -- the basic edge detection algorithm
    # and then a function to break the lines down given a threshold parameter
    @classmethod
    def find(cls, img, threshold=80, minlinelength=30, maxlinegap=10,
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
            lines = cv2.HoughLines(em, rho=1.0, theta=pi/180.0,
                                   threshold=threshold, srn=minlinelength,
                                   stn=maxlinegap)
            if lines is not None:
                lines = lines[0]
            else:
                logger.warn("no lines found.")
                return []

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
                a = cos(theta)
                b = sin(theta)
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
                                    maxLineGap=maxlinegap)
            if lines is not None:
                lines = lines[0]
            else:
                logger.warn("no lines found.")
                return []

            if nlines == -1:
                nlines = lines.shape[0]

            for l in lines[:nlines]:
                lines_fs.append(Factory.Line(img, ((l[0], l[1]), (l[2], l[3]))))

        return lines_fs


######################################################################
@apply_plugins
class Chessboard(Feature):
    """
    **SUMMARY**

    This class is used for Calibration, it uses a chessboard
    to calibrate from pixels to real world measurements.
    """

    def __init__(self, i, dim, subpixel_corners):
        self.dimensions = dim
        self.sp_corners = subpixel_corners
        at_x, at_y = np.average(self.sp_corners[:, 0], axis=0)

        posdiagsorted = sorted(self.sp_corners,
                               key=lambda corner: corner[0][0] + corner[0][1])
        #sort corners along the x + y axis
        negdiagsorted = sorted(self.sp_corners,
                               key=lambda corner: corner[0][0] - corner[0][1])
        #sort corners along the x - y axis

        points = (posdiagsorted[0][0], negdiagsorted[-1][0], posdiagsorted[-1][0],
                  negdiagsorted[0][0])
        super(Chessboard, self).__init__(i, at_x, at_y, points)

    def draw(self, no_needed_color=None):
        """
        **SUMMARY**


        Draws the chessboard corners.  We take a color param, but ignore it.

        **PARAMETERS**

        * *no_needed_color* - An RGB color triplet that isn't used


        **RETURNS**

        Nothing - this is an inplace operation that modifies the source images
        drawing layer.

        """
        cv2.drawChessboardCorners(self.image.ndarray,
                                  patternSize=self.dimensions,
                                  corners=self.sp_corners, patternWasFound=1)

    @lazyproperty
    def area(self):
        """
        **SUMMARY**

        Returns the mean of the distance between corner points in the
        chessboard Given that the chessboard is of a known size, this can be
        used as a proxy for distance from the camera

        **RETURNS**

        Returns the mean distance between the corners.

        **EXAMPLE**

        >>> img = Image("corners.jpg")
        >>> feats = img.find_chessboard_corners()
        >>> print feats[-1].area

        """
        #note, copying this from barcode means we probably need a subclass of
        #feature called "quandrangle"
        sqform = spsd.squareform(spsd.pdist(self.points, "euclidean"))
        a = sqform[0][1]
        b = sqform[1][2]
        c = sqform[2][3]
        d = sqform[3][0]
        p = sqform[0][2]
        q = sqform[1][3]
        s = (a + b + c + d) / 2.0
        return 2 * sqrt(
            (s - a) * (s - b) * (s - c) * (s - d) - (a * c + b * d + p * q) *
            (a * c + b * d - p * q) / 4)

    @classmethod
    def find(cls, img, dimensions=(8, 5), subpixel=True):
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
        gray_array = img.gray_ndarray
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


######################################################################
@apply_plugins
class TemplateMatch(Feature):
    """
    **SUMMARY**

    This class is used for template (pattern) matching in images.
    The template matching cannot handle scale or rotation.

    """

    template_image = None
    quality = 0
    w = 0
    h = 0

    def __init__(self, image, template, location, quality):
        self.template_image = template  # -- KAT - TRYING SOMETHING
        self.image = image
        self.quality = quality
        w = template.width
        h = template.height
        at_x = location[0]
        at_y = location[1]
        points = [(at_x, at_y), (at_x + w, at_y), (at_x + w, at_y + h),
                  (at_x, at_y + h)]

        super(TemplateMatch, self).__init__(image, at_x, at_y, points)

    def _template_overlaps(self, other):
        """
        Returns true if this feature overlaps another template feature.
        """
        (maxx, minx, maxy, miny) = self.extents
        overlap = False
        for pnt in other.points:
            if maxx >= pnt[0] >= minx and maxy >= pnt[1] >= miny:
                overlap = True
                break

        return overlap

    def consume(self, other):
        """
        Given another template feature, make this feature the size of the two
        features combined.
        """
        (maxx, minx, maxy, miny) = self.extents
        (maxx0, minx0, maxy0, miny0) = other.extents

        maxx = max(maxx, maxx0)
        minx = min(minx, minx0)
        maxy = max(maxy, maxy0)
        miny = min(miny, miny0)
        self.x = minx
        self.y = miny
        self.points = [(minx, miny), (minx, maxy), (maxx, maxy), (maxx, miny)]
        force_update_lazyproperties(self)

    def rescale(self, width, height):
        """
        This method keeps the feature's center the same but sets a new width
        and height
        """
        (maxx, minx, maxy, miny) = self.extents
        xc = minx + ((maxx - minx) / 2)
        yc = miny + ((maxy - miny) / 2)
        x = xc - (width / 2)
        y = yc - (height / 2)
        self.x = x
        self.y = y
        self.points = [(x, y),
                       (x + width, y),
                       (x + width, y + height),
                       (x, y + height)]
        force_update_lazyproperties(self)

    def crop(self):
        (maxx, minx, maxy, miny) = self.extents
        return self.image.crop(minx, miny, maxx - minx, maxy - miny)

    def draw(self, color=Color.GREEN, width=1):
        """
        **SUMMARY**

        Draw the bounding rectangle, default color green.

        **PARAMETERS**

        * *color* - An RGB color triplet.
        * *width* - if width is less than zero we draw the feature filled in,
         otherwise we draw the get_contour using the specified width.

        **RETURNS**

        Nothing - this is an inplace operation that modifies the source images
        drawing layer.
        """
        self.image.dl().rectangle((self.x, self.y),
                                  (self.width, self.height),
                                  color=color, width=width)

    @classmethod
    def find(cls, img, template_image=None, threshold=5,
             method="SQR_DIFF_NORM", grayscale=True, rawmatches=False):
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
            img_array = img.gray_ndarray
            template_array = template_image.gray_ndarray
        else:
            img_array = img.ndarray
            template_array = template_image.ndarray

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
            fs.append(TemplateMatch(img, template_image, (location[1],
                                                          location[0]),
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

    @classmethod
    def find_once(cls, img, template_image=None, threshold=0.2,
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
            img_array = img.gray_ndarray
            template_array = template_image.gray_ndarray
        else:
            img_array = img.ndarray
            template_array = template_image.ndarray

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
                TemplateMatch(img, template_image, (location[1], location[0]),
                              matches[location[0], location[1]]))
        return fs


######################################################################
@apply_plugins
class Circle(Feature):
    """
    **SUMMARY**

    Class for a general circle feature with a center at (x,y) and a radius r

    """

    def __init__(self, i, at_x, at_y, r):
        self.r = r
        points = [(at_x - r, at_y - r), (at_x + r, at_y - r),
                  (at_x + r, at_y + r), (at_x - r, at_y + r)]
        super(Circle, self).__init__(i, at_x, at_y, points)
        segments = 18
        rng = range(1, segments + 1)
        self.contour = []
        for theta in rng:
            rp = 2.0 * pi * float(theta) / float(segments)
            x = (r * sin(rp)) + at_x
            y = (r * cos(rp)) + at_y
            self.contour.append((x, y))

    def draw(self, color=Color.GREEN, width=1):
        """
        **SUMMARY**

        With no dimension information, color the x,y point for the feature.

        **PARAMETERS**

        * *color* - An RGB color triplet.
        * *width* - if width is less than zero we draw the feature filled in,
         otherwise we draw the get_contour using the specified width.

        **RETURNS**

        Nothing - this is an inplace operation that modifies the source images
        drawing layer.

        """
        self.image.dl().circle((self.x, self.y), self.r, color, width)

    @lazyproperty
    def mean_color(self):
        """

        **SUMMARY**

        Returns the average color within the circle.

        **RETURNS**

        Returns an RGB triplet that corresponds to the mean color of the
        feature.

        **EXAMPLE**

        >>> img = Image("lenna")
        >>> c = img.find_circle()
        >>> c[-1].mean_color()

        """
        mask = self.image.get_empty(1)
        cv2.circle(mask, center=(self.x, self.y), radius=self.r,
                   color=(255, 255, 255), thickness=-1)
        temp = cv2.mean(self.image.ndarray, mask=mask)
        return temp[0], temp[1], temp[2]

    @property
    def area(self):
        """
        Area covered by the feature -- for a pixel, 1

        **SUMMARY**

        Returns a numpy array of the area of each feature in pixels.

        **RETURNS**

        A numpy array of all the positions in the featureset.

        **EXAMPLE**

        >>> img = Image("lenna")
        >>> feats = img.find_blobs()
        >>> xs = feats.coordinates()
        >>> print xs

        """
        return self.r * self.r * pi

    @property
    def perimeter(self):
        """
        **SUMMARY**

        Returns the get_perimeter of the circle feature in pixels.
        """
        return 2 * pi * self.r

    @property
    def width(self):
        """
        **SUMMARY**

        Returns the width of the feature -- for compliance just r*2

        """
        return self.r * 2

    @property
    def height(self):
        """
        **SUMMARY**

        Returns the height of the feature -- for compliance just r*2
        """
        return self.r * 2

    @property
    def radius(self):
        """
        **SUMMARY**

        Returns the radius of the circle in pixels.

        """
        return self.r

    @property
    def diameter(self):
        """
        **SUMMARY**

        Returns the diameter of the circle in pixels.

        """
        return self.r * 2

    def crop(self, no_mask=False):
        """
        **SUMMARY**

        This function returns the largest bounding box for an image.

        **PARAMETERS**

        * *no_mask* - if no_mask=True we return the bounding box image of the
         circle. if no_mask=False (default) we return the masked circle with
         the rest of the area set to black

        **RETURNS**

        The masked circle image.

        """
        if no_mask:
            return self.image.crop(self.x, self.y, self.width,
                                   self.height, centered=True)
        else:
            mask = self.image.get_empty()
            result = self.image.get_empty()

            # if you want to shave a bit of time we go do
            # the crop before the blit
            cv2.circle(mask, center=(self.x, self.y), radius=self.r,
                       color=(255, 255, 255), thickness=-1)
            np.where(mask, self.image.ndarray, result)
            ret_value = Factory.Image(result)
            ret_value = ret_value.crop(self.x, self.y, self.width,
                                       self.height, centered=True)
            return ret_value

    @classmethod
    def find(cls, img, canny=100, threshold=350, distance=-1):
        """
        **SUMMARY**

        Perform the Hough Circle transform to extract _perfect_ circles from
        the image canny - the upper bound on a canny edge detector used to find
        circle edges.

        **PARAMETERS**

        * *threshold* - the threshold at which to count a circle. Small parts of
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

        circs = cv2.HoughCircles(img.gray_ndarray,
                                 method=cv2.cv.CV_HOUGH_GRADIENT,
                                 dp=2, minDist=distance,
                                 param1=canny, param2=threshold)
        if circs is None:
            return None
        circle_fs = FeatureSet()
        for circ in circs[0]:
            circle_fs.append(Circle(img, int(circ[0]), int(circ[1]),
                                    int(circ[2])))
        return circle_fs


###############################################################################
@apply_plugins
class KeyPoint(Feature):
    """
    **SUMMARY**

    The class is place holder for SURF/SIFT/ORB/STAR keypoints.

    """
    r = 0.00
    descriptor = None
    key_point = None

    def __init__(self, i, keypoint, descriptor=None, flavor="SURF"):
        self.key_point = keypoint
        x = keypoint.pt[0]
        y = keypoint.pt[1]
        self._r = keypoint.size / 2.0
        self.image = i
        self.octave = keypoint.octave
        self.response = keypoint.response
        self.flavor = flavor
        self.descriptor = descriptor
        r = self._r
        points = ((x + r, y + r), (x + r, y - r),
                  (x - r, y - r), (x - r, y + r))
        super(KeyPoint, self).__init__(i, x, y, points)

        segments = 18
        rng = range(1, segments + 1)
        self.points = []
        for theta in rng:
            rp = 2.0 * pi * float(theta) / float(segments)
            x = (r * sin(rp)) + self.x
            y = (r * cos(rp)) + self.y
            self.points.append((x, y))

    @property
    def object(self):
        """
        **SUMMARY**

        Returns the raw keypoint object.

        """
        return self.key_point

    @property
    def quality(self):
        """
        **SUMMARY**

        Returns the quality metric for the keypoint object.

        """
        return self.response

    @property
    def angle(self):
        """
        **SUMMARY**

        Return the angle (theta) in degrees of the feature.
        The default is 0 (horizontal).

        **RETURNS**

        An angle value in degrees.

        """
        return self.key_point.angle

    def draw(self, color=Color.GREEN, width=1):
        """
        **SUMMARY**

        Draw a circle around the feature.  Color tuple is single parameter,
        default is Green.

        **PARAMETERS**

        * *color* - An RGB color triplet.
        * *width* - if width is less than zero we draw the feature filled in,
         otherwise we draw the get_contour using the specified width.


        **RETURNS**

        Nothing - this is an inplace operation that modifies the source images
        drawing layer.

        """
        self.image.dl().circle((self.x, self.y), self._r, color, width)
        pt1 = (int(self.x), int(self.y))
        pt2 = (int(self.x + (self.radius * sin(radians(self.angle)))),
               int(self.y + (self.radius * cos(radians(self.angle)))))
        self.image.dl().line(pt1, pt2, color, width)

    @lazyproperty
    def mean_color(self):
        """
        **SUMMARY**

        Return the average color within the feature's radius

        **RETURNS**

        Returns an  RGB triplet that corresponds to the mean color of the
        feature.

        **EXAMPLE**

        >>> img = Image("lenna")
        >>> kp = img.find_keypoints()
        >>> c = kp[0].mean_color

        """
        mask = self.image.get_empty(1)
        cv2.circle(mask, center=(int(self.x), int(self.y)),
                   radius=int(self._r), color=(255, 255, 255),
                   thickness=-1)
        temp = cv2.mean(self.image.ndarray, mask)
        return temp[0], temp[1], temp[2]

    def color_distance(self, color=(0, 0, 0)):
        """
        Return the euclidean color distance of the color tuple at x,y from
        a given color (default black)
        """
        return spsd.euclidean(np.array(color), np.array(self.mean_color))

    @property
    def perimeter(self):
        """
        **SUMMARY**

        Returns the get_perimeter of the circle feature in pixels.
        """
        return 2 * pi * self._r

    @property
    def width(self):
        """
        **SUMMARY**

        Returns the width of the feature -- for compliance just r*2

        """
        return self._r * 2

    @property
    def height(self):
        """
        **SUMMARY**

        Returns the height of the feature -- for compliance just r*2
        """
        return self._r * 2

    @property
    def radius(self):
        """
        **SUMMARY**

        Returns the radius of the circle in pixels.

        """
        return self._r

    @property
    def diameter(self):
        """
        **SUMMARY**

        Returns the diameter of the circle in pixels.

        """
        return self._r * 2

    def crop(self, no_mask=False):
        """
        **SUMMARY**

        This function returns the largest bounding box for an image.

        **PARAMETERS**

        * *no_mask* - if no_mask=True we return the bounding box image of the
         circle. if no_mask=False (default) we return the masked circle with
         the rest of the area set to black

        **RETURNS**

        The masked circle image.

        """
        if no_mask:
            return self.image.crop(self.x, self.y, self.width,
                                   self.height, centered=True)
        else:
            mask = self.image.get_empty()
            result = self.image.get_empty()

            # if you want to shave a bit of time we go do
            # the crop before the blit
            cv2.circle(mask, center=(int(self.x), int(self.y)),
                       radius=int(self._r), color=(255, 255, 255),
                       thickness=-1)
            np.where(mask, self.image.ndarray, result)
            ret_value = Factory.Image(source=result)
            ret_value = ret_value.crop(self.x, self.y, self.width,
                                       self.height, centered=True)
            return ret_value

    @classmethod
    def find(cls, img, min_quality=300.00, flavor="SURF", highquality=False):
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
        kp, d = img._get_raw_keypoints(threshold=min_quality,
                                       force_reset=True,
                                       flavor=flavor,
                                       highquality=int(highquality))

        if flavor in ["ORB", "SIFT", "SURF", "BRISK", "FREAK"] \
                and kp is not None and d is not None:
            for i in range(0, len(kp)):
                fs.append(KeyPoint(img, kp[i], d[i], flavor))
        elif flavor in ["FAST", "STAR", "MSER", "Dense"] and kp is not None:
            for i in range(0, len(kp)):
                fs.append(KeyPoint(img, kp[i], None, flavor))
        else:
            logger.warning("ImageClass.Keypoints: I don't know the method "
                           "you want to use")
            return None

        return fs


######################################################################
@apply_plugins
class Motion(Feature):
    """
    **SUMMARY**

    The motion feature is used to encapsulate optical flow vectors. The feature
    holds the length and direction of the vector.

    """

    def __init__(self, i, at_x, at_y, dx, dy, window):
        """
        i    - the source image.
        at_x - the sample x pixel position on the image.
        at_y - the sample y pixel position on the image.
        dx   - the x component of the optical flow vector.
        dy   - the y component of the optical flow vector.
        wndw - the size of the sample window (we assume it is square).
        """
        self.norm_dy = 0.00
        self.norm_dx = 0.00
        self.dx = dx  # the direction of the vector
        self.dy = dy
        self.window = window  # the size of the sample window
        sz = window / 2
        # so we center at the flow vector
        points = [(at_x + sz, at_y + sz), (at_x - sz, at_y + sz),
                  (at_x + sz, at_y + sz), (at_x + sz, at_y - sz)]
        super(Motion, self).__init__(i, at_x, at_y, points)

    def draw(self, color=Color.GREEN, width=1, normalize=True):
        """
        **SUMMARY**
        Draw the optical flow vector going from the sample point along the
        length of the motion vector.

        **PARAMETERS**

        * *color* - An RGB color triplet.
        * *width* - if width is less than zero we draw the feature filled in,
         otherwise we draw the get_contour using the specified width.
        * *normalize* - normalize the vector size to the size of the block
         (i.e. the biggest optical flow vector is scaled to the size of the
          block, all other vectors are scaled relative to the longest vector).

        **RETURNS**

        Nothing - this is an inplace operation that modifies the source images
        drawing layer.

        """
        if normalize:
            win = self.window / 2
            w = sqrt((win * win) * 2)
            new_x = (self.norm_dx * w) + self.x
            new_y = (self.norm_dy * w) + self.y
        else:
            new_x = self.x + self.dx
            new_y = self.y + self.dy

        self.image.dl().line((self.x, self.y), (new_x, new_y), color, width)

    def normalize_to(self, max_magnitude):
        """
        **SUMMARY**

        This helper method normalizes the vector give an input magnitude.
        This is helpful for keeping the flow vector inside the sample window.
        """
        if max_magnitude == 0:
            self.norm_dx = 0
            self.norm_dy = 0
            return None
        mag = self.magnitude
        new_mag = mag / max_magnitude
        unit = self.unit_vector
        self.norm_dx = unit[0] * new_mag
        self.norm_dy = unit[1] * new_mag

    @property
    def magnitude(self):
        """
        Returns the magnitude of the optical flow vector.
        """
        return sqrt((self.dx * self.dx) + (self.dy * self.dy))

    @property
    def unit_vector(self):
        """
        Returns the unit vector direction of the flow vector as an (x,y) tuple.
        """
        mag = self.magnitude
        if mag != 0.00:
            return float(self.dx) / mag, float(self.dy) / mag
        else:
            return 0.00, 0.00

    @property
    def vector(self):
        """
        Returns the raw direction vector as an (x,y) tuple.
        """
        return self.dx, self.dy

    @lazyproperty
    def mean_color(self):
        """
        Return the color tuple from x,y
        **SUMMARY**

        Return a numpy array of the average color of the area covered by each
        Feature.

        **RETURNS**

        Returns an array of RGB triplets the correspond to the mean color of
        the feature.

        **EXAMPLE**

        >>> img = Image("lenna")
        >>> kp = img.find_keypoints()
        >>> c = kp.mean_color()

        """
        x = int(self.x - (self.window / 2))
        y = int(self.y - (self.window / 2))
        return self.image.crop(x, y, int(self.window),
                               int(self.window)).mean_color()

    def crop(self):
        """
        This function returns the image in the sample window around the flow
        vector.

        Returns Image
        """
        x = int(self.x - (self.window / 2))
        y = int(self.y - (self.window / 2))

        return self.image.crop(x, y, int(self.window), int(self.window))

    @classmethod
    def find(cls, img, previous_frame, window=11, aggregate=True):
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

        flow = cv2.calcOpticalFlowFarneback(prev=previous_frame.gray_ndarray,
                                            next=img.gray_ndarray,
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


######################################################################
@apply_plugins
class KeypointMatch(Feature):
    """
    This class encapsulates a keypoint match between images of an object.
    It is used to record a template of one image as it appears in another image
    """

    def __init__(self, image, template, min_rect, _homography):
        self._template = template
        self._min_rect = min_rect
        self._homography = _homography
        xmax = 0
        ymax = 0
        xmin = image.width
        ymin = image.height
        for p in min_rect:
            if p[0] > xmax:
                xmax = p[0]
            if p[0] < xmin:
                xmin = p[0]
            if p[1] > ymax:
                ymax = p[1]
            if p[1] < ymin:
                ymin = p[1]

        width = (xmax - xmin)
        height = (ymax - ymin)
        at_x = xmin + (width / 2)
        at_y = ymin + (height / 2)
        points = [(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin)]
        super(KeypointMatch, self).__init__(image, at_x, at_y, points)

    def draw(self, color=Color.GREEN, width=1):
        """
        The default drawing operation is to draw the min bounding
        rectangle in an image.

        **SUMMARY**

        Draw a small circle around the corner.  Color tuple is single
        parameter, default is Red.

        **PARAMETERS**

        * *color* - An RGB color triplet.
        * *width* - if width is less than zero we draw the feature filled in,
         otherwise we draw the get_contour using the specified width.


        **RETURNS**

        Nothing - this is an inplace operation that modifies the source images
        drawing layer.


        """
        self.image.dl().line(self._min_rect[0],
                             self._min_rect[1], color, width)
        self.image.dl().line(self._min_rect[1],
                             self._min_rect[2], color, width)
        self.image.dl().line(self._min_rect[2],
                             self._min_rect[3], color, width)
        self.image.dl().line(self._min_rect[3],
                             self._min_rect[0], color, width)

    def draw_rect(self, color=Color.GREEN, width=1):
        """
        This method draws the axes alligned square box of the template
        match. This box holds the minimum bounding rectangle that describes
        the object. If the minimum bounding rectangle is axes aligned
        then the two bounding rectangles will match.
        """
        self.image.dl().line(self.points[0], self.points[1], color, width)
        self.image.dl().line(self.points[1], self.points[2], color, width)
        self.image.dl().line(self.points[2], self.points[3], color, width)
        self.image.dl().line(self.points[3], self.points[0], color, width)

    def crop(self):
        """
        Returns a cropped image of the feature match. This cropped version is
        the axes aligned box masked to just include the image data of the
        minimum bounding rectangle.
        """
        tlc = self.top_left_corner
        raw = self.image.crop(tlc[0], tlc[1], self.width,
                              self.height)  # crop the minbouding rect
        return raw

    @lazyproperty
    def mean_color(self):
        """
        return the average color within the circle
        **SUMMARY**

        Return a numpy array of the average color of the area covered by each
        Feature.

        **RETURNS**

        Returns an array of RGB triplets the correspond to the mean color of
        the feature.

        **EXAMPLE**

        >>> img = Image("lenna")
        >>> kp = img.find_keypoints()
        >>> c = kp.mean_color()

        """
        tlc = self.top_left_corner
        # crop the minbouding rect
        raw = self.image.crop(tlc[0], tlc[0],
                              self.width, self.height)
        mask = Factory.Image((raw.width, raw.height))
        mask.dl().polygon(self._min_rect, color=Color.WHITE,
                          filled=pickle.TRUE)
        mask = mask.apply_layers()
        return cv2.mean(raw.ndarray, mask.gray_ndarray)

    @property
    def min_rect(self):
        """
        Returns the minimum bounding rectangle of the feature as a list
        of (x,y) tuples.
        """
        return self._min_rect

    @property
    def homography(self):
        """
        Returns the _homography matrix used to calulate the minimum bounding
        rectangle.
        """
        return self._homography

    @classmethod
    def find(cls, img, template, quality=500.00, min_dist=0.2, min_match=0.4):
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


######################################################################
@apply_plugins
class ShapeContextDescriptor(Feature):
    """
    Create a shape context descriptor.
    """

    def __init__(self, image, point, descriptor, blob):
        self._descriptor = descriptor
        self._source_blob = blob
        x = point[0]
        y = point[1]
        points = [(x - 1, y - 1), (x + 1, y - 1), (x + 1, y + 1),
                  (x - 1, y + 1)]
        super(ShapeContextDescriptor, self).__init__(image, x, y, points)

    def draw(self, color=Color.GREEN, width=1):
        """
        The default drawing operation is to draw the min bounding
        rectangle in an image.

        **SUMMARY**

        Draw a small circle around the corner.  Color tuple is single
        parameter, default is Red.

        **PARAMETERS**

        * *color* - An RGB color triplet.
        * *width* - if width is less than zero we draw the feature filled in,
         otherwise we draw the get_contour using the specified width.


        **RETURNS**

        Nothing - this is an inplace operation that modifies the source images
        drawing layer.

        """
        radius = 3
        if(width > radius):
            radius = width + 1
        self.image.dl().circle(center=(int(self.x), int(self.y)), radius=radius,
                               color=color, width=width)


######################################################################
@apply_plugins
class ROI(Feature):
    """
    This class creates a region of interest that inherit from one
    or more features or no features at all.
    """
    x = 0  # the center x coordinate
    y = 0  # the center y coordinate
    w = 0
    h = 0
    xtl = 0  # top left x
    ytl = 0  # top left y
    # we are going to assume x,y,w,h is our canonical form
    points = []  # point list for cross compatibility
    image = None
    sub_features = []
    _mean_color = None

    def __init__(self, x, y=None, width=None, height=None, image=None):
        """
        **SUMMARY**

        This function can handle just about whatever you throw at it
        and makes a it into a feature. Valid input items are tuples and lists
        of x,y points, features, featuresets, two x,y points, and a
        set of x,y,width,height values.


        **PARAMETERS**

        * *x* - this can be just about anything, a list or tuple of x points,
        a corner of the image, a list of (x,y) points, a Feature, a FeatureSet
        * *y* - this is usually a second point or set of y values.
        * *width* - a width
        * *height* - a height.

        **RETURNS**

        Nothing.

        **EXAMPLE**

        >>> img = Image('lenna')
        >>> x, y = np.where(img.threshold(230).get_gray_numpy() > 128)
        >>> roi = ROI(zip(x, y), img)
        >>> roi2 = ROI(x, y, img)

        """
        #After forgetting to set img=Image I put this catch
        # in to save some debugging headache.
        if isinstance(y, Factory.Image):
            self.image = y
            y = None
        elif isinstance(width, Factory.Image):
            self.image = width
            width = None
        elif isinstance(height, Factory.Image):
            self.image = height
            height = None
        else:
            self.image = image

        if image is None and isinstance(x, (Feature, FeatureSet)):
            if isinstance(x, Feature):
                self.image = x.image
            if isinstance(x, FeatureSet) and len(x) > 0:
                self.image = x[0].image

        if isinstance(x, Feature):
            self.sub_features = FeatureSet([x])
        elif isinstance(x, (list, tuple)) and len(x) > 0 \
                and isinstance(x, Feature):
            self.sub_features = FeatureSet(x)

        result = self._standardize(x, y, width, height)
        if result is None:
            logger.warning("Could not create an ROI from your data.")
            return
        self._rebase(result)

    def resize(self, width, height=None, percentage=True):
        """
        **SUMMARY**

        Contract/Expand the roi. By default use a percentage, otherwise use
        pixels. This is all done relative to the center of the roi


        **PARAMETERS**

        * *width* - the percent to grow shrink the region is the only parameter,
         otherwise it is the new ROI width
        * *height* - The new roi height in terms of pixels or a percentage.
        * *percentage* - If true use percentages (e.g. 2 doubles the size),
         otherwise use pixel values.
        * *h* - a height.

        **RETURNS**

        Nothing.

        **EXAMPLE**

        >>> roi = ROI(10, 10, 100, 100, img)
        >>> roi.resize(2)
        >>> roi.show()

        """
        if height is None and isinstance(width, (tuple, list)):
            height = width[1]
            width = width[0]
        if percentage:
            if height is None:
                height = width
            nw = self.w * width
            nh = self.h * height
            nx = self.xtl + ((self.w - nw) / 2.0)
            ny = self.ytl + ((self.h - nh) / 2.0)
            self._rebase([nx, ny, nw, nh])
        else:
            nw = self.w + width
            nh = self.h + height
            nx = self.xtl + ((self.w - nw) / 2.0)
            ny = self.ytl + ((self.h - nh) / 2.0)
            self._rebase([nx, ny, nw, nh])

    def overlaps(self, other_roi):
        for pnt in other_roi.points:
            if self.max_x >= pnt[0] >= self.min_x \
                    and self.max_y >= pnt[1] >= self.min_y:
                return True
        return False

    def translate(self, x=0, y=0):
        """
        **SUMMARY**

        Move the roi.

        **PARAMETERS**

        * *x* - Move the ROI horizontally.
        * *y* - Move the ROI vertically

        **RETURNS**

        Nothing.

        **EXAMPLE**

        >>> roi = ROI(10, 10, 100, 100, img)
        >>> roi.translate(30, 30)
        >>> roi.show()

        """
        if x == 0 and y == 0:
            return

        if y == 0 and isinstance(x, (tuple, list)):
            y = x[1]
            x = x[0]

        if isinstance(x, (float, int)) and isinstance(y, (float, int)):
            self._rebase([self.xtl + x, self.ytl + y, self.w, self.h])

    def to_xywh(self):
        """
        **SUMMARY**

        Get the ROI as a list of the top left corner's x and y position
        and the roi's width and height in pixels.

        **RETURNS**

        A list of the form [x,y,w,h]

        **EXAMPLE**

        >>> roi = ROI(10, 10, 100, 100, img)
        >>> roi.translate(30, 30)
        >>> print roi.to_xywh()

        """
        return [self.xtl, self.ytl, self.w, self.h]

    def to_tl_and_br(self):
        """
        **SUMMARY**

        Get the ROI as a list of tuples of the ROI's top left
        corner and bottom right corner.

        **RETURNS**

        A list of the form [(x,y),(x,y)]

        **EXAMPLE**

        >>> roi = ROI(10, 10, 100, 100, img)
        >>> roi.translate(30, 30)
        >>> print roi.to_tl_and_br()

        """
        return [(self.xtl, self.ytl), (self.xtl + self.w, self.ytl + self.h)]

    def to_points(self):
        """
        **SUMMARY**

        Get the ROI as a list of four points that make up the bounding
        rectangle.


        **RETURNS**

        A list of the form [(x,y),(x,y),(x,y),(x,y)]

        **EXAMPLE**

        >>> roi = ROI(10, 10, 100, 100, img)
        >>> print roi.to_points()
        """

        tl = (self.xtl, self.ytl)
        tr = (self.xtl + self.w, self.ytl)
        br = (self.xtl + self.w, self.ytl + self.h)
        bl = (self.xtl, self.ytl + self.h)
        return [tl, tr, br, bl]

    def to_unit_xywh(self):
        """
        **SUMMARY**

        Get the ROI as a list, the values are top left x, to left y,
        width and height. These values are scaled to unit values with
        respect to the source image..


        **RETURNS**

        A list of the form [x,y,w,h]

        **EXAMPLE**

        >>> roi = ROI(10, 10, 100, 100, img)
        >>> print roi.to_unit_xywh()
        """
        if self.image is None:
            return None
        srcw = float(self.image.width)
        srch = float(self.image.height)
        x, y, w, h = self.to_xywh()
        nx = 0
        ny = 0
        if x != 0:
            nx = x / srcw
        if y != 0:
            ny = y / srch

        return [nx, ny, w / srcw, h / srch]

    def to_unit_tl_and_br(self):
        """
        **SUMMARY**

        Get the ROI as a list of tuples of the ROI's top left
        corner and bottom right corner. These coordinates are in unit
        length values with respect to the source image.

        **RETURNS**

        A list of the form [(x,y),(x,y)]

        **EXAMPLE**

        >>> roi = ROI(10, 10, 100, 100, img)
        >>> roi.translate(30, 30)
        >>> print roi.to_unit_tl_and_br()

        """

        if self.image is None:
            return None
        srcw = float(self.image.width)
        srch = float(self.image.height)
        x, y, w, h = self.to_xywh()
        nx = 0
        ny = 0
        nw = w / srcw
        nh = h / srch
        if x != 0:
            nx = x / srcw
        if y != 0:
            ny = y / srch

        return [(nx, ny), (nx + nw, ny + nh)]

    def to_unit_points(self):
        """
        **SUMMARY**

        Get the ROI as a list of four points that make up the bounding
        rectangle. Each point is represented in unit coordinates with respect
        to the source image.

        **RETURNS**

        A list of the form [(x,y),(x,y),(x,y),(x,y)]

        **EXAMPLE**

        >>> roi = ROI(10, 10, 100, 100, img)
        >>> print roi.to_unit_points()
        """

        if self.image is None:
            return None
        srcw = float(self.image.width)
        srch = float(self.image.height)
        ret_value = []
        for pnt in self.to_points():
            x, y = pnt
            if x != 0:
                x /= srcw
            if y != 0:
                y /= srch
            ret_value.append((x, y))
        return ret_value

    def coord_transform_x(self, x, intype="ROI", output="SRC"):
        """
        **SUMMARY**

        Transform a single or a set of x values from one reference frame to
        another.

        Options are:

        SRC - the coordinates of the source image.
        ROI - the coordinates of the ROI
        ROI_UNIT - unit coordinates in the frame of reference of the ROI
        SRC_UNIT - unit coordinates in the frame of reference of source image.

        **PARAMETERS**

        * *x* - A list of x values or a single x value.
        * *intype* - A string indicating the input format of the data.
        * *output* - A string indicating the output format of the data.

        **RETURNS**

        A list of the transformed values.


        **EXAMPLE**

        >>> img = Image('lenna')
        >>> blobs = img.find_blobs()
        >>> roi = ROI(blobs[0])
        >>> x = roi.crop()..... /find some x values in the crop region
        >>> xt = roi.coord_transform_x(x)
        >>> #xt are no in the space of the original image.
        """
        if self.image is None:
            logger.warning("No image to perform that calculation")
            return None
        if isinstance(x, (float, int)):
            x = [x]
        intype = intype.upper()
        output = output.upper()
        if intype == output:
            return x
        return self._transform(x, self.image.width, self.w, self.xtl, intype,
                               output)

    def coord_transform_y(self, y, intype="ROI", output="SRC"):
        """
        **SUMMARY**

        Transform a single or a set of y values from one reference frame to
        another.

        Options are:

        SRC - the coordinates of the source image.
        ROI - the coordinates of the ROI
        ROI_UNIT - unit coordinates in the frame of reference of the ROI
        SRC_UNIT - unit coordinates in the frame of reference of source image.

        **PARAMETERS**

        * *y* - A list of y values or a single y value.
        * *intype* - A string indicating the input format of the data.
        * *output* - A string indicating the output format of the data.

        **RETURNS**

        A list of the transformed values.


        **EXAMPLE**

        >>> img = Image('lenna')
        >>> blobs = img.find_blobs()
        >>> roi = ROI(blobs[0])
        >>> y = roi.crop()..... /find some y values in the crop region
        >>> yt = roi.coord_transform_y(y)
        >>> #yt are no in the space of the original image.
        """

        if self.image is None:
            logger.warning("No image to perform that calculation")
            return None
        if isinstance(y, (float, int)):
            y = [y]
        intype = intype.upper()
        output = output.upper()
        if intype == output:
            return y
        return self._transform(y, self.image.height, self.h, self.ytl, intype,
                               output)

    def coord_transform_points(self, points, input="ROI", output="SRC"):
        """
        **SUMMARY**

        Transform a set of (x,y) values from one reference frame to another.

        Options are:

        SRC - the coordinates of the source image.
        ROI - the coordinates of the ROI
        ROI_UNIT - unit coordinates in the frame of reference of the ROI
        SRC_UNIT - unit coordinates in the frame of reference of source image.

        **PARAMETERS**

        * *points* - A list of (x,y) values or a single (x,y) value.
        * *intype* - A string indicating the input format of the data.
        * *output* - A string indicating the output format of the data.

        **RETURNS**

        A list of the transformed values.


        **EXAMPLE**

        >>> img = Image('lenna')
        >>> blobs = img.find_blobs()
        >>> roi = ROI(blobs[0])
        >>> pts = roi.crop()..... /find some x, y values in the crop region
        >>> pts = roi.coord_transform_points(pts)
        >>> #yt are no in the space of the original image.
        """
        if self.image is None:
            logger.warning("No image to perform that calculation")
            return None
        if isinstance(points, tuple) and len(points) == 2:
            points = [points]
        input = input.upper()
        output = output.upper()
        x = [pt[0] for pt in points]
        y = [pt[1] for pt in points]

        if input == output:
            return points

        x = self._transform(x, self.image.width, self.w, self.xtl, input,
                            output)
        y = self._transform(y, self.image.height, self.h, self.ytl, input,
                            output)
        return zip(x, y)

    @staticmethod
    def _transform(x, image_size, roi_size, offset, input, output):
        # we are going to go to src unit coordinates
        # and then we'll go back.
        if input == "SRC":
            xtemp = [xt / float(image_size) for xt in x]
        elif input == "ROI":
            xtemp = [(xt + offset) / float(image_size) for xt in x]
        elif input == "ROI_UNIT":
            xtemp = [((xt * roi_size) + offset) / float(image_size) for xt in x]
        elif input == "SRC_UNIT":
            xtemp = x
        else:
            logger.warning("Bad Parameter to coord_transform_x")
            return None

        if output == "SRC":
            ret_value = [int(xt * image_size) for xt in xtemp]
        elif output == "ROI":
            ret_value = [int((xt * image_size) - offset) for xt in xtemp]
        elif output == "ROI_UNIT":
            ret_value = [int(((xt * image_size) - offset) / float(roi_size)) for xt in
                         xtemp]
        elif output == "SRC_UNIT":
            ret_value = xtemp
        else:
            logger.warning("Bad Parameter to coord_transform_x")
            return None

        return ret_value

    def split_x(self, x, unit_vals=False, src_vals=False):
        """
        **SUMMARY**
        Split the ROI at an x value.

        x can be a list of sequentianl tuples of x split points  e.g [0.3,0.6]
        where we assume the top and bottom are also on the list.
        Use units to split as a percentage (e.g. 30% down).
        The src_vals means use coordinates of the original image.


        **PARAMETERS**

        * *x*-The split point. Can be a single point or a list of points. the
         type is determined by the flags.
        * *unit_vals* - Use unit vals for the split point. E.g. 0.5 means split
         at 50% of the ROI.
        * *src_vals* - Use x values relative to the source image rather than
         relative to the ROI.


        **RETURNS**

        Returns a feature set of ROIs split from the source ROI.

        **EXAMPLE**

        >>> roi = ROI(0, 0, 100, 100, img)
        >>> splits = roi.split_x(50) # create two ROIs

        """
        ret_value = FeatureSet()
        if unit_vals and src_vals:
            logger.warning("Not sure how you would like to split the feature")
            return None

        if not isinstance(x, (list, tuple)):
            x = [x]

        if unit_vals:
            x = self.coord_transform_x(x, intype="ROI_UNIT", output="SRC")
        elif not src_vals:
            x = self.coord_transform_x(x, intype="ROI", output="SRC")

        for xt in x:
            if xt < self.xtl or xt > self.xtl + self.w:
                logger.warning("Invalid split point.")
                return None

        x.insert(0, self.xtl)
        x.append(self.xtl + self.w)
        for i in xrange(0, len(x) - 1):
            xstart = x[i]
            xstop = x[i + 1]
            w = xstop - xstart
            ret_value.append(ROI(x=xstart, y=self.ytl, width=w, height=self.h,
                                 image=self.image))
        return ret_value

    def split_y(self, y, unit_vals=False, src_vals=False):
        """
        **SUMMARY**
        Split the ROI at an x value.

        y can be a list of sequentianl tuples of y split points  e.g [0.3,0.6]
        where we assume the top and bottom are also on the list.
        Use units to split as a percentage (e.g. 30% down).
        The src_vals means use coordinates of the original image.


        **PARAMETERS**

        * *y*-The split point. Can be a single point or a list of points. the
         type is determined by the flags.
        * *unit_vals* - Use unit vals for the split point. E.g. 0.5 means split
         at 50% of the ROI.
        * *src_vals* - Use x values relative to the source image rather than
         relative to the ROI.

        **RETURNS**

        Returns a feature set of ROIs split from the source ROI.

        **EXAMPLE**

        >>> roi = ROI(0, 0, 100, 100, img)
        >>> splits = roi.split_y(50) # create two ROIs

        """
        ret_value = FeatureSet()
        if unit_vals and src_vals:
            logger.warning("Not sure how you would like to split the feature")
            return None

        if not isinstance(y, (list, tuple)):
            y = [y]

        if unit_vals:
            y = self.coord_transform_y(y, intype="ROI_UNIT", output="SRC")
        elif not src_vals:
            y = self.coord_transform_y(y, intype="ROI", output="SRC")

        for yt in y:
            if yt < self.ytl or yt > self.ytl + self.h:
                logger.warning("Invalid split point.")
                return None

        y.insert(0, self.ytl)
        y.append(self.ytl + self.h)
        for i in xrange(0, len(y) - 1):
            ystart = y[i]
            ystop = y[i + 1]
            h = ystop - ystart
            ret_value.append(ROI(x=self.xtl, y=ystart, width=self.w, height=h,
                                 image=self.image))
        return ret_value

    def merge(self, regions):
        """
        **SUMMARY**

        Combine another region, or regions with this ROI. Everything must be
        in the source image coordinates. Regions can be a ROIs, [ROI],
        features, FeatureSets, or anything that can be cajoled into a region.

        **PARAMETERS**

        * *regions* - A region or list of regions. Regions are just about
         anything that has position.


        **RETURNS**

        Nothing, but modifies this region.

        **EXAMPLE**

        >>>  blobs = img.find_blobs()
        >>>  roi = ROI(blobs[0])
        >>>  print roi.to_xywh()
        >>>  roi.merge(blobs[2])
        >>>  print roi.to_xywh()

        """
        result = self._standardize(regions)
        if result is not None:
            xo, yo, wo, ho = result
            x = np.min([xo, self.xtl])
            y = np.min([yo, self.ytl])
            w = np.max([self.xtl + self.w, xo + wo]) - x
            h = np.max([self.ytl + self.h, yo + ho]) - y
            if self.image is not None:
                x = np.clip(x, 0, self.image.width)
                y = np.clip(y, 0, self.image.height)
                w = np.clip(w, 0, self.image.width - x)
                h = np.clip(h, 0, self.image.height - y)
            self._rebase([x, y, w, h])
            if isinstance(regions, ROI):
                self.sub_features += regions.sub_features  # ROI is not iterable error
            elif isinstance(regions, Feature):
                self.sub_features.append(regions)
            elif isinstance(regions, (list, tuple)):
                if isinstance(regions[0], ROI):
                    for reg in regions:
                        self.sub_features += reg.sub_features
                elif isinstance(regions[0], Feature):
                    for reg in regions:
                        self.sub_features.append(reg)

    def rebase(self, x, y=None, width=None, height=None):
        """

        Completely alter roi using whatever source coordinates you wish.

        """
        if isinstance(x, Feature):
            self.sub_features.append(x)
        elif isinstance(x, (list, tuple)) and len[x] > 0 \
                and isinstance(x, Feature):
            self.sub_features += list(x)
        result = self._standardize(x, y, width, height)
        if result is None:
            logger.warning("Could not create an ROI from your data.")
            return
        self._rebase(result)

    def draw(self, color=Color.GREEN, width=3):
        """
        **SUMMARY**

        This method will draw the feature on the source image.

        **PARAMETERS**

        * *color* - The color as an RGB tuple to render the image.

        **RETURNS**

        Nothing.

        **EXAMPLE**

        >>> img = Image("RedDog2.jpg")
        >>> blobs = img.find_blobs()
        >>> blobs[-1].draw()
        >>> img.show()

        """
        x, y, w, h = self.to_xywh()
        self.image.draw_rectangle(x, y, w, h, width=width, color=color)

    def show(self, color=Color.GREEN, width=2):
        """
        **SUMMARY**

        This function will automatically draw the features on the image and
        show it.

        **RETURNS**

        Nothing.

        **EXAMPLE**

        >>> img = Image("logo")
        >>> feat = img.find_blobs()
        >>> feat[-1].show() #window pops up.

        """
        self.draw(color, width)
        self.image.show()

    @lazyproperty
    def mean_color(self):
        """
        **SUMMARY**

        Return the average color within the feature as a tuple.

        **RETURNS**

        An RGB color tuple.

        **EXAMPLE**

        >>> img = Image("OWS.jpg")
        >>> blobs = img.find_blobs(128)
        >>> for b in blobs:
        >>>    if b.mean_color() == Color.WHITE:
        >>>       print "Found a white thing"

        """
        x, y, w, h = self.to_xywh()
        return self.image.crop(x, y, w, h).mean_color()

    def _rebase(self, roi):
        x, y, w, h = roi
        self.xtl = x
        self.ytl = y
        self.w = w
        self.h = h
        self.points = [(x, y), (x + w, y), (x, y + h), (x + w, y + h)]
        #WE MAY WANT TO DO A SANITY CHECK HERE
        force_update_lazyproperties(self)

    def _standardize(self, x, y=None, width=None, height=None):
        if isinstance(x, np.ndarray):
            x = x.tolist()
        if isinstance(y, np.ndarray):
            y = y.tolist()

        # make the common case fast
        if isinstance(x, (int, float)) and isinstance(y, (int, float)) \
                and isinstance(width, (int, float)) \
                and isinstance(height, (int, float)):
            if self.image is not None:
                x = np.clip(x, 0, self.image.width)
                y = np.clip(y, 0, self.image.height)
                width = np.clip(width, 0, self.image.width - x)
                height = np.clip(height, 0, self.image.height - y)

                return [x, y, width, height]
        elif isinstance(x, ROI):
            x, y, width, height = x.to_xywh()
        #If it's a feature extract what we need
        elif isinstance(x, FeatureSet) and len(x) > 0:
            #double check that everything in the list is a feature
            features = [feat for feat in x if isinstance(feat, Feature)]
            xmax = np.max([feat.max_x for feat in features])
            xmin = np.min([feat.min_x for feat in features])
            ymax = np.max([feat.max_y for feat in features])
            ymin = np.min([feat.min_y for feat in features])
            x = xmin
            y = ymin
            width = xmax - xmin
            height = ymax - ymin

        elif isinstance(x, Feature):
            the_feature = x
            x = the_feature.points[0][0]
            y = the_feature.points[0][1]
            width = the_feature.width
            height = the_feature.height

        # [x,y,w,h] (x,y,w,h)
        elif isinstance(x, (tuple, list)) and len(x) == 4 \
                and isinstance(x[0], (int, long, float)) \
                and y is None and width is None and height is None:
            x, y, width, height = x
        # x of the form [(x,y),(x1,y1),(x2,y2),(x3,y3)]
        # x of the form [[x,y],[x1,y1],[x2,y2],[x3,y3]]
        # x of the form ([x,y],[x1,y1],[x2,y2],[x3,y3])
        # x of the form ((x,y),(x1,y1),(x2,y2),(x3,y3))
        elif isinstance(x, (list, tuple)) \
                and isinstance(x[0], (list, tuple)) \
                and (len(x) == 4 and len(x[0]) == 2) \
                and y is None and width is None and height is None:
            if len(x[0]) == 2 and len(x[1]) == 2 \
                    and len(x[2]) == 2 and len(x[3]) == 2:
                xmax = np.max([x[0][0], x[1][0], x[2][0], x[3][0]])
                ymax = np.max([x[0][1], x[1][1], x[2][1], x[3][1]])
                xmin = np.min([x[0][0], x[1][0], x[2][0], x[3][0]])
                ymin = np.min([x[0][1], x[1][1], x[2][1], x[3][1]])
                x = xmin
                y = ymin
                width = xmax - xmin
                height = ymax - ymin
            else:
                logger.warning(
                    "x should be in the form  ((x,y),(x1,y1),(x2,y2),(x3,y3))")
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
                width = xmax - xmin
                height = ymax - ymin
            else:
                logger.warning(
                    "x should be in the form x = [1,2,3,4,5] y =[0,2,4,6,8]")
                return None

        # x of the form [(x,y),(x,y),(x,y),(x,y),(x,y),(x,y)]
        elif isinstance(x, (list, tuple)) and len(x) > 4 \
                and len(x[0]) == 2 and y is None and width is None and height is None:
            if isinstance(x[0][0], (int, long, float)):
                xs = [pt[0] for pt in x]
                ys = [pt[1] for pt in x]
                xmax = np.max(xs)
                ymax = np.max(ys)
                xmin = np.min(xs)
                ymin = np.min(ys)
                x = xmin
                y = ymin
                width = xmax - xmin
                height = ymax - ymin
            else:
                logger.warning(
                    "x should be in the form "
                    "[(x,y),(x,y),(x,y),(x,y),(x,y),(x,y)]")
                return None

        # x of the form [(x,y),(x1,y1)]
        elif isinstance(x, (list, tuple)) and len(x) == 2 \
                and isinstance(x[0], (list, tuple)) \
                and isinstance(x[1], (list, tuple)) \
                and y is None and width is None and height is None:
            if len(x[0]) == 2 and len(x[1]) == 2:
                xt = np.min([x[0][0], x[1][0]])
                yt = np.min([x[0][1], x[1][1]])
                width = np.abs(x[0][0] - x[1][0])
                height = np.abs(x[0][1] - x[1][1])
                x = xt
                y = yt
            else:
                logger.warning("x should be in the form [(x1,y1),(x2,y2)]")
                return None

        # x and y of the form (x,y),(x1,y2)
        elif isinstance(x, (tuple, list)) \
                and isinstance(y, (tuple, list)) \
                and width is None and height is None:
            if len(x) == 2 and len(y) == 2:
                xt = np.min([x[0], y[0]])
                yt = np.min([x[1], y[1]])
                width = np.abs(y[0] - x[0])
                height = np.abs(y[1] - x[1])
                x = xt
                y = yt

            else:
                logger.warning(
                    "if x and y are tuple it should be in the form "
                    "(x1,y1) and (x2,y2)")
                return None

        if y is None or width is None or height is None:
            logger.warning('Not a valid roi')
            return None
        elif width <= 0 or height <= 0:
            logger.warning("ROI can't have a negative dimension")
            return None

        if self.image is not None:
            x = np.clip(x, 0, self.image.width)
            y = np.clip(y, 0, self.image.height)
            width = np.clip(width, 0, self.image.width - x)
            height = np.clip(height, 0, self.image.height - y)

        return [x, y, width, height]

    def crop(self):
        ret_value = None
        if self.image is not None:
            ret_value = self.image.crop(self.xtl, self.ytl, self.w, self.h)
        return ret_value
