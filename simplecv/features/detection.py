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

import numpy as np
import pickle
import scipy.spatial.distance as spsd

from simplecv.base import cv, logger
from simplecv.color import Color
from simplecv.features.features import Feature, FeatureSet
from simplecv.image_class import Image


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


######################################################################
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
        self.vector = None
        self.y_intercept = None
        self.end_points = copy(line)

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

    def length(self):
        """

        **SUMMARY**

        This method returns the length of the line.

        **RETURNS**

        A floating point length value.

        **EXAMPLE**

        >>> img = Image("OWS.jpg")
        >>> lines = img.find_lines
        >>> for l in lines:
        >>>    if l.length() > 100:
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
        tlc = self.top_left_corner()
        return self.image.crop(tlc[0], tlc[1], self.get_width(), 
                               self.get_height())

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
        >>> c = l[0].mean_color()

        """
        (pt1, _) = self.end_points
        #we're going to walk the line, and take the mean color from all the px
        #points -- there's probably a much more optimal way to do this
        (maxx, minx, maxy, miny) = self.get_extents()

        d_x = maxx - minx
        d_y = maxy - miny
        #orient the line so it is going in the positive direction

        #if it's a straight one, we can just get mean color on the slice
        if d_x == 0.0:
            return self.image[pt1[0]:pt1[0] + 1, miny:maxy].mean_color()
        if d_y == 0.0:
            return self.image[minx:maxx, pt1[1]:pt1[1] + 1].mean_color()

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
            if img[pixel[0], pixel[1]] == (255.0, 255.0, 255.0):
                matched_pixels.append(pixel)
        matched_pixels.sort()

        return matched_pixels

    def get_angle(self):
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

    def get_vector(self):
        # this should be a lazy property
        if self.vector is None:
            self.vector = [
                float(self.end_points[1][0] - self.end_points[0][0]),
                float(self.end_points[1][1] - self.end_points[0][1])]
        return self.vector

    def dot(self, other):
        return np.dot(self.get_vector(), other.get_vector())

    def cross(self, other):
        return np.cross(self.get_vector(), other.get_vector())

    def get_y_intercept(self):
        """
        **SUMMARY**

        Returns the y intercept based on the lines equation.  Note that this
        point is potentially not contained in the image itself

        **RETURNS**

        Returns a floating point intersection value

        **EXAMPLE**

        >>> img = Image("lenna")
        >>> l = Line(img, ((50, 150), (2, 225))
        >>> b = l.get_y_intercept()
        """
        if self.y_intercept is None:
            pt1, _ = self.end_points
            #y = mx + b | b = y-mx
            self.y_intercept = pt1[1] - self.slope * pt1[0]
        return self.y_intercept

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

        return Line(self.image, ep)


######################################################################
class Barcode(Feature):
    """
    **SUMMARY**

    The Barcode Feature wrappers the object returned by find_barcode(), a zbar
    symbol

    * The x,y coordinate is the center of the code.
    * points represents the four boundary points of the feature.  Note: for QR
     codes, these points are the reference rectangls, and are quadrangular,
     rather than rectangular with other datamatrix types.
    * data is the parsed data of the code.

    **SEE ALSO**

    :py:meth:`ImageClass.findBarcodes()`
    """
    data = ""

    #given a ZXing bar
    def __init__(self, i, zbsymbol):
        self.image = i

        locs = zbsymbol.location
        if len(locs) > 4:
            xs = [l[0] for l in locs]
            ys = [l[1] for l in locs]
            xmax = np.max(xs)
            xmin = np.min(xs)
            ymax = np.max(ys)
            ymin = np.min(ys)
            points = ((xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin))
        else:
            points = copy(locs)  # hopefully this is in tl clockwise order

        super(Barcode, self).__init__(i, 0, 0, points)
        self.data = zbsymbol.data
        self.points = copy(points)
        numpoints = len(self.points)
        self.x = 0
        self.y = 0

        for pnt in self.points:
            self.x += pnt[0]
            self.y += pnt[1]

        if numpoints:
            self.x /= numpoints
            self.y /= numpoints

    def __repr__(self):
        return "%s.%s at (%d,%d), read data: %s" % (
            self.__class__.__module__, self.__class__.__name__, self.x, self.y,
            self.data)

    def draw(self, color=(255, 0, 0), width=1):
        """

        **SUMMARY**

        Draws the bounding area of the barcode, given by points.  Note that for
        QR codes, these points are the reference boxes, and so may "stray" into
        the actual code.


        **PARAMETERS**

        * *color* - An RGB color triplet.
        * *width* - if width is less than zero we draw the feature filled in,
         otherwise we draw the get_contour using the specified width.


        **RETURNS**

        Nothing - this is an inplace operation that modifies the source images
        drawing layer.


        """
        self.image.draw_line(self.points[0], self.points[1], color, width)
        self.image.draw_line(self.points[1], self.points[2], color, width)
        self.image.draw_line(self.points[2], self.points[3], color, width)
        self.image.draw_line(self.points[3], self.points[0], color, width)

    def length(self):
        """
        **SUMMARY**

        Returns the longest side of the quandrangle formed by the boundary
        points.

        **RETURNS**

        A floating point length value.

        **EXAMPLE**

        >>> img = Image("mycode.jpg")
        >>> bc = img.find_barcode()
        >>> print bc[-1].length()

        """
        sqform = spsd.squareform(spsd.pdist(self.points, "euclidean"))
        #get pairwise distances for all points
        #note that the code is a quadrilateral
        return max(sqform[0][1], sqform[1][2], sqform[2][3], sqform[3][0])

    def get_area(self):
        """
        **SUMMARY**

        Returns the area defined by the quandrangle formed by the boundary
        points


        **RETURNS**

        An integer area value.

        **EXAMPLE**

        >>> img = Image("mycode.jpg")
        >>> bc = img.find_barcode()
        >>> print bc[-1].get_area()


        """
        #calc the length of each side in a square distance matrix
        sqform = spsd.squareform(spsd.pdist(self.points, "euclidean"))

        #squareform returns a N by N matrix
        #boundry line lengths
        a = sqform[0][1]
        b = sqform[1][2]
        c = sqform[2][3]
        d = sqform[3][0]

        #diagonals
        p = sqform[0][2]
        q = sqform[1][3]

        #get_perimeter / 2
        s = (a + b + c + d) / 2.0

        #i found the formula to do this on wikihow.  Yes, I am that lame.
        #http://www.wikihow.com/Find-the-Area-of-a-Quadrilateral
        return sqrt(
            (s - a) * (s - b) * (s - c) * (s - d) - (a * c + b * d + p * q) *
            (a * c + b * d - p * q) / 4)


######################################################################
class HaarFeature(Feature):
    """
    **SUMMARY**

    The HaarFeature is a rectangle returned by the FindHaarFeature() function.

    * The x,y coordinates are defined by the center of the bounding rectangle.
    * The classifier property refers to the cascade file used for detection .
    * Points are the clockwise points of the bounding rectangle, starting in
     upper left.

    """
    classifier = ""
    _width = ""
    _height = ""
    neighbors = ''
    feature_name = 'None'

    def __init__(self, i, haarobject, haarclassifier=None, cv2flag=True):
        self.image = i
        if not cv2flag:
            ((x, y, width, height), self.neighbors) = haarobject
        else:
            (x, y, width, height) = haarobject
        at_x = x + width / 2
        at_y = y + height / 2  # set location of feature to middle of rectangle
        points = (
            (x, y), (x + width, y), (x + width, y + height), (x, y + height))

        #set bounding points of the rectangle
        self.classifier = haarclassifier
        if haarclassifier is not None:
            self.feature_name = haarclassifier.get_name()

        super(HaarFeature, self).__init__(i, at_x, at_y, points)

    def draw(self, color=(0, 255, 0), width=1):
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
        self.image.draw_line(self.points[0], self.points[1], color, width)
        self.image.draw_line(self.points[1], self.points[2], color, width)
        self.image.draw_line(self.points[2], self.points[3], color, width)
        self.image.draw_line(self.points[3], self.points[0], color, width)

    def __getstate__(self):
        sdict = self.__dict__.copy()
        if 'classifier' in sdict:
            del sdict["classifier"]
        return sdict

    def mean_color(self):
        """
        **SUMMARY**

        Find the mean color of the boundary rectangle.

        **RETURNS**

        Returns an  RGB triplet that corresponds to the mean color of the
        feature.

        **EXAMPLE**

        >>> img = Image("lenna")
        >>> face = HaarCascade("face.xml")
        >>> faces = img.find_haar_features(face)
        >>> print faces[-1].mean_color()

        """
        crop = self.image[self.points[0][0]:self.points[1][0],
                          self.points[0][1]:self.points[2][1]]
        return crop.mean_color()

    def get_area(self):
        """
        **SUMMARY**

        Returns the area of the feature in pixels.

        **RETURNS**

        The area of the feature in pixels.

        **EXAMPLE**

        >>> img = Image("lenna")
        >>> face = HaarCascade("face.xml")
        >>> faces = img.find_haar_features(face)
        >>> print faces[-1].get_area()

        """
        return self.get_width() * self.get_height()


######################################################################
class Chessboard(Feature):
    """
    **SUMMARY**

    This class is used for Calibration, it uses a chessboard
    to calibrate from pixels to real world measurements.
    """
    sp_corners = []
    dimensions = ()

    def __init__(self, i, dim, subpixel_corners):
        self.dimensions = dim
        self.sp_corners = subpixel_corners
        at_x = np.average(np.array(self.sp_corners)[:, 0])
        at_y = np.average(np.array(self.sp_corners)[:, 1])

        posdiagsorted = sorted(self.sp_corners,
                               key=lambda corner: corner[0] + corner[1])
        #sort corners along the x + y axis
        negdiagsorted = sorted(self.sp_corners,
                               key=lambda corner: corner[0] - corner[1])
        #sort corners along the x - y axis

        points = (posdiagsorted[0], negdiagsorted[-1], posdiagsorted[-1],
                  negdiagsorted[0])
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
        cv.DrawChessboardCorners(self.image.get_bitmap(), self.dimensions,
                                 self.sp_corners, 1)

    def get_area(self):
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
        >>> print feats[-1].get_area()

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


######################################################################
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
        (maxx, minx, maxy, miny) = self.get_extents()
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
        (maxx, minx, maxy, miny) = self.get_extents()
        (maxx0, minx0, maxy0, miny0) = other.get_extents()

        maxx = max(maxx, maxx0)
        minx = min(minx, minx0)
        maxy = max(maxy, maxy0)
        miny = min(miny, miny0)
        self.x = minx
        self.y = miny
        self.points = [(minx, miny), (minx, maxy), (maxx, maxy), (maxx, miny)]
        self._update_extents()

    def rescale(self, w, h):
        """
        This method keeps the feature's center the same but sets a new width
        and height
        """
        (maxx, minx, maxy, miny) = self.get_extents()
        xc = minx + ((maxx - minx) / 2)
        yc = miny + ((maxy - miny) / 2)
        x = xc - (w / 2)
        y = yc - (h / 2)
        self.x = x
        self.y = y
        self.points = [(x, y),
                       (x + w, y),
                       (x + w, y + h),
                       (x, y + h)]
        self._update_extents()

    def crop(self):
        (maxx, minx, maxy, miny) = self.get_extents()
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
                                  (self.get_width(), self.get_height()),
                                  color=color, width=width)


######################################################################
class Circle(Feature):
    """
    **SUMMARY**

    Class for a general circle feature with a center at (x,y) and a radius r

    """
    x = 0.00
    y = 0.00
    r = 0.00
    image = ""  # parent image
    points = []
    avg_color = None

    def __init__(self, i, at_x, at_y, r):
        self.r = r
        self.avg_color = None
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

    def show(self, color=Color.GREEN):
        """
        **SUMMARY**

        This function will automatically draw the features on the image and
        show it. It is a basically a shortcut function for development and is
        the same as:

        **PARAMETERS**

        * *color* - the color of the feature as an rgb triplet.

        **RETURNS**

        Nothing - this is an inplace operation that modifies the source images
        drawing layer.

        **EXAMPLE**

        >>> img = Image("logo")
        >>> feat = img.find_circle()
        >>> feat[0].show()

        """
        self.draw(color)
        self.image.show()

    def distance_from(self, point=(-1, -1)):
        """
        **SUMMARY**

        Given a point (default to center of the image), return the euclidean
        distance of x,y from this point.

        **PARAMETERS**

        * *point* - The point, as an (x,y) tuple on the image to measure
         distance from.

        **RETURNS**

        The distance as a floating point value in pixels.

        **EXAMPLE**

        >>> img = Image("OWS.jpg")
        >>> blobs = img.find_circle()
        >>> blobs[-1].distance_from(blobs[-2].coordinates())

        """
        if point[0] == -1 or point[1] == -1:
            point = np.array(self.image.size()) / 2
        return spsd.euclidean(point, [self.x, self.y])

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
        #generate the mask
        if self.avg_color is None:
            mask = self.image.get_empty(1)
            cv.Zero(mask)
            cv.Circle(mask, (self.x, self.y), self.r, color=(255, 255, 255),
                      thickness=-1)
            temp = cv.Avg(self.image.get_bitmap(), mask)
            self.avg_color = (temp[0], temp[1], temp[2])
        return self.avg_color

    def get_area(self):
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

    def get_perimeter(self):
        """
        **SUMMARY**

        Returns the get_perimeter of the circle feature in pixels.
        """
        return 2 * pi * self.r

    def get_width(self):
        """
        **SUMMARY**

        Returns the width of the feature -- for compliance just r*2

        """
        return self.r * 2

    def get_height(self):
        """
        **SUMMARY**

        Returns the height of the feature -- for compliance just r*2
        """
        return self.r * 2

    def radius(self):
        """
        **SUMMARY**

        Returns the radius of the circle in pixels.

        """
        return self.r

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
         circle. if no_mask=False (default) we return the masked circle with the
         rest of the area set to black

        **RETURNS**

        The masked circle image.

        """
        if no_mask:
            return self.image.crop(self.x, self.y, self.get_width(),
                                   self.get_height(), centered=True)
        else:
            mask = self.image.get_empty(1)
            result = self.image.get_empty()
            cv.Zero(mask)
            cv.Zero(result)
            # if you want to shave a bit of time we go do
            # the crop before the blit
            cv.Circle(mask, (self.x, self.y), self.r, color=(255, 255, 255),
                      thickness=-1)
            cv.Copy(self.image.get_bitmap(), result, mask)
            ret_value = Image(result)
            ret_value = ret_value.crop(self.x, self.y, self.get_width(),
                                       self.get_height(), centered=True)
            return ret_value


###############################################################################
class KeyPoint(Feature):
    """
    **SUMMARY**

    The class is place holder for SURF/SIFT/ORB/STAR keypoints.

    """
    x = 0.00
    y = 0.00
    r = 0.00
    image = ""  # parent image
    points = []
    __avgColor = None
    angle = 0
    octave = 0
    response = 0.00
    flavor = ""
    descriptor = None
    key_point = None

    def __init__(self, i, keypoint, descriptor=None, flavor="SURF"):
        self.key_point = keypoint
        x = keypoint.pt[1]  # KAT
        y = keypoint.pt[0]
        self._r = keypoint.size / 2.0
        self._avgColor = None
        self.image = i
        self.angle = keypoint.angle
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

    def get_object(self):
        """
        **SUMMARY**

        Returns the raw keypoint object.

        """
        return self.key_point

    def get_descriptor(self):
        """
        **SUMMARY**

        Returns the raw keypoint descriptor.

        """
        return self.descriptor

    def quality(self):
        """
        **SUMMARY**

        Returns the quality metric for the keypoint object.

        """
        return self.response

    def get_octave(self):
        """
        **SUMMARY**

        Returns the raw keypoint's octave (if it has one).

        """
        return self.octave

    def get_flavor(self):
        """
        **SUMMARY**

        Returns the type of keypoint as a string (e.g. SURF/MSER/ETC)

        """
        return self.flavor

    def get_angle(self):
        """
        **SUMMARY**

        Return the angle (theta) in degrees of the feature.
        The default is 0 (horizontal).

        **RETURNS**

        An angle value in degrees.

        """
        return self.angle

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
        pt2 = (int(self.x + (self.radius() * sin(radians(self.get_angle())))),
               int(self.y + (self.radius() * cos(radians(self.get_angle())))))
        self.image.dl().line(pt1, pt2, color, width)

    def show(self, color=Color.GREEN):
        """
        **SUMMARY**

        This function will automatically draw the features on the image and
        show it. It is a basically a shortcut function for development and is
        the same as:

        >>> img = Image("logo")
        >>> feat = img.find_blobs()
        >>> if feat: feat.draw()
        >>> img.show()

        """
        self.draw(color)
        self.image.show()

    def distance_from(self, point=(-1, -1)):
        """
        **SUMMARY**

        Given a point (default to center of the image), return the euclidean
        distance of x,y from this point
        """
        if point[0] == -1 or point[1] == -1:
            point = np.array(self.image.size()) / 2
        return spsd.euclidean(point, [self.x, self.y])

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
        >>> c = kp[0].mean_color()

        """
        #generate the mask
        if self._avgColor is None:
            mask = self.image.get_empty(1)
            cv.Zero(mask)
            cv.Circle(mask, (int(self.x), int(self.y)), int(self._r),
                      color=(255, 255, 255), thickness=-1)
            temp = cv.Avg(self.image.get_bitmap(), mask)
            self._avgColor = (temp[0], temp[1], temp[2])
        return self._avgColor

    def color_distance(self, color=(0, 0, 0)):
        """
        Return the euclidean color distance of the color tuple at x,y from
        a given color (default black)
        """
        return spsd.euclidean(np.array(color), np.array(self.mean_color()))

    def get_perimeter(self):
        """
        **SUMMARY**

        Returns the get_perimeter of the circle feature in pixels.
        """
        return 2 * pi * self._r

    def get_width(self):
        """
        **SUMMARY**

        Returns the width of the feature -- for compliance just r*2

        """
        return self._r * 2

    def get_height(self):
        """
        **SUMMARY**

        Returns the height of the feature -- for compliance just r*2
        """
        return self._r * 2

    def radius(self):
        """
        **SUMMARY**

        Returns the radius of the circle in pixels.

        """
        return self._r

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
            return self.image.crop(self.x, self.y, self.get_width(),
                                   self.get_height(), centered=True)
        else:
            mask = self.image.get_empty(1)
            result = self.image.get_empty()
            cv.Zero(mask)
            cv.Zero(result)
            # if you want to shave a bit of time we go do
            # the crop before the blit
            cv.Circle(mask, (int(self.x), int(self.y)), int(self._r),
                      color=(255, 255, 255), thickness=-1)
            cv.Copy(self.image.get_bitmap(), result, mask)
            ret_value = Image(result)
            ret_value = ret_value.crop(self.x, self.y, self.get_width(),
                                       self.get_height(), centered=True)
            return ret_value


######################################################################
class Motion(Feature):
    """
    **SUMMARY**

    The motion feature is used to encapsulate optical flow vectors. The feature
    holds the length and direction of the vector.

    """
    x = 0.0
    y = 0.0
    image = ""  # parent image
    points = []
    dx = 0.00
    dy = 0.00
    norm_dy = 0.00
    norm_dx = 0.00
    window = 7

    def __init__(self, i, at_x, at_y, dx, dy, wndw):
        """
        i    - the source image.
        at_x - the sample x pixel position on the image.
        at_y - the sample y pixel position on the image.
        dx   - the x component of the optical flow vector.
        dy   - the y component of the optical flow vector.
        wndw - the size of the sample window (we assume it is square).
        """
        self.dx = dx  # the direction of the vector
        self.dy = dy
        self.window = wndw  # the size of the sample window
        sz = wndw / 2
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

    def normalize_to(self, max_mag):
        """
        **SUMMARY**

        This helper method normalizes the vector give an input magnitude.
        This is helpful for keeping the flow vector inside the sample window.
        """
        if max_mag == 0:
            self.norm_dx = 0
            self.norm_dy = 0
            return None
        mag = self.magnitude()
        new_mag = mag / max_mag
        unit = self.unit_vector()
        self.norm_dx = unit[0] * new_mag
        self.norm_dy = unit[1] * new_mag

    def magnitude(self):
        """
        Returns the magnitude of the optical flow vector.
        """
        return sqrt((self.dx * self.dx) + (self.dy * self.dy))

    def unit_vector(self):
        """
        Returns the unit vector direction of the flow vector as an (x,y) tuple.
        """
        mag = self.magnitude()
        if mag != 0.00:
            return float(self.dx) / mag, float(self.dy) / mag
        else:
            return 0.00, 0.00

    def vector(self):
        """
        Returns the raw direction vector as an (x,y) tuple.
        """
        return self.dx, self.dy

    def window_sz(self):
        """
        Return the window size that we sampled over.
        """
        return self.window

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


######################################################################
class KeypointMatch(Feature):
    """
    This class encapsulates a keypoint match between images of an object.
    It is used to record a template of one image as it appears in another image
    """
    x = 0.00
    y = 0.00
    image = ""  # parent image
    points = []
    _min_rect = []
    _avg_color = None
    _homography = []
    _template = None

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
        #self.x = at_x
        #self.y = at_y
        points = [(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin)]
        #self._update_extents()
        #self.image = image
        #points =
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
        self.image.dl().line(self._min_rect[0], self._min_rect[1], color, width)
        self.image.dl().line(self._min_rect[1], self._min_rect[2], color, width)
        self.image.dl().line(self._min_rect[2], self._min_rect[3], color, width)
        self.image.dl().line(self._min_rect[3], self._min_rect[0], color, width)

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
        tlc = self.top_left_corner()
        raw = self.image.crop(tlc[0], tlc[1], self.get_width(),
                              self.get_height())  # crop the minbouding rect
        return raw

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
        if self._avg_color is None:
            tlc = self.top_left_corner()
            raw = self.image.crop(tlc[0], tlc[0], self.get_width(),
                                  self.get_height())  # crop the minbouding rect
            mask = Image((self.get_width(), self.get_height()))
            mask.dl().polygon(self._min_rect, color=Color.WHITE,
                              filled=pickle.TRUE)
            mask = mask.apply_layers()
            ret_value = cv.Avg(raw.get_bitmap(), mask._get_grayscale_bitmap())
            self._avg_color = ret_value
        else:
            ret_value = self._avg_color
        return ret_value

    def get_min_rect(self):
        """
        Returns the minimum bounding rectangle of the feature as a list
        of (x,y) tuples.
        """
        return self._min_rect

    def get_homography(self):
        """
        Returns the _homography matrix used to calulate the minimum bounding
        rectangle.
        """
        return self._homography


######################################################################
class ShapeContextDescriptor(Feature):
    """
    Create a shape context descriptor.
    """
    x = 0.00
    y = 0.00
    image = ""  # parent image
    points = []
    _min_rect = []
    _avg_color = None
    _descriptor = None
    _source_blob = None

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
        self.image.dl().circle((self.x, self.y), 3, color, width)


######################################################################
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

    def __init__(self, x, y=None, w=None, h=None, image=None):
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
        * *w* - a width
        * *h* - a height.

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
        if isinstance(y, Image):
            self.image = y
            y = None
        elif isinstance(w, Image):
            self.image = w
            w = None
        elif isinstance(h, Image):
            self.image = h
            h = None
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

        result = self._standardize(x, y, w, h)
        if result is None:
            logger.warning("Could not create an ROI from your data.")
            return
        self._rebase(result)

    def resize(self, w, h=None, percentage=True):
        """
        **SUMMARY**

        Contract/Expand the roi. By default use a percentage, otherwise use
        pixels. This is all done relative to the center of the roi


        **PARAMETERS**

        * *w* - the percent to grow shrink the region is the only parameter,
         otherwise it is the new ROI width
        * *h* - The new roi height in terms of pixels or a percentage.
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
        if h is None and isinstance(w, (tuple, list)):
            h = w[1]
            w = w[0]
        if percentage:
            if h is None:
                h = w
            nw = self.w * w
            nh = self.h * h
            nx = self.xtl + ((self.w - nw) / 2.0)
            ny = self.ytl + ((self.h - nh) / 2.0)
            self._rebase([nx, ny, nw, nh])
        else:
            nw = self.w + w
            nh = self.h + h
            nx = self.xtl + ((self.w - nw) / 2.0)
            ny = self.ytl + ((self.h - nh) / 2.0)
            self._rebase([nx, ny, nw, nh])

    def overlaps(self, other_roi):
        for pnt in other_roi.points:
            if self.get_max_x() >= pnt[0] >= self.get_min_x() \
                    and self.get_max_y() >= pnt[1] >= self.get_min_y():
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

    def coord_transform_pts(self, pts, intype="ROI", output="SRC"):
        """
        **SUMMARY**

        Transform a set of (x,y) values from one reference frame to another.

        Options are:

        SRC - the coordinates of the source image.
        ROI - the coordinates of the ROI
        ROI_UNIT - unit coordinates in the frame of reference of the ROI
        SRC_UNIT - unit coordinates in the frame of reference of source image.

        **PARAMETERS**

        * *pts* - A list of (x,y) values or a single (x,y) value.
        * *intype* - A string indicating the input format of the data.
        * *output* - A string indicating the output format of the data.

        **RETURNS**

        A list of the transformed values.


        **EXAMPLE**

        >>> img = Image('lenna')
        >>> blobs = img.find_blobs()
        >>> roi = ROI(blobs[0])
        >>> pts = roi.crop()..... /find some x, y values in the crop region
        >>> pts = roi.coord_transform_pts(pts)
        >>> #yt are no in the space of the original image.
        """
        if self.image is None:
            logger.warning("No image to perform that calculation")
            return None
        if isinstance(pts, tuple) and len(pts) == 2:
            pts = [pts]
        intype = intype.upper()
        output = output.upper()
        x = [pt[0] for pt in pts]
        y = [pt[1] for pt in pts]

        if intype == output:
            return pts

        x = self._transform(x, self.image.width, self.w, self.xtl, intype,
                            output)
        y = self._transform(y, self.image.height, self.h, self.ytl, intype,
                            output)
        return zip(x, y)

    @staticmethod
    def _transform(x, imgsz, roisz, offset, intype, output):
        # we are going to go to src unit coordinates
        # and then we'll go back.
        if intype == "SRC":
            xtemp = [xt / float(imgsz) for xt in x]
        elif intype == "ROI":
            xtemp = [(xt + offset) / float(imgsz) for xt in x]
        elif intype == "ROI_UNIT":
            xtemp = [((xt * roisz) + offset) / float(imgsz) for xt in x]
        elif intype == "SRC_UNIT":
            xtemp = x
        else:
            logger.warning("Bad Parameter to coord_transform_x")
            return None

        if output == "SRC":
            ret_value = [int(xt * imgsz) for xt in xtemp]
        elif output == "ROI":
            ret_value = [int((xt * imgsz) - offset) for xt in xtemp]
        elif output == "ROI_UNIT":
            ret_value = [int(((xt * imgsz) - offset) / float(roisz)) for xt in
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
            ret_value.append(ROI(xstart, self.ytl, w, self.h, self.image))
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
            ret_value.append(ROI(self.xtl, ystart, self.w, h, self.image))
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
                self.sub_features += regions
            elif isinstance(regions, Feature):
                self.sub_features.append(regions)
            elif isinstance(regions, (list, tuple)):
                if isinstance(regions[0], ROI):
                    for reg in regions:
                        self.sub_features += reg.sub_features
                elif isinstance(regions[0], Feature):
                    for reg in regions:
                        self.sub_features.append(reg)

    def rebase(self, x, y=None, w=None, h=None):
        """

        Completely alter roi using whatever source coordinates you wish.

        """
        if isinstance(x, Feature):
            self.sub_features.append(x)
        elif isinstance(x, (list, tuple)) and len[x] > 0 \
                and isinstance(x, Feature):
            self.sub_features += list(x)
        result = self._standardize(x, y, w, h)
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
        self._max_x = None
        self._max_y = None
        self._min_x = None
        self._min_y = None
        self._width = None
        self._height = None
        self.extents = None
        self.bounding_box = None
        self.xtl = x
        self.ytl = y
        self.w = w
        self.h = h
        self.points = [(x, y), (x + w, y), (x, y + h), (x + w, y + h)]
        #WE MAY WANT TO DO A SANITY CHECK HERE
        self._update_extents()

    def _standardize(self, x, y=None, w=None, h=None):
        if isinstance(x, np.ndarray):
            x = x.tolist()
        if isinstance(y, np.ndarray):
            y = y.tolist()

        # make the common case fast
        if isinstance(x, (int, float)) and isinstance(y, (int, float)) \
                and isinstance(w, (int, float)) \
                and isinstance(h, (int, float)):
            if self.image is not None:
                x = np.clip(x, 0, self.image.width)
                y = np.clip(y, 0, self.image.height)
                w = np.clip(w, 0, self.image.width - x)
                h = np.clip(h, 0, self.image.height - y)

                return [x, y, w, h]
        elif isinstance(x, ROI):
            x, y, w, h = x.to_xywh()
        #If it's a feature extract what we need
        elif isinstance(x, FeatureSet) and len(x) > 0:
            #double check that everything in the list is a feature
            features = [feat for feat in x if isinstance(feat, Feature)]
            xmax = np.max([feat.get_max_x() for feat in features])
            xmin = np.min([feat.get_min_x() for feat in features])
            ymax = np.max([feat.get_max_y() for feat in features])
            ymin = np.min([feat.get_min_y() for feat in features])
            x = xmin
            y = ymin
            w = xmax - xmin
            h = ymax - ymin

        elif isinstance(x, Feature):
            the_feature = x
            x = the_feature.points[0][0]
            y = the_feature.points[0][1]
            w = the_feature.get_width()
            h = the_feature.get_height()

        # [x,y,w,h] (x,y,w,h)
        elif isinstance(x, (tuple, list)) and len(x) == 4 \
                and isinstance(x[0], (int, long, float)) \
                and y is None and w is None and h is None:
            x, y, w, h = x
        # x of the form [(x,y),(x1,y1),(x2,y2),(x3,y3)]
        # x of the form [[x,y],[x1,y1],[x2,y2],[x3,y3]]
        # x of the form ([x,y],[x1,y1],[x2,y2],[x3,y3])
        # x of the form ((x,y),(x1,y1),(x2,y2),(x3,y3))
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
                w = xmax - xmin
                h = ymax - ymin
            else:
                logger.warning(
                    "x should be in the form x = [1,2,3,4,5] y =[0,2,4,6,8]")
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
                logger.warning(
                    "x should be in the form "
                    "[(x,y),(x,y),(x,y),(x,y),(x,y),(x,y)]")
                return None

        # x of the form [(x,y),(x1,y1)]
        elif isinstance(x, (list, tuple)) and len(x) == 2 \
                and isinstance(x[0], (list, tuple)) \
                and isinstance(x[1], (list, tuple)) \
                and y is None and w is None and h is None:
            if len(x[0]) == 2 and len(x[1]) == 2:
                xt = np.min([x[0][0], x[1][0]])
                yt = np.min([x[0][0], x[1][0]])
                w = np.abs(x[0][0] - x[1][0])
                h = np.abs(x[0][1] - x[1][1])
                x = xt
                y = yt
            else:
                logger.warning("x should be in the form [(x1,y1),(x2,y2)]")
                return None

        # x and y of the form (x,y),(x1,y2)
        elif isinstance(x, (tuple, list)) \
                and isinstance(y, (tuple, list)) \
                and w is None and h is None:
            if len(x) == 2 and len(y) == 2:
                xt = np.min([x[0], y[0]])
                yt = np.min([x[1], y[1]])
                w = np.abs(y[0] - x[0])
                h = np.abs(y[1] - x[1])
                x = xt
                y = yt

            else:
                logger.warning(
                    "if x and y are tuple it should be in the form "
                    "(x1,y1) and (x2,y2)")
                return None

        if y is None or w is None or h is None:
            logger.warning('Not a valid roi')
        elif w <= 0 or h <= 0:
            logger.warning("ROI can't have a negative dimension")
            return None

        if self.image is not None:
            x = np.clip(x, 0, self.image.width)
            y = np.clip(y, 0, self.image.height)
            w = np.clip(w, 0, self.image.width - x)
            h = np.clip(h, 0, self.image.height - y)

        return [x, y, w, h]

    def crop(self):
        ret_value = None
        if self.image is not None:
            ret_value = self.image.crop(self.xtl, self.ytl, self.w, self.h)
        return ret_value
