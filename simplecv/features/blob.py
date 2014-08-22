import re
import warnings
from math import sin, cos, pi

import cv2
import numpy as np
import scipy.spatial.distance as spsd
import scipy.stats as sps

from simplecv.base import LazyProperty, logger
from simplecv.color import Color
from simplecv.core.pluginsystem import apply_plugins
from simplecv.factory import Factory
from simplecv.features.blobmaker import BlobMaker
from simplecv.features.detection import Line, Corner
from simplecv.features.detection import ShapeContextDescriptor
from simplecv.features.features import Feature, FeatureSet


@apply_plugins
class Blob(Feature):
    """
    **SUMMARY**

    A blob is a typicall a cluster of pixels that form a feature or unique
    shape that allows it to be distinguished from the rest of the image
    Blobs typically are computed very quickly so they are used often to
    find various items in a picture based on properties.  Typically these
    things like color, shape, size, etc.   Since blobs are computed quickly
    they are typically used to narrow down search regions in an image, where
    you quickly find a blob and then that blobs region is used for more
    computational intensive type image processing.

    **EXAMPLE**

    >>> img = Image("lenna")
    >>> blobs = img.find_blobs()
    >>> blobs[-1].draw()
    >>> img.show()

    **SEE ALSO**
    :py:meth:`find_blobs`
    :py:class:`BlobMaker`
    :py:meth:`find_blobs_from_mask`

    """
    seq = ''  # the cvseq object that defines this blob
    contour = []  # the blob's outer get_perimeter as a set of (x,y) tuples
    convex_hull = []  # the convex hull get_contour as a set of (x,y) tuples
    min_rectangle = []  # the smallest box rotated to fit the blob
    # min_rectangle[0] = centroid (x,y)
    # min_rectangle[1] = (w,h)
    # min_rectangle[2] = angle

    #bounding_box = [] #get W/H and X/Y from this
    hu = []  # The seven Hu Moments
    perimeter = 0  # the length of the get_perimeter in pixels
    area = 0  # the area in pixels
    m00 = 0
    m01 = 0
    m10 = 0
    m11 = 0
    m20 = 0
    m02 = 0
    m21 = 0
    m12 = 0
    contour_appx = None
    label = ""  # A user label
    label_color = []  # what color to draw the label
    avg_color = []  # The average color of the blob's area.
    #img =  '' #Image()# the segmented image of the blob
    #hull_img = '' # Image() the image from the hull.
    #mask = '' #Image()# A mask of the blob area
    # A mask of the hull area ... we may want to use this for the image mask.
    #xmHullMask = '' #Image()
    hole_contour = []  # list of hole contours
    #mVertEdgeHist = [] #vertical edge histogram
    #mHortEdgeHist = [] #horizontal edge histgram
    pickle_skip_properties = {'img', 'hull_img', 'mask', 'hull_mask'}

    def __init__(self):
        self._scdescriptors = None
        self._complete_contour = None
        self.contour = []
        self.convex_hull = []
        self.min_rectangle = [-1, -1, -1, -1, -1]  # angle from this
        self.contour_appx = []
        self.hu = [-1, -1, -1, -1, -1, -1, -1]
        self.perimeter = 0
        self.area = 0
        self.m00 = 0
        self.m01 = 0
        self.m10 = 0
        self.m11 = 0
        self.m20 = 0
        self.m02 = 0
        self.m21 = 0
        self.m12 = 0
        self.label = "UNASSIGNED"
        self.label_color = []
        self.avg_color = [-1, -1, -1]
        self.image = None
        self.hole_contour = []
        self.points = []

    def __getstate__(self):
        newdict = {}
        for key, value in self.__dict__.items():
            if key not in Blob.pickle_skip_properties:
                newdict[key] = value
        return newdict

    def __setstate__(self, mydict):
        iplkeys = []
        for key in mydict:
            if re.search("__string", key):
                iplkeys.append(key)
            else:
                self.__dict__[key] = mydict[key]

        #once we get all the metadata loaded, go for the bitmaps
        for key in iplkeys:
            realkey = key[:-len("__string")]
            self.__dict__[realkey] = mydict[key]

    @property
    def aspect_ratio(self):
        """
        **SUMMARY**

        Return the aspect ratio of the blob

        **RETURNS**

        A single floating point value of the aspect ration.

        """
        return self.min_rectangle[1][0] / self.min_rectangle[1][1]

    def get_perimeter(self):
        """
        **SUMMARY**

        This function returns the get_perimeter as an integer number of pixel
        lengths.

        **RETURNS**

        Integer

        **EXAMPLE**

        >>> img = Image("lenna")
        >>> blobs = img.find_blobs()
        >>> print blobs[-1].get_get_perimeter()

        """

        return self.perimeter

    def hull(self):
        """
        **SUMMARY**

        This function returns the convex hull points as a list of x,y tuples.

        **RETURNS**

        A list of x,y tuples.

        **EXAMPLE**

        >>> img = Image("lenna")
        >>> blobs = img.find_blobs()
        >>> print blobs[-1].hull()

        """
        return self.convex_hull

    def get_contour(self):
        """
        **SUMMARY**

        This function returns the get_contour points as a list of x,y tuples.

        **RETURNS**

        A list of x,y tuples.

        **EXAMPLE**

        >>> img = Image("lenna")
        >>> blobs = img.find_blobs()
        >>> print blobs[-1].get_get_contour()

        """

        return self.contour

    def mean_color(self):
        """
        **SUMMARY**

        This function returns a tuple representing the average color of the
        blob.

        **RETURNS**

        A RGB triplet of the average blob colors.

        **EXAMPLE**

        >>> img = Image("lenna")
        >>> blobs = img.find_blobs()
        >>> print blobs[-1].mean_color()

        """
        box_img = self.image.crop(*self.bounding_box)
        return box_img.mean_color()

    def min_rect(self):
        """
        Returns the corners for the smallest rotated rectangle to enclose the
        blob. The points are returned as a list of (x,y) tuples.
        """
        #if( self.min_rectangle[1][0] < self.min_rectangle[1][1]):
        ang = self.min_rectangle[2]
        #else:
        #    ang =  90 + self.min_rectangle[2]
        ang = 2 * pi * (float(ang) / 360.00)
        tx = self.min_rect_x()
        ty = self.min_rect_y()
        w = self.min_rect_width() / 2.0
        h = self.min_rect_height() / 2.0
        derp = np.matrix(
            [[cos(ang), -1 * sin(ang), tx], [sin(ang), cos(ang), ty],
             [0, 0, 1]])
        # [ cos a , -sin a, tx ]
        # [ sin a , cos a , ty ]
        # [ 0     , 0     ,  1 ]
        tl = np.matrix(
            [-1.0 * w, h, 1.0])  # Kat gladly supports homo. coordinates.
        tr = np.matrix([w, h, 1.0])
        bl = np.matrix([-1.0 * w, -1.0 * h, 1.0])
        br = np.matrix([w, -1.0 * h, 1.0])
        tlp = derp * tl.transpose()
        trp = derp * tr.transpose()
        blp = derp * bl.transpose()
        brp = derp * br.transpose()
        return ((float(tlp[0, 0]), float(tlp[1, 0])),
                (float(trp[0, 0]), float(trp[1, 0])),
                (float(blp[0, 0]), float(blp[1, 0])),
                (float(brp[0, 0]), float(brp[1, 0])))

    def draw_rect(self, layer=None, color=Color.DEFAULT, width=1, alpha=128):
        """
        **SUMMARY**

        Draws the bounding rectangle for the blob.

        **PARAMETERS**

        * *color* - The color to render the blob's box.
        * *alpha* - The alpha value of the rendered blob 0 = transparent,
         255 = opaque.
        * *width* - The width of the drawn blob in pixels
        * *layer* - if layer is not None, the blob is rendered to the layer
         versus the source image.

        **RETURNS**

        Returns None, this operation works on the supplied layer or the source
        image.


        **EXAMPLE**

        >>> img = Image("lenna")
        >>> blobs = img.find_blobs()
        >>> blobs[-1].draw_rect(color=Color.RED, width=-1, alpha=128)
        >>> img.show()

        """
        if layer is None:
            layer = self.image.dl()

        if width < 1:
            layer.rectangle(self.top_left_corner,
                            (self.width, self.height), color,
                            width, filled=True, alpha=alpha)
        else:
            layer.rectangle(self.top_left_corner,
                            (self.width, self.height), color,
                            width, filled=False, alpha=alpha)

    def draw_min_rect(self, layer=None, color=Color.DEFAULT,
                      width=1, alpha=128):
        """
        **SUMMARY**

        Draws the minimum bounding rectangle for the blob. The minimum bounding
        rectangle is the smallest rotated rectangle that can enclose the blob.

        **PARAMETERS**

        * *color* - The color to render the blob's box.
        * *alpha* - The alpha value of the rendered blob 0 = transparent,
         255 = opaque.
        * *width* - The width of the drawn blob in pixels
        * *layer* - If layer is not None, the blob is rendered to the layer
        versus the source image.

        **RETURNS**

        Returns none, this operation works on the supplied layer or the source
        image.

        **EXAMPLE**

        >>> img = Image("lenna")
        >>> blobs = img.find_blobs()
        >>> for b in blobs:
        >>>      b.draw_min_rect(color=Color.RED, width=-1, alpha=128)
        >>> img.show()

        """
        if layer is None:
            layer = self.image.dl()
        (tl, tr, bl, br) = self.min_rect()
        layer.line(tl, tr, color, width=width, alpha=alpha, antialias=False)
        layer.line(bl, br, color, width=width, alpha=alpha, antialias=False)
        layer.line(tl, bl, color, width=width, alpha=alpha, antialias=False)
        layer.line(tr, br, color, width=width, alpha=alpha, antialias=False)

    def get_angle(self):
        """
        **SUMMARY**

        This method returns the angle between the horizontal and the minimum
        enclosing rectangle of the blob. The minimum enclosing rectangle IS NOT
        the bounding box. Use the bounding box for situations where you need
        only an approximation of the objects dimensions. The minimum enclosing
        rectangle is slightly harder to maninpulate but gives much better
        information about the blobs dimensions.

        **RETURNS**

        Returns the angle between the minimum bounding rectangle and the
        horizontal.

        **EXAMPLE**

        >>> img = Image("lenna")
        >>> blobs = img.find_blobs()
        >>> blobs[-1].get_angle()

        """
        #return self.min_rectangle[2]+90.00
        ret_val = self.min_rectangle[2] + 90.00
        if self.min_rectangle[1][0] >= self.min_rectangle[1][1]:
            ret_val += 90.00

        if ret_val > 90.00:
            ret_val -= 180.00
        return ret_val

    def min_rect_x(self):
        """
        **SUMMARY**

        This is the x coordinate of the centroid for the minimum bounding
        rectangle

        **RETURNS**

        An integer that is the x position of the centrod of the minimum
        bounding rectangle.

        **EXAMPLE**

        >>> img = Image("lenna")
        >>> blobs = img.find_blobs()
        >>> print blobs[-1].min_rect_x()

        """
        return self.min_rectangle[0][0]

    def min_rect_y(self):
        """
        **SUMMARY**

        This is the y coordinate of the centroid for the minimum bounding
        rectangle

        **RETURNS**

        An integer that is the y position of the centrod of the minimum
        bounding rectangle.

        **EXAMPLE**

        >>> img = Image("lenna")
        >>> blobs = img.find_blobs()
        >>> print blobs[-1].min_rect_y()

        """
        return self.min_rectangle[0][1]

    def min_rect_width(self):
        """
        **SUMMARY**

        This is the width of the minimum bounding rectangle for the blob.

        **RETURNS**

        An integer that is the width of the minimum bounding rectangle for this
        blob.

        **EXAMPLE**

        >>> img = Image("lenna")
        >>> blobs = img.find_blobs()
        >>> print blobs[-1].min_rect_width()

        """
        return self.min_rectangle[1][0]

    def min_rect_height(self):
        """
        **SUMMARY**

        This is the height, in pixels, of the minimum bounding rectangle.

        **RETURNS**

        An integer that is the height of the minimum bounding rectangle for
        this blob.

        **EXAMPLE**

        >>> img = Image("lenna")
        >>> blobs = img.find_blobs()
        >>> print blobs[-1].min_rect_height()


        """
        return self.min_rectangle[1][1]

    def rectify_major_axis(self, axis=0):
        """
        **SUMMARY**

        Rectify the blob image and the get_contour such that the major
        axis is aligned to either horizontal=0 or vertical=1. This is to say,
        we take the blob, find the longest axis, and rotate the blob such that
        the axis is either vertical or horizontal.

        **PARAMETERS**

        * *axis* - if axis is zero we rotate the blobs to fit along the
         vertical axis, otherwise we use the horizontal axis.

        **RETURNS**

        This method works in place, i.e. it rotates the blob's internal data
        structures. This method is experimental. Use at your own risk.

        **EXAMPLE**

        >>> img = Image("lenna")
        >>> blobs = img.find_blobs()
        >>> blobs[-2].img.show()
        >>> blobs[-2].rectifyToMajorAxis(1)
        >>> blobs[-2].img.show()

        """
        final_rotation = self.get_angle()
        if self.min_rect_width() > self.min_rect_height():
            final_rotation = final_rotation

        if axis > 0:
            final_rotation -= 90

        self.rotate(final_rotation)

    def rotate(self, angle):
        """
        **SUMMARY**

        Rotate the blob given an  angle in degrees. If you use this method
        most of the blob elements will
        be rotated in place , however, this will "break" drawing back to the
        original image. To draw the blob create a new layer and draw to that
        layer. Positive rotations are counter clockwise.

        **PARAMETERS**

        * *angle* - A floating point angle in degrees. Positive is
        anti-clockwise.

        **RETURNS**

        .. Warning:
          Nothing. All rotations are performed in place. This modifies the
          blob's data and will break any image write back capabilities.

        **EXAMPLE**

        >>> img = Image("lenna")
        >>> blobs = img.find_blobs()
        >>> blobs[-2].img.show()
        >>> blobs[-2].rotate(90)
        >>> blobs[-2].img.show()

        """
        #FIXME: This function should return a blob
        theta = 2 * np.pi * (angle / 360.0)
        mode = ""
        point = (self.x, self.y)
        self.image = self.image.rotate(angle, mode, point)
        self.hull_img = self.hull_img.rotate(angle, mode, point)
        self.mask = self.mask.rotate(angle, mode, point)
        self.hull_mask = self.hull_mask.rotate(angle, mode, point)

        self.contour = map(lambda x:
                           (x[0] * np.cos(theta) - x[1] * np.sin(theta),
                            x[0] * np.sin(theta) + x[1] * np.cos(theta)),
                           self.contour)
        self.convex_hull = map(lambda x:
                               (x[0] * np.cos(theta) - x[1] * np.sin(theta),
                                x[0] * np.sin(theta) + x[1] * np.cos(theta)),
                               self.convex_hull)

        if self.hole_contour is not None:
            for h in self.hole_contour:
                h = map(lambda x:
                        (x[0] * np.cos(theta) - x[1] * np.sin(theta),
                         x[0] * np.sin(theta) + x[1] * np.cos(theta)),
                        h)

    def draw_appx(self, color=Color.HOTPINK, width=-1, alpha=-1, layer=None):
        if self.contour_appx is None or len(self.contour_appx) == 0:
            return

        if not layer:
            layer = self.image.dl()

        filled = width < 1
        layer.polygon(points=self.contour_appx, color=color,
                      width=width, filled=filled, antialias=True, alpha=alpha)

    def draw(self, color=Color.GREEN, width=-1, alpha=-1, layer=None):
        """
        **SUMMARY**

        Draw the blob, in the given color, to the appropriate layer

        By default, this draws the entire blob filled in, with holes.  If you
        provide a width, an outline of the exterior and interior contours is
        drawn.

        **PARAMETERS**

        * *color* -The color to render the blob as a color tuple.
        * *alpha* - The alpha value of the rendered blob 0=transparent,
         255=opaque.
        * *width* - The width of the drawn blob in pixels, if -1 then filled
         then the polygon is filled.
        * *layer* - A source layer, if layer is not None, the blob is rendered
         to the layer versus the source image.

        **RETURNS**

        This method either works on the original source image, or on the
        drawing layer provided. The method does not modify object itself.

        **EXAMPLE**

        >>> img = Image("lenna")
        >>> blobs = img.find_blobs()
        >>> blobs[-2].draw(color=Color.PUCE, width=-1, alpha=128)
        >>> img.show()

        """
        if not layer:
            layer = self.image.dl()

        if width == -1:
            # copy the mask into 3 channels and
            # multiply by the appropriate color
            gs_bitmap = self.mask.gray_ndarray
            maskred = cv2.convertScaleAbs(gs_bitmap, alpha=color[0] / 255.0)
            maskgrn = cv2.convertScaleAbs(gs_bitmap, alpha=color[1] / 255.0)
            maskblu = cv2.convertScaleAbs(gs_bitmap, alpha=color[2] / 255.0)
            maskbit = np.dstack((maskblu, maskgrn, maskred))

            masksurface = Factory.Image(maskbit).get_pg_surface()
            masksurface.set_colorkey(Color.BLACK)
            if alpha != -1:
                masksurface.set_alpha(alpha)
            layer.surface.blit(masksurface, self.top_left_corner)
        else:
            self.draw_outline(color, alpha, width, layer)
            self.draw_holes(color, alpha, width, layer)

    def draw_outline(self, color=Color.GREEN, alpha=255, width=1, layer=None):
        """
        **SUMMARY**

        Draw the blob get_contour the provided layer -- if no layer is
        provided, draw to the source image.


        **PARAMETERS**

        * *color* - The color to render the blob.
        * *alpha* - The alpha value of the rendered poly.
        * *width* - The width of the drawn blob in pixels, -1 then the polygon
         is filled.
        * *layer* - if layer is not None, the blob is rendered to the layer
         versus the source image.


        **RETURNS**

        This method either works on the original source image, or on the
        drawing layer provided. The method does not modify object itself.

        **EXAMPLE**

        >>> img = Image("lenna")
        >>> blobs = img.find_blobs()
        >>> blobs[-2].draw_outline(color=Color.GREEN, width=3, alpha=128)
        >>> img.show()


        """

        if layer is None:
            layer = self.image.dl()

        if width < 0:
            #blit the blob in
            layer.polygon(self.contour, color, filled=True, alpha=alpha)
        else:
            lastp = self.contour[0]  # this may work better.... than the other
            for nextp in self.contour[1:]:
                layer.line(lastp, nextp, color, width=width, alpha=alpha,
                           antialias=False)
                lastp = nextp
            layer.line(self.contour[0], self.contour[-1], color, width=width,
                       alpha=alpha, antialias=False)

    def draw_holes(self, color=Color.GREEN, alpha=-1, width=-1, layer=None):
        """
        **SUMMARY**

        This method renders all of the holes (if any) that are present in
        the blob.

        **PARAMETERS**

        * *color* - The color to render the blob's holes.
        * *alpha* - The alpha value of the rendered blob hole.
        * *width* - The width of the drawn blob hole in pixels, if w=-1 then
         the polygon is filled.
        * *layer* - If layer is not None, the blob is rendered to the layer
         versus the source image.

        **RETURNS**

        This method either works on the original source image, or on the
        drawing layer provided. The method does not modify object itself.

        **EXAMPLE**

        >>> img = Image("lenna")
        >>> blobs = img.find_blobs(128)
        >>> blobs[-1].draw_holes(color=Color.GREEN, width=3, alpha=128)
        >>> img.show()

        """
        if self.hole_contour is None:
            return
        if layer is None:
            layer = self.image.dl()

        if width < 0:
            #blit the blob in
            for h in self.hole_contour:
                layer.polygon(h, color, filled=True, alpha=alpha)
        else:
            for h in self.hole_contour:
                lastp = h[0]  # this may work better.... than the other
                for nextp in h[1:]:
                    layer.line((int(lastp[0]), int(lastp[1])),
                               (int(nextp[0]), int(nextp[1])), color,
                               width=width, alpha=alpha, antialias=False)
                    lastp = nextp
                layer.line(h[0], h[-1], color, width=width, alpha=alpha,
                           antialias=False)

    def draw_hull(self, color=Color.GREEN, alpha=-1, width=-1, layer=None):
        """
        **SUMMARY**

        Draw the blob's convex hull to either the source image or to the
        specified layer given by layer.

        **PARAMETERS**

        * *color* - The color to render the blob's convex hull as an RGB
         triplet.
        * *alpha* - The alpha value of the rendered blob.
        * *width* - The width of the drawn blob in pixels, if w=-1 then the
         polygon is filled.
        * *layer* - if layer is not None, the blob is rendered to the layer
         versus the source image.

        **RETURNS**

        This method either works on the original source image, or on the
        drawing layer provided. The method does not modify object itself.

        **EXAMPLE**

        >>> img = Image("lenna")
        >>> blobs = img.find_blobs(128)
        >>> blobs[-1].draw_hull(color=Color.GREEN, width=3, alpha=128)
        >>> img.show()

        """
        if layer is None:
            layer = self.image.dl()

        if width < 0:
            #blit the blob in
            layer.polygon(self.convex_hull, color, filled=True, alpha=alpha)
        else:
            # this may work better.... than the other
            lastp = self.convex_hull[0]
            for nextp in self.convex_hull[1::]:
                layer.line(lastp, nextp, color, width=width, alpha=alpha,
                           antialias=False)
                lastp = nextp
            layer.line(self.convex_hull[0], self.convex_hull[-1], color,
                       width=width, alpha=alpha, antialias=False)

    #draw the actual pixels inside the get_contour to the layer
    def draw_mask_to_layer(self, layer=None, offset=(0, 0)):
        """
        **SUMMARY**

        Draw the actual pixels of the blob to another layer. This is handy if
        you want to examine just the pixels inside the get_contour.

        **PARAMETERS**

        * *layer* - A drawing layer upon which to apply the mask.
        * *offset* -  The offset from the top left corner where we want to
         place the mask.

        **RETURNS**

        This method either works on the original source image, or on the
        drawing layer provided. The method does not modify object itself.

        **EXAMPLE**

        >>> img = Image("lenna")
        >>> blobs = img.find_blobs(128)
        >>> dl = DrawingLayer((img.width, img.height))
        >>> blobs[-1].draw_mask_to_layer(layer=dl)
        >>> dl.show()

        """
        if layer is not None:
            layer = self.image.dl()

        layer.blit(self.image, coordinates=offset)

    def is_square(self, tolerance=0.05, ratiotolerance=0.05):
        """
        **SUMMARY**

        Given a tolerance, test if the blob is a rectangle, and how close its
        bounding rectangle's aspect ratio is to 1.0.

        **PARAMETERS**

        * *tolerance* - A percentage difference between an ideal rectangle and
         our hull mask.
        * *ratiotolerance* - A percentage difference of the aspect ratio of
         our blob and an ideal square.

        **RETURNS**

        Boolean True if our object falls within tolerance, false otherwise.

        **EXAMPLE**

        >>> img = Image("lenna")
        >>> blobs = img.find_blobs(128)
        >>> if blobs[-1].is_square():
        >>>     print "it is hip to be square."

        """
        aspect_ratio = abs(1 - self.aspect_ratio)
        if self.is_rectangle(tolerance) and aspect_ratio < ratiotolerance:
            return True
        return False

    def is_rectangle(self, tolerance=0.05):
        """
        **SUMMARY**

        Given a tolerance, test the blob against the rectangle distance to see
        if it is rectangular.

        **PARAMETERS**

        * *tolerance* - The percentage difference between our blob and its
         idealized bounding box.

        **RETURNS**

        Boolean True if the blob is withing the rectangle tolerage, false
        otherwise.

        **EXAMPLE**

        >>> img = Image("lenna")
        >>> blobs = img.find_blobs(128)
        >>> if blobs[-1].isRecangle():
        >>>     print "it is hip to be square."

        """
        if self.rectangle_distance() < tolerance:
            return True
        return False

    def rectangle_distance(self):
        """
        **SUMMARY**

        This compares the hull mask to the bounding rectangle.  Returns the
        area of the blob's hull as a fraction of the bounding rectangle.

        **RETURNS**

        The number of pixels in the blobs hull mask over the number of pixels
        in its bounding box.

        """
        _, whitecount = self.hull_mask.histogram(2)
        return abs(1.0 - float(whitecount) / (
            self.min_rect_width() * self.min_rect_height()))

    def is_circle(self, tolerance=0.05):
        """
        **SUMMARY**

        Test circle distance against a tolerance to see if the blob is
        circular.

        **PARAMETERS**

        * *tolerance* - the percentage difference between our blob and an ideal
         circle.

        **RETURNS**

        True if the feature is within tolerance for being a circle, false
        otherwise.

        """
        if self.circle_distance() < tolerance:
            return True
        return False

    def circle_distance(self):
        """
        **SUMMARY**

        Compare the hull mask to an ideal circle and count the number of pixels
        that deviate as a fraction of total area of the ideal circle.

        **RETURNS**

        The difference, as a percentage, between the hull of our blob and an
        idealized circle of our blob.

        """
        width = self.hull_mask.width
        height = self.hull_mask.height

        idealcircle = Factory.Image((width, height))
        radius = min(width, height) / 2
        idealcircle.dl().circle((width / 2, height / 2), radius, filled=True,
                                color=Color.WHITE)
        idealcircle = idealcircle.apply_layers()
        netdiff = (idealcircle - self.hull_mask) + (
            self.hull_mask - idealcircle)
        _, numwhite = netdiff.histogram(2)
        return float(numwhite) / (radius * radius * np.pi)

    def centroid(self):
        """
        **SUMMARY**

        Return the centroid (mass-determined center) of the blob. Note that
        this is different from the bounding box center.

        **RETURNS**

        An (x,y) tuple that is the center of mass of the blob.

        **EXAMPLE**
        >>> img = Image("lenna")
        >>> blobs = img.find_blobs()
        >>> img.draw_circle((blobs[-1].x, blobs[-1].y), 10, color=Color.RED)
        >>> img.draw_circle((blobs[-1].centroid()), 10, color=Color.BLUE)
        >>> img.show()

        """
        return self.m10 / self.m00, self.m01 / self.m00

    def radius(self):
        """
        **SUMMARY**

        Return the radius, the avg distance of each get_contour point from the
        centroid
        """
        return float(np.mean(spsd.cdist(self.contour, [self.centroid()])))

    def hull_radius(self):
        """
        **SUMMARY**

        Return the radius of the convex hull get_contour from the centroid
        """
        return float(np.mean(spsd.cdist(self.convex_hull, [self.centroid()])))

    @LazyProperty
    def img(self):
        #  NOTE THAT THIS IS NOT PERFECT - ISLAND WITH A LAKE WITH AN ISLAND
        #  WITH A LAKE STUFF
        tlc = self.top_left_corner
        roi = (tlc[0], tlc[1], self.width, self.height)
        roi_img = self.image.crop(*roi)
        mask = self.mask.gray_ndarray != 0  # binary mask
        array = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        array[mask] = roi_img.ndarray[mask]
        return Factory.Image(array)

    @LazyProperty
    def mask(self):
        # TODO: FIX THIS SO THAT THE INTERIOR CONTOURS GET SHIFTED AND DRAWN

        ret_value = np.zeros((self.height, self.width), np.uint8)
        l, t = self.top_left_corner

        # construct the exterior get_contour - these are tuples
        array = np.array([[(p[0] - l, p[1] - t) for p in self.contour]],
                         dtype=np.int32)

        cv2.fillPoly(ret_value, pts=array, color=(255, 255, 255), lineType=8)

        # construct the hole contours
        holes = []
        if self.hole_contour is not None:
            for h in self.hole_contour:  # -- these are lists
                holes.append(np.array([(h2[0] - l, h2[1] - t) for h2 in h],
                                      dtype=np.int32))
            if holes:
                cv2.fillPoly(ret_value, pts=holes, color=(0, 0, 0), lineType=8)
        return Factory.Image(ret_value)

    @LazyProperty
    def hull_img(self):
        tlc = self.top_left_corner
        roi = (tlc[0], tlc[1], self.width, self.height)
        roi_img = self.image.crop(*roi).ndarray
        mask = self.hull_mask.gray_ndarray != 0  # binary mask
        array = np.zeros((self.height, self.width, 3), np.uint8)
        array[mask] = roi_img[mask]
        return Factory.Image(array)

    @LazyProperty
    def hull_mask(self):
        ret_value = np.zeros((self.height, self.width, 3),
                             dtype=np.uint8)
        l, t = self.top_left_corner

        array = np.array([[(p[0] - l, p[1] - t) for p in self.convex_hull]],
                         dtype=np.int32)
        cv2.fillPoly(ret_value, pts=array, color=(255, 255, 255), lineType=8)
        return Factory.Image(ret_value)

    def get_hull_img(self):
        """
        **SUMMARY**

        The convex hull of a blob is the shape that would result if you snapped
        a rubber band around the blob. So if you had the letter "C" as your
        blob the convex hull would be the letter "O."
        This method returns an image where the source image around the convex
        hull of the blob is copied on top a black background.

        **RETURNS**
        Returns a SimpleCV Image of the convex hull, cropped to fit.

        >>> img = Image("lenna")
        >>> blobs = img.find_blobs()
        >>> blobs[-1].get_hull_img().show()

        """
        return self.hull_img

    def get_hull_mask(self):
        """
        **SUMMARY**

        The convex hull of a blob is the shape that would result if you snapped
        a rubber band around the blob. So if you had the letter "C" as your
        blob the convex hull would be the letter "O."
        This method returns an image where the area of the convex hull is white
        and the rest of the image is black. This image is cropped to the size
        of the blob.

        **RETURNS**

        Returns a binary SimpleCV image of the convex hull mask, cropped to
        fit the blob.

        **EXAMPLE**

        >>> img = Image("lenna")
        >>> blobs = img.find_blobs()
        >>> blobs[-1].get_hull_mask().show()

        """
        return self.hull_mask

    def blob_image(self):
        """
        **SUMMARY**

        This method automatically copies all of the image data around the blob
        and puts it in a new image. The resulting image has the size of the
        blob, with the blob data copied in place. Where the blob is not present
        the background is black.

        **RETURNS**

        Returns just the image of the blob (cropped to fit).

        **EXAMPLE**

        >>> img = Image("lenna")
        >>> blobs = img.find_blobs()
        >>> blobs[-1].blob_image().show()

        """
        return self.image

    def blob_mask(self):
        """
        **SUMMARY**

        This method returns an image of the blob's mask. Areas where the blob
        are present are white while all other areas are black. The image is
        cropped to match the blob area.

        **RETURNS**

        Returns a SimplecV image of the blob's mask, cropped to fit.

        **EXAMPLE**

        >>> img = Image("lenna")
        >>> blobs = img.find_blobs()
        >>> blobs[-1].blob_mask().show()

        """
        return self.mask

    def match(self, otherblob):
        """
        **SUMMARY**

        Compare the Hu moments between two blobs to see if they match.  Returns
        a comparison factor -- lower numbers are a closer match.

        **PARAMETERS**

        * *otherblob* - The other blob to compare this one to.

        **RETURNS**

        A single floating point value that is the match quality.

        **EXAMPLE**

        >>> cam = Camera()
        >>> img1 = cam.getImage()
        >>> img2 = cam.getImage()
        >>> b1 = img1.find_blobs()
        >>> b2 = img2.find_blobs()
        >>> for ba in b1:
        >>>     for bb in b2:
        >>>         print ba.match(bb)

        """
        # note: this should use cv.MatchShapes -- but that seems to be
        # broken in OpenCV 2.2  Instead, I reimplemented in numpy
        # according to the description in the docs for method I1 (reciprocal
        # log transformed abs diff)
        #return cv.MatchShapes(self.seq, otherblob.seq,
        #  cv.CV_CONTOURS_MATCH_I1)

        my_signs = np.sign(self.hu)
        my_logs = np.log(np.abs(self.hu))
        my_m = my_signs * my_logs

        other_signs = np.sign(otherblob.hu)
        other_logs = np.log(np.abs(otherblob.hu))
        other_m = other_signs * other_logs

        return np.sum(abs((1 / my_m - 1 / other_m)))

    def get_full_masked_image(self):
        """
        Get the full size image with the masked to the blob
        """
        ret_value = np.zeros((self.image.height, self.image.width, 3),
                             dtype=np.uint8)
        tlc = self.top_left_corner
        roi = (tlc[0], tlc[1], self.width, self.height)
        img_roi = self.image.crop(*roi).ndarray
        mask = self.mask.gray_ndarray != 0  # binary mask
        ret_value_roi = ret_value[Factory.Image.roi_to_slice(roi)]
        ret_value_roi[mask] = img_roi[mask]
        return Factory.Image(ret_value)

    def get_full_hull_masked_image(self):
        """
        Get the full size image with the masked to the blob
        """
        ret_value = np.zeros((self.image.height, self.image.width, 3),
                             dtype=np.uint8)
        tlc = self.top_left_corner
        roi = (tlc[0], tlc[1], self.width, self.height)
        img_roi = self.image.crop(*roi).ndarray
        mask = self.hull_mask.gray_ndarray != 0  # binary mask
        ret_value_roi = ret_value[Factory.Image.roi_to_slice(roi)]
        ret_value_roi[mask] = img_roi[mask]
        return Factory.Image(ret_value)

    def get_full_mask(self):
        """
        Get the full sized image mask
        """
        ret_value = np.zeros((self.image.height, self.image.width),
                             dtype=np.uint8)
        tlc = self.top_left_corner
        roi = (tlc[0], tlc[1], self.width, self.height)
        mask = self.mask.gray_ndarray
        ret_value[Factory.Image.roi_to_slice(roi)] = mask
        return Factory.Image(ret_value)

    def get_full_hull_mask(self):
        """
        Get the full sized image hull mask
        """
        ret_value = np.zeros((self.image.height, self.image.width),
                             dtype=np.uint8)
        tlc = self.top_left_corner
        roi = (tlc[0], tlc[1], self.width, self.height)
        mask = self.hull_mask.gray_ndarray
        ret_value[Factory.Image.roi_to_slice(roi)] = mask
        return Factory.Image(ret_value)

    def get_hull_edge_image(self):
        ret_value = np.zeros((self.image.height, self.image.width, 3),
                             dtype=np.uint8)
        tlc = self.top_left_corner
        translate = [(cs[0] - tlc[0], cs[1] - tlc[1])
                     for cs in self.convex_hull]

        cv2.polylines(ret_value, pts=[np.int32(translate)], isClosed=1,
                      color=(255, 255, 255))
        return Factory.Image(ret_value)

    def get_full_hull_edge_image(self):
        ret_value = np.zeros((self.image.height, self.image.width, 3),
                             dtype=np.uint8)
        cv2.polylines(ret_value, pts=[np.int32(self.convex_hull)], isClosed=1,
                      color=(255, 255, 255))
        return Factory.Image(ret_value)

    def get_edge_image(self):
        """
        Get the edge image for the outer get_contour (no inner holes)
        """
        ret_value = np.zeros((self.image.height, self.image.width, 3),
                             dtype=np.uint8)
        tlc = self.top_left_corner
        translate = [[cs[0] - tlc[0], cs[1] - tlc[1]] for cs in self.contour]
        cv2.polylines(ret_value, [np.int32(translate)], 1, (255, 255, 255))
        return Factory.Image(ret_value)

    def get_full_edge_image(self):
        """
        Get the edge image within the full size image.
        """
        ret_value = np.zeros((self.image.height, self.image.width, 3),
                             dtype=np.uint8)

        cv2.polylines(ret_value, pts=[np.int32(self.contour)], isClosed=1,
                      color=(255, 255, 255))
        return Factory.Image(ret_value)

    def __repr__(self):
        return "simplecv.features.blob.Blob object at (%d, %d) with area %d"\
               % (self.x, self.y, self.area)

    @staticmethod
    def _respace_points(contour, min_distance=1, max_distance=5):
        p0 = np.array(contour[-1])
        min_d = min_distance ** 2
        max_d = max_distance ** 2
        contour = [p0] + contour[:-1]
        contour = contour[:-1]
        ret_value = []  # Keep it empty, else first and last point are the same
        while len(contour) > 0:
            pnt = np.array(contour.pop())
            dist = ((p0[0] - pnt[0]) ** 2) + ((p0[1] - pnt[1]) ** 2)
            if dist > max_d:  # create the new point
                # get the unit vector from p0 to pt
                # from p0 to pt
                a = float((pnt[0] - p0[0]))
                b = float((pnt[1] - p0[1]))
                l = np.sqrt((a ** 2) + (b ** 2))
                punit = np.array([a / l, b / l])
                # make it max_distance long and add it to p0
                new_pnt = (max_distance * punit) + p0
                # push the new point onto the return value
                ret_value.append((new_pnt[0], new_pnt[1]))
                # push the new point onto the contour too
                # FIXME: "push the new point" -> ...append(pt) ?
                contour.append(pnt)
                p0 = new_pnt
            elif dist > min_d:
                p0 = np.array(pnt)
                ret_value.append(pnt)
        return ret_value

    def _filter_sc_points(self, min_distance=3, max_distance=8):
        """
        Go through ever point in the get_contour and make sure
        that it is no less than min distance to the next point
        and no more than max_distance from the the next point.
        """
        complete_contour = self._respace_points(self.contour, min_distance,
                                                max_distance)
        if self.hole_contour is not None:
            for ctr in self.hole_contour:
                complete_contour += self._respace_points(ctr, min_distance,
                                                         max_distance)
        return complete_contour

    def get_sc_descriptors(self):
        if self._scdescriptors is None:
            complete_contour = self._filter_sc_points()
            descriptors = self._generate_sc(complete_contour)
            self._scdescriptors = descriptors
            self._complete_contour = complete_contour
        return self._scdescriptors, self._complete_contour

    def _generate_sc(self, complete_contour, dsz=6, r_bound=[0.1, 2.1]):
        """
        Create the shape context objects.
        dsz - The size of descriptor as a dszxdsz histogram
        complete_contour - All of the edge points as a long list
        r_bound - Bounds on the log part of the shape context descriptor
        """
        # To compute descriptors, contour points need to be linked in circle
        p0 = np.array(complete_contour[-1])
        complete_contour = [p0] + complete_contour[:]

        data = []
        for pnt in complete_contour:  #
            temp = []
            # take each other point in the contour, center it on pnt, and
            # covert it to log polar
            for b in complete_contour:
                r = np.sqrt((b[0] - pnt[0]) ** 2 + (b[1] - pnt[1]) ** 2)
                #                if( r > 100 ):
                #                    continue

                # numpy throws an inf here that mucks the system up
                if r == 0.00:
                    continue
                r = np.log10(r)
                theta = np.arctan2(b[0] - pnt[0], b[1] - pnt[1])
                if np.isfinite(r) and np.isfinite(theta):
                    temp.append((r, theta))
            data.append(temp)

        #UHG!!! need to repeat this for all of the interior contours too
        descriptors = []
        #dsz = 6
        # for each point in the get_contour
        for point in data:
            test = np.array(point)
            # generate a 2D histrogram, and flatten it out.
            hist, _, _ = np.histogram2d(test[:, 0], test[:, 1], dsz,
                                        [r_bound, [np.pi * -1 / 2, np.pi / 2]],
                                        normed=True)
            hist = hist.reshape(1, dsz ** 2)
            if np.all(np.isfinite(hist[0])):
                descriptors.append(hist[0])
        self._scdescriptors = descriptors
        return descriptors

    def get_shape_context(self):
        """
        Return the shape context descriptors as a featureset. Corrently
        this is not used for recognition but we will perhaps use it soon.
        """
        # still need to subsample big contours
        derp = self.get_sc_descriptors()
        descriptors, complete_contour = self.get_sc_descriptors()
        fset = FeatureSet()
        for i in range(0, len(complete_contour)):
            fset.append(ShapeContextDescriptor(self.image, complete_contour[i],
                                               descriptors[i], self))

        return fset

    def show_correspondence(self, other_blob, side="left"):
        """
        This is total beta - use at your own risk.
        """
        # We're lazy right now, assume the blob images are the same size
        side = side.lower()
        my_pts = self.get_shape_context()
        your_pts = other_blob.get_shape_context()

        my_img = self.image.copy()
        your_img = other_blob.image.copy()

        my_pts = my_pts.reassign_image(my_img)
        your_pts = your_pts.reassign_image(your_img)

        my_pts.draw()
        my_img = my_img.apply_layers()
        your_pts.draw()
        your_img = your_img.apply_layers()

        result = my_img.side_by_side(your_img, side=side)
        # FIXME: unknown method, may be match method should be used?
        data = self.shape_context_match(other_blob)
        mapvals = data[0]
        color = Color()
        for i in range(0, len(self._complete_contour)):
            lhs = self._complete_contour[i]
            idx = mapvals[i]
            rhs = other_blob._complete_contour[idx]
            if side == "left":
                shift = (rhs[0] + your_img.width, rhs[1])
                result.draw_line(lhs, shift, color=color.get_random(),
                                 thickness=1)
            elif side == "bottom":
                shift = (rhs[0], rhs[1] + my_img.height)
                result.draw_line(lhs, shift, color=color.get_random(),
                                 thickness=1)
            elif side == "right":
                shift = (rhs[0] + my_img.width, rhs[1])
                result.draw_line(lhs, shift, color=color.get_random(),
                                 thickness=1)
            elif side == "top":
                shift = (lhs[0], lhs[1] + my_img.height)
                result.draw_line(lhs, shift, color=color.get_random(),
                                 thickness=1)

        return result

    def get_match_metric(self, other_blob):
        """
        This match metric is now deprecated.
        """
        # FIXME: unknown method, may be match method should be used?
        data = self.shape_context_match(other_blob)
        distances = np.array(data[1])
        sd = np.std(distances)
        x = np.mean(distances)
        min_dist = np.min(distances)
        # not sure trimmed mean is perfect
        # realistically we should have some bimodal dist
        # and we want to throw away stuff with awful matches
        # so long as the number of points is not a huge
        # chunk of our points.
        tmean = sps.tmean(distances, (min_dist, x + sd))
        return tmean

    def get_convexity_defects(self, return_points=False):
        """
        **SUMMARY**

        Get Convexity Defects of the get_contour.

        **PARAMETERS**

        *returnPoints* - Bool(False).
                         If False: Returns FeatureSet of Line(start point,
                         end point) and Corner(far point)
                         If True: Returns a list of tuples
                         (start point, end point, far point)
        **RETURNS**

        FeatureSet - A FeatureSet of Line and Corner objects
                     OR
                     A list of (start point, end point, far point)
                     See PARAMETERS.

        **EXAMPLE**

        >>> img = Image('lenna')
        >>> blobs = img.find_blobs()
        >>> blob = blobs[-1]
        >>> lines, farpoints = blob.get_convexity_defects()
        >>> lines.draw()
        >>> farpoints.draw(color=Color.RED, width=-1)
        >>> img.show()

        >>> points = blob.get_convexity_defects(return_points=True)
        >>> startpoints = zip(*points)[0]
        >>> endpoints = zip(*points)[0]
        >>> farpoints = zip(*points)[0]
        >>> print startpoints, endpoints, farpoints
        """

        if not hasattr(cv2, 'convexityDefects'):
            warnings.warn('cv2.convexityDefects is available since '
                          'OpenCV 2.4')
            return FeatureSet([])

        hull = [self.contour.index(x) for x in self.convex_hull]
        hull = np.array(hull).reshape(len(hull), 1)
        defects = cv2.convexityDefects(contour=np.array(self.contour),
                                       convexhull=hull)
        if isinstance(defects, type(None)):
            warnings.warn("Unable to find defects. "
                          "Returning Empty FeatureSet.")
            defects = []
        points = [(self.contour[defect[0][0]],
                   self.contour[defect[0][1]],
                   self.contour[defect[0][2]]) for defect in defects]

        if return_points:
            return FeatureSet(points)
        else:
            lines = FeatureSet(
                [Line(self.image, (start, end)) for start, end, far in points])
            farpoints = FeatureSet(
                [Corner(self.image, far[0], far[1]) for start, end, far in
                 points])
            features = FeatureSet([lines, farpoints])
            return features

    def bounding_circle(self):
        """
        **SUMMARY**

        This function calculates the minimum bounding circle of the blob in the
        image as an (x,y,r) tuple

        **RETURNS**

        An (x,y,r) tuple where (x,y) is the center of the circle and r is the
        radius

        **EXAMPLE**

        >>> img = Image("RatMask.png")
        >>> blobs = img.find_blobs()
        >>> print blobs[-1].bounding_circle()

        """
        points = []
        # list of contour points converted to suitable format to pass
        # into cv2.minEnclosingCircle()
        for pair in self.contour:
            points.append([[pair[0], pair[1]]])

        points = np.array(points)

        (cen, rad) = cv2.minEnclosingCircle(points)

        return cen[0], cen[1], rad

    @classmethod
    def find(cls, img, threshval=None, minsize=10, maxsize=0,
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
        >>> fs = img.find(Blobs)
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
                         p=threshconstant, inverted=True).invert(),
            img, minsize=minsize, maxsize=maxsize, appx_level=appx_level)

        if not len(blobs):
            return None

        return FeatureSet(blobs).sort_area()

    @classmethod
    def find_from_skintone(cls, img, minsize=10, maxsize=0, dilate_iter=1):
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
        >>> fs = Blob.find_from_skintone(img)
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

    @classmethod
    def find_from_palette(cld, img, palette_selection, dilate=0, minsize=5,
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
        >>> blobs = Blob.find_from_palette(img, (p[0], p[1], p[6]))
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

    @classmethod
    def smart_find(cls, img, mask=None, rect=None, thresh_level=2,
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
        >>> blobs = Blob.smart_find(img, mask=mask)
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

    @classmethod
    def find_from_mask(cls, img, mask, threshold=128, minsize=10, maxsize=0,
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
        >>> blobs = Blob.find_from_mask(img, mask)
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
        gray = mask.gray_ndarray
        val, result = cv2.threshold(gray, thresh=threshold, maxval=255,
                                    type=cv2.THRESH_BINARY)
        blobs = blobmaker.extract_from_binary(
            Factory.Image(result), img,
            minsize=minsize, maxsize=maxsize, appx_level=appx_level)
        if not len(blobs):
            return None
        return FeatureSet(blobs).sort_area()

    @classmethod
    def find_from_flood_fill(cls, img, points, tolerance=None, lower=None,
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
        >>> blerbs = Blob.find_from_flood_fill(img, ((10, 10), (20, 20), (30, 30)),
        ...                                    tolerance=30)
        >>> blerbs.show()

        **SEE ALSO**

        :py:meth:`find_blobs`
        :py:meth:`flood_fill`

        """
        mask = img.flood_fill_to_mask(points, tolerance, color=Color.WHITE,
                                      lower=lower, upper=upper,
                                      fixed_range=fixed_range)
        return cls.find_from_mask(img, mask, minsize, maxsize)

    @classmethod
    def find_from_watershed(cls, img, mask=None, erode=2, dilate=2,
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
        >>> blobs = Blob.find_from_watershed(img, mask)
        >>> blobs.show()

        **SEE ALSO**
        Color.WATERSHED_FG - The watershed foreground color
        Color.WATERSHED_BG - The watershed background color
        Color.WATERSHED_UNSURE - The watershed not sure if fg or bg color.

        """
        newmask = img.watershed(mask, erode, dilate, use_my_mask)
        if invert:
            newmask = mask.invert()
        return cls.find_from_mask(img, newmask, minsize=minsize,
                                  maxsize=maxsize)

    @classmethod
    def find_from_hue_histogram(cls, img, model, threshold=1, smooth=True,
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
        >>>> blobs = Blob.find_from_hue_histogram(img, hist)
        >>>> blobs.show()

        **SEE ALSO**

        ImageClass.get_normalized_hue_histogram()
        ImageClass.back_project_hue_histogram()

        """
        new_mask = img.back_project_hue_histogram(model=model, smooth=smooth,
                                                  full_color=False,
                                                  threshold=threshold)
        return cls.find_from_mask(img, new_mask, minsize=minsize,
                                  maxsize=maxsize)
