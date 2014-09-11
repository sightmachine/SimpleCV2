# SimpleCV Feature library
#
# Tools return basic features in feature sets

# #    x = 0.00
#     y = 0.00
#     max_x = None
#     max_y = None
#     min_x = None
#     min_y = None
#     width = None
#     height = None
#     src_img_w = None
#     src_img_h = None

from math import sqrt
import copy
import re
import types
import warnings

import cv2
import numpy as np
import scipy.spatial.distance as spsd

from simplecv.base import logger, lazyproperty
from simplecv.color import Color
from simplecv.core.pluginsystem import apply_plugins


class FeatureSet(list):
    """
    **SUMMARY**

    FeatureSet is a class extended from Python's list which has special
    functions so that it is useful for handling feature metadata on an image.

    In general, functions dealing with attributes will return numpy arrays,
    and functions dealing with sorting or filtering will return new FeatureSets

    **EXAMPLE**

    >>> image = Image("/path/to/image.png")
    >>> lines = image.find_lines()  # lines are the feature set
    >>> lines.draw()
    >>> lines.x()
    >>> lines.crop()
    """

    def __getitem__(self, key):
        """
        **SUMMARY**

        Returns a FeatureSet when sliced. Previously used to
        return list. Now it is possible to use FeatureSet member
        functions on sub-lists

        """
        if isinstance(key, types.SliceType):  # Or can use 'try:' for speed
            return FeatureSet(list.__getitem__(self, key))
        else:
            return list.__getitem__(self, key)

    def __getslice__(self, i, j):
        """
        Deprecated since python 2.0, now using __getitem__
        """
        return self.__getitem__(slice(i, j))

    def count(self):
        """
        This function returns the length / count of the all the items in the
        FeatureSet
        """

        return len(self)

    def draw(self, color=Color.GREEN, width=1, autocolor=False, alpha=-1):
        """
        **SUMMARY**

        Call the draw() method on each feature in the FeatureSet.

        **PARAMETERS**

        * *color* - The color to draw the object. Either an BGR tuple or
         a member of the :py:class:`Color` class.
        * *width* - The width to draw the feature in pixels. A value of -1
         usually indicates a filled region.
        * *autocolor* - If true a color is randomly selected for each feature.


        **RETURNS**

        Nada. Nothing. Zilch.

        **EXAMPLE**

        >>> img = Image("lenna")
        >>> feats = img.find_blobs()
        >>> feats.draw(color=Color.PUCE, width=3)
        >>> img.show()

        """
        for feat in self:
            if autocolor:
                color = Color().get_random()
            if alpha != -1:
                feat.draw(color=color, width=width, alpha=alpha)
            else:
                feat.draw(color=color, width=width)

    def show(self, color=Color.GREEN, autocolor=False, width=1):
        """
        **EXAMPLE**

        This function will automatically draw the features on the image and
        show it.
        It is a basically a shortcut function for development and is the same
        as:

        **PARAMETERS**

        * *color* - The color to draw the object. Either an BGR tuple or
         a member of the :py:class:`Color` class.
        * *width* - The width to draw the feature in pixels. A value of -1
         usually indicates a filled region.
        * *autocolor* - If true a color is randomly selected for each feature.

        **RETURNS**

        Nada. Nothing. Zilch.


        **EXAMPLE**
        >>> img = Image("logo")
        >>> feat = img.find_blobs()
        >>> if feat: feat.draw()
        >>> img.show()

        """
        self.draw(color, width, autocolor)
        self[-1].image.show()

    def reassign_image(self, new_img):
        """
        **SUMMARY**

        Return a new featureset where the features are assigned to a new image.

        **PARAMETERS**

        * *img* - the new image to which to assign the feature.

        .. Warning::
          THIS DOES NOT PERFORM A SIZE CHECK. IF YOUR NEW IMAGE IS NOT THE
          EXACT SAME SIZE YOU WILL CERTAINLY CAUSE ERRORS.

        **EXAMPLE**

        >>> img = Image("lenna")
        >>> img2 = img.invert()
        >>> l = img.find_lines()
        >>> l2 = img.reassign_image(img2)
        >>> l2.show()

        """
        ret_value = FeatureSet()
        for feat in self:
            ret_value.append(feat.reassign(new_img))
        return ret_value

    def x(self):
        """
        **SUMMARY**

        Returns a numpy array of the x (horizontal) coordinate of each feature.

        **RETURNS**

        A numpy array.

        **EXAMPLE**

        >>> img = Image("lenna")
        >>> feats = img.find_blobs()
        >>> xs = feats.x()
        >>> print xs

        """
        return np.array([feat.x for feat in self])

    def y(self):
        """
        **SUMMARY**

        Returns a numpy array of the y (vertical) coordinate of each feature.

        **RETURNS**

        A numpy array.

        **EXAMPLE**

        >>> img = Image("lenna")
        >>> feats = img.find_blobs()
        >>> xs = feats.y()
        >>> print xs

        """
        return np.array([feat.y for feat in self])

    def coordinates(self):
        """
        **SUMMARY**

        Returns a 2d numpy array of the x,y coordinates of each feature.  This
        is particularly useful if you want to use Scipy's Spatial Distance
        module

        **RETURNS**

        A numpy array of all the positions in the featureset.

        **EXAMPLE**

        >>> img = Image("lenna")
        >>> feats = img.find_blobs()
        >>> xs = feats.coordinates()
        >>> print xs


        """
        return np.array([[feat.x, feat.y] for feat in self])

    def center(self):
        return self.coordinates()

    def get_area(self):
        """
        **SUMMARY**

        Returns a numpy array of the area of each feature in pixels.

        **RETURNS**

        A numpy array of all the positions in the featureset.

        **EXAMPLE**

        >>> img = Image("lenna")
        >>> feats = img.find_blobs()
        >>> xs = feats.get_area()
        >>> print xs

        """
        return np.array([f.area for f in self])

    def sort_area(self):
        """
        **SUMMARY**

        Returns a new FeatureSet, with the largest area features first.

        **RETURNS**

        A featureset sorted based on area.

        **EXAMPLE**

        >>> img = Image("lenna")
        >>> feats = img.find_blobs()
        >>> feats = feats.sort_area()
        >>> print feats[-1]  # biggest blob
        >>> print feats[0]  # smallest blob

        """
        return FeatureSet(sorted(self, key=lambda f: f.area))

    def sort_x(self):
        """
        **SUMMARY**

        Returns a new FeatureSet, with the smallest x coordinates features
        first.

        **RETURNS**

        A featureset sorted based on area.

        **EXAMPLE**

        >>> img = Image("lenna")
        >>> feats = img.find_blobs()
        >>> feats = feats.sort_x()
        >>> print feats[-1]  # biggest blob
        >>> print feats[0]  # smallest blob

        """
        return FeatureSet(sorted(self, key=lambda f: f.x))

    def sort_y(self):
        """
        **SUMMARY**

        Returns a new FeatureSet, with the smallest y coordinates features
        first.

        **RETURNS**

        A featureset sorted based on area.

        **EXAMPLE**

        >>> img = Image("lenna")
        >>> feats = img.find_blobs()
        >>> feats = feats.sortY()
        >>> print feats[-1]  # biggest blob
        >>> print feats[0]  # smallest blob

        """
        return FeatureSet(sorted(self, key=lambda f: f.y))

    def distance_from(self, point=(-1, -1)):
        """
        **SUMMARY**

        Returns a numpy array of the distance each Feature is from a given
        coordinate. Default is the center of the image.

        **PARAMETERS**

        * *point* - A point on the image from which we will calculate distance.

        **RETURNS**

        A numpy array of distance values.

        **EXAMPLE**

        >>> img = Image("lenna")
        >>> feats = img.find_blobs()
        >>> d = feats.distance_from()
        >>> d[0]  #show the 0th blobs distance to the center.

        **TO DO**

        Make this accept other features to measure from.

        """
        if point[0] == -1 or point[1] == -1 and len(self):
            point = self[0].image.size_tuple

        return spsd.cdist(self.coordinates(), [point])[:, 0]

    def sort_distance(self, point=(-1, -1)):
        """
        **SUMMARY**

        Returns a sorted FeatureSet with the features closest to a given
        coordinate first. Default is from the center of the image.

        **PARAMETERS**

        * *point* - A point on the image from which we will calculate distance.

        **RETURNS**

        A numpy array of distance values.

        **EXAMPLE**

        >>> img = Image("lenna")
        >>> feats = img.find_blobs()
        >>> d = feats.sort_distance()
        >>> d[-1].show()  #show the 0th blobs distance to the center.


        """
        return FeatureSet(sorted(self, key=lambda f: f.distance_from(point)))

    def distance_pairs(self):
        """
        **SUMMARY**

        Returns the square-form of pairwise distances for the featureset.
        The resulting N x N array can be used to quickly look up distances
        between features.

        **RETURNS**

        A NxN np matrix of distance values.

        **EXAMPLE**

        >>> img = Image("lenna")
        >>> feats = img.find_blobs()
        >>> d = feats.distance_pairs()
        >>> print d

        """
        return spsd.squareform(spsd.pdist(self.coordinates()))

    def get_angle(self):
        """
        **SUMMARY**

        Return a numpy array of the angles (theta) of each feature.
        Note that theta is given in degrees, with 0 being horizontal.

        **RETURNS**

        An array of angle values corresponding to the features.

        **EXAMPLE**

        >>> img = Image("lenna")
        >>> l = img.find_lines()
        >>> angs = l.get_angle()
        >>> print angs


        """
        return np.array([f.angle for f in self])

    def sort_angle(self, theta=0):
        """
        Return a sorted FeatureSet with the features closest to a given angle
        first. Note that theta is given in radians, with 0 being horizontal.

        **RETURNS**

        An array of angle values corresponding to the features.

        **EXAMPLE**

        >>> img = Image("lenna")
        >>> l = img.find_lines()
        >>> l = l.sort_angle()
        >>> print angs

        """
        return FeatureSet(sorted(self,
                                 key=lambda f: abs(f.angle - theta)))

    def length(self):
        """
        **SUMMARY**

        Return a numpy array of the length (longest dimension) of each feature.

        **RETURNS**

        A numpy array of the length, in pixels, of eatch feature object.

        **EXAMPLE**

        >>> img = Image("Lenna")
        >>> l = img.find_lines()
        >>> lengt = l.length()
        >>> lengt[0] # length of the 0th element.

        """

        return np.array([f.length for f in self])

    def sort_length(self):
        """
        **SUMMARY**

        Return a sorted FeatureSet with the longest features first.

        **RETURNS**

        A sorted FeatureSet.

        **EXAMPLE**

        >>> img = Image("Lenna")
        >>> l = img.find_lines().sort_length()
        >>> lengt[-1] # length of the 0th element.

        """
        return FeatureSet(sorted(self, key=lambda f: f.length))

    def mean_color(self):
        """
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
        return np.array([f.mean_color for f in self])

    def color_distance(self, color=(0, 0, 0)):
        """
        **SUMMARY**

        Return a numpy array of the distance each features average color is
        from a given color tuple (default black, so color_distance() returns
        intensity)

        **PARAMETERS**

        * *color* - The color to calculate the distance from.

        **RETURNS**

        The distance of the average color for the feature from given color as
        a numpy array.

        **EXAMPLE**

        >>> img = Image("lenna")
        >>> circs = img.find_circle()
        >>> d = circs.color_distance(color=Color.BLUE)
        >>> print d

        """
        return spsd.cdist(self.mean_color, [color])[:, 0]

    def sort_color_distance(self, color=(0, 0, 0)):
        """
        Return a sorted FeatureSet with features closest to a given color
        first. Default is black, so sort_color_distance() will return darkest
        to brightest
        """
        return FeatureSet(sorted(self, key=lambda f: f.color_distance(color)))

    def filter(self, filterarray):
        """
        **SUMMARY**

        Return a FeatureSet which is filtered on a numpy boolean array.  This
        will let you use the attribute functions to easily screen Features out
        of return FeatureSets.

        **PARAMETERS**

        * *filterarray* - A numpy array, matching  the size of the feature set,
          made of Boolean values, we return the true values and reject the
          False value.

        **RETURNS**

        The revised feature set.

        **EXAMPLE**

        Return all lines < 200px

        # returns all lines < 200px
        >>> my_lines.filter(my_lines.length() < 200)

        # returns blobs that are nearly square
        >>> my_blobs.filter(my_blobs.get_area() > 0.9 * my_blobs.length**2)

        # any lines within 45 degrees of horizontal
        >>> my_lines.filter(abs(my_lines.get_angle()) < numpy.pi / 4)

        # only return corners in the upper diagonal of the image
        >>> my_corners.filter(my_corners.x() - my_corners.y() > 0)

        """
        return FeatureSet(list(np.array(self)[np.array(filterarray)]))

    def get_width(self):
        """
        **SUMMARY**

        Returns a nparray which is the width of all the objects in the
        FeatureSet.

        **RETURNS**

        A numpy array of width values.


        **EXAMPLE**

        >>> img = Image("NotLenna")
        >>> l = img.find_lines()
        >>> l.get_width()

        """
        return np.array([f.width for f in self])

    def get_height(self):
        """
        Returns a nparray which is the height of all the objects in the
        FeatureSet

        **RETURNS**

        A numpy array of width values.


        **EXAMPLE**

        >>> img = Image("NotLenna")
        >>> l = img.find_lines()
        >>> l.get_height()

        """
        return np.array([f.height for f in self])

    def crop(self):
        """
        **SUMMARY**

        Returns a nparray with the cropped features as SimpleCV image.

        **RETURNS**

        A SimpleCV image cropped to each image.

        **EXAMPLE**

        >>> img = Image("Lenna")
        >>> blobs = img.find_blobs(128)
        >>> for b in blobs:
        >>>   newImg = b.crop()
        >>>   newImg.show()
        >>>   time.sleep(1)

        """
        return np.array([f.crop() for f in self])

    def inside(self, region):
        """
        **SUMMARY**

        Return only the features inside the region. where region can be a
        bounding box, bounding circle, a list of tuples in a closed polygon,
        or any other featutres.

        **PARAMETERS**

        * *region*

          * A bounding box - of the form (x,y,w,h) where x,y is the upper left
           corner
          * A bounding circle of the form (x,y,r)
          * A list of x,y tuples defining a closed polygon
           e.g. ((x,y),(x,y),....)
          * Any two dimensional feature (e.g. blobs, circle ...)

        **RETURNS**

        Returns a featureset of features that are inside the region.

        **EXAMPLE**

        >>> img = Image("Lenna")
        >>> blobs = img.find_blobs()
        >>> b = blobs[-1]
        >>> lines = img.find_lines()
        >>> inside = lines.inside(b)

        **NOTE**

        This currently performs a bounding box test, not a full polygon test
        for speed.


        """
        fset = FeatureSet()
        for feat in self:
            if feat.is_contained_within(region):
                fset.append(feat)
        return fset

    def outside(self, region):
        """
        **SUMMARY**

        Return only the features outside the region. where region can be
        a bounding box, bounding circle, a list of tuples in a closed polygon,
        or any other featutres.

        **PARAMETERS**

        * *region*

          * A bounding box - of the form (x,y,w,h) where x,y is the upper
           left corner
          * A bounding circle of the form (x,y,r)
          * A list of x,y tuples defining a closed polygon
           e.g. ((x,y),(x,y),....)
          * Any two dimensional feature (e.g. blobs, circle ...)

        **RETURNS**

        Returns a featureset of features that are outside the region.


        **EXAMPLE**

        >>> img = Image("Lenna")
        >>> blobs = img.find_blobs()
        >>> b = blobs[-1]
        >>> lines = img.find_lines()
        >>> outside = lines.outside(b)

        **NOTE**

        This currently performs a bounding box test, not a full polygon test
        for speed.

        """
        fset = FeatureSet()
        for feat in self:
            if not feat.is_contained_within(region):
                fset.append(feat)
        return fset

    def overlaps(self, region):
        """
        **SUMMARY**

        Return only the features that overlap or the region. Where region can
        be a bounding box, bounding circle, a list of tuples in a closed
        polygon, or any other featutres.

        **PARAMETERS**

        * *region*

          * A bounding box - of the form (x,y,w,h) where x,y is the upper left
           corner
          * A bounding circle of the form (x,y,r)
          * A list of x,y tuples defining a closed polygon
           e.g. ((x,y),(x,y),....)
          * Any two dimensional feature (e.g. blobs, circle ...)

        **RETURNS**

        Returns a featureset of features that overlap the region.

        **EXAMPLE**

        >>> img = Image("Lenna")
        >>> blobs = img.find_blobs()
        >>> b = blobs[-1]
        >>> lines = img.find_lines()
        >>> outside = lines.overlaps(b)

        **NOTE**

        This currently performs a bounding box test, not a full polygon test
        for speed.

        """
        fset = FeatureSet()
        for feat in self:
            if feat.overlaps(region):
                fset.append(feat)
        return fset

    def above(self, region):
        """
        **SUMMARY**

        Return only the features that are above a  region. Where region can be
        a bounding box, bounding circle, a list of tuples in a closed polygon,
        or any other featutres.

        **PARAMETERS**

        * *region*

          * A bounding box - of the form (x,y,w,h) where x,y is the upper left
           corner
          * A bounding circle of the form (x,y,r)
          * A list of x,y tuples defining a closed polygon
           e.g. ((x,y),(x,y),....)
          * Any two dimensional feature (e.g. blobs, circle ...)

        **RETURNS**

        Returns a featureset of features that are above the region.

        **EXAMPLE**

        >>> img = Image("Lenna")
        >>> blobs = img.find_blobs()
        >>> b = blobs[-1]
        >>> lines = img.find_lines()
        >>> outside = lines.above(b)

        **NOTE**

        This currently performs a bounding box test, not a full polygon test
        for speed.

        """
        fset = FeatureSet()
        for feat in self:
            if feat.above(region):
                fset.append(feat)
        return fset

    def below(self, region):
        """
        **SUMMARY**

        Return only the features below the region. where region can be a
        bounding box, bounding circle, a list of tuples in a closed polygon,
        or any other featutres.

        **PARAMETERS**

        * *region*

          * A bounding box - of the form (x,y,w,h) where x,y is the upper left
           corner
          * A bounding circle of the form (x,y,r)
          * A list of x,y tuples defining a closed polygon
           e.g. ((x,y),(x,y),....)
          * Any two dimensional feature (e.g. blobs, circle ...)

        **RETURNS**

        Returns a featureset of features that are below the region.

        **EXAMPLE**

        >>> img = Image("Lenna")
        >>> blobs = img.find_blobs()
        >>> b = blobs[-1]
        >>> lines = img.find_lines()
        >>> inside = lines.below(b)

        **NOTE**

        This currently performs a bounding box test, not a full polygon test
        for speed.

        """
        fset = FeatureSet()
        for feat in self:
            if feat.below(region):
                fset.append(feat)
        return fset

    def left(self, region):
        """
        **SUMMARY**

        Return only the features left of the region. where region can be
        a bounding box, bounding circle, a list of tuples in a closed polygon,
        or any other featutres.

        **PARAMETERS**

        * *region*

          * A bounding box - of the form (x,y,w,h) where x,y is the upper left
           corner
          * A bounding circle of the form (x,y,r)
          * A list of x,y tuples defining a closed polygon
           e.g. ((x,y),(x,y),....)
          * Any two dimensional feature (e.g. blobs, circle ...)

        **RETURNS**

        Returns a featureset of features that are left of the region.

        **EXAMPLE**

        >>> img = Image("Lenna")
        >>> blobs = img.find_blobs()
        >>> b = blobs[-1]
        >>> lines = img.find_lines()
        >>> left = lines.left(b)

        **NOTE**

        This currently performs a bounding box test, not a full polygon test
        for speed.

        """
        fset = FeatureSet()
        for feat in self:
            if feat.left(region):
                fset.append(feat)
        return fset

    def right(self, region):
        """
        **SUMMARY**

        Return only the features right of the region. where region can be
        a bounding box, bounding circle, a list of tuples in a closed polygon,
        or any other featutres.

        **PARAMETERS**

        * *region*

          * A bounding box - of the form (x,y,w,h) where x,y is the upper left
           corner
          * A bounding circle of the form (x,y,r)
          * A list of x,y tuples defining a closed polygon
           e.g. ((x,y),(x,y),....)
          * Any two dimensional feature (e.g. blobs, circle ...)

        **RETURNS**

        Returns a featureset of features that are right of the region.

        **EXAMPLE**

        >>> img = Image("Lenna")
        >>> blobs = img.find_blobs()
        >>> b = blobs[-1]
        >>> lines = img.find_lines()
        >>> right = lines.right(b)

        **NOTE**

        This currently performs a bounding box test, not a full polygon test
        for speed.

        """
        fset = FeatureSet()
        for feat in self:
            if feat.right(region):
                fset.append(feat)
        return fset

    def on_image_edge(self, tolerance=1):
        """
        **SUMMARY**

        The method returns a feature set of features that are on or "near" the
        edge of the image. This is really helpful for removing features that
        are edge effects.

        **PARAMETERS**

        * *tolerance* - the distance in pixels from the edge at which a feature
          qualifies as being "on" the edge of the image.

        **RETURNS**

        Returns a featureset of features that are on the edge of the image.

        **EXAMPLE**

        >>> img = Image("./sampleimages/EdgeTest1.png")
        >>> blobs = img.find_blobs()
        >>> es = blobs.on_image_edge()
        >>> es.draw(color=Color.RED)
        >>> img.show()

        """
        fset = FeatureSet()
        for feat in self:
            if feat.on_image_edge(tolerance):
                fset.append(feat)
        return fset

    def not_on_image_edge(self, tolerance=1):
        """
        **SUMMARY**

        The method returns a feature set of features that are not on or "near"
        the edge of the image. This is really helpful for removing features
        that are edge effects.

        **PARAMETERS**

        * *tolerance* - the distance in pixels from the edge at which a feature
          qualifies as being "on" the edge of the image.

        **RETURNS**

        Returns a featureset of features that are not on the edge of the image.

        **EXAMPLE**

        >>> img = Image("./sampleimages/EdgeTest1.png")
        >>> blobs = img.find_blobs()
        >>> es = blobs.not_on_image_edge()
        >>> es.draw(color=Color.RED)
        >>> img.show()

        """
        fset = FeatureSet()
        for feat in self:
            if feat.not_on_image_edge(tolerance):
                fset.append(feat)
        return fset

    def top_left_corners(self):
        """
        **SUMMARY**

        This method returns the top left corner of each feature's bounding box.

        **RETURNS**

        A numpy array of x,y position values.

        **EXAMPLE**

        >>> img = Image("./sampleimages/EdgeTest1.png")
        >>> blobs = img.find_blobs()
        >>> tl = img.top_left_corners()
        >>> print tl[0]
        """
        return np.array([f.top_left_corner for f in self])

    def bottom_left_corners(self):
        """
        **SUMMARY**

        This method returns the bottom left corner of each feature's bounding
        box.

        **RETURNS**

        A numpy array of x,y position values.

        **EXAMPLE**

        >>> img = Image("./sampleimages/EdgeTest1.png")
        >>> blobs = img.find_blobs()
        >>> bl = img.bottom_left_corners()
        >>> print bl[0]

        """
        return np.array([f.bottom_left_corner for f in self])

    def top_right_corners(self):
        """
        **SUMMARY**

        This method returns the top right corner of each feature's bounding
        box.

        **RETURNS**

        A numpy array of x,y position values.

        **EXAMPLE**

        >>> img = Image("./sampleimages/EdgeTest1.png")
        >>> blobs = img.find_blobs()
        >>> tr = img.top_right_corners()
        >>> print tr[0]

        """
        return np.array([f.top_right_corner for f in self])

    def bottom_right_corners(self):
        """
        **SUMMARY**

        This method returns the bottom right corner of each feature's bounding
        box.

        **RETURNS**

        A numpy array of x,y position values.

        **EXAMPLE**

        >>> img = Image("./sampleimages/EdgeTest1.png")
        >>> blobs = img.find_blobs()
        >>> br = img.bottom_right_corners()
        >>> print br[0]

        """
        return np.array([f.bottom_right_corner for f in self])

    def aspect_ratios(self):
        """
        **SUMMARY**

        Return the aspect ratio of all the features in the feature set, For
        our purposes aspect ration is max(width,height)/min(width,height).

        **RETURNS**

        A numpy array of the aspect ratio of the features in the featureset.

        **EXAMPLE**

        >>> img = Image("OWS.jpg")
        >>> blobs = img.find_blobs(128)
        >>> print blobs.aspect_ratios()

        """
        return np.array([f.aspect_ratio for f in self])

    def cluster(self, method="kmeans", properties=None, k=3):
        """

        **SUMMARY**

        This function clusters the blobs in the featureSet based on the
        properties. Properties can be "color", "shape" or "position" of blobs.
        Clustering is done using K-Means or Hierarchical clustering(Ward)
        algorithm.

        **PARAMETERS**

        * *properties* - It should be a list with any combination of "color",
         "shape", "position". properties = ["color","position"].
          properties = ["position","shape"].
          properties = ["shape"]
        * *method* - if method is "kmeans", it will cluster using K-Means
         algorithm, if the method is "hierarchical", no need to spicify the
         number of clusters
        * *k* - The number of clusters(kmeans).


        **RETURNS**

        A list of featureset, each being a cluster itself.

        **EXAMPLE**

          >>> img = Image("lenna")
          >>> blobs = img.find_blobs()
          >>> clusters = blobs.cluster(method="kmeans",
          >>>       properties=["color"],
          >>>       k=5)
          >>> for i in clusters:
          >>>     i.draw(color=Color.get_random(), width=5)
          >>> img.show()

        """
        try:
            from sklearn.cluster import KMeans, Ward
            from sklearn import __version__
        except ImportError:
            logger.warning("install scikits-learning package")
            return
        fvect = []  # List of feature vector of each blob
        if not properties:
            properties = ['color', 'shape', 'position']
        if k > len(self):
            logger.warning(
                "Number of clusters cannot be greater then the number of blobs"
                " in the featureset")
            return
        for i in self:
            feature_vector = []
            if 'color' in properties:
                feature_vector.extend(i.avg_color)
            if 'shape' in properties:
                feature_vector.extend(i.hu)
            if 'position' in properties:
                feature_vector.extend(i.extents)
            if not feature_vector:
                logger.warning(
                    "properties parameter is not specified properly")
                return
            fvect.append(feature_vector)

        if method == "kmeans":

            # Ignore minor version numbers.
            sklearn_version = re.search(r'\d+\.\d+', __version__).group()

            if float(sklearn_version) > 0.11:
                k_means = KMeans(init='random', n_clusters=k,
                                 n_init=10).fit(fvect)
            else:
                k_means = KMeans(init='random', k=k, n_init=10).fit(fvect)
            k_clusters = [FeatureSet([]) for i in range(k)]
            for i in range(len(self)):
                k_clusters[k_means.labels_[i]].append(self[i])
            return k_clusters

        if method == "hierarchical":
            ward = Ward(n_clusters=int(sqrt(len(self)))).fit(
                fvect)  # n_clusters = sqrt(n)
            w_clusters = [FeatureSet([]) for i in range(int(sqrt(len(self))))]
            for i in range(len(self)):
                w_clusters[ward.labels_[i]].append(self[i])
            return w_clusters

    @property
    def image(self):
        if not len(self):
            return None
        return self[0].image

    @image.setter
    def image(self, image):
        for feat in self:
            feat.image = image


### ---------------------------------------------------------------------------
### ---------------------------------------------------------------------------
### ----------------------------FEATURE CLASS----------------------------------
### ---------------------------------------------------------------------------
### ---------------------------------------------------------------------------
@apply_plugins
class Feature(object):
    """
    **SUMMARY**

    The Feature object is an abstract class which real features descend from.
    Each feature object has:

    * a draw() method,
    * an image property, referencing the originating Image object
    * x and y coordinates
    * default functions for determining angle, area, mean_color, etc for
     FeatureSets
    * in the Feature class, these functions assume the feature is 1px

    """

    def __init__(self, i, at_x, at_y, points):
        # THE COVENANT IS THAT YOU PROVIDE THE POINTS IN THE SPECIFIED
        # FORMAT AND ALL OTHER VALUES SHALT FLOW
        self.x = at_x
        self.y = at_y
        self.image = i
        self.points = points

    def reassign(self, img):
        """
        **SUMMARY**

        Reassign the image of this feature and return an updated copy of the
        feature.

        **PARAMETERS**

        * *img* - the new image to which to assign the feature.

        .. Warning::
          THIS DOES NOT PERFORM A SIZE CHECK. IF YOUR NEW IMAGE IS NOT THE
          EXACT SAME SIZE YOU WILL CERTAINLY CAUSE ERRORS.

        **EXAMPLE**

        >>> img = Image("lenna")
        >>> img2 = img.invert()
        >>> l = img.find_lines()
        >>> l2 = img.reassignImage(img2)
        >>> l2.show()
        """
        ret_value = copy.deepcopy(self)
        if self.image.width != img.width or self.image.height != img.height:
            warnings.warn("DON'T REASSIGN IMAGES OF DIFFERENT SIZES")
        ret_value.image = img

        return ret_value

    @property
    def corners(self):
        return self.points

    @property
    def coordinates(self):
        """
        **SUMMARY**

        Returns the x,y position of the feature. This is usually the center
        coordinate.

        **RETURNS**

        Returns an (x,y) tuple of the position of the feature.

        **EXAMPLE**

        >>> img = Image("aerospace.png")
        >>> blobs = img.find_blobs()
        >>> for b in blobs:
        >>>    print b.coordinates

        """
        return self.x, self.y

    def draw(self, color=Color.GREEN):
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
        self.image[self.y, self.x] = color

    def show(self, color=Color.GREEN):
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
        self.draw(color)
        self.image.show()

    def distance_from(self, point=None):
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
        >>> blobs = img.find_blobs(128)
        >>> blobs[-1].distance_from(blobs[-2].coordinates())


        """
        if point is None:
            point = np.array(self.image.size_tuple) / 2
        return spsd.euclidean(point, [self.x, self.y])

    @property
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
        return self.image[self.y, self.x].tolist()

    def color_distance(self, color=None):
        """
        **SUMMARY**

        Return the euclidean color distance of the color tuple at x,y from
        a given color (default black).

        **PARAMETERS**

        * *color* - An RGB triplet to calculate from which to calculate the
         color distance.

        **RETURNS**

        A floating point color distance value.

        **EXAMPLE**

        >>> img = Image("OWS.jpg")
        >>> blobs = img.find_blobs(128)
        >>> for b in blobs:
        >>>    print b.color_distance(Color.WHITE):

        """
        if color is None:
            color = (0, 0, 0)
        return spsd.euclidean(np.array(color), np.array(self.mean_color))

    @property
    def angle(self):
        """
        **SUMMARY**

        Return the angle (theta) in degrees of the feature. The default is 0
        (horizontal).

        .. Warning::
          This is not a valid operation for all features.


        **RETURNS**

        An angle value in degrees.

        **EXAMPLE**

        >>> img = Image("OWS.jpg")
        >>> blobs = img.find_blobs(128)
        >>> for b in blobs:
        >>>    if b.get_angle() == 0:
        >>>       print "I AM HORIZONTAL."

        **TODO**

        Double check that values are being returned consistently.
        """
        return 0.0

    @lazyproperty
    def length(self):
        """
        **SUMMARY**

        This method returns the longest dimension of the feature
        (i.e max(width,height)).

        **RETURNS**

        A floating point length value.

        **EXAMPLE**

        >>> img = Image("OWS.jpg")
        >>> blobs = img.find_blobs(128)
        >>> for b in blobs:
        >>>    if b.length() > 200:
        >>>       print "OH MY! - WHAT A BIG FEATURE YOU HAVE!"
        >>>       print "---I bet you say that to all the features."

        **TODO**

        Should this be sqrt(x*x+y*y)?
        """
        return float(np.max([self.width, self.height]))

    def distance_to_nearest_edge(self):
        """
        **SUMMARY**

        This method returns the distance, in pixels, from the nearest image
        edge.

        **RETURNS**

        The integer distance to the nearest edge.

        **EXAMPLE**

        >>> img = Image("../sampleimages/EdgeTest1.png")
        >>> b = img.find_blobs()
        >>> b[0].distance_to_nearest_edge()

        """
        return np.min([self.min_x, self.min_y,
                       self.image.width - self.max_x,
                       self.image.height - self.max_y])

    def on_image_edge(self, tolerance=1):
        """
        **SUMMARY**

        This method returns True if the feature is less than `tolerance`
        pixels away from the nearest edge.

        **PARAMETERS**

        * *tolerance* - the distance in pixels at which a feature qualifies
          as being on the image edge.

        **RETURNS**

        True if the feature is on the edge, False otherwise.

        **EXAMPLE**

        >>> img = Image("../sampleimages/EdgeTest1.png")
        >>> b = img.find_blobs()
        >>> if b[0].on_image_edge():
        >>>     print "HELP! I AM ABOUT TO FALL OFF THE IMAGE"

        """
        # this has to be one to deal with blob library weirdness that goes deep
        # down to opencv
        return self.distance_to_nearest_edge() <= tolerance

    @property
    def aspect_ratio(self):
        """
        **SUMMARY**

        Return the aspect ratio of the feature, which for our purposes
        is max(width,height)/min(width,height).

        **RETURNS**

        A single floating point value of the aspect ration.

        **EXAMPLE**

        >>> img = Image("OWS.jpg")
        >>> blobs = img.find_blobs(128)
        >>> b[0].aspect_ratio

        """
        try:
            if self.width > self.height:
                return float(self.width / self.height)
            else:
                return float(self.height / self.width)
        except ZeroDivisionError:
            return 0.0

    @property
    def area(self):
        """
        **SUMMARY**

        Returns the area (number of pixels)  covered by the feature.

        **RETURNS**

        An integer area of the feature.

        **EXAMPLE**

        >>> img = Image("OWS.jpg")
        >>> blobs = img.find_blobs(128)
        >>> for b in blobs:
        >>>    if b.get_area() > 200:
        >>>       print b.get_area()

        """
        return self.width * self.height

    @property
    def width(self):
        """
        **SUMMARY**

        Returns the height of the feature.

        **RETURNS**

        An integer value for the feature's width.

        **EXAMPLE**

        >>> img = Image("OWS.jpg")
        >>> blobs = img.find_blobs(128)
        >>> for b in blobs:
        >>>    if b.width > b.height:
        >>>       print "wider than tall"
        >>>       b.draw()
        >>> img.show()

        """
        return self.max_x - self.min_x

    @property
    def height(self):
        """
        **SUMMARY**

        Returns the height of the feature.

        **RETURNS**

        An integer value of the feature's height.

        **EXAMPLE**

        >>> img = Image("OWS.jpg")
        >>> blobs = img.find_blobs(128)
        >>> for b in blobs:
        >>>    if b.width > b.height
        >>>       print "wider than tall"
        >>>       b.draw()
        >>> img.show()
        """
        return self.max_y - self.min_y

    def crop(self):
        """
        **SUMMARY**

        This function crops the source image to the location of the feature and
        returns a new simplecv image.

        **RETURNS**

        A SimpleCV image that is cropped to the feature position and size.

        **EXAMPLE**

        >>> img = Image("OWS.jpg")
        >>> blobs = img.find_blobs(128)
        >>> big = blobs[-1].crop()
        >>> big.show()

        """

        return self.image.crop(self.x, self.y, self.width, self.height,
                               centered=True)

    def __repr__(self):
        return "%s.%s at (%d,%d)" % (
            self.__class__.__module__, self.__class__.__name__, self.x, self.y)

    @property
    def bounding_box(self):
        """
        **SUMMARY**

        This property returns a rectangle which bounds the blob.

        **RETURNS**

        A list of [x, y, w, h] where (x, y) are the top left point of the
        rectangle and w, h are its width and height respectively.

        **EXAMPLE**

        >>> img = Image("OWS.jpg")
        >>> blobs = img.find_blobs(128)
        >>> print blobs[-1].bounding_box

        """
        return self.min_x, self.min_y, self.width, self.height

    @property
    def extents(self):
        """
        **SUMMARY**

        This property returns the maximum and minimum x and y values for the
        feature and returns them as a tuple.

        **RETURNS**

        A tuple of the extents of the feature. The order is (MaxX, MaxY,
        MinX, MinY).

        **EXAMPLE**

        >>> img = Image("OWS.jpg")
        >>> blobs = img.find_blobs(128)
        >>> print blobs[-1].get_extents()

        """
        return self.max_x, self.min_x, self.max_y, self.min_y

    @lazyproperty
    def min_y(self):
        """
        **SUMMARY**

        This method return the minimum y value of the bounding box of the
        the feature.

        **RETURNS**

        An integer value of the minimum y value of the feature.

        **EXAMPLE**

        >>> img = Image("OWS.jpg")
        >>> blobs = img.find_blobs(128)
        >>> print blobs[-1].min_y

        """
        return min(p[1] for p in self.points)

    @lazyproperty
    def max_y(self):
        """
        **SUMMARY**

        This property return the maximum y value of the bounding box of the
        the feature.

        **RETURNS**

        An integer value of the maximum y value of the feature.

        **EXAMPLE**

        >>> img = Image("OWS.jpg")
        >>> blobs = img.find_blobs(128)
        >>> print blobs[-1].max_y

        """
        return max(p[1] for p in self.points)

    @lazyproperty
    def min_x(self):
        """
        **SUMMARY**

        This property return the minimum x value of the bounding box of the
        the feature.

        **RETURNS**

        An integer value of the minimum x value of the feature.

        **EXAMPLE**

        >>> img = Image("OWS.jpg")
        >>> blobs = img.find_blobs(128)
        >>> print blobs[-1].min_x

        """
        return min(p[0] for p in self.points)

    @lazyproperty
    def max_x(self):
        """
        **SUMMARY**

        This property return the minimum x value of the bounding box of the
        the feature.

        **RETURNS**

        An integer value of the maxium x value of the feature.

        **EXAMPLE**

        >>> img = Image("OWS.jpg")
        >>> blobs = img.find_blobs(128)
        >>> print blobs[-1].max_x

        """
        return max(p[0] for p in self.points)

    @property
    def top_left_corner(self):
        """
        **SUMMARY**

        This property returns the top left corner of the bounding box of
        the blob as an (x,y) tuple.

        **RESULT**

        Returns a tupple of the top left corner.

        **EXAMPLE**

        >>> img = Image("OWS.jpg")
        >>> blobs = img.find_blobs(128)
        >>> print blobs[-1].top_left_corner

        """
        return self.min_x, self.min_y

    @property
    def bottom_right_corner(self):
        """
        **SUMMARY**

        This property returns the bottom right corner of the bounding box of
        the blob as an (x,y) tuple.

        **RESULT**

        Returns a tupple of the bottom right corner.

        **EXAMPLE**

        >>> img = Image("OWS.jpg")
        >>> blobs = img.find_blobs(128)
        >>> print blobs[-1].bottom_right_corner

        """
        return self.max_x, self.max_y

    @property
    def bottom_left_corner(self):
        """
        **SUMMARY**

        This property returns the bottom left corner of the bounding box of
        the blob as an (x,y) tuple.

        **RESULT**

        Returns a tupple of the bottom left corner.

        **EXAMPLE**

        >>> img = Image("OWS.jpg")
        >>> blobs = img.find_blobs(128)
        >>> print blobs[-1].bottom_left_corner

        """
        return self.min_x, self.max_y

    @property
    def top_right_corner(self):
        """
        **SUMMARY**

        This property returns the top right corner of the bounding box of
        the blob as an (x,y) tuple.

        **RESULT**

        Returns a tupple of the top right  corner.

        **EXAMPLE**

        >>> img = Image("OWS.jpg")
        >>> blobs = img.find_blobs(128)
        >>> print blobs[-1].top_right_corner

        """
        return self.max_x, self.min_y

    def above(self, object):
        """
        **SUMMARY**

        Return true if the feature is above the object, where object can be a
        bounding box, bounding circle, a list of tuples in a closed polygon, or
        any other featutres.

        **PARAMETERS**

        * *object*

          * A bounding box - of the form (x,y,w,h) where x,y is the upper left
           corner
          * A bounding circle of the form (x,y,r)
          * A list of x,y tuples defining a closed polygon
           e.g. ((x,y),(x,y),....)
          * Any two dimensional feature (e.g. blobs, circle ...)

        **RETURNS**

        Returns a Boolean, True if the feature is above the object, False
        otherwise.

        **EXAMPLE**

        >>> img = Image("Lenna")
        >>> blobs = img.find_blobs()
        >>> b = blobs[0]
        >>> if blobs[-1].above(b):
        >>>    print "above the biggest blob"

        """
        if isinstance(object, Feature):
            return self.max_y < object.min_y
        elif isinstance(object, tuple) or isinstance(object, np.ndarray):
            return self.max_y < object[1]
        elif isinstance(object, float) or isinstance(object, int):
            return self.max_y < object
        else:
            logger.warning(
                "SimpleCV did not recognize the input type to feature.above()."
                " This method only takes another feature, an (x,y) tuple, or a"
                " ndarray type.")
            return None

    def below(self, object):
        """
        **SUMMARY**

        Return true if the feature is below the object, where object can be a
        bounding box, bounding circle, a list of tuples in a closed polygon, or
        any other featutres.

        **PARAMETERS**

        * *object*

          * A bounding box - of the form (x,y,w,h) where x,y is the upper left
           corner
          * A bounding circle of the form (x,y,r)
          * A list of x,y tuples defining a closed polygon
           e.g. ((x,y),(x,y),....)
          * Any two dimensional feature (e.g. blobs, circle ...)

        **RETURNS**

        Returns a Boolean, True if the feature is below the object, False
        otherwise.

        **EXAMPLE**

        >>> img = Image("Lenna")
        >>> blobs = img.find_blobs()
        >>> b = blobs[0]
        >>> if blobs[-1].below(b):
        >>>    print "above the biggest blob"

        """
        if isinstance(object, Feature):
            return self.min_y > object.max_y
        elif isinstance(object, tuple) or isinstance(object, np.ndarray):
            return self.min_y > object[1]
        elif isinstance(object, float) or isinstance(object, int):
            return self.min_y > object
        else:
            logger.warning(
                "SimpleCV did not recognize the input type to feature.below()."
                " This method only takes another feature, an (x,y) tuple, or a"
                " ndarray type.")
            return None

    def right(self, object):
        """
        **SUMMARY**

        Return true if the feature is to the right object, where object can be
        a bounding box, bounding circle, a list of tuples in a closed polygon,
        or any other featutres.

        **PARAMETERS**

        * *object*

          * A bounding box - of the form (x,y,w,h) where x,y is the upper left
           corner
          * A bounding circle of the form (x,y,r)
          * A list of x,y tuples defining a closed polygon
           e.g. ((x,y),(x,y),....)
          * Any two dimensional feature (e.g. blobs, circle ...)

        **RETURNS**

        Returns a Boolean, True if the feature is to the right object, False
        otherwise.

        **EXAMPLE**

        >>> img = Image("Lenna")
        >>> blobs = img.find_blobs()
        >>> b = blobs[0]
        >>> if blobs[-1].right(b):
        >>>    print "right of the the blob"

        """
        if isinstance(object, Feature):
            return self.min_x > object.max_x
        elif isinstance(object, tuple) or isinstance(object, np.ndarray):
            return self.min_x > object[0]
        elif isinstance(object, float) or isinstance(object, int):
            return self.min_x > object
        else:
            logger.warning(
                "SimpleCV did not recognize the input type to feature.right()."
                " This method only takes another feature, an (x,y) tuple, or a"
                " ndarray type.")
            return None

    def left(self, object):
        """
        **SUMMARY**

        Return true if the feature is to the left of  the object, where object
        can be a bounding box, bounding circle, a list of tuples in a closed
        polygon, or any other featutres.

        **PARAMETERS**

        * *object*

          * A bounding box - of the form (x,y,w,h) where x,y is the upper left
           corner
          * A bounding circle of the form (x,y,r)
          * A list of x,y tuples defining a closed polygon
           e.g. ((x,y),(x,y),....)
          * Any two dimensional feature (e.g. blobs, circle ...)

        **RETURNS**

        Returns a Boolean, True if the feature is to the left of  the object,
        False otherwise.

        **EXAMPLE**

        >>> img = Image("Lenna")
        >>> blobs = img.find_blobs()
        >>> b = blobs[0]
        >>> if blobs[-1].left(b):
        >>>    print "left of  the biggest blob"


        """
        if isinstance(object, Feature):
            return self.max_x < object.min_x
        elif isinstance(object, tuple) or isinstance(object, np.ndarray):
            return self.max_x < object[0]
        elif isinstance(object, float) or isinstance(object, int):
            return self.max_x < object
        else:
            logger.warning(
                "SimpleCV did not recognize the input type to feature.left(). "
                "This method only takes another feature, an (x,y) tuple, or a "
                "ndarray type.")
            return None

    def contains(self, other):
        """
        **SUMMARY**

        Return true if the feature contains  the object, where object can be a
        bounding box, bounding circle, a list of tuples in a closed polygon, or
        any other featutres.

        **PARAMETERS**

        * *object*

          * A bounding box - of the form (x,y,w,h) where x,y is the upper left
           corner
          * A bounding circle of the form (x,y,r)
          * A list of x,y tuples defining a closed polygon
           e.g. ((x,y),(x,y),....)
          * Any two dimensional feature (e.g. blobs, circle ...)

        **RETURNS**

        Returns a Boolean, True if the feature contains the object, False
        otherwise.

        **EXAMPLE**

        >>> img = Image("Lenna")
        >>> blobs = img.find_blobs()
        >>> b = blobs[0]
        >>> if blobs[-1].contains(b):
        >>>    print "this blob is contained in the biggest blob"

        **NOTE**

        This currently performs a bounding box test, not a full polygon test
        for speed.

        """
        ret_value = False
        bounds = self.points

        if isinstance(other, Feature):  # A feature
            ret_value = True
            # this isn't completely correct
            # only tests if points lie in poly, not edges.
            for point in other.points:
                point2 = (int(point[0]), int(point[1]))
                ret_value = self._point_inside_polygon(point2, bounds)
                if not ret_value:
                    break
        # a single point
        elif ((isinstance(other, tuple) and len(other) == 2) or (
                isinstance(other, np.ndarray) and other.shape[0] == 2)):
            ret_value = self._point_inside_polygon(other, bounds)

        elif isinstance(other, tuple) and len(other) == 3:  # A circle
            #assume we are in x,y, r format
            ret_value = True
            for point in bounds:
                test = (other[0] - point[0]) * (other[0] - point[0])
                test += (other[1] - point[1]) * (other[1] - point[1])
                if test < other[2] * other[2]:
                    ret_value = False
                    break

        elif isinstance(other, tuple) and len(other) == 4 and (
                isinstance(other[0], float) or isinstance(other[0], int)):
            ret_value = (self.max_x <= other[0] + other[2] and
                         self.min_x >= other[0] and
                         self.max_y <= other[1] + other[3] and
                         self.min_y >= other[1])
        elif isinstance(other, list) and len(other) >= 4:
            # an arbitrary polygon
            ret_value = True
            for point in other:
                test = self._point_inside_polygon(point, bounds)
                if not test:
                    ret_value = False
                    break
        else:
            logger.warning(
                "SimpleCV did not recognize the input type to "
                "features.contains. This method only takes another blob, an "
                "(x,y) tuple, or a ndarray type.")

        return ret_value

    def overlaps(self, other):
        """
        **SUMMARY**

        Return true if the feature overlaps the object, where object can be a
        bounding box, bounding circle, a list of tuples in a closed polygon, or
        any other featutres.

        **PARAMETERS**

        * *object*

          * A bounding box - of the form (x,y,w,h) where x,y is the upper left
           corner
          * A bounding circle of the form (x,y,r)
          * A list of x,y tuples defining a closed polygon
           e.g. ((x,y),(x,y),....)
          * Any two dimensional feature (e.g. blobs, circle ...)

        **RETURNS**

        Returns a Boolean, True if the feature overlaps  object, False
        otherwise.

        **EXAMPLE**

        >>> img = Image("Lenna")
        >>> blobs = img.find_blobs()
        >>> b = blobs[0]
        >>> if blobs[-1].overlaps(b):
        >>>    print "This blob overlaps the biggest blob"

        Returns true if this blob contains at least one point, part of a
        collection of points, or any part of a blob.

        **NOTE**

        This currently performs a bounding box test, not a full polygon test
        for speed.

       """
        ret_value = False
        bounds = self.points

        if isinstance(other, Feature):  # A feature
            ret_value = True
            # this isn't completely correct
            # only tests if points lie in poly, not edges.
            for point in other.points:
                ret_value = self._point_inside_polygon(point, bounds)
                if ret_value:
                    break

        elif (isinstance(other, tuple) and len(other) == 2) or (
                isinstance(other, np.ndarray) and other.shape[0] == 2):
            ret_value = self._point_inside_polygon(other, bounds)

        elif isinstance(other, tuple) and len(other) == 3 and not isinstance(
                other[0], tuple):  # A circle
            #assume we are in x,y, r format
            ret_value = False
            for point in bounds:
                test = (other[0] - point[0]) * (other[0] - point[0])
                test += (other[1] - point[1]) * (other[1] - point[1])
                if test < other[2] * other[2]:
                    ret_value = True
                    break

        elif isinstance(other, tuple) and len(other) == 4 and (
                isinstance(other[0], float) or isinstance(other[0], int)):
            ret_value = (self.contains(
                (other[0], other[1])) or  # see if we contain any corner
                self.contains((other[0] + other[2], other[1])) or
                self.contains((other[0], other[1] + other[3])) or
                self.contains(
                    (other[0] + other[2], other[1] + other[3])))
        elif isinstance(other, list) and len(other) >= 3:
            # an arbitrary polygon
            ret_value = False
            for point in other:
                test = self._point_inside_polygon(point, bounds)
                if test:
                    ret_value = True
                    break
        else:
            logger.warning(
                "SimpleCV did not recognize the input type to "
                "features.overlaps. This method only takes another blob, an "
                "(x,y) tuple, or a ndarray type.")

        return ret_value

    def is_contained_within(self, other):
        """
        **SUMMARY**

        Return true if the feature is contained withing  the object other,
        where other can be a bounding box, bounding circle, a list of tuples
        in a closed polygon, or any other featutres.

        **PARAMETERS**

        * *other*

          * A bounding box - of the form (x,y,w,h) where x,y is the upper left
           corner
          * A bounding circle of the form (x,y,r)
          * A list of x,y tuples defining a closed polygon
           e.g. ((x,y),(x,y),....)
          * Any two dimensional feature (e.g. blobs, circle ...)

        **RETURNS**

        Returns a Boolean, True if the feature is above the object,
        False otherwise.

        **EXAMPLE**

        >>> img = Image("Lenna")
        >>> blobs = img.find_blobs()
        >>> b = blobs[0]
        >>> if blobs[-1].is_contained_within(b):
        >>>    print "inside the blob"

        """
        ret_value = True
        bounds = self.points

        if isinstance(other, Feature):
            # another feature do the containment test
            ret_value = other.contains(self)
        elif isinstance(other, tuple) and len(other) == 3:  # a circle
            #assume we are in x, y, r format
            for point in bounds:
                test = (other[0] - point[0]) * (other[0] - point[0])
                test += (other[1] - point[1]) * (other[1] - point[1])
                if test > other[2] * other[2]:  # radius squared:
                    ret_value = False
                    break

        elif isinstance(other, tuple) and len(other) == 4 \
                and (isinstance(other[0], float) or isinstance(other[0], int)):
            # we assume a tuple of four is (x,y,w,h)
            ret_value = (self.max_x <= other[0] + other[2] and
                         self.min_x >= other[0] and
                         self.max_y <= other[1] + other[3] and
                         self.min_y >= other[1])

        # an arbitrary polygon
        elif isinstance(other, list) and len(other) > 2:
            #everything else ....
            ret_value = True
            for point in bounds:
                test = self._point_inside_polygon(point, other)
                if not test:
                    ret_value = False
                    break

        else:
            logger.warning(
                "SimpleCV did not recognize the input type to "
                "features.contains. This method only takes another blob, an "
                "(x,y) tuple, or a ndarray type.")
            ret_value = False
        return ret_value

    @staticmethod
    def _point_inside_polygon(point, polygon):
        """
        returns true if tuple point (x,y) is inside polygon of the form
        ((a,b),(c,d),...,(a,b)) the polygon should be closed

        """
        # TODO: consider using 'shapely' lib
        # http://stackoverflow.com/questions/21612976/point-inside-polygon
        if len(polygon) < 3:
            logger.warning(
                "feature._point_inside_polygon - this is not a valid polygon")
            return False

        if not isinstance(polygon, list):
            logger.warning(
                "feature._point_inside_polygon - this is not a valid polygon")
            return False

        counter = 0
        ret_value = True
        #print "point: " + str(point)
        poly = copy.deepcopy(polygon)
        poly.append(polygon[0])
        poly_len = len(poly)
        p1 = poly[0]
        for i in range(1, poly_len + 1):
            p2 = poly[i % poly_len]
            if point[1] > np.min((p1[1], p2[1])):
                if point[1] <= np.max((p1[1], p2[1])):
                    if point[0] <= np.max((p1[0], p2[0])):
                        if p1[0] == p2[0]:
                            counter += 1
                        elif p1[1] != p2[1]:
                            test = float((point[1] - p1[1]) * (p2[0] - p1[0]))\
                                / float(((p2[1] - p1[1]) + p1[0]))
                            if point[0] <= test:
                                counter += 1
            p1 = p2

        if counter % 2 == 0:
            ret_value = False
            return ret_value
        return ret_value

    @classmethod
    def find(cls, img, method="szeliski", threshold=1000):
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
        >>> fpoints = img.find(Feature, "harris", 2000)
        >>> for f in fpoints:
        ...     f.draw()
        >>> img.show()

        **SEE ALSO**

        :py:meth:`draw_keypoint_matches`
        :py:meth:`find_keypoints`
        :py:meth:`find_keypoint_match`

        """
        if method not in ["harris", "szeliski"]:
            raise ValueError("Invalid method: {}.".format(method))

        img_array = img.to_gray()
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
