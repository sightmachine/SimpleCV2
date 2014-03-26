import sys
import warnings

import cv2
import numpy as np

from simplecv.base import logger
from simplecv.features.blob import Blob
from simplecv.features.features import FeatureSet
from simplecv.image_class import Image


class BlobMaker(object):
    """
    Blob maker encapsulates all of the get_contour extraction process and data,
    so it can be used inside the image class, or extended and used outside the
    image class. The general idea is that the blob maker provides the utilites
    that one would use for blob extraction. Later implementations may include
    tracking and other features.
    """

    def __init__(self):
        pass

    def extract_using_model(self, img, colormodel, minsize=10, maxsize=0):
        """
        Extract blobs using a color model
        img        - The input image
        colormodel - The color model to use.
        minsize    - The minimum size of the returned features.
        maxsize    - The maximum size of the returned features 0=uses the
        default value.

        Parameters:
            img - Image
            colormodel - ColorModel object
            minsize - Int
            maxsize - Int
        """
        if maxsize <= 0:
            maxsize = img.width * img.height
        gray = colormodel.threshold(img)
        blobs = self.extract_from_binary(gray, img, minsize, maxsize)
        ret_value = sorted(blobs, key=lambda x: x.area, reverse=True)
        return FeatureSet(ret_value)

    def extract(self, img, threshval=127, minsize=10, maxsize=0,
                threshblocksize=3, threshconstant=5):
        """
        This method performs a threshold operation on the input image and then
        extracts and returns the blobs.
        img       - The input image (color or b&w)
        threshval - The threshold value for the binarize operation.
         If threshval = -1 adaptive thresholding is used
        minsize   - The minimum blob size in pixels.
        maxsize   - The maximum blob size in pixels. 0=uses the default value.
        threshblocksize - The adaptive threhold block size.
        threshconstant  - The minimum to subtract off the adaptive threshold
        """
        if maxsize <= 0:
            maxsize = img.width * img.height

        #create a single channel image, thresholded to parameters

        blobs = self.extract_from_binary(
            img.binarize(threshval, 255, threshblocksize,
                         threshconstant).invert(), img, minsize, maxsize)
        ret_value = sorted(blobs, key=lambda x: x.area, reverse=True)
        return FeatureSet(ret_value)

    def extract_from_binary(self, binary_img, color_img, minsize=5, maxsize=-1,
                            appx_level=3):
        """
        This method performs blob extraction given a binary source image that
        is used to get the blob images, and a color source image.
        binary_img - The binary image with the blobs.
        color_img - The color image.
        minsize  - The minimum size of the blobs in pixels.
        maxsize  - The maximum blob size in pixels.
        * *appx_level* - The blob approximation level - an integer for the
        maximum distance between the true edge and the approximation edge -
        lower numbers yield better approximation.
        """
        #If you hit this recursion limit may god have mercy on your soul.
        #If you really are having problems set the value higher, but this means
        # you have over 10,000,000 blobs in your image.
        sys.setrecursionlimit(5000)
        #h_next moves to the next external get_contour
        #v_next() moves to the next internal get_contour
        if maxsize <= 0:
            maxsize = color_img.width * color_img.height

        ret_value = []
        ptest = (4 * 255.0) / (binary_img.width * binary_img.height)
        test = binary_img.mean_color()
        if not test:
            return FeatureSet([])
        if type(test) == tuple and len(test) == 3:
            if (test[0] == 0.00 and test[1] == 0.00 and test[2] == 0.00) or \
               (test[0] <= ptest and test[1] <= ptest and test[2] <= ptest):
                return FeatureSet([])
        else:
            if test == 0.00 or test <= ptest:
                return FeatureSet([])

        # There are a couple of weird corner cases with the opencv
        # connect components libraries - when you try to find contours
        # in an all black image, or an image with a single white pixel
        # that sits on the edge of an image the whole thing explodes
        # this check catches those bugs. -KAS
        # Also I am submitting a bug report to Willow Garage - please bare with
        # us.

        # val if two pixels are white
        #ptest = (4 * 255.0) / (binary_img.width * binary_img.height)
        #if test[0] <= ptest and test[1] <= ptest and test[2] <= ptest:
        #    return ret_value

        seq, _ = cv2.findContours(binary_img.get_gray_ndarray(),
                                  cv2.RETR_TREE,
                                  cv2.CHAIN_APPROX_SIMPLE)
        if not list(seq):
            warnings.warn("Unable to find Blobs. Retuning Empty FeatureSet.")
            return FeatureSet([])
        try:
            ret_value = self._extract_from_binary(seq, False, color_img,
                                                  minsize, maxsize, appx_level)
        except RuntimeError:
            logger.warning(
                "You exceeded the recursion limit. This means you probably "
                "have too many blobs in your image. We suggest you do some "
                "morphological operations (erode/dilate) to reduce the number "
                "of blobs in your image. This function was designed to max out"
                " at about 5000 blobs per image.")
        del seq
        return FeatureSet(ret_value)

    def _extract_from_binary(self, seq, isa_hole, color_img, minsize, maxsize,
                             appx_level):
        """
        The recursive entry point for the blob extraction. The blobs and holes
        are presented as a tree and we traverse up and across the tree.
        """
        ret_value = []

        if seq is None:
            return ret_value

        next_layer_down = []
        while True:
            # if we aren't a hole then we are an object,
            # so get and return our features
            if not isa_hole:
                temp = self._extract_data(seq, color_img, minsize, maxsize,
                                          appx_level)
                if temp is not None:
                    ret_value.append(temp)

            next_layer = seq.v_next()

            if next_layer is not None:
                next_layer_down.append(next_layer)

            seq = seq.h_next()

            if seq is None:
                break

        for next_layer in next_layer_down:
            ret_value += self._extract_from_binary(next_layer, not isa_hole,
                                                   color_img, minsize, maxsize,
                                                   appx_level)

        return ret_value

    def _extract_data(self, seq, color, minsize, maxsize, appx_level):
        """
        Extract the bulk of the data from a give blob. If the blob's are is too
        large or too small the method returns none.
        """
        if seq is None or not len(seq):
            return None
        area = cv2.contourArea(seq)
        if area < minsize or area > maxsize:
            return None

        ret_value = Blob()
        ret_value.image = color
        ret_value.area = area

        ret_value.min_rectangle = cv2.minAreaRect(seq)
        bbr = cv2.boundingRect(seq)
        ret_value.x = bbr[0] + (bbr[2] / 2)
        ret_value.y = bbr[1] + (bbr[3] / 2)
        ret_value.perimeter = cv2.arcLength(seq, closed=-1)
        if seq is not None:  # KAS
            ret_value.contour = list(seq)

            if ret_value.contour is not None:
                ret_value.contour_appx = []
                appx = cv2.approxPolyDP(
                    np.array([ret_value.contour], 'float32'), appx_level,
                    True)
                for p in appx:
                    ret_value.contour_appx.append(
                        (int(p[0][0]), int(p[0][1])))

        # so this is a bit hacky....

        # For blobs that live right on the edge of the image OpenCV reports the
        # position and width height as being one over for the true position.
        # E.g. if a blob is at (0,0) OpenCV reports its position as (1,1).
        # Likewise the width and height for the other corners is reported as
        # being one less than the width and height. This is a known bug.

        xx = bbr[0]
        yy = bbr[1]
        ww = bbr[2]
        hh = bbr[3]
        ret_value.points = [(xx, yy), (xx + ww, yy), (xx + ww, yy + hh),
                            (xx, yy + hh)]
        ret_value._update_extents()
        chull = cv2.convexHull(seq, return_points=1)
        ret_value.convex_hull = list(chull)
        # KAS -- FLAG FOR REPLACE 6/6/2012
        #get_hull_mask = self._get_hull_mask(chull)

        # KAS -- FLAG FOR REPLACE 6/6/2012
        # ret_value.hull_img = self._get_blob_as_image(chull,bb,
        #                                              color.get_bitmap(),
        #                                              get_hull_mask)

        # KAS -- FLAG FOR REPLACE 6/6/2012
        #ret_value.hull_mask = Image(get_hull_mask)

        del chull

        moments = cv2.moments(seq)

        #This is a hack for a python wrapper bug that was missing
        #the constants required from the ctype
        ret_value.m00 = area
        ret_value.m10 = moments.get('m10')
        ret_value.m01 = moments.get('m01')
        ret_value.m11 = moments.get('m11')
        ret_value.m20 = moments.get('m20')
        ret_value.m02 = moments.get('m02')
        ret_value.m21 = moments.get('m21')
        ret_value.m12 = moments.get('m12')

        ret_value.hu = cv2.HuMoments(moments)

        # KAS -- FLAG FOR REPLACE 6/6/2012
        mask = self._get_mask(seq)
        #ret_value.mask = Image(mask)

        ret_value.avg_color = self._get_avg(color.get_ndarray(), bbr, mask)
        ret_value.avg_color = ret_value.avg_color[0:3]
        #ret_value.avg_color = self._get_avg(color.get_bitmap(),
        #                                    ret_value.bounding_box, mask)
        #ret_value.avg_color = ret_value.avg_color[0:3]

        # KAS -- FLAG FOR REPLACE 6/6/2012
        #ret_value.img = self._get_blob_as_image(seq,bb,color.get_bitmap(),
        #                                        mask)

        ret_value.hole_contour = self._get_holes(seq)
        ret_value.aspect_ratio = ret_value.min_rectangle[1][0] / \
            ret_value.min_rectangle[1][1]

        return ret_value

    @staticmethod
    def _get_holes(seq):
        """
        This method returns the holes associated with a blob as a list of
        tuples.
        """
        ret_value = None
        holes = seq.v_next()
        if holes is not None:
            ret_value = [list(holes)]
            while holes.h_next() is not None:
                holes = holes.h_next()
                temp = list(holes)
                if len(temp) >= 3:  # exclude single pixel holes
                    ret_value.append(temp)
        return ret_value

    @staticmethod
    def _get_mask(seq):
        """
        Return a binary image of a particular get_contour sequence.
        """
        bbr = cv2.boundingRect(seq)
        mask = np.zeros((bbr[2], bbr[3]), dtype='uint8')
        cv2.drawContours(mask, seq, 255, 0, thickness=-1, maxLevel=0,
                         offset=(-1 * bbr[0], -1 * bbr[1]))
        holes = seq.v_next()
        if holes is not None:
            cv2.drawContours(mask, holes, 0, 255, thickness=-1, maxLevel=0,
                             offset=(-1 * bbr[0], -1 * bbr[1]))
            while holes.h_next() is not None:
                holes = holes.h_next()
                if holes is not None:
                    cv2.drawContours(mask, holes, 0, 255, 0,
                                     thickness=-1, maxLevel=0,
                                     offset=(-1 * bbr[0], -1 * bbr[1]))
        return mask

    @staticmethod
    def _get_hull_mask(hull):
        """
        Return a mask of the convex hull of a blob.
        """
        bbr = cv2.boundingRect(hull)
        mask = np.zeros((bbr[2], bbr[3]), dtype='uint8')
        cv2.drawContours(mask, hull, 255, 0, thickness=-1, maxLevel=0,
                         offset=(-1 * bbr[0], -1 * bbr[1]))
        return mask

    @staticmethod
    def _get_avg(colorbitmap, bb, mask):
        """
        Calculate the average color of a blob given the mask.
        """
        #FIXME: SetImageROI and ResetImageROI deprecated?
        #bloob field from colorbitmap should be extracted?
        #cv.SetImageROI(colorbitmap, bb)
        #may need the offset parameter
        avg = cv2.mean(colorbitmap, mask)
        #cv.ResetImageROI(colorbitmap)
        return avg

    @staticmethod
    def _get_blob_as_image(seq, bbr, colorbitmap, mask):
        """
        Return an image that contains just pixels defined by the blob sequence.
        """
        # FIXME: rewrite to cv2.copyMakeborder
        #cv.SetImageROI(colorbitmap, bb)
        output_img = mask = np.zeros((bbr[2], bbr[3]), dtype='uint8')
        #cv.Zero(output_img)
        #cv.Copy(colorbitmap, output_img, mask)
        #cv.ResetImageROI(colorbitmap)
        return Image(output_img)
