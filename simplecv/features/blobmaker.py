import sys
import warnings

import numpy as np

from simplecv.base import cv, logger
from simplecv.features.blob import Blob
from simplecv.features.features import FeatureSet
from simplecv.image_class import Image


class BlobMaker(object):
    """
    Blob maker encapsulates all of the contour extraction process and data, so
    it can be used inside the image class, or extended and used outside the
    image class. The general idea is that the blob maker provides the utilites
    that one would use for blob extraction. Later implementations may include
    tracking and other features.
    """
    mMemStorage = None

    def __init__(self):
        self.mMemStorage = cv.CreateMemStorage()

    def extractUsingModel(self, img, colormodel, minsize=10, maxsize=0):
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
        blobs = self.extractFromBinary(gray, img, minsize, maxsize)
        ret_value = sorted(blobs, key=lambda x: x.mArea, reverse=True)
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

        blobs = self.extractFromBinary(
            img.binarize(threshval, 255, threshblocksize,
                         threshconstant).invert(), img, minsize, maxsize)
        ret_value = sorted(blobs, key=lambda x: x.mArea, reverse=True)
        return FeatureSet(ret_value)

    def extractFromBinary(self, binaryImg, colorImg, minsize=5, maxsize=-1,
                          appx_level=3):
        """
        This method performs blob extraction given a binary source image that
        is used to get the blob images, and a color source image.
        binarymg - The binary image with the blobs.
        colorImg - The color image.
        minSize  - The minimum size of the blobs in pixels.
        maxSize  - The maximum blob size in pixels.
        * *appx_level* - The blob approximation level - an integer for the
        maximum distance between the true edge and the approximation edge -
        lower numbers yield better approximation.
        """
        #If you hit this recursion limit may god have mercy on your soul.
        #If you really are having problems set the value higher, but this means
        # you have over 10,000,000 blobs in your image.
        sys.setrecursionlimit(5000)
        #h_next moves to the next external contour
        #v_next() moves to the next internal contour
        if maxsize <= 0:
            maxsize = colorImg.width * colorImg.height

        ret_value = []
        test = binaryImg.mean_color()
        if test[0] == 0.00 and test[1] == 0.00 and test[2] == 0.00:
            return FeatureSet(ret_value)

        # There are a couple of weird corner cases with the opencv
        # connect components libraries - when you try to find contours
        # in an all black image, or an image with a single white pixel
        # that sits on the edge of an image the whole thing explodes
        # this check catches those bugs. -KAS
        # Also I am submitting a bug report to Willow Garage - please bare with
        # us.
        ptest = (4 * 255.0) / (
            binaryImg.width * binaryImg.height)  # val if two pixels are white
        if test[0] <= ptest and test[1] <= ptest and test[2] <= ptest:
            return ret_value

        seq = cv.FindContours(binaryImg._get_grayscale_bitmap(),
                              self.mMemStorage, cv.CV_RETR_TREE,
                              cv.CV_CHAIN_APPROX_SIMPLE)
        if not list(seq):
            warnings.warn("Unable to find Blobs. Retuning Empty FeatureSet.")
            return FeatureSet([])
        try:
            ret_value = self._extractFromBinary(seq, False, colorImg, minsize,
                                                maxsize, appx_level)
        except RuntimeError:
            logger.warning(
                "You exceeded the recursion limit. This means you probably "
                "have too many blobs in your image. We suggest you do some "
                "morphological operations (erode/dilate) to reduce the number "
                "of blobs in your image. This function was designed to max out"
                " at about 5000 blobs per image.")
        except Exception:
            logger.warning(
                "SimpleCV Find Blobs Failed - This could be an OpenCV python "
                "binding issue")
        del seq
        return FeatureSet(ret_value)

    def _extractFromBinary(self, seq, isaHole, colorImg, minsize, maxsize,
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
            if not isaHole:
                temp = self._extractData(seq, colorImg, minsize, maxsize,
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
            ret_value += self._extractFromBinary(next_layer, not isaHole,
                                                 colorImg, minsize, maxsize,
                                                 appx_level)

        return ret_value

    def _extractData(self, seq, color, minsize, maxsize, appx_level):
        """
        Extract the bulk of the data from a give blob. If the blob's are is too
        large or too small the method returns none.
        """
        if seq is None or not len(seq):
            return None
        area = cv.ContourArea(seq)
        if area < minsize or area > maxsize:
            return None

        ret_value = Blob()
        ret_value.image = color
        ret_value.mArea = area

        ret_value.mMinRectangle = cv.MinAreaRect2(seq)
        bbr = cv.BoundingRect(seq)
        ret_value.x = bbr[0] + (bbr[2] / 2)
        ret_value.y = bbr[1] + (bbr[3] / 2)
        ret_value.mPerimeter = cv.ArcLength(seq)
        if seq is not None:  # KAS
            ret_value.mContour = list(seq)
            try:
                import cv2

                if ret_value.mContour is not None:
                    ret_value.mContourAppx = []
                    appx = cv2.approxPolyDP(
                        np.array([ret_value.mContour], 'float32'), appx_level,
                        True)
                    for p in appx:
                        ret_value.mContourAppx.append(
                            (int(p[0][0]), int(p[0][1])))
            except:
                pass

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
        ret_value._updateExtents()
        chull = cv.ConvexHull2(seq, cv.CreateMemStorage(), return_points=1)
        ret_value.mConvexHull = list(chull)
        # KAS -- FLAG FOR REPLACE 6/6/2012
        #hullMask = self._getHullMask(chull,bb)

        # KAS -- FLAG FOR REPLACE 6/6/2012
        # ret_value.mHullImg = self._getBlobAsImage(chull,bb,color.get_bitmap(),
        # hullMask)

        # KAS -- FLAG FOR REPLACE 6/6/2012
        #ret_value.mHullMask = Image(hullMask)

        del chull

        moments = cv.Moments(seq)

        #This is a hack for a python wrapper bug that was missing
        #the constants required from the ctype
        ret_value.m00 = area
        try:
            ret_value.m10 = moments.m10
            ret_value.m01 = moments.m01
            ret_value.m11 = moments.m11
            ret_value.m20 = moments.m20
            ret_value.m02 = moments.m02
            ret_value.m21 = moments.m21
            ret_value.m12 = moments.m12
        except:
            ret_value.m10 = cv.GetSpatialMoment(moments, 1, 0)
            ret_value.m01 = cv.GetSpatialMoment(moments, 0, 1)
            ret_value.m11 = cv.GetSpatialMoment(moments, 1, 1)
            ret_value.m20 = cv.GetSpatialMoment(moments, 2, 0)
            ret_value.m02 = cv.GetSpatialMoment(moments, 0, 2)
            ret_value.m21 = cv.GetSpatialMoment(moments, 2, 1)
            ret_value.m12 = cv.GetSpatialMoment(moments, 1, 2)

        ret_value.mHu = cv.GetHuMoments(moments)

        # KAS -- FLAG FOR REPLACE 6/6/2012
        mask = self._getMask(seq, bbr)
        #ret_value.mMask = Image(mask)

        ret_value.mAvgColor = self._getAvg(color.get_bitmap(), bbr, mask)
        ret_value.mAvgColor = ret_value.mAvgColor[0:3]
        #ret_value.mAvgColor = self._getAvg(color.get_bitmap(),
        #                                   ret_value.mBoundingBox, mask)
        #ret_value.mAvgColor = ret_value.mAvgColor[0:3]

        # KAS -- FLAG FOR REPLACE 6/6/2012
        #ret_value.mImg = self._getBlobAsImage(seq,bb,color.get_bitmap(),mask)

        ret_value.mHoleContour = self._getHoles(seq)
        ret_value.mAspectRatio = ret_value.mMinRectangle[1][0] / \
            ret_value.mMinRectangle[1][1]

        return ret_value

    @staticmethod
    def _getHoles(seq):
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
    def _getMask(seq, bb):
        """
        Return a binary image of a particular contour sequence.
        """
        #bb = cv.BoundingRect(seq)
        mask = cv.CreateImage((bb[2], bb[3]), cv.IPL_DEPTH_8U, 1)
        cv.Zero(mask)
        cv.DrawContours(mask, seq, 255, 0, 0, thickness=-1,
                        offset=(-1 * bb[0], -1 * bb[1]))
        holes = seq.v_next()
        if holes is not None:
            cv.DrawContours(mask, holes, 0, 255, 0, thickness=-1,
                            offset=(-1 * bb[0], -1 * bb[1]))
            while holes.h_next() is not None:
                holes = holes.h_next()
                if holes is not None:
                    cv.DrawContours(mask, holes, 0, 255, 0, thickness=-1,
                                    offset=(-1 * bb[0], -1 * bb[1]))
        return mask

    @staticmethod
    def _getHullMask(hull, bb):
        """
        Return a mask of the convex hull of a blob.
        """
        bb = cv.BoundingRect(hull)
        mask = cv.CreateImage((bb[2], bb[3]), cv.IPL_DEPTH_8U, 1)
        cv.Zero(mask)
        cv.DrawContours(mask, hull, 255, 0, 0, thickness=-1,
                        offset=(-1 * bb[0], -1 * bb[1]))
        return mask

    @staticmethod
    def _getAvg(colorbitmap, bb, mask):
        """
        Calculate the average color of a blob given the mask.
        """
        cv.SetImageROI(colorbitmap, bb)
        #may need the offset parameter
        avg = cv.Avg(colorbitmap, mask)
        cv.ResetImageROI(colorbitmap)
        return avg

    @staticmethod
    def _getBlobAsImage(seq, bb, colorbitmap, mask):
        """
        Return an image that contains just pixels defined by the blob sequence.
        """
        cv.SetImageROI(colorbitmap, bb)
        output_img = cv.CreateImage((bb[2], bb[3]), cv.IPL_DEPTH_8U, 3)
        cv.Zero(output_img)
        cv.Copy(colorbitmap, output_img, mask)
        cv.ResetImageROI(colorbitmap)
        return Image(output_img)
