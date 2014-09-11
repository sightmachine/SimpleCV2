import warnings

import cv2
import numpy as np

from simplecv.base import force_update_lazyproperties
from simplecv.features.features import FeatureSet
from simplecv.factory import Factory


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

    def extract(self, img, threshold=127, minsize=10, maxsize=0,
                threshblocksize=3, threshconstant=5):
        """
        This method performs a threshold operation on the input image and then
        extracts and returns the blobs.
        img       - The input image (color or b&w)
        threshold - The threshold value for the binarize operation.
         If threshold = -1 adaptive thresholding is used
        minsize   - The minimum blob size in pixels.
        maxsize   - The maximum blob size in pixels. 0=uses the default value.
        threshblocksize - The adaptive threhold block size.
        threshconstant  - The minimum to subtract off the adaptive threshold
        """
        if maxsize <= 0:
            maxsize = img.width * img.height

        #create a single channel image, thresholded to parameters

        blobs = self.extract_from_binary(
            img.binarize(threshold=threshold, maxv=255, blocksize=threshblocksize,
                         p=threshconstant, inverted=True).invert(), color_img=img,
            minsize=minsize, maxsize=maxsize)
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
        if maxsize <= 0:
            maxsize = color_img.width * color_img.height

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

        contours, hierarchy = cv2.findContours(binary_img.to_gray(),
                                               mode=cv2.RETR_TREE,
                                               method=cv2.CHAIN_APPROX_SIMPLE)
        if not list(contours):
            warnings.warn("Unable to find Blobs. Retuning Empty FeatureSet.")
            return FeatureSet([])

        all_blobs = []
        roots = []
        for index, node in enumerate(hierarchy[0].tolist()):
            if node[3] == -1:  # has no parent
                roots.append(index)

        while len(roots):
            blobs = []
            for root_index in roots:
                blob = []
                blob.append(root_index)  # append blob index
                if hierarchy[0][root_index][2] != -1:  # blob has children
                    child_index = hierarchy[0][root_index][2]
                    blob.append(child_index)  # append blob hole
                    while hierarchy[0][child_index][0] != -1:  # has next child
                        child_index = hierarchy[0][child_index][0]
                        blob.append(child_index)  # append blob hole
                blobs.append(blob)
            all_blobs += blobs
            roots = []
            for blob in blobs:
                for index in blob[1:]:
                    if hierarchy[0][index][2] != -1:
                        child_index = hierarchy[0][index][2]
                        roots.append(child_index)  # append blob hole
                        while hierarchy[0][child_index][0] != -1:
                            child_index = hierarchy[0][child_index][0]
                            roots.append(child_index)  # append blob hole

        ret_value = []
        for blob in all_blobs:
            blob_id = blob[0]
            hole_ids = blob[1:]
            hole_contours = [contours[id][:, 0, :].tolist() for id in hole_ids]
            temp = self._extract_data(contours[blob_id], hole_contours,
                                      color_img, minsize, maxsize,
                                      appx_level)
            if temp is not None:
                ret_value.append(temp)
        return FeatureSet(ret_value)

    def _extract_data(self, contour, hole_contour, color_img, minsize,
                      maxsize, appx_level):
        """
        Extract the bulk of the data from a give blob. If the blob's are is too
        large or too small the method returns none.
        """
        if contour is None or not contour.shape[0]:
            return None
        area = cv2.contourArea(contour)
        if area < minsize or area > maxsize:
            return None

        ret_value = Factory.Blob()
        ret_value.image = color_img
        ret_value.area = area

        ret_value.min_rectangle = cv2.minAreaRect(contour)
        bbr = cv2.boundingRect(contour)
        ret_value.x = bbr[0] + (bbr[2] / 2)
        ret_value.y = bbr[1] + (bbr[3] / 2)
        ret_value.perimeter = cv2.arcLength(contour, closed=True)
        ret_value.contour = contour[:, 0, :].tolist()
        appx = cv2.approxPolyDP(np.array(contour, np.float32),
                                epsilon=appx_level, closed=True)
        ret_value.contour_appx = appx[:, 0, :].astype(int).tolist()

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
        force_update_lazyproperties(ret_value)
        chull = cv2.convexHull(contour, returnPoints=1)
        ret_value.convex_hull = chull[:, 0, :].tolist()
        # KAS -- FLAG FOR REPLACE 6/6/2012
        #get_hull_mask = self._get_mask(chull)

        # KAS -- FLAG FOR REPLACE 6/6/2012
        # ret_value.hull_img = self._get_blob_as_image(chull,bb,
        #                                              color.get_bitmap(),
        #                                              get_hull_mask)

        # KAS -- FLAG FOR REPLACE 6/6/2012
        #ret_value.hull_mask = Image(get_hull_mask)

        moments = cv2.moments(contour)

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
        #ret_value.mask = Image(mask)

        mask = self._get_mask(contour)
        ret_value.avg_color = self._get_avg(color_img, bbr, mask)[0:3]

        # KAS -- FLAG FOR REPLACE 6/6/2012
        #ret_value.img = self._get_blob_as_image(color_img,
        #                                        bbr, mask)

        ret_value.hole_contour = hole_contour

        return ret_value

    @staticmethod
    def _get_holes(contour_num, contours, hierarchy):
        """
        This method returns the holes associated with a blob as a list of
        tuples.
        """
        ret_value = []
        for i, contour in enumerate(contours):
            if hierarchy[0][i][-1] == contour_num:
                if len(contour) >= 3:  # exclude single pixel holes
                        ret_value.append(contour[:, 0, :].tolist())
        return ret_value

    @staticmethod
    def _get_mask(contour):
        """
        Return a binary image of a particular get_contour sequence.
        """
        bbr = cv2.boundingRect(contour)
        mask = np.zeros((bbr[3], bbr[2]), dtype=np.uint8)
        cv2.drawContours(mask, contours=[contour], contourIdx=0,
                         color=255, thickness=-1, maxLevel=0,
                         offset=(-1 * bbr[0], -1 * bbr[1]))
        return mask

    @staticmethod
    def _get_avg(color_array, bb, mask):
        """
        Calculate the average color of a blob given the mask.
        """
        img = color_array[Factory.Image.roi_to_slice(bb)]
        return cv2.mean(img, mask=mask)

    @staticmethod
    def _get_blob_as_image(bbr, color_array, mask):
        """
        Return an image that contains just pixels defined by the blob sequence.
        """
        img = color_array[Factory.Image.roi_to_slice(bbr)]
        output_img = np.zeros((bbr[3], bbr[2], 3), dtype=np.uint8)
        output_img[mask] = img[mask]
        return Factory.Image(output_img)
