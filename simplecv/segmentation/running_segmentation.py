import cv2
import numpy as np

from simplecv.factory import Factory
from simplecv.features.blob import Blob
from simplecv.segmentation.segmentation_base import SegmentationBase


class RunningSegmentation(SegmentationBase):
    """
    RunningSegmentation performs segmentation using a running background model.
    This model uses an accumulator which performs a running average of previous
    frames where:
    accumulator = ((1-alpha)input_image)+((alpha)accumulator)
    """

    def __init__(self, **kwargs):
        """
        Create an running background difference.
        alpha - the update weighting where:
        accumulator = ((1-alpha)input_image)+((alpha)accumulator)

        thresh - the foreground background difference threshold.
        """
        self.error = False
        self.ready = False
        self.alpha = kwargs.get('alpha', 0.7)
        self.thresh = kwargs.get('thresh', (20, 20, 20))
        self.model_img = None
        self.diff_img = None
        self.color_img = None

    def __getstate__(self):
        mydict = self.__dict__.copy()
        mydict['model_img'] = None
        mydict['diff_img'] = None
        return mydict

    def add_image(self, img):
        """
        Add a single image to the segmentation algorithm
        """
        if img is None:
            return

        self.color_img = img
        if self.model_img is None:
            self.model_img = Factory.Image(img.get_empty(3).astype(np.float32))
            self.diff_img = Factory.Image(img.get_empty(3).astype(np.float32))

        else:
            # do the difference
            diff = cv2.absdiff(self.model_img.astype(np.float32).copy(),
                               img.astype(np.float32).copy())
            self.diff_img = Factory.Image(diff)

            #update the model
            cv2.accumulateWeighted(src=img.astype(np.float32).copy(),
                                   dst=self.model_img.astype(np.float32).copy(),
                                   alpha=self.alpha)
            self.ready = True

    def is_ready(self):
        """
        Returns true if the camera has a segmented image ready.
        """
        return self.ready

    def is_error(self):
        """
        Returns true if the segmentation system has detected an error.
        Eventually we'll consruct a syntax of errors so this becomes
        more expressive
        """
        return self.error  # need to make a generic error checker

    def reset_error(self):
        """
        Clear the previous error.
        """
        self.error = False

    def reset(self):
        """
        Perform a reset of the segmentation systems underlying data.
        """
        self.model_img = None
        self.diff_img = None

    def get_raw_image(self):
        """
        Return the segmented image with white representing the foreground
        and black the background.
        """
        return self._float_to_int(self.diff_img)

    def get_segmented_image(self, white_fg=True):
        """
        Return the segmented image with white representing the foreground
        and black the background.
        """
        img = self._float_to_int(self.diff_img)
        if white_fg:
            ret_val = img.binarize(threshold=self.thresh, inverted=True)
        else:
            ret_val = img.binarize(threshold=self.thresh, inverted=True).invert()
        return ret_val

    def get_segmented_blobs(self):
        """
        return the segmented blobs from the fg/bg image
        """
        ret_val = []
        if self.color_img is not None and self.diff_img is not None:
            eight_bit = self._float_to_int(self.diff_img)
            ret_val = Blob.extract_from_binary(
                eight_bit.binarize(threshold=self.thresh, inverted=True),
                self.color_img)
        return ret_val

    @staticmethod
    def _float_to_int(img):
        """
        convert a 32bit floating point cv array to an int array
        """
        return Factory.Image(np.uint8(img))
