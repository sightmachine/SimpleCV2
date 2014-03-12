from cv2 import cv

from simplecv.features.blobmaker import BlobMaker
from simplecv.image_class import Image
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
        self.thresh = kwargs.get('alpha', (20, 20, 20))
        self.model_img = None
        self.diff_img = None
        self.color_img = None
        self.blobmaker = BlobMaker()

    def add_image(self, img):
        """
        Add a single image to the segmentation algorithm
        """
        if img is None:
            return

        self.color_img = img
        if self.model_img is None:
            self.model_img = Image(
                cv.CreateImage((img.width, img.height), cv.IPL_DEPTH_32F, 3))
            self.diff_img = Image(
                cv.CreateImage((img.width, img.height), cv.IPL_DEPTH_32F, 3))
        else:
            # do the difference
            cv.AbsDiff(self.model_img.get_bitmap(), img.get_fp_matrix(),
                       self.diff_img.get_bitmap())
            #update the model
            cv.RunningAvg(img.get_fp_matrix(), self.model_img.get_bitmap(),
                          self.alpha)
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
            ret_val = img.binarize(thresh=self.thresh)
        else:
            ret_val = img.binarize(thresh=self.thresh).invert()
        return ret_val

    def get_segmented_blobs(self):
        """
        return the segmented blobs from the fg/bg image
        """
        ret_val = []
        if self.color_img is not None and self.diff_img is not None:
            eight_bit = self._float_to_int(self.diff_img)
            ret_val = self.blobmaker.extractFromBinary(
                eight_bit.binarize(thresh=self.thresh), self.color_img)
        return ret_val

    @staticmethod
    def _float_to_int(img):
        """
        convert a 32bit floating point cv array to an int array
        """
        temp = cv.CreateImage((img.width, img.height), cv.IPL_DEPTH_8U, 3)
        cv.Convert(img.get_bitmap(), temp)

        return Image(temp)

    def __getstate__(self):
        mydict = self.__dict__.copy()
        self.blobmaker = None
        self.model_img = None
        self.diff_img = None
        del mydict['blobmaker']
        del mydict['model_img']
        del mydict['diff_img']
        return mydict

    def __setstate__(self, mydict):
        self.__dict__ = mydict
        self.blobmaker = BlobMaker()
