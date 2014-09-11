import cv2

from simplecv.factory import Factory
from simplecv.features.blobmaker import BlobMaker
from simplecv.segmentation.segmentation_base import SegmentationBase


class MOGSegmentation(SegmentationBase):
    """
    Background subtraction using mixture of gausians.
    For each pixel store a set of gaussian distributions and try to fit new
    pixels into those distributions. One of the distributions will represent
    the background.

    history - length of the pixel history to be stored
    mixtures - number of gaussian distributions to be stored per pixel
    bg_ratio - chance of a pixel being included into the background model
    noise_sigma - noise amount
    learning_rate - higher learning rate means the system will adapt faster to
     new backgrounds
    """

    def __init__(self, **kwargs):

        if not hasattr(cv2, 'BackgroundSubtractorMOG'):
            raise ImportError("A newer version of OpenCV is needed")

        self.error = False
        self.ready = False
        self.diff_img = None
        self.color_img = None
        self.blobmaker = BlobMaker()

        self.history = kwargs.get('history', 200)
        self.mixtures = kwargs.get('mixtures', 5)
        self.bg_ratio = kwargs.get('bg_ratio', 0.7)
        self.noise_sigma = kwargs.get('noise_sigma', 15)
        self.learning_rate = kwargs.get('learning_rate', 0.7)

        self.bs_mog = cv2.BackgroundSubtractorMOG(
            history=self.history, nmixtures=self.mixtures,
            backgroundRatio=self.bg_ratio, noiseSigma=self.noise_sigma)

    def add_image(self, img):
        """
        Add a single image to the segmentation algorithm
        """
        if img is None:
            return

        self.color_img = img
        self.diff_img = Factory.Image(self.bs_mog.apply(img, None,
                                                        self.learning_rate))
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
        self.diff_img = None

    def get_raw_image(self):
        """
        Return the segmented image with white representing the foreground
        and black the background.
        """
        return self.diff_img

    def get_segmented_image(self, white_fg=True):
        """
        Return the segmented image with white representing the foreground
        and black the background.
        """
        return self.diff_img

    def get_segmented_blobs(self):
        """
        return the segmented blobs from the fg/bg image
        """
        ret_val = []
        if self.color_img is not None and self.diff_img is not None:
            ret_val = self.blobmaker.extract_from_binary(self.diff_img,
                                                         self.color_img)
        return ret_val

    def __getstate__(self):
        mydict = self.__dict__.copy()
        self.blobmaker = None
        self.diff_img = None
        del mydict['blobmaker']
        del mydict['diff_img']
        return mydict

    def __setstate__(self, mydict):
        self.__dict__ = mydict
        self.blobmaker = BlobMaker()
