import cv2

from simplecv.features.blobmaker import BlobMaker
from simplecv.image import Image
from simplecv.segmentation.segmentation_base import SegmentationBase


class DiffSegmentation(SegmentationBase):
    """
    This method will do image segmentation by looking at the difference between
    two frames.

    grayOnly - use only gray images.
    threshold - The value at which we consider the color difference to
    be significant enough to be foreground imagery.

    The general usage is

    >>> from simplecv.segmentation.diff_segmentation import DiffSegmentation
    >>> from simplecv.core.camera.camera import Camera
    >>> segmentor = DiffSegmentation()
    >>> cam = Camera()
    >>> while True:
    >>>    segmentor.add_image(cam.get_image())
    >>>    if segmentor.is_ready():
    >>>        img = segmentor.get_segmented_image()

    """

    def __init__(self, **kwargs):
        self.grayonly_mode = kwargs.get('grayonly', False)
        self.threshold = kwargs.get('threshold', (10, 10, 10))
        self.error = False
        self.curr_img = None
        self.last_img = None
        self.diff_img = None
        self.color_img = None
        self.blobmaker = BlobMaker()

    def add_image(self, img):
        """
        Add a single image to the segmentation algorithm
        """
        if img is None:
            return
        if self.last_img is None:
            if self.grayonly_mode:
                self.last_img = img.to_gray()
                self.diff_img = Image(self.last_img.get_empty(1))
                self.curr_img = None
            else:
                self.last_img = img
                self.diff_img = Image(self.last_img.get_empty(3))
                self.curr_img = None
        else:
            if self.curr_img is not None:  # catch the first step
                self.last_img = self.curr_img

            if self.grayonly_mode:
                self.color_img = img
                self.curr_img = img.to_gray()
            else:
                self.color_img = img
                self.curr_img = img

            cv2.absdiff(self.curr_img.get_ndarray(),
                        self.last_img.get_ndarray(),
                        self.diff_img.get_ndarray())

    def is_ready(self):
        """
        Returns true if the camera has a segmented image ready.
        """
        if self.diff_img is None:
            return False
        else:
            return True

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
        self.curr_img = None
        self.last_img = None
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
        if white_fg:
            ret_val = self.diff_img.binarize(thresh=self.threshold)
        else:
            ret_val = self.diff_img.binarize(thresh=self.threshold).invert()
        return ret_val

    def get_segmented_blobs(self):
        """
        return the segmented blobs from the fg/bg image
        """
        ret_val = []
        if self.color_img is not None and self.diff_img is not None:
            ret_val = self.blobmaker.extract_from_binary(
                self.diff_img.binarize(thresh=self.threshold), self.color_img)
        return ret_val

    def __getstate__(self):
        mydict = self.__dict__.copy()
        self.blobmaker = None
        del mydict['blobmaker']
        return mydict

    def __setstate__(self, mydict):
        self.__dict__ = mydict
        self.blobmaker = BlobMaker()
