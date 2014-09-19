from simplecv.color_model import ColorModel
from simplecv.factory import Factory
from simplecv.features.blob import Blob
from simplecv.segmentation.segmentation_base import SegmentationBase


class ColorSegmentation(SegmentationBase):
    """
    Perform color segmentation based on a color model or color provided. This
    class uses color_model.py to create a color model.
    """

    def __init__(self):
        self.color_model = ColorModel()
        self.error = False
        self.cur_img = None
        self.truth_img = None

    def add_image(self, img):
        """
        Add a single image to the segmentation algorithm
        """
        if isinstance(img, str):
            img = Factory.Image(img)

        if isinstance(img, Factory.Image):
            self.truth_img = img
            self.cur_img = self.color_model.threshold(img)

    def is_ready(self):
        """
        Returns true if the camera has a segmented image ready.
        """
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
        self.color_model.reset()

    def get_raw_image(self):
        """
        Return the segmented image with white representing the foreground
        and black the background.
        """
        return self.cur_img

    def get_segmented_image(self, white_fg=True):
        """
        Return the segmented image with white representing the foreground
        and black the background.
        """
        return self.cur_img

    def get_segmented_blobs(self):
        """
        return the segmented blobs from the fg/bg image
        """
        return Blob.extract_from_binary(self.cur_img, self.truth_img)

    # The following are class specific methods

    def add_to_model(self, data):
        self.color_model.add(data)

    def subtract_model(self, data):
        self.color_model.remove(data)
