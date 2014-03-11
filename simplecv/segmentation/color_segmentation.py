from simplecv.color_model import ColorModel
from simplecv.features.blobmaker import BlobMaker
from simplecv.image_class import Image
from simplecv.segmentation.segmentation_base import SegmentationBase


class ColorSegmentation(SegmentationBase):
    """
    Perform color segmentation based on a color model or color provided. This
    class uses color_model.py to create a color model.
    """
    mColorModel = []
    mError = False
    mCurImg = []
    mTruthImg = []
    mBlobMaker = []

    def __init__(self):
        self.mColorModel = ColorModel()
        self.mError = False
        self.mCurImg = Image()
        self.mTruthImg = Image()
        self.mBlobMaker = BlobMaker()

    def add_image(self, img):
        """
        Add a single image to the segmentation algorithm
        """
        self.mTruthImg = img
        self.mCurImg = self.mColorModel.threshold(img)

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
        return self.mError  # need to make a generic error checker

    def reset_error(self):
        """
        Clear the previous error.
        """
        self.mError = False

    def reset(self):
        """
        Perform a reset of the segmentation systems underlying data.
        """
        self.mColorModel.reset()

    def get_raw_image(self):
        """
        Return the segmented image with white representing the foreground
        and black the background.
        """
        return self.mCurImg

    def get_segmented_image(self, white_fg=True):
        """
        Return the segmented image with white representing the foreground
        and black the background.
        """
        return self.mCurImg

    def get_segmented_blobs(self):
        """
        return the segmented blobs from the fg/bg image
        """
        return self.mBlobMaker.extractFromBinary(self.mCurImg, self.mTruthImg)

    # The following are class specific methods

    def addToModel(self, data):
        self.mColorModel.add(data)

    def subtractModel(self, data):
        self.mColorModel.remove(data)

    def __getstate__(self):
        mydict = self.__dict__.copy()
        self.mBlobMaker = None
        del mydict['blobmaker']
        return mydict

    def __setstate__(self, mydict):
        self.__dict__ = mydict
        self.mBlobMaker = BlobMaker()
