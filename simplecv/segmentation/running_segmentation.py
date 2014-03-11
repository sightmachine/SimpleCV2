from simplecv.base import cv
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

    mError = False
    mAlpha = 0.1
    mThresh = 10
    mModelImg = None
    mDiffImg = None
    mCurrImg = None
    mBlobMaker = None
    mGrayOnly = True
    mReady = False

    def __init__(self, alpha=0.7, thresh=(20, 20, 20)):
        """
        Create an running background difference.
        alpha - the update weighting where:
        accumulator = ((1-alpha)input_image)+((alpha)accumulator)

        threshold - the foreground background difference threshold.
        """
        self.mError = False
        self.mReady = False
        self.mAlpha = alpha
        self.mThresh = thresh
        self.mModelImg = None
        self.mDiffImg = None
        self.mColorImg = None
        self.mBlobMaker = BlobMaker()

    def add_image(self, img):
        """
        Add a single image to the segmentation algorithm
        """
        if img is None:
            return

        self.mColorImg = img
        if self.mModelImg is None:
            self.mModelImg = Image(
                cv.CreateImage((img.width, img.height), cv.IPL_DEPTH_32F, 3))
            self.mDiffImg = Image(
                cv.CreateImage((img.width, img.height), cv.IPL_DEPTH_32F, 3))
        else:
            # do the difference
            cv.AbsDiff(self.mModelImg.get_bitmap(), img.get_fp_matrix(),
                       self.mDiffImg.get_bitmap())
            #update the model
            cv.RunningAvg(img.get_fp_matrix(), self.mModelImg.get_bitmap(),
                          self.mAlpha)
            self.mReady = True

    def is_ready(self):
        """
        Returns true if the camera has a segmented image ready.
        """
        return self.mReady

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
        self.mModelImg = None
        self.mDiffImg = None

    def get_raw_image(self):
        """
        Return the segmented image with white representing the foreground
        and black the background.
        """
        return self._floatToInt(self.mDiffImg)

    def get_segmented_image(self, white_fg=True):
        """
        Return the segmented image with white representing the foreground
        and black the background.
        """
        retVal = None
        img = self._floatToInt(self.mDiffImg)
        if white_fg:
            retVal = img.binarize(thresh=self.mThresh)
        else:
            retVal = img.binarize(thresh=self.mThresh).invert()
        return retVal

    def get_segmented_blobs(self):
        """
        return the segmented blobs from the fg/bg image
        """
        retVal = []
        if self.mColorImg is not None and self.mDiffImg is not None:
            eightBit = self._floatToInt(self.mDiffImg)
            retVal = self.mBlobMaker.extractFromBinary(
                eightBit.binarize(thresh=self.mThresh), self.mColorImg)

        return retVal

    def _floatToInt(self, input):
        """
        convert a 32bit floating point cv array to an int array
        """
        temp = cv.CreateImage((input.width, input.height), cv.IPL_DEPTH_8U, 3)
        cv.Convert(input.get_bitmap(), temp)

        return Image(temp)

    def __getstate__(self):
        mydict = self.__dict__.copy()
        self.mBlobMaker = None
        self.mModelImg = None
        self.mDiffImg = None
        del mydict['blobmaker']
        del mydict['mModelImg']
        del mydict['mDiffImg']
        return mydict

    def __setstate__(self, mydict):
        self.__dict__ = mydict
        self.mBlobMaker = BlobMaker()
