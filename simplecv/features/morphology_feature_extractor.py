from simplecv.features.blobmaker import BlobMaker
from simplecv.features.feature_extractor_base import FeatureExtractorBase


class MorphologyFeatureExtractor(FeatureExtractorBase):
    """
    This feature extractor collects some basic morphology information about a
    given image. It is assumed that the object to be recognized is the largest
    object in the image. The user must provide a segmented white on black blob
    image.This operation then straightens the image and collects the data.
    """
    nbins = 9
    blobmaker = None
    threshold_operation = None

    def __init__(self, threshold_operation=None):
        """
        The threshold operation is a function of the form
        binaryimg = threshold(img)

        the simplest example would be:
        def binarize_wrap(img):

        """
        self.nbins = 9
        self.blobmaker = BlobMaker()
        self.threshold_operation = threshold_operation

    def set_threshold_operation(self, threshold_operation):
        """
        The threshold operation is a function of the form
        binaryimg = threshold(img)

        Example:

        >>> def binarize_wrap(img):
        >>>    return img.binarize()
        """
        self.threshold_operation = threshold_operation

    def extract(self, img):
        """
        This method takes in a image and returns some basic morphology
        characteristics about the largest blob in the image. The
        if a color image is provided the threshold operation is applied.
        """
        result = None
        if self.threshold_operation is not None:
            bw_img = self.threshold_operation(img)
        else:
            bw_img = img.binarize()

        if self.blobmaker is None:
            self.blobmaker = BlobMaker()

        fs = self.blobmaker.extract_from_binary(bw_img, img)
        if fs is not None and len(fs) > 0:
            fs = fs.sort_area()
            result = []
            result.append(fs[0].mArea / fs[0].mPerimeter)
            result.append(fs[0].mAspectRatio)
            result.append(fs[0].mHu[0])
            result.append(fs[0].mHu[1])
            result.append(fs[0].mHu[2])
            result.append(fs[0].mHu[3])
            result.append(fs[0].mHu[4])
            result.append(fs[0].mHu[5])
            result.append(fs[0].mHu[6])
        return result

    def get_field_names(self):
        """
        This method gives the names of each field in the feature vector in the
        order in which they are returned. For example, 'xpos' or 'width'
        """
        result = []
        result.append('area over perim')
        result.append('AR')
        result.append('Hu0')
        result.append('Hu1')
        result.append('Hu2')
        result.append('Hu3')
        result.append('Hu4')
        result.append('Hu5')
        result.append('Hu6')
        return result

    def get_num_fields(self):
        """
        This method returns the total number of fields in the feature vector.
        """
        return self.nbins

    def __getstate__(self):
        mydict = self.__dict__.copy()
        self.blobmaker = None
        del mydict['blobmaker']
        return mydict

    def __setstate__(self, mydict):
        self.__dict__ = mydict
        self.blobmaker = BlobMaker()
