import numpy as np

from simplecv.base import cv
from simplecv.features.feature_extractor_base import FeatureExtractorBase


class HueHistogramFeatureExtractor(FeatureExtractorBase):
    """
    Create a Hue Histogram feature extractor. This feature extractor
    takes in an image, gets the hue channel, bins the number of pixels
    with a particular Hue, and returns the results.

    nbins - the number of Hue bins.
    """
    nbins = 16

    def __init__(self, nbins=16):
        #we define the black (positive) and white (negative) regions of an
        # image to get our haar wavelet
        self.nbins = nbins

    def extract(self, img):
        """
        This feature extractor takes in a color image and returns a normalized
        color histogram of the pixel counts of each hue.
        """
        img = img.to_hls()
        h = img.get_empty(1)
        cv.Split(img.get_bitmap(), h, None, None, None)
        npa = np.array(h[:, :])
        npa = npa.reshape(1, npa.shape[0] * npa.shape[1])
        hist = np.histogram(npa, self.nbins, normed=True, range=(0, 255))
        return hist[0].tolist()

    def get_field_names(self):
        """
        This method gives the names of each field in the feature vector in the
        order in which they are returned. For example, 'xpos' or 'width'
        """
        result = []
        for i in range(self.nbins):
            name = "Hue" + str(i)
            result.append(name)
        return result

    def get_num_fields(self):
        """
        This method returns the total number of fields in the feature vector.
        """
        return self.nbins
