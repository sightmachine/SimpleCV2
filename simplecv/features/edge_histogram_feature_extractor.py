import numpy as np

from simplecv.features.feature_extractor_base import FeatureExtractorBase


class EdgeHistogramFeatureExtractor(FeatureExtractorBase):
    """
    Create a 1D edge length histogram and 1D edge angle histogram.

    This method takes in an image, applies an edge detector, and calculates
    the length and direction of lines in the image.

    bins = the number of bins
    """
    nbins = 10

    def __init__(self, bins=10):
        self.nbins = bins

    def extract(self, img):
        """
        Extract the line orientation and and length histogram.
        """
        #I am not sure this is the best normalization constant.
        result = []
        p = max(img.width, img.height) / 2
        min_line = 0.01 * p
        gap = 0.1 * p
        fs = img.find_lines(threshold=10, minlinelength=min_line,
                           maxlinegap=gap)
        ls = fs.length() / p  # normalize to image length
        angs = fs.angle()
        lhist = np.histogram(ls, self.nbins, normed=True, range=(0, 1))
        ahist = np.histogram(angs, self.nbins, normed=True, range=(-180, 180))
        result.extend(lhist[0].tolist())
        result.extend(ahist[0].tolist())
        return result

    def get_field_names(self):
        """
        Return the names of all of the length and angle fields.
        """
        result = []
        for i in range(self.nbins):
            name = "Length" + str(i)
            result.append(name)
        for i in range(self.nbins):
            name = "Angle" + str(i)
            result.append(name)

        return result

    def get_num_fields(self):
        """
        This method returns the total number of fields in the feature vector.
        """
        return self.nbins * 2
