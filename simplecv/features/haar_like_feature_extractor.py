from simplecv.features.feature_extractor_base import FeatureExtractorBase
from simplecv.features.haar_like_feature import HaarLikeFeature


class HaarLikeFeatureExtractor(FeatureExtractorBase):
    """
    This is used generate Haar like features from an image.  These
    Haar like features are used by a the classifiers of machine learning
    to help identify objects or things in the picture by their features,
    or in this case haar features.

    For a more in-depth review of Haar Like features see:
    http://en.wikipedia.org/wiki/Haar-like_features
    """

    def __init__(self, fname=None, do45=True):
        """
        fname - The feature file name
        do45 - if this is true we use the regular integral image plus the
        45 degree integral image
        """
        # we define the black (positive) and white (negative) regions of an
        # image to get our haar wavelet
        # FIXME: hope this should be self.do45 = do.45
        self.do45 = True
        self.featureset = None
        if fname is not None:
            self.read_wavelets(fname)

    def read_wavelets(self, fname, nfeats=-1):
        """
        fname = file name
        nfeats = number of features to load from file -1 -> All features
        """
        # We borrowed the wavelet file from  Chesnokov Yuriy
        # He has a great windows tutorial here:
        # http://www.codeproject.com/KB/audio-video/haar_detection.aspx
        # SimpleCV Took a vote and we think he is an all around swell guy!
        # nfeats = number of features to load
        # -1 loads all
        # otherwise loads min(nfeats,features in file)
        self.featureset = []
        ofile = open(fname, 'r')
        #line = ofile.readline()
        #count = int(line)
        temp = ofile.read()
        ofile.close()
        data = temp.split()
        count = int(data.pop(0))
        self.featureset = []
        if nfeats > -1:
            count = min(count, nfeats)
        while len(data) > 0:
            name = data.pop(0)
            nregions = int(data.pop(0))
            region = []
            for i in range(nregions):
                region.append(tuple(map(float, data[0:5])))
                data = data[5:]

            feat = HaarLikeFeature(name, region)
            self.featureset.append(feat)
        return None

    def save_wavelets(self, fname):
        """
        Save wavelets to file
        """
        ofile = open(fname, 'w')
        ofile.write(str(len(self.featureset)) + '\n\n')
        for i in range(len(self.featureset)):
            self.featureset[i].write_to_file(ofile)
        ofile.close()
        return None

    def extract(self, img):
        """
        This extractor takes in an image, creates the integral image, applies
        the Haar cascades, and returns the result as a feature vector.
        """
        regular = img.integral_image()
        result = []

        for i in range(len(self.featureset)):
            result.append(self.featureset[i].apply(regular))
        if self.do45:
            slant = img.integral_image(tilted=True)
            for i in range(len(self.featureset)):
                result.append(self.featureset[i].apply(regular))
        return result

    def get_field_names(self):
        """
        This method gives the names of each field in the feature vector in the
        order in which they are returned. For example, 'xpos' or 'width'
        """
        field_names = []
        for i in range(len(self.featureset)):
            field_names.append(self.featureset[i].mName)
        if self.do45:
            for i in range(len(self.featureset)):
                name = "Angle_" + self.featureset[i].mName
                field_names.append(name)
        return field_names

    def get_num_fields(self):
        """
        This method returns the total number of fields in the feature vector.
        """
        mult = 1
        if self.do45:
            mult = 2
        return mult * len(self.featureset)
