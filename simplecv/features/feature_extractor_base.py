import abc
import pickle


class FeatureExtractorBase(object):
    """
    The featureExtractorBase class is a way of abstracting the process of
    collecting descriptive features within an image. A feature is some
    description of the image like the mean color, or the width of a center
    image, or a histogram of edge lengths. This feature vectors can then be
    composed together and used within a machine learning algorithm to
    descriminate between different classes of objects.
    """

    __metaclass__ = abc.ABCMeta

    @classmethod
    def load(cls, fname):
        """
        load segmentation settings to file.
        """
        return pickle.load(file(fname))

    def save(self, fname):
        """
        Save segmentation settings to file.
        """
        output = open(fname, 'wb')
        pickle.dump(self, output, 2)  # use two otherwise it borks the system
        output.close()

    @abc.abstractmethod
    def extract(self, img):
        """
        Given an image extract the feature vector. The output should be a list
        object of all of the features. These features can be of any interal
        type (string, float, integer) but must contain no sub lists.
        """

    @abc.abstractmethod
    def get_field_names(self):
        """
        This method gives the names of each field in the feature vector in the
        order in which they are returned. For example, 'xpos' or 'width'
        """

    @abc.abstractmethod
    def get_num_fields(self):
        """
        This method returns the total number of fields in the feature vector.
        """
