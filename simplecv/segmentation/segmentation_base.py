import abc
import pickle


class SegmentationBase(object):
    """
    Right now I am going to keep this class as brain dead and single threaded
    as possible just so I can get the hang of abc in python. The idea behind a
    segmentation object is that you pass it frames, it does some sort of
    operations and you get a foreground / background segmented image.
    Eventually I would like these processes to by asynchronous and
    multithreaded so that they can raise specific image processing events.
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
    def add_image(self, img):
        """
        Add a single image to the segmentation algorithm
        """
        pass

    @abc.abstractmethod
    def is_ready(self):
        """
        Returns true if the camera has a segmented image ready.
        """
        return False

    @abc.abstractmethod
    def is_error(self):
        """
        Returns true if the segmentation system has detected an error.
        Eventually we'll consruct a syntax of errors so this becomes
        more expressive
        """
        return False

    @abc.abstractmethod
    def reset_error(self):
        """
        Clear the previous error.
        """
        return False

    @abc.abstractmethod
    def reset(self):
        """
        Perform a reset of the segmentation systems underlying data.
        """
        pass

    @abc.abstractmethod
    def get_raw_image(self):
        """
        Return the segmented image with white representing the foreground
        and black the background.
        """
        pass

    @abc.abstractmethod
    def get_segmented_image(self, white_fg=True):
        """
        Return the segmented image with white representing the foreground
        and black the background.
        """
        pass

    @abc.abstractmethod
    def get_segmented_blobs(self):
        """
        return the segmented blobs from the fg/bg image
        """
        pass
