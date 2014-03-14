import os

from cv2 import cv

from simplecv.base import logger, LAUNCH_PATH


class HaarCascade(object):
    """
    This class wraps HaarCascade files for the find_haar_features file.
    To use the class provide it with the path to a Haar cascade XML file and
    optionally a name.
    """
    _cache = {}

    def __init__(self, fname=None, name=None):
        #if fname.isalpha():
        #     fname = MY_CASCADES_DIR + fname + ".xml"

        if name is None:
            self._name = fname
        else:
            self._name = name

        # First checks the path given by the user,
        #  if not then checks SimpleCV's default folder
        if fname is not None:
            if os.path.exists(fname):
                self._fhandle = os.path.abspath(fname)
            else:
                self._fhandle = os.path.join(LAUNCH_PATH,
                                             'data/Features/HaarCascades',
                                             fname)
                if not os.path.exists(self._fhandle):
                    logger.warning("Could not find Haar Cascade file " + fname)
                    logger.warning("Try running the function "
                                   "img.list_haar_features() to see what is "
                                   "available")
                    return

            self._cascade = cv.Load(self._fhandle)

            if self._fhandle in HaarCascade._cache:
                self._cascade = HaarCascade._cache[self._fhandle]
                return
            HaarCascade._cache[self._fhandle] = self._cascade

    def load(self, fname=None, name=None):
        if name is None:
            self._name = fname
        else:
            self._name = name

        if fname is not None:
            if os.path.exists(fname):
                self._fhandle = os.path.abspath(fname)
            else:
                self._fhandle = os.path.join(LAUNCH_PATH,
                                             'data/Features/HaarCascades',
                                             fname)
                if not os.path.exists(self._fhandle):
                    logger.warning("Could not find Haar Cascade file " + fname)
                    logger.warning("Try running the function "
                                   "img.list_haar_features() to see what is "
                                   "available")
                    return None

            self._cascade = cv.Load(self._fhandle)

            if self._fhandle in HaarCascade._cache:
                self._cascade = HaarCascade._cache[fname]
                return
            HaarCascade._cache[self._fhandle] = self._cascade
        else:
            logger.warning("No file path mentioned.")

    def get_cascade(self):
        return self._cascade

    def get_name(self):
        return self._name

    def set_name(self, name):
        self._name = name

    def get_fhandle(self):
        return self._fhandle
