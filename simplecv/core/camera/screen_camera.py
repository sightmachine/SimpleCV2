from simplecv.base import logger
from simplecv.factory import Factory

PYSCREENSHOT_ENABLED = True
try:
    import pyscreenshot
except ImportError:
    PYSCREENSHOT_ENABLED = False


class ScreenCamera(object):
    """
    **SUMMARY**
    ScreenCapture is a camera class would allow you to capture
    all or part of the screen and return it as a color image.

    Requires the pyscreenshot Library: https://github.com/vijaym123/
    pyscreenshot

    **EXAMPLE**
    >>> sc = ScreenCamera()
    >>> res = sc.get_resolution()
    >>> print res
    >>>
    >>> img = sc.get_image()
    >>> img.show()
    """

    def __init__(self):
        super(ScreenCamera, self).__init__()
        self._roi = None
        if not PYSCREENSHOT_ENABLED:
            logger.warn("Initializing pyscreenshot failed. "
                        "pyscreenshot not found.")

    @classmethod
    def get_resolution(cls):
        """
        **DESCRIPTION**

        returns the resolution of the screenshot of the screen.

        **PARAMETERS**
        None

        **RETURNS**
        returns the resolution.

        **EXAMPLE**

        >>> img = ScreenCamera()
        >>> res = img.get_resolution()
        >>> print res
        """
        if not PYSCREENSHOT_ENABLED:
            logger.warn("pyscreenshot not found.")
            return None
        return Factory.Image(pyscreenshot.grab()).size_tuple

    def set_roi(self, roi):
        """
        **DESCRIPTION**
        To set the region of interest.

        **PARAMETERS**
        * *roi* - tuple - It is a tuple of size 4. where region of interest
                          is to the center of the screen.

        **RETURNS**
        None

        **EXAMPLE**
        >>> sc = ScreenCamera()
        >>> res = sc.get_resolution()
        >>> sc.set_roi(res[0]/4,res[1]/4,res[0]/2,res[1]/2)
        >>> img = sc.get_image()
        >>> s.show()
        """
        if isinstance(roi, tuple) and len(roi) == 4:
            self._roi = roi

    def get_image(self):
        """
        **DESCRIPTION**

        get_image function returns a Image object
        capturing the current screenshot of the screen.

        **PARAMETERS**
        None

        **RETURNS**
        Returns the region of interest if setROI is used.
        else returns the original capture of the screenshot.

        **EXAMPLE**
        >>> sc = ScreenCamera()
        >>> img = sc.get_image()
        >>> img.show()
        """
        if not PYSCREENSHOT_ENABLED:
            logger.warn("pyscreenshot not found.")
            return None

        img = Factory.Image(pyscreenshot.grab())
        try:
            if self._roi:
                img = img.crop(self._roi, centered=True)
        except Exception:
            print "Error croping the image. ROI specified is not correct."
            return None
        return img
