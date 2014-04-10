import os
import tempfile

from simplecv.base import logger
from simplecv.core.camera.frame_source import FrameSource
from simplecv.factory import Factory

PIGGYPHOTO_ENABLED = True
try:
    import piggyphoto
except ImportError:
    PIGGYPHOTO_ENABLED = False


class DigitalCamera(FrameSource):
    """
    **SUMMARY**

    The DigitalCamera takes a point-and-shoot camera or high-end slr and uses
    it as a Camera.  The current frame can always be accessed with getPreview()

    Requires the PiggyPhoto Library: https://github.com/alexdu/piggyphoto

    **EXAMPLE**

    >>> cam = DigitalCamera()
    >>> pre = cam.get_preview()
    >>> pre.find_blobs().show()
    >>>
    >>> img = cam.get_image()
    >>> img.show()

    """
    camera = None
    usbid = None
    device = None

    def __init__(self, id=0):

        if not PIGGYPHOTO_ENABLED:
            logger.warn("Initializing failed, piggyphoto not found.")
            return

        devices = piggyphoto.cameraList(autodetect=True).toList()
        if not len(devices):
            logger.warn("No compatible digital cameras attached")
            return

        self.device, self.usbid = devices[id]
        self.camera = piggyphoto.camera()

    def get_image(self):
        """
        **SUMMARY**

        Retrieve an Image-object from the camera
        with the highest quality possible.
        **RETURNS**

        A SimpleCV Image.

        **EXAMPLES**
        >>> cam = DigitalCamera()
        >>> cam.get_image().show()
        """

        if not PIGGYPHOTO_ENABLED:
            logger.warn("piggyphoto not found")
            return

        file_ind, path = tempfile.mkstemp()
        self.camera.capture_image(path)
        img = Factory.Image(path)
        os.close(file_ind)
        os.remove(path)
        return img

    def get_preview(self):
        """
        **SUMMARY**

        Retrieve an Image-object from the camera
        with the preview quality.
        **RETURNS**

        A SimpleCV Image.

        **EXAMPLES**
        >>> cam = DigitalCamera()
        >>> cam.get_preview().show()
        """
        if not PIGGYPHOTO_ENABLED:
            logger.warn("piggyphoto not found")
            return

        file_ind, path = tempfile.mkstemp()
        self.camera.capture_preview(path)
        img = Factory.Image(path)
        os.close(file_ind)
        os.remove(path)
        return img
