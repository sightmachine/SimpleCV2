import time

import numpy as np

from simplecv.core.camera.frame_source import FrameSource
from simplecv.base import logger
from simplecv.factory import Factory

FREENECT_ENABLED = True
try:
    import freenect
except ImportError:
    FREENECT_ENABLED = False


class Kinect(FrameSource):
    """
    **SUMMARY**

    This is an experimental wrapper for the Freenect python libraries
    you can get_image() and get_depth() for separate channel images

    """

    def __init__(self, device_number=0):
        """
        **SUMMARY**

        In the kinect contructor, device_number indicates which kinect to
        connect to. It defaults to 0.

        **PARAMETERS**

        * *device_number* - The index of the kinect, these go from 0 upward.
        """
        super(Kinect, self).__init__()
        self.device_number = device_number
        if not FREENECT_ENABLED:
            logger.warning("You don't seem to have the freenect library "
                           "installed. This will make it hard to use "
                           "a Kinect.")

    #this code was borrowed from
    #https://github.com/amiller/libfreenect-goodies
    def get_image(self):
        """
        **SUMMARY**

        This method returns the Kinect camera image.

        **RETURNS**

        The Kinect's color camera image.

        **EXAMPLE**

        >>> k = Kinect()
        >>> while True:
        ...     k.get_image().show()

        """
        if not FREENECT_ENABLED:
            logger.warning("You don't seem to have the freenect library "
                           "installed. This will make it hard to use "
                           "a Kinect.")
            return

        video = freenect.sync_get_video(self.device_number)[0]
        self.capture_time = time.time()
        #video = video[:, :, ::-1]  # RGB -> BGR
        return Factory.Image(video.transpose([1, 0, 2]), camera=self)

    #low bits in this depth are stripped so it fits in an 8-bit image channel
    def get_depth(self):
        """
        **SUMMARY**

        This method returns the Kinect depth image.

        **RETURNS**

        The Kinect's depth camera image as a grayscale image.

        **EXAMPLE**

        >>> k = Kinect()
        >>> while True:
        ...     d = k.get_depth()
        ...     img = k.get_image()
        ...     result = img.side_by_side(d)
        ...     result.show()
        """

        if not FREENECT_ENABLED:
            logger.warning("You don't seem to have the freenect library "
                           "installed. This will make it hard to use "
                           "a Kinect.")
            return

        depth = freenect.sync_get_depth(self.device_number)[0]
        self.capture_time = time.time()
        np.clip(depth, 0, 2 ** 10 - 1, depth)
        depth >>= 2
        depth = depth.astype(np.uint8).transpose()

        return Factory.Image(depth, camera=self)

    #we're going to also support a higher-resolution (11-bit) depth matrix
    #if you want to actually do computations with the depth
    def get_depth_matrix(self):

        if not FREENECT_ENABLED:
            logger.warning("You don't seem to have the freenect library "
                           "installed. This will make it hard to use "
                           "a Kinect.")
            return

        self.capture_time = time.time()
        return freenect.sync_get_depth(self.device_number)[0]
