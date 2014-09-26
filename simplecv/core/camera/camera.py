import threading
import subprocess
import time

import cv2

from simplecv.base import logger, SYSTEM
from simplecv.core.camera.frame_source import FrameSource
from simplecv.core.pluginsystem import apply_plugins
from simplecv.factory import Factory

# Globals
_cameras = []
_camera_polling_thread = ""
_index = []


class FrameBufferThread(threading.Thread):
    """
    **SUMMARY**

    This is a helper thread which continually debuffers the camera frames.  If
    you don't do this, cameras may constantly give you a frame behind, which
    causes problems at low sample rates. This makes sure the frames returned
    by your camera are fresh.

    """

    def run(self):
        global _cameras
        while 1:
            for cam in _cameras:
                if cam.pygame_camera:
                    cam.pygame_buffer = cam.capture.get_image(
                        cam.pygame_buffer)
                else:
                    if cam.capture.isOpened():
                        cam.capture.grab()
                cam._thread_capture_time = time.time()
            time.sleep(0.04)  # max 25 fps, if you're lucky


@apply_plugins
class Camera(FrameSource):
    """
    **SUMMARY**

    The Camera class is the class for managing input from a basic camera.  Note
    that once the camera is initialized, it will be locked from being used
    by other processes.  You can check manually if you have compatible devices
    on linux by looking for /dev/video* devices.

    This class wrappers OpenCV's cv2.VideoCapture class and associated methods.
    Read up on OpenCV's CaptureFromCAM method for more details if you need
    finer control than just basic frame retrieval

    """

    prop_map = {"width": cv2.cv.CV_CAP_PROP_FRAME_WIDTH,
                "height": cv2.cv.CV_CAP_PROP_FRAME_HEIGHT,
                "brightness": cv2.cv.CV_CAP_PROP_BRIGHTNESS,
                "contrast": cv2.cv.CV_CAP_PROP_CONTRAST,
                "saturation": cv2.cv.CV_CAP_PROP_SATURATION,
                "hue": cv2.cv.CV_CAP_PROP_HUE,
                "gain": cv2.cv.CV_CAP_PROP_GAIN,
                "exposure": cv2.cv.CV_CAP_PROP_EXPOSURE}
    #human readable to CV constant property mapping

    def __init__(self, camera_index=-1, prop_set={}, threaded=True,
                 calibrationfile=''):
        """
        **SUMMARY**

        In the camera constructor, camera_index indicates which camera
        to connect to and props is a dictionary which can be used to set
        any camera attributes
        Supported props are currently: height, width, brightness, contrast,
        saturation, hue, gain, and exposure.

        You can also specify whether you want the FrameBufferThread
        to continuously debuffer the camera.  If you specify True,
        the camera is essentially 'on' at all times.
        If you specify off, you will have to manage camera buffers.

        **PARAMETERS**

        * *camera_index* - The index of the camera, these go
                           from 0 upward, and are system specific.
        * *prop_set* - The property set for the camera
                       (i.e. a dict of camera properties).

        .. Warning::
          For most web cameras only the width and height properties are
          supported. Support for all of the other parameters varies
          by camera and operating system.

        * *threaded* - If True we constantly debuffer the camera,
                       otherwise the user. Must do this manually.

        * *calibrationfile* - A calibration file to load.
        """
        super(Camera, self).__init__()

        global _cameras
        global _camera_polling_thread
        global _index

        self.thread = ""
        self.pygame_camera = False
        self.pygame_buffer = ""
        self.index = None
        self.threaded = False
        self.capture = None

        if SYSTEM == "Linux" and -1 in _index \
                and camera_index != -1 and camera_index not in _index:
            process = subprocess.Popen(["lsof /dev/video" + str(camera_index)],
                                       shell=True, stdout=subprocess.PIPE)
            (stdoutdata, _) = process.communicate()
            if stdoutdata:
                camera_index = -1

        elif SYSTEM == "Linux" \
                and camera_index == -1 and -1 not in _index:
            process = subprocess.Popen(["lsof /dev/video*"],
                                       shell=True, stdout=subprocess.PIPE)
            (stdoutdata, _) = process.communicate()
            if stdoutdata:
                camera_index = int(stdoutdata.split("\n")[1].split()[-1][-1])

        for cam in _cameras:
            if camera_index == cam.index:
                self.threaded = cam.threaded
                self.capture = cam.capture
                self.index = cam.index
                _cameras.append(self)
                return

        # This is to add support for XIMEA cameras.
        if isinstance(camera_index, str):
            if camera_index.lower() == 'ximea':
                camera_index = 1100
                _index.append(camera_index)

        # This fixes bug with opencv not being
        # able to grab frames from webcams on linux
        self.capture = cv2.VideoCapture(camera_index)
        self.index = camera_index
        if "delay" in prop_set:
            time.sleep(prop_set['delay'])

        if SYSTEM == "Linux" and ("height" in prop_set
                                  or not self.capture.grab()):
            import pygame.camera

            pygame.camera.init()
            threaded = True  # pygame must be threaded
            if camera_index == -1:
                camera_index = 0
                self.index = camera_index
                _index.append(camera_index)
                print _index
            if 'height' in prop_set and 'width' in prop_set:
                self.capture = pygame.camera.Camera("/dev/video" +
                                                    str(camera_index),
                                                    (prop_set['width'],
                                                     prop_set['height']))
            else:
                self.capture = pygame.camera.Camera("/dev/video" +
                                                    str(camera_index))

            try:
                self.capture.start()
            except Exception, e:
                logger.warning("caught exception: %r", e)
                logger.warning("SimpleCV can't seem to find a camera on your "
                               "system, or the drivers do not work with "
                               "simplecv.")
                return
            time.sleep(0)
            self.pygame_buffer = self.capture.get_image()
            self.pygame_camera = True
        else:
            _index.append(camera_index)
            self.threaded = False
            if SYSTEM == "Windows":
                threaded = False

            if not self.capture:
                return

            #set any properties in the constructor
            for prop in prop_set.keys():
                if prop in self.prop_map:
                    self.capture.set(self.prop_map[prop], prop_set[prop])

        if threaded:
            self.threaded = True
            _cameras.append(self)
            if not _camera_polling_thread:
                _camera_polling_thread = FrameBufferThread()
                _camera_polling_thread.daemon = True
                _camera_polling_thread.start()
                time.sleep(0)  # yield to thread

        if calibrationfile:
            self.load_calibration(calibrationfile)

    #todo -- make these dynamic attributes of the Camera class
    def get_property(self, prop):
        """
        **SUMMARY**

        Retrieve the value of a given property,
        wrapper for cv2.VideoCapture.get(propid)

        .. Warning::
          For most web cameras only the width and height properties
          are supported. Support for all of the other parameters
          varies by camera and operating system.

        **PARAMETERS**

        * *prop* - The property to retrive.

        **RETURNS**

        The specified property. If it can't be found the method returns False.

        **EXAMPLE**

        >>> cam = Camera()
        >>> prop = cam.get_property("width")
        """
        if self.pygame_camera:
            if prop.lower() == 'width':
                return self.capture.get_size()[0]
            elif prop.lower() == 'height':
                return self.capture.get_size()[1]
            else:
                return False

        if prop in self.prop_map:
            return self.capture.get(self.prop_map[prop])
        return False

    def get_all_properties(self):
        """
        **SUMMARY**

        Return all properties from the camera.

        **RETURNS**

        A dict of all the camera properties.

        """
        if self.pygame_camera:
            return False
        properties = {}
        for prop in self.prop_map:
            properties[prop] = self.get_property(prop)

        return properties

    def get_image(self):
        """
        **SUMMARY**

        Retrieve an Image-object from the camera.  If you experience problems
        with stale frames from the camera's hardware buffer, increase
        the flushcache number to dequeue multiple frames before retrieval.

        We're working on how to solve this problem.

        **RETURNS**

        A SimpleCV Image from the camera.

        **EXAMPLES**

        >>> cam = Camera()
        >>> while True:
        >>>    cam.get_image().show()

        """

        if self.pygame_camera:
            return Factory.Image(self.pygame_buffer.copy())

        if not self.threaded:
            self.capture_time = time.time()
        else:
            self.capture_time = self._thread_capture_time
        if self.capture.isOpened():
            _, img = self.capture.read()
        else:
            logger.warn("Unable to open camera")
            return None
        return Factory.Image(img, camera=self)
