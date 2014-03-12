'''
SimpleCV Cameras & Devices Library

This library is used managing input from different cameras.
'''
from collections import deque
from cStringIO import StringIO
from types import InstanceType
import os
import re
import subprocess
import tempfile
import threading
import time
import urllib2
import warnings

import ctypes as ct
import cv2
from cv2 import cv
import numpy as np
import pygame as pg

FREENECT_ENABLED = True
try:
    import freenect
except ImportError:
    FREENECT_ENABLED = False

PIGGYPHOTO_ENABLED = True
try:
    import piggyphoto
except ImportError:
    PIGGYPHOTO_ENABLED = False

PYSCREENSHOT_ENABLED = True
try:
    import pyscreenshot
except ImportError:
    PYSCREENSHOT_ENABLED = False

ARAVIS_ENABLED = True
try:
    from gi.repository import Aravis
except ImportError:
    ARAVIS_ENABLED = False


PIL_ENABLED = True
try:
    from PIL import Image as PilImage
    from PIL import ImageFont as pilImageFont
    from PIL import ImageDraw as pilImageDraw
    from PIL.GifImagePlugin import getheader, getdata
except ImportError:
    try:
        import Image as PilImage
        from GifImagePlugin import getheader, getdata
    except ImportError:
        PIL_ENABLED = False

from simplecv.base import logger, SYSTEM, nparray_to_cvmat
from simplecv.color import Color
from simplecv.display import Display
from simplecv.image_class import Image, ImageSet, ColorSpace

#Globals
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
                    cv.GrabFrame(cam.capture)
                cam.set_thread_capture_time = time.time()
            time.sleep(0.04)  # max 25 fps, if you're lucky


class FrameSource(object):
    """
    **SUMMARY**

    An abstract Camera-type class, for handling multiple types of video input.
    Any sources of images inheirit from it


    """
    _calib_matrix = ""  # Intrinsic calibration matrix
    _dist_coeff = ""  # Distortion matrix
    _thread_capture_time = ''  # when the last picture was taken
    capture_time = ''  # timestamp of the last aquired image

    def __init__(self):
        return

    def get_property(self, p):
        return None

    def get_all_properties(self):
        return {}

    def get_image(self):
        return None

    def calibrate(self, image_list, grid_sz=0.03, dimensions=(8, 5)):
        """
        **SUMMARY**

        Camera calibration will help remove distortion and fisheye effects
        It is agnostic of the imagery source, and can be used with any camera

        The easiest way to run calibration is to run the
        calibrate.py file under the tools directory for SimpleCV.
        This will walk you through the calibration process.

        **PARAMETERS**

        * *image_list* - is a list of images of color calibration images.

        * *grid_sz* - is the actual grid size of the calibration grid,
                      the unit used will be the calibration unit value
                      (i.e. if in doubt use meters, or U.S. standard)

        * *dimensions* - is the the count of the *interior* corners in the
                         calibration grid. So for a grid where there are
                         4x4 black grid squares has seven interior corners.

        **RETURNS**

        The camera's intrinsic matrix.

        **EXAMPLE**

        See :py:module:calibrate.py

        """
        # This routine was adapted from code originally written by:
        # Abid. K  -- abidrahman2@gmail.com
        # See: https://github.com/abidrahmank/OpenCV-Python/blob/master/
        # Other_Examples/camera_calibration.py

        warn_thresh = 1
        n_boards = 0  # no of boards
        board_w = int(dimensions[0])  # number of horizontal corners
        board_h = int(dimensions[1])  # number of vertical corners
        n_boards = int(len(image_list))
        board_n = board_w * board_h  # no of total corners
        board_sz = (board_w, board_h)  # size of board
        if n_boards < warn_thresh:
            logger.warning("FrameSource.calibrate: We suggest using 20 or more\
                            images to perform camera calibration!")

        # creation of memory storages
        image_points = cv.CreateMat(n_boards * board_n, 2, cv.CV_32FC1)
        object_points = cv.CreateMat(n_boards * board_n, 3, cv.CV_32FC1)
        point_counts = cv.CreateMat(n_boards, 1, cv.CV_32SC1)
        intrinsic_matrix = cv.CreateMat(3, 3, cv.CV_32FC1)
        distortion_coefficient = cv.CreateMat(5, 1, cv.CV_32FC1)

        # capture frames of specified properties
        # and modification of matrix values

        successes = 0
        img_idx = 0
        # capturing required number of views
        while successes < n_boards:
            img = image_list[img_idx]
            (_, corners) = cv.FindChessboardCorners(
                img.get_grayscale_matrix(),
                board_sz,
                cv.CV_CALIB_CB_ADAPTIVE_THRESH |
                cv.CV_CALIB_CB_FILTER_QUADS)
            corners = cv.FindCornerSubPix(img.get_grayscale_matrix(),
                                          corners, (11, 11), (-1, -1),
                                          (cv.CV_TERMCRIT_EPS +
                                           cv.CV_TERMCRIT_ITER, 30, 0.1))
            # if got a good image, draw chess board
            #if found == 1:
            #    corner_count = len(corners)
            #    z = z + 1

            # if got a good image, add to matrix
            if len(corners) == board_n:
                step = successes * board_n
                k = step
                for j in range(board_n):
                    cv.Set2D(image_points, k, 0, corners[j][0])
                    cv.Set2D(image_points, k, 1, corners[j][1])
                    cv.Set2D(object_points, k, 0,
                             grid_sz * (float(j) / board_w))
                    cv.Set2D(object_points, k, 1,
                             grid_sz * (float(j) % board_w))
                    cv.Set2D(object_points, k, 2, 0.0)
                    k = k + 1
                cv.Set2D(point_counts, successes, 0, board_n)
                successes = successes + 1

        # now assigning new matrices according to view_count
        if successes < warn_thresh:
            logger.warning("FrameSource.calibrate: You have %d good images "
                           "for calibration we recommend at least %d",
                           successes, warn_thresh)

        object_points2 = cv.CreateMat(successes * board_n, 3, cv.CV_32FC1)
        image_points2 = cv.CreateMat(successes * board_n, 2, cv.CV_32FC1)
        point_counts2 = cv.CreateMat(successes, 1, cv.CV_32SC1)

        for i in range(successes * board_n):
            cv.Set2D(image_points2, i, 0, cv.Get2D(image_points, i, 0))
            cv.Set2D(image_points2, i, 1, cv.Get2D(image_points, i, 1))
            cv.Set2D(object_points2, i, 0, cv.Get2D(object_points, i, 0))
            cv.Set2D(object_points2, i, 1, cv.Get2D(object_points, i, 1))
            cv.Set2D(object_points2, i, 2, cv.Get2D(object_points, i, 2))
        for i in range(successes):
            cv.Set2D(point_counts2, i, 0, cv.Get2D(point_counts, i, 0))

        cv.Set2D(intrinsic_matrix, 0, 0, 1.0)
        cv.Set2D(intrinsic_matrix, 1, 1, 1.0)
        rcv = cv.CreateMat(n_boards, 3, cv.CV_64FC1)
        tcv = cv.CreateMat(n_boards, 3, cv.CV_64FC1)
        # camera calibration
        cv.CalibrateCamera2(object_points2, image_points2, point_counts2,
                            (img.width, img.height), intrinsic_matrix,
                            distortion_coefficient, rcv, tcv, 0)
        self._calib_matrix = intrinsic_matrix
        self._dist_coeff = distortion_coefficient
        return intrinsic_matrix

    def get_camera_matrix(self):
        """
        **SUMMARY**

        This function returns a cvMat of the camera's intrinsic matrix.
        If there is no matrix defined the function returns None.

        """
        return self._calib_matrix

    def undistort(self, image_or_2darray):
        """
        **SUMMARY**

        If given an image, apply the undistortion given by the camera's
        matrix and return the result.

        If given a 1xN 2D cvmat or a 2xN numpy array, it will un-distort
        points of measurement and return them in the original coordinate
        system.

        **PARAMETERS**

        * *image_or_2darray* - an image or an ndarray.

        **RETURNS**

        The undistorted image or the undistorted points.
        If the camera is un-calibrated we return None.

        **EXAMPLE**

        >>> img = cam.get_image()
        >>> result = cam.undistort(img)


        """
        if type(self._calib_matrix) != cv.cvmat \
                or type(self._dist_coeff) != cv.cvmat:
            logger.warning("FrameSource.undistort: This operation requires "
                           "calibration, please load the calibration matrix")
            return None

        if type(image_or_2darray) == InstanceType \
                and image_or_2darray.__class__ == Image:
            in_img = image_or_2darray  # we have an image
            ret_val = in_img.get_empty()
            cv.Undistort2(in_img.get_bitmap(), ret_val,
                          self._calib_matrix, self._dist_coeff)
            return Image(ret_val)
        else:
            mat = ''
            if type(image_or_2darray) == cv.cvmat:
                mat = image_or_2darray
            else:
                arr = cv.fromarray(np.array(image_or_2darray))
                mat = cv.CreateMat(cv.GetSize(arr)[1], 1, cv.CV_64FC2)
                cv.Merge(arr[:, 0], arr[:, 1], None, None, mat)

            upoints = cv.CreateMat(cv.GetSize(mat)[1], 1, cv.CV_64FC2)
            cv.UndistortPoints(mat, upoints, self._calib_matrix,
                               self._dist_coeff)

            #undistorted.x = (x* focalX + principalX);
            #undistorted.y = (y* focalY + principalY);
            return (np.array(upoints[:, 0]) *
                    [self.get_camera_matrix()[0, 0],
                     self.get_camera_matrix()[1, 1]] +
                    [self.get_camera_matrix()[0, 2],
                     self.get_camera_matrix()[1, 2]])[:, 0]

    def get_image_undistort(self):
        """
        **SUMMARY**

        Using the overridden get_image method we retrieve
        the image and apply the undistortion operation.


        **RETURNS**

        The latest image from the camera after applying undistortion.

        **EXAMPLE**

        >>> cam = Camera()
        >>> cam.load_calibration("mycam.xml")
        >>> while True:
        >>>    img = cam.get_image_undistort()
        >>>    img.show()

        """
        return self.undistort(self.get_image())

    def save_calibration(self, filename):
        """
        **SUMMARY**

        Save the calibration matrices to file. The file name should be
        without the extension. The default extension is .xml.

        **PARAMETERS**

        * *filename* - The file name, without an extension,
                       to which to save the calibration data.

        **RETURNS**

        Returns true if the file was saved , false otherwise.

        **EXAMPLE**

        See :py:module:calibrate.py


        """
        if type(self._calib_matrix) != cv.cvmat:
            logger.warning("FrameSource.save_calibration: \
                            No calibration matrix present, can't save.")
        else:
            cv.Save(filename + "Intrinsic.xml", self._calib_matrix)

        if type(self._dist_coeff) != cv.cvmat:
            logger.warning("FrameSource.save_calibration: \
                            No calibration distortion present, can't save.")
        else:
            cv.Save(filename + "Distortion.xml", self._dist_coeff)

        return None

    def load_calibration(self, filename):
        """
        **SUMMARY**

        Load a calibration matrix from file.
        The filename should be the stem of the calibration files names.
        e.g. If the calibration files are:
        MyWebcamIntrinsic.xml and MyWebcamDistortion.xml
        then load the calibration file "MyWebcam"

        **PARAMETERS**

        * *filename* - The file name, without an extension,
                       to which to save the calibration data.

        **RETURNS**

        Returns true if the file was loaded , false otherwise.

        **EXAMPLE**

        See :py:module:calibrate.py

        """
        self._calib_matrix = cv.Load(filename + "Intrinsic.xml")
        self._dist_coeff = cv.Load(filename + "Distortion.xml")
        return True if type(self._dist_coeff) == cv.cvmat and \
                       type(self._calib_matrix) == cv.cvmat else False

    def live(self):
        """
        **SUMMARY**

        This shows a live view of the camera.

        **EXAMPLE**

        To use it's as simple as:

        >>> cam = Camera()
        >>> cam.live()

        Left click will show mouse coordinates and color
        Right click will kill the live image
        """

        start_time = time.time()

        #from SimpleCV.Display import Display
        image = self.get_image()
        display = Display(image.size())
        image.save(display)
        col = Color.RED

        while display.is_not_done():
            image = self.get_image()
            elapsed_time = time.time() - start_time

            if display.mouse_left:
                txt = "coord: (" + str(display.mouse_x) + "," \
                      + str(display.mouse_y) + ")"
                image.dl().text(txt, (10, image.height / 2), color=col)
                txt = "color: " + str(image.get_pixel(display.mouse_x,
                                                      display.mouse_y))
                image.dl().text(txt, (10, (image.height / 2) + 10), color=col)
                print "coord: (" + str(display.mouse_x) + "," \
                      + str(display.mouse_y) + "), color: " \
                      + str(image.get_pixel(display.mouse_x, display.mouse_y))

            if elapsed_time > 0 and elapsed_time < 5:
                image.dl().text("In live mode", (10, 10), color=col)
                image.dl().text("Left click will show mouse coordinates \
                                 and color", (10, 20), color=col)
                image.dl().text("Right click will kill the live \
                                 image", (10, 30), color=col)

            image.save(display)
            if display.mouse_right:
                print "Closing Window"
                display.isplaydone = True

        pg.quit()

    def get_thread_capture_time(self):
        return self._thread_capture_time

    def set_thread_capture_time(self, capture_time):
        self._thread_capture_time = capture_time


class Camera(FrameSource):
    """
    **SUMMARY**

    The Camera class is the class for managing input from a basic camera.  Note
    that once the camera is initialized, it will be locked from being used
    by other processes.  You can check manually if you have compatible devices
    on linux by looking for /dev/video* devices.

    This class wrappers OpenCV's cvCapture class and associated methods.
    Read up on OpenCV's CaptureFromCAM method for more details if you need
    finer control than just basic frame retrieval

    """
    capture = ""  # cvCapture object
    thread = ""
    pygame_camera = False
    pygame_buffer = ""

    prop_map = {"width": cv.CV_CAP_PROP_FRAME_WIDTH,
                "height": cv.CV_CAP_PROP_FRAME_HEIGHT,
                "brightness": cv.CV_CAP_PROP_BRIGHTNESS,
                "contrast": cv.CV_CAP_PROP_CONTRAST,
                "saturation": cv.CV_CAP_PROP_SATURATION,
                "hue": cv.CV_CAP_PROP_HUE,
                "gain": cv.CV_CAP_PROP_GAIN,
                "exposure": cv.CV_CAP_PROP_EXPOSURE}
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

        global _cameras
        global _camera_polling_thread
        global _index

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
        self.capture = cv.CaptureFromCAM(camera_index)
        self.index = camera_index
        if "delay" in prop_set:
            time.sleep(prop_set['delay'])

        if SYSTEM == "Linux" and ("height" in prop_set
                                  or not cv.GrabFrame(self.capture)):
            import pygame.camera

            pygame.camera.init()
            threaded = True  # pygame must be threaded
            if camera_index == -1:
                camera_index = 0
                self.index = camera_index
                _index.append(camera_index)
                print _index
            if "height" in prop_set and "width" in prop_set:
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
                return None

            #set any properties in the constructor
            for prop in prop_set.keys():
                if prop in self.prop_map:
                    cv.SetCaptureProperty(self.capture,
                                          self.prop_map[prop],
                                          prop_set[prop])

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
        wrapper for cv.GetCaptureProperty

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
            return cv.GetCaptureProperty(self.capture, self.prop_map[prop])
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
            return Image(self.pygame_buffer.copy())

        if not self.threaded:
            cv.GrabFrame(self.capture)
            self.capture_time = time.time()
        else:
            self.capture_time = self._thread_capture_time

        frame = cv.RetrieveFrame(self.capture)
        newimg = cv.CreateImage(cv.GetSize(frame), cv.IPL_DEPTH_8U, 3)
        cv.Copy(frame, newimg)
        return Image(newimg, self)


class VirtualCamera(FrameSource):
    """
    **SUMMARY**

    The virtual camera lets you test algorithms or functions by providing
    a Camera object which is not a physically connected device.

    VirtualCamera class supports "image", "imageset" and "video" source types.

    **USAGE**

    * For image, pass the filename or URL to the image
    * For the video, the filename
    * For imageset, you can pass either a path or a list of [path, extension]
    * For directory you treat a directory to show the latest file,
      an example would be where a security camera logs images to the directory,
      calling .get_image() will get the latest in the directory

    """
    source = ""
    sourcetype = ""
    lastmtime = 0

    def __init__(self, s, st, start=1):
        """
        **SUMMARY**

        The constructor takes a source, and source type.

        **PARAMETERS**

        * *s* - the source of the imagery.
        * *st* - the type of the virtual camera. Valid strings include:
        * *start* - the number of the frame that you want to start with.

          * "image" - a single still image.
          * "video" - a video file.
          * "imageset" - a SimpleCV image set.
          * "directory" - a VirtualCamera for loading a directory

        **EXAMPLE**

        >>> vc = VirtualCamera("img.jpg", "image")
        >>> vc = VirtualCamera("video.mpg", "video")
        >>> vc = VirtualCamera("./path_to_images/", "imageset")
        >>> vc = VirtualCamera("video.mpg", "video", 300)
        >>> vc = VirtualCamera("./imgs", "directory")


        """
        self.source = s
        self.sourcetype = st
        self.counter = 0
        if start == 0:
            start = 1
        self.start = start

        if self.sourcetype not in ["video", "image", "imageset", "directory"]:
            print 'Error: In VirtualCamera(), Incorrect Source option. \
                   "%s" \nUsage:' % self.sourcetype
            print '\tVirtualCamera("filename","video")'
            print '\tVirtualCamera("filename","image")'
            print '\tVirtualCamera("./path_to_images","imageset")'
            print '\tVirtualCamera("./path_to_images","directory")'
            return None

        else:
            if isinstance(self.source, str) and not os.path.exists(
                    self.source):
                print 'Error: In VirtualCamera()\n\t"%s" \
                       was not found.' % self.source
                return None

        if self.sourcetype == "imageset":
            if isinstance(s, ImageSet):
                self.source = s
            elif isinstance(s, (list, str)):
                self.source = ImageSet()
                if isinstance(s, list):
                    self.source.load(*s)
                else:
                    self.source.load(s)
            else:
                warnings.warn('Virtual Camera is unable to figure out \
                    the contents of your ImageSet, it must be a directory, \
                    list of directories, or an ImageSet object')

        elif self.sourcetype == 'video':

            self.capture = cv.CaptureFromFile(self.source)
            cv.SetCaptureProperty(self.capture,
                                  cv.CV_CAP_PROP_POS_FRAMES,
                                  self.start - 1)

        elif self.sourcetype == 'directory':
            pass

    def get_image(self):
        """
        **SUMMARY**

        Retrieve an Image-object from the virtual camera.
        **RETURNS**

        A SimpleCV Image from the camera.

        **EXAMPLES**

        >>> cam = VirtualCamera()
        >>> while True:
        >>>    cam.get_image().show()

        """
        if self.sourcetype == 'image':
            self.counter = self.counter + 1
            return Image(self.source, self)

        elif self.sourcetype == 'imageset':
            print len(self.source)
            img = self.source[self.counter % len(self.source)]
            self.counter = self.counter + 1
            return img

        elif self.sourcetype == 'video':
            # cv.QueryFrame returns None if the video is finished
            frame = cv.QueryFrame(self.capture)
            if frame:
                img = cv.CreateImage(cv.GetSize(frame), cv.IPL_DEPTH_8U, 3)
                cv.Copy(frame, img)
                return Image(img, self)
            else:
                return None

        elif self.sourcetype == 'directory':
            img = self.find_lastest_image(self.source, 'bmp')
            self.counter = self.counter + 1
            return Image(img, self)

    def rewind(self, start=None):
        """
        **SUMMARY**

        Rewind the Video source back to the given frame.
        Available for only video sources.

        **PARAMETERS**

        start - the number of the frame that you want to rewind to.
                if not provided, the video source would be rewound
                to the starting frame number you provided or rewound
                to the beginning.

        **RETURNS**

        None

        **EXAMPLES**

        >>> cam = VirtualCamera("filename.avi", "video", 120)
        >>> i=0
        >>> while i<60:
            ... cam.get_image().show()
            ... i+=1
        >>> cam.rewind()

        """
        if self.sourcetype == 'video':
            if not start:
                cv.SetCaptureProperty(self.capture,
                                      cv.CV_CAP_PROP_POS_FRAMES,
                                      self.start - 1)
            else:
                if start == 0:
                    start = 1
                cv.SetCaptureProperty(self.capture,
                                      cv.CV_CAP_PROP_POS_FRAMES,
                                      start - 1)

        else:
            self.counter = 0

    def get_frame(self, frame):
        """
        **SUMMARY**

        Get the provided numbered frame from the video source.
        Available for only video sources.

        **PARAMETERS**

        frame -  the number of the frame

        **RETURNS**

        Image

        **EXAMPLES**

        >>> cam = VirtualCamera("filename.avi", "video", 120)
        >>> cam.get_frame(400).show()

        """
        if self.sourcetype == 'video':
            number_frame = int(cv.GetCaptureProperty(
                self.capture, cv.CV_CAP_PROP_POS_FRAMES))
            cv.SetCaptureProperty(self.capture,
                                  cv.CV_CAP_PROP_POS_FRAMES,
                                  frame - 1)
            img = self.get_image()
            cv.SetCaptureProperty(self.capture,
                                  cv.CV_CAP_PROP_POS_FRAMES,
                                  number_frame)
            return img
        elif self.sourcetype == 'imageset':
            img = None
            if frame < len(self.source):
                img = self.source[frame]
            return img
        else:
            return None

    def skip_frames(self, number):
        """
        **SUMMARY**

        Skip n number of frames.
        Available for only video sources.

        **PARAMETERS**

        n - number of frames to be skipped.

        **RETURNS**

        None

        **EXAMPLES**

        >>> cam = VirtualCamera("filename.avi", "video", 120)
        >>> i=0
        >>> while i<60:
            ... cam.get_image().show()
            ... i+=1
        >>> cam.skip_frames(100)
        >>> cam.get_image().show()

        """
        if self.sourcetype == 'video':
            number_frame = int(cv.GetCaptureProperty(
                self.capture, cv.CV_CAP_PROP_POS_FRAMES))
            cv.SetCaptureProperty(self.capture,
                                  cv.CV_CAP_PROP_POS_FRAMES,
                                  number_frame + number - 1)
        elif self.sourcetype == 'imageset':
            self.counter = (self.counter + number) % len(self.source)
        else:
            self.counter = self.counter + number

    def get_frame_number(self):
        """
        **SUMMARY**

        Get the current frame number of the video source.
        Available for only video sources.

        **RETURNS**

        * *int* - number of the frame

        **EXAMPLES**

        >>> cam = VirtualCamera("filename.avi", "video", 120)
        >>> i=0
        >>> while i<60:
            ... cam.get_image().show()
            ... i+=1
        >>> cam.skip_frames(100)
        >>> cam.get_frame_number()

        """
        if self.sourcetype == 'video':
            number_frame = int(cv.GetCaptureProperty(
                self.capture, cv.CV_CAP_PROP_POS_FRAMES))
            return number_frame
        else:
            return self.counter

    def get_current_play_time(self):
        """
        **SUMMARY**

        Get the current play time in milliseconds of the video source.
        Available for only video sources.

        **RETURNS**

        * *int* - milliseconds of time from beginning of file.

        **EXAMPLES**

        >>> cam = VirtualCamera("filename.avi", "video", 120)
        >>> i=0
        >>> while i<60:
            ... cam.get_image().show()
            ... i+=1
        >>> cam.skip_frames(100)
        >>> cam.get_current_play_time()

        """
        if self.sourcetype == 'video':
            milliseconds = int(cv.GetCaptureProperty(self.capture,
                                                     cv.CV_CAP_PROP_POS_MSEC))
            return milliseconds
        else:
            raise ValueError('sources other than video do not \
                              have play time property')

    def find_lastest_image(self, directory='.', extension='png'):
        """
        **SUMMARY**

        This function finds the latest file in a directory
        with a given extension.

        **PARAMETERS**

        directory - The directory you want to load images from (defaults '.')
        extension - The image extension you want to use (defaults to .png)

        **RETURNS**

        The filename of the latest image

        **USAGE**

        #find all .png files in 'img' directory
        >>> cam = VirtualCamera('imgs/', 'png')
        >>> cam.get_image() # Grab the latest image from that directory

        """
        max_mtime = 0
        #max_dir = None
        #max_file = None
        max_full_path = None
        for dirname, _, files in os.walk(directory):
            for fname in files:
                if fname.split('.')[-1] == extension:
                    full_path = os.path.join(dirname, fname)
                    mtime = os.stat(full_path).st_mtime
                    if mtime > max_mtime:
                        max_mtime = mtime
                        #max_dir = dirname
                        #max_file = fname
                        self.lastmtime = mtime
                        max_full_path = os.path.abspath(
                            os.path.join(dirname, fname))

        #if file is being written, block until mtime is at least 100ms old
        while time.mktime(time.localtime()) - \
                os.stat(max_full_path).st_mtime < 0.1:
            time.sleep(0)

        return max_full_path


class Kinect(FrameSource):
    """
    **SUMMARY**

    This is an experimental wrapper for the Freenect python libraries
    you can get_image() and get_depth() for separate channel images

    """
    device_number = 0

    def __init__(self, device_number=0):
        """
        **SUMMARY**

        In the kinect contructor, device_number indicates which kinect to
        connect to. It defaults to 0.

        **PARAMETERS**

        * *device_number* - The index of the kinect, these go from 0 upward.
        """
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
        >>>   k.get_image().show()

        """
        if not FREENECT_ENABLED:
            logger.warning("You don't seem to have the freenect library "
                           "installed. This will make it hard to use "
                           "a Kinect.")
            return

        video = freenect.sync_get_video(self.device_number)[0]
        self.capture_time = time.time()
        #video = video[:, :, ::-1]  # RGB -> BGR
        return Image(video.transpose([1, 0, 2]), self)

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
        >>>   d = k.get_depth()
        >>>   img = k.get_image()
        >>>   result = img.side_by_side(d)
        >>>   result.show()
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

        return Image(depth, self)

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


class JpegStreamReader(threading.Thread):
    """
    **SUMMARY**

    A Threaded class for pulling down JPEG streams and breaking up the images.
    This is handy for reading the stream of images from a IP CAmera.

    """
    url = ""
    current_frame = ""
    _thread_capture_time = ""

    def __init__(self, url):
        self.url = url

    def run(self):

        stream_file = ''

        if re.search('@', self.url):
            authstuff = re.findall('//(\S+)@', self.url)[0]
            self.url = re.sub("//\S+@", "//", self.url)
            user, password = authstuff.split(":")

            #thank you missing urllib2 manual
            #http://www.voidspace.org.uk/python/articles/urllib2.shtml#id5
            password_mgr = urllib2.HTTPPasswordMgrWithDefaultRealm()
            password_mgr.add_password(None, self.url, user, password)

            handler = urllib2.HTTPBasicAuthHandler(password_mgr)
            opener = urllib2.build_opener(handler)

            stream_file = opener.open(self.url)
        else:
            stream_file = urllib2.urlopen(self.url)

        headers = stream_file.info()
        if "content-type" in headers:
            # force ucase first char
            headers['Content-type'] = headers['content-type']

        if "Content-type" not in headers:
            logger.warning("Tried to load a JpegStream from " + self.url +
                           ", but didn't find a content-type header!")
            return

        (multipart, boundary) = headers['Content-type'].split("boundary=")
        if not re.search("multipart", multipart, re.I):
            logger.warning("Tried to load a JpegStream from " + self.url +
                           ", but the content type header was " + multipart +
                           " not multipart/replace!")
            return

        buff = ''
        data = stream_file.readline().strip()
        length = 0
        contenttype = "jpeg"

        # the first frame contains a boundarystring and some header info
        while 1:
            # print data
            if re.search(boundary, data.strip()) and len(buff):
                # we have a full jpeg in buffer.  Convert to an image
                if contenttype == "jpeg":
                    self.current_frame = buff
                    self._thread_capture_time = time.time()
                buff = ''

            if re.match("Content-Type", data, re.I):
                # set the content type, if provided (default to jpeg)
                (_, typestring) = data.split(":")
                (_, contenttype) = typestring.strip().split("/")

            if re.match("Content-Length", data, re.I):
                # once we have the content length, we know how far to go jfif
                (_, length) = data.split(":")
                length = int(length.strip())

            if re.search("JFIF", data, re.I) or \
                    re.search("\xff\xd8\xff\xdb", data) or len(data) > 55:
                # we have reached the start of the image
                buff = ''
                if length and length > len(data):
                    # read the remainder of the image
                    buff += data + stream_file.read(length - len(data))
                    if contenttype == "jpeg":
                        self.current_frame = buff
                        self._thread_capture_time = time.time()
                else:
                    while not re.search(boundary, data):
                        buff += data
                        data = stream_file.readline()

                    (endimg, _) = data.split(boundary)
                    buff += endimg
                    data = boundary
                    continue

            data = stream_file.readline()  # load the next (header) line
            time.sleep(0)  # let the other threads go

    def get_thread_capture_time(self):
        return self._thread_capture_time


class JpegStreamCamera(FrameSource):
    """
    **SUMMARY**

    The JpegStreamCamera takes a URL of a JPEG stream and treats it like
    a camera.  The current frame can always be accessed with getImage()

    Requires the Python Imaging Library:
    http://www.pythonware.com/library/pil/handbook/index.htm

    **EXAMPLE**

    Using your Android Phone as a Camera. Softwares like IP Webcam can be used.

    # your IP may be different.
    >>> cam = JpegStreamCamera("http://192.168.65.101:8080/videofeed")
    >>> img = cam.get_image()
    >>> img.show()

    """
    url = ""
    cam_thread = ""

    def __init__(self, url):
        if not PIL_ENABLED:
            logger.warning("You need the Python Image Library \
                            (PIL) to use the JpegStreamCamera")
            return
        if not url.startswith('http://'):
            url = "http://" + url
        self.url = url
        self.cam_thread = JpegStreamReader(url)
        #self.cam_thread.url = self.url
        self.cam_thread.daemon = True
        self.cam_thread.start()

    def get_image(self):
        """
        **SUMMARY**

        Return the current frame of the JpegStream being monitored

        """
        if not PIL_ENABLED:
            logger.warning("You need the Python Image Library \
                            (PIL) to use the JpegStreamCamera")
            return

        if not self.cam_thread.get_thread_capture_time():
            now = time.time()
            while not self.cam_thread.get_thread_capture_time():
                if time.time() - now > 5:
                    warnings.warn("Timeout fetching JpegStream at " + self.url)
                    return
                time.sleep(0.1)

        self.capture_time = self.cam_thread.get_thread_capture_time()
        return Image(PilImage.open(StringIO(self.cam_thread.current_frame)),
                     self)


_SANE_INIT = False


class Scanner(FrameSource):
    """
    **SUMMARY**

    The Scanner lets you use any supported SANE-compatable
    scanner as a SimpleCV camera
    List of supported devices:
    http://www.sane-project.org/sane-supported-devices.html

    Requires the PySANE wrapper for libsane.  The sane scanner object
    is available for direct manipulation at Scanner.device

    This scanner object is heavily modified from
    https://bitbucket.org/DavidVilla/pysane

    Constructor takes an index (default 0) and a list of SANE options
    (default is color mode).

    **EXAMPLE**

    >>> scan = Scanner(0, { "mode": "gray" })
    >>> preview = scan.get_preview()
    >>> stuff = preview.find_blobs(minsize = 1000)
    >>> topleft = (np.min(stuff.x()), np.min(stuff.y()))
    >>> bottomright = (np.max(stuff.x()), np.max(stuff.y()))
    >>> scan.set_roi(topleft, bottomright)
    >>> scan.set_property("resolution", 1200) #set high resolution
    >>> scan.set_property("mode", "color")
    >>> img = scan.get_image()
    >>> scan.set_roi() #reset region of interest
    >>> img.show()


    """
    usbid = None
    manufacturer = None
    model = None
    kind = None
    device = None
    max_x = None
    max_y = None

    def __init__(self, id=0, properties={"mode": "color"}):
        global _SANE_INIT
        import sane

        if not _SANE_INIT:
            try:
                sane.init()
                _SANE_INIT = True
            except Exception:
                warnings.warn("Initializing pysane failed.")
                return

        devices = sane.get_devices()
        if not len(devices):
            warnings.warn("Did not find a sane-compatable device")
            return

        self.usbid, self.manufacturer, self.model, self.kind = devices[id]

        self.device = sane.open(self.usbid)
        self.max_x = self.device.br_x
        self.max_y = self.device.br_y  # save our extents for later

        for name, value in properties.items():
            setattr(self.device, name, value)

    def get_image(self):
        """
        **SUMMARY**

        Retrieve an Image-object from the scanner.  Any ROI set with
        setROI() is taken into account.
        **RETURNS**

        A SimpleCV Image.  Note that whatever the scanner mode is,
        SimpleCV will return a 3-channel, 8-bit image.

        **EXAMPLES**
        >>> scan = Scanner()
        >>> scan.get_image().show()
        """
        return Image(self.device.scan())

    def get_preview(self):
        """
        **SUMMARY**

        Retrieve a preview-quality Image-object from the scanner.
        **RETURNS**

        A SimpleCV Image.  Note that whatever the scanner mode is,
        SimpleCV will return a 3-channel, 8-bit image.

        **EXAMPLES**
        >>> scan = Scanner()
        >>> scan.get_preview().show()
        """
        self.preview = True
        img = Image(self.device.scan())
        self.preview = False
        return img

    def get_all_properties(self):
        """
        **SUMMARY**

        Return a list of all properties and values from the scanner
        **RETURNS**

        Dictionary of active options and values.  Inactive options appear
        as "None"

        **EXAMPLES**
        >>> scan = Scanner()
        >>> print scan.get_all_properties()
        """
        props = {}
        for prop in self.device.optlist:
            val = None
            if hasattr(self.device, prop):
                val = getattr(self.device, prop)
            props[prop] = val

        return props

    def print_properties(self):

        """
        **SUMMARY**

        Print detailed information about the SANE device properties
        **RETURNS**

        Nothing

        **EXAMPLES**
        >>> scan = Scanner()
        >>> scan.print_properties()
        """
        for prop in self.device.optlist:
            try:
                print self.device[prop]
            except Exception:
                pass

    def get_property(self, prop):
        """
        **SUMMARY**
        Returns a single property value from the SANE device
        equivalent to Scanner.device.PROPERTY

        **RETURNS**
        Value for option or None if missing/inactive

        **EXAMPLES**
        >>> scan = Scanner()
        >>> print scan.get_property('mode')
        color
        """
        if hasattr(self.device, prop):
            return getattr(self.device, prop)
        return None

    def set_roi(self, topleft=(0, 0), bottomright=(-1, -1)):
        """
        **SUMMARY**
        Sets an ROI for the scanner in the current resolution.  The
        two parameters, topleft and bottomright, will default to the
        device extents, so the ROI can be reset by calling setROI with
        no parameters.

        The ROI is set by SANE in resolution independent units (default
        MM) so resolution can be changed after ROI has been set.

        **RETURNS**
        None

        **EXAMPLES**
        >>> scan = Scanner()
        >>> scan.set_roi((50, 50), (100,100))
        >>> scan.get_image().show() # a very small crop on the scanner


        """
        self.device.tl_x = self.px2mm(topleft[0])
        self.device.tl_y = self.px2mm(topleft[1])
        if bottomright[0] == -1:
            self.device.br_x = self.max_x
        else:
            self.device.br_x = self.px2mm(bottomright[0])

        if bottomright[1] == -1:
            self.device.br_y = self.max_y
        else:
            self.device.br_y = self.px2mm(bottomright[1])

    def set_property(self, prop, val):
        """
        **SUMMARY**
        Assigns a property value from the SANE device
        equivalent to Scanner.device.PROPERTY = VALUE

        **RETURNS**
        None

        **EXAMPLES**
        >>> scan = Scanner()
        >>> print scan.get_property('mode')
        color
        >>> scan.set_property("mode", "gray")
        """
        setattr(self.device, prop, val)

    def px2mm(self, pixels=1):
        """
        **SUMMARY**
        Helper function to convert native scanner resolution to millimeter
        units

        **RETURNS**
        Float value

        **EXAMPLES**
        >>> scan = Scanner()
        >>> scan.px2mm(scan.device.resolution) #return DPI in DPMM
        """
        return float(pixels * 25.4 / float(self.device.resolution))


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
            warnings.warn("Initializing failed, piggyphoto not found.")
            return

        devices = piggyphoto.cameraList(autodetect=True).toList()
        if not len(devices):
            warnings.warn("No compatible digital cameras attached")
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
            warnings.warn("piggyphoto not found")
            return

        file_ind, path = tempfile.mkstemp()
        self.camera.capture_image(path)
        img = Image(path)
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
            warnings.warn("piggyphoto not found")
            return

        file_ind, path = tempfile.mkstemp()
        self.camera.capture_preview(path)
        img = Image(path)
        os.close(file_ind)
        os.remove(path)
        return img


class ScreenCamera():
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
    _roi = None

    def __init__(self):
        if not PYSCREENSHOT_ENABLED:
            warnings.warn("Initializing pyscreenshot failed. "
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
            warnings.warn("pyscreenshot not found.")
            return None
        return Image(pyscreenshot.grab()).size()

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
        return

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
            warnings.warn("pyscreenshot not found.")
            return None

        img = Image(pyscreenshot.grab())
        try:
            if self._roi:
                img = img.crop(self._roi, centered=True)
        except Exception:
            print "Error croping the image. ROI specified is not correct."
            return None
        return img


def set_obj_param(obj, params, param_name):
    param_value = params.get(param_name)
    if param_value is not None:
        setattr(obj, param_name, param_value)


class StereoImage(object):
    """
    **SUMMARY**

    This class is for binaculor Stereopsis. That is exactrating 3D information
    from two differing views of a scene(Image). By comparing the two images,
    the relative depth information can be obtained.

    - Fundamental Matrix : F : a 3 x 3 numpy matrix, is a relationship between
      any two images of the same scene that constrains where the projection
      of points from the scene can occur in both images. see:
      http://en.wikipedia.org/wiki/Fundamental_matrix_(computer_vision)

    - Homography Matrix : H : a 3 x 3 numpy matrix,

    - ptsLeft : The matched points on the left image.

    - ptsRight : The matched points on the right image.

    -findDisparityMap and findDepthMap - provides 3D information.

    for more information on stereo vision, visit:
        http://en.wikipedia.org/wiki/Computer_stereo_vision

    **EXAMPLE**
    >>> img1 = Image('sampleimages/stereo_view1.png')
    >>> img2 = Image('sampleimages/stereo_view2.png')
    >>> stereoImg = StereoImage(img1, img2)
    >>> stereoImg.find_disparity_map(method="BM",n_disparity=20).show()
    """

    image_3d = None

    def __init__(self, img_left, img_right):
        self.image_left = img_left
        self.image_right = img_right
        if self.image_left.size() != self.image_right.size():
            logger.warning('Left and Right images should have the same size.')
            return None
        else:
            self.size = self.image_left.size()
        self.image_3d = None

    def find_fundamental_mat(self, thresh=500.00, min_dist=0.15):
        """
        **SUMMARY**

        This method returns the fundamental matrix F
        such that (P_2).T F P_1 = 0

        **PARAMETERS**

        * *thresh* - The feature quality metric. This can be any value between
                     about 300 and 500. Higher values should return fewer,
                     but higher quality features.
        * *min_dist* - The value below which the feature correspondence is
                       considered a match. This is the distance between two
                       feature vectors. Good values are between 0.05 and 0.3

        **RETURNS**
        Return None if it fails.
        * *F* -  Fundamental matrix as ndarray.
        * *matched_pts1* - the matched points (x, y) in img1
        * *matched_pts2* - the matched points (x, y) in img2

        **EXAMPLE**
        >>> img1 = Image("sampleimages/stereo_view1.png")
        >>> img2 = Image("sampleimages/stereo_view2.png")
        >>> stereoImg = StereoImage(img1,img2)
        >>> F,pts1,pts2 = stereoImg.find_fundamental_mat()

        **NOTE**
        If you deal with the fundamental matrix F directly, be aware of
        (P_2).T F P_1 = 0 where P_2 and P_1 consist of (y, x, 1)
        """

        (kpts1, desc1) = self.image_left._get_raw_keypoints(thresh)
        (kpts2, desc2) = self.image_right._get_raw_keypoints(thresh)

        if desc1 is None or desc2 is None:
            logger.warning("We didn't get any descriptors. Image might \
                            be too uniform or blurry.")
            return None

        num_pts1 = desc1.shape[0]
        num_pts2 = desc2.shape[0]

        magic_ratio = 1.00
        if num_pts1 > num_pts2:
            magic_ratio = float(num_pts1) / float(num_pts2)

        (idx, dist) = Image()._get_flann_matches(desc1, desc2)
        result = dist.squeeze() * magic_ratio < min_dist

        try:
            import cv2
        except ImportError:
            logger.warning("Can't use fundamental matrix \
                            without OpenCV >= 2.3.0")
            return None

        pts1 = np.array([kpt.pt for kpt in kpts1])
        pts2 = np.array([kpt.pt for kpt in kpts2])

        matched_pts1 = pts1[idx[result]].squeeze()
        matched_pts2 = pts2[result]
        (fnd_mat, mask) = cv2.findFundamentalMat(matched_pts1, matched_pts2,
                                                 method=cv.CV_FM_LMEDS)

        inlier_ind = mask.nonzero()[0]
        matched_pts1 = matched_pts1[inlier_ind, :]
        matched_pts2 = matched_pts2[inlier_ind, :]

        matched_pts1 = matched_pts1[:, ::-1.00]
        matched_pts2 = matched_pts2[:, ::-1.00]
        return fnd_mat, matched_pts1, matched_pts2

    def find_homography(self, thresh=500.00, min_dist=0.15):
        """
        **SUMMARY**

        This method returns the homography H such that P2 ~ H P1

        **PARAMETERS**

        * *thresh* - The feature quality metric. This can be any value between
                     about 300 and 500. Higher values should return fewer,
                     but higher quality features.
        * *min_dist* - The value below which the feature correspondence is
                       considered a match. This is the distance between two
                       feature vectors. Good values are between 0.05 and 0.3

        **RETURNS**

        Return None if it fails.
        * *H* -  homography as ndarray.
        * *matched_pts1* - the matched points (x, y) in img1
        * *matched_pts2* - the matched points (x, y) in img2

        **EXAMPLE**
        >>> img1 = Image("sampleimages/stereo_view1.png")
        >>> img2 = Image("sampleimages/stereo_view2.png")
        >>> stereoImg = StereoImage(img1,img2)
        >>> H,pts1,pts2 = stereoImg.find_homography()

        **NOTE**
        If you deal with the homography H directly, be aware of P2 ~ H P1
        where P2 and P1 consist of (y, x, 1)
        """

        (kpts1, desc1) = self.image_left._get_raw_keypoints(thresh)
        (kpts2, desc2) = self.image_right._get_raw_keypoints(thresh)

        if desc1 is None or desc2 is None:
            logger.warning("We didn't get any descriptors. Image might be \
                            too uniform or blurry.")
            return None

        num_pts1 = desc1.shape[0]
        num_pts2 = desc2.shape[0]

        magic_ratio = 1.00
        if num_pts1 > num_pts2:
            magic_ratio = float(num_pts1) / float(num_pts2)

        (idx, dist) = Image()._get_flann_matches(desc1, desc2)
        result = dist.squeeze() * magic_ratio < min_dist

        try:
            import cv2
        except ImportError:
            logger.warning("Can't use homography without OpenCV >= 2.3.0")
            return None

        pts1 = np.array([kpt.pt for kpt in kpts1])
        pts2 = np.array([kpt.pt for kpt in kpts2])

        matched_pts1 = pts1[idx[result]].squeeze()
        matched_pts2 = pts2[result]

        (hmg, mask) = cv2.findHomography(matched_pts1, matched_pts2,
                                         method=cv.CV_LMEDS)

        inlier_ind = mask.nonzero()[0]
        matched_pts1 = matched_pts1[inlier_ind, :]
        matched_pts2 = matched_pts2[inlier_ind, :]

        matched_pts1 = matched_pts1[:, ::-1.00]
        matched_pts2 = matched_pts2[:, ::-1.00]
        return hmg, matched_pts1, matched_pts2

    def find_disparity_map(self, n_disparity=16, method='BM'):
        """
        The method generates disparity map from set of stereo images.

        **PARAMETERS**

        * *method* :
                 *BM* - Block Matching algorithm, this is a real time
                 algorithm.
                 *SGBM* - Semi Global Block Matching algorithm,
                          this is not a real time algorithm.
                 *GC* - Graph Cut algorithm, This is not a real time algorithm.

        * *n_disparity* - Maximum disparity value. This should be multiple
        of 16
        * *scale* - Scale factor

        **RETURNS**

        Return None if it fails.
        Returns Disparity Map Image

        **EXAMPLE**
        >>> img1 = Image("sampleimages/stereo_view1.png")
        >>> img2 = Image("sampleimages/stereo_view2.png")
        >>> stereoImg = StereoImage(img1,img2)
        >>> disp = stereoImg.find_disparity_map(method="BM")
        """
        gray_left = self.image_left.get_grayscale_matrix()
        gray_right = self.image_right.get_grayscale_matrix()
        (rows, colums) = self.size
        #scale = int(self.image_left.depth)
        if n_disparity % 16 != 0:
            if n_disparity < 16:
                n_disparity = 16
            n_disparity = (n_disparity / 16) * 16
        try:
            if method == 'BM':
                dsp = cv.CreateMat(colums, rows, cv.CV_32F)
                state = cv.CreateStereoBMState()
                state.SADWindowSize = 41
                state.preFilterType = 1
                state.preFilterSize = 41
                state.preFilterCap = 31
                state.minDisparity = -8
                state.numberOfDisparities = n_disparity
                state.textureThreshold = 10
                #state.speckleRange = 32
                #state.speckleWindowSize = 100
                state.uniquenessRatio = 15
                cv.FindStereoCorrespondenceBM(gray_left, gray_right, dsp,
                                              state)
                dsp_visual = cv.CreateMat(colums, rows, cv.CV_8U)
                cv.Normalize(dsp, dsp_visual, 0, 256, cv.CV_MINMAX)
                dsp_visual = Image(dsp_visual)
                return Image(dsp_visual.get_bitmap(),
                             colorSpace=ColorSpace.GRAY)

            elif method == 'GC':
                dsp_left = cv.CreateMat(colums, rows, cv.CV_32F)
                dsp_right = cv.CreateMat(colums, rows, cv.CV_32F)
                state = cv.CreateStereoGCState(n_disparity, 8)
                state.minDisparity = -8
                cv.FindStereoCorrespondenceGC(gray_left, gray_right, dsp_left,
                                              dsp_right, state, 0)
                dsp_left_visual = cv.CreateMat(colums, rows, cv.CV_8U)
                cv.Normalize(dsp_left, dsp_left_visual, 0, 256, cv.CV_MINMAX)
                #cv.Scale(dsp_left, dsp_left_visual, -scale)
                dsp_left_visual = Image(dsp_left_visual)
                return Image(dsp_left_visual.get_bitmap(),
                             colorSpace=ColorSpace.GRAY)

            elif method == 'SGBM':
                try:
                    import cv2

                    ver = cv2.__version__
                    if ver.startswith("$Rev :"):
                        logger.warning(
                            "Can't use SGBM without OpenCV >= 2.4.0")
                        return None
                except ImportError:
                    logger.warning("Can't use SGBM without OpenCV >= 2.4.0")
                    return None
                state = cv2.StereoSGBM()
                state.SADWindowSize = 41
                state.preFilterCap = 31
                state.minDisparity = 0
                state.numberOfDisparities = n_disparity
                #state.speckleRange = 32
                #state.speckleWindowSize = 100
                state.disp12MaxDiff = 1
                state.fullDP = False
                state.P1 = 8 * 1 * 41 * 41
                state.P2 = 32 * 1 * 41 * 41
                state.uniquenessRatio = 15
                dsp = state.compute(self.image_left.get_gray_numpy(),
                                    self.image_right.get_gray_numpy())
                return Image(dsp)

            else:
                logger.warning("Unknown method. Choose one method amoung \
                                BM or SGBM or GC !")
                return None

        except Exception:
            logger.warning("Error in computing the Disparity Map, may be \
                            due to the Images are stereo in nature.")
            return None

    def eline(self, point, fnd_mat, which_image):
        """
        **SUMMARY**

        This method returns, line feature object.

        **PARAMETERS**

        * *point* - Input point (x, y)
        * *fnd_mat* - Fundamental matrix.
        * *which_image* - Index of the image (1 or 2) that contains the point

        **RETURNS**

        epipolar line, in the form of line feature object.

        **EXAMPLE**

        >>> img1 = Image("sampleimages/stereo_view1.png")
        >>> img2 = Image("sampleimages/stereo_view2.png")
        >>> stereoImg = StereoImage(img1,img2)
        >>> F,pts1,pts2 = stereoImg.find_fundamental_mat()
        >>> point = pts2[0]
        >>> #find corresponding Epipolar line in the left image.
        >>> epiline = mapper.eline(point, F, 1)
        """

        from simplecv.features.detection import Line

        pts1 = (0, 0)
        pts2 = self.size
        pt_cvmat = cv.CreateMat(1, 1, cv.CV_32FC2)
        # OpenCV seems to use (y, x) coordinate.
        pt_cvmat[0, 0] = (point[1], point[0])
        line = cv.CreateMat(1, 1, cv.CV_32FC3)
        cv.ComputeCorrespondEpilines(pt_cvmat, which_image,
                                     nparray_to_cvmat(fnd_mat), line)
        line_np_array = np.array(line).squeeze()
        line_np_array = line_np_array[[1.00, 0, 2]]
        pts1 = (pts1[0], (-line_np_array[2] - line_np_array[0] * pts1[0])
                / line_np_array[1])
        pts2 = (pts2[0], (-line_np_array[2] - line_np_array[0] * pts2[0])
                / line_np_array[1])
        if which_image == 1:
            return Line(self.image_left, [pts1, pts2])
        elif which_image == 2:
            return Line(self.image_right, [pts1, pts2])

    def project_point(self, point, hmg, which_image):
        """
        **SUMMARY**

        This method returns the corresponding point (x, y)

        **PARAMETERS**

        * *point* - Input point (x, y)
        * *which_image* - Index of the image (1 or 2) that contains the point
        * *hmg* - Homography that can be estimated
                  using StereoCamera.find_homography()

        **RETURNS**

        Corresponding point (x, y) as tuple

        **EXAMPLE**

        >>> img1 = Image("sampleimages/stereo_view1.png")
        >>> img2 = Image("sampleimages/stereo_view2.png")
        >>> stereoImg = StereoImage(img1,img2)
        >>> F,pts1,pts2 = stereoImg.find_fundamental_mat()
        >>> point = pts2[0]
        >>> #finds corresponding  point in the left image.
        >>> projectPoint = stereoImg.project_point(point, H, 1)
        """

        hmg = np.matrix(hmg)
        point = np.matrix((point[1], point[0], 1.00))
        if which_image == 1.00:
            corres_pt = hmg * point.T
        else:
            corres_pt = np.linalg.inv(hmg) * point.T
        corres_pt = corres_pt / corres_pt[2]
        return float(corres_pt[1]), float(corres_pt[0])

    def get_3d_image(self, rpj_mat, method="BM", state=None):
        """
        **SUMMARY**

        This method returns the 3D depth image using reprojectImageTo3D method.

        **PARAMETERS**

        * *rpj_mat* - reprojection Matrix (disparity to depth matrix)
        * *method* - Stereo Correspondonce method to be used.
                   - "BM" - Stereo BM
                   - "SGBM" - Stereo SGBM
        * *state* - dictionary corresponding to parameters of
                    stereo correspondonce.
                    SADWindowSize - odd int
                    numberOfDisparities - int
                    minDisparity  - int
                    preFilterCap - int
                    preFilterType - int (only BM)
                    speckleRange - int
                    speckleWindowSize - int
                    P1 - int (only SGBM)
                    P2 - int (only SGBM)
                    fullDP - Bool (only SGBM)
                    uniquenessRatio - int
                    textureThreshold - int (only BM)

        **RETURNS**

        SimpleCV.Image representing 3D depth Image
        also StereoImage.Image3D gives OpenCV 3D Depth Image of CV_32F type.

        **EXAMPLE**

        >>> lImage = Image("l.jpg")
        >>> rImage = Image("r.jpg")
        >>> stereo = StereoImage(lImage, rImage)
        >>> rpj_mat = cv.Load("Q.yml")
        >>> stereo.get_3d_image(rpj_mat).show()

        >>> state = {"SADWindowSize":9, "numberOfDisparities":112}
        >>> stereo.get_3d_image(rpj_mat, "BM", state).show()
        >>> stereo.get_3d_image(rpj_mat, "SGBM", state).show()
        """
        img_left = self.image_left
        img_right = self.image_right
        cv2flag = True
        try:
            import cv2
        except ImportError:
            cv2flag = False
        import cv2.cv as cv

        (rows, colums) = self.size
        if method == "BM":
            sbm = cv.CreateStereoBMState()
            disparity = cv.CreateMat(colums, rows, cv.CV_32F)
            if not state:
                state = {"SADWindowSize": 9, "numberOfDisparities": 112,
                         "preFilterType": 1, "speckleWindowSize": 0,
                         "minDisparity": -39, "textureThreshold": 507,
                         "preFilterCap": 61, "uniquenessRatio": 0,
                         "speckleRange": 8, "preFilterSize": 5}

            set_obj_param(sbm, state, "SADWindowSize")
            set_obj_param(sbm, state, "preFilterCap")
            set_obj_param(sbm, state, "minDisparity")
            set_obj_param(sbm, state, "numberOfDisparities")
            set_obj_param(sbm, state, "uniquenessRatio")
            set_obj_param(sbm, state, "speckleRange")
            set_obj_param(sbm, state, "speckleWindowSize")
            set_obj_param(sbm, state, "textureThreshold")
            set_obj_param(sbm, state, "preFilterType")

            gray_left = img_left.get_grayscale_matrix()
            gray_right = img_right.get_grayscale_matrix()
            cv.FindStereoCorrespondenceBM(gray_left, gray_right, disparity,
                                          sbm)
            disparity_visual = cv.CreateMat(colums, rows, cv.CV_8U)

        elif method == "SGBM":
            if not cv2flag:
                warnings.warn("Can't Use SGBM without OpenCV >= 2.4. \
                               Use SBM instead.")
            sbm = cv2.StereoSGBM()
            if not state:
                state = {"SADWindowSize": 9, "numberOfDisparities": 96,
                         "minDisparity": -21, "speckleWindowSize": 0,
                         "preFilterCap": 61, "uniquenessRatio": 7,
                         "speckleRange": 8, "disp12MaxDiff": 1,
                         "fullDP": False}
                set_obj_param(sbm, state, "disp12MaxDiff")

            set_obj_param(sbm, state, "SADWindowSize")
            set_obj_param(sbm, state, "preFilterCap")
            set_obj_param(sbm, state, "minDisparity")
            set_obj_param(sbm, state, "numberOfDisparities")
            set_obj_param(sbm, state, "P1")
            set_obj_param(sbm, state, "P2")
            set_obj_param(sbm, state, "uniquenessRatio")
            set_obj_param(sbm, state, "speckleRange")
            set_obj_param(sbm, state, "speckleWindowSize")
            set_obj_param(sbm, state, "fullDP")

            disparity = sbm.compute(img_left.get_gray_numpy_cv2(),
                                    img_right.get_gray_numpy_cv2())

        else:
            warnings.warn("Unknown method. Returning None")
            return None

        if cv2flag:
            if not isinstance(rpj_mat, np.ndarray):
                rpj_mat = np.array(rpj_mat)
            if not isinstance(disparity, np.ndarray):
                disparity = np.array(disparity)
            image_3d = cv2.reprojectImageTo3D(disparity, rpj_mat,
                                              ddepth=cv2.cv.CV_32F)
            image_3d_normalize = cv2.normalize(image_3d, alpha=0, beta=255,
                                               norm_type=cv2.cv.CV_MINMAX,
                                               dtype=cv2.cv.CV_8UC3)
            ret_value = Image(image_3d_normalize, cv2image=True)
        else:
            image_3d = cv.CreateMat(self.image_left.size()[1],
                                    self.image_left.size()[0], cv2.cv.CV_32FC3)
            image_3d_normalize = cv.CreateMat(self.image_left.size()[1],
                                              self.image_left.size()[0],
                                              cv2.cv.CV_8UC3)
            cv.ReprojectImageTo3D(disparity, image_3d, rpj_mat)
            cv.Normalize(image_3d, image_3d_normalize, 0, 255,
                         cv.CV_MINMAX, cv2.cv.CV_8UC3)
            ret_value = Image(image_3d_normalize)
        self.image_3d = image_3d
        return ret_value

    def get_3d_image_from_disparity(self, disparity, rpj_mat):
        """
        **SUMMARY**

        This method returns the 3D depth image using reprojectImageTo3D method.

        **PARAMETERS**
        * *disparity* - Disparity Image
        * *rpj_mat* - reprojection Matrix (disparity to depth matrix)

        **RETURNS**

        SimpleCV.Image representing 3D depth Image
        also StereoCamera.Image3D gives OpenCV 3D Depth Image of CV_32F type.

        **EXAMPLE**

        >>> lImage = Image("l.jpg")
        >>> rImage = Image("r.jpg")
        >>> stereo = StereoCamera()
        >>> rpj_mat = cv.Load("Q.yml")
        >>> disp = stereo.find_disparity_map()
        >>> stereo.get_3d_image_from_disparity(disp, rpj_mat)
        """
        cv2flag = True
        try:
            import cv2
        except ImportError:
            cv2flag = False
            import cv2.cv as cv

        if cv2flag:
            if not isinstance(rpj_mat, np.ndarray):
                rpj_mat = np.array(rpj_mat)
            disparity = disparity.get_numpy_cv2()
            image_3d = cv2.reprojectImageTo3D(disparity, rpj_mat,
                                              ddepth=cv2.cv.CV_32F)
            image_3d_normalize = cv2.normalize(image_3d, alpha=0, beta=255,
                                               norm_type=cv2.cv.CV_MINMAX,
                                               dtype=cv2.cv.CV_8UC3)
            ret_value = Image(image_3d_normalize, cv2image=True)
        else:
            disparity = disparity.get_matrix()
            image_3d = cv.CreateMat(self.image_left.size()[1],
                                    self.image_left.size()[0],
                                    cv2.cv.CV_32FC3)
            image_3d_normalize = cv.CreateMat(self.image_left.size()[1],
                                              self.image_left.size()[0],
                                              cv2.cv.CV_8UC3)
            cv.ReprojectImageTo3D(disparity, image_3d, rpj_mat)
            cv.Normalize(image_3d, image_3d_normalize, 0, 255,
                         cv.CV_MINMAX, cv2.cv.CV_8UC3)
            ret_value = Image(image_3d_normalize)
        self.image_3d = image_3d
        return ret_value


class StereoCamera(object):
    """
    Stereo Camera is a class dedicated for calibration stereo camera.
    It also has functionalites for rectification and getting undistorted
    Images.

    This class can be used to calculate various parameters
    related to both the camera's :
      -> Camera Matrix
      -> Distortion coefficients
      -> Rotation and Translation matrix
      -> Rectification transform (rotation matrix)
      -> Projection matrix in the new (rectified) coordinate systems
      -> Disparity-to-depth mapping matrix (Q)
    """

    def __init__(self):
        return

    def stereo_calibration(self, cam_left, cam_right, nboards=30,
                           chessboard=(8, 5), grid_size=0.027,
                           win_size=(352, 288)):
        """

        **SUMMARY**

        Stereo Calibration is a way in which you obtain the parameters that
        will allow you to calculate 3D information of the scene.
        Once both the camera's are initialized.
        Press [Space] once chessboard is identified in both the camera's.
        Press [esc] key to exit the calibration process.

        **PARAMETERS**

        * cam_left - Left camera index.
        * cam_right - Right camera index.
        * nboards - Number of samples or multiple views of the chessboard in
                    different positions and orientations with your stereo
                    camera
        * chessboard - A tuple of Cols, Rows in
                       the chessboard (used for calibration).
        * grid_size - chessboard grid size in real units
        * win_size - This is the window resolution.

        **RETURNS**

        A tuple of the form (cm1, cm2, d1, d2, r, t, e, f) on success
        cm1 - Camera Matrix for left camera,
        cm2 - Camera Matrix for right camera,
        d1 - Vector of distortion coefficients for left camera,
        d2 - Vector of distortion coefficients for right camera,
        r - Rotation matrix between the left and the right
            camera coordinate systems,
        t - Translation vector between the left and the right
            coordinate systems of the cameras,
        e - Essential matrix,
        f - Fundamental matrix

        **EXAMPLE**

        >>> StereoCam = StereoCamera()
        >>> calibration = StereoCam.stereo_calibration(1,2,nboards=40)

        **Note**

        Press space to capture the images.

        """
        count = 0
        left = "Left"
        right = "Right"
        try:
            capture_left = cv.CaptureFromCAM(cam_left)
            cv.SetCaptureProperty(capture_left, cv.CV_CAP_PROP_FRAME_WIDTH,
                                  win_size[0])
            cv.SetCaptureProperty(capture_left, cv.CV_CAP_PROP_FRAME_HEIGHT,
                                  win_size[1])
            frame_left = cv.QueryFrame(capture_left)
            cv.FindChessboardCorners(frame_left, chessboard)

            capture_right = cv.CaptureFromCAM(cam_right)
            cv.SetCaptureProperty(capture_right, cv.CV_CAP_PROP_FRAME_WIDTH,
                                  win_size[0])
            cv.SetCaptureProperty(capture_right, cv.CV_CAP_PROP_FRAME_HEIGHT,
                                  win_size[1])
            frame_right = cv.QueryFrame(capture_right)
            cv.FindChessboardCorners(frame_right, chessboard)
        except Exception:
            print "Error Initialising the Left and Right camera"
            return None

        cols = nboards * chessboard[0] * chessboard[1]
        image_points1 = cv.CreateMat(1, cols, cv.CV_64FC2)
        image_points2 = cv.CreateMat(1, cols, cv.CV_64FC2)

        object_points = cv.CreateMat(1, cols, cv.CV_64FC3)
        num_points = cv.CreateMat(1, nboards, cv.CV_32S)

        # the intrinsic camera matrices
        cm1 = cv.CreateMat(3, 3, cv.CV_64F)
        cm2 = cv.CreateMat(3, 3, cv.CV_64F)

        # the distortion coefficients of both cameras
        d1 = cv.CreateMat(1, 5, cv.CV_64F)
        d2 = cv.CreateMat(1, 5, cv.CV_64F)

        # matrices governing the rotation and translation from camera 1
        # to camera 2
        r = cv.CreateMat(3, 3, cv.CV_64F)
        t = cv.CreateMat(3, 1, cv.CV_64F)

        # the essential and fundamental matrices
        e = cv.CreateMat(3, 3, cv.CV_64F)
        f = cv.CreateMat(3, 3, cv.CV_64F)

        while True:
            frame_left = cv.QueryFrame(capture_left)
            cv.Flip(frame_left, frame_left, 1)
            frame_right = cv.QueryFrame(capture_right)
            cv.Flip(frame_right, frame_right, 1)
            k = cv.WaitKey(3)

            cor1 = cv.FindChessboardCorners(frame_left, chessboard)
            if cor1[0]:
                cv.DrawChessboardCorners(frame_left, chessboard,
                                         cor1[1], cor1[0])
                cv.ShowImage(left, frame_left)

            cor2 = cv.FindChessboardCorners(frame_right, chessboard)
            if cor2[0]:
                cv.DrawChessboardCorners(frame_right, chessboard,
                                         cor2[1], cor2[0])
                cv.ShowImage(right, frame_right)

            cbrd_mlt = chessboard[0] * chessboard[1]
            if cor1[0] and cor2[0] and k == 0x20:
                print count
                for i in range(0, len(cor1[1])):
                    cv.Set1D(image_points1, count * cbrd_mlt + i,
                             cv.Scalar(cor1[1][i][0], cor1[1][i][1]))
                    cv.Set1D(image_points2, count * cbrd_mlt + i,
                             cv.Scalar(cor2[1][i][0], cor2[1][i][1]))

                count += 1

                if count == nboards:
                    cv.DestroyAllWindows()
                    for i in range(nboards):
                        for j in range(chessboard[1]):
                            for k in range(chessboard[0]):
                                cv.Set1D(object_points,
                                         i * cbrd_mlt + j * chessboard[0] + k,
                                         (k * grid_size, j * grid_size, 0))

                    for i in range(nboards):
                        cv.Set1D(num_points, i, cbrd_mlt)

                    cv.SetIdentity(cm1)
                    cv.SetIdentity(cm2)
                    cv.Zero(d1)
                    cv.Zero(d2)

                    print "Running stereo calibration..."
                    del cam_left
                    del cam_right
                    cv.StereoCalibrate(
                        object_points, image_points1, image_points2,
                        num_points, cm1, d1, cm2, d2, win_size, r, t, e, f,
                        flags=cv.CV_CALIB_SAME_FOCAL_LENGTH
                        | cv.CV_CALIB_ZERO_TANGENT_DIST)

                    print "Done."
                    return cm1, cm2, d1, d2, r, t, e, f

            cv.ShowImage(left, frame_left)
            cv.ShowImage(right, frame_right)
            if k == 0x1b:
                print "ESC pressed. Exiting. \
                       WARNING: NOT ENOUGH CHESSBOARDS FOUND YET"
                cv.DestroyAllWindows()
                break

    def save_calibration(self, calibration=None, fname="Stereo", cdir="."):
        """

        **SUMMARY**

        save_calibration is a method to save the StereoCalibration parameters
        such as CM1, CM2, D1, D2, R, T, E, F of stereo pair.
        This method returns True on success and saves the calibration
        in the following format.
        StereoCM1.txt
        StereoCM2.txt
        StereoD1.txt
        StereoD2.txt
        StereoR.txt
        StereoT.txt
        StereoE.txt
        StereoF.txt

        **PARAMETERS**

        calibration - is a tuple os the form (CM1, CM2, D1, D2, R, T, E, F)
        CM1 -> Camera Matrix for left camera,
        CM2 -> Camera Matrix for right camera,
        D1 -> Vector of distortion coefficients for left camera,
        D2 -> Vector of distortion coefficients for right camera,
        R -> Rotation matrix between the left and the right
             camera coordinate systems,
        T -> Translation vector between the left and the right
             coordinate systems of the cameras,
        E -> Essential matrix,
        F -> Fundamental matrix


        **RETURNS**

        return True on success and saves the calibration files.

        **EXAMPLE**

        >>> StereoCam = StereoCamera()
        >>> calibration = StereoCam.stereo_calibration(1,2,nboards=40)
        >>> StereoCam.save_calibration(calibration,fname="Stereo1")
        """
        filenames = (fname + "CM1.txt", fname + "CM2.txt", fname + "D1.txt",
                     fname + "D2.txt", fname + "R.txt", fname + "T.txt",
                     fname + "E.txt", fname + "F.txt")
        try:
            (cm1, cm2, d1, d2, r, t, e, f) = calibration
            cv.Save("{0}/{1}".format(cdir, filenames[0]), cm1)
            cv.Save("{0}/{1}".format(cdir, filenames[1]), cm2)
            cv.Save("{0}/{1}".format(cdir, filenames[2]), d1)
            cv.Save("{0}/{1}".format(cdir, filenames[3]), d2)
            cv.Save("{0}/{1}".format(cdir, filenames[4]), r)
            cv.Save("{0}/{1}".format(cdir, filenames[5]), t)
            cv.Save("{0}/{1}".format(cdir, filenames[6]), e)
            cv.Save("{0}/{1}".format(cdir, filenames[7]), f)
            print "Calibration parameters written \
                   to directory '{0}'.".format(cdir)
            return True

        except Exception:
            return False

    def load_calibration(self, fname="Stereo", dir="."):
        """

        **SUMMARY**

        load_calibration is a method to load the StereoCalibration parameters
        such as CM1, CM2, D1, D2, R, T, E, F of stereo pair.
        This method loads from calibration files and return calibration
        on success else return false.

        **PARAMETERS**

        fname - is the prefix of the calibration files.
        dir - is the directory in which files are present.

        **RETURNS**

        a tuple of the form (CM1, CM2, D1, D2, R, T, E, F) on success.
        CM1 - Camera Matrix for left camera
        CM2 - Camera Matrix for right camera
        D1 - Vector of distortion coefficients for left camera
        D2 - Vector of distortion coefficients for right camera
        R - Rotation matrix between the left and the right
            camera coordinate systems
        T - Translation vector between the left and the right
            coordinate systems of the cameras
        E - Essential matrix
        F - Fundamental matrix
        else returns false

        **EXAMPLE**

        >>> StereoCam = StereoCamera()
        >>> loadedCalibration = StereoCam.load_calibration(fname="Stereo1")

        """
        filenames = (fname + "CM1.txt", fname + "CM2.txt", fname + "D1.txt",
                     fname + "D2.txt", fname + "R.txt", fname + "T.txt",
                     fname + "E.txt", fname + "F.txt")
        try:
            cm1 = cv.Load("{0}/{1}".format(dir, filenames[0]))
            cm2 = cv.Load("{0}/{1}".format(dir, filenames[1]))
            d1 = cv.Load("{0}/{1}".format(dir, filenames[2]))
            d2 = cv.Load("{0}/{1}".format(dir, filenames[3]))
            r = cv.Load("{0}/{1}".format(dir, filenames[4]))
            t = cv.Load("{0}/{1}".format(dir, filenames[5]))
            e = cv.Load("{0}/{1}".format(dir, filenames[6]))
            f = cv.Load("{0}/{1}".format(dir, filenames[7]))
            print "Calibration files loaded from dir '{0}'.".format(dir)
            return cm1, cm2, d1, d2, r, t, e, f

        except Exception:
            return False

    def stereo_rectify(self, calibration=None, win_size=(352, 288)):
        """

        **SUMMARY**

        Computes rectification transforms for each head
        of a calibrated stereo camera.

        **PARAMETERS**

        calibration - is a tuple os the form (CM1, CM2, D1, D2, R, T, E, F)
        CM1 - Camera Matrix for left camera,
        CM2 - Camera Matrix for right camera,
        D1 - Vector of distortion coefficients for left camera,
        D2 - Vector of distortion coefficients for right camera,
        R - Rotation matrix between the left and the right
            camera coordinate systems,
        T - Translation vector between the left and the right
            coordinate systems of the cameras,
        E - Essential matrix,
        F - Fundamental matrix

        **RETURNS**

        On success returns a a tuple of the format -> (R1, R2, P1, P2, Q, roi)
        R1 - Rectification transform (rotation matrix) for the left camera.
        R2 - Rectification transform (rotation matrix) for the right camera.
        P1 - Projection matrix in the new (rectified) coordinate systems
             for the left camera.
        P2 - Projection matrix in the new (rectified) coordinate systems
             for the right camera.
        Q - disparity-to-depth mapping matrix.

        **EXAMPLE**

        >>> StereoCam = StereoCamera()
        >>> calibration = StereoCam.load_calibration(fname="Stereo1")
        >>> rectification = StereoCam.stereo_rectify(calibration)

        """
        (cm1, cm2, d1, d2, r, t, e, f) = calibration
        r1 = cv.CreateMat(3, 3, cv.CV_64F)
        r2 = cv.CreateMat(3, 3, cv.CV_64F)
        p1 = cv.CreateMat(3, 4, cv.CV_64F)
        p2 = cv.CreateMat(3, 4, cv.CV_64F)
        q = cv.CreateMat(4, 4, cv.CV_64F)

        print "Running stereo rectification..."

        (leftroi, rightroi) = cv.StereoRectify(cm1, cm2, d1, d2, win_size, r,
                                               t, r1, r2, p1, p2, q)
        roi = []
        roi.append(max(leftroi[0], rightroi[0]))
        roi.append(max(leftroi[1], rightroi[1]))
        roi.append(min(leftroi[2], rightroi[2]))
        roi.append(min(leftroi[3], rightroi[3]))
        print "Done."
        return r1, r2, p1, p2, q, roi

    def get_images_undistort(self, img_left, img_right, calibration,
                             rectification, win_size=(352, 288)):
        """
        **SUMMARY**
        Rectify two images from the calibration and rectification parameters.

        **PARAMETERS**
        * *img_left* - Image captured from left camera
                       and needs to be rectified.
        * *img_right* - Image captures from right camera
                       and need to be rectified.
        * *calibration* - A calibration tuple of the format
                          (CM1, CM2, D1, D2, R, T, E, F)
        * *rectification* - A rectification tuple of the format
                           (R1, R2, P1, P2, Q, roi)

        **RETURNS**
        returns rectified images in a tuple -> (img_left,img_right)

        **EXAMPLE**
        >>> StereoCam = StereoCamera()
        >>> calibration = StereoCam.load_calibration(fname="Stereo1")
        >>> rectification = StereoCam.stereo_rectify(calibration)
        >>> img_left = cam_left.get_image()
        >>> img_right = cam_right.get_image()
        >>> rectLeft,rectRight = StereoCam.get_images_undistort(img_left,
                                       img_right,calibration,rectification)
        """
        img_left = img_left.get_matrix()
        img_right = img_right.get_matrix()
        (cm1, cm2, d1, d2, r, t, e, f) = calibration
        (r1, r2, p1, p2, q, roi) = rectification

        dst1 = cv.CloneMat(img_left)
        dst2 = cv.CloneMat(img_right)
        map1x = cv.CreateMat(win_size[1], win_size[0], cv.CV_32FC1)
        map2x = cv.CreateMat(win_size[1], win_size[0], cv.CV_32FC1)
        map1y = cv.CreateMat(win_size[1], win_size[0], cv.CV_32FC1)
        map2y = cv.CreateMat(win_size[1], win_size[0], cv.CV_32FC1)

        #print "Rectifying images..."
        cv.InitUndistortRectifyMap(cm1, d1, r1, p1, map1x, map1y)
        cv.InitUndistortRectifyMap(cm2, d2, r2, p2, map2x, map2y)

        cv.Remap(img_left, dst1, map1x, map1y)
        cv.Remap(img_right, dst2, map2x, map2y)
        return Image(dst1), Image(dst2)

    def get_3d_image(self, left_index, right_index, rpj_mat, method="BM",
                     state=None):
        """
        **SUMMARY**

        This method returns the 3D depth image using reprojectImageTo3D method.

        **PARAMETERS**

        * *left_index* - Index of left camera
        * *right_index* - Index of right camera
        * *rpj_mat* - reprojection Matrix (disparity to depth matrix)
        * *method* - Stereo Correspondonce method to be used.
                   - "BM" - Stereo BM
                   - "SGBM" - Stereo SGBM
        * *state* - dictionary corresponding to parameters of
                    stereo correspondonce.
                    SADWindowSize - odd int
                    n_disparity - int
                    min_disparity  - int
                    preFilterCap - int
                    preFilterType - int (only BM)
                    speckleRange - int
                    speckleWindowSize - int
                    P1 - int (only SGBM)
                    P2 - int (only SGBM)
                    fullDP - Bool (only SGBM)
                    uniquenessRatio - int
                    textureThreshold - int (only BM)


        **RETURNS**

        SimpleCV.Image representing 3D depth Image
        also StereoCamera.Image3D gives OpenCV 3D Depth Image of CV_32F type.

        **EXAMPLE**

        >>> lImage = Image("l.jpg")
        >>> rImage = Image("r.jpg")
        >>> stereo = StereoCamera()
        >>> Q = cv.Load("Q.yml")
        >>> stereo.get_3d_image(1, 2, Q).show()

        >>> state = {"SADWindowSize":9, "n_disparity":112, "min_disparity":-39}
        >>> stereo.get_3d_image(1, 2, Q, "BM", state).show()
        >>> stereo.get_3d_image(1, 2, Q, "SGBM", state).show()
        """
        cv2flag = True
        try:
            import cv2
        except ImportError:
            cv2flag = False
            import cv2.cv as cv
        if cv2flag:
            cam_left = cv2.VideoCapture(left_index)
            cam_right = cv2.VideoCapture(right_index)
            if cam_left.isOpened():
                _, img_left = cam_left.read()
            else:
                warnings.warn("Unable to open left camera")
                return None
            if cam_right.isOpened():
                _, img_right = cam_right.read()
            else:
                warnings.warn("Unable to open right camera")
                return None
            img_left = Image(img_left, cv2image=True)
            img_right = Image(img_right, cv2image=True)
        else:
            cam_left = cv.CaptureFromCAM(left_index)
            cam_right = cv.CaptureFromCAM(right_index)
            img_left = cv.QueryFrame(cam_left)
            if img_left is None:
                warnings.warn("Unable to open left camera")
                return None

            img_right = cv.QueryFrame(cam_right)
            if img_right is None:
                warnings.warn("Unable to open right camera")
                return None

            img_left = Image(img_left, cv2image=True)
            img_right = Image(img_right, cv2image=True)

        del cam_left
        del cam_right

        stereo_images = StereoImage(img_left, img_right)
        image_3d_normalize = stereo_images.get_3d_image(rpj_mat, method, state)
        #self.image_3d = stereo_images.image_3d
        return image_3d_normalize


class AVTCameraThread(threading.Thread):
    camera = None
    running = True
    verbose = False
    lock = None
    logger = None
    framerate = 0

    def __init__(self, camera):
        super(AVTCameraThread, self).__init__()
        self._stop = threading.Event()
        self.camera = camera
        self.lock = threading.Lock()
        self.name = 'Thread-Camera-ID-' + str(self.camera.uniqueid)

    def run(self):
        counter = 0
        timestamp = time.time()

        while self.running:
            self.lock.acquire()
            self.camera.runCommand("AcquisitionStart")
            frame = self.camera._get_frame(1000)

            if frame:
                img = Image(PilImage.fromstring(
                    self.camera.imgformat,
                    (self.camera.width, self.camera.height),
                    frame.ImageBuffer[:int(frame.ImageBufferSize)]))
            self.camera._buffer.appendleft(img)

            self.camera.runCommand("AcquisitionStop")
            self.lock.release()
            counter += 1
            time.sleep(0.01)

            if time.time() - timestamp >= 1:
                self.camera.framerate = counter
                counter = 0
                timestamp = time.time()

    def stop(self):
        self._stop.set()

    def stopped(self):
        return self._stop.isSet()


AVT_CAMERA_ERRORS = [
    ("ePvErrSuccess", "No error"),
    ("ePvErrCameraFault", "Unexpected camera fault"),
    ("ePvErrInternalFault", "Unexpected fault in PvApi or driver"),
    ("ePvErrBadHandle", "Camera handle is invalid"),
    ("ePvErrBadParameter", "Bad parameter to API call"),
    ("ePvErrBadSequence", "Sequence of API calls is incorrect"),
    ("ePvErrNotFound", "Camera or attribute not found"),
    ("ePvErrAccessDenied", "Camera cannot be opened in the specified mode"),
    ("ePvErrUnplugged", "Camera was unplugged"),
    ("ePvErrInvalidSetup", "Setup is invalid (an attribute is invalid)"),
    ("ePvErrResources", "System/network resources or memory not available"),
    ("ePvErrBandwidth", "1394 bandwidth not available"),
    ("ePvErrQueueFull", "Too many frames on queue"),
    ("ePvErrBufferTooSmall", "Frame buffer is too small"),
    ("ePvErrCancelled", "Frame cancelled by user"),
    ("ePvErrDataLost", "The data for the frame was lost"),
    ("ePvErrDataMissing", "Some data in the frame is missing"),
    ("ePvErrTimeout", "Timeout during wait"),
    ("ePvErrOutOfRange", "Attribute value is out of the expected range"),
    ("ePvErrWrongType", "Attribute is not this type (wrong access function)"),
    ("ePvErrForbidden", "Attribute write forbidden at this time"),
    ("ePvErrUnavailable", "Attribute is not available at this time"),
    ("ePvErrFirewall", "A firewall is blocking the traffic (Windows only)"),
]


def pverr(errcode):
    if errcode:
        raise Exception(": ".join(AVT_CAMERA_ERRORS[errcode]))


class AVTCamera(FrameSource):
    """
    **SUMMARY**
    AVTCamera is a ctypes wrapper for the Prosilica/Allied Vision cameras,
    such as the "manta" series.

    These require the PvAVT binary driver from Allied Vision:
    http://www.alliedvisiontec.com/us/products/1108.html

    Note that as of time of writing the new VIMBA driver is not available
    for Mac/Linux - so this uses the legacy PvAVT drive

    Props to Cixelyn, whos py-avt-pvapi module showed how to get much
    of this working https://bitbucket.org/Cixelyn/py-avt-pvapi

    All camera properties are directly from the PvAVT manual -- if not
    specified it will default to whatever the camera state is.  Cameras
    can either by

    **EXAMPLE**
    >>> cam = AVTCamera(0, {"width": 656, "height": 492})
    >>>
    >>> img = cam.get_image()
    >>> img.show()
    """

    _buffer = None  # Buffer to store images
    _buffersize = 10  # Number of images to keep
    # in the rolling image buffer for threads
    _lastimage = None  # Last image loaded into memory
    _thread = None
    _framerate = 0
    threaded = False
    _pvinfo = {}
    _properties = {
        "AcqEndTriggerEvent": ("Enum", "R/W"),
        "AcqEndTriggerMode": ("Enum", "R/W"),
        "AcqRecTriggerEvent": ("Enum", "R/W"),
        "AcqRecTriggerMode": ("Enum", "R/W"),
        "AcqStartTriggerEvent": ("Enum", "R/W"),
        "AcqStartTriggerMode": ("Enum", "R/W"),
        "FrameRate": ("Float32", "R/W"),
        "FrameStartTriggerDelay": ("Uint32", "R/W"),
        "FrameStartTriggerEvent": ("Enum", "R/W"),
        "FrameStartTriggerMode": ("Enum", "R/W"),
        "FrameStartTriggerOverlap": ("Enum", "R/W"),
        "AcquisitionFrameCount": ("Uint32", "R/W"),
        "AcquisitionMode": ("Enum", "R/W"),
        "RecorderPreEventCount": ("Uint32", "R/W"),
        "ConfigFileIndex": ("Enum", "R/W"),
        "ConfigFilePowerup": ("Enum", "R/W"),
        "DSPSubregionBottom": ("Uint32", "R/W"),
        "DSPSubregionLeft": ("Uint32", "R/W"),
        "DSPSubregionRight": ("Uint32", "R/W"),
        "DSPSubregionTop": ("Uint32", "R/W"),
        "DefectMaskColumnEnable": ("Enum", "R/W"),
        "ExposureAutoAdjustTol": ("Uint32", "R/W"),
        "ExposureAutoAlg": ("Enum", "R/W"),
        "ExposureAutoMax": ("Uint32", "R/W"),
        "ExposureAutoMin": ("Uint32", "R/W"),
        "ExposureAutoOutliers": ("Uint32", "R/W"),
        "ExposureAutoRate": ("Uint32", "R/W"),
        "ExposureAutoTarget": ("Uint32", "R/W"),
        "ExposureMode": ("Enum", "R/W"),
        "ExposureValue": ("Uint32", "R/W"),
        "GainAutoAdjustTol": ("Uint32", "R/W"),
        "GainAutoMax": ("Uint32", "R/W"),
        "GainAutoMin": ("Uint32", "R/W"),
        "GainAutoOutliers": ("Uint32", "R/W"),
        "GainAutoRate": ("Uint32", "R/W"),
        "GainAutoTarget": ("Uint32", "R/W"),
        "GainMode": ("Enum", "R/W"),
        "GainValue": ("Uint32", "R/W"),
        "LensDriveCommand": ("Enum", "R/W"),
        "LensDriveDuration": ("Uint32", "R/W"),
        "LensVoltage": ("Uint32", "R/V"),
        "LensVoltageControl": ("Uint32", "R/W"),
        "IrisAutoTarget": ("Uint32", "R/W"),
        "IrisMode": ("Enum", "R/W"),
        "IrisVideoLevel": ("Uint32", "R/W"),
        "IrisVideoLevelMax": ("Uint32", "R/W"),
        "IrisVideoLevelMin": ("Uint32", "R/W"),
        "VsubValue": ("Uint32", "R/C"),
        "WhitebalAutoAdjustTol": ("Uint32", "R/W"),
        "WhitebalAutoRate": ("Uint32", "R/W"),
        "WhitebalMode": ("Enum", "R/W"),
        "WhitebalValueRed": ("Uint32", "R/W"),
        "WhitebalValueBlue": ("Uint32", "R/W"),
        "EventAcquisitionStart": ("Uint32", "R/C 40000"),
        "EventAcquisitionEnd": ("Uint32", "R/C 40001"),
        "EventFrameTrigger": ("Uint32", "R/C 40002"),
        "EventExposureEnd": ("Uint32", "R/C 40003"),
        "EventAcquisitionRecordTrigger": ("Uint32", "R/C 40004"),
        "EventSyncIn1Rise": ("Uint32", "R/C 40010"),
        "EventSyncIn1Fall": ("Uint32", "R/C 40011"),
        "EventSyncIn2Rise": ("Uint32", "R/C 40012"),
        "EventSyncIn2Fall": ("Uint32", "R/C 40013"),
        "EventSyncIn3Rise": ("Uint32", "R/C 40014"),
        "EventSyncIn3Fall": ("Uint32", "R/C 40015"),
        "EventSyncIn4Rise": ("Uint32", "R/C 40016"),
        "EventSyncIn4Fall": ("Uint32", "R/C 40017"),
        "EventOverflow": ("Uint32", "R/C 65534"),
        "EventError": ("Uint32", "R/C"),
        "EventNotification": ("Enum", "R/W"),
        "EventSelector": ("Enum", "R/W"),
        "EventsEnable1": ("Uint32", "R/W"),
        "BandwidthCtrlMode": ("Enum", "R/W"),
        "ChunkModeActive": ("Boolean", "R/W"),
        "NonImagePayloadSize": ("Unit32", "R/V"),
        "PayloadSize": ("Unit32", "R/V"),
        "StreamBytesPerSecond": ("Uint32", "R/W"),
        "StreamFrameRateConstrain": ("Boolean", "R/W"),
        "StreamHoldCapacity": ("Uint32", "R/V"),
        "StreamHoldEnable": ("Enum", "R/W"),
        "TimeStampFrequency": ("Uint32", "R/C"),
        "TimeStampValueHi": ("Uint32", "R/V"),
        "TimeStampValueLo": ("Uint32", "R/V"),
        "Height": ("Uint32", "R/W"),
        "RegionX": ("Uint32", "R/W"),
        "RegionY": ("Uint32", "R/W"),
        "Width": ("Uint32", "R/W"),
        "PixelFormat": ("Enum", "R/W"),
        "TotalBytesPerFrame": ("Uint32", "R/V"),
        "BinningX": ("Uint32", "R/W"),
        "BinningY": ("Uint32", "R/W"),
        "CameraName": ("String", "R/W"),
        "DeviceFirmwareVersion": ("String", "R/C"),
        "DeviceModelName": ("String", "R/W"),
        "DevicePartNumber": ("String", "R/C"),
        "DeviceSerialNumber": ("String", "R/C"),
        "DeviceVendorName": ("String", "R/C"),
        "FirmwareVerBuild": ("Uint32", "R/C"),
        "FirmwareVerMajor": ("Uint32", "R/C"),
        "FirmwareVerMinor": ("Uint32", "R/C"),
        "PartClass": ("Uint32", "R/C"),
        "PartNumber": ("Uint32", "R/C"),
        "PartRevision": ("String", "R/C"),
        "PartVersion": ("String", "R/C"),
        "SerialNumber": ("String", "R/C"),
        "SensorBits": ("Uint32", "R/C"),
        "SensorHeight": ("Uint32", "R/C"),
        "SensorType": ("Enum", "R/C"),
        "SensorWidth": ("Uint32", "R/C"),
        "UniqueID": ("Uint32", "R/C"),
        "Strobe1ControlledDuration": ("Enum", "R/W"),
        "Strobe1Delay": ("Uint32", "R/W"),
        "Strobe1Duration": ("Uint32", "R/W"),
        "Strobe1Mode": ("Enum", "R/W"),
        "SyncIn1GlitchFilter": ("Uint32", "R/W"),
        "SyncInLevels": ("Uint32", "R/V"),
        "SyncOut1Invert": ("Enum", "R/W"),
        "SyncOut1Mode": ("Enum", "R/W"),
        "SyncOutGpoLevels": ("Uint32", "R/W"),
        "DeviceEthAddress": ("String", "R/C"),
        "HostEthAddress": ("String", "R/C"),
        "DeviceIPAddress": ("String", "R/C"),
        "HostIPAddress": ("String", "R/C"),
        "GvcpRetries": ("Uint32", "R/W"),
        "GvspLookbackWindow": ("Uint32", "R/W"),
        "GvspResentPercent": ("Float32", "R/W"),
        "GvspRetries": ("Uint32", "R/W"),
        "GvspSocketBufferCount": ("Enum", "R/W"),
        "GvspTimeout": ("Uint32", "R/W"),
        "HeartbeatInterval": ("Uint32", "R/W"),
        "HeartbeatTimeout": ("Uint32", "R/W"),
        "MulticastEnable": ("Enum", "R/W"),
        "MulticastIPAddress": ("String", "R/W"),
        "PacketSize": ("Uint32", "R/W"),
        "StatDriverType": ("Enum", "R/V"),
        "StatFilterVersion": ("String", "R/C"),
        "StatFrameRate": ("Float32", "R/V"),
        "StatFramesCompleted": ("Uint32", "R/V"),
        "StatFramesDropped": ("Uint32", "R/V"),
        "StatPacketsErroneous": ("Uint32", "R/V"),
        "StatPacketsMissed": ("Uint32", "R/V"),
        "StatPacketsReceived": ("Uint32", "R/V"),
        "StatPacketsRequested": ("Uint32", "R/V"),
        "StatPacketResent": ("Uint32", "R/V")
    }

    class AVTCameraInfo(ct.Structure):
        """
        AVTCameraInfo is an internal ctypes.Structure-derived class which
        contains metadata about cameras on the local network.

        Properties include:
        * UniqueId
        * CameraName
        * ModelName
        * PartNumber
        * SerialNumber
        * FirmwareVersion
        * PermittedAccess
        * InterfaceId
        * InterfaceType
        """
        _fields_ = [
            ("StructVer", ct.c_ulong),
            ("UniqueId", ct.c_ulong),
            ("CameraName", ct.c_char * 32),
            ("ModelName", ct.c_char * 32),
            ("PartNumber", ct.c_char * 32),
            ("SerialNumber", ct.c_char * 32),
            ("FirmwareVersion", ct.c_char * 32),
            ("PermittedAccess", ct.c_long),
            ("InterfaceId", ct.c_ulong),
            ("InterfaceType", ct.c_int)
        ]

        def __repr__(self):
            return "<SimpleCV.Camera.AVTCameraInfo " \
                   "- UniqueId: %s>" % self.UniqueId

    class AVTFrame(ct.Structure):
        _fields_ = [
            ("image_buffer", ct.POINTER(ct.c_char)),
            ("image_buffer_size", ct.c_ulong),
            ("ancillary_buffer", ct.c_int),
            ("ancillary_buffer_size", ct.c_int),
            ("Context", ct.c_int * 4),
            ("_reserved1", ct.c_ulong * 8),

            ("Status", ct.c_int),
            ("ImageSize", ct.c_ulong),
            ("AncillarySize", ct.c_ulong),
            ("Width", ct.c_ulong),
            ("Height", ct.c_ulong),
            ("RegionX", ct.c_ulong),
            ("RegionY", ct.c_ulong),
            ("Format", ct.c_int),
            ("BitDepth", ct.c_ulong),
            ("BayerPattern", ct.c_int),
            ("FrameCount", ct.c_ulong),
            ("TimestampLo", ct.c_ulong),
            ("TimestampHi", ct.c_ulong),
            ("_reserved2", ct.c_ulong * 32)
        ]

        def __init__(self, buffersize):
            self.image_buffer = ct.create_string_buffer(buffersize)
            self.image_buffer_size = ct.c_ulong(buffersize)
            self.ancillary_buffer = 0
            self.ancillary_buffer_size = 0
            self.img = None
            #self.hasImage = False
            self.frame = None

    def __del__(self):
        #This function should disconnect from the AVT Camera
        pverr(self.dll.PvCameraClose(self.handle))

    def __init__(self, camera_id=-1, properties={}, threaded=False):
        #~ super(AVTCamera, self).__init__()

        if SYSTEM == "Windows":
            self.dll = ct.windll.LoadLibrary("PvAPI.dll")
        elif SYSTEM == "Darwin":
            self.dll = ct.CDLL("libPvAPI.dylib", ct.RTLD_GLOBAL)
        else:
            self.dll = ct.CDLL("libPvAPI.so")

        if not self._pvinfo.get("initialized", False):
            self.dll.PvInitialize()
            self._pvinfo['initialized'] = True
        #initialize.  Note that we rely on listAllCameras being the next
        #call, since it blocks on cameras initializing

        camlist = self.list_all_cameras()

        if not len(camlist):
            raise Exception("Couldn't find any cameras with the PvAVT "
                            "driver. Use SampleViewer to confirm you have one "
                            "connected.")

        if camera_id < 9000:  # camera was passed as an index reference
            if camera_id == -1:  # accept -1 for "first camera"
                camera_id = 0

            camera_id = camlist[camera_id].UniqueId

        camera_id = long(camera_id)
        self.handle = ct.c_uint()
        init_count = 0
        #wait until camera is availble:
        while self.dll.PvCameraOpen(camera_id, 0, ct.byref(self.handle)) != 0:
            if init_count > 4:  # Try to connect 5 times before giving up
                raise Exception('Could not connect to camera, please \
                                 verify with SampleViewer you can connect')
            init_count += 1
            time.sleep(1)  # sleep and retry to connect to camera in a second

        pverr(self.dll.PvCaptureStart(self.handle))
        self.uniqueid = camera_id

        self.set_property("AcquisitionMode", "SingleFrame")
        self.set_property("FrameStartTriggerMode", "Freerun")

        if properties.get("mode", "RGB") == 'gray':
            self.set_property("PixelFormat", "Mono8")
        else:
            self.set_property("PixelFormat", "Rgb24")

        #give some compatablity with other cameras
        if properties.get("mode", ""):
            properties.pop("mode")

        if properties.get("height", ""):
            properties["Height"] = properties["height"]
            properties.pop("height")

        if properties.get("width", ""):
            properties["Width"] = properties["width"]
            properties.pop("width")

        for prop in properties:
            self.set_property(prop, properties[prop])

        if threaded:
            self._thread = AVTCameraThread(self)
            self._thread.daemon = True
            self._buffer = deque(maxlen=self._buffersize)
            self._thread.start()
            self.threaded = True

        self._refresh_frame_stats()

    def restart(self):
        """
        This tries to restart the camera thread
        """
        self._thread.stop()
        self._thread = AVTCameraThread(self)
        self._thread.daemon = True
        self._buffer = deque(maxlen=self._buffersize)
        self._thread.start()

    def list_all_cameras(self):
        """
        **SUMMARY**
        List all cameras attached to the host

        **RETURNS**
        List of AVTCameraInfo objects, otherwise empty list

        """
        camlist = (self.AVTCameraInfo * 100)()
        starttime = time.time()
        while int(camlist[0].UniqueId) == 0 and time.time() - starttime < 10:
            self.dll.PvCameraListEx(ct.byref(camlist), 100,
                                    None, ct.sizeof(self.AVTCameraInfo))
            time.sleep(0.1)  # keep checking for cameras until timeout

        return [cam for cam in camlist if cam.UniqueId != 0]

    def run_command(self, command):
        """
        **SUMMARY**
        Runs a PvAVT Command on the camera

        Valid Commands include:
        * FrameStartTriggerSoftware
        * AcquisitionAbort
        * AcquisitionStart
        * AcquisitionStop
        * ConfigFileLoad
        * ConfigFileSave
        * TimeStampReset
        * TimeStampValueLatch

        **RETURNS**

        0 on success

        **EXAMPLE**
        >>>c = AVTCamera()
        >>>c.run_command("TimeStampReset")
        """
        return self.dll.PvCommandRun(self.handle, command)

    def get_property(self, name):
        """
        **SUMMARY**
        This retrieves the value of the AVT Camera attribute

        There are around 140 properties for the AVT Camera, so reference the
        AVT Camera and Driver Attributes pdf that is provided with
        the driver for detailed information

        Note that the error codes are currently ignored, so empty values
        may be returned.

        **EXAMPLE**
        >>>c = AVTCamera()
        >>>print c.get_property("ExposureValue")
        """
        valtype, _ = self._properties.get(name, (None, None))

        if not valtype:
            return None

        val = ''
        err = 0
        if valtype == "Enum":
            val = ct.create_string_buffer(100)
            vallen = ct.c_long()
            err = self.dll.PvAttrEnumGet(self.handle, name, val,
                                         100, ct.byref(vallen))
            val = str(val[:vallen.value])
        elif valtype == "Uint32":
            val = ct.c_uint()
            err = self.dll.PvAttrUint32Get(self.handle, name, ct.byref(val))
            val = int(val.value)
        elif valtype == "Float32":
            val = ct.c_float()
            err = self.dll.PvAttrFloat32Get(self.handle, name, ct.byref(val))
            val = float(val.value)
        elif valtype == "String":
            val = ct.create_string_buffer(100)
            vallen = ct.c_long()
            err = self.dll.PvAttrStringGet(self.handle, name, val,
                                           100, ct.byref(vallen))
            val = str(val[:vallen.value])
        elif valtype == "Boolean":
            val = ct.c_bool()
            err = self.dll.PvAttrBooleanGet(self.handle, name, ct.byref(val))
            val = bool(val.value)

        #TODO, handle error codes

        return val

    #TODO, implement the PvAttrRange* functions
    #def get_property_range(self, name)

    def get_all_properties(self):
        """
        **SUMMARY**
        This returns a dict with the name and current value of the
        documented PvAVT attributes

        CAVEAT: it addresses each of the properties individually, so
        this may take time to run if there's network latency

        **EXAMPLE**
        >>>c = AVTCamera(0)
        >>>props = c.get_all_properties()
        >>>print props['ExposureValue']

        """
        props = {}
        for name in self._properties.keys():
            props[name] = self.get_property(name)
        return props

    def set_property(self, name, value, skip_buffer_size_check=False):
        """
        **SUMMARY**
        This sets the value of the AVT Camera attribute.

        There are around 140 properties for the AVT Camera, so reference the
        AVT Camera and Driver Attributes pdf that is provided with
        the driver for detailed information

        By default, we will also refresh the height/width and bytes per
        frame we're expecting -- you can manually bypass this if you want speed

        Returns the raw PvAVT error code (0 = success)

        **Example**
        >>>c = AVTCamera()
        >>>c.set_property("ExposureValue", 30000)
        >>>c.get_image().show()
        """
        valtype, _ = self._properties.get(name, (None, None))

        if not valtype:
            return None

        if valtype == "Uint32":
            err = self.dll.PvAttrUint32Set(self.handle, name,
                                           ct.c_uint(int(value)))
        elif valtype == "Float32":
            err = self.dll.PvAttrFloat32Set(self.handle, name,
                                            ct.c_float(float(value)))
        elif valtype == "Enum":
            err = self.dll.PvAttrEnumSet(self.handle, name, str(value))
        elif valtype == "String":
            err = self.dll.PvAttrStringSet(self.handle, name, str(value))
        elif valtype == "Boolean":
            err = self.dll.PvAttrBooleanSet(self.handle, name,
                                            ct.c_bool(bool(value)))

        #just to be safe, re-cache the camera metadata
        if not skip_buffer_size_check:
            self._refresh_frame_stats()

        return err

    def get_image(self):
        """
        **SUMMARY**
        Extract an Image from the Camera, returning the value. No matter
        what the image characteristics on the camera, the Image returned
        will be RGB 8 bit depth, if camera is in greyscale mode it will
        be 3 identical channels.

        **EXAMPLE**
        >>>c = AVTCamera()
        >>>c.get_image().show()
        """

        if self.threaded:
            self._thread.lock.acquire()
            try:
                img = self._buffer.pop()
                self._lastimage = img
            except IndexError:
                img = self._lastimage
            self._thread.lock.release()

        else:
            self.run_command("AcquisitionStart")
            frame = self._get_frame()
            img = Image(PilImage.fromstring(
                self.imgformat, (self.width, self.height),
                frame.image_buffer[:int(frame.image_buffer_size)]))
            self.run_command("AcquisitionStop")

        return img

    def setup_async_mode(self):
        self.set_property('AcquisitionMode', 'SingleFrame')
        self.set_property('FrameStartTriggerMode', 'Software')

    def setup_sync_mode(self):
        self.set_property('AcquisitionMode', 'Continuous')
        self.set_property('FrameStartTriggerMode', 'FreeRun')

    def unbuffer(self):
        img = Image(PilImage.fromstring(
            self.imgformat, (self.width, self.height),
            self.frame.ImageBuffer[:int(self.frame.ImageBufferSize)]))

        return img

    def _refresh_frame_stats(self):
        self.width = self.get_property("Width")
        self.height = self.get_property("Height")
        self.buffersize = self.get_property("TotalBytesPerFrame")
        self.pixelformat = self.get_property("PixelFormat")
        self.imgformat = 'RGB'
        if self.pixelformat == 'Mono8':
            self.imgformat = 'L'

    def _get_frame(self, timeout=2000):
        #return the AVTFrame object from the camera, timeout in ms
        #need to multiply by bitdepth
        try:
            frame = self.AVTFrame(self.buffersize)
            pverr(self.dll.PvCaptureQueueFrame(self.handle,
                                               ct.byref(frame), None))
            try:
                pverr(self.dll.PvCaptureWaitForFrameDone(self.handle,
                                                         ct.byref(frame),
                                                         timeout))
            except Exception, e:
                print "Exception waiting for frame: ", e
                raise e

        except Exception, e:
            print "Exception aquiring frame: ", e
            raise e

        return frame


class GigECamera(Camera):
    """
        GigE Camera driver via Aravis
    """
    plist = [
        'available_pixel_formats',
        'available_pixel_formats_as_display_names',
        'available_pixel_formats_as_strings',
        'binning',
        'device_id',
        'exposure_time',
        'exposure_time_bounds',
        'frame_rate',
        'frame_rate_bounds',
        'gain',
        'gain_bounds',
        'height_bounds',
        'model_name',
        'payload',
        'pixel_format',
        'pixel_format_as_string',
        'region',
        'sensor_size',
        'trigger_source',
        'vendor_name',
        'width_bounds'
    ]

    #def __init__(self, camera_id=None, properties={}, threaded=False):
    def __init__(self, properties={}, threaded=False):
        if not ARAVIS_ENABLED:
            print "GigE is supported by the Aravis library, download and \
                   build from https://github.com/sightmachine/aravis"
            print "Note that you need to set GI_TYPELIB_PATH=$GI_TYPELIB_PATH:\
                   (PATH_TO_ARAVIS)/src for the GObject Introspection"
            return

        self._cam = Aravis.Camera.new(None)

        self._pixel_mode = "RGB"
        if properties.get("mode", False):
            self._pixel_mode = properties.pop("mode")

        if self._pixel_mode == "gray":
            self._cam.set_pixel_format(Aravis.PIXEL_FORMAT_MONO_8)
        else:
            #we'll use bayer (basler cams)
            self._cam.set_pixel_format(Aravis.PIXEL_FORMAT_BAYER_BG_8)
            #TODO, deal with other pixel formats

        if properties.get("roi", False):
            roi = properties['roi']
            self._cam.set_region(*roi)
            #TODO, check sensor size

        if properties.get("width", False):
            #TODO, set internal function to scale results of getimage
            pass

        if properties.get("framerate", False):
            self._cam.set_frame_rate(properties['framerate'])

        self._stream = self._cam.create_stream(None, None)

        payload = self._cam.get_payload()
        self._stream.push_buffer(Aravis.Buffer.new_allocate(payload))
        [_, _, width, height] = self._cam.get_region()
        self._height, self._width = height, width

    def get_image(self):
        if not ARAVIS_ENABLED:
            warnings.warn("Initializing failed, Aravis library not found.")
            return
        try:
            import cv2
        except ImportError:
            logger.warning("Can't work OpenCV >= 2.3.0")
            return None
        camera = self._cam
        camera.start_acquisition()
        buff = self._stream.pop_buffer()
        self.capture_time = buff.timestamp_ns / 1000000.0
        img = np.fromstring(ct.string_at(buff.data_address(), buff.size),
                            dtype=np.uint8).reshape(self._height, self._width)
        rgb = cv2.cvtColor(img, cv2.COLOR_BAYER_BG2BGR)
        self._stream.push_buffer(buff)
        camera.stop_acquisition()
        #TODO, we should handle software triggering
        #(separate capture and get image events)

        return Image(rgb)

    def get_property_list(self):
        return self.plist

    def get_property(self, name=None):
        '''
        This function get's the properties availble to the camera

        Usage:
        > camera.get_property('region')
        > (0, 0, 128, 128)

        Available Properties:
        see function camera.get_property_list()
        '''

        if name is None:
            print "You need to provide a property, available properties are:"
            print ""
            for prop in self.get_property_list():
                print prop
            return

        stringval = "get_{}".format(name)
        try:
            return getattr(self._cam, stringval)()
        except Exception:
            print 'Property {} does not appear to exist'.format(name)
            return None

    def set_property(self, name=None, *args):
        '''
        This function sets the property available to the camera

        Usage:
        > camera.set_property('region',(256,256))

        Available Properties:
        see function camera.get_property_list()
        '''

        if name is None:
            print "You need to provide a property, available properties are:"
            print ""
            for prop in self.get_property_list():
                print prop
            return

        if len(args) <= 0:
            print "You must provide a value to set"
            return

        stringval = "set_{}".format(name)
        try:
            return getattr(self._cam, stringval)(*args)
        except Exception:
            print 'Property {} does not appear to exist or value\
                   is not in correct format'.format(name)
            return None

    def get_all_properties(self):
        '''
        This function just prints out all the properties available to the
        camera
        '''

        for prop in self.get_property_list():
            print "{}: {}".format(prop, self.get_property(prop))
