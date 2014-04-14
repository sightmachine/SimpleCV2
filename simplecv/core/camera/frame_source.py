import pickle
import time

import cv2
import numpy as np
import pygame as pg

from simplecv.base import logger
from simplecv.color import Color
from simplecv.display import Display
from simplecv.factory import Factory


class FrameSource(object):
    """
    **SUMMARY**

    An abstract Camera-type class, for handling multiple types of video input.
    Any sources of images inheirit from it


    """

    def __init__(self):
        self._calib_matrix = ""  # Intrinsic calibration matrix
        self._dist_coeff = ""  # Distortion matrix
        self._thread_capture_time = ''  # when the last picture was taken
        self.capture_time = ''  # timestamp of the last aquired image

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
        image_points = np.zeros((n_boards * board_n, 2), dtype=np.float32)
        object_points = np.zeros((n_boards * board_n, 3), dtype=np.float32)
        point_counts = np.zeros((n_boards, 1), dtype=np.float32)

        # capture frames of specified properties
        # and modification of matrix values

        successes = 0
        img_idx = 0
        # capturing required number of views
        while successes < n_boards:
            img = image_list[img_idx]
            retval, corners = cv2.findChessboardCorners(
                image=img.get_gray_ndarray(),
                patternSize=board_sz,
                flags=cv2.CALIB_CB_ADAPTIVE_THRESH |
                cv2.CALIB_CB_FILTER_QUADS)

            #if not retval:
            #FIXME: check retval should be implemented

            cv2.cornerSubPix(img.get_gray_ndarray(),
                             corners, (11, 11), (-1, -1),
                             (cv2.cv.CV_TERMCRIT_EPS +
                              cv2.cv.CV_TERMCRIT_ITER, 30, 0.1))
            # if got a good image, draw chess board
            #if found == 1:
            #    corner_count = len(corners)
            #    z = z + 1

            # if got a good image, add to matrix
            if len(corners) == board_n:
                step = successes * board_n
                k = step
                for j in range(board_n):
                    image_points[k, 0] = corners[j, 0, 0]
                    image_points[k, 1] = corners[j, 0, 1]
                    object_points[k, 0] = grid_sz * (float(j) / board_w)
                    object_points[k, 1] = grid_sz * (float(j) % board_w)
                    object_points[k, 2] = 0.0
                    k += 1
                point_counts[successes, 0] = board_n
                successes += 1

        # now assigning new matrices according to view_count
        if successes < warn_thresh:
            logger.warning("FrameSource.calibrate: You have %d good images "
                           "for calibration we recommend at least %d",
                           successes, warn_thresh)

        object_points2 = np.zeros((successes * board_n, 3), dtype=np.float32)
        image_points2 = np.zeros((successes * board_n, 2), dtype=np.float32)
        point_counts2 = np.zeros((successes, 1), dtype=np.float32)

        for i in range(successes * board_n):
            image_points2[i, 0] = image_points[i, 0]
            image_points2[i, 1] = image_points[i, 1]
            object_points2[i, 0] = object_points[i, 0]
            object_points2[i, 1] = object_points[i, 1]
            object_points2[i, 2] = object_points[i, 2]
        for i in range(successes):
            point_counts2[i, 0] = point_counts[i, 0]

        # camera calibration
        _, cam_matrix, dist_cft, _, _, _ = cv2.calibrateCamera(object_points2,
                                                               image_points2,
                                                               img.size)

        self._calib_matrix = cam_matrix
        self._dist_coeff = dist_cft
        return cam_matrix

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
        if not isinstance(self._calib_matrix, np.ndarray) \
                or not isinstance(self._dist_coeff, np.ndarray):
            logger.warning("FrameSource.undistort: This operation requires "
                           "calibration, please load the calibration matrix")
            return None

        if isinstance(image_or_2darray, Factory.Image):
            in_img = image_or_2darray  # we have an image
            ret_val = cv2.undistort(in_img.get_ndarray(), self._calib_matrix,
                                    self._dist_coeff)
            return Factory.Image(ret_val)
        elif isinstance(image_or_2darray, np.ndarray):
            mat = image_or_2darray
            return cv2.undistort(mat, self._calib_matrix, self._dist_coeff)
        else:
            logger.warning("FrameSource.undistort: image_or_2darray should be "
                           "Image or numpy.ndarray")
            return None

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
        if not isinstance(self._calib_matrix, np.ndarray):
            logger.warning("FrameSource.save_calibration: "
                           "No calibration matrix present, can't save.")
        else:
            output = open(filename + "Intrinsic.bin", 'wb')
            pickle.dump(self._calib_matrix, output)
            output.close()

        if not isinstance(self._dist_coeff, np.ndarray):
            logger.warning("FrameSource.save_calibration: "
                           "No calibration distortion present, can't save.")
        else:
            output = open(filename + "Distortion.bin", 'wb')
            pickle.dump(self._dist_coeff, output)
            output.close()

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
        with open(filename + "Intrinsic.bin") as f:
            self._calib_matrix = pickle.load(f)
        with open(filename + "Distortion.bin") as f:
            self._dist_coeff = pickle.load(f)
        if isinstance(self._dist_coeff, np.ndarray) \
                and isinstance(self._calib_matrix, np.ndarray):
            return True
        else:
            return False

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
        display = Display(image.size)
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
                display.done = True

        pg.quit()

    def get_thread_capture_time(self):
        return self._thread_capture_time

    def set_thread_capture_time(self, capture_time):
        self._thread_capture_time = capture_time
