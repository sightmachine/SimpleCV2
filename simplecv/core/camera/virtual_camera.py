import os
import time

import cv2

from simplecv.core.camera.frame_source import FrameSource
from simplecv.base import logger
from simplecv.image_set import ImageSet
from simplecv.factory import Factory


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
        >>> vc1 = VirtualCamera("video.mpg", "video")
        >>> vc2 = VirtualCamera("./path_to_images/", "imageset")
        >>> vc3 = VirtualCamera("video.mpg", "video", 300)
        >>> vc4 = VirtualCamera("./imgs", "directory")


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
            return

        else:
            if isinstance(self.source, str) and not os.path.exists(
                    self.source):
                print 'Error: In VirtualCamera()\n\t"%s" \
                       was not found.' % self.source
                return

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
                logger.warn('Virtual Camera is unable to figure out \
                    the contents of your ImageSet, it must be a directory, \
                    list of directories, or an ImageSet object')

        elif self.sourcetype == 'video':
            self.capture = cv2.VideoCapture(self.source)
            self.capture.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, self.start - 1)

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
            self.counter += 1
            return Factory.Image(self.source, camera=self)

        elif self.sourcetype == 'imageset':
            print len(self.source)
            img = self.source[self.counter % len(self.source)]
            self.counter += 1
            return img

        elif self.sourcetype == 'video':
            ret_value, img = self.capture.read()
            return Factory.Image(img, camera=self) if ret_value else None

        elif self.sourcetype == 'directory':
            img = self.find_lastest_image(self.source, 'bmp')
            self.counter += 1
            return Factory.Image(img, camera=self)

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
                self.capture.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, self.start - 1)
            else:
                if start == 0:
                    start = 1
                self.capture.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, start - 1)
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
            number_frame = int(self.capture.get(cv2.cv.CV_CAP_PROP_POS_FRAMES))
            self.capture.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, frame - 1)
            img = self.get_image()
            self.capture.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, number_frame)
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
            number_frame = int(self.capture.get(cv2.cv.CV_CAP_PROP_POS_FRAMES))
            self.capture.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,
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
        >>> while i < 60:
        ...     cam.get_image().show()
        ...     i += 1
        >>> cam.skip_frames(100)
        >>> cam.get_frame_number()

        """
        if self.sourcetype == 'video':
            return int(self.capture.get(cv2.cv.CV_CAP_PROP_POS_FRAMES))
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
        ...     cam.get_image().show()
        ...     i+=1
        >>> cam.skip_frames(100)
        >>> cam.get_current_play_time()

        """
        if self.sourcetype == 'video':
            return int(self.capture.get(cv2.cv.CV_CAP_PROP_POS_MSEC))
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
