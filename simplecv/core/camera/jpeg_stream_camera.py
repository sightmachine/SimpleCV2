from cStringIO import StringIO
import re
import threading
import time
import urllib2


from simplecv.base import logger
from simplecv.core.camera.frame_source import FrameSource
from simplecv.factory import Factory

try:
    from PIL import Image as PilImage
except:
    import Image as PilImage


class JpegStreamReader(threading.Thread):
    """
    **SUMMARY**

    A Threaded class for pulling down JPEG streams and breaking up the images.
    This is handy for reading the stream of images from a IP Camera.

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
        if not self.cam_thread.get_thread_capture_time():
            now = time.time()
            while not self.cam_thread.get_thread_capture_time():
                if time.time() - now > 5:
                    logger.warn("Timeout fetching JpegStream at " + self.url)
                    return
                time.sleep(0.1)

        self.capture_time = self.cam_thread.get_thread_capture_time()
        return Factory.Image(PilImage.open(
            StringIO(self.cam_thread.current_frame)), camera=self)
