from cStringIO import StringIO
import cStringIO
import glob
import itertools
import math
import os
import random
import re
import sys
import tempfile
import time
import types
import urllib2
import warnings

from numpy import int32
from numpy import uint8
import cv2
import numpy as np
import scipy.cluster.vq as scv
import scipy.linalg as nla  # for linear algebra / least squares
import scipy.ndimage as ndimage
import scipy.spatial.distance as spsd
import scipy.stats.stats as sss  # for auto white balance

from simplecv.base import (init_options_handler, logger,
                           is_number, is_tuple, int_to_bin,
                           IMAGE_FORMATS, LAUNCH_PATH, MAX_DIMENSION)
from simplecv.color import Color, ColorCurve
from simplecv.display import Display
from simplecv.drawing_layer import DrawingLayer
from simplecv.exif import process_file
from simplecv.features.features import FeatureSet, Feature
from simplecv.linescan import LineScan
from simplecv.stream import JpegStreamer, VideoStream
from simplecv.tracking.cam_shift_tracker import camshiftTracker
from simplecv.tracking.lk_tracker import lkTracker
from simplecv.tracking.mf_tracker import mfTracker
from simplecv.tracking.surf_tracker import surfTracker
from simplecv.tracking.track_set import TrackSet


if not init_options_handler.headless:
    import pygame as pg

PIL_ENABLED = True
try:
    from PIL import Image as PilImage
    from PIL.GifImagePlugin import getheader, getdata
except ImportError:
    try:
        import Image as PilImage
        from GifImagePlugin import getheader, getdata
    except ImportError:
        PIL_ENABLED = False

OCR_ENABLED = True
try:
    import tesseract
except ImportError:
    OCR_ENABLED = False

ZXING_ENABLED = True
try:
    import zxing
except ImportError:
    ZXING_ENABLED = False


class ColorSpace(object):
    """
    **SUMMARY**

    The colorspace  class is used to encapsulate the color space of a given
    image.
    This class acts like C/C++ style enumerated type.


    See: http://stackoverflow.com/questions/2122706/
    detect-color-space-with-opencv
    """
    UNKNOWN = 0
    BGR = 1
    GRAY = 2
    RGB = 3
    HLS = 4
    HSV = 5
    XYZ = 6
    YCrCb = 7


class ImageSet(list):
    """
    **SUMMARY**

    This is an abstract class for keeping a list of images.  It has a few
    advantages in that you can use it to auto load data sets from a directory
    or the net.

    Keep in mind it inherits from a list too, so all the functionality a
    normal python list has this will too.

    **EXAMPLES**


    >>> imgs = ImageSet()
    >>> imgs.download("ninjas")
    >>> imgs.show(ninjas)


    or you can load a directory path:

    >>> imgs = ImageSet('/path/to/imgs/')
    >>> imgs.show()

    This will download and show a bunch of random ninjas.  If you want to
    save all those images locally then just use:

    >>> imgs.save()

    You can also load up the sample images that come with simplecv as:

    >>> imgs = ImageSet('samples')
    >>> imgs.filelist
    >>> logo = imgs.find('simplecv.png')

    **TO DO**

    Eventually this should allow us to pull image urls / paths from csv files.
    The method also allow us to associate an arbitraty bunch of data with each
    image, and on load/save pickle that data or write it to a CSV file.

    """
    filelist = None

    def __init__(self, directory=None):
        if not directory:
            return

        if isinstance(directory, list):
            if isinstance(directory[0], Image):
                super(ImageSet, self).__init__(directory)
            elif isinstance(directory[0], str) \
                    or isinstance(directory[0], unicode):
                super(ImageSet, self).__init__(map(Image, directory))

        elif directory.lower() == 'samples' or directory.lower() == 'sample':
            pth = LAUNCH_PATH
            pth = os.path.realpath(pth)
            directory = os.path.join(pth, 'data/sampleimages')
            self.load(directory)
        else:
            self.load(directory)

    def download(self, tag=None, number=10, size='thumb'):
        """
        **SUMMARY**

        This function downloads images from Google Image search based
        on the tag you provide. The number is the number of images you
        want to have in the list. Valid values for size are 'thumb', 'small',
        'medium', 'large' or a tuple of exact dimensions i.e. (640,480).
        Note that 'thumb' is exceptionally faster than others.

        .. Warning::
          This requires the python library Beautiful Soup to be installed
          http://www.crummy.com/software/BeautifulSoup/

        **PARAMETERS**

        * *tag* - A string of tag values you would like to download.
        * *number* - An integer of the number of images to try and download.
        * *size* - the size of the images to download. Valid options a tuple
          of the exact size or a string of the following approximate sizes:

          * thumb ~ less than 128x128
          * small  ~ approximately less than 640x480 but larger than 128x128
          * medium ~  approximately less than 1024x768 but larger than 640x480.
          * large ~ > 1024x768

        **RETURNS**

        Nothing - but caches local copy of images.

        **EXAMPLE**

        >>> imgs = ImageSet()
        >>> imgs.download("ninjas")
        >>> imgs.show(ninjas)


        """

        try:
            from BeautifulSoup import BeautifulSoup

        except:
            print "You need to install Beatutiul Soup to use this function"
            print "to install you can use:"
            print "easy_install beautifulsoup"
            return

        invalid_size_msg = "I don't understand what size images you want. " \
                           "Valid options: 'thumb', 'small', 'medium', " \
                           "'large' or a tuple of exact dimensions " \
                           "i.e. (640,480)."

        if isinstance(size, basestring):
            size = size.lower()
            if size == 'thumb':
                size_param = ''
            elif size == 'small':
                size_param = '&tbs=isz:s'
            elif size == 'medium':
                size_param = '&tbs=isz:m'
            elif size == 'large':
                size_param = '&tbs=isz:l'
            else:
                print invalid_size_msg
                return None

        elif type(size) == tuple:
            width, height = size
            size_param = '&tbs=isz:ex,iszw:' + \
                         str(width) + ',iszh:' + str(height)

        else:
            print invalid_size_msg
            return None

        # Used to extract imgurl parameter value from a URL
        imgurl_re = re.compile('(?<=(&|\?)imgurl=)[^&]*((?=&)|$)')

        add_set = ImageSet()
        candidate_count = 0

        while len(add_set) < number:
            opener = urllib2.build_opener()
            opener.addheaders = [('User-agent', 'Mozilla/5.0')]
            url = ("http://www.google.com/search?tbm=isch&q="
                   + urllib2.quote(tag)
                   + size_param + "&start=" + str(candidate_count))
            page = opener.open(url)
            soup = BeautifulSoup(page)

            img_urls = []

            # Gets URLs of the thumbnail images
            if size == 'thumb':
                imgs = soup.findAll('img')
                for img in imgs:
                    dl_url = str(dict(img.attrs)['src'])
                    img_urls.append(dl_url)

            # Gets the direct image URLs
            else:
                for link_tag in soup.findAll(
                        'a', {'href': re.compile('imgurl=')}):
                    # URL to an image as given by Google Images
                    dirty_url = link_tag.get('href')
                    # The direct URL to the image
                    dl_url = str(re.search(imgurl_re, dirty_url).group())
                    img_urls.append(dl_url)

            for dl_url in img_urls:
                try:
                    add_img = Image(dl_url, verbose=False)

                    # Don't know a better way to check if the image was
                    # actually returned
                    if add_img.height != 0 and add_img.width != 0:
                        add_set.append(add_img)

                except:
                    #do nothing
                    pass

                if len(add_set) >= number:
                    break

        self.extend(add_set)

    def upload(self, dest, api_key=None, api_secret=None, verbose=True):
        """

        **SUMMARY**

        Uploads all the images to imgur or flickr or dropbox. In verbose mode
        URL values are printed.


        **PARAMETERS**

        * *api_key* - a string of the API key.
        * *api_secret* - (required only for flickr and dropbox ) a string of
         the API secret.
        * *verbose* - If verbose is true all values are printed to the screen


        **RETURNS**

        if uploading is successful

        - Imgur return the original image URL on success and None if it fails.
        - Flick returns True on success, else returns False.
        - dropbox returns True on success.


        **EXAMPLE**

        TO upload image to imgur::

          >>> imgset = ImageSet("/home/user/Desktop")
          >>> result = imgset.upload( 'imgur',"MY_API_KEY1234567890" )
          >>> print "Uploaded To: " + result[0]


        To upload image to flickr::

          >>> imgset.upload('flickr','api_key','api_secret')
          >>> # Once the api keys and secret keys are cached.
          >>> imgset.upload('flickr')

        To upload image to dropbox::

          >>> imgset.upload('dropbox','api_key','api_secret')
          >>> # Once the api keys and secret keys are cached.
          >>> imgset.upload('dropbox')

        **NOTES**

        .. Warning::
          This method requires two packages to be installed
          -PyCurl
          -flickr api.
          -dropbox


        .. Warning::
          You must supply your own API key.


        Find more about API keys:

        - http://imgur.com/register/api_anon
        - http://www.flickr.com/services/api/misc.api_keys.html
        - https://www.dropbox.com/developers/start/setup#python


        """
        try:
            for i in self:
                i.upload(dest, api_key, api_secret, verbose)
            return True

        except:
            return False

    def show(self, showtime=0.25):
        """
        **SUMMARY**

        This is a quick way to show all the items in a ImageSet.
        The time is in seconds. You can also provide a decimal value, so
        showtime can be 1.5, 0.02, etc.
        to show each image.

        **PARAMETERS**

        * *showtime* - the time, in seconds, to show each image in the set.

        **RETURNS**

        Nothing.

        **EXAMPLE**

        >>> imgs = ImageSet()
        >>> imgs.download("ninjas")
        >>> imgs.show()


       """

        for i in self:
            i.show()
            time.sleep(showtime)

    def _get_app_ext(self, loops=0):
        """ Application extension. Part that specifies amount of loops.
        if loops is 0, if goes on infinitely.
        """
        bb = "\x21\xFF\x0B"  # application extension
        bb += "NETSCAPE2.0"
        bb += "\x03\x01"
        if loops == 0:
            loops = 2 ** 16 - 1
        bb += int_to_bin(loops)
        bb += '\x00'  # end
        return bb

    def _get_graphics_control_ext(self, duration=0.1):
        """ Graphics Control Extension. A sort of header at the start of
        each image. Specifies transparency and duration. """
        bb = '\x21\xF9\x04'
        bb += '\x08'  # no transparency
        bb += int_to_bin(int(duration * 100))  # in 100th of seconds
        bb += '\x00'  # no transparent color
        bb += '\x00'  # end
        return bb

    def _write_gif(self, filename, duration=0.1, loops=0, dither=1):
        """ Given a set of images writes the bytes to the specified stream.
        """
        frames = 0
        previous = None
        fp = open(filename, 'wb')

        if not PIL_ENABLED:
            logger.warning("Need PIL to write animated gif files.")
            return

        converted = []

        for img in self:
            if not isinstance(img, PilImage.Image):
                pil_img = img.get_pil()
            else:
                pil_img = img

            converted.append((pil_img.convert('P', dither=dither),
                              img._get_header_anim()))

        try:
            for img, header_anim in converted:
                if not previous:
                    # gather data
                    palette = getheader(img)[1]
                    data = getdata(img)
                    imdes, data = data[0], data[1:]
                    header = header_anim
                    appext = self._get_app_ext(loops)
                    graphext = self._get_graphics_control_ext(duration)

                    # write global header
                    fp.write(header)
                    fp.write(palette)
                    fp.write(appext)

                    # write image
                    fp.write(graphext)
                    fp.write(imdes)
                    for d in data:
                        fp.write(d)

                else:
                    # gather info (compress difference)
                    data = getdata(img)
                    imdes, data = data[0], data[1:]
                    graphext = self._get_graphics_control_ext(duration)

                    # write image
                    fp.write(graphext)
                    fp.write(imdes)
                    for d in data:
                        fp.write(d)

                previous = img.copy()
                frames = frames + 1

            fp.write(";")  # end gif

        finally:
            fp.close()
            return frames

    def save(self, destination=None, dt=0.2, verbose=False, displaytype=None):
        """

        **SUMMARY**

        This is a quick way to save all the images in a data set.
        Or to Display in webInterface.

        If you didn't specify a path one will randomly be generated.
        To see the location the files are being saved to then pass
        verbose = True.

        **PARAMETERS**

        * *destination* - path to which images should be saved, or name of gif
        * file. If this ends in .gif, the pictures will be saved accordingly.
        * *dt* - time between frames, for creating gif files.
        * *verbose* - print the path of the saved files to the console.
        * *displaytype* - the method use for saving or displaying images.


        valid values are:

        * 'notebook' - display to the ipython notebook.
        * None - save to a temporary file.

        **RETURNS**

        Nothing.

        **EXAMPLE**

        >>> imgs = ImageSet()
        >>> imgs.download("ninjas")
        >>> imgs.save(destination="ninjas_folder", verbose=True)

        >>> imgs.save(destination="ninjas.gif", verbose=True)

        """
        if displaytype == 'notebook':
            try:
                from IPython.core.display import Image as IPImage
            except ImportError:
                print "You need IPython Notebooks to use this display mode"
                return
            from IPython.core import display as idisplay

            for i in self:
                tf = tempfile.NamedTemporaryFile(suffix=".png")
                loc = tf.name
                tf.close()
                i.save(loc)
                idisplay.display(IPImage(filename=loc))
                return
        else:
            if destination:
                if destination.endswith(".gif"):
                    return self._write_gif(destination, dt)
                else:
                    for i in self:
                        i.save(path=destination, temp=True, verbose=verbose)
            else:
                for i in self:
                    i.save(verbose=verbose)

    def show_paths(self):
        """
        **SUMMARY**

        This shows the file paths of all the images in the set.

        If they haven't been saved to disk then they will not have a filepath


        **RETURNS**

        Nothing.

        **EXAMPLE**

        >>> imgs = ImageSet()
        >>> imgs.download("ninjas")
        >>> imgs.save(verbose=True)
        >>> imgs.show_paths()


        **TO DO**

        This should return paths as a list too.

        """

        for i in self:
            print i.filename

    def _read_gif(self, filename):
        """ read_gif(filename)

        Reads images from an animated GIF file. Returns the number of images
        loaded.
        """

        if not PIL_ENABLED:
            return
        elif not os.path.isfile(filename):
            return

        pil_img = PilImage.open(filename)
        pil_img.seek(0)

        pil_images = []
        try:
            while True:
                pil_images.append(pil_img.copy())
                pil_img.seek(pil_img.tell() + 1)

        except EOFError:
            pass

        loaded = 0
        for img in pil_images:
            self.append(Image(img))
            loaded += 1

        return loaded

    def load(self, directory=None, extension=None, sort_by=None):
        """
        **SUMMARY**

        This function loads up files automatically from the directory you pass
        it.  If you give it an extension it will only load that extension
        otherwise it will try to load all know file types in that directory.

        extension should be in the format:
        extension = 'png'

        **PARAMETERS**

        * *directory* - The path or directory from which to load images.
        * *extension* - The extension to use. If none is given png is the
         default.
        * *sort_by* - Sort the directory based on one of the following
         parameters passed as strings.
          * *time* - the modification time of the file.
          * *name* - the name of the file.
          * *size* - the size of the file.

          The default behavior is to leave the directory unsorted.

        **RETURNS**

        The number of images in the image set.

        **EXAMPLE**

        >>> imgs = ImageSet()
        >>> imgs.load("images/faces")
        >>> imgs.load("images/eyes", "png")

        """
        if not directory:
            logger.warning("You need to give a directory to load files from.")
            return

        if not os.path.exists(directory):
            logger.warning("Invalid image path given.")
            return

        if extension:
            #regexes to ignore case
            regex_list = ['[' + letter + letter.upper() + ']'
                          for letter in extension]
            regex = ''.join(regex_list)
            regex = "*." + regex
            formats = [os.path.join(directory, regex)]

        else:
            formats = [os.path.join(directory, x) for x in IMAGE_FORMATS]

        file_set = [glob.glob(p) for p in formats]
        full_set = []
        for f in file_set:
            for i in f:
                full_set.append(i)

        file_set = full_set
        if sort_by is not None:
            if sort_by.lower() == "time":
                file_set = sorted(file_set, key=os.path.getmtime)
            if sort_by.lower() == "name":
                file_set = sorted(file_set)
            if sort_by.lower() == "size":
                file_set = sorted(file_set, key=os.path.getsize)

        self.filelist = dict()

        for i in file_set:
            tmp = None
            try:
                tmp = Image(i)
                if tmp is not None and tmp.width > 0 and tmp.height > 0:
                    if sys.platform.lower() == 'win32' \
                            or sys.platform.lower() == 'win64':
                        self.filelist[tmp.filename.split('\\')[-1]] = tmp
                    else:
                        self.filelist[tmp.filename.split('/')[-1]] = tmp
                    self.append(tmp)
            except:
                continue
        return len(self)

    def standardize(self, width, height):
        """
        **SUMMARY**

        Resize every image in the set to a standard size.

        **PARAMETERS**

        * *width* - the width that we want for every image in the set.
        * *height* - the height that we want for every image in the set.

        **RETURNS**

        A new image set where every image in the set is scaled to the desired
        size.

        **EXAMPLE**

        >>>> iset = ImageSet("./b/")
        >>>> thumbnails = iset.standardize(64,64)
        >>>> for t in thumbnails:
        >>>>   t.show()

        """
        ret_val = ImageSet()
        for i in self:
            ret_val.append(i.resize(width, height))
        return ret_val

    def dimensions(self):
        """
        **SUMMARY**

        Return an np.array that are the width and height of every image in the
        image set.

        **PARAMETERS**

        --NONE--

        **RETURNS**
        A 2xN numpy array where N is the number of images in the set. The first
        column is the width, and the second collumn is the height.

        **EXAMPLE**
        >>> iset = ImageSet("./b/")
        >>> sz = iset.dimensions()
        >>> np.max(sz[:,0]) # returns the largest width in the set.

        """
        ret_val = []
        for i in self:
            ret_val.append((i.width, i.height))
        return np.array(ret_val)

    def average(self, mode="first", size=(None, None)):
        """
        **SUMMARY**

        Casts each in the image set into a 32F image, averages them together
        and returns the results.
        If the images are different sizes the method attempts to standarize
        them.

        **PARAMETERS**

        * *mode* -
          * "first" - resize everything to the size of the first image.
          * "max" - resize everything to be the max width and max height of
           the set.
          * "min" - resize everything to be the min width and min height of
           the set.
          * "average" - resize everything to be the average width and height of
           the set
          * "fixed" - fixed, use the size tuple provided.

        * *size* - if the mode is set to fixed use this tuple as the size of
         the resulting image.

        **RETURNS**

        Returns a single image that is the average of all the values.

        **EXAMPLE**

        >>> imgs = ImageSet()
        >>> imgs.load("images/faces")
        >>> result = imgs.average(mode="first")
        >>> result.show()

        **TODO**
        * Allow the user to pass in an offset parameters that blit the images
        into the resutl.
        """
        fw = 0
        fh = 0
        # figger out how we will handle everything
        if len(self) <= 0:
            return ImageSet()

        vals = self.dimensions()
        if mode.lower() == "first":
            fw = self[0].width
            fh = self[0].height
        elif mode.lower() == "fixed":
            fw = size[0]
            fh = size[1]
        elif mode.lower() == "max":
            fw = np.max(vals[:, 0])
            fh = np.max(vals[:, 1])
        elif mode.lower() == "min":
            fw = np.min(vals[:, 0])
            fh = np.min(vals[:, 1])
        elif mode.lower() == "average":
            fw = int(np.average(vals[:, 0]))
            fh = int(np.average(vals[:, 1]))
        #determine if we really need to resize the images
        t1 = np.sum(vals[:, 0] - fw)
        t2 = np.sum(vals[:, 1] - fh)
        if t1 != 0 or t2 != 0:
            resized = self.standardize(fw, fh)
        else:
            resized = self
        # Now do the average calculation
        accumulator = np.zeros((fw, fh, 3), dtype=np.uint8)
        alpha = float(1.0 / len(resized))
        beta = float((len(resized) - 1.0) / len(resized))
        for i in resized:
            cv2.addWeighted(i.get_ndarray(), alpha,
                            accumulator, beta, 0, accumulator)
        ret_val = Image(accumulator)
        return ret_val

    def __getitem__(self, key):
        """
        **SUMMARY**

        Returns a ImageSet when sliced. Previously used to
        return list. Now it is possible to ImageSet member
        functions on sub-lists

        """
        if isinstance(key, types.SliceType):  # Or can use 'try:' for speed
            return ImageSet(list.__getitem__(self, key))
        else:
            return list.__getitem__(self, key)

    def __getslice__(self, i, j):
        """
        Deprecated since python 2.0, now using __getitem__
        """
        return self.__getitem__(slice(i, j))


class Image(object):
    """
    **SUMMARY**

    The Image class is the heart of SimpleCV and allows you to convert to and
    from a number of source types with ease.  It also has intelligent buffer
    management, so that modified copies of the Image required for algorithms
    such as edge detection, etc can be cached and reused when appropriate.


    Image are converted into 8-bit, 3-channel images in RGB colorspace.
    It will automatically handle conversion from other representations into
    this standard format.  If dimensions are passed, an empty image is created.

    **EXAMPLE**

    >>> i = Image("/path/to/image.png")
    >>> c = Camera().get_image()


    You can also just load the SimpleCV logo using:

    >>> img = Image("simplecv")
    >>> img2 = Image("logo")
    >>> img3 = Image("logo_inverted")
    >>> img4 = Image("logo_transparent")

    Or you can load an image from a URL:

    >>> img = Image("http://www.simplecv.org/image.png")

    """

    width = 0  # width and height in px
    height = 0
    depth = 0
    filename = ""  # source filename
    filehandle = ""  # filehandle if used
    camera = ""
    _mLayers = []

    _mDoHuePalette = False
    _mPaletteBins = None
    _mPalette = None
    _mPaletteMembers = None
    _mPalettePercentages = None

    _barcodeReader = ""  # property for the ZXing barcode reader

    # these are buffer frames for various operations on the image
    _bitmap = ""  # the bitmap (iplimage)  representation of the image
    _matrix = ""  # the matrix (cvmat) representation
    _grayMatrix = ""  # the gray scale (cvmat) representation -KAS
    _graybitmap = ""  # a reusable 8-bit grayscale bitmap
    _equalizedgraybitmap = ""  # the above bitmap, normalized
    _blobLabel = ""  # the label image for blobbing
    _edgeMap = ""  # holding reference for edge map
    _cannyparam = (0, 0)  # parameters that created _edgeMap
    _pil = ""  # holds a PIL object in buffer
    _numpy = ""  # numpy form buffer
    _grayNumpy = ""  # grayscale numpy for keypoint stuff
    _colorSpace = ColorSpace.UNKNOWN  # Colorspace Object
    _pgsurface = ""
    _cv2Numpy = None  # numpy array for OpenCV >= 2.3
    _cv2GrayNumpy = None  # grayscale numpy array for OpenCV >= 2.3
    # to store grid details | Format -> [gridIndex, gridDimensions]
    _gridLayer = [None, [0, 0]]

    #For DFT Caching
    _DFT = []  # an array of 2 channel (real,imaginary) 64F images

    #Keypoint caching values
    _mKeyPoints = None
    _mKPDescriptors = None
    _mKPFlavor = "NONE"

    #temp files
    _tempFiles = []

    #when we empty the buffers, populate with this:
    _initialized_buffers = {
        "_bitmap": "",
        "_matrix": "",
        "_grayMatrix": "",
        "_graybitmap": "",
        "_equalizedgraybitmap": "",
        "_blobLabel": "",
        "_edgeMap": "",
        "_cannyparam": (0, 0),
        "_pil": "",
        "_numpy": "",
        "_grayNumpy": "",
        "_pgsurface": "",
        "_cv2GrayNumpy": "",
        "_cv2Numpy": ""}

    # The variables _uncroppedX and _uncroppedY are used to buffer the points
    # when we crop the image.
    _uncroppedX = 0
    _uncroppedY = 0

    def __repr__(self):
        if len(self.filename) == 0:
            fn = "None"
        else:
            fn = self.filename
        return "<SimpleCV.Image Object size:(%d, %d), filename: (%s), " \
               "at memory location: (%s)>" \
               % (self.width, self.height, fn, hex(id(self)))

    #initialize the frame
    #parameters: source designation (filename)
    #todo: handle camera/capture from file cases (detect on file extension)
    def __init__(self, source=None, camera=None,
                 color_space=ColorSpace.UNKNOWN, verbose=True, sample=False,
                 webp=False):
        """
        **SUMMARY**

        The constructor takes a single polymorphic parameter, which it tests
        to see how it should convert into an RGB image.  Supported types
        include:

        **PARAMETERS**

        * *source* - The source of the image. This can be just about anything,
          a numpy arrray, a file name, a width and height tuple, a url. Certain
          strings such as "lenna" or "logo" are loaded automatically for quick
          testing.

        * *camera* - A camera to pull a live image.

        * *colorspace* - A default camera color space. If none is specified
         this will usually default to the BGR colorspace.

        * *sample* - This is set to true if you want to load some of the
         included sample images without having to specify the complete path


        **EXAMPLES**

        >>> img = Image('simplecv')
        >>> img2 = Image('test.png')
        >>> img3 = Image('http://www.website.com/my_image.jpg')
        >>> img.show()

        **NOTES**

        OpenCV: iplImage and cvMat types
        Python Image Library: Image type
        Filename: All opencv supported types (jpg, png, bmp, gif, etc)
        URL: The source can be a url, but must include the http://

        """
        self._ndarray = None  # contains image data as numpy.ndarray
        self._mLayers = []
        self.camera = camera
        self._colorSpace = color_space
        #Keypoint Descriptors
        self._mKeyPoints = []
        self._mKPDescriptors = []
        self._mKPFlavor = "NONE"
        #Pallete Stuff
        self._mDoHuePalette = False
        self._mPaletteBins = None
        self._mPalette = None
        self._mPaletteMembers = None
        self._mPalettePercentages = None
        #Temp files
        self._tempFiles = []

        #Check if need to load from URL
        if isinstance(source, basestring) \
                and (source.lower().startswith("http://")
                     or source.lower().startswith("https://")):
            #try:
            # added spoofed user agent for images that
            # are blocking bots (like wikipedia)
            user_agent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_7_4) " \
                         "AppleWebKit/536.5 (KHTML, like Gecko) " \
                         "Chrome/19.0.1084.54 Safari/536.5"
            req = urllib2.Request(source, headers={'User-Agent': user_agent})
            img_file = urllib2.urlopen(req)
            #except:
            #if verbose:
            #print "Couldn't open Image from URL:" + source
            #return None

            im = StringIO(img_file.read())
            source = PilImage.open(im).convert("RGB")

        #Check if loaded from base64 URI
        if isinstance(source, basestring) \
                and (source.lower().startswith("data:image/png;base64,")):
            img = source[22:].decode("base64")
            im = StringIO(img)
            source = PilImage.open(im).convert("RGB")

        #This section loads custom built-in images
        if isinstance(source, basestring):
            tmpname = source.lower()

            if tmpname == "simplecv" or tmpname == "logo":
                imgpth = os.path.join(LAUNCH_PATH,
                                      'data/sampleimages', 'simplecv.png')
                source = imgpth
            elif tmpname == "simplecv_inverted" \
                    or tmpname == "inverted" or tmpname == "logo_inverted":
                imgpth = os.path.join(LAUNCH_PATH,
                                      'data/sampleimages',
                                      'simplecv_inverted.png')
                source = imgpth
            elif tmpname == "lenna":
                imgpth = os.path.join(LAUNCH_PATH,
                                      'data/sampleimages',
                                      'lenna.png')
                source = imgpth
            elif tmpname == "lyle":
                imgpth = os.path.join(LAUNCH_PATH,
                                      'data/sampleimages',
                                      'LyleJune1973.png')
                source = imgpth
            elif tmpname == "parity":
                choice = random.choice(['LyleJune1973.png', 'lenna.png'])
                imgpth = os.path.join(LAUNCH_PATH, 'data/sampleimages', choice)
                source = imgpth

            elif sample:
                imgpth = os.path.join(LAUNCH_PATH, 'data/sampleimages', source)
                source = imgpth

        if isinstance(source, (tuple, list)):
            w = int(source[0])
            h = int(source[1])
            if color_space == ColorSpace.GRAY:
                self._ndarray = np.zeros((h, w), np.uint8)
            else:
                self._ndarray = np.zeros((h, w, 3), np.uint8)

        elif isinstance(source, np.ndarray):
            if len(source.shape) == 3 and source.shape[2] == 3:
                # we have a three channel array
                self._ndarray = source
                self._colorSpace = color_space
            elif len(source.shape) == 2:
                # we have a single channel array
                self._ndarray = source
                self._colorSpace = ColorSpace.GRAY
            else:
                raise IOError('Cant create image from ndarray with '
                              'shape {}.'.format(source.shape))

        elif isinstance(source, str):
            if source == '':
                raise IOError("No filename provided to Image constructor")
            elif not os.path.exists(source):
                raise IOError("Filename provided does not exist")

            if webp or source.split('.')[-1] == 'webp':
                try:
                    from webm import decode as webm_decode
                except ImportError:
                    logger.warning(
                        'The webm module or latest PIL / PILLOW module '
                        'needs to be installed to load webp files: '
                        'https://github.com/sightmachine/python-webm')
                    return
                with open(source, "rb") as f:
                    webp_image_data = bytearray(f.read())
                result = webm_decode.DecodeRGB(webp_image_data)
                self._pil = PilImage.frombuffer(
                    "RGB", (result.width, result.height),
                    str(result.bitmap), "raw", "RGB", 0, 1
                )
                self.filename = source
                self._ndarray = np.asarray(self._pil, dtype=np.uint8)
                self._colorSpace = ColorSpace.RGB
            else:
                self.filename = source
                self._ndarray = cv2.imread(self.filename)
                if self._ndarray is None:
                    raise Exception('Failed to create an image array')
                self._colorSpace = ColorSpace.BGR

        elif webp and isinstance(source, cStringIO.InputType):
            source.seek(0)  # set the stringIO to the begining
            try:
                self._pil = PilImage.open(source)
            except:
                raise Exception('Failed to load webp image using PIL')
            self._ndarray = np.asarray(self._pil, dtype=np.uint8)
            self._colorSpace = ColorSpace.RGB

        elif isinstance(source, pg.Surface):
            self._pgsurface = source
            self._ndarray = cv2.transpose(
                pg.surfarray.array3d(self._pgsurface).copy())
            self._colorSpace = ColorSpace.RGB

        elif PIL_ENABLED and isinstance(source, PilImage.Image):
            if source.mode != 'RGB':
                source = source.convert('RGB')
            self._pil = source
            self._ndarray = np.asarray(self._pil, dtype=np.uint8)
            self._colorSpace = ColorSpace.RGB

        else:
            raise Exception('Unsupported source type')

        self.height = self._ndarray.shape[0]
        self.width = self._ndarray.shape[1]
        self.dtype = self._ndarray.dtype

    # FIXME: __del__ prevents garbage collection of Image objects
    def __del__(self):
        """
        This is called when the instance is about to be destroyed also called
         a destructor.
        """
        try:
            for i in self._tempFiles:
                if i[1]:
                    os.remove(i[0])
        except:
            pass

    def get_exif_data(self):
        """
        **SUMMARY**

        This function extracts the exif data from an image file like JPEG or
        TIFF. The data is returned as a dict.

        **RETURNS**

        A dictionary of key value pairs. The value pairs are defined in the
        exif.py file.

        **EXAMPLE**

        >>> img = Image("./SimpleCV/data/sampleimages/OWS.jpg")
        >>> data = img.get_exif_data()
        >>> data['Image GPSInfo'].values

        **NOTES**

        * Compliments of: http://exif-py.sourceforge.net/

        * See also: http://en.wikipedia.org/wiki/Exchangeable_image_file_format

        **See Also**

        :py:class:`EXIF`
        """
        import os
        import string

        if len(self.filename) < 5 or self.filename is None:
            # I am not going to warn, better of img sets
            # logger.warning("ImageClass.get_exif_data: This image did not come
            # from a file, can't get EXIF data.")
            return {}

        file_name, file_extension = os.path.splitext(self.filename)
        file_extension = string.lower(file_extension)
        if file_extension != '.jpeg' and file_extension != '.jpg' \
                and file_extension != 'tiff' and file_extension != '.tif':
            return {}

        raw = open(self.filename, 'rb')
        data = process_file(raw)
        return data

    def live(self):
        """
        **SUMMARY**

        This shows a live view of the camera.
        * Left click will show mouse coordinates and color.
        * Right click will kill the live image.

        **RETURNS**

        Nothing. In place method.

        **EXAMPLE**

        >>> cam = Camera()
        >>> cam.live()

        """

        start_time = time.time()

        from simplecv.display import Display

        i = self
        d = Display(i.size())
        i.save(d)
        col = Color.RED

        while d.is_not_done():
            i = self
            i.clear_layers()
            elapsed_time = time.time() - start_time

            if d.mouse_left:
                txt = "coord: (" + str(d.mouse_x) + "," + str(d.mouse_y) + ")"
                i.dl().text(txt, (10, i.height / 2), color=col)
                txt = "color: " + str(i.get_pixel(d.mouse_x, d.mouse_y))
                i.dl().text(txt, (10, (i.height / 2) + 10), color=col)
                print "coord: (" + str(d.mouse_x) + "," + str(d.mouse_y) \
                    + "), color: " + str(i.get_pixel(d.mouse_x, d.mouse_y))

            if elapsed_time > 0 and elapsed_time < 5:
                i.dl().text("In live mode", (10, 10), color=col)
                i.dl().text("Left click will show mouse coordinates and color",
                            (10, 20), color=col)
                i.dl().text("Right click will kill the live image", (10, 30),
                            color=col)

            i.save(d)
            if d.mouse_right:
                print "Closing Window"
                d.done = True

        pg.quit()

    def get_color_space(self):
        """
        **SUMMARY**

        Returns the value matched in the color space class

        **RETURNS**

        Integer corresponding to the color space.

        **EXAMPLE**

        >>> if image.get_color_space() == ColorSpace.RGB:

        **SEE ALSO**

        :py:class:`ColorSpace`

        """
        return self._colorSpace

    def is_rgb(self):
        """
        **SUMMARY**

        Returns true if this image uses the RGB colorspace.

        **RETURNS**

        True if the image uses the RGB colorspace, False otherwise.

        **EXAMPLE**

        >>> if img.is_rgb():
        ...     r, g, b = img.split_channels()

        **SEE ALSO**

        :py:meth:`to_rgb`


        """
        return self._colorSpace == ColorSpace.RGB

    def is_bgr(self):
        """
        **SUMMARY**

        Returns true if this image uses the BGR colorspace.

        **RETURNS**

        True if the image uses the BGR colorspace, False otherwise.

        **EXAMPLE**

        >>> if img.is_bgr():
        ...     b, g, r = img.split_channels()

        **SEE ALSO**

        :py:meth:`to_bgr`

        """
        return self._colorSpace == ColorSpace.BGR

    def is_hsv(self):
        """
        **SUMMARY**

        Returns true if this image uses the HSV colorspace.

        **RETURNS**

        True if the image uses the HSV colorspace, False otherwise.

        **EXAMPLE**

        >>> if img.is_hsv():
        ...     h, s ,v = img.split_channels()

        **SEE ALSO**

        :py:meth:`to_hsv`

        """
        return self._colorSpace == ColorSpace.HSV

    def is_hls(self):
        """
        **SUMMARY**

        Returns true if this image uses the HLS colorspace.

        **RETURNS**

        True if the image uses the HLS colorspace, False otherwise.

        **EXAMPLE**

        >>> if img.is_hls():
        ...     h, l, s = img.split_channels()

        **SEE ALSO**

        :py:meth:`to_hls`

        """
        return self._colorSpace == ColorSpace.HLS

    def is_xyz(self):
        """
        **SUMMARY**

        Returns true if this image uses the XYZ colorspace.

        **RETURNS**

        True if the image uses the XYZ colorspace, False otherwise.

        **EXAMPLE**

        >>> if img.is_xyz():
        ...     x, y, z = img.split_channels()

        **SEE ALSO**

        :py:meth:`to_xyz`

        """
        return self._colorSpace == ColorSpace.XYZ

    def is_gray(self):
        """
        **SUMMARY**

        Returns true if this image uses the Gray colorspace.

        **RETURNS**

        True if the image uses the Gray colorspace, False otherwise.

        **EXAMPLE**

        >>> if img.is_gray():
        ...     print "The image is in Grayscale."

        **SEE ALSO**

        :py:meth:`to_gray`

        """
        return self._colorSpace == ColorSpace.GRAY

    def is_ycrcb(self):
        """
        **SUMMARY**

        Returns true if this image uses the YCrCb colorspace.

        **RETURNS**

        True if the image uses the YCrCb colorspace, False otherwise.

        **EXAMPLE**

        >>> if img.is_ycrcb():
        ...     y, cr, cb = img.split_channels()

        **SEE ALSO**

        :py:meth:`to_ycrcb`

        """
        return self._colorSpace == ColorSpace.YCrCb

    @staticmethod
    def convert(ndarray, from_color_space, to_color_space):
        """ Converts a numpy array from one color space to another

        :param ndarray: array to convert
        :type name: numpy.ndarray.
        :param from_color_space: color space to convert from.
        :type state: int.
        :param from_color_space: color space to convert to.
        :type state: int.
        :returns: instance of numpy.ndarray.
        """
        if from_color_space == to_color_space \
                or (from_color_space == ColorSpace.UNKNOWN
                    and to_color_space == ColorSpace.BGR):
            return ndarray.copy()

        color_space_to_string = {
            ColorSpace.UNKNOWN: 'BGR',  # Unknown handled as default BGR
            ColorSpace.BGR: 'BGR',
            ColorSpace.GRAY: 'GRAY',
            ColorSpace.RGB: 'RGB',
            ColorSpace.HLS: 'HLS',
            ColorSpace.HSV: 'HSV',
            ColorSpace.XYZ: 'XYZ',
            ColorSpace.YCrCb: 'YCR_CB'
        }

        converter_str = 'COLOR_{}2{}'.format(
            color_space_to_string[from_color_space],
            color_space_to_string[to_color_space])

        try:
            converter = getattr(cv2, converter_str)
        except AttributeError:
            # convert to BGR first
            converter_bgr_str = 'COLOR_{}2BGR'.format(
                color_space_to_string[from_color_space])
            converter_str = 'COLOR_BGR2{}'.format(
                color_space_to_string[to_color_space])

            converter_bgr = getattr(cv2, converter_bgr_str)
            converter = getattr(cv2, converter_str)

            new_ndarray = cv2.cvtColor(ndarray, converter_bgr)
            return cv2.cvtColor(new_ndarray, converter)
        else:
            return cv2.cvtColor(ndarray, converter)

    def to_rgb(self):
        """
        **SUMMARY**

        This method attemps to convert the image to the RGB colorspace.
        If the color space is unknown we assume it is in the BGR format

        **RETURNS**

        Returns the converted image if the conversion was successful,
        otherwise None is returned.

        **EXAMPLE**

        >>> img = Image("lenna")
        >>> RGBImg = img.to_rgb()

        **SEE ALSO**

        :py:meth:`is_rgb`

        """
        rgb_array = Image.convert(self._ndarray, self._colorSpace,
                                  ColorSpace.RGB)
        return Image(rgb_array, color_space=ColorSpace.RGB)

    def to_bgr(self):
        """
        **SUMMARY**

        This method attemps to convert the image to the BGR colorspace.
        If the color space is unknown we assume it is in the BGR format.

        **RETURNS**

        Returns the converted image if the conversion was successful,
        otherwise None is returned.

        **EXAMPLE**

        >>> img = Image("lenna")
        >>> BGRImg = img.to_bgr()

        **SEE ALSO**

        :py:meth:`is_bgr`

        """
        bgr_array = Image.convert(self._ndarray, self._colorSpace,
                                  ColorSpace.BGR)
        return Image(bgr_array, color_space=ColorSpace.BGR)

    def to_hls(self):
        """
        **SUMMARY**

        This method attempts to convert the image to the HLS colorspace.
        If the color space is unknown we assume it is in the BGR format.

        **RETURNS**

        Returns the converted image if the conversion was successful,
        otherwise None is returned.

        **EXAMPLE**

        >>> img = Image("lenna")
        >>> HLSImg = img.to_hls()

        **SEE ALSO**

        :py:meth:`is_hls`

        """
        hls_array = Image.convert(self._ndarray, self._colorSpace,
                                  ColorSpace.HLS)
        return Image(hls_array, color_space=ColorSpace.HLS)

    def to_hsv(self):
        """
        **SUMMARY**

        This method attempts to convert the image to the HSV colorspace.
        If the color space is unknown we assume it is in the BGR format

        **RETURNS**

        Returns the converted image if the conversion was successful,
        otherwise None is returned.

        **EXAMPLE**

        >>> img = Image("lenna")
        >>> HSVImg = img.to_hsv()

        **SEE ALSO**

        :py:meth:`is_hsv`

        """
        hsv_array = Image.convert(self._ndarray, self._colorSpace,
                                  ColorSpace.HSV)
        return Image(hsv_array, color_space=ColorSpace.HSV)

    def to_xyz(self):
        """
        **SUMMARY**

        This method attemps to convert the image to the XYZ colorspace.
        If the color space is unknown we assume it is in the BGR format

        **RETURNS**

        Returns the converted image if the conversion was successful,
        otherwise None is returned.

        **EXAMPLE**

        >>> img = Image("lenna")
        >>> XYZImg = img.to_xyz()

        **SEE ALSO**

        :py:meth:`is_xyz`

        """
        xyz_array = Image.convert(self._ndarray, self._colorSpace,
                                  ColorSpace.XYZ)
        return Image(xyz_array, color_space=ColorSpace.XYZ)

    def to_gray(self):
        """
        **SUMMARY**

        This method attemps to convert the image to the grayscale colorspace.
        If the color space is unknown we assume it is in the BGR format.

        **RETURNS**

        A grayscale SimpleCV image if successful.
        otherwise None is returned.

        **EXAMPLE**

        >>> img = Image("lenna")
        >>> img.to_gray().binarize().show()

        **SEE ALSO**

        :py:meth:`is_gray`
        :py:meth:`binarize`

        """
        gray_array = Image.convert(self._ndarray, self._colorSpace,
                                   ColorSpace.GRAY)
        return Image(gray_array, color_space=ColorSpace.GRAY)

    def to_ycrcb(self):
        """
        **SUMMARY**

        This method attemps to convert the image to the YCrCb colorspace.
        If the color space is unknown we assume it is in the BGR format

        **RETURNS**

        Returns the converted image if the conversion was successful,
        otherwise None is returned.

        **EXAMPLE**

        >>> img = Image("lenna")
        >>> RGBImg = img.to_ycrcb()

        **SEE ALSO**

        :py:meth:`is_ycrcb`

        """

        ycrcb_array = Image.convert(self._ndarray, self._colorSpace,
                                    ColorSpace.YCrCb)
        return Image(ycrcb_array, color_space=ColorSpace.YCrCb)

    def get_empty(self, channels=3):
        """
        **SUMMARY**

        Create a new, empty OpenCV bitmap with the specified number of channels
        (default 3).
        This method basically creates an empty copy of the image. This is handy
        for interfacing with OpenCV functions directly.

        **PARAMETERS**

        * *channels* - The number of channels in the returned OpenCV image.

        **RETURNS**

        Returns an black OpenCV IplImage that matches the width, height, and
        color depth of the source image.

        **EXAMPLE**

        >>> img = Image("lenna")
        >>> rawImg  = img.get_empty()
        >>> cv2.SomeOpenCVFunc(img.get_bitmap(),rawImg)

        **SEE ALSO**

        :py:meth:`get_bitmap`
        :py:meth:`get_fp_matrix`
        :py:meth:`get_pil`
        :py:meth:`get_numpy`
        :py:meth:`get_gray_numpy`
        :py:meth:`get_grayscale_matrix`

        """
        shape = [self.height, self.width]
        if channels > 1:
            shape.append(channels)
        return np.zeros(shape, dtype=self.dtype)

    def get_bitmap(self):
        """
        **SUMMARY**

        Retrieve the bitmap (iplImage) of the Image.  This is useful if you
        want to use functions from OpenCV with SimpleCV's image class

        **RETURNS**

        Returns black OpenCV IplImage from this image.

        **EXAMPLE**

        >>> img = Image("lenna")
        >>> rawImg  = img.get_bitmap()
        >>> rawOut  = img.get_empty()
        >>> cv2.SomeOpenCVFunc(rawImg,rawOut)

        **SEE ALSO**

        :py:meth:`get_empty`
        :py:meth:`get_fp_matrix`
        :py:meth:`get_pil`
        :py:meth:`get_numpy`
        :py:meth:`get_gray_numpy`
        :py:meth:`get_grayscale_matrix`

        """
        raise Exception('Deprecated. use get_ndarray()')

    def get_matrix(self):
        """
        **SUMMARY**

        Get the matrix (cvMat) version of the image, required for some OpenCV
        algorithms.

        **RETURNS**

        Returns the OpenCV CvMat version of this image.

        **EXAMPLE**

        >>> img = Image("lenna")
        >>> rawImg  = img.get_matrix()
        >>> rawOut  = img.get_empty()
        >>> cv2.SomeOpenCVFunc(rawImg,rawOut)

        **SEE ALSO**

        :py:meth:`get_empty`
        :py:meth:`get_bitmap`
        :py:meth:`get_fp_matrix`
        :py:meth:`get_pil`
        :py:meth:`get_numpy`
        :py:meth:`get_gray_numpy`
        :py:meth:`get_grayscale_matrix`

        """
        raise Exception('Deprecated use get_ndarray()')

    def get_fp_ndarray(self):
        """
        **SUMMARY**

        Converts the standard int bitmap to a floating point bitmap.
        This is handy for some OpenCV functions.


        **RETURNS**

        Returns the floating point OpenCV CvMat version of this image.

        **EXAMPLE**

        >>> img = Image("lenna")
        >>> rawImg  = img.get_fp_matrix()
        >>> rawOut  = img.get_empty()
        >>> cv2.SomeOpenCVFunc(rawImg,rawOut)

        **SEE ALSO**

        :py:meth:`get_empty`
        :py:meth:`get_bitmap`
        :py:meth:`get_matrix`
        :py:meth:`get_pil`
        :py:meth:`get_numpy`
        :py:meth:`get_gray_numpy`
        :py:meth:`get_grayscale_matrix`

        """
        return self._ndarray.astype(np.float32)

    def get_pil(self):
        """
        **SUMMARY**

        Get a PIL Image object for use with the Python Image Library
        This is handy for some PIL functions.


        **RETURNS**

        Returns the Python Imaging Library (PIL) version of this image.

        **EXAMPLE**

        >>> img = Image("lenna")
        >>> rawImg  = img.get_pil()


        **SEE ALSO**

        :py:meth:`get_empty`
        :py:meth:`get_bitmap`
        :py:meth:`get_matrix`
        :py:meth:`get_fp_matrix`
        :py:meth:`get_numpy`
        :py:meth:`get_gray_numpy`
        :py:meth:`get_grayscale_matrix`

        """
        if not PIL_ENABLED:
            return None
        if not self._pil:
            rgb_array = self.to_rgb().get_ndarray()
            self._pil = PilImage.fromstring("RGB", self.size(),
                                            rgb_array.tostring())
        return self._pil

    def get_gray_ndarray(self):
        """
        **SUMMARY**

        Return a grayscale Numpy array of the image.

        **RETURNS**

        Returns the image, converted first to grayscale and then converted to
        a 2D numpy array.

        **EXAMPLE**

        >>> img = Image("lenna")
        >>> rawImg  = img.get_gray_numpy()

        **SEE ALSO**

        :py:meth:`get_empty`
        :py:meth:`get_bitmap`
        :py:meth:`get_matrix`
        :py:meth:`get_pil`
        :py:meth:`get_numpy`
        :py:meth:`get_gray_numpy`
        :py:meth:`get_grayscale_matrix`

        """
        return Image.convert(self._ndarray, self._colorSpace, ColorSpace.GRAY)

    def get_numpy(self):
        """
        **SUMMARY**

        Get a Numpy array of the image in width x height x RGB dimensions

        **RETURNS**

        Returns the image, converted first to grayscale and then converted to
        a 3D numpy array.

        **EXAMPLE**

        >>> img = Image("lenna")
        >>> rawImg  = img.get_numpy()

        **SEE ALSO**

        :py:meth:`get_empty`
        :py:meth:`get_bitmap`
        :py:meth:`get_matrix`
        :py:meth:`get_pil`
        :py:meth:`get_gray_numpy`
        :py:meth:`get_grayscale_matrix`

        """
        raise Exception('Deprecated. use get_ndarray()')

    def get_ndarray(self):
        """
        **SUMMARY**

        Get a Numpy array of the image in width x height x chanels dimensions
        compatible with OpenCV >= 2.3

        **RETURNS**

        Returns the 3D numpy array of the image compatible with OpenCV >= 2.3

        **EXAMPLE**

        >>> img = Image("lenna")
        >>> rawImg  = img.get_ndarray()

        **SEE ALSO**

        :py:meth:`get_empty`
        :py:meth:`get_bitmap`
        :py:meth:`get_matrix`
        :py:meth:`get_pil`
        :py:meth:`get_gray_numpy`
        :py:meth:`get_grayscale_matrix`
        :py:meth:`get_numpy`
        :py:meth:`get_gray_numpy_cv2`

        """
        return self._ndarray

    def _get_grayscale_bitmap(self):
        raise Exception('Deprecated use get_gray_ndarray()')

    def get_grayscale_matrix(self):
        """
        **SUMMARY**

        Get the grayscale matrix (cvMat) version of the image, required for
        some OpenCV algorithms.

        **RETURNS**

        Returns the OpenCV CvMat version of this image.

        **EXAMPLE**

        >>> img = Image("lenna")
        >>> rawImg  = img.get_grayscale_matrix()
        >>> rawOut  = img.get_empty()
        >>> cv2.SomeOpenCVFunc(rawImg,rawOut)

        **SEE ALSO**

        :py:meth:`get_empty`
        :py:meth:`get_bitmap`
        :py:meth:`get_fp_matrix`
        :py:meth:`get_pil`
        :py:meth:`get_numpy`
        :py:meth:`get_gray_numpy`
        :py:meth:`get_matrix`

        """
        raise Exception('Deprecated use get_gray_ndarray()')

    def _get_equalized_grayscale_bitmap(self):
        raise Exception('Deprecated use cv2.equalizeHist(gray_array)')

    def equalize(self):
        """
        **SUMMARY**

        Perform a histogram equalization on the image.

        **RETURNS**

        Returns a grayscale simplecv Image.

        **EXAMPLE**

        >>> img = Image("lenna")
        >>> img = img.equalize()

        """
        equalized_array = cv2.equalizeHist(self.get_gray_ndarray())
        return Image(equalized_array, color_space=ColorSpace.GRAY)

    def get_pg_surface(self):
        """
        **SUMMARY**

        Returns the image as a pygame surface.  This is used for rendering the
        display

        **RETURNS**

        A pygame surface object used for rendering.


        """
        if self._pgsurface:
            return self._pgsurface
        else:
            self._pgsurface = pg.image.fromstring(
                self.to_rgb().get_ndarray().tostring(), self.size(), "RGB")
            return self._pgsurface

    def to_string(self):
        """
        **SUMMARY**

        Returns the image as a string, useful for moving data around.


        **RETURNS**

        The image, converted to rgb, then converted to a string.

        """
        return self.to_rgb().get_ndarray().tostring()

    def save(self, filehandle_or_filename="", mode="", verbose=False,
             temp=False, path=None, filename=None, clean_temp=False, **params):
        """
        **SUMMARY**

        Save the image to the specified filename.  If no filename is provided
        then it will use the filename the Image was loaded from or the last
        place it was saved to. You can save to lots of places, not just files.
        For example you can save to the Display, a JpegStream, VideoStream,
        temporary file, or Ipython Notebook.


        Save will implicitly render the image's layers before saving, but the
        layers are
        not applied to the Image itself.


        **PARAMETERS**

        * *filehandle_or_filename* - the filename to which to store the file.
         The method will infer the file type.

        * *mode* - This flag is used for saving using pul.

        * *verbose* - If this flag is true we return the path where we saved
         the file.

        * *temp* - If temp is True we save the image as a temporary file and
         return the path

        * *path* - path where temporary files needed to be stored

        * *filename* - name(Prefix) of the temporary file.

        * *clean_temp* - This flag is made True if tempfiles are tobe deleted
         once the object is to be destroyed.

        * *params* - This object is used for overloading the PIL save methods.
          In particular this method is useful for setting the jpeg compression
          level. For JPG see this documentation:
          http://www.pythonware.com/library/pil/handbook/format-jpeg.htm

        **EXAMPLES**

        To save as a temporary file just use:

        >>> img = Image('simplecv')
        >>> img.save(temp=True)

        It will return the path that it saved to.

        Save also supports IPython Notebooks when passing it a Display object
        that has been instainted with the notebook flag.

        To do this just use::

          >>> disp = Display(displaytype='notebook')
          >>> img.save(disp)

        .. Note::
          You must have IPython notebooks installed for this to work path and
          filename are valid if and only if temp is set to True.

        .. attention::
          We need examples for all save methods as they are unintuitve.

        """
        #TODO, we use the term mode here when we mean format
        #TODO, if any params are passed, use PIL

        if temp:
            import glob

            if filename is None:
                filename = 'Image'
            if path is None:
                path = tempfile.gettempdir()
            if glob.os.path.exists(path):
                path = glob.os.path.abspath(path)
                imagefiles = glob.glob(
                    glob.os.path.join(path, filename + "*.png"))
                num = [0]
                for img in imagefiles:
                    num.append(int(glob.re.findall('[0-9]+$', img[:-4])[-1]))
                num.sort()
                fnum = num[-1] + 1
                filename = glob.os.path.join(
                    path, filename + ("%07d" % fnum) + ".png")
                self._tempFiles.append((filename, clean_temp))
                self.save(self._tempFiles[-1][0])
                return self._tempFiles[-1][0]
            else:
                print "Path does not exist!"

        else:
            if filename:
                filehandle_or_filename = filename + ".png"

        if not filehandle_or_filename:
            if self.filename:
                filehandle_or_filename = self.filename
            else:
                filehandle_or_filename = self.filehandle

        if len(self._mLayers):
            saveimg = self.apply_layers()
        else:
            saveimg = self

        if self._colorSpace != ColorSpace.BGR \
                and self._colorSpace != ColorSpace.GRAY:
            saveimg = saveimg.to_bgr()

        if not isinstance(filehandle_or_filename, basestring):

            fh = filehandle_or_filename

            if not PIL_ENABLED:
                logger.warning("You need the python image library to save by "
                               "filehandle")
                return 0

            if isinstance(fh, JpegStreamer):
                fh.jpgdata = StringIO()
                saveimg.get_pil().save(
                    fh.jpgdata, "jpeg",
                    **params)  # save via PIL to a StringIO handle
                fh.refreshtime = time.time()
                self.filename = ""
                self.filehandle = fh

            elif isinstance(fh, VideoStream):
                self.filename = ""
                self.filehandle = fh
                fh.write_frame(saveimg)

            elif isinstance(fh, Display):

                if fh.displaytype == 'notebook':
                    try:
                        from IPython.core.display import Image as IPImage
                    except ImportError:
                        print "You need IPython Notebooks to use this " \
                              "display mode"
                        return

                    from IPython.core import display as idisplay

                    tf = tempfile.NamedTemporaryFile(suffix=".png")
                    loc = tf.name
                    tf.close()
                    self.save(loc)
                    idisplay.display(IPImage(filename=loc))
                    return
                else:
                    #self.filename = ""
                    self.filehandle = fh
                    fh.write_frame(saveimg)

            else:
                if not mode:
                    mode = "jpeg"

                try:
                     # The latest version of PIL / PILLOW supports webp,
                     # try this first, if not gracefully fallback
                    saveimg.get_pil().save(fh, mode, **params)
                     # set the filename for future save operations
                    self.filehandle = fh
                    self.filename = ""
                    return 1
                except Exception, e:
                    if mode.lower() != 'webp':
                        raise e

            if verbose:
                print self.filename

            if not mode.lower() == 'webp':
                return 1

        #make a temporary file location if there isn't one
        if not filehandle_or_filename:
            filename = tempfile.mkstemp(suffix=".png")[-1]
        else:
            filename = filehandle_or_filename

        #allow saving in webp format
        if mode == 'webp' or re.search('\.webp$', filename):
            try:
                #newer versions of PIL support webp format, try that first
                self.get_pil().save(filename, **params)
            except:
                # if PIL doesn't support it, maybe we have
                # the python-webm library
                try:
                    from webm import encode as webm_encode
                    from webm.handlers import BitmapHandler, WebPHandler
                except:
                    logger.warning(
                        'You need the webm library to save to webp format. '
                        'You can download from: '
                        'https://github.com/sightmachine/python-webm')
                    return 0

                #PNG_BITMAP_DATA = bytearray(
                #   Image.open(PNG_IMAGE_FILE).tostring())
                png_bitmap_data = bytearray(self.to_string())
                image_width = self.width
                image_height = self.height

                image = BitmapHandler(
                    png_bitmap_data, BitmapHandler.RGB,
                    image_width, image_height, image_width * 3
                )
                result = webm_encode.EncodeRGB(image)

                if isinstance(filehandle_or_filename, cStringIO.InputType):
                    filehandle_or_filename.write(result.data)
                else:
                    file(filename.format("RGB"), "wb").write(result.data)
                return 1
        # if the user is passing kwargs use the PIL save method.
        # usually this is just the compression rate for the image
        if params:
            if not mode:
                mode = "jpeg"
            saveimg.get_pil().save(filename, mode, **params)
            return 1

        if filename:
            if self.is_gray():
                cv2.imwrite(filename, saveimg.get_ndarray())
            else:
                cv2.imwrite(filename, saveimg.to_bgr().get_ndarray())

            # set the filename for future save operations
            self.filename = filename
            self.filehandle = ""
        elif self.filename:
            if self.is_gray():
                cv2.imwrite(filename, saveimg.get_ndarray())
            else:
                cv2.imwrite(filename, saveimg.to_bgr().get_ndarray())
        else:
            return 0

        if verbose:
            print self.filename

        if temp:
            return filename
        else:
            return 1

    def copy(self):
        """
        **SUMMARY**

        Return a full copy of the Image's bitmap.  Note that this is different
        from using python's implicit copy function in that only the bitmap
        itself is copied. This method essentially performs a deep copy.

        **RETURNS**

        A copy of this SimpleCV image.

        **EXAMPLE**

        >>> img = Image("logo")
        >>> img2 = img.copy()

        """
        return Image(self._ndarray.copy(), color_space=self._colorSpace)

    def upload(self, dest, api_key=None, api_secret=None, verbose=True):
        """
        **SUMMARY**

        Uploads image to imgur or flickr or dropbox. In verbose mode URL values
        are printed.

        **PARAMETERS**

        * *api_key* - a string of the API key.
        * *api_secret* (required only for flickr and dropbox ) - a string of
         the API secret.
        * *verbose* - If verbose is true all values are printed to the screen


        **RETURNS**

        if uploading is successful

        - Imgur return the original image URL on success and None if it fails.
        - Flick returns True on success, else returns False.
        - dropbox returns True on success.


        **EXAMPLE**

        TO upload image to imgur::

          >>> img = Image("lenna")
          >>> result = img.upload( 'imgur',"MY_API_KEY1234567890" )
          >>> print "Uploaded To: " + result[0]


        To upload image to flickr::

          >>> img.upload('flickr','api_key','api_secret')
          >>> # Once the api keys and secret keys are cached.
          >>> img.invert().upload('flickr')


        To upload image to dropbox::

          >>> img.upload('dropbox','api_key','api_secret')
          >>> # Once the api keys and secret keys are cached.
          >>> img.invert().upload('dropbox')


        **NOTES**

        .. Warning::
          This method requires two packages to be installed

          - PyCurl
          - flickr api.
          - dropbox


        .. Warning::
          You must supply your own API key. See here:

          - http://imgur.com/register/api_anon
          - http://www.flickr.com/services/api/misc.api_keys.html
          - https://www.dropbox.com/developers/start/setup#python

        """
        if dest == 'imgur':
            try:
                import pycurl
            except ImportError:
                print "PycURL Library not installed."
                return

            response = StringIO()
            c = pycurl.Curl()
            values = [("key", api_key),
                      ("image", (c.FORM_FILE, self.filename))]
            c.setopt(c.URL, "http://api.imgur.com/2/upload.xml")
            c.setopt(c.HTTPPOST, values)
            c.setopt(c.WRITEFUNCTION, response.write)
            c.perform()
            c.close()

            match = re.search(r'<hash>(\w+).*?<deletehash>(\w+)'
                              r'.*?<original>(http://[\w.]+/[\w.]+)',
                              response.getvalue(), re.DOTALL)
            if match:
                if verbose:
                    print "Imgur page: http://imgur.com/" + match.group(1)
                    print "Original image: " + match.group(3)
                    print "Delete page: http://imgur.com/delete/" \
                          + match.group(2)
                return [match.group(1), match.group(3), match.group(2)]
            else:
                if verbose:
                    print "The API Key given is not valid"
                return None

        elif dest == 'flickr':
            global temp_token
            flickr = None
            try:
                import flickrapi
            except ImportError:
                print "Flickr API is not installed. Please install it from " \
                      "http://pypi.python.org/pypi/flickrapi"
                return False
            try:
                if not (api_key is None and api_secret is None):
                    self.flickr = flickrapi.FlickrAPI(api_key, api_secret,
                                                      cache=True)
                    self.flickr.cache = flickrapi.SimpleCache(timeout=3600,
                                                              max_entries=200)
                    self.flickr.authenticate_console('write')
                    temp_token = (api_key, api_secret)
                else:
                    try:
                        self.flickr = flickrapi.FlickrAPI(temp_token[0],
                                                          temp_token[1],
                                                          cache=True)
                        self.flickr.authenticate_console('write')
                    except NameError:
                        print "API key and Secret key are not set."
                        return
            except:
                print "The API Key and Secret Key are not valid"
                return False
            if self.filename:
                try:
                    self.flickr.upload(self.filename, self.filehandle)
                except:
                    print "Uploading Failed !"
                    return False
            else:
                tf = self.save(temp=True)
                self.flickr.upload(tf, "Image")
            return True

        elif dest == 'dropbox':
            global dropbox_token
            access_type = 'dropbox'
            try:
                from dropbox import client, rest, session
                import webbrowser
            except ImportError:
                print "Dropbox API is not installed. For more info refer : " \
                      "https://www.dropbox.com/developers/start/setup#python "
                return False
            try:
                if 'dropbox_token' not in globals() \
                        and api_key is not None and api_secret is not None:
                    sess = session.DropboxSession(api_key, api_secret,
                                                  access_type)
                    request_token = sess.obtain_request_token()
                    url = sess.build_authorize_url(request_token)
                    webbrowser.open(url)
                    print "Please visit this website and press the 'Allow' " \
                          "button, then hit 'Enter' here."
                    raw_input()
                    access_token = sess.obtain_access_token(request_token)
                    dropbox_token = client.DropboxClient(sess)
                else:
                    if dropbox_token:
                        pass
                    else:
                        return None
            except:
                print "The API Key and Secret Key are not valid"
                return False
            if self.filename:
                try:
                    f = open(self.filename)
                    dropbox_token.put_file(
                        '/SimpleCVImages/' + os.path.split(self.filename)[-1],
                        f)
                except:
                    print "Uploading Failed !"
                    return False
            else:
                tf = self.save(temp=True)
                f = open(tf)
                dropbox_token.put_file('/SimpleCVImages/' + 'Image', f)
                return True

    def scale(self, width, height=-1, interpolation=cv2.INTER_LINEAR):
        """
        **SUMMARY**

        Scale the image to a new width and height.

        If no height is provided, the width is considered a scaling value.

        **PARAMETERS**

        * *width* - either the new width in pixels, if the height parameter
          is > 0, or if this value is a floating point value, this is the
          scaling factor.

        * *height* - the new height in pixels.

        * *interpolation* - how to generate new pixels that don't match the
         original pixels. Argument goes direction to cv2.resize.
        See http://docs.opencv.org/modules/imgproc/doc/
        geometric_transformations.html?highlight=resize#cv2.resize
        for more details

        **RETURNS**

        The resized image.

        **EXAMPLE**

        >>> img.scale(200, 100)  # scales the image to 200px x 100px
        >>> img.scale(2.0)  # enlarges the image to 2x its current size


        .. Warning::
          The two value scale command is deprecated. To set width and height
          use the resize function.

        :py:meth:`resize`

        """
        w, h = width, height
        if height == -1:
            w = int(self.width * width)
            h = int(self.height * width)
            if w > MAX_DIMENSION or h > MAX_DIMENSION or h < 1 or w < 1:
                logger.warning("Holy Heck! You tried to make an image really "
                               "big or impossibly small. I can't scale that")
                return self

        scaled_array = cv2.resize(self.get_ndarray(), (w, h),
                                  interpolation=interpolation)
        return Image(scaled_array, color_space=self._colorSpace)

    def resize(self, w=None, h=None):
        """
        **SUMMARY**

        This method resizes an image based on a width, a height, or both.
        If either width or height is not provided the value is inferred by
        keeping the aspect ratio.
        If both values are provided then the image is resized accordingly.

        **PARAMETERS**

        * *width* - The width of the output image in pixels.

        * *height* - The height of the output image in pixels.

        **RETURNS**

        Returns a resized image, if the size is invalid a warning is issued and
        None is returned.

        **EXAMPLE**

        >>> img = Image("lenna")
        >>> img2 = img.resize(w=1024) # h is guessed from w
        >>> img3 = img.resize(h=1024) # w is guessed from h
        >>> img4 = img.resize(w=200, h=100)

        """
        ret_val = None
        if w is None and h is None:
            logger.warning("Image.resize has no parameters. "
                           "No operation is performed")
            return None
        elif w is not None and h is None:
            sfactor = float(w) / float(self.width)
            h = int(sfactor * float(self.height))
        elif w is None and h is not None:
            sfactor = float(h) / float(self.height)
            w = int(sfactor * float(self.width))
        if w > MAX_DIMENSION or h > MAX_DIMENSION:
            logger.warning("Image.resize Holy Heck! You tried to make an "
                           "image really big or impossibly small. "
                           "I can't scale that")
            return ret_val

        saceld_array = cv2.resize(self._ndarray, (w, h))
        return Image(saceld_array, color_space=self._colorSpace)

    def smooth(self, algorithm_name='gaussian', aperture=(3, 3), sigma=0,
               spatial_sigma=0, grayscale=False):
        """
        **SUMMARY**

        Smooth the image, by default with the Gaussian blur.  If desired,
        additional algorithms and apertures can be specified.  Optional
        parameters are passed directly to OpenCV's functions.

        If grayscale is true the smoothing operation is only performed on a
        single channel otherwise the operation is performed on each channel
        of the image.

        for OpenCV versions >= 2.3.0 it is advisible to take a look at
               - :py:meth:`bilateral_filter`
               - :py:meth:`median_filter`
               - :py:meth:`blur`
               - :py:meth:`gaussian_blur`

        **PARAMETERS**

        * *algorithm_name* - valid options are 'blur' or gaussian, 'bilateral',
         and 'median'.

          * `Median Filter <http://en.wikipedia.org/wiki/Median_filter>`_

          * `Gaussian Blur <http://en.wikipedia.org/wiki/Gaussian_blur>`_

          * `Bilateral Filter <http://en.wikipedia.org/wiki/Bilateral_filter>`_

        * *aperture* - A tuple for the aperture of the gaussian blur as an
                       (x,y) tuple.

        .. Warning::
          These must be odd numbers.

        * *sigma* -

        * *spatial_sigma* -

        * *grayscale* - Return just the grayscale image.



        **RETURNS**

        The smoothed image.

        **EXAMPLE**

        >>> img = Image("Lenna")
        >>> img2 = img.smooth()
        >>> img3 = img.smooth('median')

        **SEE ALSO**

        :py:meth:`bilateral_filter`
        :py:meth:`median_filter`
        :py:meth:`blur`

        """
        if is_tuple(aperture):
            win_x, win_y = aperture
            if win_x <= 0 or win_y <= 0 or win_x % 2 == 0 or win_y % 2 == 0:
                logger.warning("The aperture (x,y) must be odd number and "
                               "greater than 0.")
                return None
        else:
            raise ValueError("Please provide a tuple to aperture, "
                             "got: %s" % type(aperture))

        window = (win_x, win_y)
        if algorithm_name == "blur":
            return self.blur(window, grayscale)
        elif algorithm_name == "bilateral":
            return self.bilateral_filter(diameter=win_x, grayscale=grayscale)
        elif algorithm_name == "median":
            return self.median_filter(window, grayscale)
        else:
            return self.gaussian_blur(window, sigma, spatial_sigma, grayscale)

    def median_filter(self, window=None, grayscale=False):
        """
        **SUMMARY**

        Smooths the image, with the median filter. Performs a median filtering
        operation to denoise/despeckle the image.
        The optional parameter is the window size.
        see : http://en.wikipedia.org/wiki/Median_filter

        **Parameters**

        * *window* - should be in the form a tuple (win_x,win_y). Where win_x
         should be equal to win_y. By default it is set to 3x3,
         i.e window = (3x3).

        **Note**

        win_x and win_y should be greater than zero, a odd number and equal.

        For OpenCV versions >= 2.3.0
        cv2.medianBlur function is called.

        """
        if is_tuple(window):
            win_x, win_y = window
            if win_x >= 0 and win_y >= 0 and win_x % 2 == 1 and win_y % 2 == 1:
                if win_x != win_y:
                    win_x = win_y
            else:
                logger.warning("The aperture (win_x, win_y) must be odd "
                               "number and greater than 0.")
                return None

        elif is_number(window):
            win_x = window
        else:
            win_x = 3  # set the default aperture window size (3x3)

        if grayscale:
            img_medianblur = cv2.medianBlur(self.get_gray_ndarray(), win_x)
            return Image(img_medianblur, color_space=ColorSpace.GRAY)
        else:
            img_medianblur = cv2.medianBlur(self._ndarray, win_x)
            return Image(img_medianblur, color_space=self._colorSpace)

    def bilateral_filter(self, diameter=5, sigma_color=10, sigma_space=10,
                         grayscale=False):
        """
        **SUMMARY**

        Smooths the image, using bilateral filtering. Potential of bilateral
        filtering is for the removal of texture.
        The optional parameter are diameter, sigma_color, sigma_space.

        Bilateral Filter
        see : http://en.wikipedia.org/wiki/Bilateral_filter
        see : http://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/
        MANDUCHI1/Bilateral_Filtering.html

        **Parameters**

        * *diameter* - A tuple for the window of the form (diameter,diameter).
                       By default window = (3x3).
                       (for OpenCV versions <= 2.3.0)
                     - Diameter of each pixel neighborhood that is used during
                       filtering. ( for OpenCV versions >= 2.3.0)


        * *sigma_color* - Filter the specified value in the color space. A
         larger value of the parameter means that farther colors within the
         pixel neighborhood (see sigma_space ) will be mixed together,
         resulting in larger areas of semi-equal color.

        * *sigma_space* - Filter the specified value in the coordinate space.
         A larger value of the parameter means that farther pixels will
         influence each other as long as their colors are close enough

        **NOTE**
        For OpenCV versions <= 2.3.0
        -- this acts as Convience function derived from the :py:meth:`smooth`
           method.
        -- where aperture(window) is (diameter,diameter)
        -- sigma_color and sigmanSpace become obsolete

        For OpenCV versions higher than 2.3.0. i.e >= 2.3.0
        -- cv2.bilateralFilter function is called
        -- If the sigma_color and sigma_space values are small (< 10),
           the filter will not have much effect, whereas if they are large
           (> 150), they will have a very strong effect, making the image look
           'cartoonish'
        -- It is recommended to use diamter=5 for real time applications, and
           perhaps diameter=9 for offile applications that needs heavy noise
           filtering.
        """
        if is_tuple(diameter):
            win_x, win_y = diameter
            if win_x >= 0 and win_y >= 0 and win_x % 2 == 1 and win_y % 2 == 1:
                if win_x != win_y:
                    diameter = (win_x, win_y)
            else:
                logger.warning("The aperture (win_x,win_y) must be odd number "
                               "and greater than 0.")
                return None

        elif is_number(diameter):
            pass
        else:
            win_x = 3  # set the default aperture window size (3x3)
            diameter = (win_x, win_x)

        if grayscale:
            img_bilateral = cv2.bilateralFilter(self.get_gray_ndarray(),
                                                diameter, sigma_color,
                                                sigma_space)
            return Image(img_bilateral, color_space=ColorSpace.GRAY)
        else:
            img_bilateral = cv2.bilateralFilter(self._ndarray, diameter,
                                                sigma_color, sigma_space)
            return Image(img_bilateral, color_space=self._colorSpace)

    def blur(self, window=None, grayscale=False):
        """
        **SUMMARY**

        Smoothes an image using the normalized box filter.
        The optional parameter is window.

        see : http://en.wikipedia.org/wiki/Blur

        **Parameters**

        * *window* - should be in the form a tuple (win_x,win_y).
                   - By default it is set to 3x3, i.e window = (3x3).

        **NOTE**
        For OpenCV versions <= 2.3.0
        -- this acts as Convience function derived from the :py:meth:`smooth`
           method.

        For OpenCV versions higher than 2.3.0. i.e >= 2.3.0
        -- cv2.blur function is called
        """
        if is_tuple(window):
            win_x, win_y = window
            if win_x <= 0 or win_y <= 0:
                logger.warning("win_x and win_y should be greater than 0.")
                return None
        elif is_number(window):
            window = (window, window)
        else:
            window = (3, 3)

        if grayscale:
            img_blur = cv2.blur(self.get_gray_ndarray(), window)
            return Image(img_blur, color_space=ColorSpace.GRAY)
        else:
            img_blur = cv2.blur(self._ndarray, window)
            return Image(img_blur, color_space=self._colorSpace)

    def gaussian_blur(self, window=None, sigma_x=0, sigma_y=0,
                      grayscale=False):
        """
        **SUMMARY**

        Smoothes an image, typically used to reduce image noise and reduce
        detail.
        The optional parameter is window.

        see : http://en.wikipedia.org/wiki/Gaussian_blur

        **Parameters**

        * *window* - should be in the form a tuple (win_x,win_y). Where win_x
                     and win_y should be positive and odd.
                   - By default it is set to 3x3, i.e window = (3x3).

        * *sigma_x* - Gaussian kernel standard deviation in X direction.

        * *sigma_y* - Gaussian kernel standard deviation in Y direction.

        * *grayscale* - If true, the effect is applied on grayscale images.

        **NOTE**
        For OpenCV versions <= 2.3.0
        -- this acts as Convience function derived from the :py:meth:`smooth`
           method.

        For OpenCV versions higher than 2.3.0. i.e >= 2.3.0
        -- cv2.GaussianBlur function is called
        """
        if is_tuple(window):
            win_x, win_y = window
            if win_x >= 0 and win_y >= 0 and win_x % 2 == 1 and win_y % 2 == 1:
                pass
            else:
                logger.warning("The aperture (win_x,win_y) must be odd number "
                               "and greater than 0.")
                return None

        elif is_number(window):
            window = (window, window)
        else:
            window = (3, 3)  # set the default aperture window size (3x3)

        image_gauss = cv2.GaussianBlur(self._ndarray, window, sigma_x,
                                       None, sigma_y)

        if grayscale:
            image_gauss = cv2.GaussianBlur(self.get_gray_ndarray(), window,
                                           sigma_x, None, sigma_y)
            return Image(image_gauss, color_space=ColorSpace.GRAY)
        else:
            image_gauss = cv2.GaussianBlur(self._ndarray, window, sigma_x,
                                           None, sigma_y)
            return Image(image_gauss, color_space=self._colorSpace)

    def invert(self):
        """
        **SUMMARY**

        Invert (negative) the image note that this can also be done with the
        unary minus (-) operator. For binary image this turns black into white
        and white into black (i.e. white is the new black).

        **RETURNS**

        The opposite of the current image.

        **EXAMPLE**

        >>> img  = Image("polar_bear_in_the_snow.png")
        >>> img.invert().save("black_bear_at_night.png")

        **SEE ALSO**

        :py:meth:`binarize`

        """
        return -self

    def grayscale(self):
        """
        **SUMMARY**

        This method returns a gray scale version of the image. It makes
        everything look like an old movie.

        **RETURNS**

        A grayscale SimpleCV image.

        **EXAMPLE**

        >>> img = Image("lenna")
        >>> img.grayscale().binarize().show()

        **SEE ALSO**

        :py:meth:`binarize`
        """
        raise Exception('Deprecated! use to_gray()')

    def flip_horizontal(self):
        """
        **SUMMARY**

        Horizontally mirror an image.


        .. Warning::
          Note that flip does not mean rotate 180 degrees! The two are
          different.

        **RETURNS**

        The flipped SimpleCV image.

        **EXAMPLE**

        >>> img = Image("lenna")
        >>> upsidedown = img.flip_horizontal()


        **SEE ALSO**

        :py:meth:`flip_vertical`
        :py:meth:`rotate`

        """
        flip_array = cv2.flip(self._ndarray, 1)
        return Image(flip_array, color_space=self._colorSpace)

    def flip_vertical(self):
        """
        **SUMMARY**

        Vertically mirror an image.


        .. Warning::
          Note that flip does not mean rotate 180 degrees! The two are
          different.

        **RETURNS**

        The flipped SimpleCV image.

        **EXAMPLE**

        >>> img = Image("lenna")
        >>> img = img.flip_vertical()


        **SEE ALSO**

        :py:meth:`rotate`
        :py:meth:`flip_horizontal`

        """
        flip_array = cv2.flip(self._ndarray, 0)
        return Image(flip_array, color_space=self._colorSpace)

    def stretch(self, thresh_low=0, thresh_high=255):
        """
        **SUMMARY**

        The stretch filter works on a greyscale image, if the image
        is color, it returns a greyscale image.  The filter works by
        taking in a lower and upper threshold.  Anything below the lower
        threshold is pushed to black (0) and anything above the upper
        threshold is pushed to white (255)

        **PARAMETERS**

        * *thresh_low* - The lower threshold for the stretch operation.
          This should be a value between 0 and 255.

        * *thresh_high* - The upper threshold for the stretch operation.
          This should be a value between 0 and 255.

        **RETURNS**

        A gray scale version of the image with the appropriate histogram
        stretching.


        **EXAMPLE**

        >>> img = Image("orson_welles.jpg")
        >>> img2 = img.stretch(56.200)
        >>> img2.show()

        **NOTES**

        TODO - make this work on RGB images with thresholds for each channel.

        **SEE ALSO**

        :py:meth:`binarize`
        :py:meth:`equalize`

        """
        threshold, array = cv2.threshold(self.get_gray_ndarray(), thresh_low,
                                         255, cv2.THRESH_TOZERO)
        array = cv2.bitwise_not(array)
        threshold, array = cv2.threshold(array, 255 - thresh_high, 255,
                                         cv2.THRESH_TOZERO)
        array = cv2.bitwise_not(array)
        return Image(array, color_space=ColorSpace.GRAY)

    def gamma_correct(self, gamma=1):

        """
        **DESCRIPTION**

        Transforms an image according to Gamma Correction also known as
        Power Law Transform.

        **PARAMETERS**

        * *gamma* - A non-negative real number.

        **RETURNS**

        A Gamma corrected image.

        **EXAMPLE**

        >>> img = Image(simplecv)
        >>> img.show()
        >>> img.gamma_correct(1.5).show()
        >>> img.gamma_correct(0.7).show()

        """
        if gamma < 0:
            return "Gamma should be a non-negative real number"
        scale = 255.0
        dst = (((1.0 / scale) * self._ndarray) ** gamma) * scale
        return Image(dst.astype(self.dtype), color_space=self._colorSpace)

    def binarize(self, thresh=None, maxv=255, blocksize=0, p=5):
        """
        **SUMMARY**

        Do a binary threshold the image, changing all values below thresh to
        maxv and all above to black.  If a color tuple is provided, each color
        channel is thresholded separately.


        If threshold is -1 (default), an adaptive method (OTSU's method) is
        used.
        If then a blocksize is specified, a moving average over each region of
        block*block pixels a threshold is applied where threshold =
        local_mean - p.

        **PARAMETERS**

        * *thresh* - the threshold as an integer or an (r,g,b) tuple , where
          pixels below (darker) than thresh are set to to max value,
          and all values above this value are set to black. If this parameter
          is -1 we use Otsu's method.

        * *maxv* - The maximum value for pixels below the threshold. Ordinarily
         this should be 255 (white)

        * *blocksize* - the size of the block used in the adaptive binarize
          operation.

        .. Warning::
          This parameter must be an odd number.

        * *p* - The difference from the local mean to use for thresholding
         in Otsu's method.

        **RETURNS**

        A binary (two colors, usually black and white) SimpleCV image. This
        works great for the find_blobs family of functions.

        **EXAMPLE**

        Example of a vanila threshold versus an adaptive threshold:

        >>> img = Image("orson_welles.jpg")
        >>> b1 = img.binarize(128)
        >>> b2 = img.binarize(blocksize=11,p=7)
        >>> b3 = b1.side_by_side(b2)
        >>> b3.show()


        **NOTES**

        `Otsu's Method Description<http://en.wikipedia.org/wiki/Otsu's_method>`

        **SEE ALSO**

        :py:meth:`threshold`
        :py:meth:`find_blobs`
        :py:meth:`invert`
        :py:meth:`dilate`
        :py:meth:`erode`

        """
        if is_tuple(thresh):
            b = self._ndarray[:, :, 0].copy()
            g = self._ndarray[:, :, 1].copy()
            r = self._ndarray[:, :, 2].copy()

            r = cv2.threshold(r, thresh[0], maxv, cv2.THRESH_BINARY_INV)[1]
            g = cv2.threshold(g, thresh[1], maxv, cv2.THRESH_BINARY_INV)[1]
            b = cv2.threshold(b, thresh[2], maxv, cv2.THRESH_BINARY_INV)[1]
            array = r + g + b
            return Image(array, color_space=ColorSpace.GRAY)

        elif thresh is None:
            if blocksize:
                array = cv2.adaptiveThreshold(self.get_gray_ndarray(), maxv,
                                              cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                              cv2.THRESH_BINARY_INV,
                                              blocksize, p)
            else:
                array = cv2.threshold(
                    self.get_gray_ndarray(), -1, float(maxv),
                    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
            return Image(array, color_space=ColorSpace.GRAY)
        else:
            # desaturate the image, and apply the new threshold
            array = cv2.threshold(self.get_gray_ndarray(), thresh,
                                  maxv, cv2.THRESH_BINARY_INV)[1]
            return Image(array, color_space=ColorSpace.GRAY)

    def mean_color(self, color_space=None):
        """
        **SUMMARY**

        This method finds the average color of all the pixels in the image and
        displays tuple in the colorspace specfied by the user.
        If no colorspace is specified , (B,G,R) colorspace is taken as default.

        **RETURNS**

        A tuple of the average image values. Tuples are in the channel order.
        *For most images this means the results are (B,G,R).*

        **EXAMPLE**

        >>> img = Image('lenna')
        >>> # returns tuple in Image's colorspace format.
        >>> colors = img.mean_color()
        >>> colors1 = img.mean_color('BGR')   # returns tuple in (B,G,R) format
        >>> colors2 = img.mean_color('RGB')   # returns tuple in (R,G,B) format
        >>> colors3 = img.mean_color('HSV')   # returns tuple in (H,S,V) format
        >>> colors4 = img.mean_color('XYZ')   # returns tuple in (X,Y,Z) format
        >>> colors5 = img.mean_color('Gray')  # returns float of mean intensity
        >>> colors6 = img.mean_color('YCrCb') # returns tuple in Y,Cr,Cb format
        >>> colors7 = img.mean_color('HLS')   # returns tuple in (H,L,S) format


        """
        if color_space is None:
            array = self._ndarray
            if len(self._ndarray.shape) == 2:
                return np.average(array)
        elif color_space == 'BGR':
            array = self.to_bgr().get_ndarray()
        elif color_space == 'RGB':
            array = self.to_rgb().get_ndarray()
        elif color_space == 'HSV':
            array = self.to_hsv().get_ndarray()
        elif color_space == 'XYZ':
            array = self.to_xyz().get_ndarray()
        elif color_space == 'Gray':
            array = self.get_gray_ndarray()
            return np.average(array)
        elif color_space == 'YCrCb':
            array = self.to_ycrcb().get_ndarray()
        elif color_space == 'HLS':
            array = self.to_hls().get_ndarray()
        else:
            logger.warning("Image.meanColor: There is no supported conversion "
                           "to the specified colorspace. Use one of these as "
                           "argument: 'BGR' , 'RGB' , 'HSV' , 'Gray' , 'XYZ' "
                           ", 'YCrCb' , 'HLS' .")
            return None
        return (np.average(array[:, :, 0]),
                np.average(array[:, :, 1]),
                np.average(array[:, :, 2]))

    def find_corners(self, maxnum=50, minquality=0.04, mindistance=1.0):
        """
        **SUMMARY**

        This will find corner Feature objects and return them as a FeatureSet
        strongest corners first.  The parameters give the number of corners to
        look for, the minimum quality of the corner feature, and the minimum
        distance between corners.

        **PARAMETERS**

        * *maxnum* - The maximum number of corners to return.

        * *minquality* - The minimum quality metric. This shoudl be a number
         between zero and one.

        * *mindistance* - The minimum distance, in pixels, between successive
         corners.

        **RETURNS**

        A featureset of :py:class:`Corner` features or None if no corners are
         found.


        **EXAMPLE**

        Standard Test:

        >>> img = Image("data/sampleimages/simplecv.png")
        >>> corners = img.find_corners()
        >>> if corners: True

        True

        Validation Test:

        >>> img = Image("data/sampleimages/black.png")
        >>> corners = img.find_corners()
        >>> if not corners: True

        True

        **SEE ALSO**

        :py:class:`Corner`
        :py:meth:`find_keypoints`

        """
        corner_coordinates = cv2.goodFeaturesToTrack(self.get_gray_ndarray(),
                                                     maxnum, minquality,
                                                     mindistance)

        corner_features = []
        for x, y in corner_coordinates[:, 0, :]:
            corner_features.append(Corner(self, x, y))

        return FeatureSet(corner_features)

    def find_blobs(self, threshval=None, minsize=10, maxsize=0,
                   threshblocksize=0, threshconstant=5, appx_level=3):
        """

        **SUMMARY**

        Find blobs  will look for continuous
        light regions and return them as Blob features in a FeatureSet.
        Parameters specify the binarize filter threshold value, and minimum and
        maximum size for blobs. If a threshold value is -1, it will use an
        adaptive threshold.  See binarize() for more information about
        thresholding.  The threshblocksize and threshconstant parameters are
        only used for adaptive threshold.


        **PARAMETERS**

        * *threshval* - the threshold as an integer or an (r,g,b) tuple , where
          pixels below (darker) than thresh are set to to max value,
          and all values above this value are set to black. If this parameter
          is -1 we use Otsu's method.

        * *minsize* - the minimum size of the blobs, in pixels, of the returned
         blobs. This helps to filter out noise.

        * *maxsize* - the maximim size of the blobs, in pixels, of the returned
         blobs.

        * *threshblocksize* - the size of the block used in the adaptive
          binarize operation. *TODO - make this match binarize*

        * *appx_level* - The blob approximation level - an integer for the
          maximum distance between the true edge and the
          approximation edge - lower numbers yield better approximation.

          .. warning::
            This parameter must be an odd number.

        * *threshconstant* - The difference from the local mean to use for
         thresholding in Otsu's method. *TODO - make this match binarize*


        **RETURNS**

        Returns a featureset (basically a list) of :py:class:`blob` features.
        If no blobs are found this method returns None.

        **EXAMPLE**

        >>> img = Image("lenna")
        >>> fs = img.find_blobs()
        >>> if fs is not None:
        >>>     fs.draw()

        **NOTES**

        .. Warning::
          For blobs that live right on the edge of the image OpenCV reports the
          position and width height as being one over for the true position.
          E.g. if a blob is at (0,0) OpenCV reports its position as (1,1).
          Likewise the width and height for the other corners is reported as
          being one less than the width and height. This is a known bug.

        **SEE ALSO**
        :py:meth:`threshold`
        :py:meth:`binarize`
        :py:meth:`invert`
        :py:meth:`dilate`
        :py:meth:`erode`
        :py:meth:`find_blobs_from_palette`
        :py:meth:`smart_find_blobs`
        """
        if maxsize == 0:
            maxsize = self.width * self.height
        #create a single channel image, thresholded to parameters

        blobmaker = BlobMaker()
        blobs = blobmaker.extract_from_binary(
            self.binarize(threshval, 255, threshblocksize,
                          threshconstant).invert(),
            self, minsize=minsize, maxsize=maxsize, appx_level=appx_level)

        if not len(blobs):
            return None

        return FeatureSet(blobs).sort_area()

    def find_skintone_blobs(self, minsize=10, maxsize=0, dilate_iter=1):
        """
        **SUMMARY**

        Find Skintone blobs will look for continuous
        regions of Skintone in a color image and return them as Blob features
        in a FeatureSet. Parameters specify the binarize filter threshold
        value, and minimum and maximum size for blobs. If a threshold value is
        -1, it will use an adaptive threshold.  See binarize() for more
        information about thresholding.  The threshblocksize and threshconstant
        parameters are only used for adaptive threshold.


        **PARAMETERS**

        * *minsize* - the minimum size of the blobs, in pixels, of the returned
         blobs. This helps to filter out noise.

        * *maxsize* - the maximim size of the blobs, in pixels, of the returned
         blobs.

        * *dilate_iter* - the number of times to run the dilation operation.

        **RETURNS**

        Returns a featureset (basically a list) of :py:class:`blob` features.
        If no blobs are found this method returns None.

        **EXAMPLE**

        >>> img = Image("lenna")
        >>> fs = img.find_skintone_blobs()
        >>> if fs is not None:
        >>>     fs.draw()

        **NOTES**
        It will be really awesome for making UI type stuff, where you want to
        track a hand or a face.

        **SEE ALSO**
        :py:meth:`threshold`
        :py:meth:`binarize`
        :py:meth:`invert`
        :py:meth:`dilate`
        :py:meth:`erode`
        :py:meth:`find_blobs_from_palette`
        :py:meth:`smart_find_blobs`
        """
        if maxsize == 0:
            maxsize = self.width * self.height
        mask = self.get_skintone_mask(dilate_iter)
        blobmaker = BlobMaker()
        blobs = blobmaker.extract_from_binary(mask, self, minsize=minsize,
                                              maxsize=maxsize)
        if not len(blobs):
            return None
        return FeatureSet(blobs).sort_area()

    def get_skintone_mask(self, dilate_iter=0):
        """
        **SUMMARY**

        Find Skintone mask will look for continuous
        regions of Skintone in a color image and return a binary mask where the
        white pixels denote Skintone region.

        **PARAMETERS**

        * *dilate_iter* - the number of times to run the dilation operation.


        **RETURNS**

        Returns a binary mask.

        **EXAMPLE**

        >>> img = Image("lenna")
        >>> mask = img.findSkintoneMask()
        >>> mask.show()

        """
        if self.is_ycrcb():
            ycrcb = self._ndarray
        else:
            ycrcb = self.to_ycrcb().get_ndarray()

        y = np.zeros((256, 1), dtype=uint8)
        y[5:] = 255
        cr = np.zeros((256, 1), dtype=uint8)
        cr[140:180] = 255
        cb = np.zeros((256, 1), dtype=uint8)
        cb[77:135] = 255

        y_array = ycrcb[:, :, 0]
        cr_array = ycrcb[:, :, 1]
        cb_array = ycrcb[:, :, 2]

        y_array = cv2.LUT(y_array, y)
        cr_array = cv2.LUT(cr_array, cr)
        cb_array = cv2.LUT(cb_array, cb)

        array = np.dstack((y_array, cr_array, cb_array))

        mask = Image(array, color_space=ColorSpace.YCrCb)
        mask = mask.binarize((128, 128, 128))
        mask = mask.to_rgb().binarize()
        mask.dilate(dilate_iter)
        return mask

    # this code is based on code that's based on code from
    # http://blog.jozilla.net/2008/06/27/
    # fun-with-python-opencv-and-face-detection/
    def find_haar_features(self, cascade, scale_factor=1.2, min_neighbors=2,
                           use_canny=cv2.cv.CV_HAAR_DO_CANNY_PRUNING,
                           min_size=(20, 20), max_size=(1000, 1000)):
        """
        **SUMMARY**

        A Haar like feature cascase is a really robust way of finding the
        location of a known object. This technique works really well for a few
        specific applications like face, pedestrian, and vehicle detection. It
        is worth noting that this approach **IS NOT A MAGIC BULLET** . Creating
        a cascade file requires a large number of images that have been sorted
        by a human.vIf you want to find Haar Features (useful for face
        detection among other purposes) this will return Haar feature objects
        in a FeatureSet.

        For more information, consult the cv2.CascadeClassifier documentation.

        To see what features are available run img.list_haar_features() or you
        can provide your own haarcascade file if you have one available.

        Note that the cascade parameter can be either a filename, or a
        HaarCascade loaded with cv2.CascadeClassifier(),
        or a SimpleCV HaarCascade object.

        **PARAMETERS**

        * *cascade* - The Haar Cascade file, this can be either the path to a
          cascade file or a HaarCascased SimpleCV object that has already been
          loaded.

        * *scale_factor* - The scaling factor for subsequent rounds of the
          Haar cascade (default 1.2) in terms of a percentage
          (i.e. 1.2 = 20% increase in size)

        * *min_neighbors* - The minimum number of rectangles that makes up an
          object. Ususally detected faces are clustered around the face, this
          is the number of detections in a cluster that we need for detection.
          Higher values here should reduce false positives and decrease false
          negatives.

        * *use-canny* - Whether or not to use Canny pruning to reject areas
         with too many edges (default yes, set to 0 to disable)

        * *min_size* - Minimum window size. By default, it is set to the size
          of samples the classifier has been trained on ((20,20) for face
          detection)

        * *max_size* - Maximum window size. By default, it is set to the size
          of samples the classifier has been trained on ((1000,1000) for face
          detection)

        **RETURNS**

        A feature set of HaarFeatures

        **EXAMPLE**

        >>> faces = HaarCascade(
            ...         "./SimpleCV/data/Features/HaarCascades/face.xml",
            ...         "myFaces")
        >>> cam = Camera()
        >>> while True:
        >>>     f = cam.get_image().find_haar_features(faces)
        >>>     if f is not None:
        >>>          f.show()

        **NOTES**

        OpenCV Docs:
        - http://opencv.willowgarage.com/documentation/python/
          objdetect_cascade_classification.html

        Wikipedia:
        - http://en.wikipedia.org/wiki/Viola-Jones_object_detection_framework
        - http://en.wikipedia.org/wiki/Haar-like_features

        The video on this pages shows how Haar features and cascades work to
        located faces:
        - http://dismagazine.com/dystopia/evolved-lifestyles/8115/
        anti-surveillance-how-to-hide-from-machines/

        """

        #lovely.  This segfaults if not present
        from simplecv.features.haar_cascade import HaarCascade

        if isinstance(cascade, basestring):
            cascade = HaarCascade(cascade)
            if not cascade.get_cascade():
                return None
        elif isinstance(cascade, HaarCascade):
            pass
        else:
            logger.warning('Could not initialize HaarCascade. '
                           'Enter Valid cascade value.')
            return None

        haar_classify = cv2.CascadeClassifier(cascade.get_fhandle())
        objects = haar_classify.detectMultiScale(
            self.get_gray_ndarray(), scaleFactor=scale_factor,
            minNeighbors=min_neighbors, minSize=min_size,
            flags=use_canny)

        if objects is not None:
            return FeatureSet(
                [HaarFeature(self, o, cascade, True) for o in objects])

        return None

    def draw_circle(self, ctr, rad, color=(0, 0, 0), thickness=1):
        """
        **SUMMARY**

        Draw a circle on the image.

        **PARAMETERS**

        * *ctr* - The center of the circle as an (x,y) tuple.
        * *rad* - The radius of the circle in pixels
        * *color* - A color tuple (default black)
        * *thickness* - The thickness of the circle, -1 means filled in.

        **RETURNS**

        .. Warning::
          This is an inline operation. Nothing is returned, but a circle is
          drawn on the images's drawing layer.

        **EXAMPLE**

        >>> img = Image("lenna")
        >>> img.draw_circle(
            ...    (img.width / 2, img.height / 2),
            ...    r=50, color=Colors.RED, width=3)
        >>> img.show()

        **NOTES**

        .. Warning::
          Note that this function is depricated, try to use
          DrawingLayer.circle() instead.

        **SEE ALSO**

        :py:meth:`draw_line`
        :py:meth:`draw_text`
        :py:meth:`dl`
        :py:meth:`draw_rectangle`
        :py:class:`DrawingLayer`

        """
        if thickness < 0:
            self.get_drawing_layer().circle((int(ctr[0]), int(ctr[1])),
                                            int(rad), color, int(thickness),
                                            filled=True)
        else:
            self.get_drawing_layer().circle((int(ctr[0]), int(ctr[1])),
                                            int(rad), color, int(thickness))

    def draw_line(self, pt1, pt2, color=(0, 0, 0), thickness=1):
        """
        **SUMMARY**
        Draw a line on the image.


        **PARAMETERS**

        * *pt1* - the first point for the line (tuple).
        * *pt2* - the second point on the line (tuple).
        * *color* - a color tuple (default black).
        * *thickness* the thickness of the line in pixels.

        **RETURNS**

        .. Warning::
          This is an inline operation. Nothing is returned, but a circle is
          drawn on the images's
          drawing layer.

        **EXAMPLE**

        >>> img = Image("lenna")
        >>> img.draw_line(
            ...    (0,0), (img.width, img.height),
            ...      color=Color.RED, thickness=3)
        >>> img.show()

        **NOTES**

        .. Warning::
           Note that this function is depricated, try to use
           DrawingLayer.line() instead.

        **SEE ALSO**

        :py:meth:`draw_text`
        :py:meth:`dl`
        :py:meth:`draw_circle`
        :py:meth:`draw_rectangle`

        """
        pt1 = (int(pt1[0]), int(pt1[1]))
        pt2 = (int(pt2[0]), int(pt2[1]))
        self.get_drawing_layer().line(pt1, pt2, color, thickness)

    def size(self):
        """
        **SUMMARY**

        Returns a tuple that lists the width and height of the image.

        **RETURNS**

        The width and height as a tuple.


        """
        return self.width, self.height

    def is_empty(self):
        """
        **SUMMARY**

        Checks if the image is empty by checking its width and height.

        **RETURNS**

        True if the image's size is (0, 0), False for any other size.

        """
        return self.size() == (0, 0)

    def split(self, cols, rows):
        """
        **SUMMARY**

        This method can be used to brak and image into a series of image
        chunks. Given number of cols and rows, splits the image into a cols x
        rows 2d array of cropped images

        **PARAMETERS**

        * *rows* - an integer number of rows.
        * *cols* - an integer number of cols.

        **RETURNS**

        A list of SimpleCV images.

        **EXAMPLE**

        >>> img = Image("lenna")
        >>> quadrant =img.split(2,2)
        >>> for f in quadrant:
        >>>    f.show()
        >>>    time.sleep(1)


        **NOTES**

        TODO: This should return and ImageList

        """
        crops = []

        wratio = self.width / cols
        hratio = self.height / rows

        for i in range(rows):
            row = []
            for j in range(cols):
                row.append(self.crop(j * wratio, i * hratio, wratio, hratio))
            crops.append(row)

        return crops

    def split_channels(self):
        """
        **SUMMARY**

        Split the channels of an image.

        **RETURNS**

        A tuple of of 3 image objects.

        **EXAMPLE**

        >>> img = Image("lenna")
        >>> data = img.split_channels()
        >>> for d in data:
        >>>    d.show()
        >>>    time.sleep(1)

        **SEE ALSO**

        :py:meth:`merge_channels`
        """
        chanel_0 = self._ndarray[:, :, 0]
        chanel_1 = self._ndarray[:, :, 1]
        chanel_2 = self._ndarray[:, :, 2]

        return (Image(chanel_0, color_space=ColorSpace.GRAY),
                Image(chanel_1, color_space=ColorSpace.GRAY),
                Image(chanel_2, color_space=ColorSpace.GRAY))

    def merge_channels(self, r=None, g=None, b=None,
                       color_space=ColorSpace.UNKNOWN):
        """
        **SUMMARY**

        Merge channels is the oposite of split_channels. The image takes one
        image for each of the R,G,B channels and then recombines them into a
        single image. Optionally any of these channels can be None.

        **PARAMETERS**

        * *r* - The r or last channel  of the result SimpleCV Image.
        * *g* - The g or center channel of the result SimpleCV Image.
        * *b* - The b or first channel of the result SimpleCV Image.


        **RETURNS**

        A SimpleCV Image.

        **EXAMPLE**

        >>> img = Image("lenna")
        >>> r, g, b = img.split_channels()
        >>> r = r.binarize()
        >>> g = g.binarize()
        >>> b = b.binarize()
        >>> result = img.merge_channels(r, g, b)
        >>> result.show()


        **SEE ALSO**
        :py:meth:`split_channels`

        """
        if r is None and g is None and b is None:
            logger.warning("Image.merge_channels - we need at least "
                           "one valid channel")
            return None

        if r is None:
            r = Image(self.size(), color_space=ColorSpace.GRAY)
        if g is None:
            g = Image(self.size(), color_space=ColorSpace.GRAY)
        if b is None:
            b = Image(self.size(), color_space=ColorSpace.GRAY)

        array = np.dstack((r.get_ndarray(),
                           g.get_ndarray(),
                           b.get_ndarray()))
        return Image(array, color_space=color_space)

    def apply_hls_curve(self, hcurve, lcurve, scurve):
        """
        **SUMMARY**

        Apply a color correction curve in HSL space. This method can be used
        to change values for each channel. The curves are
        :py:class:`ColorCurve` class objects.

        **PARAMETERS**

        * *hcurve* - the hue ColorCurve object.
        * *lcurve* - the lightnes / value ColorCurve object.
        * *scurve* - the saturation ColorCurve object

        **RETURNS**

        A SimpleCV Image

        **EXAMPLE**

        >>> img = Image("lenna")
        >>> hc = ColorCurve([[0,0], [100, 120], [180, 230], [255, 255]])
        >>> lc = ColorCurve([[0,0], [90, 120], [180, 230], [255, 255]])
        >>> sc = ColorCurve([[0,0], [70, 110], [180, 230], [240, 255]])
        >>> img2 = img.apply_hls_curve(hc,lc,sc)

        **SEE ALSO**

        :py:class:`ColorCurve`
        :py:meth:`apply_rgb_curve`
        """
        #TODO CHECK ROI
        #TODO CHECK CURVE SIZE
        #TODO CHECK CURVE SIZE

        # Move to HLS space
        array = self.to_hls().get_ndarray()

        # now apply the color curve correction
        array[:, :, 0] = np.take(hcurve.curve, array[:, :, 0])
        array[:, :, 1] = np.take(lcurve.curve, array[:, :, 1])
        array[:, :, 2] = np.take(scurve.curve, array[:, :, 2])

        # Move back to original color space
        array = Image.convert(array, ColorSpace.HLS, self._colorSpace)
        return Image(array, color_space=self._colorSpace)

    def apply_rgb_curve(self, rcurve, gcurve, bcurve):
        """
        **SUMMARY**

        Apply a color correction curve in RGB space. This method can be used
        to change values for each channel. The curves are
        :py:class:`ColorCurve` class objects.

        **PARAMETERS**

        * *rcurve* - the red ColorCurve object, or appropriately formatted
         list
        * *gcurve* - the green ColorCurve object, or appropriately formatted
         list
        * *bcurve* - the blue ColorCurve object, or appropriately formatted
         list

        **RETURNS**

        A SimpleCV Image

        **EXAMPLE**

        >>> img = Image("lenna")
        >>> rc = ColorCurve([[0,0], [100, 120], [180, 230], [255, 255]])
        >>> gc = ColorCurve([[0,0], [90, 120], [180, 230], [255, 255]])
        >>> bc = ColorCurve([[0,0], [70, 110], [180, 230], [240, 255]])
        >>> img2 = img.apply_rgb_curve(rc,gc,bc)

        **SEE ALSO**

        :py:class:`ColorCurve`
        :py:meth:`apply_hls_curve`

        """
        if isinstance(bcurve, list):
            bcurve = ColorCurve(bcurve)
        if isinstance(gcurve, list):
            gcurve = ColorCurve(gcurve)
        if isinstance(rcurve, list):
            rcurve = ColorCurve(rcurve)

        array = self._ndarray.copy()
        array[:, :, 0] = np.take(bcurve.curve, array[:, :, 0])
        array[:, :, 1] = np.take(gcurve.curve, array[:, :, 1])
        array[:, :, 2] = np.take(rcurve.curve, array[:, :, 2])
        return Image(array, color_space=self._colorSpace)

    def apply_intensity_curve(self, curve):
        """
        **SUMMARY**

        Intensity applied to all three color channels

        **PARAMETERS**

        * *curve* - a ColorCurve object, or 2d list that can be conditioned
         into one

        **RETURNS**

        A SimpleCV Image

        **EXAMPLE**

        >>> img = Image("lenna")
        >>> cc = ColorCurve([[0,0], [100, 120], [180, 230], [255, 255]])
        >>> img2 = img.apply_rgb_curve(cc)

        **SEE ALSO**

        :py:class:`ColorCurve`
        :py:meth:`apply_hls_curve`

        """
        return self.apply_rgb_curve(curve, curve, curve)

    def color_distance(self, color=Color.BLACK):
        """
        **SUMMARY**

        Returns an image representing the distance of each pixel from a given
        color tuple, scaled between 0 (the given color) and 255. Pixels distant
        from the given tuple will appear as brighter and pixels closest to the
        target color will be darker.


        By default this will give image intensity (distance from pure black)

        **PARAMETERS**

        * *color*  - Color object or Color Tuple

        **RETURNS**

        A SimpleCV Image.

        **EXAMPLE**

        >>> img = Image("logo")
        >>> img2 = img.color_distance(color=Color.BLACK)
        >>> img2.show()


        **SEE ALSO**

        :py:meth:`binarize`
        :py:meth:`hue_distance`
        :py:meth:`find_blobs_from_mask`
        """
        # reshape our matrix to 1xN
        pixels = self._ndarray.copy().reshape(-1, 3)

        # calculate the distance each pixel is
        distances = spsd.cdist(pixels, [color])
        distances *= (255.0 / distances.max())  # normalize to 0 - 255
        array = distances.reshape(self.width, self.height)
        return Image(array)

    def hue_distance(self, color=Color.BLACK, minsaturation=20, minvalue=20,
                     maxvalue=255):
        """
        **SUMMARY**

        Returns an image representing the distance of each pixel from the given
        hue of a specific color.  The hue is "wrapped" at 180, so we have to
        take the shorter of the distances between them -- this gives a hue
        distance of max 90, which we'll scale into a 0-255 grayscale image.

        The minsaturation and minvalue are optional parameters to weed out very
        weak hue signals in the picture, they will be pushed to max distance
        [255]


        **PARAMETERS**

        * *color* - Color object or Color Tuple.
        * *minsaturation*  - the minimum saturation value for color
         (from 0 to 255).
        * *minvalue*  - the minimum hue value for the color
         (from 0 to 255).

        **RETURNS**

        A simpleCV image.

        **EXAMPLE**

        >>> img = Image("logo")
        >>> img2 = img.hue_distance(color=Color.BLACK)
        >>> img2.show()

        **SEE ALSO**

        :py:meth:`binarize`
        :py:meth:`hue_distance`
        :py:meth:`morph_open`
        :py:meth:`morph_close`
        :py:meth:`morph_gradient`
        :py:meth:`find_blobs_from_mask`

        """
        if isinstance(color, (float, int, long, complex)):
            color_hue = color
        else:
            color_hue = Color.hsv(color)[0]

        # again, gets transposed to vsh
        vsh_matrix = self.to_hsv().get_ndarray().reshape(-1, 3)
        hue_channel = np.cast['int'](vsh_matrix[:, 2])

        if color_hue < 90:
            hue_loop = 180
        else:
            hue_loop = -180
        #set whether we need to move back or forward on the hue circle

        distances = np.minimum(np.abs(hue_channel - color_hue),
                               np.abs(hue_channel - (color_hue + hue_loop)))
        #take the minimum distance for each pixel

        distances = np.where(
            np.logical_and(
                vsh_matrix[:, 0] > minvalue,
                vsh_matrix[:, 1] > minsaturation),
            distances * (255.0 / 90.0),  # normalize 0 - 90 -> 0 - 255
            # use the maxvalue if it false outside
            # of our value/saturation tolerances
            255.0)

        return Image(distances.reshape(self.width, self.height))

    def erode(self, iterations=1, kernelsize=3):
        """
        **SUMMARY**

        Apply a morphological erosion. An erosion has the effect of removing
        small bits of noise and smothing blobs.

        This implementation uses the default openCV 3X3 square kernel

        Erosion is effectively a local minima detector, the kernel moves over
        the image and takes the minimum value inside the kernel.
        iterations - this parameters is the number of times to apply/reapply
        the operation

        * See: http://en.wikipedia.org/wiki/Erosion_(morphology).

        * See: http://opencv.willowgarage.com/documentation/cpp/
         image_filtering.html#cv-erode

        * Example Use: A threshold/blob image has 'salt and pepper' noise.

        * Example Code: /examples/MorphologyExample.py

        **PARAMETERS**

        * *iterations* - the number of times to run the erosion operation.

        **RETURNS**

        A SimpleCV image.

        **EXAMPLE**

        >>> img = Image("lenna")
        >>> derp = img.binarize()
        >>> derp.erode(3).show()

        **SEE ALSO**
        :py:meth:`dilate`
        :py:meth:`binarize`
        :py:meth:`morph_open`
        :py:meth:`morph_close`
        :py:meth:`morph_gradient`
        :py:meth:`find_blobs_from_mask`

        """
        kern = cv2.getStructuringElement(cv2.MORPH_RECT,
                                         (kernelsize, kernelsize), (1, 1))
        array = cv2.erode(self._ndarray, kern, iterations=iterations)
        return Image(array, color_space=self._colorSpace)

    def dilate(self, iterations=1):
        """
        **SUMMARY**

        Apply a morphological dilation. An dilation has the effect of smoothing
        blobs while intensifying the amount of noise blobs.
        This implementation uses the default openCV 3X3 square kernel
        Erosion is effectively a local maxima detector, the kernel moves over
        the image and takes the maxima value inside the kernel.

        * See: http://en.wikipedia.org/wiki/Dilation_(morphology)

        * See: http://opencv.willowgarage.com/documentation/cpp/
         image_filtering.html#cv-dilate

        * Example Use: A part's blob needs to be smoother

        * Example Code: ./examples/MorphologyExample.py

        **PARAMETERS**

        * *iterations* - the number of times to run the dilation operation.

        **RETURNS**

        A SimpleCV image.

        **EXAMPLE**

        >>> img = Image("lenna")
        >>> derp = img.binarize()
        >>> derp.dilate(3).show()

        **SEE ALSO**

        :py:meth:`erode`
        :py:meth:`binarize`
        :py:meth:`morph_open`
        :py:meth:`morph_close`
        :py:meth:`morph_gradient`
        :py:meth:`find_blobs_from_mask`

        """
        kern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3), (1, 1))
        array = cv2.dilate(self._ndarray, kern, iterations=iterations)
        return Image(array, color_space=self._colorSpace)

    def morph_open(self):
        """
        **SUMMARY**

        morphologyOpen applies a morphological open operation which is
        effectively an erosion operation followed by a morphological dilation.
        This operation helps to 'break apart' or 'open' binary regions which
        are close together.


        * `Morphological opening on Wikipedia <http://en.wikipedia.org/wiki/
         Opening_(morphology)>`_

        * `OpenCV documentation <http://opencv.willowgarage.com/documentation/
         cpp/image_filtering.html#cv-morphologyex>`_

        * Example Use: two part blobs are 'sticking' together.

        * Example Code: ./examples/MorphologyExample.py

        **RETURNS**

        A SimpleCV image.

        **EXAMPLE**

        >>> img = Image("lenna")
        >>> derp = img.binarize()
        >>> derp.morph_open.show()

        **SEE ALSO**

        :py:meth:`erode`
        :py:meth:`dilate`
        :py:meth:`binarize`
        :py:meth:`morph_close`
        :py:meth:`morph_gradient`
        :py:meth:`find_blobs_from_mask`

        """
        kern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3), (1, 1))
        array = cv2.morphologyEx(src=self._ndarray, op=cv2.MORPH_OPEN,
                                 kernel=kern, anchor=(1, 1), iterations=1)
        return Image(array, color_space=self._colorSpace)

    def morph_close(self):
        """
        **SUMMARY**

        morphologyClose applies a morphological close operation which is
        effectively a dilation operation followed by a morphological erosion.
        This operation helps to 'bring together' or 'close' binary regions
        which are close together.


        * See: `Closing <http://en.wikipedia.org/wiki/Closing_(morphology)>`_

        * See: `Morphology from OpenCV <http://opencv.willowgarage.com/
         documentation/cpp/image_filtering.html#cv-morphologyex>`_

        * Example Use: Use when a part, which should be one blob is really two
         blobs.

        * Example Code: ./examples/MorphologyExample.py

        **RETURNS**

        A SimpleCV image.

        **EXAMPLE**

        >>> img = Image("lenna")
        >>> derp = img.binarize()
        >>> derp.morph_close.show()

        **SEE ALSO**

        :py:meth:`erode`
        :py:meth:`dilate`
        :py:meth:`binarize`
        :py:meth:`morph_open`
        :py:meth:`morph_gradient`
        :py:meth:`find_blobs_from_mask`

        """
        kern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3), (1, 1))
        array = cv2.morphologyEx(src=self._ndarray, op=cv2.MORPH_CLOSE,
                                 kernel=kern, anchor=(1, 1), iterations=1)
        return Image(array, color_space=self._colorSpace)

    def morph_gradient(self):
        """
        **SUMMARY**

        The morphological gradient is the difference betwen the morphological
        dilation and the morphological gradient. This operation extracts the
        edges of a blobs in the image.


        * `See Morph Gradient of Wikipedia <http://en.wikipedia.org/wiki/
        Morphological_Gradient>`_

        * `OpenCV documentation <http://opencv.willowgarage.com/documentation/
         cpp/image_filtering.html#cv-morphologyex>`_

        * Example Use: Use when you have blobs but you really just want to know
         the blob edges.

        * Example Code: ./examples/MorphologyExample.py


        **RETURNS**

        A SimpleCV image.

        **EXAMPLE**

        >>> img = Image("lenna")
        >>> derp = img.binarize()
        >>> derp.morph_gradient.show()

        **SEE ALSO**

        :py:meth:`erode`
        :py:meth:`dilate`
        :py:meth:`binarize`
        :py:meth:`morph_open`
        :py:meth:`morph_close`
        :py:meth:`find_blobs_from_mask`

        """
        kern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3), (1, 1))
        array = cv2.morphologyEx(self._ndarray, cv2.MORPH_GRADIENT, kern)
        return Image(array, color_space=self._colorSpace)

    def histogram(self, numbins=50):
        """
        **SUMMARY**

        Return a numpy array of the 1D histogram of intensity for pixels in
        the image
        Single parameter is how many "bins" to have.


        **PARAMETERS**

        * *numbins* - An interger number of bins in a histogram.

        **RETURNS**

        A list of histogram bin values.

        **EXAMPLE**

        >>> img = Image('lenna')
        >>> hist = img.histogram()

        **SEE ALSO**

        :py:meth:`hue_histogram`

        """
        hist, bin_edges = np.histogram(self.get_gray_ndarray(), bins=numbins)
        return hist.tolist()

    def hue_histogram(self, bins=179, dynamic_range=True):

        """
        **SUMMARY**

        Returns the histogram of the hue channel for the image


        **PARAMETERS**

        * *numbins* - An interger number of bins in a histogram.

        **RETURNS**

        A list of histogram bin values.

        **SEE ALSO**

        :py:meth:`histogram`

        """
        if dynamic_range:
            return np.histogram(self.to_hsv().get_ndarray()[:, :, 2],
                                bins=bins)[0]
        else:
            return np.histogram(self.to_hsv().get_ndarray()[:, :, 2],
                                bins=bins, range=(0.0, 360.0))[0]

    def hue_peaks(self, bins=179):
        """
        **SUMMARY**

        Takes the histogram of hues, and returns the peak hue values, which
        can be useful for determining what the "main colors" in a picture.

        The bins parameter can be used to lump hues together, by default it
        is 179 (the full resolution in OpenCV's HSV format)

        Peak detection code taken from https://gist.github.com/1178136
        Converted from/based on a MATLAB script at
        http://billauer.co.il/peakdet.html

        Returns a list of tuples, each tuple contains the hue, and the fraction
        of the image that has it.

        **PARAMETERS**

        * *bins* - the integer number of bins, between 0 and 179.

        **RETURNS**

        A list of (hue,fraction) tuples.

        """
        # keyword arguments:
        # y_axis -- A list containg the signal over which to find peaks
        # x_axis -- A x-axis whose values correspond to the
        # 'y_axis' list and is used
        #     in the return to specify the postion of the peaks.
        #     If omitted the index
        #     of the y_axis is used. (default: None)
        # lookahead -- (optional) distance to look ahead from a peak
        # candidate to
        #     determine if it is the actual peak (default: 500)
        #     '(sample / period) / f' where '4 >= f >= 1.25' might be a good
        #      value
        # delta -- (optional) this specifies a minimum difference between
        #     a peak and the following points, before a peak may be considered
        #     a peak. Useful to hinder the algorithm from picking up false
        #     peaks towards to end of the signal. To work well delta should
        #     be set to 'delta >= RMSnoise * 5'.
        #     (default: 0)
        #         Delta function causes a 20% decrease in speed, when omitted
        #         Correctly used it can double the speed of the algorithm
        # return --  Each cell of the lists contains a tupple of:
        #     (position, peak_value)
        #     to get the average peak value
        #     do 'np.mean(maxtab, 0)[1]' on the results

        y_axis, x_axis = np.histogram(self.to_hsv().get_ndarray()[:, :, 2],
                                      bins=bins)
        x_axis = x_axis[0:bins]
        lookahead = int(bins / 17)
        delta = 0

        maxtab = []
        mintab = []
        dump = []  # Used to pop the first hit which always if false

        length = len(y_axis)
        if x_axis is None:
            x_axis = range(length)

        #perform some checks
        if length != len(x_axis):
            raise ValueError("Input vectors y_axis and "
                             "x_axis must have same length")
        if lookahead < 1:
            raise ValueError("Lookahead must be above '1' in value")
        if not (np.isscalar(delta) and delta >= 0):
            raise ValueError("delta must be a positive number")

        #needs to be a numpy array
        y_axis = np.asarray(y_axis)

        #maxima and minima candidates are temporarily stored in
        #mx and mn respectively
        mn, mx = np.Inf, -np.Inf

        #Only detect peak if there is 'lookahead' amount of points after it
        for index, (x, y) in enumerate(
                zip(x_axis[:-lookahead], y_axis[:-lookahead])):
            if y > mx:
                mx = y
                mxpos = x
            if y < mn:
                mn = y
                mnpos = x

            ####look for max####
            if y < mx - delta and mx != np.Inf:
                # Maxima peak candidate found
                # look ahead in signal to ensure that
                # this is a peak and not jitter
                if y_axis[index:index + lookahead].max() < mx:
                    maxtab.append((mxpos, mx))
                    dump.append(True)
                    #set algorithm to only find minima now
                    mx = np.Inf
                    mn = np.Inf

            ####look for min####
            if y > mn + delta and mn != -np.Inf:
                # Minima peak candidate found
                # look ahead in signal to ensure that
                # this is a peak and not jitter
                if y_axis[index:index + lookahead].min() > mn:
                    mintab.append((mnpos, mn))
                    dump.append(False)
                    #set algorithm to only find maxima now
                    mn = -np.Inf
                    mx = -np.Inf

        #Remove the false hit on the first value of the y_axis
        try:
            if dump[0]:
                maxtab.pop(0)
                #print "pop max"
            else:
                mintab.pop(0)
                #print "pop min"
            del dump
        except IndexError:
            #no peaks were found, should the function return empty lists?
            pass

        huetab = []
        for hue, pixelcount in maxtab:
            huetab.append((hue, pixelcount / float(self.width * self.height)))
        return huetab

    def __getitem__(self, coord):
        if not (isinstance(coord, (list, tuple)) and len(coord) == 2):
            raise Exception('Not implemented for {}'.format(coord))
        if isinstance(coord[0], types.SliceType) \
                or isinstance(coord[1], types.SliceType):
            return Image(self._ndarray[coord],
                         color_space=self._colorSpace)
        else:
            return self._ndarray[coord].tolist()

    def __setitem__(self, coord, value):
        if not (isinstance(coord, (list, tuple)) and len(coord) == 2):
            raise Exception('Not implemented for {}'.format(coord))
        self._ndarray[coord] = value

    def __sub__(self, other):
        if isinstance(other, Image):
            if self.size() != other.size():
                warnings.warn("Both images should have same dimensions. "
                              "Returning None.")
                return None
            array = cv2.subtract(self._ndarray, other.get_ndarray())
            return Image(array, color_space=self._colorSpace)
        else:
            array = (self._ndarray - other).astype(self.dtype)
            return Image(array, color_space=self._colorSpace)

    def __add__(self, other):
        if isinstance(other, Image):
            if self.size() != other.size():
                warnings.warn("Both images should have same dimensions. "
                              "Returning None.")
                return None
            array = cv2.add(self._ndarray, other.get_ndarray())
            return Image(array, color_space=self._colorSpace)
        else:
            array = self._ndarray + other
            return Image(array, color_space=self._colorSpace)

    def __and__(self, other):
        if isinstance(other, Image):
            if self.size() != other.size():
                warnings.warn("Both images should have same dimensions. "
                              "Returning None.")
                return None
            array = self._ndarray & other.get_ndarray()
            return Image(array, color_space=self._colorSpace)
        else:
            array = self._ndarray & other
            return Image(array, color_space=self._colorSpace)

    def __or__(self, other):
        if isinstance(other, Image):
            if self.size() != other.size():
                warnings.warn("Both images should have same dimensions. "
                              "Returning None.")
                return None
            array = self._ndarray | other.get_ndarray()
            return Image(array, color_space=self._colorSpace)
        else:
            array = self._ndarray | other
            return Image(array, color_space=self._colorSpace)

    def __div__(self, other):
        if isinstance(other, Image):
            if self.size() != other.size():
                warnings.warn("Both images should have same dimensions. "
                              "Returning None.")
                return None
            array = cv2.divide(self._ndarray, other.get_ndarray())
            return Image(array, color_space=self._colorSpace)
        else:
            array = (self._ndarray / other).astype(self.dtype)
            return Image(array, color_space=self._colorSpace)

    def __mul__(self, other):
        if isinstance(other, Image):
            if self.size() != other.size():
                warnings.warn("Both images should have same dimensions. "
                              "Returning None.")
                return None
            array = cv2.multiply(self._ndarray, other.get_ndarray())
            return Image(array, color_space=self._colorSpace)
        else:
            array = (self._ndarray * other).astype(self.dtype)
            return Image(array, color_space=self._colorSpace)

    def __pow__(self, power):
        if isinstance(power, int):
            array = cv2.pow(self._ndarray, power)
            return Image(array, color_space=self._colorSpace)
        else:
            raise ValueError('Cant make exponentiation with this type')

    def __neg__(self):
        array = ~self._ndarray
        return Image(array, color_space=self._colorSpace)

    def __invert__(self):
        return self.__neg__()

    def max(self, other):
        """
        **SUMMARY**

        The maximum value of my image, and the other image, in each channel
        If other is a number, returns the maximum of that and the number

        **PARAMETERS**

        * *other* - Image of the same size or a number.

        **RETURNS**

        A SimpelCV image.

        """
        if isinstance(other, Image):
            if self.size() != other.size():
                warnings.warn("Both images should have same dimensions. "
                              "Returning None.")
                return None
            array = cv2.max(self._ndarray, other.get_ndarray())
            return Image(array, color_space=self._colorSpace)
        else:
            array = np.maximum(self._ndarray, other)
            return Image(array, color_space=self._colorSpace)

    def min(self, other):
        """
        **SUMMARY**

        The minimum value of my image, and the other image, in each channel
        If other is a number, returns the minimum of that and the number

        **Parameter**

        * *other* - Image of the same size or number

        **Returns**

        IMAGE
        """

        if isinstance(other, Image):
            if self.size() != other.size():
                warnings.warn("Both images should have same dimensions. "
                              "Returning None.")
                return None
            array = cv2.min(self._ndarray, other.get_ndarray())
            return Image(array, color_space=self._colorSpace)
        else:
            array = np.minimum(self._ndarray, other)
            return Image(array, color_space=self._colorSpace)

    def _clear_buffers(self, clearexcept="_bitmap"):
        for k, v in self._initialized_buffers.items():
            if k == clearexcept:
                continue
            self.__dict__[k] = v

    def find_barcode(self, do_zlib=True, zxing_path=""):
        """
        **SUMMARY**

        This function requires zbar and the zbar python wrapper
        to be installed or zxing and the zxing python library.

        **ZBAR**

        To install please visit:
        http://zbar.sourceforge.net/

        On Ubuntu Linux 12.04 or greater:
        sudo apt-get install python-zbar


        **ZXING**

        If you have the python-zxing library installed, you can find 2d and 1d
        barcodes in your image.  These are returned as Barcode feature objects
        in a FeatureSet.  The single parameter is the ZXing_path along with
        setting the do_zlib flag to False. You do not need the parameter if you
        don't have the ZXING_LIBRARY env parameter set.

        You can clone python-zxing at:

        http://github.com/oostendo/python-zxing

        **INSTALLING ZEBRA CROSSING**

        * Download the latest version of zebra crossing from:
         http://code.google.com/p/zxing/

        * unpack the zip file where ever you see fit

          >>> cd zxing-x.x, where x.x is the version number of zebra crossing
          >>> ant -f core/build.xml
          >>> ant -f javase/build.xml

          This should build the library, but double check the readme

        * Get our helper library

          >>> git clone git://github.com/oostendo/python-zxing.git
          >>> cd python-zxing
          >>> python setup.py install

        * Our library does not have a setup file. You will need to add
           it to your path variables. On OSX/Linux use a text editor to modify
           your shell file (e.g. .bashrc)

          export ZXING_LIBRARY=<FULL PATH OF ZXING LIBRARY - (i.e. step 2)>
          for example:

          export ZXING_LIBRARY=/my/install/path/zxing-x.x/

          On windows you will need to add these same variables to the system
          variable, e.g.

          http://www.computerhope.com/issues/ch000549.htm

        * On OSX/Linux source your shell rc file (e.g. source .bashrc). Windows
         users may need to restart.

        * Go grab some barcodes!

        .. Warning::
          Users on OSX may see the following error:

          RuntimeWarning: tmpnam is a potential security risk to your program

          We are working to resolve this issue. For normal use this should not
          be a problem.

        **Returns**

        A :py:class:`FeatureSet` of :py:class:`Barcode` objects. If no barcodes
         are detected the method returns None.

        **EXAMPLE**

        >>> bc = cam.getImage()
        >>> barcodes = img.findBarcodes()
        >>> for b in barcodes:
        >>>     b.draw()

        **SEE ALSO**

        :py:class:`FeatureSet`
        :py:class:`Barcode`

        """
        if do_zlib:
            try:
                import zbar
            except:
                logger.warning('The zbar library is not installed, please '
                               'install to read barcodes')
                return None

            #configure zbar
            scanner = zbar.ImageScanner()
            scanner.parse_config('enable')
            raw = self.get_pil().convert('L').tostring()
            width = self.width
            height = self.height

            # wrap image data
            image = zbar.Image(width, height, 'Y800', raw)

            # scan the image for barcodes
            scanner.scan(image)
            barcode = None
            # extract results
            for symbol in image:
                # do something useful with results
                barcode = symbol
            # clean up
            del image

        else:
            if not ZXING_ENABLED:
                warnings.warn("Zebra Crossing (ZXing) Library not installed. "
                              "Please see the release notes.")
                return None

            if not self._barcodeReader:
                if not zxing_path:
                    self._barcodeReader = zxing.BarCodeReader()
                else:
                    self._barcodeReader = zxing.BarCodeReader(zxing_path)

            tmp_filename = os.tmpnam() + ".png"
            self.save(tmp_filename)
            barcode = self._barcodeReader.decode(tmp_filename)
            os.unlink(tmp_filename)

        if barcode:
            f = Barcode(self, barcode)
            return FeatureSet([f])
        else:
            return None

    #this function contains two functions -- the basic edge detection algorithm
    #and then a function to break the lines down given a threshold parameter
    def find_lines(self, threshold=80, minlinelength=30, maxlinegap=10,
                   cannyth1=50, cannyth2=100, use_standard=False, nlines=-1,
                   maxpixelgap=1):
        """
        **SUMMARY**

        find_lines will find line segments in your image and returns line
        feature objects in a FeatureSet. This method uses the Hough
        (pronounced "HUFF") transform.

        See http://en.wikipedia.org/wiki/Hough_transform

        **PARAMETERS**

        * *threshold* - which determines the minimum "strength" of the line.
        * *minlinelength* - how many pixels long the line must be to be
         returned.
        * *maxlinegap* - how much gap is allowed between line segments to
         consider them the same line .
        * *cannyth1* - thresholds used in the edge detection step, refer to
         :py:meth:`_get_edge_map` for details.
        * *cannyth2* - thresholds used in the edge detection step, refer to
         :py:meth:`_get_edge_map` for details.
        * *use_standard* - use standard or probabilistic Hough transform.
        * *nlines* - maximum number of lines for return.
        * *maxpixelgap* - how much distance between pixels is allowed to
         consider them the same line.

        **RETURNS**

        Returns a :py:class:`FeatureSet` of :py:class:`Line` objects. If no
         lines are found the method returns None.

        **EXAMPLE**

        >>> img = Image("lenna")
        >>> lines = img.find_lines()
        >>> lines.draw()
        >>> img.show()

        **SEE ALSO**
        :py:class:`FeatureSet`
        :py:class:`Line`
        :py:meth:`edges`

        """
        em = self._get_edge_map(cannyth1, cannyth2)

        lines_fs = FeatureSet()
        if use_standard:
            lines = cv2.HoughLines(em, 1.0, math.pi/180.0, threshold,
                                   srn=minlinelength,
                                   stn=maxlinegap)[0]
            if nlines == -1:
                nlines = lines.shape[0]
            # All white points (edges) in Canny edge image
            y, x = np.where(em > 128)  #
            # Put points in dictionary for fast checkout if point is white
            pts = dict((p, 1) for p in zip(x, y))

            w, h = self.width - 1, self.height - 1
            for rho, theta in lines[:nlines]:
                ep = []
                ls = []
                a = math.cos(theta)
                b = math.sin(theta)
                # Find endpoints of line on the image's edges
                if round(b, 4) == 0:  # slope of the line is infinity
                    ep.append((int(round(abs(rho))), 0))
                    ep.append((int(round(abs(rho))), h))
                elif round(a, 4) == 0:  # slope of the line is zero
                    ep.append((0, int(round(abs(rho)))))
                    ep.append((w, int(round(abs(rho)))))
                else:
                    # top edge
                    x = rho / float(a)
                    if 0 <= x <= w:
                        ep.append((int(round(x)), 0))
                    # bottom edge
                    x = (rho - h * b) / float(a)
                    if 0 <= x <= w:
                        ep.append((int(round(x)), h))
                    # left edge
                    y = rho / float(b)
                    if 0 <= y <= h:
                        ep.append((0, int(round(y))))
                    # right edge
                    y = (rho - w * a) / float(b)
                    if 0 <= y <= h:
                        ep.append((w, int(round(y))))
                # remove duplicates if line crosses the image at corners
                ep = list(set(ep))
                ep.sort()
                brl = self.bresenham_line(ep[0], ep[1])

                # Follow the points on Bresenham's line. Look for white points.
                # If the distance between two adjacent white points (dist) is
                # less than or equal maxpixelgap then consider them the same
                # line. If dist is bigger maxpixelgap then check if length of
                # the line is bigger than minlinelength. If so then add line.

                # distance between two adjacent white points
                dist = float('inf')
                len_l = float('-inf')  # length of the line
                for p in brl:
                    if p in pts:
                        # found the end of the previous line and
                        # the start of the new line
                        if dist > maxpixelgap:
                            if len_l >= minlinelength:
                                if ls:
                                    # If the gap between current line and
                                    # previous is less than maxlinegap then
                                    # merge this lines
                                    l = ls[-1]
                                    gap = round(math.sqrt(
                                        (start_p[0] - l[1][0]) ** 2 +
                                        (start_p[1] - l[1][1]) ** 2))
                                    if gap <= maxlinegap:
                                        ls.pop()
                                        start_p = l[0]
                                ls.append((start_p, last_p))
                            # First white point of the new line found
                            dist = 1
                            len_l = 1
                            start_p = p  # first endpoint of the line
                        else:
                            # dist is less than or equal maxpixelgap,
                            # so line doesn't end yet
                            len_l += dist
                            dist = 1
                        last_p = p  # last white point
                    else:
                        dist += 1

                for l in ls:
                    lines_fs.append(Line(self, l))
            lines_fs = lines_fs[:nlines]
        else:
            lines = cv2.HoughLinesP(em, 1.0, math.pi/180.0, threshold,
                                    minLineLength=minlinelength,
                                    maxLineGap=maxlinegap)[0]
            if nlines == -1:
                nlines = lines.shape[0]

            for l in lines[:nlines]:
                lines_fs.append(Line(self, ((l[0], l[1]), (l[2], l[3]))))

        return lines_fs

    def find_chessboard(self, dimensions=(8, 5), subpixel=True):
        """
        **SUMMARY**

        Given an image, finds a chessboard within that image.  Returns the
        Chessboard featureset.
        The Chessboard is typically used for calibration because of its evenly
        spaced corners.


        The single parameter is the dimensions of the chessboard, typical one
        can be found in \SimpleCV\tools\CalibGrid.png

        **PARAMETERS**

        * *dimensions* - A tuple of the size of the chessboard in width and
         height in grid objects.
        * *subpixel* - Boolean if True use sub-pixel accuracy, otherwise use
         regular pixel accuracy.

        **RETURNS**

        A :py:class:`FeatureSet` of :py:class:`Chessboard` objects. If no
         chessboards are found None is returned.

        **EXAMPLE**

        >>> img = cam.getImage()
        >>> cb = img.find_chessboard()
        >>> cb.draw()

        **SEE ALSO**

        :py:class:`FeatureSet`
        :py:class:`Chessboard`

        """
        gray_array = self.get_gray_ndarray()
        equalized_grayscale_array = cv2.equalizeHist(gray_array)
        corners = cv2.findChessboardCorners(
            equalized_grayscale_array, dimensions,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
        if len(corners[1]) == dimensions[0] * dimensions[1]:
            if subpixel:
                sp_corners = cv2.cornerSubPix(
                    gray_array, corners[1], (11, 11), (-1, -1),
                    (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS,
                     10, 0.01))
            else:
                sp_corners = corners[1]
            return FeatureSet([Chessboard(self, dimensions, sp_corners)])
        else:
            return None

    def edges(self, t1=50, t2=100):
        """
        **SUMMARY**

        Finds an edge map Image using the Canny edge detection method. Edges
        will be brighter than the surrounding area.

        The t1 parameter is roughly the "strength" of the edge required, and
        the value between t1 and t2 is used for edge linking.

        For more information:

        * http://opencv.willowgarage.com/documentation/python/
        imgproc_feature_detection.html

        * http://en.wikipedia.org/wiki/Canny_edge_detector

        **PARAMETERS**

        * *t1* - Int - the lower Canny threshold.
        * *t2* - Int - the upper Canny threshold.

        **RETURNS**

        A SimpleCV image where the edges are white on a black background.

        **EXAMPLE**

        >>> cam = Camera()
        >>> while True:
        >>>    cam.getImage().edges().show()


        **SEE ALSO**

        :py:meth:`find_lines`

        """
        return Image(self._get_edge_map(t1, t2), color_space=self._colorSpace)

    def _get_edge_map(self, t1=50, t2=100):
        """
        Return the binary bitmap which shows where edges are in the image.
        The two parameters determine how much change in the image determines
        an edge, and how edges are linked together.  For more information
        refer to:

        http://en.wikipedia.org/wiki/Canny_edge_detector
        http://opencv.willowgarage.com/documentation/python/
        imgproc_feature_detection.html?highlight=canny#Canny
        """

        if self._edgeMap and self._cannyparam[0] == t1 \
                and self._cannyparam[1] == t2:
            return self._edgeMap

        self._edgeMap = cv2.Canny(self.get_gray_ndarray(), t1, t2)
        self._cannyparam = (t1, t2)

        return self._edgeMap

    def rotate(self, angle, fixed=True, point=None, scale=1.0):
        """
        **SUMMARY***

        This function rotates an image around a specific point by the given
        angle. By default in "fixed" mode, the returned Image is the same
        dimensions as the original Image, and the contents will be scaled to
        fit. In "full" mode the contents retain the original size, and the
        Image object will scale by default, the point is the center of the
        image. You can also specify a scaling parameter

        .. Note:
          that when fixed is set to false selecting a rotation point has no
          effect since the image is move to fit on the screen.

        **PARAMETERS**

        * *angle* - angle in degrees positive is clockwise, negative is counter
         clockwise
        * *fixed* - if fixed is true,keep the original image dimensions,
         otherwise scale the image to fit the rotation
        * *point* - the point about which we want to rotate, if none is
         defined we use the center.
        * *scale* - and optional floating point scale parameter.

        **RETURNS**

        The rotated SimpleCV image.

        **EXAMPLE**

        >>> img = Image('logo')
        >>> img2 = img.rotate(73.00, point=(img.width / 2, img.height / 2))
        >>> img3 = img.rotate(73.00,
            ...               fixed=False,
            ...               point=(img.width / 2, img.height / 2))
        >>> img4 = img2.side_by_side(img3)
        >>> img4.show()

        **SEE ALSO**

        :py:meth:`rotate90`

        """
        if point is None:
            point = [-1, -1]
        if point[0] == -1 or point[1] == -1:
            point[0] = (self.width - 1) / 2
            point[1] = (self.height - 1) / 2

        # first we create what we thing the rotation matrix should be
        rot_mat = cv2.getRotationMatrix2D((float(point[0]),
                                           float(point[1])),
                                          float(angle), float(scale))
        if fixed:
            array = cv2.warpAffine(self._ndarray, rot_mat, self.size())
            return Image(array, color_space=self._colorSpace)

        # otherwise, we're expanding the matrix to
        # fit the image at original size
        a1 = np.array([0, 0, 1])
        b1 = np.array([self.width, 0, 1])
        c1 = np.array([self.width, self.height, 1])
        d1 = np.array([0, self.height, 1])
        # So we have defined our image ABC in homogenous coordinates
        # and apply the rotation so we can figure out the image size
        a = np.dot(rot_mat, a1)
        b = np.dot(rot_mat, b1)
        c = np.dot(rot_mat, c1)
        d = np.dot(rot_mat, d1)
        # I am not sure about this but I think the a/b/c/d are transposed
        # now we calculate the extents of the rotated components.
        min_y = min(a[1], b[1], c[1], d[1])
        min_x = min(a[0], b[0], c[0], d[0])
        max_y = max(a[1], b[1], c[1], d[1])
        max_x = max(a[0], b[0], c[0], d[0])
        # from the extents we calculate the new size
        new_width = np.ceil(max_x - min_x)
        new_height = np.ceil(max_y - min_y)
        # now we calculate a new translation
        tx = 0
        ty = 0
        # calculate the translation that will get us centered in the new image
        if min_x < 0:
            tx = -1.0 * min_x
        elif max_x > new_width - 1:
            tx = -1.0 * (max_x - new_width)

        if min_y < 0:
            ty = -1.0 * min_y
        elif max_y > new_height - 1:
            ty = -1.0 * (max_y - new_height)

        # now we construct an affine map that will the rotation and scaling
        # we want with the the corners all lined up nicely
        # with the output image.
        src = ((a1[0], a1[1]), (b1[0], b1[1]), (c1[0], c1[1]))
        dst = ((a[0] + tx, a[1] + ty),
               (b[0] + tx, b[1] + ty),
               (c[0] + tx, c[1] + ty))

        # calculate the translation of the corners to center the image
        # use these new corner positions as the input to cvGetAffineTransform
        rot_mat = cv2.getAffineTransform(
            np.array(src).astype(np.float32),
            np.array(dst).astype(np.float32))
        array = cv2.warpAffine(self._ndarray, rot_mat,
                               (int(new_width), int(new_height)))
        return Image(array, color_space=self._colorSpace)

    def transpose(self):
        """
        **SUMMARY**

        Does a fast 90 degree rotation to the right with a flip.

        .. Warning::
          Subsequent calls to this function *WILL NOT* keep rotating it to the
          right!!!
          This function just does a matrix transpose so following one transpose
          by another will just yield the original image.

        **RETURNS**

        The rotated SimpleCV Image.

        **EXAMPLE**

        >>> img = Image("logo")
        >>> img2 = img.transpose()
        >>> img2.show()

        **SEE ALSO**

        :py:meth:`rotate`


        """
        array = cv2.transpose(self._ndarray)
        return Image(array, color_space=self._colorSpace)

    def shear(self, cornerpoints):
        """
        **SUMMARY**

        Given a set of new corner points in clockwise order, return a shear-ed
        image that transforms the image contents.  The returned image is the
        same dimensions.

        **PARAMETERS**

        * *cornerpoints* - a 2x4 tuple of points. The order is
         (top_left, top_right, bottom_left, bottom_right)

        **RETURNS**

        A simpleCV image.

        **EXAMPLE**

        >>> img = Image("lenna")
        >>> points = ((50, 0), (img.width + 50, 0),
            ...       (img.width, img.height), (0, img.height))
        >>> img.shear(points).show()

        **SEE ALSO**

        :py:meth:`transform_affine`
        :py:meth:`warp`
        :py:meth:`rotate`

        http://en.wikipedia.org/wiki/Transformation_matrix

        """
        src = ((0, 0), (self.width - 1, 0), (self.width - 1, self.height - 1))
        rot_matrix = cv2.getAffineTransform(
            np.array(src).astype(np.float32),
            np.array(cornerpoints).astype(np.float32))
        return self.transform_affine(rot_matrix)

    def transform_affine(self, rot_matrix):
        """
        **SUMMARY**

        This helper function for shear performs an affine rotation using the
        supplied matrix. The matrix can be a either an openCV mat or an
        np.ndarray type. The matrix should be a 2x3

        **PARAMETERS**

        * *rot_matrix* - A 2x3 numpy array or CvMat of the affine transform.

        **RETURNS**

        The rotated image. Note that the rotation is done in place, i.e.
        the image is not enlarged to fit the transofmation.

        **EXAMPLE**

        >>> img = Image("lenna")
        >>> points = ((50, 0), (img.width + 50, 0),
        ...           (img.width, img.height), (0, img.height))
        >>> src = ((0, 0), (img.width - 1, 0),
        ...        (img.width - 1, img.height - 1))
        >>> rot_matrix = cv2.getAffineTransform(src, points)
        >>> img.transform_affine(rot_matrix).show()

        **SEE ALSO**

        :py:meth:`shear`
        :py:meth`warp`
        :py:meth:`transform_perspective`
        :py:meth:`rotate`

        http://en.wikipedia.org/wiki/Transformation_matrix

        """
        array = cv2.warpAffine(self._ndarray, rot_matrix, self.size())
        return Image(array, color_space=self._colorSpace)

    def warp(self, cornerpoints):
        """
        **SUMMARY**

        This method performs and arbitrary perspective transform.
        Given a new set of corner points in clockwise order frin top left,
        return an Image with the images contents warped to the new coordinates.
        The returned image will be the same size as the original image


        **PARAMETERS**

        * *cornerpoints* - A list of four tuples corresponding to the
         destination corners in the order of
         (top_left,top_right,bottom_left,bottom_right)

        **RETURNS**

        A simpleCV Image with the warp applied. Note that this operation does
        not enlarge the image.

        **EXAMPLE**

        >>> img = Image("lenna")
        >>> points = ((30, 30), (img.width - 10, 70),
            ...       (img.width - 1 - 40, img.height - 1 + 30),
            ...       (20, img.height + 10))
        >>> img.warp(points).show()

        **SEE ALSO**

        :py:meth:`shear`
        :py:meth:`transform_affine`
        :py:meth:`transform_perspective`
        :py:meth:`rotate`

        http://en.wikipedia.org/wiki/Transformation_matrix

        """
        #original coordinates
        src = np.array(((0, 0), (self.width - 1, 0),
                        (self.width - 1, self.height - 1),
                        (0, self.height - 1))).astype(np.float32)
        # figure out the warp matrix
        p_wrap = cv2.getPerspectiveTransform(
            src, np.array(cornerpoints).astype(np.float32))
        return self.transform_perspective(p_wrap)

    def transform_perspective(self, rot_matrix):
        """
        **SUMMARY**

        This helper function for warp performs an affine rotation using the
        supplied matrix.
        The matrix can be a either an openCV mat or an np.ndarray type.
        The matrix should be a 3x3

       **PARAMETERS**
            * *rot_matrix* - Numpy Array or CvMat

        **RETURNS**

        The rotated image. Note that the rotation is done in place, i.e. the
        image is not enlarged to fit the transofmation.

        **EXAMPLE**

        >>> img = Image("lenna")
        >>> points = ((50,0), (img.width + 50, 0),
            ...       (img.width, img.height), (0, img.height))
        >>> src = ((30, 30), (img.width - 10, 70),
            ...    (img.width - 1 - 40, img.height - 1 + 30),
            ...    (20, img.height + 10))
        >>> result = cv2.getPerspectiveTransform(
        ...     np.array(src).astype(np.float32),
        ...     np.array(points).astype(np.float32))
        >>> img.transform_perspective(result).show()


        **SEE ALSO**

        :py:meth:`shear`
        :py:meth:`warp`
        :py:meth:`transform_perspective`
        :py:meth:`rotate`

        http://en.wikipedia.org/wiki/Transformation_matrix

        """
        array = cv2.warpPerspective(src=self._ndarray, dsize=self.size(),
                                    M=rot_matrix, flags=cv2.INTER_CUBIC)
        return Image(array, color_space=self._colorSpace)

    def get_pixel(self, x, y):
        """
        **SUMMARY**

        This function returns the RGB value for a particular image pixel given
        a specific row and column.

        .. Warning::
          this function will always return pixels in RGB format even if the
          image is BGR format.

        **PARAMETERS**

            * *x* - Int the x pixel coordinate.
            * *y* - Int the y pixel coordinate.

        **RETURNS**

        A color value that is a three element integer tuple.

        **EXAMPLE**

        >>> img = Image(logo)
        >>> color = img.get_pixel(10,10)


        .. Warning::
          We suggest that this method be used sparingly. For repeated pixel
          access use python array notation. I.e. img[x][y].

        """
        ret_val = None
        if x < 0 or x >= self.width:
            logger.warning("get_pixel: X value is not valid.")
        elif y < 0 or y >= self.height:
            logger.warning("get_pixel: Y value is not valid.")
        else:
            ret_val = self[x, y]
        return ret_val

    def get_gray_pixel(self, x, y):
        """
        **SUMMARY**

        This function returns the gray value for a particular image pixel given
         a specific row and column.

        .. Warning::
          This function will always return pixels in RGB format even if the
          image is BGR format.

        **PARAMETERS**

        * *x* - Int the x pixel coordinate.
        * *y* - Int the y pixel coordinate.

        **RETURNS**

        A gray value integer between 0 and 255.

        **EXAMPLE**

        >>> img = Image(logo)
        >>> color = img.get_gray_pixel(10,10)


        .. Warning::
          We suggest that this method be used sparingly. For repeated pixel
          access use python array notation. I.e. img[x][y].

        """
        ret_val = None
        if x < 0 or x >= self.width:
            logger.warning("get_gray_pixel: X value is not valid.")
        elif y < 0 or y >= self.height:
            logger.warning("get_gray_pixel: Y value is not valid.")
        else:
            ret_val = self.get_gray_ndarray()[x, y]
        return ret_val

    def get_vert_scanline(self, column):
        """
        **SUMMARY**

        This function returns a single column of RGB values from the image as
        a numpy array. This is handy if you want to crawl the image looking
        for an edge.

        **PARAMETERS**

        * *column* - the column number working from left=0 to right=img.width.

        **RETURNS**

        A numpy array of the pixel values. Ususally this is in BGR format.

        **EXAMPLE**

        >>> img = Image("lenna")
        >>> myColor = [0, 0, 0]
        >>> sl = img.get_vert_scanline(423)
        >>> sll = sl.tolist()
        >>> for p in sll:
        >>>    if p == myColor:
        >>>        # do something

        **SEE ALSO**

        :py:meth:`get_horz_scanline_gray`
        :py:meth:`get_horz_scanline`
        :py:meth:`get_vert_scanline_gray`
        :py:meth:`get_vert_scanline`

        """
        ret_val = None
        if column < 0 or column >= self.width:
            logger.warning("get_vert_scanline: column value is not valid.")
        else:
            ret_val = self._ndarray[:, column]
        return ret_val

    def get_horz_scanline(self, row):
        """
        **SUMMARY**

        This function returns a single row of RGB values from the image.
        This is handy if you want to crawl the image looking for an edge.

        **PARAMETERS**

        * *row* - the row number working from top=0 to bottom=img.height.

        **RETURNS**

        A a lumpy numpy array of the pixel values. Ususally this is in BGR
        format.

        **EXAMPLE**

        >>> img = Image("lenna")
        >>> myColor = [0,0,0]
        >>> sl = img.get_horz_scanline(422)
        >>> sll = sl.tolist()
        >>> for p in sll:
        >>>    if p == myColor:
        >>>        # do something

        **SEE ALSO**

        :py:meth:`get_horz_scanline_gray`
        :py:meth:`get_vert_scanline_gray`
        :py:meth:`get_vert_scanline`

        """
        ret_val = None
        if row < 0 or row >= self.height:
            logger.warning("get_horz_scanline: row value is not valid.")
        else:
            ret_val = self._ndarray[row]
        return ret_val

    def get_vert_scanline_gray(self, column):
        """
        **SUMMARY**

        This function returns a single column of gray values from the image as
        a numpy array. This is handy if you want to crawl the image looking
        for an edge.

        **PARAMETERS**

        * *column* - the column number working from left=0 to right=img.width.

        **RETURNS**

        A a lumpy numpy array of the pixel values.

        **EXAMPLE**

        >>> img = Image("lenna")
        >>> myColor = [255]
        >>> sl = img.get_vert_scanline_gray(421)
        >>> sll = sl.tolist()
        >>> for p in sll:
        >>>    if p == myColor:
        >>>        # do something

        **SEE ALSO**

        :py:meth:`get_horz_scanline_gray`
        :py:meth:`get_horz_scanline`
        :py:meth:`get_vert_scanline`

        """
        ret_val = None
        if column < 0 or column >= self.width:
            logger.warning("getHorzRGBScanline: row value is not valid.")
        else:
            ret_val = self.get_gray_ndarray()[:, column]
        return ret_val

    def get_horz_scanline_gray(self, row):
        """
        **SUMMARY**

        This function returns a single row of gray values from the image as
        a numpy array. This is handy if you want to crawl the image looking
        for an edge.

        **PARAMETERS**

        * *row* - the row number working from top=0 to bottom=img.height.

        **RETURNS**

        A a lumpy numpy array of the pixel values.

        **EXAMPLE**

        >>> img = Image("lenna")
        >>> myColor = [255]
        >>> sl = img.get_horz_scanline_gray(420)
        >>> sll = sl.tolist()
        >>> for p in sll:
        >>>    if p == myColor:
        >>>        # do something

        **SEE ALSO**

        :py:meth:`get_horz_scanline_gray`
        :py:meth:`get_horz_scanline`
        :py:meth:`get_vert_scanline_gray`
        :py:meth:`get_vert_scanline`

        """
        ret_val = None
        if row < 0 or row >= self.height:
            logger.warning("get_horz_scanline_gray: row value is not valid.")
        else:
            ret_val = self.get_gray_ndarray()[row]
        return ret_val

    @staticmethod
    def roi_to_slice(roi):
        x, y, w, h = roi
        return slice(y, y + h), slice(x, x + w)

    def crop(self, x, y=None, w=None, h=None, centered=False, smart=False):
        """

        **SUMMARY**

        Consider you want to crop a image with the following dimension::

            (x,y)
            +--------------+
            |              |
            |              |h
            |              |
            +--------------+
                  w      (x1,y1)


        Crop attempts to use the x and y position variables and the w and h
        width and height variables to crop the image. When centered is false,
        x and y define the top and left of the cropped rectangle. When centered
        is true the function uses x and y as the centroid of the cropped
        region.

        You can also pass a feature into crop and have it automatically return
        the cropped image within the bounding outside area of that feature

        Or parameters can be in the form of a
         - tuple or list : (x,y,w,h) or [x,y,w,h]
         - two points : (x,y),(x1,y1) or [(x,y),(x1,y1)]

        **PARAMETERS**

        * *x* - An integer or feature.
              - If it is a feature we crop to the features dimensions.
              - This can be either the top left corner of the image or the
                center cooridnate of the the crop region.
              - or in the form of tuple/list. i,e (x,y,w,h) or [x,y,w,h]
              - Otherwise in two point form. i,e [(x,y),(x1,y1)] or (x,y)
        * *y* - The y coordinate of the center, or top left corner  of the
                crop region.
              - Otherwise in two point form. i,e (x1,y1)
        * *w* - Int - the width of the cropped region in pixels.
        * *h* - Int - the height of the cropped region in pixels.
        * *centered*  - Boolean - if True we treat the crop region as being
          the center coordinate and a width and height. If false we treat it as
          the top left corner of the crop region.
        * *smart* - Will make sure you don't try and crop outside the image
         size, so if your image is 100x100 and you tried a crop like
         img.crop(50,50,100,100), it will autoscale the crop to the max width.


        **RETURNS**

        A SimpleCV Image cropped to the specified width and height.

        **EXAMPLE**

        >>> img = Image('lenna')
        >>> img.crop(50, 40, 128, 128).show()
        >>> img.crop((50, 40, 128, 128)).show() #roi
        >>> img.crop([50, 40, 128, 128]) #roi
        >>> img.crop((50, 40), (178, 168)) # two point form
        >>> img.crop([(50, 40),(178, 168)]) # two point form
        >>> # list of x's and y's
        >>> img.crop([x1, x2, x3, x4, x5], [y1, y1, y3, y4, y5])
        >>> img.crop([(x, y), (x, y), (x, y), (x, y), (x, y)] # list of (x,y)
        >>> img.crop(x, y, 100, 100, smart=True)
        **SEE ALSO**

        :py:meth:`embiggen`
        :py:meth:`region_select`
        """

        if smart:
            if x > self.width:
                x = self.width
            elif x < 0:
                x = 0
            elif y > self.height:
                y = self.height
            elif y < 0:
                y = 0
            elif (x + w) > self.width:
                w = self.width - x
            elif (y + h) > self.height:
                h = self.height - y

        if isinstance(x, np.ndarray):
            x = x.tolist()
        if isinstance(y, np.ndarray):
            y = y.tolist()

        #If it's a feature extract what we need
        if isinstance(x, Feature):
            feature = x
            x = feature.points[0][0]
            y = feature.points[0][1]
            w = feature.get_width()
            h = feature.get_height()

        elif isinstance(x, (tuple, list)) and len(x) == 4 \
                and isinstance(x[0], (int, long, float)) \
                and y is None and w is None and h is None:
            x, y, w, h = x
        # x of the form [(x,y),(x1,y1),(x2,y2),(x3,y3)]
        # x of the form [[x,y],[x1,y1],[x2,y2],[x3,y3]]
        # x of the form ([x,y],[x1,y1],[x2,y2],[x3,y3])
        # x of the form ((x,y),(x1,y1),(x2,y2),(x3,y3))
        # x of the form (x,y,x1,y2) or [x,y,x1,y2]
        elif isinstance(x, (list, tuple)) \
                and isinstance(x[0], (list, tuple)) \
                and (len(x) == 4 and len(x[0]) == 2) \
                and y is None and w is None and h is None:
            if len(x[0]) == 2 and len(x[1]) == 2 \
                    and len(x[2]) == 2 and len(x[3]) == 2:
                xmax = np.max([x[0][0], x[1][0], x[2][0], x[3][0]])
                ymax = np.max([x[0][1], x[1][1], x[2][1], x[3][1]])
                xmin = np.min([x[0][0], x[1][0], x[2][0], x[3][0]])
                ymin = np.min([x[0][1], x[1][1], x[2][1], x[3][1]])
                x = xmin
                y = ymin
                w = xmax - xmin
                h = ymax - ymin
            else:
                logger.warning("x should be in the form  "
                               "((x,y),(x1,y1),(x2,y2),(x3,y3))")
                return None

        # x,y of the form [x1,x2,x3,x4,x5....] and y similar
        elif isinstance(x, (tuple, list)) \
                and isinstance(y, (tuple, list)) \
                and len(x) > 4 and len(y) > 4:
            if isinstance(x[0], (int, long, float)) \
                    and isinstance(y[0], (int, long, float)):
                xmax = np.max(x)
                ymax = np.max(y)
                xmin = np.min(x)
                ymin = np.min(y)
                x = xmin
                y = ymin
                w = xmax - xmin
                h = ymax - ymin
            else:
                logger.warning("x should be in the form "
                               "x = [1, 2, 3, 4, 5] y = [0, 2, 4, 6, 8]")
                return None

        # x of the form [(x,y),(x,y),(x,y),(x,y),(x,y),(x,y)]
        elif isinstance(x, (list, tuple)) and len(x) > 4 \
                and len(x[0]) == 2 and y is None and w is None and h is None:
            if isinstance(x[0][0], (int, long, float)):
                xs = [pt[0] for pt in x]
                ys = [pt[1] for pt in x]
                xmax = np.max(xs)
                ymax = np.max(ys)
                xmin = np.min(xs)
                ymin = np.min(ys)
                x = xmin
                y = ymin
                w = xmax - xmin
                h = ymax - ymin
            else:
                logger.warning("x should be in the form "
                               "[(x,y),(x,y),(x,y),(x,y),(x,y),(x,y)]")
                return None

        # x of the form [(x,y),(x1,y1)]
        elif isinstance(x, (list, tuple)) and len(x) == 2 \
                and isinstance(x[0], (list, tuple)) \
                and isinstance(x[1], (list, tuple)) \
                and y is None and w is None and h is None:
            if len(x[0]) == 2 and len(x[1]) == 2:
                xt = np.min([x[0][0], x[1][0]])
                yt = np.min([x[0][0], x[1][0]])
                w = np.abs(x[0][0] - x[1][0])
                h = np.abs(x[0][1] - x[1][1])
                x = xt
                y = yt
            else:
                logger.warning("x should be in the form [(x1,y1),(x2,y2)]")
                return None

        # x and y of the form (x,y),(x1,y2)
        elif isinstance(x, (tuple, list)) and isinstance(y, (tuple, list)) \
                and w is None and h is None:
            if len(x) == 2 and len(y) == 2:
                xt = np.min([x[0], y[0]])
                yt = np.min([x[1], y[1]])
                w = np.abs(y[0] - x[0])
                h = np.abs(y[1] - x[1])
                x = xt
                y = yt

            else:
                logger.warning("if x and y are tuple it should be in the form "
                               "(x1,y1) and (x2,y2)")
                return None

        if y is None or w is None or h is None:
            print "Please provide an x, y, width, height to function"

        if w <= 0 or h <= 0:
            logger.warning("Can't do a negative crop!")
            return None

        if x < 0 or y < 0:
            logger.warning("Crop will try to help you, but you have a "
                           "negative crop position, your width and height "
                           "may not be what you want them to be.")

        if centered:
            rectangle = (int(x - (w / 2)), int(y - (h / 2)), int(w), int(h))
        else:
            rectangle = (int(x), int(y), int(w), int(h))

        (top_roi, bottom_roi) = self._rect_overlap_rois(
            (rectangle[2], rectangle[3]), (self.width, self.height),
            (rectangle[0], rectangle[1]))

        if bottom_roi is None:
            logger.warning("Hi, your crop rectangle doesn't even overlap your "
                           "image. I have no choice but to return None.")
            return None

        array = self._ndarray[
            bottom_roi[1]:bottom_roi[1] + bottom_roi[3],
            bottom_roi[0]:bottom_roi[0] + bottom_roi[2]]

        img = Image(array, color_space=self._colorSpace)

        #Buffering the top left point (x, y) in a image.
        img._uncroppedX = self._uncroppedX + int(x)
        img._uncroppedY = self._uncroppedY + int(y)
        return img

    def region_select(self, x1, y1, x2, y2):
        """
        **SUMMARY**

        Region select is similar to crop, but instead of taking a position and
        width and height values it simply takes two points on the image and
        returns the selected region. This is very helpful for creating
        interactive scripts that require the user to select a region.

        **PARAMETERS**

        * *x1* - Int - Point one x coordinate.
        * *y1* - Int  - Point one y coordinate.
        * *x2* - Int  - Point two x coordinate.
        * *y2* - Int  - Point two y coordinate.

        **RETURNS**

        A cropped SimpleCV Image.

        **EXAMPLE**

        >>> img = Image("lenna")
        >>> # often this comes from a mouse click
        >>> subreg = img.region_select(10, 10, 100, 100)
        >>> subreg.show()

        **SEE ALSO**

        :py:meth:`crop`

        """
        w = abs(x1 - x2)
        h = abs(y1 - y2)

        ret_val = None
        if w <= 0 or h <= 0 or w > self.width or h > self.height:
            logger.warning("region_select: the given values will not fit in "
                           "the image or are too small.")
        else:
            xf = x2
            if x1 < x2:
                xf = x1
            yf = y2
            if y1 < y2:
                yf = y1
            ret_val = self.crop(xf, yf, w, h)

        return ret_val

    def clear(self):
        """
        **SUMMARY**

        This is a slightly unsafe method that clears out the entire image state
        it is usually used in conjunction with the drawing blobs to fill in
        draw a single large blob in the image.

        .. Warning:
          Do not use this method unless you have a particularly compelling
          reason.

        """
        if self.is_gray():
            self._ndarray = np.zeros(self.size(), dtype=self.dtype)
        else:
            self._ndarray = np.zeros((self.width, self.height, 3),
                                     dtype=self.dtype)
        self._clear_buffers()

    def draw(self, features, color=Color.GREEN, width=1, autocolor=False):
        """
        **SUMMARY**

        This is a method to draw Features on any given image.

        **PARAMETERS**

        * *features* - FeatureSet or any Feature
         (eg. Line, Circle, Corner, etc)
        * *color*    - Color of the Feature to be drawn
        * *width*    - width of the Feature to be drawn
        * *autocolor*- If true a color is randomly selected for each feature

        **RETURNS**
        None

        **EXAMPLE**

        img = Image("lenna")
        lines = img.equalize().find_lines()
        img.draw(lines)
        img.show()
        """
        if isinstance(features, Image):
            warnings.warn("You need to pass drawable features.")
            return
        if hasattr(features, 'draw'):
            from copy import deepcopy

            if isinstance(features, FeatureSet):
                cfeatures = deepcopy(features)
                for cfeat in cfeatures:
                    cfeat.image = self
                cfeatures.draw(color, width, autocolor)
            else:
                cfeatures = deepcopy(features)
                cfeatures.image = self
                cfeatures.draw(color, width)
        else:
            warnings.warn("You need to pass drawable features.")

    def draw_text(self, text="", x=None, y=None, color=Color.BLUE,
                  fontsize=16):
        """
        **SUMMARY**

        This function draws the string that is passed on the screen at the
        specified coordinates.

        The Default Color is blue but you can pass it various colors

        The text will default to the center of the screen if you don't pass
        it a value


        **PARAMETERS**

        * *text* - String - the text you want to write. ASCII only please.
        * *x* - Int - the x position in pixels.
        * *y* - Int - the y position in pixels.
        * *color* - Color object or Color Tuple
        * *fontsize* - Int - the font size - roughly in points.

        **RETURNS**

        Nothing. This is an in place function. Text is added to the Images
        drawing layer.

        **EXAMPLE**

        >>> img = Image("lenna")
        >>> img.draw_text("xamox smells like cool ranch doritos.",
            ...          50, 50, color=Color.BLACK, fontsize=48)
        >>> img.show()

        **SEE ALSO**

        :py:meth:`dl`
        :py:meth:`draw_circle`
        :py:meth:`draw_rectangle`

        """
        if x is None:
            x = self.width / 2
        if y is None:
            y = self.height / 2

        self.get_drawing_layer().set_font_size(fontsize)
        self.get_drawing_layer().text(text, (x, y), color)

    def draw_rectangle(self, x, y, w, h, color=Color.RED, width=1, alpha=255):
        """
        **SUMMARY**

        Draw a rectangle on the screen given the upper left corner of the
        rectangle and the width and height.

        **PARAMETERS**

        * *x* - the x position.
        * *y* - the y position.
        * *w* - the width of the rectangle.
        * *h* - the height of the rectangle.
        * *color* - an RGB tuple indicating the desired color.
        * *width* - the width of the rectangle, a value less than or equal to
         zero means filled in completely.
        * *alpha* - the alpha value on the interval from 255 to 0, 255 is
         opaque, 0 is completely transparent.

        **RETURNS**

        None - this operation is in place and adds the rectangle to the drawing
        layer.

        **EXAMPLE**

        >>> img = Image("lenna")
        >>> img.draw_rectange(50, 50, 100, 123)
        >>> img.show()

        **SEE ALSO**

        :py:meth:`dl`
        :py:meth:`draw_circle`
        :py:meth:`draw_rectangle`
        :py:meth:`apply_layers`
        :py:class:`DrawingLayer`

        """
        if width < 1:
            self.get_drawing_layer().rectangle((x, y), (w, h), color,
                                               filled=True, alpha=alpha)
        else:
            self.get_drawing_layer().rectangle((x, y), (w, h), color, width,
                                               alpha=alpha)

    def draw_rotated_rectangle(self, boundingbox, color=Color.RED, width=1):
        """
        **SUMMARY**

        Draw the minimum bouding rectangle. This rectangle is a series of four
        points.

        **TODO**

        **KAT FIX THIS**
        """
        raise Exception('not implemented')
        # cv2.ellipse(self._ndarray, box=boundingbox, color=color,
        #             thicness=width)

    def show(self, type='window'):
        """
        **SUMMARY**

        This function automatically pops up a window and shows the current
        image.

        **PARAMETERS**

        * *type* - this string can have one of two values, either 'window', or
         'browser'. Window opens a display window, while browser opens the
         default web browser to show an image.

        **RETURNS**

        This method returns the display object. In the case of the window this
        is a JpegStreamer object. In the case of a window a display
        object is returned.

        **EXAMPLE**

        >>> img = Image("lenna")
        >>> img.show()
        >>> img.show('browser')

        **SEE ALSO**

        :py:class:`JpegStreamer`
        :py:class:`Display`

        """

        if type == 'browser':
            import webbrowser

            js = JpegStreamer(8080)
            self.save(js)
            webbrowser.open("http://localhost:8080", 2)
            return js
        elif type == 'window':
            from simplecv.display import Display

            if init_options_handler.on_notebook:
                d = Display(displaytype='notebook')
            else:
                d = Display(self.size())
            self.save(d)
            return d
        else:
            print "Unknown type to show"

    def _surface_to_image(self, surface):
        imgarray = pg.surfarray.array3d(surface)
        ret_val = Image(imgarray)
        ret_val._colorSpace = ColorSpace.RGB
        return ret_val.to_bgr().transpose()

    def _image_to_surface(self, img):
        return pg.image.fromstring(img.get_pil().tostring(), img.size(), "RGB")
        #return pg.surfarray.make_surface(img.to_rgb().get_numpy())

    def to_pygame_surface(self):
        """
        **SUMMARY**

        Converts this image to a pygame surface. This is useful if you want
        to treat an image as a sprite to render onto an image. An example
        would be rendering blobs on to an image.

        .. Warning::
          *THIS IS EXPERIMENTAL*. We are plannng to remove this functionality
          sometime in the near future.

        **RETURNS**

        The image as a pygame surface.

        **SEE ALSO**


        :py:class:`DrawingLayer`
        :py:meth:`insert_drawing_layer`
        :py:meth:`add_drawing_layer`
        :py:meth:`dl`
        :py:meth:`to_pygame_surface`
        :py:meth:`get_drawing_layer`
        :py:meth:`remove_drawing_layer`
        :py:meth:`clear_layers`
        :py:meth:`layers`
        :py:meth:`merged_layers`
        :py:meth:`apply_layers`
        :py:meth:`draw_text`
        :py:meth:`draw_rectangle`
        :py:meth:`draw_circle`
        :py:meth:`blit`

        """
        return pg.image.fromstring(self.get_pil().tostring(), self.size(),
                                   "RGB")

    def add_drawing_layer(self, layer=None):
        """
        **SUMMARY**

        Push a new drawing layer onto the back of the layer stack

        **PARAMETERS**

        * *layer* - The new drawing layer to add.

        **RETURNS**

        The index of the new layer as an integer.

        **EXAMPLE**

        >>> img = Image("Lenna")
        >>> myLayer = DrawingLayer((img.width,img.height))
        >>> img.add_drawing_layer(myLayer)

        **SEE ALSO**

        :py:class:`DrawingLayer`
        :py:meth:`insertDrawinglayer`
        :py:meth:`addDrawinglayer`
        :py:meth:`dl`
        :py:meth:`to_pygame_surface`
        :py:meth:`get_drawing_layer`
        :py:meth:`remove_drawing_layer`
        :py:meth:`clear_layers`
        :py:meth:`layers`
        :py:meth:`merged_layers`
        :py:meth:`apply_layers`
        :py:meth:`draw_text`
        :py:meth:`draw_rectangle`
        :py:meth:`draw_circle`
        :py:meth:`blit`

        """

        if not isinstance(layer, DrawingLayer):
            return "Please pass a DrawingLayer object"

        if not layer:
            layer = DrawingLayer(self.size())
        self._mLayers.append(layer)
        return len(self._mLayers) - 1

    def insert_drawing_layer(self, layer, index):
        """
        **SUMMARY**

        Insert a new layer into the layer stack at the specified index.

        **PARAMETERS**

        * *layer* - A drawing layer with crap you want to draw.
        * *index* - The index at which to insert the layer.

        **RETURNS**

        None - that's right - nothing.

        **EXAMPLE**

        >>> img = Image("Lenna")
        >>> myLayer1 = DrawingLayer((img.width, img.height))
        >>> myLayer2 = DrawingLayer((img.width, img.height))
        >>> #Draw on the layers
        >>> img.insert_drawing_layer(myLayer1, 1) # on top
        >>> img.insert_drawing_layer(myLayer2, 2) # on the bottom


        **SEE ALSO**

        :py:class:`DrawingLayer`
        :py:meth:`addDrawinglayer`
        :py:meth:`dl`
        :py:meth:`to_pygame_surface`
        :py:meth:`get_drawing_layer`
        :py:meth:`remove_drawing_layer`
        :py:meth:`clear_layers`
        :py:meth:`layers`
        :py:meth:`merged_layers`
        :py:meth:`apply_layers`
        :py:meth:`draw_text`
        :py:meth:`draw_rectangle`
        :py:meth:`draw_circle`
        :py:meth:`blit`

        """
        self._mLayers.insert(index, layer)

    def remove_drawing_layer(self, index=-1):
        """
        **SUMMARY**

        Remove a layer from the layer stack based on the layer's index.

        **PARAMETERS**

        * *index* - Int - the index of the layer to remove.

        **RETURNS**

        This method returns the removed drawing layer.

        **EXAMPLES**

        >>> img = Image("Lenna")
        >>> img.remove_drawing_layer(1)  # removes the layer with index = 1
        >>> # if no index is specified it removes the top layer
        >>> img.remove_drawing_layer()


        **SEE ALSO**

        :py:class:`DrawingLayer`
        :py:meth:`addDrawinglayer`
        :py:meth:`dl`
        :py:meth:`to_pygame_surface`
        :py:meth:`get_drawing_layer`
        :py:meth:`remove_drawing_layer`
        :py:meth:`clear_layers`
        :py:meth:`layers`
        :py:meth:`merged_layers`
        :py:meth:`apply_layers`
        :py:meth:`draw_text`
        :py:meth:`draw_rectangle`
        :py:meth:`draw_circle`
        :py:meth:`blit`

        """
        try:
            return self._mLayers.pop(index)
        except IndexError:
            print 'Not a valid index or No layers to remove!'

    def get_drawing_layer(self, index=-1):
        """
        **SUMMARY**

        Return a drawing layer based on the provided index.  If not provided,
        will default to the top layer.  If no layers exist, one will be created

        **PARAMETERS**

        * *index* - returns the drawing layer at the specified index.

        **RETURNS**

        A drawing layer.

        **EXAMPLE**

        >>> img = Image("Lenna")
        >>> myLayer1 = DrawingLayer((img.width,img.height))
        >>> myLayer2 = DrawingLayer((img.width,img.height))
        >>> #Draw on the layers
        >>> img.insert_drawing_layer(myLayer1,1) # on top
        >>> img.insert_drawing_layer(myLayer2,2) # on the bottom
        >>> layer2 =img.get_drawing_layer(2)

        **SEE ALSO**

        :py:class:`DrawingLayer`
        :py:meth:`addDrawinglayer`
        :py:meth:`dl`
        :py:meth:`to_pygame_surface`
        :py:meth:`get_drawing_layer`
        :py:meth:`remove_drawing_layer`
        :py:meth:`clear_layers`
        :py:meth:`layers`
        :py:meth:`merged_layers`
        :py:meth:`apply_layers`
        :py:meth:`draw_text`
        :py:meth:`draw_rectangle`
        :py:meth:`draw_circle`
        :py:meth:`blit`

        """
        if not len(self._mLayers):
            layer = DrawingLayer(self.size())
            self.add_drawing_layer(layer)
        try:
            return self._mLayers[index]
        except IndexError:
            print 'Not a valid index'

    def dl(self, index=-1):
        """
        **SUMMARY**

        Alias for :py:meth:`get_drawing_layer`

        """
        return self.get_drawing_layer(index)

    def clear_layers(self):
        """
        **SUMMARY**

        Remove all of the drawing layers.

        **RETURNS**

        None.

        **EXAMPLE**

        >>> img = Image("Lenna")
        >>> myLayer1 = DrawingLayer((img.width,img.height))
        >>> myLayer2 = DrawingLayer((img.width,img.height))
        >>> img.insert_drawing_layer(myLayer1,1) # on top
        >>> img.insert_drawing_layer(myLayer2,2) # on the bottom
        >>> img.clear_layers()

        **SEE ALSO**

        :py:class:`DrawingLayer`
        :py:meth:`dl`
        :py:meth:`to_pygame_surface`
        :py:meth:`get_drawing_layer`
        :py:meth:`remove_drawing_layer`
        :py:meth:`layers`
        :py:meth:`merged_layers`
        :py:meth:`apply_layers`
        :py:meth:`draw_text`
        :py:meth:`draw_rectangle`
        :py:meth:`draw_circle`
        :py:meth:`blit`

        """
        for i in self._mLayers:
            self._mLayers.remove(i)

        return None

    def layers(self):
        """
        **SUMMARY**

        Return the array of DrawingLayer objects associated with the image.

        **RETURNS**

        A list of of drawing layers.

        **SEE ALSO**

        :py:class:`DrawingLayer`
        :py:meth:`add_drawing_layer`
        :py:meth:`dl`
        :py:meth:`to_pygame_surface`
        :py:meth:`get_drawing_layer`
        :py:meth:`remove_drawing_layer`
        :py:meth:`merged_layers`
        :py:meth:`apply_layers`
        :py:meth:`draw_text`
        :py:meth:`draw_rectangle`
        :py:meth:`draw_circle`
        :py:meth:`blit`

        """
        return self._mLayers

        #render the image.

    def _render_image(self, layer):
        img_surf = self.get_pg_surface().copy()
        img_surf.blit(layer.surface, (0, 0))
        return Image(img_surf)

    def merged_layers(self):
        """
        **SUMMARY**

        Return all DrawingLayer objects as a single DrawingLayer.

        **RETURNS**

        Returns a drawing layer with all of the drawing layers of this image
        merged into one.

        **EXAMPLE**

        >>> img = Image("Lenna")
        >>> myLayer1 = DrawingLayer((img.width,img.height))
        >>> myLayer2 = DrawingLayer((img.width,img.height))
        >>> img.insert_drawing_layer(myLayer1,1) # on top
        >>> img.insert_drawing_layer(myLayer2,2) # on the bottom
        >>> derp = img.merged_layers()

        **SEE ALSO**

        :py:class:`DrawingLayer`
        :py:meth:`add_drawing_layer`
        :py:meth:`dl`
        :py:meth:`to_pygame_surface`
        :py:meth:`get_drawing_layer`
        :py:meth:`remove_drawing_layer`
        :py:meth:`layers`
        :py:meth:`apply_layers`
        :py:meth:`draw_text`
        :py:meth:`draw_rectangle`
        :py:meth:`draw_circle`
        :py:meth:`blit`

        """
        final = DrawingLayer(self.size())
        for layers in self._mLayers:  # compose all the layers
            layers.render_to_other_layer(final)
        return final

    def apply_layers(self, indicies=-1):
        """
        **SUMMARY**

        Render all of the layers onto the current image and return the result.
        Indicies can be a list of integers specifying the layers to be used.

        **PARAMETERS**

        * *indicies* -  Indicies can be a list of integers specifying the
         layers to be used.

        **RETURNS**

        The image after applying the drawing layers.

        **EXAMPLE**

        >>> img = Image("Lenna")
        >>> myLayer1 = DrawingLayer((img.width,img.height))
        >>> myLayer2 = DrawingLayer((img.width,img.height))
        >>> #Draw some stuff
        >>> img.insert_drawing_layer(myLayer1,1) # on top
        >>> img.insert_drawing_layer(myLayer2,2) # on the bottom
        >>> derp = img.apply_layers()

        **SEE ALSO**

        :py:class:`DrawingLayer`
        :py:meth:`dl`
        :py:meth:`to_pygame_surface`
        :py:meth:`get_drawing_layer`
        :py:meth:`remove_drawing_layer`
        :py:meth:`layers`
        :py:meth:`draw_text`
        :py:meth:`draw_rectangle`
        :py:meth:`draw_circle`
        :py:meth:`blit`

        """
        if not len(self._mLayers):
            return self

        if indicies == -1 and len(self._mLayers) > 0:
            final = self.merged_layers()
            img_surf = self.get_pg_surface().copy()
            img_surf.blit(final.surface, (0, 0))
            return Image(img_surf)
        else:
            final = DrawingLayer((self.width, self.height))
            ret_val = self
            indicies.reverse()
            for idx in indicies:
                ret_val = self._mLayers[idx].render_to_other_layer(final)
            img_surf = self.get_pg_surface().copy()
            img_surf.blit(final.surface, (0, 0))
            indicies.reverse()
            return Image(img_surf)

    def adaptive_scale(self, resolution, fit=True):
        """
        **SUMMARY**

        Adapative Scale is used in the Display to automatically
        adjust image size to match the display size. This method attempts to
        scale an image to the desired resolution while keeping the aspect ratio
        the same. If fit is False we simply crop and center the image to the
        resolution. In general this method should look a lot better than
        arbitrary cropping and scaling.

        **PARAMETERS**

        * *resolution* - The size of the returned image as a (width,height)
         tuple.
        * *fit* - If fit is true we try to fit the image while maintaining the
         aspect ratio. If fit is False we crop and center the image to fit the
         resolution.

        **RETURNS**

        A SimpleCV Image.

        **EXAMPLE**

        This is typically used in this instance:

        >>> d = Display((800, 600))
        >>> i = Image((640, 480))
        >>> i.save(d)

        Where this would scale the image to match the display size of 800x600

        """

        wndw_ar = float(resolution[0]) / float(resolution[1])
        img_ar = float(self.width) / float(self.height)
        img = self
        targetx = 0
        targety = 0
        targetw = resolution[0]
        targeth = resolution[1]
        if self.size() == resolution:  # we have to resize
            ret_val = self
        elif img_ar == wndw_ar and fit:
            ret_val = img.scale(resolution[0], resolution[1])
            return ret_val
        elif fit:
            #scale factors
            ret_val = np.zeros((resolution[1], resolution[0], 3),
                               dtype='uint8')
            wscale = (float(self.width) / float(resolution[0]))
            hscale = (float(self.height) / float(resolution[1]))
            if wscale > 1:  # we're shrinking what is the percent reduction
                wscale = 1 - (1.0 / wscale)
            else:  # we need to grow the image by a percentage
                wscale = 1.0 - wscale
            if hscale > 1:
                hscale = 1 - (1.0 / hscale)
            else:
                hscale = 1.0 - hscale
            if wscale == 0:  # if we can get away with not scaling do that
                targetx = 0
                targety = (resolution[1] - self.height) / 2
                targetw = img.width
                targeth = img.height
            elif hscale == 0:  # if we can get away with not scaling do that
                targetx = (resolution[0] - img.width) / 2
                targety = 0
                targetw = img.width
                targeth = img.height
            elif wscale < hscale:  # the width has less distortion
                sfactor = float(resolution[0]) / float(self.width)
                targetw = int(float(self.width) * sfactor)
                targeth = int(float(self.height) * sfactor)
                if targetw > resolution[0] or targeth > resolution[1]:
                    #aw shucks that still didn't work do the other way instead
                    sfactor = float(resolution[1]) / float(self.height)
                    targetw = int(float(self.width) * sfactor)
                    targeth = int(float(self.height) * sfactor)
                    targetx = (resolution[0] - targetw) / 2
                    targety = 0
                else:
                    targetx = 0
                    targety = (resolution[1] - targeth) / 2
                img = img.scale(targetw, targeth)
            else:  # the height has more distortion
                sfactor = float(resolution[1]) / float(self.height)
                targetw = int(float(self.width) * sfactor)
                targeth = int(float(self.height) * sfactor)
                if targetw > resolution[0] or targeth > resolution[1]:
                    # aw shucks that still didn't work do the other way instead
                    sfactor = float(resolution[0]) / float(self.width)
                    targetw = int(float(self.width) * sfactor)
                    targeth = int(float(self.height) * sfactor)
                    targetx = 0
                    targety = (resolution[1] - targeth) / 2
                else:
                    targetx = (resolution[0] - targetw) / 2
                    targety = 0
                img = img.scale(targetw, targeth)

        else:  # we're going to crop instead
            # center a too small image
            if self.width <= resolution[0] and self.height <= resolution[1]:
                #we're too small just center the thing
                ret_val = np.zeros((resolution[1], resolution[0], 3),
                                   dtype='uint8')
                targetx = (resolution[0] / 2) - (self.width / 2)
                targety = (resolution[1] / 2) - (self.height / 2)
                targeth = self.height
                targetw = self.width
            # crop too big on both axes
            elif self.width > resolution[0] and self.height > resolution[1]:
                targetw = resolution[0]
                targeth = resolution[1]
                targetx = 0
                targety = 0
                x = (self.width - resolution[0]) / 2
                y = (self.height - resolution[1]) / 2
                img = img.crop(x, y, targetw, targeth)
                return img
            # height too big
            elif self.width <= resolution[0] and self.height > resolution[1]:
                # crop along the y dimension and center along the x dimension
                ret_val = np.zeros((resolution[1], resolution[0], 3),
                                   dtype='uint8')
                targetw = self.width
                targeth = resolution[1]
                targetx = (resolution[0] - self.width) / 2
                targety = 0
                x = 0
                y = (self.height - resolution[1]) / 2
                img = img.crop(x, y, targetw, targeth)

            # width too big
            elif self.width > resolution[0] and self.height <= resolution[1]:
                # crop along the y dimension and center along the x dimension
                ret_val = np.zeros((resolution[1], resolution[0], 3),
                                   dtype='uint8')
                targetw = resolution[0]
                targeth = self.height
                targetx = 0
                targety = (resolution[1] - self.height) / 2
                x = (self.width - resolution[0]) / 2
                y = 0
                img = img.crop(x, y, targetw, targeth)

        ret_val[targety:targety + targeth,
                targetx:targetx + targetw] = img._ndarray
        ret_val = Image(ret_val, color_space=self._colorSpace)
        return ret_val

    def blit(self, img, pos=None, alpha=None, mask=None, alpha_mask=None):
        """
        **SUMMARY**

        Blit aka bit blit - which in ye olden days was an acronym for bit-block
        transfer. In other words blit is when you want to smash two images
        together, or add one image to another. This method takes in a second
        simplecv image, and then allows you to add to some point on the calling
        image. A general blit command will just copy all of the image. You can
        also copy the image with an alpha value to the source image is
        semi-transparent. A binary mask can be used to blit non-rectangular
        image onto the souce image. An alpha mask can be used to do and
        arbitrarily transparent image to this image. Both the mask and alpha
        masks are SimpleCV Images.

        **PARAMETERS**

        * *img* - an image to place ontop of this image.
        * *pos* - an (x,y) position tuple of the top left corner of img on this
         image. Note that these values can be negative.
        * *alpha* - a single floating point alpha value
         (0=see the bottom image, 1=see just img, 0.5 blend the two 50/50).
        * *mask* - a binary mask the same size as the input image.
         White areas are blitted, black areas are not blitted.
        * *alpha_mask* - an alpha mask where each grayscale value maps how much
        of each image is shown.

        **RETURNS**

        A SimpleCV Image. The size will remain the same.

        **EXAMPLE**

        >>> topImg = Image("top.png")
        >>> bottomImg = Image("bottom.png")
        >>> mask = Image("mask.png")
        >>> aMask = Image("alpphaMask.png")
        >>> bottomImg.blit(top, pos=(100, 100)).show()
        >>> bottomImg.blit(top, alpha=0.5).show()
        >>> bottomImg.blit(top, pos=(100, 100), mask=mask).show()
        >>> bottomImg.blit(top, pos=(-10, -10), alpha_mask=aMask).show()

        **SEE ALSO**

        :py:meth:`create_binary_mask`
        :py:meth:`create_alpha_mask`

        """
        if pos is None:
            pos = (0, 0)

        (top_roi, bottom_roi) = self._rect_overlap_rois(
            (img.width, img.height), (self.width, self.height), pos)

        if alpha:
            top_img = img.copy().crop(*top_roi)
            bottom_img = self.copy().crop(*bottom_roi)
            alpha = float(alpha)
            beta = float(1.00 - alpha)
            gamma = float(0.00)
            blit_array = cv2.addWeighted(top_img.get_ndarray(), alpha,
                                         bottom_img.get_ndarray(), beta, gamma)
            array = self._ndarray.copy()
            array[bottom_roi[1]:bottom_roi[1] + bottom_roi[3],
                  bottom_roi[0]:bottom_roi[0] + bottom_roi[2]] = blit_array
            return Image(array, color_space=self._colorSpace)
        elif alpha_mask:
            if alpha_mask.size() != img.size():
                logger.warning("Image.blit: your mask and image don't match "
                               "sizes, if the mask doesn't fit, you can not "
                               "blit! Try using the scale function.")
                return None
            top_img = img.copy().crop(*top_roi)
            bottom_img = self.copy().crop(*bottom_roi)
            mask_img = alpha_mask.crop(*top_roi)
            # Apply mask to img
            top_array = top_img.get_ndarray().astype(np.float32)
            gray_mask_array = mask_img.get_gray_ndarray()
            gray_mask_array = gray_mask_array.astype(np.float32) / 255.0
            gray_mask_array = np.dstack((gray_mask_array, gray_mask_array,
                                         gray_mask_array))
            masked_top_array = cv2.multiply(top_array, gray_mask_array)
            # Apply inverted mask to img
            bottom_array = bottom_img.get_ndarray().astype(np.float32)
            inv_graymask_array = mask_img.invert().get_gray_ndarray()
            inv_graymask_array = inv_graymask_array.astype(np.float32) / 255.0
            inv_graymask_array = np.dstack((inv_graymask_array,
                                            inv_graymask_array,
                                            inv_graymask_array))
            masked_bottom_array = cv2.multiply(bottom_array,
                                               inv_graymask_array)

            blit_array = cv2.add(masked_top_array, masked_bottom_array)
            blit_array = blit_array.astype(self.dtype)

            array = self._ndarray.copy()
            array[bottom_roi[1]:bottom_roi[1] + bottom_roi[3],
                  bottom_roi[0]:bottom_roi[0] + bottom_roi[2]] = blit_array
            return Image(array, color_space=self._colorSpace)

        elif mask:
            if mask.size() != img.size():
                logger.warning("Image.blit: your mask and image don't match "
                               "sizes, if the mask doesn't fit, you can not "
                               "blit! Try using the scale function. ")
                return None
            top_img = img.copy().crop(*top_roi)
            mask_img = mask.crop(*top_roi)
            # Apply mask to img
            top_array = top_img.get_ndarray()
            gray_mask_array = mask_img.get_gray_ndarray()
            binary_mask = gray_mask_array != 0
            array = self._ndarray.copy()
            array[Image.roi_to_slice(bottom_roi)][binary_mask] = \
                top_array[binary_mask]
            return Image(array, color_space=self._colorSpace)

        else:  # vanilla blit
            top_img = img.copy().crop(*top_roi)
            array = self._ndarray.copy()
            array[bottom_roi[1]:bottom_roi[1] + bottom_roi[3],
                  bottom_roi[0]:bottom_roi[0] + bottom_roi[2]] = \
                top_img.get_ndarray()
            return Image(array, color_space=self._colorSpace)

    def side_by_side(self, image, side="right", scale=True):
        """
        **SUMMARY**

        Combine two images as a side by side images. Great for before and after
        images.

        **PARAMETERS**

        * *side* - what side of this image to place the other image on.
          choices are ('left'/'right'/'top'/'bottom').

        * *scale* - if true scale the smaller of the two sides to match the
          edge touching the other image. If false we center the smaller
          of the two images on the edge touching the larger image.

        **RETURNS**

        A new image that is a combination of the two images.

        **EXAMPLE**

        >>> img = Image("lenna")
        >>> img2 = Image("orson_welles.jpg")
        >>> img3 = img.side_by_side(img2)

        **TODO**

        Make this accept a list of images.

        """
        # there is probably a cleaner way to do this, but I know I hit every
        # case when they are enumerated
        if side == "top":
            return image.side_by_side(self, "bottom", scale)
        elif side == "bottom":
            if self.width > image.width:
                # scale the other image width to fit
                resized = image.resize(w=self.width) if scale else image
                topimage = self
                w = self.width
            else:  # our width is smaller than the other image
                # scale the other image width to fit
                topimage = self.resize(w=image.width) if scale else self
                resized = image
                w = image.width
            h = topimage.height + resized.height
            xc = (topimage.width - resized.width) / 2
            array = np.zeros((h, w, 3), dtype=self.dtype)
            if xc > 0:
                array[:topimage.height, :topimage.width] = \
                    topimage.get_ndarray()
                array[h - resized.height:, xc:xc + resized.width] = \
                    resized.get_ndarray()
            else:
                array[:topimage.height, abs(xc):abs(xc) + topimage.width] = \
                    topimage.get_ndarray()
                array[h - resized.height:, :resized.width] = \
                    resized.get_ndarray()
            return Image(array, color_space=self._colorSpace)
        elif side == "right":
            return image.side_by_side(self, "left", scale)
        else:  # default to left
            if self.height > image.height:
                # scale the other image height to fit
                resized = image.resize(h=self.height) if scale else image
                rightimage = self
                h = rightimage.height
            else:  # our height is smaller than the other image
                #scale our height to fit
                rightimage = self.resize(h=image.height) if scale else self
                h = image.height
                resized = image
            w = rightimage.width + resized.width
            yc = (rightimage.height - resized.height) / 2
            array = np.zeros((h, w, 3), dtype=self.dtype)
            if yc > 0:
                array[:rightimage.height, w - rightimage.width:] = \
                    rightimage.get_ndarray()
                array[yc:yc + resized.height, :resized.width] = \
                    resized.get_ndarray()
            else:
                array[abs(yc):abs(yc) + rightimage.height,
                      w - rightimage.width:] = rightimage.get_ndarray()
                array[:resized.height, :resized.width] = resized.get_ndarray()
            return Image(array, color_space=self._colorSpace)

    def embiggen(self, size=None, color=Color.BLACK, pos=None):
        """
        **SUMMARY**

        Make the canvas larger but keep the image the same size.

        **PARAMETERS**

        * *size* - width and heigt tuple of the new canvas or give a single
         vaule in which to scale the image size, for instance size=2 would make
         the image canvas twice the size

        * *color* - the color of the canvas

        * *pos* - the position of the top left corner of image on the new
         canvas, if none the image is centered.

        **RETURNS**

        The enlarged SimpleCV Image.

        **EXAMPLE**

        >>> img = Image("lenna")
        >>> img = img.embiggen((1024, 1024), color=Color.BLUE)
        >>> img.show()

        """
        if not self.is_bgr():
            logger.warning("Image.embiggen works only with bgr image")
            return None

        if not isinstance(size, tuple) and size > 1:
            size = (self.width * size, self.height * size)

        if size is None or size[0] < self.width or size[1] < self.height:
            logger.warning("Image.embiggen: the size provided is invalid")
            return None
        array = np.zeros((size[1], size[0], 3), dtype=self.dtype)
        array[:, :, :] = color[::-1]  # RBG to BGR
        if pos is None:
            pos = (((size[0] - self.width) / 2), ((size[1] - self.height) / 2))
        (top_roi, bottom_roi) = self._rect_overlap_rois(
            (self.width, self.height), size, pos)
        if top_roi is None or bottom_roi is None:
            logger.warning("Image.embiggen: the position of the old image "
                           "doesn't make sense, there is no overlap")
            return None
        blit_array = self._ndarray[top_roi[1]:top_roi[1] + top_roi[3],
                                   top_roi[0]:top_roi[0] + top_roi[2]]
        array[bottom_roi[1]:bottom_roi[1] + bottom_roi[3],
              bottom_roi[0]:bottom_roi[0] + bottom_roi[2]] = blit_array
        return Image(array, color_space=self._colorSpace)

    def _rect_overlap_rois(self, top, bottom, pos):
        """
        top is a rectangle (w,h)
        bottom is a rectangle (w,h)
        pos is the top left corner of the top rectangle with respect to the
        bottom rectangle's top left corner method returns none if the two
        rectangles do not overlap. Otherwise returns the top rectangle's
        ROI (x,y,w,h) and the bottom rectangle's ROI (x,y,w,h)
        """
        # the position of the top rect coordinates
        # give bottom top right = (0,0)
        tr = (pos[0] + top[0], pos[1])
        tl = pos
        br = (pos[0] + top[0], pos[1] + top[1])
        bl = (pos[0], pos[1] + top[1])

        # do an overlap test to weed out corner cases and errors
        def in_bounds((w, h), (x, y)):
            ret_val = True
            if x < 0 or y < 0 or x > w or y > h:
                ret_val = False
            return ret_val

        trc = in_bounds(bottom, tr)
        tlc = in_bounds(bottom, tl)
        brc = in_bounds(bottom, br)
        blc = in_bounds(bottom, bl)
        if not trc and not tlc and not brc and not blc:  # no overlap
            return None, None
        # easy case top is fully inside bottom
        elif trc and tlc and brc and blc:
            t_ret = (0, 0, top[0], top[1])
            b_ret = (pos[0], pos[1], top[0], top[1])
            return t_ret, b_ret
        # let's figure out where the top rectangle sits on the bottom
        # we clamp the corners of the top rectangle to live inside
        # the bottom rectangle and from that get the x,y,w,h
        tl = (np.clip(tl[0], 0, bottom[0]), np.clip(tl[1], 0, bottom[1]))
        br = (np.clip(br[0], 0, bottom[0]), np.clip(br[1], 0, bottom[1]))

        bx = tl[0]
        by = tl[1]
        bw = abs(tl[0] - br[0])
        bh = abs(tl[1] - br[1])
        # now let's figure where the bottom rectangle is in the top rectangle
        # we do the same thing with different coordinates
        pos = (-1 * pos[0], -1 * pos[1])
        #recalculate the bottoms's corners with respect to the top.
        tr = (pos[0] + bottom[0], pos[1])
        tl = pos
        br = (pos[0] + bottom[0], pos[1] + bottom[1])
        bl = (pos[0], pos[1] + bottom[1])
        tl = (np.clip(tl[0], 0, top[0]), np.clip(tl[1], 0, top[1]))
        br = (np.clip(br[0], 0, top[0]), np.clip(br[1], 0, top[1]))
        tx = tl[0]
        ty = tl[1]
        tw = abs(br[0] - tl[0])
        th = abs(br[1] - tl[1])
        return (tx, ty, tw, th), (bx, by, bw, bh)

    def create_binary_mask(self, color1=(0, 0, 0), color2=(255, 255, 255)):
        """
        **SUMMARY**

        Generate a binary mask of the image based on a range of rgb values.
        A binary mask is a black and white image where the white area is kept
        and the black area is removed.

        This method is used by specifying two colors as the range between the
        minimum and maximum values that will be masked white.

        **PARAMETERS**

        * *color1* - The starting color range for the mask..
        * *color2* - The end of the color range for the mask.

        **RETURNS**

        A binary (black/white) image mask as a SimpleCV Image.

        **EXAMPLE**

        >>> img = Image("lenna")
        >>> mask = img.create_binary_mask(color1=(0, 128, 128),
        ...                               color2=(255, 255, 255))
        >>> mask.show()

        **SEE ALSO**

        :py:meth:`create_binary_mask`
        :py:meth:`create_alpha_mask`
        :py:meth:`blit`
        :py:meth:`threshold`

        """
        if not self.is_bgr():
            logger.warning("create_binary_mask works only with BGR image")
            return None
        if color1[0] - color2[0] == 0 \
                or color1[1] - color2[1] == 0 \
                or color1[2] - color2[2] == 0:
            logger.warning("No color range selected, the result will be "
                           "black, returning None instead.")
            return None
        if color1[0] > 255 or color1[0] < 0 \
                or color1[1] > 255 or color1[1] < 0 \
                or color1[2] > 255 or color1[2] < 0 \
                or color2[0] > 255 or color2[0] < 0 \
                or color2[1] > 255 or color2[1] < 0 \
                or color2[2] > 255 or color2[2] < 0:
            logger.warning("One of the tuple values falls "
                           "outside of the range of 0 to 255")
            return None
        # converting to BGR
        color1 = tuple(reversed(color1))
        color2 = tuple(reversed(color2))

        results = []
        for index, color in enumerate(zip(color1, color2)):
            chanel = cv2.inRange(self._ndarray[:, :, index],
                                 np.array(min(color)),
                                 np.array(max(color)))
            results.append(chanel)
        array = cv2.bitwise_and(results[0], results[1])
        array = cv2.bitwise_and(array, results[2]).astype(self.dtype)
        return Image(array, color_space=ColorSpace.GRAY)

    def apply_binary_mask(self, mask, bg_color=Color.BLACK):
        """
        **SUMMARY**

        Apply a binary mask to the image. The white areas of the mask will be
        kept, and the black areas removed. The removed areas will be set to the
        color of bg_color.

        **PARAMETERS**

        * *mask* - the binary mask image. White areas are kept, black areas are
         removed.
        * *bg_color* - the color of the background on the mask.

        **RETURNS**

        A binary (black/white) image mask as a SimpleCV Image.

        **EXAMPLE**

        >>> img = Image("lenna")
        >>> mask = img.create_binary_mask(color1=(0, 128, 128),
        ...                               color2=(255, 255, 255))
        >>> result = img.apply_binary_mask(mask)
        >>> result.show()

        **SEE ALSO**

        :py:meth:`create_binary_mask`
        :py:meth:`create_alpha_mask`
        :py:meth:`apply_binary_mask`
        :py:meth:`blit`
        :py:meth:`threshold`

        """
        if self.size() != mask.size():
            logger.warning("Image.apply_binary_mask: your mask and image "
                           "don't match sizes, if the mask doesn't fit, you "
                           "can't apply it! Try using the scale function. ")
            return None

        array = np.zeros((self.height, self.width, 3), self.dtype)
        array = cv2.add(array, np.array(tuple(reversed(bg_color)),
                                        dtype=self.dtype))
        binary_mask = mask.get_gray_ndarray() != 0

        array[binary_mask] = self._ndarray[binary_mask]
        return Image(array, color_space=self._colorSpace)

    def create_alpha_mask(self, hue=60, hue_lb=None, hue_ub=None):
        """
        **SUMMARY**

        Generate a grayscale or binary mask image based either on a hue or an
        RGB triplet that can be used like an alpha channel. In the resulting
        mask, the hue/rgb_color will be treated as transparent (black).

        When a hue is used the mask is treated like an 8bit alpha channel.
        When an RGB triplet is used the result is a binary mask.
        rgb_thresh is a distance measure between a given a pixel and the mask
        value that we will add to the mask. For example, if rgb_color=(0,255,0)
        and rgb_thresh=5 then any pixel within five color values of the
        rgb_color will be added to the mask (e.g. (0,250,0),(5,255,0)....)

        Invert flips the mask values.


        **PARAMETERS**

        * *hue* - a hue used to generate the alpha mask.
        * *hue_lb* - the upper value  of a range of hue values to use.
        * *hue_ub* - the lower value  of a range of hue values to use.

        **RETURNS**

        A grayscale alpha mask as a SimpleCV Image.

        >>> img = Image("lenna")
        >>> mask = img.create_alpha_mask(hue_lb=50, hue_ub=70)
        >>> mask.show()

        **SEE ALSO**

        :py:meth:`create_binary_mask`
        :py:meth:`create_alpha_mask`
        :py:meth:`apply_binary_mask`
        :py:meth:`blit`
        :py:meth:`threshold`

        """

        if hue < 0 or hue > 180:
            logger.warning("Invalid hue color, valid hue range is 0 to 180.")
            return None

        if not self.is_hsv():
            hsv = self.to_hsv()
        else:
            hsv = self.copy()
        h = hsv.get_ndarray()[:, :, 0]
        v = hsv.get_ndarray()[:, :, 2]
        hlut = np.zeros(256, dtype=np.uint8)
        if hue_lb is not None and hue_ub is not None:
            hlut[hue_lb:hue_ub] = 255
        else:
            hlut[hue] = 255
        mask = cv2.LUT(h, hlut)[:, :, 0]
        array = hsv.get_empty(1)
        array = np.where(mask, v, array)
        return Image(array, color_space=ColorSpace.GRAY)

    def apply_pixel_function(self, func):
        """
        **SUMMARY**

        apply a function to every pixel and return the result
        The function must be of the form int (r,g,b)=func((r,g,b))

        **PARAMETERS**

        * *func* - a function pointer to a function of the form
         (r,g.b) = func((r,g,b))

        **RETURNS**

        A simpleCV image after mapping the function to the image.

        **EXAMPLE**

        >>> def derp(pixels):
        >>>     b, g, r = pixels
        >>>     return int(b * .2), int(r * .3), int(g * .5)
        >>>
        >>> img = Image("lenna")
        >>> img2 = img.apply_pixel_function(derp)

        """
        # there should be a way to do this faster using numpy vectorize
        # but I can get vectorize to work with the three channels together...
        # have to split them
        #TODO: benchmark this against vectorize
        pixels = np.array(self._ndarray).reshape(-1, 3).tolist()
        result = np.array(map(func, pixels), dtype=uint8).reshape((
            self.width, self.height, 3))
        return Image(result)

    def integral_image(self, tilted=False):
        """
        **SUMMARY**

        Calculate the integral image and return it as a numpy array.
        The integral image gives the sum of all of the pixels above and to the
        right of a given pixel location. It is useful for computing Haar
        cascades. The return type is a numpy array the same size of the image.
        The integral image requires 32Bit values which are not easily supported
        by the simplecv Image class.

        **PARAMETERS**

        * *tilted*  - if tilted is true we tilt the image 45 degrees and then
         calculate the results.

        **RETURNS**

        A numpy array of the values.

        **EXAMPLE**

        >>> img = Image("logo")
        >>> derp = img.integral_image()

        **SEE ALSO**

        http://en.wikipedia.org/wiki/Summed_area_table
        """

        if tilted:
            _, _, array = cv2.integral3(self.get_gray_ndarray())
        else:
            array = cv2.integral(self.get_gray_ndarray())
        return array

    def convolve(self, kernel=None, center=None):
        """
        **SUMMARY**

        Convolution performs a shape change on an image.  It is similiar to
        something like a dilate.  You pass it a kernel in the form of a list,
        np.array, or cvMat

        **PARAMETERS**

        * *kernel* - The convolution kernel. As list, set or Numpy Array.
        * *center* - If true we use the center of the kernel.

        **RETURNS**

        The image after we apply the convolution.

        **EXAMPLE**

        >>> img = Image("data/sampleimages/simplecv.png")
        >>> kernel = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        >>> conv = img.convolve()

        **SEE ALSO**

        http://en.wikipedia.org/wiki/Convolution

        """
        if kernel is None:
            kernel = np.array(((1, 0, 0), (0, 1, 0), (0, 0, 1)))
        elif isinstance(kernel, (list, set)):
            kernel = np.array(kernel)
        elif isinstance(kernel, np.ndarray):
            pass
        else:
            logger.warning("Image.convolve: kernel should be numpy array.")
            return None

        if center is None:
            array = cv2.filter2D(self._ndarray, -1, kernel)
        else:
            array = cv2.filter2D(self._ndarray, -1, kernel, anchor=center)
        return Image(array, color_space=self._colorSpace)

    def find_template(self, template_image=None, threshold=5,
                      method="SQR_DIFF_NORM", grayscale=True,
                      rawmatches=False):
        """
        **SUMMARY**

        This function searches an image for a template image.  The template
        image is a smaller image that is searched for in the bigger image.
        This is a basic pattern finder in an image.  This uses the standard
        OpenCV template (pattern) matching and cannot handle scaling or
        rotation

        Template matching returns a match score for every pixel in the image.
        Often pixels that are near to each other and a close match to the
        template are returned as a match. If the threshold is set too low
        expect to get a huge number of values. The threshold parameter is in
        terms of the number of standard deviations from the mean match value
        you are looking

        For example, matches that are above three standard deviations will
        return 0.1% of the pixels. In a 800x600 image this means there will be
        800*600*0.001 = 480 matches.

        This method returns the locations of wherever it finds a match above a
        threshold. Because of how template matching works, very often multiple
        instances of the template overlap significantly. The best approach is
        to find the centroid of all of these values. We suggest using an
        iterative k-means approach to find the centroids.


        **PARAMETERS**

        * *template_image* - The template image.
        * *threshold* - Int
        * *method* -

          * SQR_DIFF_NORM - Normalized square difference
          * SQR_DIFF      - Square difference
          * CCOEFF        -
          * CCOEFF_NORM   -
          * CCORR         - Cross correlation
          * CCORR_NORM    - Normalize cross correlation
        * *grayscale* - Boolean - If false, template Match is found using BGR
         image.

        **EXAMPLE**

        >>> image = Image("/path/to/img.png")
        >>> pattern_image = image.crop(100, 100, 100, 100)
        >>> found_patterns = image.find_template(pattern_image)
        >>> found_patterns.draw()
        >>> image.show()

        **RETURNS**

        This method returns a FeatureSet of TemplateMatch objects.

        """
        if template_image is None:
            logger.info("Need image for matching")
            return
        if template_image.width > self.width:
            logger.info("Image too wide")
            return
        if template_image.height > self.height:
            logger.info("Image too tall")
            return

        check = 0  # if check = 0 we want maximal value, otherwise minimal
        # minimal
        if method is None or method == "" or method == "SQR_DIFF_NORM":
            method = cv2.TM_SQDIFF_NORMED
            check = 1
        elif method == "SQR_DIFF":  # minimal
            method = cv2.TM_SQDIFF
            check = 1
        elif method == "CCOEFF":  # maximal
            method = cv2.TM_CCOEFF
        elif method == "CCOEFF_NORM":  # maximal
            method = cv2.TM_CCOEFF_NORMED
        elif method == "CCORR":  # maximal
            method = cv2.TM_CCORR
        elif method == "CCORR_NORM":  # maximal
            method = cv2.TM_CCORR_NORMED
        else:
            logger.warning("ooops.. I don't know what template matching "
                           "method you are looking for.")
            return None

        #choose template matching method to be used
        if grayscale:
            matches = cv2.matchTemplate(self.get_gray_ndarray(),
                                        template_image.get_gray_ndarray(),
                                        method)
        else:
            matches = cv2.matchTemplate(self._ndarray,
                                        template_image.get_ndarray(),
                                        method)
        mean = np.mean(matches)
        sd = np.std(matches)
        if check > 0:
            compute = np.where((matches < mean - threshold * sd))
        else:
            compute = np.where((matches > mean + threshold * sd))

        mapped = map(tuple, np.column_stack(compute))
        fs = FeatureSet()
        for location in mapped:
            fs.append(
                TemplateMatch(self, template_image, (location[1], location[0]),
                              matches[location[0], location[1]]))

        if rawmatches:
            return fs
        # cluster overlapping template matches
        finalfs = FeatureSet()
        if len(fs) > 0:
            finalfs.append(fs[0])
            for f in fs:
                match = False
                for f2 in finalfs:
                    if f2._template_overlaps(f):  # if they overlap
                        f2.consume(f)  # merge them
                        match = True
                        break

                if not match:
                    finalfs.append(f)

            # rescale the resulting clusters to fit the template size
            for f in finalfs:
                f.rescale(template_image.width, template_image.height)
            fs = finalfs
        return fs

    def find_template_once(self, template_image=None, threshold=0.2,
                           method="SQR_DIFF_NORM", grayscale=True):
        """
        **SUMMARY**

        This function searches an image for a single template image match.The
        template image is a smaller image that is searched for in the bigger
        image. This is a basic pattern finder in an image.  This uses the
        standard OpenCV template (pattern) matching and cannot handle scaling
        or rotation

        This method returns the single best match if and only if that
        match less than the threshold (greater than in the case of
        some methods).

        **PARAMETERS**

        * *template_image* - The template image.
        * *threshold* - Int
        * *method* -

          * SQR_DIFF_NORM - Normalized square difference
          * SQR_DIFF      - Square difference
          * CCOEFF        -
          * CCOEFF_NORM   -
          * CCORR         - Cross correlation
          * CCORR_NORM    - Normalize cross correlation
        * *grayscale* - Boolean - If false, template Match is found using BGR
         image.

        **EXAMPLE**

        >>> image = Image("/path/to/img.png")
        >>> pattern_image = image.crop(100, 100, 100, 100)
        >>> found_patterns = image.find_template_once(pattern_image)
        >>> found_patterns.draw()
        >>> image.show()

        **RETURNS**

        This method returns a FeatureSet of TemplateMatch objects.

        """
        if template_image is None:
            logger.info("Need image for template matching.")
            return
        if template_image.width > self.width:
            logger.info("Template image is too wide for the given image.")
            return
        if template_image.height > self.height:
            logger.info("Template image too tall for the given image.")
            return

        check = 0  # if check = 0 we want maximal value, otherwise minimal
        # minimal
        if method is None or method == "" or method == "SQR_DIFF_NORM":
            method = cv2.TM_SQDIFF_NORMED
            check = 1
        elif method == "SQR_DIFF":  # minimal
            method = cv2.TM_SQDIFF
            check = 1
        elif method == "CCOEFF":  # maximal
            method = cv2.TM_CCOEFF
        elif method == "CCOEFF_NORM":  # maximal
            method = cv2.TM_CCOEFF_NORMED
        elif method == "CCORR":  # maximal
            method = cv2.TM_CCORR
        elif method == "CCORR_NORM":  # maximal
            method = cv2.TM_CCORR_NORMED
        else:
            logger.warning("ooops.. I don't know what template matching "
                           "method you are looking for.")
            return None
        #choose template matching method to be used
        if grayscale:
            matches = cv2.matchTemplate(self.get_gray_ndarray(),
                                        template_image.get_gray_ndarray(),
                                        method)
        else:
            matches = cv2.matchTemplate(self._ndarray,
                                        template_image.get_ndarray(), method)
        if check > 0:
            if np.min(matches) <= threshold:
                compute = np.where(matches == np.min(matches))
            else:
                return []
        else:
            if np.max(matches) >= threshold:
                compute = np.where(matches == np.max(matches))
            else:
                return []
        mapped = map(tuple, np.column_stack(compute))
        fs = FeatureSet()
        for location in mapped:
            fs.append(
                TemplateMatch(self, template_image, (location[1], location[0]),
                              matches[location[0], location[1]]))
        return fs

    def read_text(self):
        """
        **SUMMARY**

        This function will return any text it can find using OCR on the
        image.

        Please note that it does not handle rotation well, so if you need
        it in your application try to rotate and/or crop the area so that
        the text would be the same way a document is read

        **RETURNS**

        A String

        **EXAMPLE**

        >>> img = Imgae("somethingwithtext.png")
        >>> text = img.read_text()
        >>> print text

        **NOTE**

        If you're having run-time problems I feel bad for your son,
        I've got 99 problems but dependencies ain't one:

        http://code.google.com/p/tesseract-ocr/
        http://code.google.com/p/python-tesseract/

        """

        if not OCR_ENABLED:
            return "Please install the correct OCR library required - " \
                   "http://code.google.com/p/tesseract-ocr/ " \
                   "http://code.google.com/p/python-tesseract/"

        api = tesseract.TessBaseAPI()
        api.SetOutputName("outputName")
        api.Init(".", "eng", tesseract.OEM_DEFAULT)
        api.SetPageSegMode(tesseract.PSM_AUTO)

        jpgdata = StringIO()
        self.get_pil().save(jpgdata, "jpeg")
        jpgdata.seek(0)
        stringbuffer = jpgdata.read()
        result = tesseract.ProcessPagesBuffer(stringbuffer, len(stringbuffer),
                                              api)
        return result

    def find_circle(self, canny=100, thresh=350, distance=-1):
        """
        **SUMMARY**

        Perform the Hough Circle transform to extract _perfect_ circles from
        the image canny - the upper bound on a canny edge detector used to find
        circle edges.

        **PARAMETERS**

        * *thresh* - the threshold at which to count a circle. Small parts of
          a circle get added to the accumulator array used internally to the
          array. This value is the minimum threshold. Lower thresholds give
          more circles, higher thresholds give fewer circles.

        .. ::Warning:
          If this threshold is too high, and no circles are found the
          underlying OpenCV routine fails and causes a segfault.

        * *distance* - the minimum distance between each successive circle in
          pixels. 10 is a good starting value.

        **RETURNS**

        A feature set of Circle objects.

        **EXAMPLE**

        >>> img = Image("lenna")
        >>> circs = img.find_circle()
        >>> for c in circs:
        >>>    print c


        """

        # a distnace metric for how apart our circles should be
        # this is sa good bench mark
        if distance < 0:
            distance = 1 + max(self.width, self.height) / 50

        circs = cv2.HoughCircles(self.get_gray_ndarray(),
                                 cv2.cv.CV_HOUGH_GRADIENT,
                                 2, distance, param1=canny, param2=thresh)
        if circs is None:
            return None
        circle_fs = FeatureSet()
        for circ in circs[0]:
            circle_fs.append(Circle(self, int(circ[0]), int(circ[1]),
                                    int(circ[2])))
        return circle_fs

    def white_balance(self, method="Simple"):
        """
        **SUMMARY**

        Attempts to perform automatic white balancing.
        Gray World see:
        http://scien.stanford.edu/pages/labsite/2000/psych221/
        projects/00/trek/GWimages.html

        Robust AWB:
        http://scien.stanford.edu/pages/labsite/2010/psych221/
        projects/2010/JasonSu/robustawb.html

        http://scien.stanford.edu/pages/labsite/2010/psych221/
        projects/2010/JasonSu/Papers/
        Robust%20Automatic%20White%20Balance%20Algorithm%20using
        %20Gray%20Color%20Points%20in%20Images.pdf

        Simple AWB:
        http://www.ipol.im/pub/algo/lmps_simplest_color_balance/
        http://scien.stanford.edu/pages/labsite/2010/psych221/
        projects/2010/JasonSu/simplestcb.html



        **PARAMETERS**

        * *method* - The method to use for white balancing. Can be one of the
         following:

          * `Gray World <http://scien.stanford.edu/pages/labsite/2000/psych221/
            projects/00/trek/GWimages.html>`_

          * `Robust AWB <http://scien.stanford.edu/pages/labsite/2010/psych221/
            projects/2010/JasonSu/robustawb.html>`_

          * `Simple AWB <http://www.ipol.im/pub/algo/
            lmps_simplest_color_balance/>`_


        **RETURNS**

        A SimpleCV Image.

        **EXAMPLE**

        >>> img = Image("lenna")
        >>> img2 = img.white_balance()

        """
        if not self.is_bgr():
            logger.warning("Image.white_balance: works only with BGR image")
            return None

        if method == "GrayWorld":
            avg = cv2.mean(self._ndarray)
            bf = float(avg[0])
            gf = float(avg[1])
            rf = float(avg[2])
            af = (bf + gf + rf) / 3.0
            if bf == 0.00:
                b_factor = 1.00
            else:
                b_factor = af / bf

            if gf == 0.00:
                g_factor = 1.00
            else:
                g_factor = af / gf

            if rf == 0.00:
                r_factor = 1.00
            else:
                r_factor = af / rf

            b = self._ndarray[:, :, 0]
            g = self._ndarray[:, :, 1]
            r = self._ndarray[:, :, 2]

            bfloat = cv2.convertScaleAbs(b.astype(np.float32), alpha=b_factor)
            gfloat = cv2.convertScaleAbs(g.astype(np.float32), alpha=g_factor)
            rfloat = cv2.convertScaleAbs(r.astype(np.float32), alpha=r_factor)

            (min_b, max_b, min_b_loc, max_b_loc) = cv2.minMaxLoc(bfloat)
            (min_g, max_g, min_g_loc, max_g_loc) = cv2.minMaxLoc(gfloat)
            (min_r, max_r, min_r_loc, max_r_loc) = cv2.minMaxLoc(rfloat)
            scale = max([max_r, max_g, max_b])
            sfactor = 1.00
            if scale > 255:
                sfactor = 255.00 / float(scale)

            b = cv2.convertScaleAbs(bfloat, alpha=sfactor)
            g = cv2.convertScaleAbs(gfloat, alpha=sfactor)
            r = cv2.convertScaleAbs(rfloat, alpha=sfactor)

            array = np.dstack((b, g, r)).astype(self.dtype)
            return Image(array, color_space=ColorSpace.BGR)
        elif method == "Simple":
            thresh = 0.003
            sz = self.width * self.height
            bcf = sss.cumfreq(self._ndarray[:, :, 0], numbins=256)
            # get our cumulative histogram of values for this color
            bcf = bcf[0]

            blb = -1  # our upper bound
            bub = 256  # our lower bound
            lower_thresh = 0.00
            upper_thresh = 0.00
            #now find the upper and lower thresh% of our values live
            while lower_thresh < thresh:
                blb = blb + 1
                lower_thresh = bcf[blb] / sz
            while upper_thresh < thresh:
                bub = bub - 1
                upper_thresh = (sz - bcf[bub]) / sz

            gcf = sss.cumfreq(self._ndarray[:, :, 1], numbins=256)
            gcf = gcf[0]
            glb = -1  # our upper bound
            gub = 256  # our lower bound
            lower_thresh = 0.00
            upper_thresh = 0.00
            #now find the upper and lower thresh% of our values live
            while lower_thresh < thresh:
                glb = glb + 1
                lower_thresh = gcf[glb] / sz
            while upper_thresh < thresh:
                gub = gub - 1
                upper_thresh = (sz - gcf[gub]) / sz

            rcf = sss.cumfreq(self._ndarray[:, :, 2], numbins=256)
            rcf = rcf[0]
            rlb = -1  # our upper bound
            rub = 256  # our lower bound
            lower_thresh = 0.00
            upper_thresh = 0.00
            #now find the upper and lower thresh% of our values live
            while lower_thresh < thresh:
                rlb = rlb + 1
                lower_thresh = rcf[rlb] / sz
            while upper_thresh < thresh:
                rub = rub - 1
                upper_thresh = (sz - rcf[rub]) / sz
            #now we create the scale factors for the remaining pixels
            rlbf = float(rlb)
            rubf = float(rub)
            glbf = float(glb)
            gubf = float(gub)
            blbf = float(blb)
            bubf = float(bub)

            r_lut = np.ones((256, 1), dtype=uint8)
            g_lut = np.ones((256, 1), dtype=uint8)
            b_lut = np.ones((256, 1), dtype=uint8)
            for i in range(256):
                if i <= rlb:
                    r_lut[i][0] = 0
                elif i >= rub:
                    r_lut[i][0] = 255
                else:
                    rf = ((float(i) - rlbf) * 255.00 / (rubf - rlbf))
                    r_lut[i][0] = int(rf)
                if i <= glb:
                    g_lut[i][0] = 0
                elif i >= gub:
                    g_lut[i][0] = 255
                else:
                    gf = ((float(i) - glbf) * 255.00 / (gubf - glbf))
                    g_lut[i][0] = int(gf)
                if i <= blb:
                    b_lut[i][0] = 0
                elif i >= bub:
                    b_lut[i][0] = 255
                else:
                    bf = ((float(i) - blbf) * 255.00 / (bubf - blbf))
                    b_lut[i][0] = int(bf)
            return self.apply_lut(r_lut, g_lut, b_lut)

    def apply_lut(self, r_lut=None, b_lut=None, g_lut=None):
        """
        **SUMMARY**

        Apply LUT allows you to apply a LUT (look up table) to the pixels in a
        image. Each LUT is just an array where each index in the array points
        to its value in the result image. For example r_lut[0]=255 would change
        all pixels where the red channel is zero to the value 255.

        **PARAMETERS**

        * *r_lut* - np.array of size (256x1) with dtype=uint8.
        * *g_lut* - np.array of size (256x1) with dtype=uint8.
        * *b_lut* - np.array of size (256x1) with dtype=uint8.

        .. warning::
          The dtype is very important. Will throw the following error without
          it: error: dst.size() == src.size() &&
          dst.type() == CV_MAKETYPE(lut.depth(), src.channels())


        **RETURNS**

        The SimpleCV image remapped using the LUT.

        **EXAMPLE**

        This example saturates the red channel:

        >>> rlut = np.ones((256, 1), dtype=uint8) * 255
        >>> img=img.apply_lut(r_lut=rlut)


        NOTE:

        -==== BUG NOTE ====-
        This method seems to error on the LUT map for some versions of OpenCV.
        I am trying to figure out why. -KAS
        """
        if not self.is_bgr():
            logger.warning("Image.apply_lut: works only with BGR image")
            return None
        b = self._ndarray[:, :, 0]
        g = self._ndarray[:, :, 1]
        r = self._ndarray[:, :, 2]
        if r_lut is not None:
            r = cv2.LUT(r, r_lut)
        if g_lut is not None:
            g = cv2.LUT(g, g_lut)
        if b_lut is not None:
            b = cv2.LUT(b, b_lut)
        array = np.dstack((b, g, r))
        return Image(array, color_space=self._colorSpace)

    def _get_raw_keypoints(self, thresh=500.00, flavor="SURF", highquality=1,
                           force_reset=False):
        """
        .. _get_raw_keypoints:
        This method finds keypoints in an image and returns them as the raw
        keypoints and keypoint descriptors. When this method is called it
        caches a the features and keypoints locally for quick and easy access.

        Parameters:
        min_quality - The minimum quality metric for SURF descriptors. Good
                      values range between about 300.00 and 600.00

        flavor - a string indicating the method to use to extract features.
                 A good primer on how feature/keypoint extractiors can be found
                 here:

                 http://en.wikipedia.org/wiki/
                 Feature_detection_(computer_vision)

                 http://www.cg.tu-berlin.de/fileadmin/fg144/
                 Courses/07WS/compPhoto/Feature_Detection.pdf


                 "SURF" - extract the SURF features and descriptors. If you
                 don't know what to use, use this.
                 See: http://en.wikipedia.org/wiki/SURF

                 "STAR" - The STAR feature extraction algorithm
                 See: http://pr.willowgarage.com/wiki/Star_Detector

                 "FAST" - The FAST keypoint extraction algorithm
                 See: http://en.wikipedia.org/wiki/
                 Corner_detection#AST_based_feature_detectors

                 All the flavour specified below are for
                 OpenCV versions >= 2.4.0:

                 "MSER" - Maximally Stable Extremal Regions algorithm

                 See: http://en.wikipedia.org/
                 wiki/Maximally_stable_extremal_regions

                 "Dense" - Dense Scale Invariant Feature Transform.

                 See: http://www.vlfeat.org/api/dsift.html

                 "ORB" - The Oriented FAST and Rotated BRIEF

                 See: http://www.willowgarage.com/sites/default/
                 files/orb_final.pdf

                 "SIFT" - Scale-invariant feature transform

                 See: http://en.wikipedia.org/wiki/
                 Scale-invariant_feature_transform

                 "BRISK" - Binary Robust Invariant Scalable Keypoints

                  See: http://www.asl.ethz.ch/people/lestefan/personal/BRISK

                 "FREAK" - Fast Retina Keypoints

                  See: http://www.ivpe.com/freak.htm
                  Note: It's a keypoint descriptor and not a KeyPoint detector.
                  SIFT KeyPoints are detected and FERAK is used to extract
                  keypoint descriptor.

        highquality - The SURF descriptor comes in two forms, a vector of 64
                      descriptor values and a vector of 128 descriptor values.
                      The latter are "high" quality descriptors.

        force_reset - If keypoints have already been calculated for this image
                     those keypoints are returned veresus recalculating the
                     values. If force reset is True we always recalculate the
                     values, otherwise we will used the cached copies.

        Returns:
        A tuple of keypoint objects and optionally a numpy array of the
        descriptors.

        Example:
        >>> img = Image("aerospace.jpg")
        >>> kp,d = img._get_raw_keypoints()

        Notes:
        If you would prefer to work with the raw keypoints and descriptors each
        image keeps a local cache of the raw values. These are named:

        self._mKeyPoints # A tuple of keypoint objects
        See: http://opencv.itseez.com/modules/features2d/doc/
        common_interfaces_of_feature_detectors.html#keypoint-keypoint
        self._mKPDescriptors # The descriptor as a floating point numpy array
        self._mKPFlavor = "NONE" # The flavor of the keypoints as a string.

        See Also:
         ImageClass._get_raw_keypoints(self, thresh=500.00,
                                     force_reset=False,
                                     flavor="SURF", highquality=1)
         ImageClass._get_flann_matches(self,sd,td)
         ImageClass.find_keypoint_match(self, template, quality=500.00,
                                      minDist=0.2, minMatch=0.4)
         ImageClass.draw_keypoint_matches(self, template, thresh=500.00,
                                        minDist=0.15, width=1)

        """
        if force_reset:
            self._mKeyPoints = None
            self._mKPDescriptors = None

        _detectors = ["SIFT", "SURF", "FAST", "STAR", "FREAK", "ORB", "BRISK",
                      "MSER", "Dense"]
        _descriptors = ["SIFT", "SURF", "ORB", "FREAK", "BRISK"]
        if flavor not in _detectors:
            warnings.warn("Invalid choice of keypoint detector.")
            return None, None

        if self._mKeyPoints is not None and self._mKPFlavor == flavor:
            return self._mKeyPoints, self._mKPDescriptors

        if hasattr(cv2, flavor):

            if flavor == "SURF":
                # cv2.SURF(hessianThreshold, nOctaves,
                #          nOctaveLayers, extended, upright)
                detector = cv2.SURF(thresh, 4, 2, highquality, 1)
                self._mKeyPoints, self._mKPDescriptors = \
                    detector.detect(self.get_gray_ndarray(), None, False)
                if len(self._mKeyPoints) == 0:
                    return None, None
                if highquality == 1:
                    self._mKPDescriptors = self._mKPDescriptors.reshape(
                        (-1, 128))
                else:
                    self._mKPDescriptors = self._mKPDescriptors.reshape(
                        (-1, 64))

            elif flavor in _descriptors:
                detector = getattr(cv2, flavor)()
                self._mKeyPoints, self._mKPDescriptors = \
                    detector.detectAndCompute(self.get_gray_ndarray(), None,
                                              False)
            elif flavor == "MSER":
                if hasattr(cv2, "FeatureDetector_create"):
                    detector = cv2.FeatureDetector_create("MSER")
                    self._mKeyPoints = detector.detect(self.get_gray_ndarray())
        elif flavor == "STAR":
            detector = cv2.StarDetector()
            self._mKeyPoints = detector.detect(self.get_gray_ndarray())
        elif flavor == "FAST":
            if not hasattr(cv2, "FastFeatureDetector"):
                warnings.warn("You need OpenCV >= 2.4.0 to support FAST")
                return None, None
            detector = cv2.FastFeatureDetector(int(thresh), True)
            self._mKeyPoints = detector.detect(self.get_gray_ndarray(), None)
        elif hasattr(cv2, "FeatureDetector_create"):
            if flavor in _descriptors:
                extractor = cv2.DescriptorExtractor_create(flavor)
                if flavor == "FREAK":
                    flavor = "SIFT"
                detector = cv2.FeatureDetector_create(flavor)
                self._mKeyPoints = detector.detect(self.get_gray_ndarray())
                self._mKeyPoints, self._mKPDescriptors = extractor.compute(
                    self.get_gray_ndarray(), self._mKeyPoints)
            else:
                detector = cv2.FeatureDetector_create(flavor)
                self._mKeyPoints = detector.detect(self.get_gray_ndarray())
        else:
            warnings.warn("simplecv can't seem to find appropriate function "
                          "with your OpenCV version.")
            return None, None
        return self._mKeyPoints, self._mKPDescriptors

    @staticmethod
    def _get_flann_matches(sd, td):
        """
        Summary:
        This method does a fast local approximate nearest neighbors (FLANN)
        calculation between two sets of feature vectors. The result are two
        numpy arrays the first one is a list of indexes of the matches and the
        second one is the match distance value. For the match indices or idx,
        the index values correspond to the values of td, and the value in the
        array is the index in td. I. I.e. j = idx[i] is where td[i] matches
        sd[j]. The second numpy array, at the index i is the match distance
        between td[i] and sd[j]. Lower distances mean better matches.

        Parameters:
        sd - A numpy array of feature vectors of any size.
        td - A numpy array of feature vectors of any size, this vector is used
             for indexing and the result arrays will have a length matching
             this vector.

        Returns:
        Two numpy arrays, the first one, idx, is the idx of the matches of the
        vector td with sd. The second one, dist, is the distance value for the
        closest match.

        Example:
        >>> kpt,td = img1._get_raw_keypoints()  # t is template
        >>> kps,sd = img2._get_raw_keypoints()  # s is source
        >>> idx,dist = img1._get_flann_matches(sd, td)
        >>> j = idx[42]
        >>> print kps[j] # matches kp 42
        >>> print dist[i] # the match quality.

        Notes:
        If you would prefer to work with the raw keypoints and descriptors each
        image keeps a local cache of the raw values. These are named:

        self._mKeyPoints # A tuple of keypoint objects
        See: http://opencv.itseez.com/modules/features2d/doc/
        common_interfaces_of_feature_detectors.html#keypoint-keypoint
        self._mKPDescriptors # The descriptor as a floating point numpy array
        self._mKPFlavor = "NONE" # The flavor of the keypoints as a string.

        See:
         ImageClass._get_raw_keypoints(self, thresh=500.00, forceReset=False,
                                     flavor="SURF", highQuality=1)
         ImageClass._get_flann_matches(self, sd, td)
         ImageClass.draw_keypoint_matches(self, template, thresh=500.00,
                                        minDist=0.15, width=1)
         ImageClass.find_keypoints(self, min_quality=300.00,
                                  flavor="SURF", highQuality=False)
         ImageClass.find_keypoint_match(self, template, quality=500.00,
                                      minDist=0.2, minMatch=0.4)
        """
        flann_index_kdtree = 1  # bug: flann enums are missing
        flann_params = dict(algorithm=flann_index_kdtree, trees=4)
        flann = cv2.flann_Index(sd, flann_params)
        # FIXME: need to provide empty dict
        idx, dist = flann.knnSearch(td, 1, params={})
        del flann
        return idx, dist

    def draw_keypoint_matches(self, template, thresh=500.00, min_dist=0.15,
                              width=1):
        """
        **SUMMARY**

        Draw keypoints draws a side by side representation of two images,
        calculates keypoints for both images, determines the keypoint
        correspondences, and then draws the correspondences. This method is
        helpful for debugging keypoint calculations and also looks really
        cool :) The parameters mirror the parameters used for
        findKeypointMatches to assist with debugging

        **PARAMETERS**

        * *template* - A template image.
        * *quality* - The feature quality metric. This can be any value between
          about 300 and 500. Higher values should return fewer, but higher
          quality features.
        * *min_dist* - The value below which the feature correspondence is
          considered a match. This is the distance between two feature vectors
          Good values are between 0.05 and 0.3
        * *width* - The width of the drawn line.

        **RETURNS**

        A side by side image of the template and source image with each feature
        correspondence draw in a different color.

        **EXAMPLE**

        >>> img = cam.getImage()
        >>> template = Image("myTemplate.png")
        >>> result = img.draw_keypoint_matches(self,template,300.00,0.4):

        **NOTES**

        If you would prefer to work with the raw keypoints and descriptors each
        image keeps a local cache of the raw values. These are named:

        self._mKeyPoints # A tuple of keypoint objects
        See: http://opencv.itseez.com/modules/features2d/doc/
        common_interfaces_of_feature_detectors.html#keypoint-keypoint
        self._mKPDescriptors # The descriptor as a floating point numpy array
        self._mKPFlavor = "NONE" # The flavor of the keypoints as a string.

        **SEE ALSO**

        :py:meth:`draw_keypoint_matches`
        :py:meth:`find_keypoints`
        :py:meth:`find_keypoint_match`

        """
        if template is None:
            return None

        result_img = template.side_by_side(self, scale=False)
        hdif = (self.height - template.height) / 2
        skp, sd = self._get_raw_keypoints(thresh)
        tkp, td = template._get_raw_keypoints(thresh)
        if td is None or sd is None:
            logger.warning("We didn't get any descriptors. Image might be too "
                           "uniform or blurry.")
            return result_img
        template_points = float(td.shape[0])
        sample_points = float(sd.shape[0])
        magic_ratio = 1.00
        if sample_points > template_points:
            magic_ratio = float(sd.shape[0]) / float(td.shape[0])
        # match our keypoint descriptors
        idx, dist = self._get_flann_matches(sd, td)
        p = dist[:, 0]
        result = p * magic_ratio < min_dist
        for i in range(0, len(idx)):
            if result[i]:
                pt_a = (tkp[i].pt[1], tkp[i].pt[0] + hdif)
                pt_b = (skp[idx[i]].pt[1] + template.width, skp[idx[i]].pt[0])
                result_img.draw_line(pt_a, pt_b, color=Color.get_random(),
                                     thickness=width)
        return result_img

    def find_keypoint_match(self, template, quality=500.00, min_dist=0.2,
                            min_match=0.4):
        """
        **SUMMARY**

        find_keypoint_match allows you to match a template image with another
        image using SURF keypoints. The method extracts keypoints from each
        image, uses the Fast Local Approximate Nearest Neighbors algorithm to
        find correspondences between the feature points, filters the
        correspondences based on quality, and then, attempts to calculate
        a homography between the two images. This homography allows us to draw
        a matching bounding box in the source image that corresponds to the
        template. This method allows you to perform matchs the ordinarily fail
        when using the find_template method. This method should be able to
        handle a reasonable changes in camera orientation and illumination.
        Using a template that is close to the target image will yield much
        better results.

        .. Warning::
          This method is only capable of finding one instance of the template
          in an image. If more than one instance is visible the homography
          calculation and the method will fail.

        **PARAMETERS**

        * *template* - A template image.
        * *quality* - The feature quality metric. This can be any value between
          about 300 and 500. Higher values should return fewer, but higher
          quality features.
        * *min_dist* - The value below which the feature correspondence is
           considered a match. This is the distance between two feature
           vectors. Good values are between 0.05 and 0.3
        * *min_match* - The percentage of features which must have matches to
          proceed with homography calculation. A value of 0.4 means 40% of
          features must match. Higher values mean better matches are used.
          Good values are between about 0.3 and 0.7


        **RETURNS**

        If a homography (match) is found this method returns a feature set with
        a single KeypointMatch feature. If no match is found None is returned.

        **EXAMPLE**

        >>> template = Image("template.png")
        >>> img = camera.getImage()
        >>> fs = img.find_keypoint_match(template)
        >>> if fs is not None:
        >>>      fs.draw()
        >>>      img.show()

        **NOTES**

        If you would prefer to work with the raw keypoints and descriptors each
        image keeps a local cache of the raw values. These are named:

        | self._mKeyPoints # A Tuple of keypoint objects
        | self._mKPDescriptors # The descriptor as a floating point numpy array
        | self._mKPFlavor = "NONE" # The flavor of the keypoints as a string.
        | `See Documentation <http://opencv.itseez.com/modules/features2d/doc/
        | common_interfaces_of_feature_detectors.html#keypoint-keypoint>`_

        **SEE ALSO**

        :py:meth:`_get_raw_keypoints`
        :py:meth:`_get_flann_matches`
        :py:meth:`draw_keypoint_matches`
        :py:meth:`find_keypoints`

        """
        if template is None:
            return None

        skp, sd = self._get_raw_keypoints(quality)
        tkp, td = template._get_raw_keypoints(quality)
        if skp is None or tkp is None:
            warnings.warn("I didn't get any keypoints. Image might be too "
                          "uniform or blurry.")
            return None

        template_points = float(td.shape[0])
        sample_points = float(sd.shape[0])
        magic_ratio = 1.00
        if sample_points > template_points:
            magic_ratio = float(sd.shape[0]) / float(td.shape[0])

        # match our keypoint descriptors
        idx, dist = self._get_flann_matches(sd, td)
        p = dist[:, 0]
        result = p * magic_ratio < min_dist
        pr = result.shape[0] / float(dist.shape[0])

        # if more than min_match % matches we go ahead and get the data
        if pr > min_match and len(result) > 4:
            lhs = []
            rhs = []
            for i in range(0, len(idx)):
                if result[i]:
                    lhs.append((tkp[i].pt[1], tkp[i].pt[0]))
                    rhs.append((skp[idx[i]].pt[0], skp[idx[i]].pt[1]))

            rhs_pt = np.array(rhs)
            lhs_pt = np.array(lhs)
            if len(rhs_pt) < 16 or len(lhs_pt) < 16:
                return None
            (homography, mask) = cv2.findHomography(lhs_pt, rhs_pt, cv2.RANSAC,
                                                    ransacReprojThreshold=1.0)
            w = template.width
            h = template.height

            pts = np.array([[0, 0], [0, h], [w, h], [w, 0]], dtype=np.float32)

            ppts = cv2.perspectiveTransform(np.array([pts]), homography)

            pt0i = (ppts[0][0][0], ppts[0][0][1])
            pt1i = (ppts[0][1][0], ppts[0][1][1])
            pt2i = (ppts[0][2][0], ppts[0][2][1])
            pt3i = (ppts[0][3][0], ppts[0][3][1])

            #construct the feature set and return it.
            fs = FeatureSet()
            fs.append(KeypointMatch(self, template, (pt0i, pt1i, pt2i, pt3i),
                                    homography))
            # the homography matrix is necessary for many purposes like image
            # stitching.
            # No need to add homography as it is already being
            # fs.append(homography)
            # added in KeyPointMatch class.
            return fs
        else:
            return None

    def find_keypoints(self, min_quality=300.00, flavor="SURF",
                       highquality=False):
        """
        **SUMMARY**

        This method finds keypoints in an image and returns them as a feature
        set. Keypoints are unique regions in an image that demonstrate some
        degree of invariance to changes in camera pose and illumination. They
        are helpful for calculating homographies between camera views, object
        rotations, and multiple view overlaps.

        We support four keypoint detectors and only one form of keypoint
        descriptors. Only the surf flavor of keypoint returns feature and
        descriptors at this time.

        **PARAMETERS**

        * *min_quality* - The minimum quality metric for SURF descriptors.
          Good values range between about 300.00 and 600.00

        * *flavor* - a string indicating the method to use to extract features.
          A good primer on how feature/keypoint extractiors can be found in
          `feature detection on wikipedia <http://en.wikipedia.org/wiki/
          Feature_detection_(computer_vision)>`_
          and
          `this tutorial. <http://www.cg.tu-berlin.de/fileadmin/fg144/
          Courses/07WS/compPhoto/Feature_Detection.pdf>`_


          * "SURF" - extract the SURF features and descriptors. If you don't
           know what to use, use this.

            See: http://en.wikipedia.org/wiki/SURF

          * "STAR" - The STAR feature extraction algorithm

            See: http://pr.willowgarage.com/wiki/Star_Detector

          * "FAST" - The FAST keypoint extraction algorithm

            See: http://en.wikipedia.org/wiki/
            Corner_detection#AST_based_feature_detectors

          All the flavour specified below are for OpenCV versions >= 2.4.0 :

          * "MSER" - Maximally Stable Extremal Regions algorithm

            See: http://en.wikipedia.org/wiki/Maximally_stable_extremal_regions

          * "Dense" -

          * "ORB" - The Oriented FAST and Rotated BRIEF

            See: http://www.willowgarage.com/sites/default/files/orb_final.pdf

          * "SIFT" - Scale-invariant feature transform

            See: http://en.wikipedia.org/wiki/Scale-invariant_feature_transform

          * "BRISK" - Binary Robust Invariant Scalable Keypoints

            See: http://www.asl.ethz.ch/people/lestefan/personal/BRISK

           * "FREAK" - Fast Retina Keypoints

             See: http://www.ivpe.com/freak.htm
             Note: It's a keypoint descriptor and not a KeyPoint detector.
             SIFT KeyPoints are detected and FERAK is used to extract
             keypoint descriptor.

        * *highquality* - The SURF descriptor comes in two forms, a vector of
          64 descriptor values and a vector of 128 descriptor values. The
          latter are "high" quality descriptors.

        **RETURNS**

        A feature set of KeypointFeatures. These KeypointFeatures let's you
        draw each feature, crop the features, get the feature descriptors, etc.

        **EXAMPLE**

        >>> img = Image("aerospace.jpg")
        >>> fs = img.find_keypoints(flavor="SURF", min_quality=500,
            ...                    highquality=True)
        >>> fs = fs.sort_area()
        >>> fs[-1].draw()
        >>> img.draw()

        **NOTES**

        If you would prefer to work with the raw keypoints and descriptors each
        image keeps a local cache of the raw values. These are named:

        :py:meth:`_get_raw_keypoints`
        :py:meth:`_get_flann_matches`
        :py:meth:`draw_keypoint_matches`
        :py:meth:`find_keypoints`

        """

        fs = FeatureSet()

        if highquality:
            kp, d = self._get_raw_keypoints(thresh=min_quality,
                                            force_reset=True,
                                            flavor=flavor, highquality=1)
        else:
            kp, d = self._get_raw_keypoints(thresh=min_quality,
                                            force_reset=True,
                                            flavor=flavor, highquality=0)

        if flavor in ["ORB", "SIFT", "SURF", "BRISK", "FREAK"] \
                and kp is not None and d is not None:
            for i in range(0, len(kp)):
                fs.append(KeyPoint(self, kp[i], d[i], flavor))
        elif flavor in ["FAST", "STAR", "MSER", "Dense"] and kp is not None:
            for i in range(0, len(kp)):
                fs.append(KeyPoint(self, kp[i], None, flavor))
        else:
            logger.warning("ImageClass.Keypoints: I don't know the method "
                           "you want to use")
            return None

        return fs

    def find_motion(self, previous_frame, window=11, aggregate=True):
        """
        **SUMMARY**

        find_motion performs an optical flow calculation. This method attempts
        to find motion between two subsequent frames of an image. You provide
        it with the previous frame image and it returns a feature set of motion
        fetures that are vectors in the direction of motion.

        **PARAMETERS**

        * *previous_frame* - The last frame as an Image.
        * *window* - The block size for the algorithm. For the the HS and LK
          methods this is the regular sample grid at which we return motion
          samples. For the block matching method this is the matching window
          size.
        * *method* - The algorithm to use as a string.
          Your choices are:

          * 'BM' - default block matching robust but slow - if you are unsure
           use this.

          * 'LK' - `Lucas-Kanade method <http://en.wikipedia.org/
          wiki/Lucas%E2%80%93Kanade_method>`_

          * 'HS' - `Horn-Schunck method <http://en.wikipedia.org/
          wiki/Horn%E2%80%93Schunck_method>`_

        * *aggregate* - If aggregate is true, each of our motion features is
          the average of motion around the sample grid defined by window. If
          aggregate is false we just return the the value as sampled at the
          window grid interval. For block matching this flag is ignored.

        **RETURNS**

        A featureset of motion objects.

        **EXAMPLES**

        >>> cam = Camera()
        >>> img1 = cam.getImage()
        >>> img2 = cam.getImage()
        >>> motion = img2.find_motion(img1)
        >>> motion.draw()
        >>> img2.show()

        **SEE ALSO**

        :py:class:`Motion`
        :py:class:`FeatureSet`

        """
        if self.size() != previous_frame.size():
            logger.warning("Image.find_motion: To find motion the current "
                           "and previous frames must match")
            return None

        flow = cv2.calcOpticalFlowFarneback(previous_frame.get_gray_ndarray(),
                                            self.get_gray_ndarray(), None, 0.5,
                                            1, window, 1, 7, 1.5, 0)
        fs = FeatureSet()
        max_mag = 0.00
        w = math.floor(float(window) / 2.0)
        cx = ((self.width - window) / window) + 1  # our sample rate
        cy = ((self.height - window) / window) + 1
        vx = 0.00
        vy = 0.00
        xf = flow[:, :, 0]
        yf = flow[:, :, 1]
        for x in range(0, int(cx)):  # go through our sample grid
            for y in range(0, int(cy)):
                xi = (x * window) + w  # calculate the sample point
                yi = (y * window) + w
                if aggregate:
                    lowx = int(xi - w)
                    highx = int(xi + w)
                    lowy = int(yi - w)
                    highy = int(yi + w)
                    # get the average x/y components in the output
                    xderp = xf[lowy:highy, lowx:highx]
                    yderp = yf[lowy:highy, lowx:highx]
                    vx = np.average(xderp)
                    vy = np.average(yderp)
                else:  # other wise just sample
                    vx = xf[yi, xi]
                    vy = yf[yi, xi]

                mag = (vx * vx) + (vy * vy)
                # calculate the max magnitude for normalizing our vectors
                if mag > max_mag:
                    max_mag = mag
                # add the sample to the feature set
                fs.append(Motion(self, xi, yi, vx, vy, window))
        return fs

    def _generate_palette(self, bins, hue, centroids=None):
        """
        **SUMMARY**

        This is the main entry point for palette generation. A palette, for our
        purposes, is a list of the main colors in an image. Creating a palette
        with 10 bins, tries to cluster the colors in rgb space into ten
        distinct groups. In hue space we only look at the hue channel. All of
        the relevant palette data is cached in the image
        class.

        **PARAMETERS**

        * *bins* - an integer number of bins into which to divide the colors in
         the image.
        * *hue* - if hue is true we do only cluster on the image hue values.
        * *centroids* - A list of tuples that are the initial k-means
        estimates. This is handy if you want consisten results from the
        palettize.

        **RETURNS**

        Nothing, but creates the image's cached values for:

        self._mDoHuePalette
        self._mPaletteBins
        self._mPalette
        self._mPaletteMembers
        self._mPalettePercentages


        **EXAMPLE**

        >>> img._generate_palette(bins=42)

        **NOTES**

        The hue calculations should be siginificantly faster than the generic
        RGB calculation as it works in a one dimensional space. Sometimes the
        underlying scipy method freaks out about k-means initialization with
        the following warning:

        UserWarning: One of the clusters is empty. Re-run kmean with
        a different initialization.

        This shouldn't be a real problem.

        **SEE ALSO**

        ImageClass.get_palette(self, bins=10, hue=False)
        ImageClass.re_palette(self, palette, hue=False)
        ImageClass.draw_palette_colors(self, size=(-1, -1), horizontal=True,
                                     bins=10, hue=False)
        ImageClass.palettize(self, bins=10 ,hue=False)
        ImageClass.binarize_from_palette(self, palette_selection)
        ImageClass.find_blobs_from_palette(self, palette_selection, dilate = 0,
                                        minsize=5, maxsize=0)
        """
        # FIXME: There is a performance issue

        if self._mPaletteBins != bins or self._mDoHuePalette != hue:
            total = float(self.width * self.height)
            percentages = []
            result = None
            if not hue:
                # reshape our matrix to 1xN
                pixels = np.array(self._ndarray).reshape(-1, 3)
                if centroids is None:
                    result = scv.kmeans(pixels, bins)
                else:
                    if isinstance(centroids, list):
                        centroids = np.array(centroids, dtype=np.uint8)
                    result = scv.kmeans(pixels, centroids)

                self._mPaletteMembers = scv.vq(pixels, result[0])[0]

            else:
                hsv = self
                if not self.is_hsv():
                    hsv = self.to_hsv()

                h = hsv._ndarray[:, :, 0]
                pixels = h.reshape(-1, 1)

                if centroids is None:
                    result = scv.kmeans(pixels, bins)
                else:
                    if isinstance(centroids, list):
                        centroids = np.array(centroids, dtype=np.uint8)
                        centroids = centroids.reshape(centroids.shape[0], 1)
                    result = scv.kmeans(pixels, centroids)

                self._mPaletteMembers = scv.vq(pixels, result[0])[0]

            for i in range(0, bins):
                count = np.where(self._mPaletteMembers == i)
                v = float(count[0].shape[0]) / total
                percentages.append(v)

            self._mDoHuePalette = hue
            self._mPaletteBins = bins
            self._mPalette = np.array(result[0], dtype=np.uint8)
            self._mPalettePercentages = percentages

    def get_palette(self, bins=10, hue=False, centroids=None):
        """
        **SUMMARY**

        This method returns the colors in the palette of the image. A palette
        is the set of the most common colors in an image. This method is
        helpful for segmentation.

        **PARAMETERS**

        * *bins* - an integer number of bins into which to divide the colors in
         the image.
        * *hue*  - if hue is true we do only cluster on the image hue values.
        * *centroids* - A list of tuples that are the initial k-means
         estimates. This is handy if you want consisten results from the
         palettize.

        **RETURNS**

        A numpy array of the BGR color tuples.

        **EXAMPLE**

        >>> p = img.get_palette(bins=42)
        >>> print p[2]

        **NOTES**

        The hue calculations should be siginificantly faster than the generic
        RGB calculation as it works in a one dimensional space. Sometimes the
        underlying scipy method freaks out about k-means initialization with
        the following warning:

        .. Warning::
          One of the clusters is empty. Re-run kmean with a different
          initialization. This shouldn't be a real problem.

        **SEE ALSO**

        :py:meth:`re_palette`
        :py:meth:`draw_palette_colors`
        :py:meth:`palettize`
        :py:meth:`get_palette`
        :py:meth:`binarize_from_palette`
        :py:meth:`find_blobs_from_palette`

        """
        self._generate_palette(bins, hue, centroids)
        return self._mPalette

    def re_palette(self, palette, hue=False):
        """
        **SUMMARY**

        re_palette takes in the palette from another image and attempts to
        apply it to this image. This is helpful if you want to speed up the
        palette computation for a series of images (like those in a video
        stream).

        **PARAMETERS**

        * *palette* - The pre-computed palette from another image.
        * *hue* - Boolean Hue - if hue is True we use a hue palette, otherwise
         we use a BGR palette.

        **RETURNS**

        A SimpleCV Image.

        **EXAMPLE**

        >>> img = Image("lenna")
        >>> img2 = Image("logo")
        >>> p = img.get_palette()
        >>> result = img2.re_palette(p)
        >>> result.show()

        **SEE ALSO**

        :py:meth:`re_palette`
        :py:meth:`draw_palette_colors`
        :py:meth:`palettize`
        :py:meth:`get_palette`
        :py:meth:`binarize_from_palette`
        :py:meth:`find_blobs_from_palette`

        """
        ret_val = None
        if hue:
            if not self.is_hsv():
                hsv = self.to_hsv()
            else:
                hsv = self.copy()

            h = hsv.get_ndarray()[:, :, 0]
            pixels = h.reshape(-1, 1)
            result = scv.vq(pixels, palette)
            derp = palette[result[0]]
            ret_val = Image(derp.reshape(self.height, self.width))
            ret_val = ret_val.rotate(-90, fixed=False)
            ret_val._mDoHuePalette = True
            ret_val._mPaletteBins = len(palette)
            ret_val._mPalette = palette
            ret_val._mPaletteMembers = result[0]

        else:
            result = scv.vq(self.get_ndarray().reshape(-1, 3), palette)
            ret_val = Image(
                palette[result[0]].reshape(self.width, self.height, 3))
            ret_val._mDoHuePalette = False
            ret_val._mPaletteBins = len(palette)
            ret_val._mPalette = palette
            pixels = np.array(self.get_ndarray()).reshape(-1, 3)
            ret_val._mPaletteMembers = scv.vq(pixels, palette)[0]

        percentages = []
        total = self.width * self.height
        for i in range(0, len(palette)):
            count = np.where(self._mPaletteMembers == i)
            v = float(count[0].shape[0]) / total
            percentages.append(v)
        self._mPalettePercentages = percentages
        return ret_val

    def draw_palette_colors(self, size=(-1, -1), horizontal=True, bins=10,
                            hue=False):
        """
        **SUMMARY**

        This method returns the visual representation (swatches) of the palette
        in an image. The palette is orientated either horizontally or
        vertically, and each color is given an area proportional to the number
        of pixels that have that color in the image. The palette is arranged as
        it is returned from the clustering algorithm. When size is left
        to its default value, the palette size will match the size of the
        orientation, and then be 10% of the other dimension. E.g. if our image
        is 640X480 the horizontal palette will be (640x48) likewise the
        vertical palette will be (480x64)

        If a Hue palette is used this method will return a grayscale palette.

        **PARAMETERS**

        * *bins* - an integer number of bins into which to divide the colors in
          the image.
        * *hue* - if hue is true we do only cluster on the image hue values.
        * *size* - The size of the generated palette as a (width,height) tuple,
          if left default we select
          a size based on the image so it can be nicely displayed with the
          image.
        * *horizontal* - If true we orientate our palette horizontally,
         otherwise vertically.

        **RETURNS**

        A palette swatch image.

        **EXAMPLE**

        >>> p = img1.draw_palette_colors()
        >>> img2 = img1.side_by_side(p, side="bottom")
        >>> img2.show()

        **NOTES**

        The hue calculations should be siginificantly faster than the generic
        RGB calculation as it works in a one dimensional space. Sometimes the
        underlying scipy method freaks out about k-means initialization with
        the following warning:

        .. Warning::
          One of the clusters is empty. Re-run kmean with a different
          initialization. This shouldn't be a real problem.

        **SEE ALSO**

        :py:meth:`re_palette`
        :py:meth:`draw_palette_colors`
        :py:meth:`palettize`
        :py:meth:`get_palette`
        :py:meth:`binarize_from_palette`
        :py:meth:`find_blobs_from_palette`

        """
        self._generate_palette(bins, hue)
        ret_val = None
        if not hue:
            if horizontal:
                if size[0] == -1 or size[1] == -1:
                    size = (int(self.width), int(self.height * .1))
                pal = np.zeros((size[1], size[0], 3), dtype=np.uint8)
                idx_l = 0
                idx_h = 0
                for i in range(0, min(bins, self._mPalette.shape[0])):
                    idx_h = np.clip(
                        idx_h +
                        (self._mPalettePercentages[i] * float(size[0])),
                        0, size[0] - 1)
                    roi = (int(idx_l), 0, int(idx_h - idx_l), size[1])
                    roiimage = pal[roi[1]:roi[1] + roi[3],
                                   roi[0]:roi[0] + roi[2]]
                    color = np.array((
                        float(self._mPalette[i][2]),
                        float(self._mPalette[i][1]),
                        float(self._mPalette[i][0])))
                    roiimage += color
                    pal[roi[1]:roi[1] + roi[3],
                        roi[0]:roi[0] + roi[2]] = roiimage
                    idx_l = idx_h
                ret_val = Image(pal)
            else:
                if size[0] == -1 or size[1] == -1:
                    size = (int(self.width * .1), int(self.height))
                pal = np.zeros((size[1], size[0], 3), dtype=np.uint8)
                idx_l = 0
                idx_h = 0
                for i in range(0, min(bins, self._mPalette.shape[0])):
                    idx_h = np.clip(
                        idx_h + self._mPalettePercentages[i] * size[1], 0,
                        size[1] - 1)
                    roi = (0, int(idx_l), size[0], int(idx_h - idx_l))
                    roiimage = pal[roi[1]:roi[1] + roi[3],
                                   roi[0]:roi[0] + roi[2]]
                    color = np.array((
                        float(self._mPalette[i][2]),
                        float(self._mPalette[i][1]),
                        float(self._mPalette[i][0])))
                    roiimage += color
                    pal[roi[1]:roi[1] + roi[3],
                        roi[0]:roi[0] + roi[2]] = roiimage
                    idx_l = idx_h
                ret_val = Image(pal)
        else:  # do hue
            if horizontal:
                if size[0] == -1 or size[1] == -1:
                    size = (int(self.width), int(self.height * .1))
                pal = np.zeros((size[1], size[0], 3), dtype=np.uint8)
                idx_l = 0
                idx_h = 0
                for i in range(0, min(bins, self._mPalette.shape[0])):
                    idx_h = np.clip(
                        idx_h +
                        (self._mPalettePercentages[i] * float(size[0])),
                        0, size[0] - 1)
                    roi = (int(idx_l), 0, int(idx_h - idx_l), size[1])
                    roiimage = pal[roi[1]:roi[1] + roi[3],
                                   roi[0]:roi[0] + roi[2]]
                    roiimage += self._mPalette[i]
                    pal[roi[1]:roi[1] + roi[3],
                        roi[0]:roi[0] + roi[2]] = roiimage
                    idx_l = idx_h
                ret_val = Image(pal)
            else:
                if size[0] == -1 or size[1] == -1:
                    size = (int(self.width * .1), int(self.height))
                pal = np.zeros((size[1], size[0], 3), dtype=np.uint8)
                idx_l = 0
                idx_h = 0
                for i in range(0, min(bins, self._mPalette.shape[0])):
                    idx_h = np.clip(
                        idx_h + self._mPalettePercentages[i] * size[1], 0,
                        size[1] - 1)
                    roi = (0, int(idx_l), size[0], int(idx_h - idx_l))
                    roiimage = pal[roi[1]:roi[1] + roi[3],
                                   roi[0]:roi[0] + roi[2]]
                    roiimage += self._mPalette[i]
                    pal[roi[1]:roi[1] + roi[3],
                        roi[0]:roi[0] + roi[2]] = roiimage
                    idx_l = idx_h
                ret_val = Image(pal)
        return ret_val

    def palettize(self, bins=10, hue=False, centroids=None):
        """
        **SUMMARY**

        This method analyzes an image and determines the most common colors
        using a k-means algorithm. The method then goes through and replaces
        each pixel with the centroid of the clutsters found by k-means. This
        reduces the number of colors in an image to the number of bins. This
        can be particularly handy for doing segementation based on color.

        **PARAMETERS**

        * *bins* - an integer number of bins into which to divide the colors
          in the image.
        * *hue* - if hue is true we do only cluster on the image hue values.


        **RETURNS**

        An image matching the original where each color is replaced with its
        palette value.

        **EXAMPLE**

        >>> img2 = img1.palettize()
        >>> img2.show()

        **NOTES**

        The hue calculations should be siginificantly faster than the generic
        RGB calculation as it works in a one dimensional space. Sometimes the
        underlying scipy method freaks out about k-means initialization with
        the following warning:

        .. Warning::
          UserWarning: One of the clusters is empty. Re-run kmean with a
          different initialization. This shouldn't be a real problem.

        **SEE ALSO**

        :py:meth:`re_palette`
        :py:meth:`draw_palette_colors`
        :py:meth:`palettize`
        :py:meth:`get_palette`
        :py:meth:`binarize_from_palette`
        :py:meth:`find_blobs_from_palette`

        """
        ret_val = None
        self._generate_palette(bins, hue, centroids)
        if hue:
            derp = self._mPalette[self._mPaletteMembers]
            ret_val = Image(derp.reshape(self.height, self.width))
            ret_val = ret_val.rotate(-90, fixed=False)
        else:
            ret_val = Image(
                self._mPalette[self._mPaletteMembers].reshape(self.width,
                                                              self.height, 3))
        return ret_val

    def find_blobs_from_palette(self, palette_selection, dilate=0, minsize=5,
                                maxsize=0, appx_level=3):
        """
        **SUMMARY**

        This method attempts to use palettization to do segmentation and
        behaves similar to the find_blobs blob in that it returs a feature set
        of blob objects. Once a palette has been extracted using get_palette()
        we can then select colors from that palette to be labeled white within
        our blobs.

        **PARAMETERS**

        * *palette_selection* - color triplets selected from our palette that
          will serve turned into blobs. These values can either be a 3xN numpy
          array, or a list of RGB triplets.
        * *dilate* - the optional number of dilation operations to perform on
          the binary image prior to performing blob extraction.
        * *minsize* - the minimum blob size in pixels
        * *maxsize* - the maximim blob size in pixels.
        * *appx_level* - The blob approximation level - an integer for the
          maximum distance between the true edge and the approximation edge -
          lower numbers yield better approximation.

        **RETURNS**

        If the method executes successfully a FeatureSet of Blobs is returned
        from the image. If the method fails a value of None is returned.

       **EXAMPLE**

        >>> img = Image("lenna")
        >>> p = img.get_palette()
        >>> blobs = img.find_blobs_from_palette((p[0], p[1], p[6]))
        >>> blobs.draw()
        >>> img.show()

        **SEE ALSO**

        :py:meth:`re_palette`
        :py:meth:`draw_palette_colors`
        :py:meth:`palettize`
        :py:meth:`get_palette`
        :py:meth:`binarize_from_palette`
        :py:meth:`find_blobs_from_palette`

        """

        #we get the palette from find palete
        #ASSUME: GET PALLETE WAS CALLED!
        bwimg = self.binarize_from_palette(palette_selection)
        if dilate > 0:
            bwimg = bwimg.dilate(dilate)

        if maxsize == 0:
            maxsize = self.width * self.height
        #create a single channel image, thresholded to parameters

        blobmaker = BlobMaker()
        blobs = blobmaker.extract_from_binary(bwimg,
                                              self, minsize=minsize,
                                              maxsize=maxsize,
                                              appx_level=appx_level)

        if not len(blobs):
            return None
        return blobs

    def binarize_from_palette(self, palette_selection):
        """
        **SUMMARY**

        This method uses the color palette to generate a binary (black and
        white) image. Palaette selection is a list of color tuples retrieved
        from img.get_palette(). The provided values will be drawn white
        while other values will be black.

        **PARAMETERS**

        palette_selection - color triplets selected from our palette that will
        serve turned into blobs. These values can either be a 3xN numpy array,
        or a list of RGB triplets.

        **RETURNS**

        This method returns a black and white images, where colors that are
        close to the colors in palette_selection are set to white

        **EXAMPLE**

        >>> img = Image("lenna")
        >>> p = img.get_palette()
        >>> b = img.binarize_from_palette((p[0], p[1], [6]))
        >>> b.show()

        **SEE ALSO**

        :py:meth:`re_palette`
        :py:meth:`draw_palette_colors`
        :py:meth:`palettize`
        :py:meth:`get_palette`
        :py:meth:`binarize_from_palette`
        :py:meth:`find_blobs_from_palette`

        """

        #we get the palette from find palete
        #ASSUME: GET PALLETE WAS CALLED!
        if self._mPalette is None:
            logger.warning("Image.binarize_from_palette: No palette exists, "
                           "call get_palette())")
            return None
        ret_val = None
        img = self.palettize(self._mPaletteBins, hue=self._mDoHuePalette)
        if not self._mDoHuePalette:
            npimg = img.get_ndarray()
            white = np.array([255, 255, 255])
            black = np.array([0, 0, 0])

            for p in palette_selection:
                npimg = np.where(npimg != p, npimg, white)

            npimg = np.where(npimg != white, black, white)
            ret_val = Image(npimg)
        else:
            npimg = img.get_ndarray()[:, :, 1]
            white = np.array([255])
            black = np.array([0])

            for p in palette_selection:
                npimg = np.where(npimg != p, npimg, white)

            npimg = np.where(npimg != white, black, white)
            ret_val = Image(npimg)
        return ret_val

    def skeletonize(self, radius=5):
        """
        **SUMMARY**

        Skeletonization is the process of taking in a set of blobs (here blobs
        are white on a black background) and finding a squigly line that would
        be the back bone of the blobs were they some sort of vertebrate animal.
        Another way of thinking about skeletonization is that it finds a series
        of lines that approximates a blob's shape.

        A good summary can be found here:

        http://www.inf.u-szeged.hu/~palagyi/skel/skel.html

        **PARAMETERS**

        * *radius* - an intenger that defines how roughly how wide a blob must
          be to be added to the skeleton, lower values give more skeleton
          lines, higher values give fewer skeleton lines.

        **EXAMPLE**

        >>> cam = Camera()
        >>> while True:
        ...     img = cam.getImage()
        ...     b = img.binarize().invert()
        ...     s = img.skeletonize()
        ...     r = b - s
        ...     r.show()


        **NOTES**

        This code was a suggested improvement by Alex Wiltchko, check out his
        awesome blog here:

        http://alexbw.posterous.com/

        """
        img_array = self.get_gray_ndarray()
        distance_img = ndimage.distance_transform_edt(img_array)
        morph_laplace_img = ndimage.morphological_laplace(distance_img,
                                                          (radius, radius))
        skeleton = morph_laplace_img < morph_laplace_img.min() / 2
        ret_val = np.zeros([self.width, self.height])
        ret_val[skeleton] = 255
        ret_val = ret_val.astype(self.dtype)
        return Image(ret_val, color_space=ColorSpace.GRAY)

    def smart_threshold(self, mask=None, rect=None):
        """
        **SUMMARY**

        smart_threshold uses a method called grabCut, also called graph cut, to
        automagically generate a grayscale mask image. The dumb version of
        threshold just uses color, smart_threshold looks at both color and
        edges to find a blob. To work smart_threshold needs either a rectangle
        that bounds the object you want to find, or a mask. If you use
        a rectangle make sure it holds the complete object. In the case of
        a mask, it need not be a normal binary mask, it can have the normal
        white foreground and black background, but also a light and dark gray
        values that correspond to areas that are more likely to be foreground
        and more likely to be background. These values can be found in the
        color class as Color.BACKGROUND, Color.FOREGROUND,
        Color.MAYBE_BACKGROUND, and Color.MAYBE_FOREGROUND.

        **PARAMETERS**

        * *mask* - A grayscale mask the same size as the image using the 4 mask
         color values
        * *rect* - A rectangle tuple of the form (x_position, y_position,
          width, height)

        **RETURNS**

        A grayscale image with the foreground / background values assigned to:

        * BACKGROUND = (0,0,0)

        * MAYBE_BACKGROUND = (64,64,64)

        * MAYBE_FOREGROUND =  (192,192,192)

        * FOREGROUND = (255,255,255)

        **EXAMPLE**

        >>> img = Image("RatTop.png")
        >>> mask = Image((img.width,img.height))
        >>> mask.dl().circle((100, 100), 80, color=Color.MAYBE_BACKGROUND,
        ...                  filled=True)
        >>> mask.dl().circle((100, 100), 60, color=Color.MAYBE_FOREGROUND,
        ...                  filled=True)
        >>> mask.dl().circle((100 ,100), 40, color=Color.FOREGROUND,
        ...                  filled=True)
        >>> mask = mask.apply_layers()
        >>> new_mask = img.smart_threshold(mask=mask)
        >>> new_mask.show()

        **NOTES**

        http://en.wikipedia.org/wiki/Graph_cuts_in_computer_vision

        **SEE ALSO**

        :py:meth:`smart_find_blobs`

        """
        ret_val = None
        if mask is not None:
            gray_array = mask.get_gray_ndarray()
            # translate the human readable images to something
            # opencv wants using a lut
            lut = np.zeros((256, 1), dtype=uint8)
            lut[255] = 1
            lut[64] = 2
            lut[192] = 3
            gray_array = cv2.LUT(gray_array, lut)
            mask_in = gray_array.copy()
            # get our image in a flavor grab cut likes
            npimg = self._ndarray
            # require by opencv
            tmp1 = np.zeros((1, 13 * 5))
            tmp2 = np.zeros((1, 13 * 5))
            # do the algorithm
            cv2.grabCut(npimg, mask_in, None, tmp1, tmp2, 10,
                        mode=cv2.GC_INIT_WITH_MASK)
            # remap the color space
            lut = np.zeros((256, 1), dtype=uint8)
            lut[1] = 255
            lut[2] = 64
            lut[3] = 192
            output = cv2.LUT(mask_in, lut)
            ret_val = Image(output, color_space=ColorSpace.GRAY)

        elif rect is not None:
            npimg = self._ndarray
            tmp1 = np.zeros((1, 13 * 5))
            tmp2 = np.zeros((1, 13 * 5))
            mask = np.zeros((self.height, self.width), dtype=np.uint8)
            cv2.grabCut(npimg, mask, rect, tmp1, tmp2, 10,
                        mode=cv2.GC_INIT_WITH_RECT)
            lut = np.zeros((256, 1), dtype=uint8)
            lut[1] = 255
            lut[2] = 64
            lut[3] = 192
            array = cv2.LUT(mask, lut)
            ret_val = Image(array, color_space=ColorSpace.GRAY)
        else:
            logger.warning("ImageClass.findBlobsSmart requires either a mask "
                           "or a selection rectangle. Failure to provide one "
                           "of these causes your bytes to splinter and bit "
                           "shrapnel to hit your pipeline making it asplode "
                           "in a ball of fire. Okay... not really")
        return ret_val

    def smart_find_blobs(self, mask=None, rect=None, thresh_level=2,
                         appx_level=3):
        """
        **SUMMARY**

        smart_find_blobs uses a method called grabCut, also called graph cut,
        to  automagically determine the boundary of a blob in the image. The
        dumb find blobs just uses color threshold to find the boundary,
        smart_find_blobs looks at both color and edges to find a blob. To work
        smart_find_blobs needs either a rectangle that bounds the object you
        want to find, or a mask. If you use a rectangle make sure it holds the
        complete object. In the case of a mask, it need not be a normal binary
        mask, it can have the normal white foreground and black background, but
        also a light and dark gray values that correspond to areas that are
        more likely to be foreground and more likely to be background. These
        values can be found in the color class as Color.BACKGROUND,
        Color.FOREGROUND, Color.MAYBE_BACKGROUND, and Color.MAYBE_FOREGROUND.

        **PARAMETERS**

        * *mask* - A grayscale mask the same size as the image using the 4 mask
         color values
        * *rect* - A rectangle tuple of the form (x_position, y_position,
         width, height)
        * *thresh_level* - This represents what grab cut values to use in the
         mask after the graph cut algorithm is run,

          * 1  - means use the foreground, maybe_foreground, and
            maybe_background values
          * 2  - means use the foreground and maybe_foreground values.
          * 3+ - means use just the foreground

        * *appx_level* - The blob approximation level - an integer for the
          maximum distance between the true edge and the approximation edge -
          lower numbers yield better approximation.


        **RETURNS**

        A featureset of blobs. If everything went smoothly only a couple of
        blobs should be present.

        **EXAMPLE**

        >>> img = Image("RatTop.png")
        >>> mask = Image((img.width,img.height))
        >>> mask.dl().circle((100, 100), 80, color=Color.MAYBE_BACKGROUND,
            ...              filled=True
        >>> mask.dl().circle((100, 100), 60, color=Color.MAYBE_FOREGROUND,
            ...              filled=True)
        >>> mask.dl().circle((100, 100), 40, color=Color.FOREGROUND,
            ...              filled=True)
        >>> mask = mask.apply_layers()
        >>> blobs = img.smart_find_blobs(mask=mask)
        >>> blobs.draw()
        >>> blobs.show()

        **NOTES**

        http://en.wikipedia.org/wiki/Graph_cuts_in_computer_vision

        **SEE ALSO**

        :py:meth:`smart_threshold`

        """
        result = self.smart_threshold(mask, rect)
        binary = None
        ret_val = None

        if result:
            if thresh_level == 1:
                result = result.threshold(192)
            elif thresh_level == 2:
                result = result.threshold(128)
            elif thresh_level > 2:
                result = result.threshold(1)
            bm = BlobMaker()
            ret_val = bm.extract_from_binary(result, self, appx_level)

        return ret_val

    def threshold(self, value):
        """
        **SUMMARY**

        We roll old school with this vanilla threshold function. It takes your
        image converts it to grayscale, and applies a threshold. Values above
        the threshold are white, values below the threshold are black
        (note this is in contrast to binarize... which is a stupid function
        that drives me up a wall). The resulting black and white image is
        returned.

        **PARAMETERS**

        * *value* - the threshold, goes between 0 and 255.

        **RETURNS**

        A black and white SimpleCV image.

        **EXAMPLE**

        >>> img = Image("purplemonkeydishwasher.png")
        >>> result = img.threshold(42)

        **NOTES**

        THRESHOLD RULES BINARIZE DROOLS!

        **SEE ALSO**

        :py:meth:`binarize`

        """
        gray = self.get_gray_ndarray()
        _, array = cv2.threshold(gray, value, 255, cv2.THRESH_BINARY)
        return Image(array, color_space=ColorSpace.GRAY)

    def flood_fill(self, points, tolerance=None, color=Color.WHITE, lower=None,
                   upper=None, fixed_range=True):
        """
        **SUMMARY**

        FloodFill works just like ye olde paint bucket tool in your favorite
        image manipulation program. You select a point (or a list of points),
        a color, and a tolerance, and flood_fill will start at that point,
        looking for pixels within the tolerance from your intial pixel. If the
        pixel is in tolerance, we will convert it to your color, otherwise the
        method will leave the pixel alone. The method accepts both single
        values, and triplet tuples for the tolerance values. If you require
        more control over your tolerance you can use the upper and lower
        values. The fixed range parameter let's you toggle between setting the
        tolerance with repect to the seed pixel, and using a tolerance that is
        relative to the adjacent pixels. If fixed_range is true the method will
        set its tolerance with respect to the seed pixel, otherwise the
        tolerance will be with repsect to adjacent pixels.

        **PARAMETERS**

        * *points* - A tuple, list of tuples, or np.array of seed points for
         flood fill
        * *tolerance* - The color tolerance as a single value or a triplet.
        * *color* - The color to replace the flood_fill pixels with
        * *lower* - If tolerance does not provide enough control you can
          optionally set the upper and lower values
          around the seed pixel. This value can be a single value or a triplet.
          This will override the tolerance variable.
        * *upper* - If tolerance does not provide enough control you can
          optionally set the upper and lower values around the seed pixel. This
          value can be a single value or a triplet. This will override the
          tolerance variable.
        * *fixed_range* - If fixed_range is true we use the seed_pixel +/-
          tolerance. If fixed_range is false, the tolerance is +/- tolerance of
          the values of the adjacent pixels to the pixel under test.

        **RETURNS**

        An Image where the values similar to the seed pixel have been replaced
        by the input color.

        **EXAMPLE**

        >>> img = Image("lenna")
        >>> img2 = img.flood_fill(((10, 10), (54, 32)), tolerance=(10, 10, 10),
        ...                       color=Color.RED)
        >>> img2.show()

        **SEE ALSO**

        :py:meth:`flood_fill_to_mask`
        :py:meth:`find_flood_fill_blobs`

        """
        if isinstance(color, np.ndarray):
            color = color.tolist()
        elif isinstance(color, dict):
            color = (color['R'], color['G'], color['B'])

        if isinstance(points, tuple):
            points = np.array(points)
        # first we guess what the user wants to do
        # if we get and int/float convert it to a tuple
        if upper is None and lower is None and tolerance is None:
            upper = (0, 0, 0)
            lower = (0, 0, 0)

        if tolerance is not None and isinstance(tolerance, (float, int)):
            tolerance = (int(tolerance), int(tolerance), int(tolerance))

        if lower is not None and isinstance(lower, (float, int)):
            lower = (int(lower), int(lower), int(lower))
        elif lower is None:
            lower = tolerance

        if upper is not None and isinstance(upper, (float, int)):
            upper = (int(upper), int(upper), int(upper))
        elif upper is None:
            upper = tolerance

        if isinstance(points, tuple):
            points = np.array(points)

        flags = cv2.FLOODFILL_MASK_ONLY
        if fixed_range:
            flags |= cv2.FLOODFILL_FIXED_RANGE

        mask = np.zeros((self.height + 2, self.width + 2), dtype=np.uint8)
        array = self._ndarray.copy()

        if len(points.shape) != 1:
            for p in points:
                cv2.floodFill(array, mask, tuple(p),
                              color, lower, upper, flags)
        else:
            cv2.floodFill(array, mask, tuple(points),
                          color, lower, upper, flags)
        return Image(array)

    def flood_fill_to_mask(self, points, tolerance=None, color=Color.WHITE,
                           lower=None, upper=None, fixed_range=True,
                           mask=None):
        """
        **SUMMARY**

        flood_fill_to_mask works sorta paint bucket tool in your favorite image
        manipulation program. You select a point (or a list of points), a
        color, and a tolerance, and flood_fill will start at that point,
        looking for pixels within the tolerance from your intial pixel. If the
        pixel is in tolerance, we will convert it to your color, otherwise the
        method will leave the pixel alone. Unlike regular flood_fill,
        flood_fill_to_mask, will return a binary mask of your flood fill
        operation. This is handy if you want to extract blobs from an area, or
        create a selection from a region. The method takes in an optional mask.
        Non-zero values of the mask act to block the flood fill operations.
        This is handy if you want to use an edge image to "stop" the flood fill
        operation within a particular region.

        The method accepts both single values, and triplet tuples for the
        tolerance values. If you require more control over your tolerance you
        can use the upper and lower values. The fixed range parameter let's you
        toggle between setting the tolerance with repect to the seed pixel, and
        using a tolerance that is relative to the adjacent pixels. If
        fixed_range is true the method will set its tolerance with respect to
        the seed pixel, otherwise the tolerance will be with repsect to
        adjacent pixels.

        **PARAMETERS**

        * *points* - A tuple, list of tuples, or np.array of seed points for
          flood fill
        * *tolerance* - The color tolerance as a single value or a triplet.
        * *color* - The color to replace the flood_fill pixels with
        * *lower* - If tolerance does not provide enough control you can
          optionally set the upper and lower values around the seed pixel. This
          value can be a single value or a triplet. This will override
          the tolerance variable.
        * *upper* - If tolerance does not provide enough control you can
          optionally set the upper and lower values around the seed pixel. This
          value can be a single value or a triplet. This will override
          the tolerance variable.
        * *fixed_range* - If fixed_range is true we use the seed_pixel +/-
          tolerance. If fixed_range is false, the tolerance is +/- tolerance of
          the values of the adjacent pixels to the pixel under test.
        * *mask* - An optional mask image that can be used to control the flood
          fill operation. the output of this function will include the mask
          data in the input mask.

        **RETURNS**

        An Image where the values similar to the seed pixel have been replaced
        by the input color.

        **EXAMPLE**

        >>> img = Image("lenna")
        >>> mask = img.edges()
        >>> mask= img.flood_fill_to_mask(((10, 10), (54, 32)),
        ...                              tolerance=(10, 10, 10), mask=mask)
        >>> mask.show

        **SEE ALSO**

        :py:meth:`flood_fill`
        :py:meth:`find_flood_fill_blobs`

        """
        if isinstance(color, np.ndarray):
            color = color.tolist()
        elif isinstance(color, dict):
            color = (color['R'], color['G'], color['B'])

        if isinstance(points, tuple):
            points = np.array(points)

        # first we guess what the user wants to do
        # if we get and int/float convert it to a tuple
        if upper is None and lower is None and tolerance is None:
            upper = (0, 0, 0)
            lower = (0, 0, 0)

        if tolerance is not None and isinstance(tolerance, (float, int)):
            tolerance = (int(tolerance), int(tolerance), int(tolerance))

        if lower is not None and isinstance(lower, (float, int)):
            lower = (int(lower), int(lower), int(lower))
        elif lower is None:
            lower = tolerance

        if upper is not None and isinstance(upper, (float, int)):
            upper = (int(upper), int(upper), int(upper))
        elif upper is None:
            upper = tolerance

        if isinstance(points, tuple):
            points = np.array(points)

        flags = cv2.FLOODFILL_MASK_ONLY
        if fixed_range:
            flags |= cv2.FLOODFILL_FIXED_RANGE

        #opencv wants a mask that is slightly larger
        if mask is None:
            local_mask = np.zeros((self.height + 2, self.width + 2),
                                  dtype=np.uint8)
        else:
            local_mask = mask.embiggen(size=(self.width + 2, self.height + 2))
            local_mask = local_mask.get_gray_ndarray()

        temp = self.get_gray_ndarray()
        if len(points.shape) != 1:
            for p in points:
                cv2.floodFill(temp, local_mask, tuple(p), color,
                              lower, upper, flags)
        else:
            cv2.floodFill(temp, local_mask, tuple(points), color,
                          lower, upper, flags)

        ret_val = Image(local_mask)
        ret_val = ret_val.crop(1, 1, self.width, self.height)
        return ret_val

    def find_blobs_from_mask(self, mask, threshold=128, minsize=10, maxsize=0,
                             appx_level=3):
        """
        **SUMMARY**

        This method acts like find_blobs, but it lets you specifiy blobs
        directly by providing a mask image. The mask image must match the size
        of this image, and the mask should have values > threshold where you
        want the blobs selected. This method can be used with binarize, dialte,
        erode, flood_fill, edges etc to get really nice segmentation.

        **PARAMETERS**

        * *mask* - The mask image, areas lighter than threshold will be counted
          as blobs. Mask should be the same size as this image.
        * *threshold* - A single threshold value used when we binarize the
          mask.
        * *minsize* - The minimum size of the returned blobs.
        * *maxsize*  - The maximum size of the returned blobs, if none is
          specified we peg this to the image size.
        * *appx_level* - The blob approximation level - an integer for the
          maximum distance between the true edge and the approximation edge -
          lower numbers yield better approximation.


        **RETURNS**

        A featureset of blobs. If no blobs are found None is returned.

        **EXAMPLE**

        >>> img = Image("Foo.png")
        >>> mask = img.binarize().dilate(2)
        >>> blobs = img.find_blobs_from_mask(mask)
        >>> blobs.show()

        **SEE ALSO**

        :py:meth:`find_blobs`
        :py:meth:`binarize`
        :py:meth:`threshold`
        :py:meth:`dilate`
        :py:meth:`erode`
        """
        if maxsize == 0:
            maxsize = self.width * self.height
        #create a single channel image, thresholded to parameters
        if mask.size() != self.size():
            logger.warning("Image.find_blobs_from_mask - your mask does "
                           "not match the size of your image")
            return None

        blobmaker = BlobMaker()
        gray = mask.get_gray_ndarray()
        val, result = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        blobs = blobmaker.extract_from_binary(
            Image(result, color_space=ColorSpace.GRAY), self,
            minsize=minsize, maxsize=maxsize, appx_level=appx_level)
        if not len(blobs):
            return None
        return FeatureSet(blobs).sort_area()

    def find_flood_fill_blobs(self, points, tolerance=None, lower=None,
                              upper=None,
                              fixed_range=True, minsize=30, maxsize=-1):
        """

        **SUMMARY**

        This method lets you use a flood fill operation and pipe the results to
        find_blobs. You provide the points to seed flood_fill and the rest is
        taken care of.

        flood_fill works just like ye olde paint bucket tool in your favorite
        image manipulation program. You select a point (or a list of points),
        a color, and a tolerance, and flood_fill will start at that point,
        looking for pixels within the tolerance from your intial pixel. If the
        pixel is in tolerance, we will convert it to your color, otherwise the
        method will leave the pixel alone. The method accepts both single
        values, and triplet tuples for the tolerance values. If you require
        more control over your tolerance you can use the upper and lower
        values. The fixed range parameter let's you toggle between setting the
        tolerance with repect to the seed pixel, and using a tolerance that is
        relative to the adjacent pixels. If fixed_range is true the method will
        set its tolerance with respect to the seed pixel, otherwise the
        tolerance will be with repsect to adjacent pixels.

        **PARAMETERS**

        * *points* - A tuple, list of tuples, or np.array of seed points for
          flood fill.
        * *tolerance* - The color tolerance as a single value or a triplet.
        * *color* - The color to replace the flood_fill pixels with
        * *lower* - If tolerance does not provide enough control you can
          optionally set the upper and lower values around the seed pixel.
          This value can be a single value or a triplet. This will override
          the tolerance variable.
        * *upper* - If tolerance does not provide enough control you can
          optionally set the upper and lower values around the seed pixel.
           This value can be a single value or a triplet. This will override
          the tolerance variable.
        * *fixed_range* - If fixed_range is true we use the seed_pixel +/-
          tolerance. If fixed_range is false, the tolerance is +/- tolerance
          of the values of the adjacent pixels to the pixel under test.
        * *minsize* - The minimum size of the returned blobs.
        * *maxsize* - The maximum size of the returned blobs, if none is
          specified we peg this to the image size.

        **RETURNS**

        A featureset of blobs. If no blobs are found None is returned.

        An Image where the values similar to the seed pixel have been replaced
        by the input color.

        **EXAMPLE**

        >>> img = Image("lenna")
        >>> blerbs = img.find_flood_fill_blobs(((10, 10), (20, 20), (30, 30)),
            ...                             tolerance=30)
        >>> blerbs.show()

        **SEE ALSO**

        :py:meth:`find_blobs`
        :py:meth:`flood_fill`

        """
        mask = self.flood_fill_to_mask(points, tolerance, color=Color.WHITE,
                                       lower=lower, upper=upper,
                                       fixed_range=fixed_range)
        return self.find_blobs_from_mask(mask, minsize, maxsize)

    def _do_dft(self, grayscale=False):
        """
        **SUMMARY**

        This private method peforms the discrete Fourier transform on an input
        image. The transform can be applied to a single channel gray image or
        to each channel of the image. Each channel generates a 64F 2 channel
        IPL image corresponding to the real and imaginary components of the
        DFT. A list of these IPL images are then cached in the private member
        variable _DFT.


        **PARAMETERS**

        * *grayscale* - If grayscale is True we first covert the image to
          grayscale, otherwise we perform the operation on each channel.

        **RETURNS**

        nothing - but creates a locally cached list of IPL imgaes corresponding
        to the real and imaginary components of each channel.

        **EXAMPLE**

        >>> img = Image('logo.png')
        >>> img._do_dft()
        >>> img._DFT[0] # get the b channel Re/Im components

        **NOTES**

        http://en.wikipedia.org/wiki/Discrete_Fourier_transform
        http://math.stackexchange.com/questions/1002/
        fourier-transform-for-dummies

        **TO DO**

        This method really needs to convert the image to an optimal DFT size.
        http://opencv.itseez.com/modules/core/doc/
        operations_on_arrays.html#getoptimaldftsize

        """
        width, height = self.size()
        if grayscale and (len(self._DFT) == 0 or len(self._DFT) == 3):
            self._DFT = []
            img = self.get_gray_ndarray()
            data = img.astype(np.float64)
            blank = np.zeros((height, width))
            src = np.dstack((data, blank))
            dst = cv2.dft(src)
            self._DFT.append(dst)
        elif not grayscale and len(self._DFT) < 2:
            width, height = self.size()
            self._DFT = []
            img = self._ndarray.copy()
            b = img[:, :, 0]
            g = img[:, :, 1]
            r = img[:, :, 2]
            chanels = [b, g, r]
            for c in chanels:
                data = c.astype(np.float64)
                blank = np.zeros((height, width))
                src = np.dstack((data, blank))
                dst = cv2.dft(src)
                self._DFT.append(dst)

    def _get_dft_clone(self, grayscale=False):
        """
        **SUMMARY**

        This method works just like _do_dft but returns a deep copy
        of the resulting array which can be used in destructive operations.

        **PARAMETERS**

        * *grayscale* - If grayscale is True we first covert the image to
          grayscale, otherwise we perform the operation on each channel.

        **RETURNS**

        A deep copy of the cached DFT real/imaginary image list.

        **EXAMPLE**

        >>> img = Image('logo.png')
        >>> myDFT = img._get_dft_clone()
        >>> SomeCVFunc(myDFT[0])

        **NOTES**

        http://en.wikipedia.org/wiki/Discrete_Fourier_transform
        http://math.stackexchange.com/questions/1002/
        fourier-transform-for-dummies

        **SEE ALSO**

        ImageClass._do_dft()

        """
        # this is needs to be switched to the optimal
        # DFT size for faster processing.
        self._do_dft(grayscale)
        ret_val = []
        if grayscale:
            ret_val.append(self._DFT[0].copy())
        else:
            for img in self._DFT:
                ret_val.append(img.copy())
        return ret_val

    def raw_dft_image(self, grayscale=False):
        """
        **SUMMARY**

        This method returns the **RAW** DFT transform of an image as a list of
        IPL Images. Each result image is a two channel 64f image where the
        irst channel is the real component and the second channel is the
        imaginary component. If the operation is performed on an RGB image and
        grayscale is False the result is a list of these images of the form
        [b, g, r].

        **PARAMETERS**

        * *grayscale* - If grayscale is True we first covert the image to
          grayscale, otherwise we perform the operation on each channel.

        **RETURNS**

        A list of the DFT images (see above). Note that this is a shallow copy
        operation.

        **EXAMPLE**

        >>> img = Image('logo.png')
        >>> myDFT = img.raw_dft_image()
        >>> for c in myDFT:
        >>>    #do some operation on the DFT

        **NOTES**

        http://en.wikipedia.org/wiki/Discrete_Fourier_transform
        http://math.stackexchange.com/questions/1002/
        fourier-transform-for-dummies

        **SEE ALSO**

        :py:meth:`raw_dft_image`
        :py:meth:`get_dft_log_magnitude`
        :py:meth:`apply_dft_filter`
        :py:meth:`high_pass_filter`
        :py:meth:`low_pass_filter`
        :py:meth:`band_pass_filter`
        :py:meth:`inverse_dft`
        :py:meth:`apply_butterworth_filter`
        :py:meth:`inverse_dft`
        :py:meth:`apply_gaussian_filter`
        :py:meth:`apply_unsharp_mask`

        """
        self._do_dft(grayscale)
        return self._DFT

    def get_dft_log_magnitude(self, grayscale=False):
        """
        **SUMMARY**

        This method returns the log value of the magnitude image of the DFT
        transform. This method is helpful for examining and comparing the
        results of DFT transforms. The log component helps to "squish" the
        large floating point values into an image that can be rendered easily.

        In the image the low frequency components are in the corners of the
        image and the high frequency components are in the center of the image.

        **PARAMETERS**

        * *grayscale* - if grayscale is True we perform the magnitude operation
           of the grayscale image otherwise we perform the operation on each
           channel.

        **RETURNS**

        Returns a SimpleCV image corresponding to the log magnitude of the
        input image.

        **EXAMPLE**

        >>> img = Image("RedDog2.jpg")
        >>> img.get_dft_log_magnitude().show()
        >>> lpf = img.low_pass_filter(img.width/10.img.height/10)
        >>> lpf.get_dft_log_magnitude().show()

        **NOTES**

        * http://en.wikipedia.org/wiki/Discrete_Fourier_transform
        * http://math.stackexchange.com/questions/1002/
          fourier-transform-for-dummies

        **SEE ALSO**

        :py:meth:`raw_dft_image`
        :py:meth:`get_dft_log_magnitude`
        :py:meth:`apply_dft_filter`

        :py:meth:`high_pass_filter`
        :py:meth:`low_pass_filter`
        :py:meth:`band_pass_filter`
        :py:meth:`inverse_dft`
        :py:meth:`apply_butterworth_filter`
        :py:meth:`inverse_dft`
        :py:meth:`apply_gaussian_filter`
        :py:meth:`apply_unsharp_mask`


        """
        dft = self._get_dft_clone(grayscale)
        if grayscale:
            chans = [self.get_empty(1)]
        else:
            chans = [self.get_empty(1), self.get_empty(1), self.get_empty(1)]

        for i in range(0, len(chans)):
            data = dft[i][:, :, 0]
            blank = dft[i][:, :, 1]
            data = cv2.pow(data, 2.0)
            blank = cv2.pow(blank, 2.0)
            data += blank
            data = cv2.pow(data, 0.5)
            data += 1  # 1 + Mag
            data = cv2.log(data)  # log(1 + Mag)
            min_val, max_val, pt1, pt2 = cv2.minMaxLoc(data)
            denom = max_val - min_val
            if denom == 0:
                denom = 1
            data = data / denom - min_val / denom  # scale
            data = cv2.multiply(data, data, scale=255.0)
            chans[i] = np.copy(data).astype(self.dtype)
        if grayscale:
            ret_val = Image(chans[0], color_space=ColorSpace.GRAY)
        else:
            ret_val = Image(np.dstack(tuple(chans)))
        return ret_val

    def _bounds_from_percentage2(self, float_val, bound):
        return np.clip(int(float_val * bound), 0, bound)

    def _bounds_from_percentage(self, float_val, bound):
        return np.clip(int(float_val * (bound / 2.00)), 0, (bound / 2))

    def apply_dft_filter(self, flt, grayscale=False):
        """
        **SUMMARY**

        This function allows you to apply an arbitrary filter to the DFT of an
        image. This filter takes in a gray scale image, whiter values are kept
        and black values are rejected. In the DFT image, the lower frequency
        values are in the corners of the image, while the higher frequency
        components are in the center. For example, a low pass filter has white
        squares in the corners and is black everywhere else.

        **PARAMETERS**

        * *grayscale* - if this value is True we perfrom the operation on the
          DFT of the gray version of the image and the result is gray image.
          If grayscale is true we perform the operation on each channel and the
          recombine them to create the result.

        * *flt* - A grayscale filter image. The size of the filter must match
          the size of the image.

        **RETURNS**

        A SimpleCV image after applying the filter.

        **EXAMPLE**

        >>>  filter = Image("MyFilter.png")
        >>>  myImage = Image("MyImage.png")
        >>>  result = myImage.apply_dft_filter(filter)
        >>>  result.show()

        **SEE ALSO**

        :py:meth:`raw_dft_image`
        :py:meth:`get_dft_log_magnitude`
        :py:meth:`apply_dft_filter`
        :py:meth:`high_pass_filter`
        :py:meth:`low_pass_filter`
        :py:meth:`band_pass_filter`
        :py:meth:`inverse_dft`
        :py:meth:`apply_butterworth_filter`
        :py:meth:`inverse_dft`
        :py:meth:`apply_gaussian_filter`
        :py:meth:`apply_unsharp_mask`

        **TODO**

        Make this function support a separate filter image for each channel.
        """
        if isinstance(flt, DFT):
            filteredimage = flt.apply_filter(self, grayscale)
            return filteredimage

        if flt.size() != self.size():
            logger.warning("Image.apply_dft_filter - Your filter must match "
                           "the size of the image")
        dft = []
        if grayscale:
            dft = self._get_dft_clone(grayscale)
            flt64f = flt.get_gray_ndarray().astype(np.float64)
            final_filt = np.dstack((flt64f, flt64f))
            for i in range(len(dft)):
                dft[i] = cv2.mulSpectrums(dft[i], final_filt, 0)
        else:  # break down the filter and then do each channel
            dft = self._get_dft_clone(grayscale)
            flt = flt.get_ndarray()
            b = flt[:, :, 0]
            g = flt[:, :, 1]
            r = flt[:, :, 2]
            chans = [b, g, r]
            for i in range(0, len(chans)):
                flt64f = np.copy(chans[i])
                final_filt = np.dstack((flt64f, flt64f))
                if dft[i].dtype != final_filt.dtype:
                    final_filt = final_filt.astype(dft[i].dtype)
                dft[i] = cv2.mulSpectrums(dft[i], final_filt, 0)
        return self._inverse_dft(dft)

    def high_pass_filter(self, x_cutoff, y_cutoff=None, grayscale=False):
        """
        **SUMMARY**

        This method applies a high pass DFT filter. This filter enhances
        the high frequencies and removes the low frequency signals. This has
        the effect of enhancing edges. The frequencies are defined as going
        between 0.00 and 1.00 and where 0 is the lowest frequency in the image
        and 1.0 is the highest possible frequencies. Each of the frequencies
        are defined with respect to the horizontal and vertical signal. This
        filter isn't perfect and has a harsh cutoff that causes ringing
        artifacts.

        **PARAMETERS**

        * *x_cutoff* - The horizontal frequency at which we perform the cutoff.
          A separate frequency can be used for the b,g, and r signals by
          providing list of values. The frequency is defined between zero to
          one, where zero is constant component and 1 is the highest possible
          frequency in the image.

        * *y_cutoff* - The cutoff frequencies in the y direction. If none are
          provided we use the same values as provided for x.

        * *grayscale* - if this value is True we perfrom the operation on the
          DFT of the gray version of the image and the result is gray image.
          If grayscale is true we perform the operation on each channel and
          the recombine them to create the result.

        **RETURNS**

        A SimpleCV Image after applying the filter.

        **EXAMPLE**

        >>> img = Image("SimpleCV/data/sampleimages/RedDog2.jpg")
        >>> img.get_dft_log_magnitude().show()
        >>> hpf = img.high_pass_filter([0.2, 0.1, 0.2])
        >>> hpf.show()
        >>> hpf.get_dft_log_magnitude().show()

        **NOTES**

        This filter is far from perfect and will generate a lot of ringing
        artifacts.

        * See: http://en.wikipedia.org/wiki/Ringing_(signal)
        * See: http://en.wikipedia.org/wiki/High-pass_filter#Image

        **SEE ALSO**

        :py:meth:`raw_dft_image`
        :py:meth:`get_dft_log_magnitude`
        :py:meth:`apply_dft_filter`
        :py:meth:`high_pass_filter`
        :py:meth:`low_pass_filter`
        :py:meth:`band_pass_filter`
        :py:meth:`inverse_dft`
        :py:meth:`apply_butterworth_filter`
        :py:meth:`inverse_dft`
        :py:meth:`apply_gaussian_filter`
        :py:meth:`apply_unsharp_mask`

        """
        if isinstance(x_cutoff, float):
            x_cutoff = [x_cutoff, x_cutoff, x_cutoff]
        if isinstance(y_cutoff, float):
            y_cutoff = [y_cutoff, y_cutoff, y_cutoff]
        if y_cutoff is None:
            y_cutoff = [x_cutoff[0], x_cutoff[1], x_cutoff[2]]

        for i in range(0, len(x_cutoff)):
            x_cutoff[i] = self._bounds_from_percentage(x_cutoff[i], self.width)
            y_cutoff[i] = self._bounds_from_percentage(y_cutoff[i],
                                                       self.height)

        filter = None
        h = self.height
        w = self.width

        if grayscale:
            filter = self.get_empty(1)
            filter += 255  # make everything white

            # now make all of the corners black
            cv2.rectangle(filter, (0, 0), (x_cutoff[0], y_cutoff[0]),
                          0, thickness=-1)  # TL
            cv2.rectangle(filter, (0, h - y_cutoff[0]), (x_cutoff[0], h),
                          0, thickness=-1)  # BL
            cv2.rectangle(filter, (w - x_cutoff[0], 0), (w, y_cutoff[0]),
                          0, thickness=-1)  # TR
            cv2.rectangle(filter, (w - x_cutoff[0], h - y_cutoff[0]), (w, h),
                          0, thickness=-1)  # BR

            scv_filt = Image(filter, color_space=ColorSpace.GRAY)
        else:
            # I need to looking into CVMERGE/SPLIT... I would really
            # need to know how much memory we're allocating here
            filter_b = self.get_empty(1) + 255  # make everything white
            filter_g = self.get_empty(1) + 255  # make everything white
            filter_r = self.get_empty(1) + 255  # make everything white

            # now make all of the corners black
            temp = [filter_b, filter_g, filter_r]
            i = 0
            for f in temp:
                cv2.rectangle(f, (0, 0), (x_cutoff[i], y_cutoff[i]), 0,
                              thickness=-1)
                cv2.rectangle(f, (0, h - y_cutoff[i]), (x_cutoff[i], h), 0,
                              thickness=-1)
                cv2.rectangle(f, (w - x_cutoff[i], 0), (w, y_cutoff[i]), 0,
                              thickness=-1)
                cv2.rectangle(f, (w - x_cutoff[i], h - y_cutoff[i]), (w, h), 0,
                              thickness=-1)
                i = i + 1

            filter = np.dstack(tuple(temp))
            scv_filt = Image(filter, color_space=ColorSpace.BGR)

        return self.apply_dft_filter(scv_filt, grayscale)

    def low_pass_filter(self, x_cutoff, y_cutoff=None, grayscale=False):
        """
        **SUMMARY**

        This method applies a low pass DFT filter. This filter enhances
        the low frequencies and removes the high frequency signals. This has
        the effect of reducing noise. The frequencies are defined as going
        between 0.00 and 1.00 and where 0 is the lowest frequency in the image
        and 1.0 is the highest possible frequencies. Each of the frequencies
        are defined with respect to the horizontal and vertical signal. This
        filter isn't perfect and has a harsh cutoff that causes ringing
        artifacts.

        **PARAMETERS**

        * *x_cutoff* - The horizontal frequency at which we perform the cutoff.
          A separate frequency can be used for the b,g, and r signals by
          providing a list of values. The frequency is defined between zero to
          one, where zero is constant component and 1 is the highest possible
          frequency in the image.

        * *y_cutoff* - The cutoff frequencies in the y direction. If none are
          provided we use the same values as provided for x.

        * *grayscale* - if this value is True we perfrom the operation on the
          DFT of the gray version of the image and the result is gray image.
          If grayscale is true we perform the operation on each channel and the
          recombine them to create the result.

        **RETURNS**

        A SimpleCV Image after applying the filter.

        **EXAMPLE**

        >>> img = Image("SimpleCV/data/sampleimages/RedDog2.jpg")
        >>> img.get_dft_log_magnitude().show()
        >>> lpf = img.low_pass_filter([0.2, 0.2, 0.05])
        >>> lpf.show()
        >>> lpf.get_dft_log_magnitude().show()

        **NOTES**

        This filter is far from perfect and will generate a lot of ringing
        artifacts.

        See: http://en.wikipedia.org/wiki/Ringing_(signal)
        See: http://en.wikipedia.org/wiki/Low-pass_filter

        **SEE ALSO**

        :py:meth:`raw_dft_image`
        :py:meth:`get_dft_log_magnitude`
        :py:meth:`apply_dft_filter`
        :py:meth:`high_pass_filter`
        :py:meth:`low_pass_filter`
        :py:meth:`band_pass_filter`
        :py:meth:`inverse_dft`
        :py:meth:`apply_butterworth_filter`
        :py:meth:`inverse_dft`
        :py:meth:`apply_gaussian_filter`
        :py:meth:`apply_unsharp_mask`

        """
        if isinstance(x_cutoff, float):
            x_cutoff = [x_cutoff, x_cutoff, x_cutoff]
        if isinstance(y_cutoff, float):
            y_cutoff = [y_cutoff, y_cutoff, y_cutoff]
        if y_cutoff is None:
            y_cutoff = [x_cutoff[0], x_cutoff[1], x_cutoff[2]]

        for i in range(0, len(x_cutoff)):
            x_cutoff[i] = self._bounds_from_percentage(x_cutoff[i], self.width)
            y_cutoff[i] = self._bounds_from_percentage(y_cutoff[i],
                                                       self.height)

        filter = None
        h = self.height
        w = self.width

        if grayscale:
            filter = self.get_empty(1)

            #now make all of the corners white
            cv2.rectangle(filter, (0, 0), (x_cutoff[0], y_cutoff[0]), 255,
                          thickness=-1)  # TL
            cv2.rectangle(filter, (0, h - y_cutoff[0]), (x_cutoff[0], h), 255,
                          thickness=-1)  # BL
            cv2.rectangle(filter, (w - x_cutoff[0], 0), (w, y_cutoff[0]), 255,
                          thickness=-1)  # TR
            cv2.rectangle(filter, (w - x_cutoff[0], h - y_cutoff[0]), (w, h),
                          255, thickness=-1)  # BR
            scv_filt = Image(filter, color_space=ColorSpace.GRAY)

        else:
            # I need to looking into CVMERGE/SPLIT... I would really need
            # to know how much memory we're allocating here
            filter_b = self.get_empty(1)
            filter_g = self.get_empty(1)
            filter_r = self.get_empty(1)

            # now make all of the corners white
            temp = [filter_b, filter_g, filter_r]
            i = 0
            for f in temp:
                cv2.rectangle(f, (0, 0), (x_cutoff[i], y_cutoff[i]), 255,
                              thickness=-1)
                cv2.rectangle(f, (0, h - y_cutoff[i]), (x_cutoff[i], h), 255,
                              thickness=-1)
                cv2.rectangle(f, (w - x_cutoff[i], 0), (w, y_cutoff[i]), 255,
                              thickness=-1)
                cv2.rectangle(f, (w - x_cutoff[i], h - y_cutoff[i]), (w, h),
                              255, thickness=-1)
                i = i + 1

            filter = np.dstack(tuple(temp))
            scv_filt = Image(filter, color_space=ColorSpace.BGR)

        return self.apply_dft_filter(scv_filt, grayscale)

    #FIXME: need to decide BGR or RGB
    # ((rx_begin,ry_begin)(gx_begin,gy_begin)(bx_begin,by_begin))
    # or (x,y)
    def band_pass_filter(self, x_cutoff_low, x_cutoff_high, y_cutoff_low=None,
                         y_cutoff_high=None, grayscale=False):
        """
        **SUMMARY**

        This method applies a simple band pass DFT filter. This filter enhances
        the a range of frequencies and removes all of the other frequencies.
        This allows a user to precisely select a set of signals to display.
        The frequencies are defined as going between 0.00 and 1.00 and where 0
        is the lowest frequency in the image and 1.0 is the highest possible
        frequencies. Each of the frequencies are defined with respect to the
        horizontal and vertical signal. This filter isn't perfect and has a
        harsh cutoff that causes ringing artifacts.

        **PARAMETERS**

        * *x_cutoff_low*  - The horizontal frequency at which we perform the
          cutoff of the low frequency signals. A separate frequency can be used
          for the b,g, and r signals by providing a list of values. The
          frequency is defined between zero to one, where zero is constant
          component and 1 is the highest possible frequency in the image.

        * *x_cutoff_high* - The horizontal frequency at which we perform the
          cutoff of the high frequency signals. Our filter passes signals
          between x_cutoff_low and x_cutoff_high. A separate frequency can be
          used for the b, g, and r channels by providing a list of values. The
          frequency is defined between zero to one, where zero is constant
          component and 1 is the highest possible frequency in the image.

        * *y_cutoff_low* - The low frequency cutoff in the y direction. If none
          are provided we use the same values as provided for x.

        * *y_cutoff_high* - The high frequency cutoff in the y direction. If
          none are provided we use the same values as provided for x.

        * *grayscale* - if this value is True we perfrom the operation on the
          DFT of the gray version of the image and the result is gray image.
          If grayscale is true we perform the operation on each channel and
          the recombine them to create the result.

        **RETURNS**

        A SimpleCV Image after applying the filter.

        **EXAMPLE**

        >>> img = Image("SimpleCV/data/sampleimages/RedDog2.jpg")
        >>> img.get_dft_log_magnitude().show()
        >>> lpf = img.band_pass_filter([0.2, 0.2, 0.05],[0.3, 0.3, 0.2])
        >>> lpf.show()
        >>> lpf.get_dft_log_magnitude().show()

        **NOTES**

        This filter is far from perfect and will generate a lot of ringing
        artifacts.

        See: http://en.wikipedia.org/wiki/Ringing_(signal)

        **SEE ALSO**

        :py:meth:`raw_dft_image`
        :py:meth:`get_dft_log_magnitude`
        :py:meth:`apply_dft_filter`
        :py:meth:`high_pass_filter`
        :py:meth:`low_pass_filter`
        :py:meth:`band_pass_filter`
        :py:meth:`inverse_dft`
        :py:meth:`apply_butterworth_filter`
        :py:meth:`inverse_dft`
        :py:meth:`apply_gaussian_filter`
        :py:meth:`apply_unsharp_mask`

        """

        if isinstance(x_cutoff_low, float):
            x_cutoff_low = [x_cutoff_low, x_cutoff_low, x_cutoff_low]
        if isinstance(y_cutoff_low, float):
            y_cutoff_low = [y_cutoff_low, y_cutoff_low, y_cutoff_low]
        if isinstance(x_cutoff_high, float):
            x_cutoff_high = [x_cutoff_high, x_cutoff_high, x_cutoff_high]
        if isinstance(y_cutoff_high, float):
            y_cutoff_high = [y_cutoff_high, y_cutoff_high, y_cutoff_high]

        if y_cutoff_low is None:
            y_cutoff_low = [x_cutoff_low[0], x_cutoff_low[1], x_cutoff_low[2]]
        if y_cutoff_high is None:
            y_cutoff_high = [x_cutoff_high[0], x_cutoff_high[1],
                             x_cutoff_high[2]]

        for i in range(0, len(x_cutoff_low)):
            x_cutoff_low[i] = self._bounds_from_percentage(x_cutoff_low[i],
                                                           self.width)
            x_cutoff_high[i] = self._bounds_from_percentage(x_cutoff_high[i],
                                                            self.width)
            y_cutoff_high[i] = self._bounds_from_percentage(y_cutoff_high[i],
                                                            self.height)
            y_cutoff_low[i] = self._bounds_from_percentage(y_cutoff_low[i],
                                                           self.height)

        filter = None
        h = self.height
        w = self.width
        if grayscale:
            filter = self.get_empty(1)

            # now make all of the corners white
            cv2.rectangle(filter, (0, 0), (x_cutoff_high[0], y_cutoff_high[0]),
                          255, thickness=-1)  # TL
            cv2.rectangle(filter, (0, h - y_cutoff_high[0]),
                          (x_cutoff_high[0], h), 255, thickness=-1)  # BL
            cv2.rectangle(filter, (w - x_cutoff_high[0], 0),
                          (w, y_cutoff_high[0]), 255, thickness=-1)  # TR
            cv2.rectangle(filter, (w - x_cutoff_high[0], h - y_cutoff_high[0]),
                          (w, h), 255, thickness=-1)  # BR
            cv2.rectangle(filter, (0, 0), (x_cutoff_low[0], y_cutoff_low[0]),
                          0, thickness=-1)  # TL
            cv2.rectangle(filter, (0, h - y_cutoff_low[0]),
                          (x_cutoff_low[0], h), 0, thickness=-1)  # BL
            cv2.rectangle(filter, (w - x_cutoff_low[0], 0),
                          (w, y_cutoff_low[0]), 0, thickness=-1)  # TR
            cv2.rectangle(filter, (w - x_cutoff_low[0], h - y_cutoff_low[0]),
                          (w, h), 0, thickness=-1)  # BR
            scv_filt = Image(filter, color_space=ColorSpace.GRAY)

        else:
            # I need to looking into CVMERGE/SPLIT... I would really need
            # to know how much memory we're allocating here
            filter_b = self.get_empty(1)
            filter_g = self.get_empty(1)
            filter_r = self.get_empty(1)

            #now make all of the corners black
            temp = [filter_b, filter_g, filter_r]
            i = 0
            for f in temp:
                cv2.rectangle(f, (0, 0), (x_cutoff_high[i], y_cutoff_high[i]),
                              255, thickness=-1)  # TL
                cv2.rectangle(f, (0, h - y_cutoff_high[i]),
                              (x_cutoff_high[i], h), 255, thickness=-1)  # BL
                cv2.rectangle(f, (w - x_cutoff_high[i], 0),
                              (w, y_cutoff_high[i]), 255, thickness=-1)  # TR
                cv2.rectangle(f, (w - x_cutoff_high[i], h - y_cutoff_high[i]),
                              (w, h), 255, thickness=-1)  # BR
                cv2.rectangle(f, (0, 0), (x_cutoff_low[i], y_cutoff_low[i]), 0,
                              thickness=-1)  # TL
                cv2.rectangle(f, (0, h - y_cutoff_low[i]),
                              (x_cutoff_low[i], h), 0, thickness=-1)  # BL
                cv2.rectangle(f, (w - x_cutoff_low[i], 0),
                              (w, y_cutoff_low[i]), 0, thickness=-1)  # TR
                cv2.rectangle(f, (w - x_cutoff_low[i], h - y_cutoff_low[i]),
                              (w, h), 0, thickness=-1)  # BR
                i = i + 1

            filter = np.dstack(tuple(temp))
            scv_filt = Image(filter, color_space=ColorSpace.BGR)

        return self.apply_dft_filter(scv_filt, grayscale)

    def _inverse_dft(self, input):
        """
        **SUMMARY**
        **PARAMETERS**
        **RETURNS**
        **EXAMPLE**
        NOTES:
        SEE ALSO:
        """
        # a destructive IDFT operation for internal calls
        if len(input) == 1:
            dftimg = cv2.dft(input[0], flags=cv2.DFT_INVERSE)
            data = dftimg[:, :, 0].copy()
            min, max, pt1, pt2 = cv2.minMaxLoc(data)
            denom = max - min
            if denom == 0:
                denom = 1
            data = data / denom - min / denom
            data = cv2.multiply(data, data, scale=255.0)
            result = np.copy(data).astype(np.uint8)  # convert
            ret_val = Image(result)
        else:  # DO RGB separately
            results = []
            for i in range(0, len(input)):
                dftimg = cv2.dft(input[i], flags=cv2.DFT_INVERSE)
                data = dftimg[:, :, 0].copy()
                min, max, pt1, pt2 = cv2.minMaxLoc(data)
                denom = max - min
                if denom == 0:
                    denom = 1
                data = data / denom - min / denom  # scale
                data = cv2.multiply(data, data, scale=255.0)
                result = np.copy(data).astype(np.uint8)
                results.append(result)
            ret_val = Image(np.dstack((results[0], results[1], results[2])))
        return ret_val

    def inverse_dft(self, raw_dft_image):
        """
        **SUMMARY**

        This method provides a way of performing an inverse discrete Fourier
        transform on a real/imaginary image pair and obtaining the result as
        a simplecv image. This method is helpful if you wish to perform custom
        filter development.

        **PARAMETERS**

        * *raw_dft_image* - A list object with either one or three IPL images.
          Each image should have a 64f depth and contain two channels (the real
          and the imaginary).

        **RETURNS**

        A simpleCV image.

        **EXAMPLE**

        Note that this is an example, I don't recommend doing this unless you
        know what you are doing.

        >>> raw = img.getRawDFT()
        >>> cv2.SomeOperation(raw)
        >>> result = img.inverse_dft(raw)
        >>> result.show()

        **SEE ALSO**

        :py:meth:`raw_dft_image`
        :py:meth:`get_dft_log_magnitude`
        :py:meth:`apply_dft_filter`
        :py:meth:`high_pass_filter`
        :py:meth:`low_pass_filter`
        :py:meth:`band_pass_filter`
        :py:meth:`inverse_dft`
        :py:meth:`apply_butterworth_filter`
        :py:meth:`inverse_dft`
        :py:meth:`apply_gaussian_filter`
        :py:meth:`apply_unsharp_mask`

        """
        input = []
        if len(raw_dft_image) == 1:
            input.append(self._DFT[0].copy())
        else:
            for img in raw_dft_image:
                input.append(img.copy())
        return self._inverse_dft(input)

    def apply_butterworth_filter(self, dia=400, order=2, highpass=False,
                                 grayscale=False):
        """
        **SUMMARY**

        Creates a butterworth filter of 64x64 pixels, resizes it to fit
        image, applies DFT on image using the filter.
        Returns image with DFT applied on it

        **PARAMETERS**

        * *dia* - int Diameter of Butterworth low pass filter
        * *order* - int Order of butterworth lowpass filter
        * *highpass*: BOOL True: highpass filterm False: lowpass filter
        * *grayscale*: BOOL

        **EXAMPLE**

        >>> im = Image("lenna")
        >>> img = im.apply_butterworth_filter(dia=400, order=2,
        ...                                   highpass=True, grayscale=False)

        Output image: http://i.imgur.com/5LS3e.png

        >>> img = im.apply_butterworth_filter(dia=400, order=2,
        ...                                   highpass=False, grayscale=False)

        Output img: http://i.imgur.com/QlCAY.png

        >>> # take image from here: http://i.imgur.com/O0gZn.png
        >>> im = Image("grayscale_lenn.png")
        >>> img = im.apply_butterworth_filter(dia=400, order=2,
        ...                                   highpass=True, grayscale=True)

        Output img: http://i.imgur.com/BYYnp.png

        >>> img = im.apply_butterworth_filter(dia=400, order=2,
        ...                           highpass=False, grayscale=True)

        Output img: http://i.imgur.com/BYYnp.png

        **SEE ALSO**

        :py:meth:`raw_dft_image`
        :py:meth:`get_dft_log_magnitude`
        :py:meth:`apply_dft_filter`
        :py:meth:`high_pass_filter`
        :py:meth:`low_pass_filter`
        :py:meth:`band_pass_filter`
        :py:meth:`inverse_dft`
        :py:meth:`apply_butterworth_filter`
        :py:meth:`inverse_dft`
        :py:meth:`apply_gaussian_filter`
        :py:meth:`apply_unsharp_mask`

        """
        #reimplemented with faster, vectorized filter kernel creation
        w, h = self.size()
        intensity_scale = 2 ** 8 - 1  # for now 8-bit
        sz_x = 64  # for now constant, symmetric
        sz_y = 64  # for now constant, symmetric
        x0 = sz_x / 2.0  # for now, on center
        y0 = sz_y / 2.0  # for now, on center
        # efficient "vectorized" computation
        x, y = np.meshgrid(np.arange(sz_x), np.arange(sz_y))
        d = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)
        flt = intensity_scale / (1.0 + (d / dia) ** (order * 2))
        if highpass:  # then invert the filter
            flt = intensity_scale - flt

        # numpy arrays are in row-major form...
        # doesn't matter for symmetric filter
        flt = Image(flt)
        flt_re = flt.resize(w, h)
        img = self.apply_dft_filter(flt_re, grayscale)
        return img

    def apply_gaussian_filter(self, dia=400, highpass=False, grayscale=False):
        """
        **SUMMARY**

        Creates a gaussian filter of 64x64 pixels, resizes it to fit
        image, applies DFT on image using the filter.
        Returns image with DFT applied on it

        **PARAMETERS**

        * *dia* -  int - diameter of Gaussian filter
        * *highpass*: BOOL True: highpass filter False: lowpass filter
        * *grayscale*: BOOL

        **EXAMPLE**

        >>> im = Image("lenna")
        >>> img = im.apply_gaussian_filter(dia=400, highpass=True,
        ...                                grayscale=False)

        Output image: http://i.imgur.com/DttJv.png

        >>> img = im.apply_gaussian_filter(dia=400, highpass=False,
        ...                                grayscale=False)

        Output img: http://i.imgur.com/PWn4o.png

        >>> # take image from here: http://i.imgur.com/O0gZn.png
        >>> im = Image("grayscale_lenn.png")
        >>> img = im.apply_gaussian_filter(dia=400, highpass=True,
        ...                                grayscale=True)

        Output img: http://i.imgur.com/9hX5J.png

        >>> img = im.apply_gaussian_filter(dia=400, highpass=False,
        ...                                grayscale=True)

        Output img: http://i.imgur.com/MXI5T.png

        **SEE ALSO**

        :py:meth:`raw_dft_image`
        :py:meth:`get_dft_log_magnitude`
        :py:meth:`apply_dft_filter`
        :py:meth:`high_pass_filter`
        :py:meth:`low_pass_filter`
        :py:meth:`band_pass_filter`
        :py:meth:`inverse_dft`
        :py:meth:`apply_butterworth_filter`
        :py:meth:`inverse_dft`
        :py:meth:`apply_gaussian_filter`
        :py:meth:`apply_unsharp_mask`

        """
        #reimplemented with faster, vectorized filter kernel creation
        w, h = self.size()
        intensity_scale = 2 ** 8 - 1  # for now 8-bit
        sz_x = 64  # for now constant, symmetric
        sz_y = 64  # for now constant, symmetric
        x0 = sz_x / 2.0  # for now, on center
        y0 = sz_y / 2.0  # for now, on center
        # efficient "vectorized" computation
        x, y = np.meshgrid(np.arange(sz_x), np.arange(sz_y))
        d = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)
        flt = intensity_scale * np.exp(-0.5 * (d / dia) ** 2)
        if highpass:  # then invert the filter
            flt = intensity_scale - flt
        # numpy arrays are in row-major form...
        # doesn't matter for symmetric filter
        flt = Image(flt)
        flt_re = flt.resize(w, h)
        img = self.apply_dft_filter(flt_re, grayscale)
        return img

    def apply_unsharp_mask(self, boost=1, dia=400, grayscale=False):
        """
        **SUMMARY**

        This method applies unsharp mask or highboost filtering
        on image depending upon the boost value provided.
        DFT is applied on image using gaussian lowpass filter.
        A mask is created subtracting the DFT image from the original
        iamge. And then mask is added in the image to sharpen it.
        unsharp masking => image + mask
        highboost filtering => image + (boost)*mask

        **PARAMETERS**

        * *boost* - int  boost = 1 => unsharp masking, boost > 1 => highboost
          filtering
        * *dia* - int Diameter of Gaussian low pass filter
        * *grayscale* - BOOL

        **EXAMPLE**

        Gaussian Filters:

        >>> im = Image("lenna")
        >>> # highboost filtering
        >>> img = im.apply_unsharp_mask(2, grayscale=False)


        output image: http://i.imgur.com/A1pZf.png

        >>> img = im.apply_unsharp_mask(1, grayscale=False) # unsharp masking

        output image: http://i.imgur.com/smCdL.png

        >>> # take image from here: http://i.imgur.com/O0gZn.png
        >>> im = Image("grayscale_lenn.png")
        >>> # highboost filtering
        >>> img = im.apply_unsharp_mask(2, grayscale=True)

        output image: http://i.imgur.com/VtGzl.png

        >>> img = im.apply_unsharp_mask(1,grayscale=True) #unsharp masking

        output image: http://i.imgur.com/bywny.png

        **SEE ALSO**

        :py:meth:`raw_dft_image`
        :py:meth:`get_dft_log_magnitude`
        :py:meth:`apply_dft_filter`
        :py:meth:`high_pass_filter`
        :py:meth:`low_pass_filter`
        :py:meth:`band_pass_filter`
        :py:meth:`inverse_dft`
        :py:meth:`apply_butterworth_filter`
        :py:meth:`inverse_dft`
        :py:meth:`apply_gaussian_filter`
        :py:meth:`apply_unsharp_mask`

        """
        if boost < 0:
            print "boost >= 1"
            return None

        lp_im = self.apply_gaussian_filter(dia=dia, grayscale=grayscale,
                                           highpass=False)
        im = Image(self.get_bitmap())
        mask = im - lp_im
        img = im
        for i in range(boost):
            img = img + mask
        return img

    def list_haar_features(self):
        '''
        This is used to list the built in features available for HaarCascade
        feature detection.  Just run this function as:

        >>> img.list_haar_features()

        Then use one of the file names returned as the input to the
        findHaarFeature() function. So you should get a list, more than likely
        you will see face.xml, to use it then just

        >>> img.find_haar_features('face.xml')
        '''

        features_directory = os.path.join(LAUNCH_PATH,
                                          'data/Features/HaarCascades')
        features = os.listdir(features_directory)
        print features

    def _copy_avg(self, src, dst, roi, levels, levels_f, mode):
        '''
        Take the value in an ROI, calculate the average / peak hue
        and then set the output image roi to the value.
        '''

        src_roi = src._ndarray[roi[1]:roi[1] + roi[3],
                               roi[0]:roi[0] + roi[2]]
        dst_roi = dst[roi[1]:roi[1] + roi[3],
                      roi[0]:roi[0] + roi[2]]
        if mode:  # get the peak hue for an area
            h = Image(src_roi).hue_histogram()
            my_hue = np.argmax(h)
            c = (float(my_hue), float(255), float(255), float(0))
            dst_roi += c
        else:  # get the average value for an area optionally set levels
            avg = cv2.mean(src_roi)
            avg = (float(avg[0]), float(avg[1]), float(avg[2]))
            if levels is not None:
                avg = (int(avg[0] / levels) * levels_f,
                       int(avg[1] / levels) * levels_f,
                       int(avg[2] / levels) * levels_f)
            dst_roi += avg

        dst[roi[1]:roi[1] + roi[3],
            roi[0]:roi[0] + roi[2]] = dst_roi

    def pixelize(self, block_size=10, region=None, levels=None, do_hue=False):
        """
        **SUMMARY**

        Pixelation blur, like the kind used to hide naughty bits on your
        favorite tv show.

        **PARAMETERS**

        * *block_size* - the blur block size in pixels, an integer is an square
           blur, a tuple is rectangular.
        * *region* - do the blur in a region in format (x_position, y_position,
          width, height)
        * *levels* - the number of levels per color channel. This makes the
          image look like an 8-bit video game.
        * *do_hue* - If this value is true we calculate the peak hue for the
          area, not the average color for the area.

        **RETURNS**

        Returns the image with the pixelation blur applied.

        **EXAMPLE**

        >>> img = Image("lenna")
        >>> result = img.pixelize(16, (200, 180, 250, 250), levels=4)
        >>> img.show()

        """

        if isinstance(block_size, int):
            block_size = (block_size, block_size)

        ret_val = self.get_empty()

        levels_f = 0.00
        if levels is not None:
            levels = 255 / int(levels)
            if levels <= 1:
                levels = 2
            levels_f = float(levels)

        if region is not None:
            xs = region[0]
            ys = region[1]
            w = region[2]
            h = region[3]
            ret_val = self._ndarray.copy()
            ret_val[ys:ys + w, xs:xs + h] = 0
        else:
            xs = 0
            ys = 0
            w = self.width
            h = self.height

        #if( region is None ):
        hc = w / block_size[0]  # number of horizontal blocks
        vc = h / block_size[1]  # number of vertical blocks
        #when we fit in the blocks, we're going to spread the round off
        #over the edges 0->x_0, 0->y_0  and x_0+hc*block_size
        x_lhs = int(np.ceil(
            float(w % block_size[0]) / 2.0))  # this is the starting point
        y_lhs = int(np.ceil(float(h % block_size[1]) / 2.0))
        x_rhs = int(np.floor(
            float(w % block_size[0]) / 2.0))  # this is the starting point
        y_rhs = int(np.floor(float(h % block_size[1]) / 2.0))
        x_0 = xs + x_lhs
        y_0 = ys + y_lhs
        x_f = (x_0 + (block_size[0] * hc))  # this would be the end point
        y_f = (y_0 + (block_size[1] * vc))

        for i in range(0, hc):
            for j in range(0, vc):
                xt = x_0 + (block_size[0] * i)
                yt = y_0 + (block_size[1] * j)
                roi = (xt, yt, block_size[0], block_size[1])
                self._copy_avg(self, ret_val, roi, levels, levels_f, do_hue)

        if x_lhs > 0:  # add a left strip
            xt = xs
            wt = x_lhs
            ht = block_size[1]
            for j in range(0, vc):
                yt = y_0 + (j * block_size[1])
                roi = (xt, yt, wt, ht)
                self._copy_avg(self, ret_val, roi, levels, levels_f, do_hue)

        if x_rhs > 0:  # add a right strip
            xt = (x_0 + (block_size[0] * hc))
            wt = x_rhs
            ht = block_size[1]
            for j in range(0, vc):
                yt = y_0 + (j * block_size[1])
                roi = (xt, yt, wt, ht)
                self._copy_avg(self, ret_val, roi, levels, levels_f, do_hue)

        if y_lhs > 0:  # add a left strip
            yt = ys
            ht = y_lhs
            wt = block_size[0]
            for i in range(0, hc):
                xt = x_0 + (i * block_size[0])
                roi = (xt, yt, wt, ht)
                self._copy_avg(self, ret_val, roi, levels, levels_f, do_hue)

        if y_rhs > 0:  # add a right strip
            yt = (y_0 + (block_size[1] * vc))
            ht = y_rhs
            wt = block_size[0]
            for i in range(0, hc):
                xt = x_0 + (i * block_size[0])
                roi = (xt, yt, wt, ht)
                self._copy_avg(self, ret_val, roi, levels, levels_f, do_hue)

        #now the corner cases
        if x_lhs > 0 and y_lhs > 0:
            roi = (xs, ys, x_lhs, y_lhs)
            self._copy_avg(self, ret_val, roi, levels, levels_f, do_hue)

        if x_rhs > 0 and y_rhs > 0:
            roi = (x_f, y_f, x_rhs, y_rhs)
            self._copy_avg(self, ret_val, roi, levels, levels_f, do_hue)

        if x_lhs > 0 and y_rhs > 0:
            roi = (xs, y_f, x_lhs, y_rhs)
            self._copy_avg(self, ret_val, roi, levels, levels_f, do_hue)

        if x_rhs > 0 and y_lhs > 0:
            roi = (x_f, ys, x_rhs, y_lhs)
            self._copy_avg(self, ret_val, roi, levels, levels_f, do_hue)

        if do_hue:
            ret_val = cv2.cvtColor(ret_val, cv2.COLOR_HSV2BGR)

        return Image(ret_val)

    def anonymize(self, block_size=10, features=None, transform=None):
        """
        **SUMMARY**

        Anonymize, for additional privacy to images.

        **PARAMETERS**

        * *features* - A list with the Haar like feature cascades that should
           be matched.
        * *block_size* - The size of the blocks for the pixelize function.
        * *transform* - A function, to be applied to the regions matched
          instead of pixelize.
        * This function must take two arguments: the image and the region
          it'll be applied to,
        * as in region = (x, y, width, height).

        **RETURNS**

        Returns the image with matching regions pixelated.

        **EXAMPLE**

        >>> img = Image("lenna")
        >>> anonymous = img.anonymize()
        >>> anonymous.show()

        >>> def my_function(img, region):
        >>>     x, y, width, height = region
        >>>     img = img.crop(x, y, width, height)
        >>>     return img
        >>>
        >>>img = Image("lenna")
        >>>transformed = img.anonymize(transform = my_function)

        """

        regions = []

        if features is None:
            regions.append(self.find_haar_features("face"))
            regions.append(self.find_haar_features("profile"))
        else:
            for feature in features:
                regions.append(self.find_haar_features(feature))

        found = [f for f in regions if f is not None]

        img = self.copy()

        if found:
            for feature_set in found:
                for region in feature_set:
                    rect = (region.top_left_corner()[0],
                            region.top_left_corner()[1],
                            region.get_width(), region.get_height())
                    if transform is None:
                        img = img.pixelize(block_size=block_size, region=rect)
                    else:
                        img = transform(img, rect)

        return img

    def edge_intersections(self, pt0, pt1, width=1, canny1=0, canny2=100):
        """
        **SUMMARY**

        Find the outermost intersection of a line segment and the edge image
        and return a list of the intersection points. If no intersections are
        found the method returns an empty list.

        **PARAMETERS**

        * *pt0* - an (x,y) tuple of one point on the intersection line.
        * *pt1* - an (x,y) tuple of the second point on the intersection line.
        * *width* - the width of the line to use. This approach works better
                    when for cases where the edges on an object are not always
                    closed and may have holes.
        * *canny1* - the lower bound of the Canny edge detector parameters.
        * *canny2* - the upper bound of the Canny edge detector parameters.

        **RETURNS**

        A list of two (x,y) tuples or an empty list.

        **EXAMPLE**

        >>> img = Image("SimpleCV")
        >>> a = (25, 100)
        >>> b = (225, 110)
        >>> pts = img.edge_intersections(a, b, width=3)
        >>> e = img.edges(0, 100)
        >>> e.draw_line(a, b, color=Color.RED)
        >>> e.draw_circle(pts[0], 10, color=Color.GREEN)
        >>> e.draw_circle(pts[1], 10, color=Color.GREEN)
        >>> e.show()

        img = Image("SimpleCV")
        a = (25,100)
        b = (225,100)
        pts = img.edge_intersections(a,b,width=3)
        e = img.edges(0,100)
        e.draw_line(a,b,color=Color.RED)
        e.draw_circle(pts[0],10,color=Color.GREEN)
        e.draw_circle(pts[1],10,color=Color.GREEN)
        e.show()


        """
        w = abs(pt0[0] - pt1[0])
        h = abs(pt0[1] - pt1[1])
        x = np.min([pt0[0], pt1[0]])
        y = np.min([pt0[1], pt1[1]])
        if w <= 0:
            w = width
            x = np.clip(x - (width / 2), 0, x - (width / 2))
        if h <= 0:
            h = width
            y = np.clip(y - (width / 2), 0, y - (width / 2))
        #got some corner cases to catch here
        p0p = np.array([(pt0[0] - x, pt0[1] - y)])
        p1p = np.array([(pt1[0] - x, pt1[1] - y)])
        edges = self.crop(x, y, w, h)._get_edge_map(canny1, canny2)
        line = np.zeros((h, w), np.uint8)
        cv2.line(line, ((pt0[0] - x), (pt0[1] - y)),
                ((pt1[0] - x), (pt1[1] - y)), 255.00, width, 8)
        line = cv2.multiply(line, edges)
        intersections = line.transpose()
        (xs, ys) = np.where(intersections == 255)
        points = zip(xs, ys)
        if len(points) == 0:
            return [None, None]
        a = np.argmin(spsd.cdist(p0p, points, 'cityblock'))
        b = np.argmin(spsd.cdist(p1p, points, 'cityblock'))
        pt_a = (int(xs[a] + x), int(ys[a] + y))
        pt_b = (int(xs[b] + x), int(ys[b] + y))
        # we might actually want this to be list of all the points
        return [pt_a, pt_b]

    def fit_contour(self, initial_curve, window=(11, 11),
                    params=(0.1, 0.1, 0.1), do_appx=True, appx_level=1):
        """

        **SUMMARY**

        This method tries to fit a list of points to lines in the image. The
        list of points is a list of (x,y) tuples that are near (i.e. within the
        window size) of the line you want to fit in the image. This method uses
        a binary such as the result of calling edges.

        This method is based on active contours. Please see this reference:
        http://en.wikipedia.org/wiki/Active_contour_model

        **PARAMETERS**

        * *initial_curve* - region of the form [(x0,y0),(x1,y1)...] that are
          the initial conditions to fit.
        * *window* - the search region around each initial point to look for
          a solution.
        * *params* - The alpha, beta, and gamma parameters for the active
          contours algorithm as a list [alpha, beta, gamma].
        * *do_appx* - post process the snake into a polynomial approximation.
          Basically this flag will clean up the output of the get_contour
          algorithm.
        * *appx_level* - how much to approximate the snake, higher numbers mean
          more approximation.

        **DISCUSSION**

        THIS SECTION IS QUOTED FROM: http://users.ecs.soton.ac.uk/msn/
        book/new_demo/Snakes/
        There are three components to the Energy Function:

        * Continuity
        * Curvature
        * Image (Gradient)

        Each Weighted by Specified Parameter:

        Total Energy = Alpha*Continuity + Beta*Curvature + Gamma*Image

        Choose different values dependent on Feature to extract:

        * Set alpha high if there is a deceptive Image Gradient
        * Set beta  high if smooth edged Feature, low if sharp edges
        * Set gamma high if contrast between Background and Feature is low


        **RETURNS**

        A list of (x,y) tuples that approximate the curve. If you do not use
        approximation the list should be the same length as the input list
        length.

        **EXAMPLE**

        >>> img = Image("lenna")
        >>> edges = img.edges(t1=120, t2=155)
        >>> guess = [(311, 284), (313, 270),
            ...      (320, 259), (330, 253), (347, 245)]
        >>> result = edges.fit_contour(guess)
        >>> img.draw_points(guess, color=Color.RED)
        >>> img.draw_points(result, color=Color.GREEN)
        >>> img.show()

        """
        raise Exception('deprecated. cv2 has no SnakeImage')
        # alpha = [params[0]]
        # beta = [params[1]]
        # gamma = [params[2]]
        # if window[0] % 2 == 0:
        #     window = (window[0] + 1, window[1])
        #     logger.warn("Yo dawg, just a heads up, snakeFitPoints wants an "
        #                 "odd window size. I fixed it for you, but you may "
        #                 "want to take a look at your code.")
        # if window[1] % 2 == 0:
        #     window = (window[0], window[1] + 1)
        #     logger.warn("Yo dawg, just a heads up, snakeFitPoints wants an "
        #                 "odd window size. I fixed it for you, but you may "
        #                 "want to take a look at your code.")
        # raw = cv.SnakeImage(self._get_grayscale_bitmap(), initial_curve,
        #                     alpha, beta, gamma, window,
        #                     (cv.CV_TERMCRIT_ITER, 10, 0.01))
        # if do_appx:
        #     appx = cv2.approxPolyDP(np.array([raw], 'float32'), appx_level,
        #                             True)
        #     ret_val = []
        #     for p in appx:
        #         ret_val.append((int(p[0][0]), int(p[0][1])))
        # else:
        #     ret_val = raw
        #
        # return ret_val

    def fit_edge(self, guess, window=10, threshold=128, measurements=5,
                 darktolight=True, lighttodark=True, departurethreshold=1):
        """
        **SUMMARY**

        Fit edge in a binary/gray image using an initial guess and the least
        squares method. The functions returns a single line

        **PARAMETERS**

        * *guess* - A tuples of the form ((x0,y0),(x1,y1)) which is an
          approximate guess
        * *window* - A window around the guess to search.
        * *threshold* - the threshold above which we count a pixel as a line
        * *measurements* -the number of line projections to use for fitting
        the line

        TODO: Constrict a line to black to white or white to black
        Right vs. Left orientation.

        **RETURNS**

        A a line object
        **EXAMPLE**
      """
        search_lines = FeatureSet()
        fit_points = FeatureSet()
        x1 = guess[0][0]
        x2 = guess[1][0]
        y1 = guess[0][1]
        y2 = guess[1][1]
        dx = float((x2 - x1)) / (measurements - 1)
        dy = float((y2 - y1)) / (measurements - 1)
        s = np.zeros((measurements, 2))
        lpstartx = np.zeros(measurements)
        lpstarty = np.zeros(measurements)
        lpendx = np.zeros(measurements)
        lpendy = np.zeros(measurements)
        linefitpts = np.zeros((measurements, 2))

        # obtain equation for initial guess line
        # vertical line must be handled as special
        # case since slope isn't defined
        if x1 == x2:
            m = 0
            mo = 0
            b = x1
            for i in xrange(0, measurements):
                s[i][0] = x1
                s[i][1] = y1 + i * dy
                lpstartx[i] = s[i][0] + window
                lpstarty[i] = s[i][1]
                lpendx[i] = s[i][0] - window
                lpendy[i] = s[i][1]
                cur_line = Line(self, ((lpstartx[i], lpstarty[i]),
                                       (lpendx[i], lpendy[i])))
                search_lines.append(cur_line)
                tmp = self.get_threshold_crossing(
                    (int(lpstartx[i]), int(lpstarty[i])),
                    (int(lpendx[i]), int(lpendy[i])), threshold=threshold,
                    lighttodark=lighttodark, darktolight=darktolight,
                    departurethreshold=departurethreshold)
                fit_points.append(Circle(self, tmp[0], tmp[1], 3))
                linefitpts[i] = tmp

        else:
            m = float((y2 - y1)) / (x2 - x1)
            b = y1 - m * x1
            mo = -1 / m  # slope of orthogonal line segments

            # obtain points for measurement along the initial guess line
            for i in xrange(0, measurements):
                s[i][0] = x1 + i * dx
                s[i][1] = y1 + i * dy
                fx = (math.sqrt(math.pow(window, 2)) / (1 + mo)) / 2
                fy = fx * mo
                lpstartx[i] = s[i][0] + fx
                lpstarty[i] = s[i][1] + fy
                lpendx[i] = s[i][0] - fx
                lpendy[i] = s[i][1] - fy
                cur_line = Line(self, ((lpstartx[i], lpstarty[i]),
                                       (lpendx[i], lpendy[i])))
                search_lines.append(cur_line)
                tmp = self.get_threshold_crossing(
                    (int(lpstartx[i]), int(lpstarty[i])),
                    (int(lpendx[i]), int(lpendy[i])), threshold=threshold,
                    lighttodark=lighttodark, darktolight=darktolight,
                    departurethreshold=departurethreshold)
                fit_points.append((tmp[0], tmp[1]))
                linefitpts[i] = tmp

        x = linefitpts[:, 0]
        y = linefitpts[:, 1]
        ymin = np.min(y)
        ymax = np.max(y)
        xmax = np.max(x)
        xmin = np.min(x)

        if (xmax - xmin) > (ymax - ymin):
            # do the least squares
            a = np.vstack([x, np.ones(len(x))]).T
            m, c = nla.lstsq(a, y)[0]
            y0 = int(m * xmin + c)
            y1 = int(m * xmax + c)
            final_line = Line(self, ((xmin, y0), (xmax, y1)))
        else:
            # do the least squares
            a = np.vstack([y, np.ones(len(y))]).T
            m, c = nla.lstsq(a, x)[0]
            x0 = int(ymin * m + c)
            x1 = int(ymax * m + c)
            final_line = Line(self, ((x0, ymin), (x1, ymax)))

        return final_line, search_lines, fit_points

    def get_threshold_crossing(self, pt1, pt2, threshold=128, darktolight=True,
                               lighttodark=True, departurethreshold=1):
        """
        **SUMMARY**

        This function takes in an image and two points, calculates the
        intensity profile between the points, and returns the single point at
        which the profile crosses an intensity

        **PARAMETERS**

        * *p1, p2* - the starting and ending points in tuple form e.g. (1,2)
        * *threshold* pixel value of desired threshold crossing
        * *departurethreshold* - noise reduction technique.  requires this
         many points to be above the threshold to trigger crossing

        **RETURNS**

        A a lumpy numpy array of the pixel values. Ususally this is in BGR
        format.

        **EXAMPLE**

        >>> img = Image("lenna")
        >>> myColor = [0,0,0]
        >>> sl = img.get_horz_scanline(422)
        >>> sll = sl.tolist()
        >>> for p in sll:
        >>>    if p == myColor:
        >>>        # do something

        **SEE ALSO**

        :py:meth:`get_horz_scanline_gray`
        :py:meth:`get_vert_scanline_gray`
        :py:meth:`get_vert_scanline`

        """
        linearr = self.get_diagonal_scanline_grey(pt1, pt2)
        ind = 0
        crossing = -1
        if departurethreshold == 1:
            while ind < linearr.size - 1:
                if darktolight:
                    if linearr[ind] <= threshold \
                            and linearr[ind + 1] > threshold:
                        crossing = ind
                        break
                if lighttodark:
                    if linearr[ind] >= threshold \
                            and linearr[ind + 1] < threshold:
                        crossing = ind
                        break
                ind = ind + 1
            if crossing != -1:
                xind = pt1[0] + int(
                    round((pt2[0] - pt1[0]) * crossing / linearr.size))
                yind = pt1[1] + int(
                    round((pt2[1] - pt1[1]) * crossing / linearr.size))
                ret_val = (xind, yind)
            else:
                ret_val = (-1, -1)
                print 'Edgepoint not found.'
        else:
            while ind < linearr.size - (departurethreshold + 1):
                if darktolight:
                    if linearr[ind] <= threshold and \
                            (linearr[ind + 1:ind + 1 + departurethreshold]
                             > threshold).all():
                        crossing = ind
                        break
                if lighttodark:
                    if linearr[ind] >= threshold and \
                            (linearr[ind + 1:ind + 1 + departurethreshold]
                             < threshold).all():
                        crossing = ind
                        break
                ind = ind + 1
            if crossing != -1:
                xind = pt1[0] + int(
                    round((pt2[0] - pt1[0]) * crossing / linearr.size))
                yind = pt1[1] + int(
                    round((pt2[1] - pt1[1]) * crossing / linearr.size))
                ret_val = (xind, yind)
            else:
                ret_val = (-1, -1)
                print 'Edgepoint not found.'
        return ret_val

    def get_diagonal_scanline_grey(self, pt1, pt2):
        """
        **SUMMARY**

        This function returns a single line of greyscale values from the image.
        TODO: speed inprovements and RGB tolerance

        **PARAMETERS**

        * *pt1, pt2* - the starting and ending points in tuple form e.g. (1,2)

        **RETURNS**

        An array of the pixel values.

        **EXAMPLE**

        >>> img = Image("lenna")
        >>> sl = img.get_diagonal_scanline_grey((100, 200), (300, 400))


        **SEE ALSO**

        :py:meth:`get_horz_scanline_gray`
        :py:meth:`get_vert_scanline_gray`
        :py:meth:`get_vert_scanline`

        """
        if not self.is_gray():
            img = self.to_gray()
        else:
            img = self

        width = round(math.sqrt(
            math.pow(pt2[0] - pt1[0], 2) + math.pow(pt2[1] - pt1[1], 2)))
        ret_val = np.zeros(width)

        for x in range(0, ret_val.size):
            xind = pt1[0] + int(round((pt2[0] - pt1[0]) * x / ret_val.size))
            yind = pt1[1] + int(round((pt2[1] - pt1[1]) * x / ret_val.size))
            current_pixel = img.get_pixel(xind, yind)
            ret_val[x] = current_pixel[0]
        return ret_val

    def fit_lines(self, guesses, window=10, threshold=128):
        """
        **SUMMARY**

        Fit lines in a binary/gray image using an initial guess and the least
        squares method. The lines are returned as a line feature set.

        **PARAMETERS**

        * *guesses* - A list of tuples of the form ((x0,y0),(x1,y1)) where each
          of the lines is an approximate guess.
        * *window* - A window around the guess to search.
        * *threshold* - the threshold above which we count a pixel as a line

        **RETURNS**

        A feature set of line features, one per guess.

        **EXAMPLE**


        >>> img = Image("lsq.png")
        >>> guesses = [((313, 150), (312, 332)), ((62, 172), (252, 52)),
        ...            ((102, 372), (182, 182)), ((372, 62), (572, 162)),
        ...            ((542, 362), (462, 182)), ((232, 412), (462, 423))]
        >>> l = img.fit_lines(guesses, window=10)
        >>> l.draw(color=Color.RED, width=3)
        >>> for g in guesses:
        >>>    img.draw_line(g[0], g[1], color=Color.YELLOW)

        >>> img.show()
        """

        ret_val = FeatureSet()
        i = 0
        for g in guesses:
            # Guess the size of the crop region from the line
            # guess and the window.
            ymin = np.min([g[0][1], g[1][1]])
            ymax = np.max([g[0][1], g[1][1]])
            xmin = np.min([g[0][0], g[1][0]])
            xmax = np.max([g[0][0], g[1][0]])

            xmin_w = np.clip(xmin - window, 0, self.width)
            xmax_w = np.clip(xmax + window, 0, self.width)
            ymin_w = np.clip(ymin - window, 0, self.height)
            ymax_w = np.clip(ymax + window, 0, self.height)
            temp = self.crop(xmin_w, ymin_w, xmax_w - xmin_w, ymax_w - ymin_w)
            temp = temp.get_gray_ndarray()

            # pick the lines above our threshold
            x, y = np.where(temp > threshold)
            pts = zip(x, y)
            gpv = np.array([float(g[0][0] - xmin_w), float(g[0][1] - ymin_w)])
            gpw = np.array([float(g[1][0] - xmin_w), float(g[1][1] - ymin_w)])

            def line_segment_to_point(p):
                w = gpw
                v = gpv
                #print w,v
                p = np.array([float(p[0]), float(p[1])])
                l2 = np.sum((w - v) ** 2)
                t = float(np.dot((p - v), (w - v))) / float(l2)
                if t < 0.00:
                    return np.sqrt(np.sum((p - v) ** 2))
                elif t > 1.0:
                    return np.sqrt(np.sum((p - w) ** 2))
                else:
                    project = v + (t * (w - v))
                    return np.sqrt(np.sum((p - project) ** 2))

            # http://stackoverflow.com/questions/849211/
            # shortest-distance-between-a-point-and-a-line-segment

            distances = np.array(map(line_segment_to_point, pts))
            closepoints = np.where(distances < window)[0]

            pts = np.array(pts)

            if len(closepoints) < 3:
                continue

            good_pts = pts[closepoints]
            good_pts = good_pts.astype(float)

            x = good_pts[:, 0]
            y = good_pts[:, 1]
            # do the shift from our crop
            # generate the line values
            x = x + xmin_w
            y = y + ymin_w

            ymin = np.min(y)
            ymax = np.max(y)
            xmax = np.max(x)
            xmin = np.min(x)

            if (xmax - xmin) > (ymax - ymin):
                # do the least squares
                a = np.vstack([x, np.ones(len(x))]).T
                m, c = nla.lstsq(a, y)[0]
                y0 = int(m * xmin + c)
                y1 = int(m * xmax + c)
                ret_val.append(Line(self, ((xmin, y0), (xmax, y1))))
            else:
                # do the least squares
                a = np.vstack([y, np.ones(len(y))]).T
                m, c = nla.lstsq(a, x)[0]
                x0 = int(ymin * m + c)
                x1 = int(ymax * m + c)
                ret_val.append(Line(self, ((x0, ymin), (x1, ymax))))

        return ret_val

    def fit_line_points(self, guesses, window=(11, 11), samples=20,
                        params=(0.1, 0.1, 0.1)):
        """
        **DESCRIPTION**

        This method uses the snakes / active get_contour approach in an attempt
        to fit a series of points to a line that may or may not be exactly
        linear.

        **PARAMETERS**

        * *guesses* - A set of lines that we wish to fit to. The lines are
          specified as a list of tuples of (x,y) tuples.
          E.g. [((x0,y0),(x1,y1))....]
        * *window* - The search window in pixels for the active contours
          approach.
        * *samples* - The number of points to sample along the input line,
          these are the initial conditions for active contours method.
        * *params* - the alpha, beta, and gamma values for the active
          contours routine.

        **RETURNS**

        A list of fitted get_contour points. Each get_contour is a list of
        (x,y) tuples.

        **EXAMPLE**

        >>> img = Image("lsq.png")
        >>> guesses = [((313, 150), (312, 332)), ((62, 172), (252, 52)),
        ...            ((102, 372), (182, 182)), ((372, 62), (572, 162)),
        ...            ((542, 362), (462, 182)), ((232, 412), (462, 423))]
        >>> r = img.fit_line_points(guesses)
        >>> for rr in r:
        >>>    img.draw_line(rr[0], rr[1], color=Color.RED, width=3)
        >>> for g in guesses:
        >>>    img.draw_line(g[0], g[1], color=Color.YELLOW)

        >>> img.show()

        """
        pts = []
        for g in guesses:
            #generate the approximation
            best_guess = []
            dx = float(g[1][0] - g[0][0])
            dy = float(g[1][1] - g[0][1])
            l = np.sqrt((dx * dx) + (dy * dy))
            if l <= 0:
                logger.warning("Can't Do snakeFitPoints without "
                               "OpenCV >= 2.3.0")
                return

            dx = dx / l
            dy = dy / l
            for i in range(-1, samples + 1):
                t = i * (l / samples)
                best_guess.append(
                    (int(g[0][0] + (t * dx)), int(g[0][1] + (t * dy))))
            # do the snake fitting
            appx = self.fit_contour(best_guess, window=window, params=params,
                                    do_appx=False)
            pts.append(appx)

        return pts

    def draw_points(self, pts, color=Color.RED, sz=3, width=-1):
        """
        **DESCRIPTION**

        A quick and dirty points rendering routine.

        **PARAMETERS**

        * *pts* - pts a list of (x,y) points.
        * *color* - a color for our points.
        * *sz* - the circle radius for our points.
        * *width* - if -1 fill the point, otherwise the size of point border

        **RETURNS**

        None - This is an inplace operation.

        **EXAMPLE**

        >>> img = Image("lenna")
        >>> img.draw_points([(10,10),(30,30)])
        >>> img.show()
        """
        for p in pts:
            self.draw_circle(p, sz, color, width)
        return None

    def sobel(self, xorder=1, yorder=1, do_gray=True, aperture=5):
        """
        **DESCRIPTION**

        Sobel operator for edge detection

        **PARAMETERS**

        * *xorder* - int - Order of the derivative x.
        * *yorder* - int - Order of the derivative y.
        * *do_gray* - Bool - grayscale or not.
        * *aperture* - int - Size of the extended Sobel kernel. It must be 1,
          3, 5, or 7.

        **RETURNS**

        Image with sobel opeartor applied on it

        **EXAMPLE**

        >>> img = Image("lenna")
        >>> s = img.sobel()
        >>> s.show()
        """
        ret_val = None
        if aperture != 1 and aperture != 3 and aperture != 5 and aperture != 7:
            logger.warning("Bad Sobel Aperture, values are [1,3,5,7].")
            return None

        if do_gray:
            dst = cv2.Sobel(self.get_gray_ndarray(), cv2.CV_32F, xorder,
                            yorder, ksize=aperture)
            minv = np.min(dst)
            maxv = np.max(dst)
            cscale = 255 / (maxv - minv)
            shift = -1 * minv

            t = np.zeros(self.size(), dtype=np.uint8)
            t = cv2.convertScaleAbs(dst, t, cscale, shift / 255.0)
            ret_val = Image(t)

        else:
            layers = self.split_channels(grayscale=False)
            sobel_layers = []
            for layer in layers:
                dst = cv2.Sobel(layer.get_gray_numpy(), cv2.CV_32F, xorder,
                                yorder, ksize=aperture)

                minv = np.min(dst)
                maxv = np.max(dst)
                cscale = 255 / (maxv - minv)
                shift = -1 * minv

                t = np.zeros(self.size(), dtype=np.uint8)
                t = cv2.convertScaleAbs(dst, t, cscale, shift / 255.0)
                sobel_layers.append(Image(t))
            b, g, r = sobel_layers

            ret_val = self.merge_channels(b, g, r)
        return ret_val

    def track(self, method="CAMShift", ts=None, img=None, bb=None, **kwargs):
        """

        **DESCRIPTION**

        Tracking the object surrounded by the bounding box in the given
        image or TrackSet.

        **PARAMETERS**

        * *method* - str - The Tracking Algorithm to be applied
        * *ts* - TrackSet - SimpleCV.Features.TrackSet.
        * *img* - Image - Image to be tracked or list - List of Images to be
          tracked.
        * *bb* - tuple - Bounding Box tuple (x, y, w, h)


        **Optional Parameters**

        *CAMShift*

        CAMShift Tracker is based on mean shift thresholding algorithm which is
        combined with an adaptive region-sizing step. Histogram is calcualted
        based on the mask provided. If mask is not provided, hsv transformed
        image of the provided image is thresholded using inRange function
        (band thresholding).

        lower HSV and upper HSV values are used inRange function. If the user
        doesn't provide any range values, default range values are used.

        Histogram is back projected using previous images to get an appropriate
        image and it passed to camshift function to find the object in the
        image. Users can decide the number of images to be used in back
        projection by providing num_frames.

        lower - Lower HSV value for inRange thresholding. tuple of (H, S, V).
                Default : (0, 60, 32)
        upper - Upper HSV value for inRange thresholding. tuple of (H, S, V).
                Default: (180, 255, 255)
        mask - Mask to calculate Histogram. It's better if you don't provide
               one. Default: calculated using above thresholding ranges.
        num_frames - number of frames to be backtracked. Default: 40

        *LK*

        LK Tracker is based on Optical Flow method. In brief, optical flow can
        be defined as the apparent motion of objects caused by the relative
        motion between an observer and the scene. (Wikipedia).

        LK Tracker first finds some good feature points in the given bounding
        box in the image. These are the tracker points. In consecutive frames,
        optical flow of these feature points is calculated. Users can limit the
        number of feature points by provideing maxCorners and qualityLevel.
        Number of features will always be less than maxCorners. These feature
        points are calculated using Harris Corner detector. It returns a matrix
        with each pixel having some quality value. Only good features are used
        based upon the qualityLevel provided. better features have better
        quality measure and hence are more suitable to track.

        Users can set minimum distance between each features by providing
        minDistance.

        LK tracker finds optical flow using a number of pyramids and users
        can set this number by providing maxLevel and users can set size of the
        search window for Optical Flow by setting winSize.

        docs from http://docs.opencv.org/
        maxCorners - Maximum number of corners to return in
                     goodFeaturesToTrack. If there are more corners than are
                     found, the strongest of them is returned. Default: 4000
        qualityLevel - Parameter characterizing the minimal accepted quality of
                       image corners. The parameter value is multiplied by the
                       best corner quality measure, which is the minimal
                       eigenvalue or the Harris function response. The corners
                       with the quality measure less than the product are
                       rejected. For example, if the best corner has the
                       quality measure = 1500,  and the qualityLevel=0.01 ,
                       then all the corners with the quality measure less than
                       15 are rejected. Default: 0.08
        minDistance - Minimum possible Euclidean distance between the returned
                      corners. Default: 2
        blockSize - Size of an average block for computing a derivative
                    covariation matrix over each pixel neighborhood. Default: 3
        winSize - size of the search window at each pyramid level.
                  Default: (10, 10)
        maxLevel - 0-based maximal pyramid level number; if set to 0, pyramids
                   are not used (single level), Default: 10 if set to 1, two
                   levels are used, and so on

        *SURF*

        SURF based tracker finds keypoints in the template and computes the
        descriptor. The template is chosen based on the bounding box provided
        with the first image. The image is cropped and stored as template. SURF
        keypoints are found and descriptor is computed for the template and
        stored.

        SURF keypoints are found in the image and its descriptor is computed.
        Image keypoints and template keypoints are matched using K-nearest
        neighbor algorithm. Matched keypoints are filtered according to the knn
        distance of the points. Users can set this criteria by setting
        distance. Density Based Clustering algorithm (DBSCAN) is applied on
        the matched keypoints to filter out points that are in background.
        DBSCAN creates a cluster of object points anc background points. These
        background points are discarded. Users can set certain parameters for
        DBSCAN which are listed below.

        K-means is applied on matched KeyPoints with k=1 to find the center of
        the cluster and then bounding box is predicted based upon the position
        of all the object KeyPoints.

        eps_val - eps for DBSCAN. The maximum distance between two samples for
          them to be considered as in the same neighborhood. default: 0.69
        min_samples - min number of samples in DBSCAN. The number of samples
          in a neighborhood for a point to be considered as a core point.
          default: 5
        distance - thresholding KNN distance of each feature. if
        KNN distance > distance, point is discarded. default: 100

        *MFTrack*

        Median Flow tracker is similar to LK tracker (based on Optical Flow),
        but it's more advanced, better and faster.

        In MFTrack, tracking points are decided based upon the number of
        horizontal and vertical points and window size provided by the user.
        Unlike LK Tracker, good features are not found which saves a huge
        amount of time.

        feature points are selected symmetrically in the bounding box.
        Total number of feature points to be tracked = numM * numN.

        If the width and height of bounding box is 200 and 100 respectively,
        and numM = 10 and numN = 10, there will be 10 points in the bounding
        box equally placed(10 points in 200 pixels) in each row. and 10 equally
        placed points (10 points in 100 pixels) in each column. So total number
        of tracking points = 100.

        numM > 0
        numN > 0 (both may not be equal)

        users can provide a margin around the bounding box that will be
        considered to place feature points and calculate optical flow.
        Optical flow is calculated from frame1 to frame2 and from frame2 to
        frame1. There might be some points which give inaccurate optical flow,
        to eliminate these points the above method is used. It is called
        forward-backward error tracking. Optical Flow seach window size can be
        set usung winsize_lk.

        For each point, comparision is done based on the quadratic area around
        it. The length of the square window can be set using winsize.

        numM        - Number of points to be tracked in the bounding box
                      in height direction.
                      default: 10

        numN        - Number of points to be tracked in the bounding box
                      in width direction.
                      default: 10

        margin      - Margin around the bounding box.
                      default: 5

        winsize_lk  - Optical Flow search window size.
                      default: 4

        winsize     - Size of quadratic area around the point which is
                      compared. default: 10


        Available Tracking Methods

         - CamShift
         - LK
         - SURF
         - MFTrack


        **RETURNS**

        SimpleCV.Features.TrackSet

        Returns a TrackSet with all the necessary attributes.

        **HOW TO**

        >>> ts = img.track("camshift", img=img1, bb=bb)


        Here TrackSet is returned. All the necessary attributes will be
        included in the trackset. After getting the trackset you need not
        provide the bounding box or image. You provide TrackSet as parameter
        to track(). Bounding box and image will be taken from the trackset.
        So. now

        >>> ts = new_img.track("camshift", ts)

        The new Tracking feature will be appended to the given trackset and
        that will be returned.
        So, to use it in loop::

          img = cam.getImage()
          bb = (img.width/4,img.height/4,img.width/4,img.height/4)
          ts = img.track(img=img, bb=bb)
          while (True):
              img = cam.getImage()
              ts = img.track("camshift", ts=ts)

          ts = []
          while (some_condition_here):
              img = cam.getImage()
              ts = img.track("camshift",ts,img0,bb)


        now here in first loop iteration since ts is empty, img0 and bb will
        be considered. New tracking object will be created and added in ts
        (TrackSet) After first iteration, ts is not empty and hence the
        previous image frames and bounding box will be taken from ts and img0
        and bb will be ignored.

        # Instead of loop, give a list of images to be tracked.

        ts = []
        imgs = [img1, img2, img3, ..., imgN]
        ts = img0.track("camshift", ts, imgs, bb)
        ts.drawPath()
        ts[-1].image.show()

        Using Optional Parameters:

        for CAMShift

        >>> ts = []
        >>> ts = img.track("camshift", ts, img1, bb, lower=(40, 100, 100),
            ...            upper=(100, 250, 250))

        You can provide some/all/None of the optional parameters listed
        for CAMShift.

        for LK

        >>> ts = []
        >>> ts = img.track("lk", ts, img1, bb, maxCorners=4000,
            ...            qualityLevel=0.5, minDistance=3)

        You can provide some/all/None of the optional parameters listed for LK.

        for SURF

        >>> ts = []
        >>> ts = img.track("surf", ts, img1, bb, eps_val=0.7, min_samples=8,
            ...            distance=200)

        You can provide some/all/None of the optional parameters listed
        for SURF.

        for MFTrack
        >>> ts = []
        >>> ts = img.track("mftrack", ts, img1, bb, numM=12, numN=12,
            ...            winsize=15)

        You can provide some/all/None of the optional parameters listed for
        MFTrack.

        Check out Tracking examples provided in the SimpleCV source code.

        READ MORE:

        CAMShift Tracker:
        Uses meanshift based CAMShift thresholding technique. Blobs and objects
        with single tone or tracked very efficiently. CAMshift should be
        preferred if you are trying to track faces. It is optimized to track
        faces.

        LK (Lucas Kanade) Tracker:
        It is based on LK Optical Flow. It calculates Optical flow in frame1
        to frame2 and also in frame2 to frame1 and using back track error,
        filters out false positives.

        SURF based Tracker:
        Matches keypoints from the template image and the current frame.
        flann based matcher is used to match the keypoints.
        Density based clustering is used classify points as in-region
        (of bounding box) and out-region points. Using in-region points, new
        bounding box is predicted using k-means.

        Median Flow Tracker:

        Media Flow Tracker is the base tracker that is used in OpenTLD. It is
        based on Optical Flow. It calculates optical flow of the points in the
        bounding box from frame 1 to frame 2 and from frame 2 to frame 1 and
        using back track error, removes false positives. As the name suggests,
        it takes the median of the flow, and eliminates points.
        """
        if not ts and not img:
            print "Invalid Input. Must provide FeatureSet or Image"
            return None

        if not ts and not bb:
            print "Invalid Input. Must provide Bounding Box with Image"
            return None

        if not ts:
            ts = TrackSet()
        else:
            img = ts[-1].image
            bb = ts[-1].bb

        if type(img) == list:
            ts = self.track(method, ts, img[0], bb, **kwargs)
            for i in img:
                ts = i.track(method, ts, **kwargs)
            return ts

        # Issue #256 - (Bug) Memory management issue due to too many number
        # of images.
        nframes = 300
        if 'nframes' in kwargs:
            nframes = kwargs['nframes']

        if len(ts) > nframes:
            ts.trimList(50)

        if method.lower() == "camshift":
            track = camshiftTracker(self, bb, ts, **kwargs)
            ts.append(track)

        elif method.lower() == "lk":
            track = lkTracker(self, bb, ts, img, **kwargs)
            ts.append(track)

        elif method.lower() == "surf":
            try:
                from scipy.spatial import distance as dist
                from sklearn.cluster import DBSCAN
            except ImportError:
                logger.warning("sklearn required")
                return None
            if not hasattr(cv2, "FeatureDetector_create"):
                warnings.warn("OpenCV >= 2.4.3 required. Returning None.")
                return None
            track = surfTracker(self, bb, ts, **kwargs)
            ts.append(track)

        elif method.lower() == "mftrack":
            track = mfTracker(self, bb, ts, img, **kwargs)
            ts.append(track)

        return ts

    def _to32f(self):
        """
        **SUMMARY**

        Convert this image to a 32bit floating point image.

        """
        return self._ndarray.astype(np.float32)

    def __getstate__(self):
        return dict(colorspace=self._colorSpace,
                    image=self.apply_layers().get_ndarray())

    def __setstate__(self, mydict):
        self._ndarray = mydict['image']
        self._colorSpace = mydict['colorspace']
        self.height = self._ndarray.shape[0]
        self.width = self._ndarray.shape[1]
        self.dtype = self._ndarray.dtype

    def get_area(self):
        '''
        Returns the area of the Image.
        '''
        return self.width * self.height

    def _get_header_anim(self):
        """ Animation header. To replace the getheader()[0] """
        bb = "GIF89a"
        bb += int_to_bin(self.size()[0])
        bb += int_to_bin(self.size()[1])
        bb += "\x87\x00\x00"
        return bb

    def rotate270(self):
        """
        **DESCRIPTION**

        Rotate the image 270 degrees to the left, the same as 90 degrees to
        the right. This is the same as rotate_right()

        **RETURNS**

        A SimpleCV image.

        **EXAMPLE**

        >>>> img = Image('lenna')
        >>>> img.rotate270().show()

        """
        array = cv2.flip(self._ndarray, 0)  # vertical
        array = cv2.transpose(array)
        return Image(array, color_space=self._colorSpace)

    def rotate90(self):
        """
        **DESCRIPTION**

        Rotate the image 90 degrees to the left, the same as 270 degrees to the
        right. This is the same as rotate_right()

        **RETURNS**

        A SimpleCV image.

        **EXAMPLE**

        >>>> img = Image('lenna')
        >>>> img.rotate90().show()

        """

        array = cv2.transpose(self._ndarray)
        array = cv2.flip(array, 0)  # vertical
        return Image(array, color_space=self._colorSpace)

    def rotate_left(self):  # same as 90
        """
        **DESCRIPTION**

        Rotate the image 90 degrees to the left.
        This is the same as rotate 90.

        **RETURNS**

        A SimpleCV image.

        **EXAMPLE**

        >>>> img = Image('lenna')
        >>>> img.rotate_left().show()

        """

        return self.rotate90()

    def rotate_right(self):  # same as 270
        """
        **DESCRIPTION**

        Rotate the image 90 degrees to the right.
        This is the same as rotate 270.

        **RETURNS**

        A SimpleCV image.

        **EXAMPLE**

        >>>> img = Image('lenna')
        >>>> img.rotate_right().show()

        """

        return self.rotate270()

    def rotate180(self):
        """
        **DESCRIPTION**

        Rotate the image 180 degrees to the left/right.
        This is the same as rotate 90.

        **RETURNS**

        A SimpleCV image.

        **EXAMPLE**

        >>>> img = Image('lenna')
        >>>> img.rotate180().show()
        """
        array = cv2.flip(self._ndarray, 0)  # vertical
        array = cv2.flip(array, 1)  # horizontal
        return Image(array, color_space=self._colorSpace)

    def vertical_histogram(self, bins=10, threshold=128, normalize=False,
                           for_plot=False):
        """

        **DESCRIPTION**

        This method generates histogram of the number of grayscale pixels
        greater than the provided threshold. The method divides the image
        into a number evenly spaced vertical bins and then counts the number
        of pixels where the pixel is greater than the threshold. This method
        is helpful for doing basic morphological analysis.

        **PARAMETERS**

        * *bins* - The number of bins to use.
        * *threshold* - The grayscale threshold. We count pixels greater than
          this value.
        * *normalize* - If normalize is true we normalize the bin countsto sum
          to one. Otherwise we return the number of pixels.
        * *for_plot* - If this is true we return the bin indicies, the bin
          counts, and the bin widths as a tuple. We can use these values in
          pyplot.bar to quickly plot the histogram.


        **RETURNS**

        The default settings return the raw bin counts moving from left to
        right on the image. If for_plot is true we return a tuple that
        contains a list of bin labels, the bin counts, and the bin widths.
        This tuple can be used to plot the histogram using
        matplotlib.pyplot.bar function.


        **EXAMPLE**

          >>> import matplotlib.pyplot as plt
          >>> img = Image('lenna')
          >>> plt.bar(*img.vertical_histogram(threshold=128, bins=10,
              ...                            normalize=False, for_plot=True),
              ...     color='y')
          >>> plt.show()


        **NOTES**

        See: http://docs.scipy.org/doc/numpy/reference/generated/
        numpy.histogram.html
        See: http://matplotlib.org/api/
        pyplot_api.html?highlight=hist#matplotlib.pyplot.hist

        """
        if bins <= 0:
            raise Exception("Not enough bins")

        img = self.get_gray_ndarray()
        pts = np.where(img > threshold)
        y = pts[1]
        hist = np.histogram(y, bins=bins, range=(0, self.height),
                            normed=normalize)
        ret_val = None
        if for_plot:
            # for using matplotlib bar command
            # bin labels, bin values, bin width
            ret_val = (hist[1][0:-1], hist[0], self.height / bins)
        else:
            ret_val = hist[0]
        return ret_val

    def horizontal_histogram(self, bins=10, threshold=128, normalize=False,
                             for_plot=False):
        """

        **DESCRIPTION**

        This method generates histogram of the number of grayscale pixels
        greater than the provided threshold. The method divides the image
        into a number evenly spaced horizontal bins and then counts the number
        of pixels where the pixel is greater than the threshold. This method
        is helpful for doing basic morphological analysis.

        **PARAMETERS**

        * *bins* - The number of bins to use.
        * *threshold* - The grayscale threshold. We count pixels greater than
          this value.
        * *normalize* - If normalize is true we normalize the bin counts to sum
          to one. Otherwise we return the number of pixels.
        * *for_plot* - If this is true we return the bin indicies, the bin
          counts, and the bin widths as a tuple. We can use these values in
          pyplot.bar to quickly plot the histogram.


        **RETURNS**

        The default settings return the raw bin counts moving from top to
        bottom on the image. If for_plot is true we return a tuple that
        contains a list of bin labels, the bin counts, and the bin widths.
        This tuple can be used to plot the histogram using
        matplotlib.pyplot.bar function.

        **EXAMPLE**

        >>>> import matplotlib.pyplot as plt
        >>>> img = Image('lenna')
        >>>> plt.bar(img.horizontal_histogram(threshold=128, bins=10,
        ...                                   normalize=False, for_plot=True),
        ...          color='y')
        >>>> plt.show())

        **NOTES**

        See: http://docs.scipy.org/doc/numpy/reference/generated/
        numpy.histogram.html
        See: http://matplotlib.org/api/
        pyplot_api.html?highlight=hist#matplotlib.pyplot.hist

        """
        if bins <= 0:
            raise Exception("Not enough bins")

        img = self.get_gray_ndarray()
        pts = np.where(img > threshold)
        x = pts[0]
        hist = np.histogram(x, bins=bins, range=(0, self.width),
                            normed=normalize)
        ret_val = None
        if for_plot:
            # for using matplotlib bar command
            # bin labels, bin values, bin width
            ret_val = (hist[1][0:-1], hist[0], self.width / bins)
        else:
            ret_val = hist[0]
        return ret_val

    def get_line_scan(self, x=None, y=None, pt1=None, pt2=None, channel=-1):
        """
        **SUMMARY**

        This function takes in a channel of an image or grayscale by default
        and then pulls out a series of pixel values as a linescan object
        than can be manipulated further.

        **PARAMETERS**

        * *x* - Take a vertical line scan at the column x.
        * *y* - Take a horizontal line scan at the row y.
        * *pt1* - Take a line scan between two points on the line the line
          scan values always go in the +x direction
        * *pt2* - Second parameter for a non-vertical or horizontal line scan.
        * *channel* - To select a channel. eg: selecting a channel RED,GREEN
          or BLUE. If set to -1 it operates with gray scale values


        **RETURNS**

        A SimpleCV.LineScan object or None if the method fails.

        **EXAMPLE**

        >>>> import matplotlib.pyplot as plt
        >>>> img = Image('lenna')
        >>>> a = img.get_line_scan(x=10)
        >>>> b = img.get_line_scan(y=10)
        >>>> c = img.get_line_scan(pt1 = (10,10), pt2 = (500,500))
        >>>> plt.plot(a)
        >>>> plt.plot(b)
        >>>> plt.plot(c)
        >>>> plt.show()

        """

        if channel == -1:
            img = self.get_gray_ndarray()
        else:
            try:
                img = self._ndarray[:, :, channel]
            except IndexError:
                print 'Channel missing!'
                return None

        ret_val = None
        if x is not None and y is None and pt1 is None and pt2 is None:
            if x >= 0 and x < self.width:
                ret_val = LineScan(img[x, :])
                ret_val.image = self
                ret_val.pt1 = (x, 0)
                ret_val.pt2 = (x, self.height)
                ret_val.col = x
                x = np.ones((1, self.height))[0] * x
                y = range(0, self.height, 1)
                pts = zip(x, y)
                ret_val.point_loc = pts
            else:
                warnings.warn(
                    "ImageClass.get_line_scan - that is not valid scanline.")
                return None

        elif x is None and y is not None and pt1 is None and pt2 is None:
            if y >= 0 and y < self.height:
                ret_val = LineScan(img[:, y])
                ret_val.image = self
                ret_val.pt1 = (0, y)
                ret_val.pt2 = (self.width, y)
                ret_val.row = y
                y = np.ones((1, self.width))[0] * y
                x = range(0, self.width, 1)
                pts = zip(x, y)
                ret_val.point_loc = pts

            else:
                warnings.warn("ImageClass.get_line_scan - "
                              "that is not valid scanline.")
                return None

        elif isinstance(pt1, (tuple, list)) and isinstance(pt2, (tuple, list))\
                and len(pt1) == 2 and len(pt2) == 2 \
                and x is None and y is None:

            pts = self.bresenham_line(pt1, pt2)
            ret_val = LineScan([img[p[0], p[1]] for p in pts])
            ret_val.point_loc = pts
            ret_val.image = self
            ret_val.pt1 = pt1
            ret_val.pt2 = pt2

        else:
            # an invalid combination - warn
            warnings.warn("ImageClass.get_line_scan - that is not valid "
                          "scanline.")
            return None
        ret_val.channel = channel
        return ret_val

    def set_line_scan(self, linescan, x=None, y=None, pt1=None, pt2=None,
                      channel=-1):
        """
        **SUMMARY**

        This function helps you put back the linescan in the image.

        **PARAMETERS**

        * *linescan* - LineScan object
        * *x* - put  line scan at the column x.
        * *y* - put line scan at the row y.
        * *pt1* - put line scan between two points on the line the line scan
          values always go in the +x direction
        * *pt2* - Second parameter for a non-vertical or horizontal line scan.
        * *channel* - To select a channel. eg: selecting a channel RED,GREEN
          or BLUE. If set to -1 it operates with gray scale values


        **RETURNS**

        A SimpleCV.Image

        **EXAMPLE**

        >>> img = Image('lenna')
        >>> a = img.get_line_scan(x=10)
        >>> for index in range(len(a)):
            ... a[index] = 0
        >>> newimg = img.set_line_scan(a, x=50)
        >>> newimg.show()
        # This will show you a black line in column 50.

        """
        #retVal = self.to_gray()
        if channel == -1:
            img = np.copy(self.get_gray_ndarray())
        else:
            try:
                img = np.copy(self.get_ndarray()[:, :, channel])
            except IndexError:
                warnings.warn('Channel missing!')
                return None

        if x is None and y is None and pt1 is None and pt2 is None:
            if linescan.pt1 is None or linescan.pt2 is None:
                warnings.warn("ImageClass.set_line_scan: No coordinates to "
                              "re-insert linescan.")
                return None
            else:
                pt1 = linescan.pt1
                pt2 = linescan.pt2
                if pt1[0] == pt2[0] and np.abs(pt1[1] - pt2[1]) == self.height:
                    x = pt1[0]  # vertical line
                    pt1 = None
                    pt2 = None

                elif pt1[1] == pt2[1] \
                        and np.abs(pt1[0] - pt2[0]) == self.width:
                    y = pt1[1]  # horizontal line
                    pt1 = None
                    pt2 = None

        ret_val = None
        if x is not None and y is None and pt1 is None and pt2 is None:
            if x >= 0 and x < self.width:
                if len(linescan) != self.height:
                    linescan = linescan.resample(self.height)
                #check for number of points
                #linescan = np.array(linescan)
                img[x, :] = np.clip(linescan[:], 0, 255)
            else:
                warnings.warn("ImageClass.set_line_scan: No coordinates to "
                              "re-insert linescan.")
                return None
        elif x is None and y is not None and pt1 is None and pt2 is None:
            if y >= 0 and y < self.height:
                if len(linescan) != self.width:
                    linescan = linescan.resample(self.width)
                #check for number of points
                #linescan = np.array(linescan)
                img[:, y] = np.clip(linescan[:], 0, 255)
            else:
                warnings.warn("ImageClass.set_line_scan: No coordinates to "
                              "re-insert linescan.")
                return None
        elif isinstance(pt1, (tuple, list)) and isinstance(pt2, (tuple, list))\
                and len(pt1) == 2 and len(pt2) == 2 \
                and x is None and y is None:

            pts = self.bresenham_line(pt1, pt2)
            if len(linescan) != len(pts):
                linescan = linescan.resample(len(pts))
            #linescan = np.array(linescan)
            linescan = np.clip(linescan[:], 0, 255)
            idx = 0
            for pt in pts:
                img[pt[0], pt[1]] = linescan[idx]
                idx = idx + 1
        else:
            warnings.warn("ImageClass.set_line_scan: No coordinates to "
                          "re-insert linescan.")
            return None
        if channel == -1:
            ret_val = Image(img)
        else:
            temp = np.copy(self.get_ndarray())
            temp[:, :, channel] = img
            ret_val = Image(temp)
        return ret_val

    def replace_line_scan(self, linescan, x=None, y=None, pt1=None, pt2=None,
                          channel=None):
        """

        **SUMMARY**

        This function easily lets you replace the linescan in the image.
        Once you get the LineScan object, you might want to edit it. Perform
        some task, apply some filter etc and now you want to put it back where
        you took it from. By using this function, it is not necessary to
        specify where to put the data. It will automatically replace where you
        took the LineScan from.

        **PARAMETERS**

        * *linescan* - LineScan object
        * *x* - put  line scan at the column x.
        * *y* - put line scan at the row y.
        * *pt1* - put line scan between two points on the line the line scan
          values always go in the +x direction
        * *pt2* - Second parameter for a non-vertical or horizontal line scan.
        * *channel* - To select a channel. eg: selecting a channel RED,GREEN
          or BLUE. If set to -1 it operates with gray scale values


        **RETURNS**

        A SimpleCV.Image

        **EXAMPLE**

        >>> img = Image('lenna')
        >>> a = img.get_line_scan(x=10)
        >>> for index in range(len(a)):
            ... a[index] = 0
        >>> newimg = img.replace_line_scan(a)
        >>> newimg.show()
        # This will show you a black line in column 10.

        """

        if x is None and y is None \
                and pt1 is None and pt2 is None and channel is None:

            if linescan.channel == -1:
                img = np.copy(self.get_gray_ndarray())
            else:
                try:
                    img = np.copy(self.get_ndarray()[:, :, linescan.channel])
                except IndexError:
                    print 'Channel missing!'
                    return None

            if linescan.row is not None:
                if len(linescan) == self.width:
                    ls = np.clip(linescan, 0, 255)
                    img[:, linescan.row] = ls[:]
                else:
                    warnings.warn("LineScan Size and Image size do not match")
                    return None

            elif linescan.col is not None:
                if len(linescan) == self.height:
                    ls = np.clip(linescan, 0, 255)
                    img[linescan.col, :] = ls[:]
                else:
                    warnings.warn("LineScan Size and Image size do not match")
                    return None
            elif linescan.pt1 and linescan.pt2:
                pts = self.bresenham_line(linescan.pt1, linescan.pt2)
                if len(linescan) != len(pts):
                    linescan = linescan.resample(len(pts))
                ls = np.clip(linescan[:], 0, 255)
                idx = 0
                for pt in pts:
                    img[pt[0], pt[1]] = ls[idx]
                    idx = idx + 1

            if linescan.channel == -1:
                ret_val = Image(img)
            else:
                temp = np.copy(self.get_ndarray())
                temp[:, :, linescan.channel] = img
                ret_val = Image(temp)

        else:
            if channel is None:
                ret_val = self.set_line_scan(linescan, x, y, pt1, pt2,
                                             linescan.channel)
            else:
                ret_val = self.set_line_scan(linescan, x, y, pt1, pt2, channel)
        return ret_val

    def get_pixels_online(self, pt1, pt2):
        """
        **SUMMARY**

        Return all of the pixels on an arbitrary line.

        **PARAMETERS**

        * *pt1* - The first pixel coordinate as an (x,y) tuple or list.
        * *pt2* - The second pixel coordinate as an (x,y) tuple or list.

        **RETURNS**

        Returns a list of RGB pixels values.

        **EXAMPLE**

        >>>> img = Image('something.png')
        >>>> img.get_pixels_online( (0,0), (img.width/2,img.height/2) )
        """
        ret_val = None
        if isinstance(pt1, (tuple, list)) and isinstance(pt2, (tuple, list)) \
                and len(pt1) == 2 and len(pt2) == 2:
            pts = self.bresenham_line(pt1, pt2)
            ret_val = [self.get_pixel(p[0], p[1]) for p in pts]
        else:
            warnings.warn("ImageClass.get_pixels_online - The line you "
                          "provided is not valid")

        return ret_val

    def bresenham_line(self, (x, y), (x2, y2)):
        """
        Brensenham line algorithm

        cribbed from: http://snipplr.com/view.php?codeview&id=22482

        This is just a helper method
        """
        if not 0 <= x <= self.width - 1 \
                or not 0 <= y <= self.height - 1 \
                or not 0 <= x2 <= self.width - 1 \
                or not 0 <= y2 <= self.height - 1:
            l = Line(self, ((x, y), (x2, y2))).crop_to_image_edges()
            if l:
                ep = list(l.end_points)
                ep.sort()
                x, y = ep[0]
                x2, y2 = ep[1]
            else:
                return []

        steep = 0
        coords = []
        dx = abs(x2 - x)
        if (x2 - x) > 0:
            sx = 1
        else:
            sx = -1
        dy = abs(y2 - y)
        if (y2 - y) > 0:
            sy = 1
        else:
            sy = -1
        if dy > dx:
            steep = 1
            x, y = y, x
            dx, dy = dy, dx
            sx, sy = sy, sx
        d = (2 * dy) - dx
        for i in range(0, dx):
            if steep:
                coords.append((y, x))
            else:
                coords.append((x, y))
            while d >= 0:
                y = y + sy
                d = d - (2 * dx)
            x = x + sx
            d = d + (2 * dy)
        coords.append((x2, y2))
        return coords

    def uncrop(self, list_of_pts):  # (x,y),(x2,y2)):
        """
        **SUMMARY**

        This function allows us to translate a set of points from the crop
        window back to the coordinate of the source window.

        **PARAMETERS**

        * *list_of_pts* - set of points from cropped image.

        **RETURNS**

        Returns a list of coordinates in the source image.

        **EXAMPLE**

        >> img = Image('lenna')
        >> croppedImg = img.crop(10,20,250,500)
        >> sourcePts = croppedImg.uncrop([(2,3),(56,23),(24,87)])
        """
        return [(i[0] + self._uncroppedX, i[1] + self._uncroppedY) for i in
                list_of_pts]

    def grid(self, dimensions=(10, 10), color=(0, 0, 0), width=1,
             antialias=True, alpha=-1):
        """
        **SUMMARY**

        Draw a grid on the image

        **PARAMETERS**

        * *dimensions* - No of rows and cols as an (rows,xols) tuple or list.
        * *color* - Grid's color as a tuple or list.
        * *width* - The grid line width in pixels.
        * *antialias* - Draw an antialiased object
        * *aplha* - The alpha blending for the object. If this value is -1 then
          the layer default value is used. A value of 255 means opaque, while
          0 means transparent.

        **RETURNS**

        Returns the index of the drawing layer of the grid

        **EXAMPLE**

        >>>> img = Image('something.png')
        >>>> img.grid([20, 20], (255, 0, 0))
        >>>> img.grid((20, 20), (255, 0, 0), 1, True, 0)
        """
        ret_val = self.copy()
        try:
            step_row = self.size()[1] / dimensions[0]
            step_col = self.size()[0] / dimensions[1]
        except ZeroDivisionError:
            return None

        i = 1
        j = 1

        grid = DrawingLayer(self.size())  # add a new layer for grid
        while (i < dimensions[0]) and (j < dimensions[1]):
            if i < dimensions[0]:
                grid.line((0, step_row * i), (self.size()[0], step_row * i),
                          color, width, antialias, alpha)
                i = i + 1
            if j < dimensions[1]:
                grid.line((step_col * j, 0), (step_col * j, self.size()[1]),
                          color, width, antialias, alpha)
                j = j + 1
        # store grid layer index
        ret_val._gridLayer[0] = ret_val.add_drawing_layer(grid)
        ret_val._gridLayer[1] = dimensions
        return ret_val

    def remove_grid(self):

        """
        **SUMMARY**

                Remove Grid Layer from the Image.

        **PARAMETERS**

                None

        **RETURNS**

                Drawing Layer corresponding to the Grid Layer

        **EXAMPLE**

        >>>> img = Image('something.png')
        >>>> img.grid([20,20],(255,0,0))
        >>>> gridLayer = img.remove_grid()

        """

        if self._gridLayer[0] is not None:
            grid = self.remove_drawing_layer(self._gridLayer[0])
            self._gridLayer = [None, [0, 0]]
            return grid
        else:
            return None

    def find_grid_lines(self):

        """
        **SUMMARY**

        Return Grid Lines as a Line Feature Set

        **PARAMETERS**

        None

        **RETURNS**

        Grid Lines as a Feature Set

        **EXAMPLE**

        >>>> img = Image('something.png')
        >>>> img.grid([20,20],(255,0,0))
        >>>> lines = img.find_grid_lines()

        """

        grid_index = self.get_drawing_layer(self._gridLayer[0])
        if self._gridLayer[0] == -1:
            print "Cannot find grid on the image, Try adding a grid first"

        line_fs = FeatureSet()
        try:
            step_row = self.size()[1] / self._gridLayer[1][0]
            step_col = self.size()[0] / self._gridLayer[1][1]
        except ZeroDivisionError:
            return None

        i = 1
        j = 1

        while i < self._gridLayer[1][0]:
            line_fs.append(Line(self, ((0, step_row * i),
                                      (self.size()[0], step_row * i))))
            i = i + 1
        while j < self._gridLayer[1][1]:
            line_fs.append(Line(self, ((step_col * j, 0),
                                      (step_col * j, self.size()[1]))))
            j = j + 1

        return line_fs

    def logical_and(self, img, grayscale=True):
        """
        **SUMMARY**

        Perform bitwise AND operation on images

        **PARAMETERS**

        img - the bitwise operation to be performed with
        grayscale

        **RETURNS**

        SimpleCV.ImageClass.Image

        **EXAMPLE**

        >>> img = Image("something.png")
        >>> img1 = Image("something_else.png")
        >>> img.logical_and(img1, grayscale=False)
        >>> img.logical_and(img1)

        """
        if self.size() != img.size():
            logger.warning("Both images must have same sizes")
            return None
        if grayscale:
            retval = cv2.bitwise_and(self.get_gray_ndarray(),
                                     img.get_gray_ndarray())
        else:
            retval = cv2.bitwise_and(self.get_ndarray(), img.get_ndarray())
        return Image(retval)

    def logical_nand(self, img, grayscale=True):
        """
        **SUMMARY**

        Perform bitwise NAND operation on images

        **PARAMETERS**

        img - the bitwise operation to be performed with
        grayscale

        **RETURNS**

        SimpleCV.ImageClass.Image

        **EXAMPLE**

        >>> img = Image("something.png")
        >>> img1 = Image("something_else.png")
        >>> img.logical_nand(img1, grayscale=False)
        >>> img.logical_nand(img1)

        """
        if self.size() != img.size():
            logger.warning("Both images must have same sizes")
            return None
        if grayscale:
            retval = cv2.bitwise_and(self.get_gray_ndarray(),
                                     img.get_gray_ndarray())
        else:
            retval = cv2.bitwise_and(self.get_ndarray(), img.get_ndarray())
        retval = cv2.bitwise_not(retval)
        return Image(retval)

    def logical_or(self, img, grayscale=True):
        """
        **SUMMARY**

        Perform bitwise OR operation on images

        **PARAMETERS**

        img - the bitwise operation to be performed with
        grayscale

        **RETURNS**

        SimpleCV.ImageClass.Image

        **EXAMPLE**

        >>> img = Image("something.png")
        >>> img1 = Image("something_else.png")
        >>> img.logical_or(img1, grayscale=False)
        >>> img.logical_or(img1)

        """
        if self.size() != img.size():
            logger.warning("Both images must have same sizes")
            return None
        if grayscale:
            retval = cv2.bitwise_or(self.get_gray_ndarray(),
                                    img.get_gray_ndarray())
        else:
            retval = cv2.bitwise_or(self.get_ndarray(), img.get_ndarray())
        return Image(retval)

    def logical_xor(self, img, grayscale=True):
        """
        **SUMMARY**

        Perform bitwise XOR operation on images

        **PARAMETERS**

        img - the bitwise operation to be performed with
        grayscale

        **RETURNS**

        SimpleCV.ImageClass.Image

        **EXAMPLE**

        >>> img = Image("something.png")
        >>> img1 = Image("something_else.png")
        >>> img.logical_xor(img1, grayscale=False)
        >>> img.logical_xor(img1)

        """
        if self.size() != img.size():
            logger.warning("Both images must have same sizes")
            return None
        if grayscale:
            retval = cv2.bitwise_xor(self.get_gray_ndarray(),
                                     img.get_gray_ndarray())
        else:
            retval = cv2.bitwise_xor(self.get_ndarray(), img.get_ndarray())
        return Image(retval)

    def match_sift_key_points(self, template, quality=200):
        """
        **SUMMARY**

        matchSIFTKeypoint allows you to match a template image with another
        image using SIFT keypoints. The method extracts keypoints from each
        image, uses the Fast Local Approximate Nearest Neighbors algorithm to
        find correspondences between the feature points, filters the
        correspondences based on quality. This method should be able to handle
        a reasonable changes in camera orientation and illumination. Using a
        template that is close to the target image will yield much better
        results.

        **PARAMETERS**

        * *template* - A template image.
        * *quality* - The feature quality metric. This can be any value
          between about 100 and 500. Lower values should return fewer, but
          higher quality features.

        **RETURNS**

        A Tuple of lists consisting of matched KeyPoints found on the image
        and matched keypoints found on the template. keypoints are sorted
        according to lowest distance.

        **EXAMPLE**

        >>> camera = Camera()
        >>> template = Image("template.png")
        >>> img = camera.get_image()
        >>> fs = img.match_sift_key_points(template)

        **SEE ALSO**

        :py:meth:`_get_raw_keypoints`
        :py:meth:`_get_flann_matches`
        :py:meth:`draw_keypoint_matches`
        :py:meth:`find_keypoints`

        """
        if not hasattr(cv2, "FeatureDetector_create"):
            warnings.warn("OpenCV >= 2.4.3 required")
            return None
        if template is None:
            return None
        detector = cv2.FeatureDetector_create("SIFT")
        descriptor = cv2.DescriptorExtractor_create("SIFT")
        img = self.get_ndarray()
        template_img = template.get_ndarray()

        skp = detector.detect(img)
        skp, sd = descriptor.compute(img, skp)

        tkp = detector.detect(template_img)
        tkp, td = descriptor.compute(template_img, tkp)

        idx, dist = self._get_flann_matches(sd, td)
        dist = dist[:, 0] / 2500.0
        dist = dist.reshape(-1, ).tolist()
        idx = idx.reshape(-1).tolist()
        indices = range(len(dist))
        indices.sort(key=lambda i: dist[i])
        dist = [dist[i] for i in indices]
        idx = [idx[i] for i in indices]
        sfs = []
        for i, dis in itertools.izip(idx, dist):
            if dis < quality:
                sfs.append(KeyPoint(template, skp[i], sd, "SIFT"))
            else:
                break  # since sorted

        idx, dist = self._get_flann_matches(td, sd)
        dist = dist[:, 0] / 2500.0
        dist = dist.reshape(-1, ).tolist()
        idx = idx.reshape(-1).tolist()
        indices = range(len(dist))
        indices.sort(key=lambda i: dist[i])
        dist = [dist[i] for i in indices]
        idx = [idx[i] for i in indices]
        tfs = []
        for i, dis in itertools.izip(idx, dist):
            if dis < quality:
                tfs.append(KeyPoint(template, tkp[i], td, "SIFT"))
            else:
                break

        return sfs, tfs

    def draw_sift_key_point_match(self, template, distance=200, num=-1,
                                  width=1):
        """
        **SUMMARY**

        Draw SIFT keypoints draws a side by side representation of two images,
        calculates keypoints for both images, determines the keypoint
        correspondences, and then draws the correspondences. This method is
        helpful for debugging keypoint calculations and also looks really
        cool :) The parameters mirror the parameters used for
        findKeypointMatches to assist with debugging

        **PARAMETERS**

        * *template* - A template image.
        * *distance* - This can be any value between about 100 and 500. Lower
                       value should return less number of features but higher
                       quality features.
        * *num* -   Number of features you want to draw. Features are sorted
                    according to the dist from min to max.
        * *width* - The width of the drawn line.

        **RETURNS**

        A side by side image of the template and source image with each feature
        correspondence draw in a different color.

        **EXAMPLE**

        >>> cam = Camera()
        >>> img = cam.get_image()
        >>> template = Image("myTemplate.png")
        >>> result = img.draw_sift_key_point_match(template, 300.00):

        **SEE ALSO**

        :py:meth:`draw_keypoint_matches`
        :py:meth:`find_keypoints`
        :py:meth:`find_keypoint_match`

        """
        if template is None:
            return
        result_img = template.side_by_side(self, scale=False)
        hdif = (self.height - template.height) / 2
        sfs, tfs = self.match_sift_key_points(template, distance)
        maxlen = min(len(sfs), len(tfs))
        if num < 0 or num > maxlen:
            num = maxlen
        for i in range(num):
            skp = sfs[i]
            tkp = tfs[i]
            pt_a = (int(tkp.y), int(tkp.x) + hdif)
            pt_b = (int(skp.y) + template.width, int(skp.x))
            result_img.draw_line(pt_a, pt_b, color=Color.get_random(),
                                 thickness=width)
        return result_img

    def stega_encode(self, message):
        """
        **SUMMARY**

        A simple steganography tool for hidding messages in images.
        **PARAMETERS**

        * *message* -A message string that you would like to encode.

        **RETURNS**

        Your message encoded in the returning image.

        **EXAMPLE**

        >>>> img = Image('lenna')
        >>>> img2 = img.stega_encode("HELLO WORLD!")
        >>>> img2.save("TopSecretImg.png")
        >>>> img3 = Image("TopSecretImg.png")
        >>>> img3.stega_decode()

        **NOTES**

        More here:
        http://en.wikipedia.org/wiki/Steganography
        You will need to install stepic:
        http://domnit.org/stepic/doc/pydoc/stepic.html

        You may need to monkey with jpeg compression
        as it seems to degrade the encoded message.

        PNG sees to work quite well.

        """

        try:
            import stepic
        except ImportError:
            logger.warning("stepic library required")
            return None
        warnings.simplefilter("ignore")
        pil_img = PilImage.frombuffer("RGB", self.size(), self.to_string())
        stepic.encode_inplace(pil_img, message)
        ret_val = Image(pil_img)
        return ret_val.flip_vertical()

    def stega_decode(self):
        """
        **SUMMARY**

        A simple steganography tool for hidding and finding
        messages in images.

        **RETURNS**

        Your message decoded in the image.

        **EXAMPLE**

        >>>> img = Image('lenna')
        >>>> img2 = img.stega_encode("HELLO WORLD!")
        >>>> img2.save("TopSecretImg.png")
        >>>> img3 = Image("TopSecretImg.png")
        >>>> img3.stega_decode()

        **NOTES**

        More here:
        http://en.wikipedia.org/wiki/Steganography
        You will need to install stepic:
        http://domnit.org/stepic/doc/pydoc/stepic.html

        You may need to monkey with jpeg compression
        as it seems to degrade the encoded message.

        PNG sees to work quite well.

        """
        try:
            import stepic
        except ImportError:
            logger.warning("stepic library required")
            return None
        warnings.simplefilter("ignore")
        pil_img = PilImage.frombuffer("RGB", self.size(), self.to_string())
        result = stepic.decode(pil_img)
        return result

    def find_features(self, method="szeliski", threshold=1000):
        """
        **SUMMARY**

        Find szeilski or Harris features in the image.
        Harris features correspond to Harris corner detection in the image.

        Read more:

        Harris Features: http://en.wikipedia.org/wiki/Corner_detection
        szeliski Features: http://research.microsoft.com/en-us/um/people/
        szeliski/publications.htm

        **PARAMETERS**

        * *method* - Features type
        * *threshold* - threshold val

        **RETURNS**

        A list of Feature objects corrseponding to the feature points.

        **EXAMPLE**

        >>> img = Image("corner_sample.png")
        >>> fpoints = img.find_features("harris", 2000)
        >>> for f in fpoints:
            ... f.draw()
        >>> img.show()

        **SEE ALSO**

        :py:meth:`draw_keypoint_matches`
        :py:meth:`find_keypoints`
        :py:meth:`find_keypoint_match`

        """
        img = self.get_gray_ndarray()
        blur = cv2.GaussianBlur(img, (3, 3), 0)

        ix = cv2.Sobel(blur, cv2.CV_32F, 1, 0)
        iy = cv2.Sobel(blur, cv2.CV_32F, 0, 1)

        ix_ix = np.multiply(ix, ix)
        iy_iy = np.multiply(iy, iy)
        ix_iy = np.multiply(ix, iy)

        ix_ix_blur = cv2.GaussianBlur(ix_ix, (5, 5), 0)
        iy_iy_blur = cv2.GaussianBlur(iy_iy, (5, 5), 0)
        ix_iy_blur = cv2.GaussianBlur(ix_iy, (5, 5), 0)

        harris_thresh = threshold * 5000
        alpha = 0.06
        det_a = ix_ix_blur * iy_iy_blur - ix_iy_blur ** 2
        trace_a = ix_ix_blur + iy_iy_blur
        feature_list = []
        if method == "szeliski":
            harmonic_mean = det_a / trace_a
            for j, i in np.argwhere(harmonic_mean > threshold):
                feature_list.append(
                    Feature(self, i, j, ((i, j), (i, j), (i, j), (i, j))))

        elif method == "harris":
            harris_function = det_a - (alpha * trace_a * trace_a)
            for j, i in np.argwhere(harris_function > harris_thresh):
                feature_list.append(
                    Feature(self, i, j, ((i, j), (i, j), (i, j), (i, j))))
        else:
            logger.warning("Invalid method.")
            return None
        return feature_list

    def watershed(self, mask=None, erode=2, dilate=2, use_my_mask=False):
        """
        **SUMMARY**

        Implements the Watershed algorithm on the input image.

        Read more:

        Watershed: "http://en.wikipedia.org/wiki/Watershed_(image_processing)"

        **PARAMETERS**

        * *mask* - an optional binary mask. If none is provided we do a
          binarize and invert.
        * *erode* - the number of times to erode the mask to find the
          foreground.
        * *dilate* - the number of times to dilate the mask to find possible
          background.
        * *use_my_mask* - if this is true we do not modify the mask.

        **RETURNS**

        The Watershed image

        **EXAMPLE**

        >>> img = Image("/data/sampleimages/wshed.jpg")
        >>> img1 = img.watershed()
        >>> img1.show()

        # here is an example of how to create your own mask

        >>> img = Image('lenna')
        >>> myMask = Image((img.width, img.height))
        >>> myMask = myMask.flood_fill((0, 0), color=Color.WATERSHED_BG)
        >>> mask = img.threshold(128)
        >>> myMask = (myMask - mask.dilate(2) + mask.erode(2))
        >>> result = img.watershed(mask=myMask, use_my_mask=True)

        **SEE ALSO**
        Color.WATERSHED_FG - The watershed foreground color
        Color.WATERSHED_BG - The watershed background color
        Color.WATERSHED_UNSURE - The watershed not sure if fg or bg color.

        TODO: Allow the user to pass in a function that defines the watershed
        mask.
        """

        output = self.get_empty(3)
        if mask is None:
            mask = self.binarize().invert()
        newmask = None
        if not use_my_mask:
            newmask = Image((self.width, self.height))
            newmask = newmask.flood_fill((0, 0), color=Color.WATERSHED_BG)
            dilate_erode_sum = mask.dilate(dilate) + mask.erode(erode)
            newmask = (newmask - dilate_erode_sum.to_bgr())
        else:
            newmask = mask
        m = np.int32(newmask.get_gray_ndarray())
        cv2.watershed(self._ndarray, m)
        m = cv2.convertScaleAbs(m)
        ret, thresh = cv2.threshold(m, 0, 255, cv2.cv.CV_THRESH_OTSU)
        ret_val = Image(thresh)
        return ret_val

    def find_blobs_from_watershed(self, mask=None, erode=2, dilate=2,
                                  use_my_mask=False, invert=False, minsize=20,
                                  maxsize=None):
        """
        **SUMMARY**

        Implements the watershed algorithm on the input image with an optional
        mask and then uses the mask to find blobs.

        Read more:

        Watershed: "http://en.wikipedia.org/wiki/Watershed_(image_processing)"

        **PARAMETERS**

        * *mask* - an optional binary mask. If none is provided we do a
          binarize and invert.
        * *erode* - the number of times to erode the mask to find the
          foreground.
        * *dilate* - the number of times to dilate the mask to find possible
          background.
        * *use_my_mask* - if this is true we do not modify the mask.
        * *invert* - invert the resulting mask before finding blobs.
        * *minsize* - minimum blob size in pixels.
        * *maxsize* - the maximum blob size in pixels.

        **RETURNS**

        A feature set of blob features.

        **EXAMPLE**

        >>> img = Image("/data/sampleimages/wshed.jpg")
        >>> mask = img.threshold(100).dilate(3)
        >>> blobs = img.find_blobs_from_watershed(mask)
        >>> blobs.show()

        **SEE ALSO**
        Color.WATERSHED_FG - The watershed foreground color
        Color.WATERSHED_BG - The watershed background color
        Color.WATERSHED_UNSURE - The watershed not sure if fg or bg color.

        """
        newmask = self.watershed(mask, erode, dilate, use_my_mask)
        if invert:
            newmask = mask.invert()
        return self.find_blobs_from_mask(newmask, minsize=minsize,
                                         maxsize=maxsize)

    def max_value(self, locations=False):
        """
        **SUMMARY**
        Returns the brightest/maximum pixel value in the
        grayscale image. This method can also return the
        locations of pixels with this value.

        **PARAMETERS**

        * *locations* - If true return the location of pixels
           that have this value.

        **RETURNS**

        The maximum value and optionally the list of points as
        a list of (x,y) tuples.

        **EXAMPLE**

        >>> img = Image("lenna")
        >>> max = img.max_value()
        >>> min, pts = img.min_value(locations=True)
        >>> img2 = img.stretch(min,max)

        """
        if locations:
            val = np.max(self.get_gray_ndarray())
            x, y = np.where(self.get_gray_ndarray() == val)
            locs = zip(x.tolist(), y.tolist())
            return int(val), locs
        else:
            val = np.max(self.get_gray_ndarray())
            return int(val)

    def min_value(self, locations=False):
        """
        **SUMMARY**
        Returns the darkest/minimum pixel value in the
        grayscale image. This method can also return the
        locations of pixels with this value.

        **PARAMETERS**

        * *locations* - If true return the location of pixels
           that have this value.

        **RETURNS**

        The minimum value and optionally the list of points as
        a list of (x,y) tuples.

        **EXAMPLE**

        >>> img = Image("lenna")
        >>> max = img.max_value()
        >>> min, pts = img.min_value(locations=True)
        >>> img2 = img.stretch(min,max)

        """
        if locations:
            val = np.min(self.get_gray_ndarray())
            x, y = np.where(self.get_gray_ndarray() == val)
            locs = zip(x.tolist(), y.tolist())
            return int(val), locs
        else:
            val = np.min(self.get_gray_ndarray())
            return int(val)

    def find_keypoint_clusters(self, num_of_clusters=5, order='dsc',
                               flavor='surf'):
        '''
        This function is meant to try and find interesting areas of an
        image. It does this by finding keypoint clusters in an image.
        It uses keypoint (ORB) detection to locate points of interest
        and then uses kmeans clustering to get the X,Y coordinates of
        those clusters of keypoints. You provide the expected number
        of clusters and you will get back a list of the X,Y coordinates
        and rank order of the number of Keypoints around those clusters

        **PARAMETERS**
        * num_of_clusters - The number of clusters you are looking for
          (default: 5)
        * order - The rank order you would like the points returned in, dsc or
          asc, (default: dsc)
        * flavor - The keypoint type, or 'corner' for just corners


        **EXAMPLE**

        >>> img = Image('simplecv')
        >>> clusters = img.find_keypoint_clusters()
        >>> clusters.draw()
        >>> img.show()

        **RETURNS**

        FeatureSet
        '''
        if flavor.lower() == 'corner':
            keypoints = self.find_corners()  # fallback to corners
        else:
            keypoints = self.find_keypoints(
                flavor=flavor.upper())  # find the keypoints
        if keypoints is None or keypoints <= 0:
            return None

        xypoints = np.array([(f.x, f.y) for f in keypoints])
        # find the clusters of keypoints
        xycentroids, xylabels = scv.kmeans2(xypoints, num_of_clusters)
        xycounts = np.array([])

        # count the frequency of occurences for sorting
        for i in range(num_of_clusters):
            xycounts = np.append(xycounts, len(np.where(xylabels == i)[-1]))

        # sort based on occurence
        merged = np.msort(np.hstack((np.vstack(xycounts), xycentroids)))
        clusters = [c[1:] for c in
                    merged]  # strip out just the values ascending
        if order.lower() == 'dsc':
            clusters = clusters[::-1]  # reverse if descending

        fs = FeatureSet()
        for x, y in clusters:  # map the values to a feature set
            f = Corner(self, x, y)
            fs.append(f)

        return fs

    def get_freak_descriptor(self, flavor="SURF"):
        """
        **SUMMARY**

        Compute FREAK Descriptor of given keypoints.
        FREAK - Fast Retina Keypoints.
        Read more: http://www.ivpe.com/freak.htm

        Keypoints can be extracted using following detectors.

        - SURF
        - SIFT
        - BRISK
        - ORB
        - STAR
        - MSER
        - FAST
        - Dense

        **PARAMETERS**

        * *flavor* - Detector (see above list of detectors) - string

        **RETURNS**

        * FeatureSet* - A feature set of KeyPoint Features.
        * Descriptor* - FREAK Descriptor

        **EXAMPLE**

        >>> img = Image("lenna")
        >>> fs, des = img.get_freak_descriptor("ORB")

        """
        if cv2.__version__.startswith('$Rev:'):
            warnings.warn("OpenCV version >= 2.4.2 requierd")
            return None

        if int(cv2.__version__.replace('.', '0')) < 20402:
            warnings.warn("OpenCV version >= 2.4.2 requierd")
            return None

        flavors = ["SIFT", "SURF", "BRISK", "ORB", "STAR", "MSER", "FAST",
                   "Dense"]
        if flavor not in flavors:
            warnings.warn("Unkown Keypoints detector. Returning None.")
            return None
        detector = cv2.FeatureDetector_create(flavor)
        extractor = cv2.DescriptorExtractor_create("FREAK")
        self._mKeyPoints = detector.detect(self.get_gray_ndarray())
        self._mKeyPoints, self._mKPDescriptors = extractor.compute(
            self.get_gray_ndarray(),
            self._mKeyPoints)
        fs = FeatureSet()
        for i in range(len(self._mKeyPoints)):
            fs.append(KeyPoint(self, self._mKeyPoints[i],
                               self._mKPDescriptors[i], flavor))

        return fs, self._mKPDescriptors

    def get_gray_histogram_counts(self, bins=255, limit=-1):
        '''
        This function returns a list of tuples of greyscale pixel counts
        by frequency.  This would be useful in determining the dominate
        pixels (peaks) of the greyscale image.

        **PARAMETERS**

        * *bins* - The number of bins for the hisogram, defaults to 255
          (greyscale)
        * *limit* - The number of counts to return, default is all

        **RETURNS**

        * List * - A list of tuples of (frequency, value)

        **EXAMPLE**

        >>> img = Image("lenna")
        >>> counts = img.get_gray_histogram_counts()
        >>> # the most dominate pixel color tuple of frequency and value
        >>> counts[0]
        >>> counts[1][1] # the second most dominate pixel color value
        '''

        hist = self.histogram(bins)
        vals = [(e, h) for h, e in enumerate(hist)]
        vals.sort()
        vals.reverse()

        if limit == -1:
            limit = bins

        return vals[:limit]

    def gray_peaks(self, bins=255, delta=0, lookahead=15):
        """
        **SUMMARY**

        Takes the histogram of a grayscale image, and returns the peak
        grayscale intensity values.

        The bins parameter can be used to lump grays together, by default it is
        set to 255

        Returns a list of tuples, each tuple contains the grayscale intensity,
        and the fraction of the image that has it.

        **PARAMETERS**

        * *bins* - the integer number of bins, between 1 and 255.

        * *delta* - the minimum difference betweena peak and the following
                    points, before a peak may be considered a peak.Useful to
                    hinder the algorithm from picking up false peaks towards
                    to end of the signal.

        * *lookahead* - the distance to lookahead from a peakto determine if it
                        is an actual peak, should be an integer greater than 0.

        **RETURNS**

        A list of (grays,fraction) tuples.

        **NOTE**

        Implemented using the techniques used in huetab()

        """

        # The bins are the no of edges bounding an histogram.
        # Thus bins= Number of bars in histogram+1
        # As range() function is exclusive,
        # hence bins+2 is passed as parameter.

        y_axis, x_axis = np.histogram(self.get_gray_ndarray(),
                                      bins=range(bins + 2))
        x_axis = x_axis[0:bins + 1]
        maxtab = []
        mintab = []
        length = len(y_axis)
        if x_axis is None:
            x_axis = range(length)

        #perform some checks
        if length != len(x_axis):
            raise ValueError("Input vectors y_axis and x_axis must have "
                             "same length")
        if lookahead < 1:
            raise ValueError("Lookahead must be above '1' in value")
        if not (np.isscalar(delta) and delta >= 0):
            raise ValueError("delta must be a positive number")

        #needs to be a numpy array
        y_axis = np.asarray(y_axis)

        #maxima and minima candidates are temporarily stored in
        #mx and mn respectively
        mn, mx = np.Inf, -np.Inf

        #Only detect peak if there is 'lookahead' amount of points after it
        for index, (x, y) in enumerate(zip(x_axis[:-lookahead],
                                           y_axis[:-lookahead])):
            if y > mx:
                mx = y
                mxpos = x
            if y < mn:
                mn = y
                mnpos = x

            ####look for max####
            if y < mx - delta and mx != np.Inf:
                # Maxima peak candidate found
                # look ahead in signal to ensure that this
                # is a peak and not jitter
                if y_axis[index:index + lookahead].max() < mx:
                    maxtab.append((mxpos, mx))
                    #set algorithm to only find minima now
                    mx = np.Inf
                    mn = np.Inf

            if y > mn + delta and mn != -np.Inf:
                # Minima peak candidate found
                # look ahead in signal to ensure that
                # this is a peak and not jitter
                if y_axis[index:index + lookahead].min() > mn:
                    mintab.append((mnpos, mn))
                    #set algorithm to only find maxima now
                    mn = -np.Inf
                    mx = -np.Inf

        ret_val = []
        for intensity, pixelcount in maxtab:
            ret_val.append(
                (intensity, pixelcount / float(self.width * self.height)))
        return ret_val

    def tv_denoising(self, gray=False, weight=50, eps=0.0002, max_iter=200,
                     resize=1):
        """
        **SUMMARY**

        Performs Total Variation Denoising, this filter tries to minimize the
        total-variation of the image.

        see : http://en.wikipedia.org/wiki/Total_variation_denoising

        **Parameters**

        * *gray* - Boolean value which identifies the colorspace of
            the input image. If set to True, filter uses gray scale values,
            otherwise colorspace is used.

        * *weight* - Denoising weight, it controls the extent of denoising.

        * *eps* - Stopping criteria for the algorithm. If the relative
          difference of the cost function becomes less than this value, the
          algorithm stops.

        * *max_iter* - Determines the maximum number of iterations the
          algorithm goes through for optimizing.

        * *resize* - Parameter to scale up/down the image. If set to
            1 filter is applied on the original image. This parameter is
            mostly to speed up the filter.

        **NOTE**

        This function requires Scikit-image library to be installed!
        To install scikit-image library run::

            sudo pip install -U scikit-image

        Read More: http://scikit-image.org/

        """

        try:
            from skimage.filter import denoise_tv_chambolle
        except ImportError:
            logger.warn('Scikit-image Library not installed!')
            return None

        img = self.copy()

        if resize <= 0:
            print 'Enter a valid resize value'
            return None

        if resize != 1:
            img = img.resize(int(img.width * resize), int(img.height * resize))

        if gray is True:
            img = img.get_gray_numpy()
            multichannel = False
        elif gray is False:
            img = img.get_ndarray()
            multichannel = True
        else:
            warnings.warn('gray value not valid')
            return None

        denoise_mat = denoise_tv_chambolle(img, weight, eps, max_iter,
                                           multichannel)
        ret_val = img * denoise_mat

        ret_val = Image(ret_val)
        if resize != 1:
            return ret_val.resize(int(ret_val.width / resize),
                                  int(ret_val.width / resize))
        else:
            return ret_val

    # FIXME: following functions should be merged
    def motion_blur(self, intensity=15, direction='NW'):
        """
        **SUMMARY**

        Performs the motion blur of an Image. Uses different filters to find
        out the motion blur in different directions.

        see : https://en.wikipedia.org/wiki/Motion_blur

        **Parameters**

        * *intensity* - The intensity of the motion blur effect. Basically
           defines the size of the filter used in the process. It has to be an
           integer. 0 intensity implies no blurring.

        * *direction* - The direction of the motion. It is a string taking
            values left, right, up, down as well as N, S, E, W for north,
            south, east, west and NW, NE, SW, SE for northwest and so on.
            default is NW

        **RETURNS**

        An image with the specified motion blur filter applied.

        **EXAMPLE**
        >>> i = Image ('lenna')
        >>> mb = i.motion_blur()
        >>> mb.show()

        """
        mid = int(intensity / 2)
        tmp = np.identity(intensity)

        if intensity == 0:
            warnings.warn("0 intensity means no blurring")
            return self

        elif intensity % 2 is 0:
            div = mid
            for i in range(mid, intensity - 1):
                tmp[i][i] = 0
        else:
            div = mid + 1
            for i in range(mid + 1, intensity - 1):
                tmp[i][i] = 0

        if direction == 'right' or direction.upper() == 'E':
            kernel = np.concatenate(
                (np.zeros((1, mid)), np.ones((1, mid + 1))), axis=1)
        elif direction == 'left' or direction.upper() == 'W':
            kernel = np.concatenate(
                (np.ones((1, mid + 1)), np.zeros((1, mid))), axis=1)
        elif direction == 'up' or direction.upper() == 'N':
            kernel = np.concatenate(
                (np.ones((1 + mid, 1)), np.zeros((mid, 1))), axis=0)
        elif direction == 'down' or direction.upper() == 'S':
            kernel = np.concatenate(
                (np.zeros((mid, 1)), np.ones((mid + 1, 1))), axis=0)
        elif direction.upper() == 'NW':
            kernel = tmp
        elif direction.upper() == 'NE':
            kernel = np.fliplr(tmp)
        elif direction.upper() == 'SW':
            kernel = np.flipud(tmp)
        elif direction.upper() == 'SE':
            kernel = np.flipud(np.fliplr(tmp))
        else:
            warnings.warn("Please enter a proper direction")
            return None

        retval = self.convolve(kernel=kernel / div)
        return retval

    def motion_blur2(self, intensity=15, angle=0):
        """
        **SUMMARY**

        Performs the motion blur of an Image given the intensity and angle

        see : https://en.wikipedia.org/wiki/Motion_blur

        **Parameters**

        * *intensity* - The intensity of the motion blur effect. Governs the
            size of the kernel used in convolution

        * *angle* - Angle in degrees at which motion blur will occur. Positive
            is Clockwise and negative is Anti-Clockwise. 0 blurs from left to
            right


        **RETURNS**

        An image with the specified motion blur applied.

        **EXAMPLE**
        >>> img = Image ('lenna')
        >>> blur = img.motion_blur(40,45)
        >>> blur.show()

        """

        intensity = int(intensity)

        if intensity <= 1:
            logger.warning('power less than 1 will result in no change')
            return self

        kernel = np.zeros((intensity, intensity))

        rad = math.radians(angle)
        x1, y1 = intensity / 2, intensity / 2

        x2 = int(x1 - (intensity - 1) / 2 * math.sin(rad))
        y2 = int(y1 - (intensity - 1) / 2 * math.cos(rad))

        line = self.bresenham_line((x1, y1), (x2, y2))

        x = [p[0] for p in line]
        y = [p[1] for p in line]

        kernel[x, y] = 1
        kernel = kernel / len(line)
        return self.convolve(kernel=kernel)

    def recognize_face(self, recognizer=None):
        """
        **SUMMARY**

        Find faces in the image using FaceRecognizer and predict their class.

        **PARAMETERS**

        * *recognizer*   - Trained FaceRecognizer object

        **EXAMPLES**

        >>> cam = Camera()
        >>> img = cam.get_image()
        >>> recognizer = FaceRecognizer()
        >>> recognizer.load("training.xml")
        >>> print img.recognize_face(recognizer)
        """
        if not hasattr(cv2, "createFisherFaceRecognizer"):
            warnings.warn("OpenCV >= 2.4.4 required to use this.")
            return None

        if not isinstance(recognizer, FaceRecognizer):
            warnings.warn("SimpleCV.Features.FaceRecognizer object required.")
            return None

        w, h = recognizer.image_size
        label = recognizer.predict(self.resize(w, h))
        return label

    def find_and_recognize_faces(self, recognizer, cascade=None):
        """
        **SUMMARY**

        Predict the class of the face in the image using FaceRecognizer.

        **PARAMETERS**

        * *recognizer*  - Trained FaceRecognizer object

        * *cascade*     -haarcascade which would identify the face
                         in the image.

        **EXAMPLES**

        >>> cam = Camera()
        >>> img = cam.get_image()
        >>> recognizer = FaceRecognizer()
        >>> recognizer.load("training.xml")
        >>> feat = img.find_and_recognize_faces(recognizer, "face.xml")
        >>> for feature, label, confidence in feat:
            ... i = feature.crop()
            ... i.draw_text(str(label))
            ... i.show()
        """
        if not hasattr(cv2, "createFisherFaceRecognizer"):
            warnings.warn("OpenCV >= 2.4.4 required to use this.")
            return None

        if not isinstance(recognizer, FaceRecognizer):
            warnings.warn("SimpleCV.Features.FaceRecognizer object required.")
            return None

        if not cascade:
            cascade = os.path.join(LAUNCH_PATH,
                                   'data/Features/HaarCascades/face.xml')

        faces = self.find_haar_features(cascade)
        if not faces:
            warnings.warn("Faces not found in the image.")
            return None

        ret_val = []
        for face in faces:
            label, confidence = face.crop().recognize_face(recognizer)
            ret_val.append([face, label, confidence])
        return ret_val

    def channel_mixer(self, channel='r', weight=(100, 100, 100)):
        """
        **SUMMARY**

        Mixes channel of an RGB image based on the weights provided. The output
        is given at the channel provided in the parameters. Basically alters
        the value of one channelg of an RGB image based in the values of other
        channels and itself. If the image is not RGB then first converts the
        image to RGB and then mixes channel

        **PARAMETERS**

        * *channel* - The output channel in which the values are to be
        replaced. It can have either 'r' or 'g' or 'b'

        * *weight* - The weight of each channel in calculation of the mixed
        channel. It is a tuple having 3 values mentioning the percentage of the
        value of the channels, from -200% to 200%

        **RETURNS**

        A SimpleCV RGB Image with the provided channel replaced with the mixed
        channel.

        **EXAMPLE**

        >>> img = Image("lenna")
        >>> img2 = img.channel_mixer()
        >>> Img3 = img.channel_mixer(channel='g', weights=(3, 2, 1))

        **NOTE**

        Read more at http://docs.gimp.org/en/plug-in-colors-channel-mixer.html

        """
        r, g, b = self.split_channels()
        if weight[0] > 200 or weight[1] > 200 or weight[2] >= 200:
            if weight[0] < -200 or weight[1] < -200 or weight[2] < -200:
                warnings.warn('Value of weights can be from -200 to 200%')
                return None

        weight = map(float, weight)
        channel = channel.lower()
        if channel == 'r':
            r = r * (weight[0] / 100.0) + \
                g * (weight[1] / 100.0) + \
                b * (weight[2] / 100.0)
        elif channel == 'g':
            g = r * (weight[0] / 100.0) + \
                g * (weight[1] / 100.0) + \
                b * (weight[2] / 100.0)
        elif channel == 'b':
            b = r * (weight[0] / 100.0) + \
                g * (weight[1] / 100.0) + \
                b * (weight[2] / 100.0)
        else:
            warnings.warn('Please enter a valid channel(r/g/b)')
            return None

        ret_val = self.merge_channels(r=r, g=g, b=b)
        return ret_val

    def prewitt(self):
        """
        **SUMMARY**

        Prewitt operator for edge detection

        **PARAMETERS**

        None

        **RETURNS**

        Image with prewitt opeartor applied on it

        **EXAMPLE**

        >>> img = Image("lenna")
        >>> p = img.prewitt()
        >>> p.show()

        **NOTES**

        Read more at: http://en.wikipedia.org/wiki/Prewitt_operator

        """
        img = self.copy()
        grayimg = img.to_gray()
        gx = [[1, 1, 1], [0, 0, 0], [-1, -1, -1]]
        gy = [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]
        grayx = grayimg.convolve(gx)
        grayy = grayimg.convolve(gy)
        grayxnp = np.uint64(grayx.get_gray_ndarray())
        grayynp = np.uint64(grayy.get_gray_ndarray())
        ret_val = Image(np.sqrt(grayxnp ** 2 + grayynp ** 2))
        return ret_val

    def edge_snap(self, point_list, step=1):
        """
        **SUMMARY**

        Given a List of points finds edges closet to the line joining two
        successive points, edges are returned as a FeatureSet of
        Lines.

        Note : Image must be binary, it is assumed that prior conversion is
        done

        **Parameters**

       * *point_list* - List of points to be checked for nearby edges.

        * *step* - Number of points to skip if no edge is found in vicinity.
                   Keep this small if you want to sharply follow a curve

        **RETURNS**

        * FeatureSet * - A FeatureSet of Lines

        **EXAMPLE**

        >>> image = Image("logo").edges()
        >>> edgeLines = image.edge_snap([(50, 50), (230, 200)])
        >>> edgeLines.draw(color=Color.YELLOW, width=3)
        """
        img_array = self.get_gray_ndarray()
        c1 = np.count_nonzero(img_array)
        c2 = np.count_nonzero(img_array - 255)

        #checking that all values are 0 and 255
        if c1 + c2 != img_array.size:
            raise ValueError("Image must be binary")

        if len(point_list) < 2:
            return None

        final_list = [point_list[0]]
        feature_set = FeatureSet()
        last = point_list[0]
        for point in point_list[1:None]:
            final_list += self._edge_snap2(last, point, step)
            last = point

        last = final_list[0]
        for point in final_list:
            feature_set.append(Line(self, (last, point)))
            last = point
        return feature_set

    def _edge_snap2(self, start, end, step):
        """
        **SUMMARY**

        Given a two points returns a list of edge points closet to the line
        joining the points. Point is a tuple of two numbers

        Note : Image must be binary

        **Parameters**

        * *start* - First Point

        * *end* - Second Point

        * *step* - Number of points to skip if no edge is found in vicinity
                   Keep this low to detect sharp curves

        **RETURNS**

        * List * - A list of tuples , each tuple contains (x,y) values

        """

        edge_map = np.copy(self.get_gray_ndarray())

        #Size of the box around a point which is checked for edges.
        box = step * 4

        xmin = min(start[0], end[0])
        xmax = max(start[0], end[0])
        ymin = min(start[1], end[1])
        ymax = max(start[1], end[1])

        line = self.bresenham_line(start, end)

        #List of Edge Points.
        final_list = []
        i = 0

        #Closest any point has ever come to the end point
        overall_min_dist = None

        while i < len(line):

            x, y = line[i]

            #Get the matrix of points fromx around current point.
            region = edge_map[x - box:x + box, y - box:y + box]

            #Condition at the boundary of the image
            if region.shape[0] == 0 or region.shape[1] == 0:
                i += step
                continue

            #Index of all Edge points
            index_list = np.argwhere(region > 0)
            if index_list.size > 0:

                #Center the coordinates around the point
                index_list -= box
                min_dist = None

                # Incase multiple edge points exist, choose the one closest
                # to the end point
                for ix, iy in index_list:
                    dist = math.hypot(x + ix - end[0], iy + y - end[1])
                    if min_dist is None or dist < min_dist:
                        dx, dy = ix, iy
                        min_dist = dist

                # The distance of the new point is compared with the least
                # distance computed till now, the point is rejected if it's
                # comparitively more. This is done so that edge points don't
                # wrap around a curve instead of heading towards the end point
                if overall_min_dist is not None \
                        and min_dist > overall_min_dist * 1.1:
                    i += step
                    continue

                if overall_min_dist is None or min_dist < overall_min_dist:
                    overall_min_dist = min_dist

                # Reset the points in the box so that they are not detected
                # during the next iteration.
                edge_map[x - box:x + box, y - box:y + box] = 0

                # Keep all the points in the bounding box
                if xmin <= x + dx <= xmax and ymin <= y + dx <= ymax:
                    #Add the point to list and redefine the line
                    line = [(x + dx, y + dy)] \
                        + self.bresenham_line((x + dx, y + dy), end)
                    final_list += [(x + dx, y + dy)]

                    i = 0

            i += step
        final_list += [end]
        return final_list

    def get_lightness(self):
        """
        **SUMMARY**

        This method converts the given RGB image to grayscale using the
        Lightness method.

        **Parameters**

        None

        **RETURNS**

        A GrayScale image with values according to the Lightness method

        **EXAMPLE**
        >>> img = Image ('lenna')
        >>> out = img.get_lightness()
        >>> out.show()

        **NOTES**

        Algorithm used: value = (MAX(R,G,B) + MIN(R,G,B))/2

        """
        if self.is_bgr() or self._colorSpace == ColorSpace.UNKNOWN:
            img_mat = np.array(self._ndarray, dtype=np.int)
            ret_val = np.array((np.max(img_mat, 2) + np.min(img_mat, 2)) / 2,
                               dtype=np.uint8)
        else:
            logger.warnings('Input a RGB image')
            return None

        return Image(ret_val)

    def get_luminosity(self):
        """
        **SUMMARY**

        This method converts the given RGB image to grayscale using the
        Luminosity method.

        **Parameters**

        None

        **RETURNS**

        A GrayScale image with values according to the Luminosity method

        **EXAMPLE**
        >>> img = Image ('lenna')
        >>> out = img.get_luminosity()
        >>> out.show()

        **NOTES**

        Algorithm used: value =  0.21 R + 0.71 G + 0.07 B

        """
        if self.is_bgr() or self._colorSpace == ColorSpace.UNKNOWN:
            img_mat = np.array(self._ndarray, dtype=np.int)
            ret_val = np.array(np.average(img_mat, 2, (0.07, 0.71, 0.21)),
                               dtype=np.uint8)
        else:
            logger.warnings('Input a RGB image')
            return None

        return Image(ret_val)

    def get_average(self):
        """
        **SUMMARY**

        This method converts the given RGB image to grayscale by averaging out
        the R,G,B values.

        **Parameters**

        None

        **RETURNS**

        A GrayScale image with values according to the Average method

        **EXAMPLE**
        >>> img = Image ('lenna')
        >>> out = img.get_average()
        >>> out.show()

        **NOTES**

        Algorithm used: value =  (R+G+B)/3

        """
        if self.is_bgr() or self._colorSpace == ColorSpace.UNKNOWN:
            img_mat = np.array(self._ndarray, dtype=np.int)
            ret_val = np.array(img_mat.mean(2), dtype=np.uint8)
        else:
            logger.warnings('Input a RGB image')
            return None

        return Image(ret_val)

    def smart_rotate(self, bins=18, point=[-1, -1], auto=True, threshold=80,
                     min_length=30, max_gap=10, t1=150, t2=200, fixed=True):
        """
        **SUMMARY**

        Attempts to rotate the image so that the most significant lines are
        approximately parellel to horizontal or vertical edges.

        **Parameters**


        * *bins* - The number of bins the lines will be grouped into.

        * *point* - the point about which to rotate, refer :py:meth:`rotate`

        * *auto* - If true point will be computed to the mean of centers of all
            the lines in the selected bin. If auto is True, value of point is
            ignored

        * *threshold* - which determines the minimum "strength" of the line
            refer :py:meth:`find_lines` for details.

        * *min_length* - how many pixels long the line must be to be returned,
            refer :py:meth:`find_lines` for details.

        * *max_gap* - how much gap is allowed between line segments to consider
            them the same line .refer to :py:meth:`find_lines` for details.

        * *t1* - thresholds used in the edge detection step,
            refer to :py:meth:`_get_edge_map` for details.

        * *t2* - thresholds used in the edge detection step,
            refer to :py:meth:`_get_edge_map` for details.

        * *fixed* - if fixed is true,keep the original image dimensions,
            otherwise scale the image to fit the rotation , refer to
            :py:meth:`rotate`

        **RETURNS**

        A rotated image

        **EXAMPLE**
        >>> i = Image ('image.jpg')
        >>> i.smart_rotate().show()

        """
        lines = self.find_lines(threshold, min_length, max_gap, t1, t2)

        if len(lines) == 0:
            logger.warning("No lines found in the image")
            return self

        # Initialize empty bins
        binn = [[] for i in range(bins)]

        #Convert angle to bin number
        conv = lambda x: int(x + 90) / bins

        #Adding lines to bins
        [binn[conv(line.get_angle())].append(line) for line in lines]

        #computing histogram, value of each column is total length of all lines
        #in the bin
        hist = [sum([line.length() for line in lines]) for lines in binn]

        #The maximum histogram
        index = np.argmax(np.array(hist))

        #Good ol weighted mean, for the selected bin
        avg = sum([line.get_angle() * line.length() for line in binn[index]]) \
            / sum([line.length() for line in binn[index]])

        #Mean of centers of all lines in selected bin
        if auto:
            x = sum([line.end_points[0][0] + line.end_points[1][0]
                     for line in binn[index]]) / 2 / len(binn[index])
            y = sum([line.end_points[0][1] + line.end_points[1][1]
                     for line in binn[index]]) / 2 / len(binn[index])
            point = [x, y]

        #Determine whether to rotate the lines to vertical or horizontal
        if -45 <= avg <= 45:
            return self.rotate(avg, fixed=fixed, point=point)
        elif avg > 45:
            return self.rotate(avg - 90, fixed=fixed, point=point)
        else:
            return self.rotate(avg + 90, fixed=fixed, point=point)
            #Congratulations !! You did a smart thing

    def normalize(self, new_min=0, new_max=255, min_cut=2, max_cut=98):
        """
        **SUMMARY**

        Performs image normalization and yeilds a linearly normalized gray
        image. Also known as contrast strestching.

        see : http://en.wikipedia.org/wiki/Normalization_(image_processing)

        **Parameters**

        * *new_min* - The minimum of the new range over which the image is
        normalized

        * *new_max* - The maximum of the new range over which the image is
        normalized

        * *min_cut* - A number between 0 to 100. The threshold percentage
        for the current minimum value selection. This helps us to avoid the
        effect of outlying pixel with either very low value

        * *max_cut* - A number between 0 to 100. The threshold percentage for
        the current minimum value selection. This helps us to avoid the effect
        of outlying pixel with either very low value

        **RETURNS**

        A normalized grayscale image.

        **EXAMPLE**
        >>> img = Image('lenna')
        >>> norm = img.normalize()
        >>> norm.show()

        """
        if new_min < 0 or new_max > 255:
            warnings.warn("new_min and new_max can vary from 0-255")
            return None
        if new_max < new_min:
            warnings.warn("new_min should be less than new_max")
            return None
        if min_cut > 100 or max_cut > 100:
            warnings.warn("min_cut and max_cut")
            return None
        #avoiding the effect of odd pixels
        try:
            hist = self.get_gray_histogram_counts()
            freq, val = zip(*hist)
            maxfreq = (freq[0] - freq[-1]) * max_cut / 100.0
            minfreq = (freq[0] - freq[-1]) * min_cut / 100.0
            closest_match = lambda a, l: min(l, key=lambda x: abs(x - a))
            maxval = closest_match(maxfreq, val)
            minval = closest_match(minfreq, val)
            ret_val = (self.grayscale() - minval) \
                * ((new_max - new_min) / float(maxval - minval)) + new_min
        #catching zero division in case there are very less intensities present
        #Normalizing based on absolute max and min intensities present
        except ZeroDivisionError:
            maxval = self.max_value()
            minval = self.min_value()
            ret_val = (self.grayscale() - minval) \
                * ((new_max - new_min) / float(maxval - minval)) + new_min
        #catching the case where there is only one intensity throughout
        except:
            warnings.warn(
                "All pixels of the image have only one intensity value")
            return None
        return ret_val

    def get_normalized_hue_histogram(self, roi=None):
        """
        **SUMMARY**

        This method generates a normalized hue histogram for the image
        or the ROI within the image. The hue histogram is a 2D hue/saturation
        numpy array histogram with a shape of 180x256. This histogram can
        be used for histogram back projection.

        **PARAMETERS**

        * *roi* - Anything that can be cajoled into being an ROI feature
          including a tuple of (x,y,w,h), a list of points, or another feature.

        **RETURNS**

        A normalized 180x256 numpy array that is the hue histogram.

        **EXAMPLE**

        >>> img = Image('lenna')
        >>> roi = (0,0,100,100)
        >>> hist = img.get_normalized_hue_histogram(roi)

        **SEE ALSO**

        ImageClass.back_project_hue_histogram()
        ImageClass.find_blobs_from_hue_histogram()

        """
        from simplecv.features.detection import ROI

        if roi:  # roi is anything that can be taken to be an roi
            roi = ROI(roi, self)
            hsv = roi.crop().to_hsv().get_ndarray()
        else:
            hsv = self.to_hsv().get_ndarray()
        hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
        return hist

    def back_project_hue_histogram(self, model, smooth=True, full_color=False,
                                   threshold=None):
        """
        **SUMMARY**

        This method performs hue histogram back projection on the image. This
        is a very quick and easy way of matching objects based on color. Given
        a hue histogram taken from another image or an roi within the image we
        attempt to find all pixels that are similar to the colors inside the
        histogram. The result can either be a grayscale image that shows the
        matches or a color image.


        **PARAMETERS**

        * *model* - The histogram to use for pack projection. This can either
        be a histogram, anything that can be converted into an ROI for the
        image (like an x,y,w,h tuple or a feature, or another image.

        * *smooth* - A bool, True means apply a smoothing operation after doing
        the back project to improve the results.

        * *full_color* - return the results as a color image where pixels
        included in the back projection are rendered as their source colro.

        * *threshold* - If this value is not None, we apply a threshold to the
        result of back projection to yield a binary image. Valid values are
        from 1 to 255.

        **RETURNS**

        A SimpleCV Image rendered according to the parameters provided.

        **EXAMPLE**

        >>>> img = Image('lenna')

        Generate a hist

        >>>> hist = img.get_normalized_hue_histogram((0, 0, 50, 50))
        >>>> a = img.back_project_hue_histogram(hist)
        >>>> b = img.back_project_hue_histogram((0, 0, 50, 50))  # same result
        >>>> c = img.back_project_hue_histogram(Image('lyle'))

        **SEE ALSO**
        ImageClass.get_normalized_hue_histogram()
        ImageClass.find_blobs_from_hue_histogram()

        """
        if model is None:
            warnings.warn('Backproject requires a model')
            return None
        # this is the easier test, try to cajole model into ROI
        if isinstance(model, Image):
            model = model.get_normalized_hue_histogram()
        if not isinstance(model, np.ndarray) or model.shape != (180, 256):
            model = self.get_normalized_hue_histogram(model)
        if isinstance(model, np.ndarray) and model.shape == (180, 256):
            hsv = self.to_hsv().get_ndarray()
            dst = cv2.calcBackProject(
                [hsv], [0, 1], model, [0, 180, 0, 256], 1)
            if smooth:
                disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                cv2.filter2D(dst, -1, disc, dst)
            result = Image(dst)
            result = result.to_bgr()
            if threshold:
                result = result.threshold(threshold)
            if full_color:
                temp = Image((self.width, self.height))
                result = temp.blit(self, alpha_mask=result)
            return result
        else:
            warnings.warn('Backproject model does not appear to be valid')
            return None

    def find_blobs_from_hue_histogram(self, model, threshold=1, smooth=True,
                                      minsize=10, maxsize=None):
        """
        **SUMMARY**

        This method performs hue histogram back projection on the image and
        uses the results to generate a FeatureSet of blob objects. This is a
        very quick and easy way of matching objects based on color. Given a hue
        histogram taken from another image or an roi within the image we
        attempt to find all pixels that are similar to the colors inside the
        histogram.

        **PARAMETERS**

        * *model* - The histogram to use for pack projection. This can either
        be a histogram, anything that can be converted into an ROI for the
        image (like an x,y,w,h tuple or a feature, or another image.

        * *smooth* - A bool, True means apply a smoothing operation after doing
        the back project to improve the results.

        * *threshold* - If this value is not None, we apply a threshold to the
        result of back projection to yield a binary image. Valid values are
        from 1 to 255.

        * *minsize* - the minimum blob size in pixels.

        * *maxsize* - the maximum blob size in pixels.

        **RETURNS**

        A FeatureSet of blob objects or None if no blobs are found.

        **EXAMPLE**

        >>>> img = Image('lenna')

        Generate a hist

        >>>> hist = img.get_normalized_hue_histogram((0, 0, 50, 50))
        >>>> blobs = img.find_blobs_from_hue_histogram(hist)
        >>>> blobs.show()

        **SEE ALSO**

        ImageClass.get_normalized_hue_histogram()
        ImageClass.back_project_hue_histogram()

        """
        new_mask = self.back_project_hue_histogram(model, smooth,
                                                   full_color=False,
                                                   threshold=threshold)
        return self.find_blobs_from_mask(new_mask, minsize=minsize,
                                         maxsize=maxsize)

    def filter(self, flt, grayscale=False):
        """
        **SUMMARY**

        This function allows you to apply an arbitrary filter to the DFT of an
        image. This filter takes in a gray scale image, whiter values are kept
        and black values are rejected. In the DFT image, the lower frequency
        values are in the corners of the image, while the higher frequency
        components are in the center. For example, a low pass filter has white
        squares in the corners and is black everywhere else.

        **PARAMETERS**

        * *flt* - A DFT filter

        * *grayscale* - if this value is True we perfrom the operation on the
        DFT of the gray version of the image and the result is gray image. If
        grayscale is true we perform the operation on each channel and the
        recombine them to create the result.

        **RETURNS**

        A SimpleCV image after applying the filter.

        **EXAMPLE**

        >>>  filter = DFT.create_gaussian_filter()
        >>>  myImage = Image("MyImage.png")
        >>>  result = myImage.filter(filter)
        >>>  result.show()
        """
        filteredimage = flt.apply_filter(self, grayscale)
        return filteredimage

# FIXME: circular import
from simplecv.features.detection import Barcode, Corner, HaarFeature, Line,\
    Chessboard, TemplateMatch, Circle, KeyPoint, Motion, KeypointMatch
from simplecv.features.facerecognizer import FaceRecognizer
from simplecv.features.blobmaker import BlobMaker
from simplecv.dft import DFT
