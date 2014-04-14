import glob
import os
import re
import sys
import tempfile
import time
import types
import urllib2

from numpy import uint8
import cv2
import numpy as np


from simplecv import DATA_DIR
from simplecv.base import logger, int_to_bin, IMAGE_FORMATS
from simplecv.image import Image

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
            directory = os.path.join(DATA_DIR, 'sampleimages')
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

        def get_header_anim(image):
            """ Animation header. To replace the getheader()[0] """
            bb = "GIF89a"
            bb += int_to_bin(image.width)
            bb += int_to_bin(image.height)
            bb += "\x87\x00\x00"
            return bb

        for img in self:
            if not isinstance(img, PilImage.Image):
                pil_img = img.get_pil()
            else:
                pil_img = img

            converted.append((pil_img.convert('P', dither=dither),
                              get_header_anim(img)))

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
        accumulator = np.zeros((fh, fw, 3), dtype=np.uint8)
        alpha = float(1.0 / len(resized))
        beta = float((len(resized) - 1.0) / len(resized))
        for i in resized:
            accumulator = cv2.addWeighted(src1=i.get_ndarray(),
                                          alpha=alpha,
                                          src2=accumulator,
                                          beta=beta, gamma=0)
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
