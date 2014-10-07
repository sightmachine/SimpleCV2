"""
Image loaders.
"""
import os
import random
import urllib2
import cStringIO
from simplecv.core.pluginsystem import plugin_list

try:
    from PIL import Image as PilImage
except:
    import Image as PilImage

import cv2
import numpy as np

from simplecv import DATA_DIR
from simplecv.core.image import Image


class ImageLoaderBase(object):

    @staticmethod
    def can_load(**kwargs):
        return False

    @staticmethod
    def load(**kwargs):
        raise NotImplementedError()


@plugin_list('loaders')
class ImageLoader(object):

    builtin_loaders = []
    loaders = []

    @staticmethod
    def register(loader_cls):
        ImageLoader.builtin_loaders.append(loader_cls)
        return loader_cls

    @staticmethod
    def load(**kwargs):
        for loader in ImageLoader.builtin_loaders:
            if loader.can_load(**kwargs):
                return loader.load(**kwargs)
        for loader in ImageLoader.loaders:
            if loader.can_load(**kwargs):
                return loader.load(**kwargs)

        msg = 'Cannot load image from {}'.format(kwargs.get('source'))
        raise ValueError(msg)


@ImageLoader.register
class Cv2ImageLoader(ImageLoaderBase):
    """
    Image loader that uses cv2 to load an image
    """

    # Based on
    # http://docs.opencv.org/modules/highgui/doc/
    # reading_and_writing_images_and_video.html#imread
    SUPPORTED_EXTENSIONS = (
        '.bmp', '.dib',
        '.jpeg', '.jpg', '.jpe', '.jp2',
        '.png',
        '.pbm', '.pgm', '.ppm',
        '.sr', '.ras',
        '.tiff', '.tif',
    )

    @staticmethod
    def can_load(**kwargs):
        source = kwargs.get('source')
        if isinstance(source, basestring):
            if source == '':
                return False
            elif not os.path.exists(source):
                return False

            # check extension
            fname, ext = os.path.splitext(source)
            if ext.lower() in Cv2ImageLoader.SUPPORTED_EXTENSIONS:
                return True
        return False

    @staticmethod
    def load(**kwargs):
        source = kwargs.get('source')
        color_space = kwargs.get('color_space')
        if (color_space == Image.GRAY):
            array = cv2.imread(source, 0)
        else:
            array = cv2.imread(source)
            color_space = Image.BGR
        if array is None:
            raise Exception('Failed to create an image array')
        return array, color_space, source


@ImageLoader.register
class SampleImageLoader(ImageLoaderBase):

    SUPPORTED_SAMPLE_IMAGES = (
        'simplecv',
        'logo',
        'simplecv_inverted',
        'inverted',
        'logo_inverted',
        'lenna',
        'lyle',
        'parity',
    )

    @staticmethod
    def can_load(**kwargs):
        source = kwargs.get('source')
        sample = kwargs.get('sample')
        if sample is True or \
                isinstance(source, basestring) and \
                source in SampleImageLoader.SUPPORTED_SAMPLE_IMAGES:
            return True
        else:
            return False

    @staticmethod
    def load(**kwargs):
        source = kwargs.get('source')
        sample = kwargs.get('sample')
        if isinstance(source, basestring):
            tmpname = source.lower()
            if tmpname == 'simplecv' or tmpname == 'logo':
                source = os.path.join(DATA_DIR, 'sampleimages', 'simplecv.png')
            elif tmpname == "simplecv_inverted" \
                    or tmpname == 'inverted' or tmpname == 'logo_inverted':
                source = os.path.join(DATA_DIR, 'sampleimages',
                                      'simplecv_inverted.png')
            elif tmpname == 'lenna':
                source = os.path.join(DATA_DIR, 'sampleimages', 'lenna.png')
            elif tmpname == 'lyle':
                source = os.path.join(DATA_DIR, 'sampleimages',
                                      'LyleJune1973.png')
            elif tmpname == 'parity':
                choice = random.choice(['LyleJune1973.png', 'lenna.png'])
                source = os.path.join(DATA_DIR, 'sampleimages', choice)

            elif sample is True:
                source = os.path.join(DATA_DIR, 'sampleimages', source)

            return Cv2ImageLoader.load(source=source)
        else:
            raise Exception('Cannot load image from {}'.format(source))


@ImageLoader.register
class HttpImageLoader(ImageLoaderBase):

    USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_7_4) " \
                 "AppleWebKit/536.5 (KHTML, like Gecko) " \
                 "Chrome/19.0.1084.54 Safari/536.5"

    @staticmethod
    def can_load(**kwargs):
        source = kwargs.get('source')
        if isinstance(source, basestring):
            source_lower = source.lower()
            if source_lower.startswith("http://") \
                    or source_lower.startswith("https://"):
                return True
        return False

    @staticmethod
    def load(**kwargs):
        source = kwargs.get('source')
        if isinstance(source, basestring):
            source_lower = source.lower()
            if source_lower.startswith("http://") \
                    or source_lower.startswith("https://"):
                headers = {'User-Agent': HttpImageLoader.USER_AGENT}
                req = urllib2.Request(source, headers=headers)
                img_file = urllib2.urlopen(req)
                img_array = np.asarray(bytearray(img_file.read()),
                                       dtype=np.uint8)
                array = cv2.imdecode(img_array, cv2.CV_LOAD_IMAGE_COLOR)
                if array is None:
                    raise Exception('Failed to create an image array')
                return array, Image.BGR, None

        raise Exception('Cannot load image from {}'.format(source))


@ImageLoader.register
class RawPngImageLoader(ImageLoaderBase):

    @staticmethod
    def can_load(**kwargs):
        source = kwargs.get('source')
        if isinstance(source, basestring) \
                and (source.lower().startswith("data:image/png;base64,")):
            return True
        else:
            return False

    @staticmethod
    def load(**kwargs):
        source = kwargs.get('source')
        if isinstance(source, basestring) \
                and (source.lower().startswith("data:image/png;base64,")):
            array = cv2.imdecode(np.fromstring(source),
                                 cv2.CV_LOAD_IMAGE_COLOR)
            if array is None:
                raise Exception('Failed to create an image array')
            return array, Image.BGR, None
        else:
            raise Exception('Cannot load image from {}'.format(source))


@ImageLoader.register
class ListTupleImageLoader(ImageLoaderBase):

    @staticmethod
    def can_load(**kwargs):
        source = kwargs.get('source')
        if isinstance(source, (tuple, list)) \
                and len(source) == 2:
            return True
        else:
            return False

    @staticmethod
    def load(**kwargs):
        source = kwargs.get('source')
        color_space = kwargs.get('color_space')
        if isinstance(source, (tuple, list)) \
                and len(source) == 2:
            w, h = map(int, source)
            if color_space == Image.GRAY:
                array = np.zeros((h, w), np.uint8)
            else:
                array = np.zeros((h, w, 3), np.uint8)
            return array, color_space, None
        else:
            raise Exception('Cannot load image from {}'.format(source))


@ImageLoader.register
class WebpImageLoader(ImageLoaderBase):

    @staticmethod
    def can_load(**kwargs):
        source = kwargs.get('source')
        webp = kwargs.get('webp')
        if webp is True:
            return True
        if isinstance(source, basestring):
            if source == '':
                return False
            elif not os.path.exists(source):
                return False

            # check extension
            fname, ext = os.path.splitext(source)
            if ext.lower() == '.webp':
                return True
        return False

    @staticmethod
    def load(**kwargs):
        source = kwargs.get('source')
        if isinstance(source, basestring):
            if source == '':
                raise IOError("No filename provided to Image constructor")
            elif not os.path.exists(source):
                raise IOError("Filename provided does not exist")

            try:
                from webm import decode as webm_decode
            except ImportError:
                msg = 'The webm module or latest PIL / PILLOW module ' \
                      'needs to be installed to load webp files: ' \
                      'https://github.com/sightmachine/python-webm'
                raise Exception(msg)

            with open(source, "rb") as f:
                webp_image_data = bytearray(f.read())

            result = webm_decode.DecodeRGB(webp_image_data)
            pil_img = PilImage.frombuffer(
                "RGB", (result.width, result.height),
                str(result.bitmap), "raw", "RGB", 0, 1)
            array = np.asarray(pil_img, dtype=np.uint8)
            return array, Image.RGB, source
        elif isinstance(source, cStringIO.InputType):
            source.seek(0)  # set the stringIO to the begining
            try:
                pil_img = PilImage.open(source)
            except:
                raise Exception('Failed to load webp image using PIL')
            array = np.asarray(pil_img, dtype=np.uint8)
            return array, Image.RGB, None
        else:
            raise Exception('Cannot load image from {}'.format(source))


@ImageLoader.register
class PilImageLoader(ImageLoaderBase):

    @staticmethod
    def can_load(**kwargs):
        source = kwargs.get('source')
        if isinstance(source, PilImage.Image):
            return True
        else:
            return False

    @staticmethod
    def load(**kwargs):
        source = kwargs.get('source')
        if isinstance(source, PilImage.Image):
            if source.mode != 'RGB':
                source = source.convert('RGB')
            array = np.asarray(source, dtype=np.uint8)
            return array, Image.RGB, None
        else:
            raise Exception('Cannot load image from {}'.format(source))
