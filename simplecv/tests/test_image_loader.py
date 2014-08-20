import numpy as np
from nose.tools import assert_equals, assert_is_none

from simplecv.core.image.loader import ImageLoader, Cv2ImageLoader, \
                                       SampleImageLoader, HttpImageLoader, \
                                       RawPngImageLoader, \
                                       ListTupleImageLoader, \
                                       WebpImageLoader, \
                                       PygameImageLoader, \
                                       PilImageLoader
from simplecv.image import Image
from simplecv.tests.utils import skipped


def test_Cv2ImageLoader():
    assert not Cv2ImageLoader.can_load()
    assert not Cv2ImageLoader.can_load(source="unknownimage.jpg")
    assert not Cv2ImageLoader.can_load(source="")
    assert Cv2ImageLoader.can_load(source="../data/sampleimages/lenna.png")

    data = Cv2ImageLoader.load(source="../data/sampleimages/lenna.png")
    array, colorspace, source = data

    assert_equals(array.shape, (512, 512, 3))
    assert_equals(colorspace, Image.BGR)

def test_SampleImageLoader():
    assert not SampleImageLoader.can_load()
    assert SampleImageLoader.can_load(source="lenna", sample=True)
    sample_images = SampleImageLoader.SUPPORTED_SAMPLE_IMAGES
    for img in sample_images:
        SampleImageLoader.load(source=img, sample=True)

# TODO: rewrite with mock
@skipped
def test_HttpImageLoader():
    assert HttpImageLoader.can_load(source="http://simplecv.org/sites/all/"\
                                    "themes/kalypso/images/SM_logo_color.png")

    assert HttpImageLoader.can_load(source="https://github.com/sightmachine/"\
                                    "SimpleCV2/blob/2.0/develop/simplecv/data/"
                                    "sampleimages/simplecv.png")

    assert not HttpImageLoader.can_load(source="../data/sampleimages/"\
                                        "simplecv.png")

    data = HttpImageLoader.load(source="https://raw.githubusercontent.com/"\
                                "sightmachine/SimpleCV2/2.0/develop/"\
                                "simplecv/data/sampleimages/lenna.png")
    array, colorspace, filename = data

    assert_equals(array.shape, (512, 512, 3))
    assert_equals(colorspace, Image.BGR)
"""
def test_RawPngImageLoader():
    f = open("../data/test/standard/raw_png_data_simplecv", "r")
    png_data = f.read()
    assert RawPngImageLoader.can_load(source=png_data)
    assert not RawPngImageLoader.can_load(source="unknown_source")

    data = RawPngImageLoader.load(source=png_data)
    array, colorspace, filename = data
    assert_equals(array.shape, (250, 250, 3))
    assert_equals(colorspace, Image.BGR)
"""

def test_ListTupleImageLoader():
    assert ListTupleImageLoader.can_load(source=(100, 100))
    assert ListTupleImageLoader.can_load(source=[100, 100])
    assert not ListTupleImageLoader.can_load(source=(10))
    assert not ListTupleImageLoader.can_load(source=(10, 10, 100))

    data = ListTupleImageLoader.load(source=(100, 80), color_space=Image.RGB)
    arr, color_space, filename = data
    assert_equals(arr.data, np.zeros((80, 100, 3), np.uint8).data)
    assert_equals(color_space, Image.RGB)

    data = ListTupleImageLoader.load(source=(100, 80), color_space=Image.GRAY)
    arr, color_space, filename = data
    assert_equals(arr.data, np.zeros((80, 100), np.uint8).data)
    assert_equals(color_space, Image.GRAY)

def test_WebpImageLoader():
    assert WebpImageLoader.can_load(source="../data/sampleimages/simplecv.webp",
                                webp=True)
    assert WebpImageLoader.can_load(source="../data/sampleimages/simplecv.webp")
    assert not WebpImageLoader.can_load(source="")
    assert not WebpImageLoader.can_load(source=np.zeros((10, 10)))
    assert not WebpImageLoader.can_load(source="../data/sampleimages/unknown.webp")

    data = WebpImageLoader.load(source="../data/sampleimages/simplecv.webp")
    array, color_space, filename = data
    assert_equals(array.shape, (250, 250, 3))
    assert_equals(color_space, Image.RGB)

def test_PygameImageLoader():
    img = Image("lenna")
    pg_surface = img.get_pg_surface()

    assert PygameImageLoader.can_load(source=pg_surface)
    assert not PygameImageLoader.can_load(source=img)

    data = PygameImageLoader.load(source=pg_surface)
    array, color_space, filename = data

    assert_equals(array.shape, (512, 512, 3))
    assert_equals(color_space, Image.RGB)

def test_PilImageLoader():
    img = Image("simplecv")
    pil_img = img.get_pil()

    assert PilImageLoader.can_load(source=pil_img)
    assert not PilImageLoader.can_load(source=img)

    data = PilImageLoader.load(source=pil_img)
    array, color_space, filename = data

    assert_equals(array.shape, (250, 250, 3))
    assert_equals(color_space, Image.RGB)
