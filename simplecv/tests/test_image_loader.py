import numpy as np
from nose.tools import assert_equals, assert_true, assert_false
import mock

from simplecv.core.image.loader import ImageLoader, Cv2ImageLoader, \
                                       SampleImageLoader, HttpImageLoader, \
                                       RawPngImageLoader, \
                                       ListTupleImageLoader, \
                                       WebpImageLoader, \
                                       PilImageLoader
from simplecv.image import Image
from simplecv.tests.utils import sampleimage_path

simplecv_image = sampleimage_path('simplecv.png')


def test_cv2_image_loader():
    assert not Cv2ImageLoader.can_load()
    assert not Cv2ImageLoader.can_load(source="unknownimage.jpg")
    assert not Cv2ImageLoader.can_load(source="")
    assert Cv2ImageLoader.can_load(source="../data/sampleimages/lenna.png")

    data = Cv2ImageLoader.load(source="../data/sampleimages/lenna.png")
    array, colorspace, source = data

    assert_equals(array.shape, (512, 512, 3))
    assert_equals(colorspace, Image.BGR)


def test_sample_image_loader():
    assert not SampleImageLoader.can_load()
    assert SampleImageLoader.can_load(source="lenna", sample=True)
    sample_images = SampleImageLoader.SUPPORTED_SAMPLE_IMAGES
    for img in sample_images:
        SampleImageLoader.load(source=img, sample=True)


@mock.patch('simplecv.core.image.loader.urllib2')
def test_http_image_loader(urllib2_mock):
    myurl = "http://server.org/my.png"
    with open(simplecv_image) as f:
        urllib2_mock.urlopen.return_value = f

        assert_true(HttpImageLoader.can_load(source=myurl))
        data = HttpImageLoader.load(source=myurl)

        array, colorspace, filename = data

        assert_equals(array.shape, (250, 250, 3))
        assert_equals(colorspace, Image.BGR)

    assert_false(HttpImageLoader.can_load(
        source="../data/sampleimages/simplecv.png"))


def test_raw_png_image_loader():
    png_data = ('data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAQAAAAECAIAA'
                'AAmkwkpAAAACXBIWXMAAAsTAAALEwEAmpwYAAAAB3RJTUUH3gwJDgkPF/v+'
                'pgAAABl0RVh0Q29tbWVudABDcmVhdGVkIHdpdGggR0lNUFeBDhcAAAAiSUR'
                'BVAjXTcqxCQAgEACxvLj/ymchgqkzFQOSJ8tnq7viABf9CQFv5HakAAAAAE'
                'lFTkSuQmCC')
    assert RawPngImageLoader.can_load(source=png_data)
    assert not RawPngImageLoader.can_load(source="unknown_source")

    data = RawPngImageLoader.load(source=png_data)
    array, colorspace, filename = data
    assert_equals(array.shape, (4, 4, 3))
    assert_equals(colorspace, Image.BGR)
    expected = [[[255, 255, 255], [  0, 255, 255], [  0, 255, 255], [255, 255, 255]],
                [[  0,   0, 255], [  0,   0,   0], [  0,   0,   0], [  0, 255,   0]],
                [[  0,   0, 255], [  0,   0,   0], [  0,   0,   0], [  0, 255,   0]],
                [[255, 255, 255], [255,   0,   0], [255,   0,   0], [255, 255, 255]]]
    assert_equals(expected, array.tolist())


def test_list_tuple_image_loader():
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


def test_webp_image_loader():
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


def test_pil_image_loader():
    img = Image("simplecv")
    pil_img = img.get_pil()

    assert PilImageLoader.can_load(source=pil_img)
    assert not PilImageLoader.can_load(source=img)

    data = PilImageLoader.load(source=pil_img)
    array, color_space, filename = data

    assert_equals(array.shape, (250, 250, 3))
    assert_equals(color_space, Image.RGB)
