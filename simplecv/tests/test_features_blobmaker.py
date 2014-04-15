from nose.tools import assert_equals

from simplecv.color import Color
from simplecv.color_model import ColorModel
from simplecv.features.blobmaker import BlobMaker
from simplecv.image import Image


def test_blobmaker_extract():
    img = Image((400, 400))
    nparray = img.get_ndarray()
    nparray[50:100, 50:100] = (255, 255, 255)
    nparray[150:225, 150:225] = (255, 255, 255)
    nparray[250:350, 250:350] = (255, 255, 255)

    bm = BlobMaker()
    blobs = bm.extract(img, maxsize=-1)

    assert_equals(len(blobs), 3)

    blobs = bm.extract(img, maxsize=9000, minsize=3000)
    assert_equals(len(blobs), 1)

    img = Image((1, 1))
    bin_img = Image((1, 1))
    blobs = bm.extract_from_binary(bin_img, img, maxsize=-1)
    assert_equals(len(blobs), 0)


def test_blobmaker_extract_using_model():
    cm = ColorModel()
    cm.add(Color.RED)
    cm.add(Color.GREEN)

    img = Image((400, 400))
    nparray = img.get_ndarray()
    nparray[:, :] = (0, 0, 255)
    nparray[50:100, 50:100] = (255, 0, 0)
    nparray[150:225, 150:225] = (0, 255, 0)
    nparray[250:350, 250:350] = (255, 0, 0)

    bm = BlobMaker()
    blobs = bm.extract_using_model(img, cm, maxsize=-1)
    assert_equals(len(blobs), 3)

    blobs = bm.extract(img, maxsize=9000, minsize=3000)
    assert_equals(len(blobs), 1)
