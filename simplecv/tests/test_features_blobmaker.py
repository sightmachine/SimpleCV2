from nose.tools import assert_equals, assert_is_instance

from simplecv.color import Color
from simplecv.color_model import ColorModel
from simplecv.features.blobmaker import BlobMaker
from simplecv.image import Image


def test_blobmaker_extract():
    img = Image((400, 400))
    nparray = img.ndarray
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
    nparray = img.ndarray
    nparray[:, :] = (0, 0, 255)
    nparray[50:100, 50:100] = (255, 0, 0)
    nparray[150:225, 150:225] = (0, 255, 0)
    nparray[250:350, 250:350] = (255, 0, 0)

    bm = BlobMaker()
    blobs = bm.extract_using_model(img, cm, maxsize=-1)
    assert_equals(len(blobs), 3)

    blobs = bm.extract(img, maxsize=9000, minsize=3000)
    assert_equals(len(blobs), 1)


def test_blobmaker_blob_data():
    img = Image("../data/sampleimages/blockhead.png")
    blobber = BlobMaker()
    blobs = blobber.extract(img)
    assert blobs
    assert_equals(7, len(blobs))
    for b in blobs:
        assert b.area > 0
        assert b.perimeter > 0
        assert sum(b.avg_color) > 0
        assert sum(b.bounding_box) > 0
        assert b.m00 != 0
        assert b.m01 != 0
        assert b.m10 != 0
        assert b.m11 != 0
        assert b.m20 != 0
        assert b.m02 != 0
        assert b.m21 != 0
        assert b.m12 != 0
        assert sum(b.hu) > 0


def test_blobmaker_blob_methods():
    img = Image("../data/sampleimages/blockhead.png")
    blobber = BlobMaker()
    blobs = blobber.extract(img)
    first = blobs[0]
    for b in blobs:
        assert_is_instance(b.width, int)
        assert_is_instance(b.height, int)
        assert_is_instance(b.area, float)
        assert_is_instance(b.max_x, int)
        assert_is_instance(b.min_x, int)
        assert_is_instance(b.max_y, int)
        assert_is_instance(b.min_y, int)
        assert_is_instance(b.min_rect_width, float)
        assert_is_instance(b.min_rect_height, float)
        assert_is_instance(b.min_rect_x, float)
        assert_is_instance(b.min_rect_y, float)
        assert_is_instance(b.contour, list)
        assert_is_instance(b.aspect_ratio, float)
        assert_is_instance(b.angle, float)
        assert_is_instance(b.above(first), bool)
        assert_is_instance(b.below(first), bool)
        assert_is_instance(b.left(first), bool)
        assert_is_instance(b.right(first), bool)
        assert_is_instance(b.contains(first), bool)
        assert_is_instance(b.overlaps(first), bool)

        assert_is_instance(b.image, Image)
        assert_is_instance(b.mask, Image)
        assert_is_instance(b.hull_img, Image)
        assert_is_instance(b.hull_mask, Image)
        b.rectify_major_axis()
        assert_is_instance(b.image, Image)
        assert_is_instance(b.mask, Image)
        assert_is_instance(b.hull_img, Image)
        assert_is_instance(b.hull_mask, Image)
