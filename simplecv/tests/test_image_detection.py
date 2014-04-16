import pickle

import cv2
import numpy as np
from nose.tools import assert_equals, assert_is_instance, assert_greater

from simplecv.image import Image
from simplecv.tests.utils import perform_diff
from simplecv.features.blobmaker import BlobMaker
from simplecv.features.blob import Blob
from simplecv.features.features import FeatureSet, Feature
from simplecv.features.detection import Corner, Line
from simplecv.color import Color

#images
contour_hiearachy = "../data/sampleimages/contour_hiearachy.png"
testimage = "../data/sampleimages/9dots4lines.png"
testimage2 = "../data/sampleimages/aerospace.jpg"
testbarcode = "../data/sampleimages/barcode.png"


def test_detection_find_corners():
    img = Image(testimage2)
    corners = img.find_corners(25)
    corners.draw()
    assert_greater(len(corners), 0)
    result = [img]
    name_stem = "test_detection_find_corners"
    perform_diff(result, name_stem)


def test_image_histogram():
    img = Image(testimage2)
    for i in img.histogram(25):
        assert_is_instance(i, int)


def test_detection_lines():
    img = Image(testimage2)
    lines = img.find_lines()
    assert lines
    lines.draw()
    result = [img]
    name_stem = "test_detection_lines"
    perform_diff(result, name_stem)


def test_detection_lines_standard():
    img = Image(testimage2)
    lines = img.find_lines(use_standard=True)
    assert lines
    lines.draw()
    result = [img]
    name_stem = "test_detection_lines_standard"
    perform_diff(result, name_stem)


def test_detection_feature_measures():
    img = Image(testimage2)

    fs = FeatureSet()
    fs.append(Corner(img, 5, 5))
    fs.append(Line(img, ((2, 2), (3, 3))))
    bm = BlobMaker()
    result = bm.extract(img)
    assert result
    assert_equals(5, len(result))
    fs.extend(result)

    for f in fs:
        c = f.mean_color()
        assert_equals(3, len(c))
        assert all(map(lambda a: isinstance(a, (float, int)), c))

        pts = f.coordinates()
        assert_is_instance(pts, np.ndarray)
        assert_equals((2, ), pts.shape)

        assert_is_instance(f.get_area(), (float, int))
        assert_is_instance(f.length(), (float, int))
        assert_is_instance(f.color_distance(), (float, int))
        assert_is_instance(f.get_angle(), (float, int))
        # distance from center of image
        assert_is_instance(f.distance_from(), (float, int))

    fs1 = fs.sort_distance()
    assert_equals(7, len(fs1))
    assert all(map(lambda a: isinstance(a, Feature), fs1))
    fs2 = fs.sort_angle()
    assert_equals(7, len(fs2))
    assert all(map(lambda a: isinstance(a, Feature), fs2))
    fs3 = fs.sort_length()
    assert_equals(7, len(fs3))
    assert all(map(lambda a: isinstance(a, Feature), fs3))
    fs4 = fs.sort_color_distance()
    assert_equals(7, len(fs4))
    assert all(map(lambda a: isinstance(a, Feature), fs4))
    fs5 = fs.sort_area()
    assert_equals(7, len(fs5))
    assert all(map(lambda a: isinstance(a, Feature), fs5))


def test_detection_blobs_appx():
    img = Image("lenna")
    blobs = img.find_blobs()
    assert_equals(29, len(blobs))
    blobs[-1].draw(color=Color.RED)
    blobs[-1].draw_appx(color=Color.BLUE)
    result = [img]

    img2 = Image("lenna")
    blobs = img2.find_blobs(appx_level=11)
    assert_equals(29, len(blobs))
    blobs[-1].draw(color=Color.RED)
    blobs[-1].draw_appx(color=Color.BLUE)
    result.append(img2)

    name_stem = "test_detection_blobs_appx"
    perform_diff(result, name_stem)
    assert blobs is not None


def test_detection_blobs():
    result = []
    img = Image(testbarcode)
    blobs = img.find_blobs()
    blobs.draw(color=Color.RED)
    assert_equals(5, len(blobs))
    result.append(img)

    img = Image(contour_hiearachy)
    blobs = img.find_blobs()
    assert_equals(10, len(blobs))
    blobs.draw(color=Color.RED)
    result.append(img)

    #TODO - WE NEED BETTER COVERAGE HERE
    name_stem = "test_detection_blobs"
    perform_diff(result, name_stem)
    assert blobs is not None


def test_detection_blobs_lazy():
    img = Image("lenna")
    b = img.find_blobs()
    result = []

    s = pickle.dumps(b[-1])  # use two otherwise it w
    b2 = pickle.loads(s)

    result.append(b[-1].img)
    result.append(b[-1].mask)
    result.append(b[-1].hull_img)
    result.append(b[-1].hull_mask)

    result.append(b2.img)
    result.append(b2.mask)
    result.append(b2.hull_img)
    result.append(b2.hull_mask)

    #TODO - WE NEED BETTER COVERAGE HERE
    name_stem = "test_detection_blobs_lazy"
    perform_diff(result, name_stem)


def test_detection_blobs_adaptive():
    img = Image(testimage)
    blobs = img.find_blobs(threshblocksize=99)
    assert blobs
    blobs.draw(color=Color.RED)
    result = [img]
    name_stem = "test_detection_blobs_adaptive"
    perform_diff(result, name_stem)


def test_detection_blobs_smallimages():
    # Check if segfault occurs or not
    img = Image("../data/sampleimages/blobsegfaultimage.png")
    blobs = img.find_blobs()
    assert blobs is None
    # if no segfault, pass


def test_detection_blobs_convexity_defects():
    if not hasattr(cv2, 'convexityDefects'):
        return

    img = Image('lenna')
    blobs = img.find_blobs()
    b = blobs[-1]
    feat = b.get_convexity_defects()
    assert_greater(len(feat), 0)
    points = b.get_convexity_defects(return_points=True)
    assert_greater(len(points), 0)


def test_detection_barcode():
    try:
        import zbar
    except:
        return

    img1 = Image(testimage)
    img2 = Image(testbarcode)

    nocode = img1.find_barcode()
    assert nocode is None  # we should find no barcode in our test image
    code = img2.find_barcode()
    code.draw()
    assert code.points
    result = [img1, img2]
    name_stem = "test_detection_barcode"
    perform_diff(result, name_stem)


def test_detection_x():
    img = Image(testimage)
    tmp_x = img.find_lines().x()[0]
    assert tmp_x > 0
    assert img.size[0]


def test_detection_y():
    img = Image(testimage)
    tmp_y = img.find_lines().y()[0]
    assert_greater(tmp_y, 0)
    assert img.size[0]


def test_detection_area():
    img = Image(testimage2)
    bm = BlobMaker()
    result = bm.extract(img)
    for b in result:
        assert_greater(b.get_area(), 0)


def test_detection_angle():
    angles = Image(testimage).find_lines().get_angle()
    for angle in angles:
        assert_is_instance(angle, float)


def test_detection_length():
    img = Image(testimage)
    val = img.find_lines().length()
    assert_is_instance(val, np.ndarray)
    assert len(val)


def test_detection_sortangle():
    img = Image(testimage)
    val = img.find_lines().sort_angle()
    assert_greater(val[1].x, val[0].x)


def test_detection_sortarea():
    img = Image(testimage)
    bm = BlobMaker()
    result = bm.extract(img)
    val = result.sort_area()
    assert val
    assert_equals(3, len(val))
    assert all(map(lambda a: isinstance(a, Blob), val))


def test_detection_sort_length():
    img = Image(testimage)
    val = img.find_lines().sort_length()
    assert val
    assert_equals(17, len(val))
    assert all(map(lambda a: isinstance(a, Line), val))
