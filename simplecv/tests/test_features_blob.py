from nose.tools import assert_equals, assert_almost_equals, assert_is_instance

from simplecv.color import Color
from simplecv.color_model import ColorModel
from simplecv.features.blob import Blob
from simplecv.features.contour import Contour
from simplecv.image import Image
from simplecv.tests.utils import perform_diff, perform_diff_blobs, skipped


def test_hull():
    img = Image(source="lenna")
    blobs = img.find(Blob)
    blob = blobs[-1]
    chull = blob.convex_hull


def test_blob_draw_rect():
    img = Image(source="lenna")
    blobs = img.find(Blob)
    blob = blobs[-1]
    blob.draw_rect(color=Color.BLUE, width=-1, alpha=128)

    img1 = Image(source="simplecv")
    blobs = img1.find(Blob)
    blob = blobs[-1]
    blob.draw_rect(color=Color.RED, width=2, alpha=255)

    imgs = [img, img1]
    name_stem = "test_blob_draw_rect"
    perform_diff(imgs, name_stem, 0.0)


def test_blob_rectify_major_axis():
    img = Image(source="lenna")
    blobs = img.find(Blob)
    blobs_1 = blobs[-1]
    blobs_1.rectify_major_axis()
    blobs_2 = blobs[-2]
    blobs_2.rectify_major_axis(1)

    blobs1 = img.find(Blob)
    blobs1_1 = blobs1[-1]
    blobs1_1.rotate(blobs1_1.angle)
    blobs1_2 = blobs[-2]
    blobs1_2.rectify_major_axis(1)

    perform_diff_blobs(blobs_1, blobs1_1)
    perform_diff_blobs(blobs_2, blobs1_2)


def test_blob_draw_appx():
    nblob = Blob()

    img = Image(source="simplecv")
    blobs = img.find(Blob)
    blob = blobs[-1]
    blob.contour_appx.draw(color=Color.GREEN, width=-1, alpha=128)

    img1 = Image(source="lenna")
    blobs1 = img1.find(Blob)
    blob1 = blobs1[-2]
    blob1.contour_appx.draw(color=Color.RED, width=3, alpha=255)

    result = [img, img1]
    name_stem = "test_blob_draw_appx"

    perform_diff(result, name_stem, 0.0)


def test_blob_is_square():
    img = Image((400, 400))
    nparray = img
    nparray[100:300, 100:300] = (255, 255, 255)

    blobs = img.find(Blob)
    blob = blobs[0]
    assert blob.is_square()

    img1 = Image((400, 400))
    nparray1 = img1
    nparray1[50:350, 100:300] = (255, 255, 255)
    blobs1 = img1.find(Blob)
    blob1 = blobs1[0]
    assert not blob1.is_square()


def test_blob_centroid():
    img = Image((400, 400))
    nparray = img
    nparray[100:300, 100:300] = (255, 255, 255)

    blobs = img.find(Blob)
    blob = blobs[0]

    assert_equals(blob.centroid, (199.5, 199.5))


def test_blob_radius():
    img = Image((400, 400))
    nparray = img
    nparray[100:300, 100:300] = (255, 255, 255)

    blobs = img.find(Blob)
    blob = blobs[0]

    assert_equals(int(blob.radius), 140)


def test_blob_hull_radius():
    img = Image((400, 400))
    nparray = img
    nparray[100:300, 100:300] = (255, 255, 255)

    blobs = img.find(Blob)
    blob = blobs[0]

    assert_equals(int(blob.hull_radius), 140)


def test_blob_match():
    img = Image((400, 400))
    nparray = img
    nparray[50:150, 50:150] = (255, 255, 255)
    nparray[100:150, 150:250] = (255, 255, 255)
    nparray[200:300, 50:150] = (255, 255, 255)
    nparray[250:300, 150:250] = (255, 255, 255)

    blobs = img.find(Blob)
    blob = blobs[0]
    blob1 = blobs[1]

    if blob.match(blob1) > 6e-10:
        assert False


def test_blob_repr():
    img = Image((400, 400))
    nparray = img
    nparray[50:150, 50:150] = (255, 255, 255)
    nparray[100:150, 150:250] = (255, 255, 255)

    blobs = img.find(Blob)
    blob = blobs[0]

    bstr = "simplecv.features.blob.Blob object at (150, 100) with " \
           "area 14701"

    assert_equals(blob.__repr__(), bstr)


def test_blob_get_sc_descriptors():
    img = Image((400, 400))
    nparray = img
    nparray[50:150, 50:150] = (255, 255, 255)
    nparray[250:350, 150:250] = (255, 255, 255)

    blobs = img.find(Blob)
    blob0 = blobs[0]
    blob1 = blobs[1]

    scd0, cc0 = blob0.get_sc_descriptors()
    scd1, cc1 = blob1.get_sc_descriptors()

    assert_equals(len(scd0), len(scd1))
    assert_equals(len(cc0), len(cc1))

    if cc0[0][0] > cc1[0][0]:
        for index in range(len(cc0)):
            assert_almost_equals(cc0[index][0] - 100, cc1[index][0], 5)
            assert_almost_equals(cc0[index][1] - 200, cc1[index][1], 5)
            #cc2.append((ccp[0]-100, ccp[1]-200))
    else:
        for index in range(len(cc0)):
            assert_almost_equals(cc0[index][0] + 100, cc1[index][0], 5)
            assert_almost_equals(cc0[index][1] + 200, cc1[index][1], 5)

    for index in range(len(scd0)):
        assert_equals(scd0[index].data, scd1[index].data)

# broken utility
@skipped
def test_blob_show_correspondence():
    img = Image((400,400))
    nparray = img
    nparray[50:150, 50:150] = (255, 255, 255)
    nparray[250:350, 150:250] = (255, 255, 255)

    blobs = img.find(Blob)
    blob0 = blobs[0]
    blob1 = blobs[1]


def test_get_shape_context():
    img = Image((400, 400))
    nparray = img
    nparray[50:150, 50:150] = (255, 255, 255)
    nparray[250:350, 150:250] = (255, 255, 255)

    blobs = img.find(Blob)
    blob0 = blobs[0]
    blob0.shape_context

def test_blob_extract():
    img = Image((400, 400))
    nparray = img
    nparray[50:100, 50:100] = (255, 255, 255)
    nparray[150:225, 150:225] = (255, 255, 255)
    nparray[250:350, 250:350] = (255, 255, 255)

    blobs = Blob.extract(img, maxsize=-1)

    assert_equals(len(blobs), 3)

    blobs = Blob.extract(img, maxsize=9000, minsize=3000)
    assert_equals(len(blobs), 1)

    img = Image((1, 1))
    bin_img = Image((1, 1))
    blobs = Blob.extract_from_binary(bin_img, img, maxsize=-1)
    assert_equals(len(blobs), 0)


def test_blob_extract_using_model():
    cm = ColorModel()
    cm.add(Color.RED)
    cm.add(Color.GREEN)

    img = Image((400, 400))
    nparray = img
    nparray[:, :] = (0, 0, 255)
    nparray[50:100, 50:100] = (255, 0, 0)
    nparray[150:225, 150:225] = (0, 255, 0)
    nparray[250:350, 250:350] = (255, 0, 0)

    blobs = Blob.extract_using_model(img, cm, maxsize=-1)
    assert_equals(len(blobs), 3)

    blobs = Blob.extract(img, maxsize=9000, minsize=3000)
    assert_equals(len(blobs), 1)


def test_blob_extract_blob_data():
    img = Image("../data/sampleimages/blockhead.png")
    blobs = Blob.extract(img)
    assert blobs
    assert_equals(7, len(blobs))
    for b in blobs:
        assert b.area > 0
        assert b.perimeter > 0
        assert sum(b.avg_color) > 0
        assert sum(b.bounding_box) > 0
        assert b.moments['m00'] != 0
        assert b.moments['m01'] != 0
        assert b.moments['m10'] != 0
        assert b.moments['m11'] != 0
        assert b.moments['m20'] != 0
        assert b.moments['m02'] != 0
        assert b.moments['m21'] != 0
        assert b.moments['m12'] != 0
        assert sum(b.hu) > 0


def test_blob_extract_blob_methods():
    img = Image("../data/sampleimages/blockhead.png")
    blobs = Blob.extract(img)
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
        assert_is_instance(b.contour, Contour)
        assert_is_instance(b.aspect_ratio, float)
        assert_is_instance(b.angle, float)
        assert_is_instance(b.above(first), bool)
        assert_is_instance(b.below(first), bool)
        assert_is_instance(b.left(first), bool)
        assert_is_instance(b.right(first), bool)
        assert_is_instance(b.contains(first), bool)
        assert_is_instance(b.overlaps(first), bool)

        assert_is_instance(b.contour.to_image(), Image)
        assert_is_instance(b.contour.to_mask(), Image)
        assert_is_instance(b.convex_hull.to_image(), Image)
        assert_is_instance(b.convex_hull.to_mask(), Image)
        b.rectify_major_axis()
        assert_is_instance(b.contour.to_image(), Image)
        assert_is_instance(b.contour.to_mask(), Image)
        assert_is_instance(b.convex_hull.to_image(), Image)
        assert_is_instance(b.convex_hull.to_mask(), Image)