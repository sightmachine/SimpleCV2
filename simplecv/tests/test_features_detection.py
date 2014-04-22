from nose.tools import assert_equals, assert_almost_equals

from simplecv.image import Image
from simplecv.features.detection import Line
from simplecv.features.detection import Barcode
from simplecv.tests.utils import perform_diff

BARCODE_IMAGE = "../data/sampleimages/barcode.png"

def test_line_draw():
    img = Image((100, 100))
    line = Line(img, ((20, 20), (20,80)))
    line.draw(color=(255, 255, 255), width=1)

    img1 = Image((100, 100))
    np_array = img1.get_ndarray()
    np_array[20:80, 20:20] = (255, 255, 255)

    assert_equals(img.get_ndarray().data, img1.get_ndarray().data)

def test_line_length():
    line = Line(None, ((20, 20), (20,80)))
    assert_equals(line.length(), 60)

def test_line_crop():
    img = Image("../data/sampleimages/EdgeTest2.png")
    l = img.find_lines().sort_area()
    l = l[-5:-1]
    results = []
    for ls in l:
        results.append(ls.crop())
    name_stem = "test_line_crop"
    perform_diff(results, name_stem, tolerance=3.0)

def test_line_mean_color():
    img = Image((100, 100))
    np_array = img.get_ndarray()
    np_array[:100, :30] = (255, 0, 0)
    np_array[:100, 30:60] = (0, 255, 0)
    np_array[:100, 60:90] = (0, 0, 255)

    l1 = Line(img, ((10, 10), (80, 10)))
    expected_mean_color = (255*20/70.0, 255*30/70.0, 255*20/70.0)
    assert_equals(l1.mean_color(), expected_mean_color)

    l2 = Line(img, ((10, 10), (10, 80)))
    expected_mean_color = (255.0, 0.0, 0.0)
    assert_equals(l2.mean_color(), expected_mean_color)

    l3 = Line(img, ((25, 25), (35, 35)))
    expected_mean_color = (255.0/2.0, 255.0/2.0, 0.0)
    l3.draw(color=(255, 255, 255), width=2)
    img.show()
    assert_equals(l3.mean_color(), expected_mean_color)

def test_line_find_intersection():
    l1 = Line(None, ((50, 10), (50, 80)))
    l2 = Line(None, ((10, 50), (80, 50)))
    assert_equals(l1.find_intersection(l2), (50, 50))
    assert_equals(l2.find_intersection(l1), (50, 50))

    l3 = Line(None, ((10, 10), (80, 80)))
    l4 = Line(None, ((80, 10), (10, 80)))
    assert_equals(l3.find_intersection(l4), (45.0, 45.0))

def test_line_parallel():
    img = None
    l1 = Line(img, ((100, 200), (300, 400)))
    l2 = Line(img, ((200, 300), (400, 500)))
    assert l1.is_parallel(l2)


def test_line_perp():
    img = None
    l1 = Line(img, ((100, 200), (100, 400)))
    l2 = Line(img, ((200, 300), (400, 300)))
    assert l1.is_perpendicular(l2)

    l1 = Line(None, ((50, 10), (50, 80)))
    l2 = Line(None, ((10, 50), (80, 50)))
    assert l1.find_intersection(l2)
    assert l2.find_intersection(l1)

def test_line_img_intersection():
    img = Image((512, 512))
    for x in range(200, 400):
        img[200, x] = (255.0, 255.0, 255.0)
    l = Line(img, ((300, 100), (300, 500)))
    assert_equals([(300, 200)], l.img_intersections(img))

def test_line_get_angle():
    l1 = Line(None, ((50, 10), (50, 80)))
    l2 = Line(None, ((10, 50), (80, 50)))
    assert_equals(l1.get_angle(), 90.0)
    assert_equals(l2.get_angle(), 0.0)

def test_line_crop_to_image_edges():
    img = Image((512, 512))
    l = Line(img, ((-10, -5), (400, 400)))
    l_cr = l.crop_to_image_edges()
    assert_tuple_equal(((0, 5), (400, 400)), l_cr.end_points)

def test_line_extend_to_image_edges():
    img = Image((512, 512))
    l = Line(img, ((10, 10), (30, 30)))
    l_ext = l.extend_to_image_edges()
    assert_list_equal([(0, 0), (511, 511)], l_ext.end_points)

def test_line_get_vector():
    l1 = Line(None, ((50, 10), (50, 80)))
    l2 = Line(None, ((10, 50), (80, 50)))
    assert_equals(l1.get_vector(), [0.0, 70.0])

def test_line_dot():
    l1 = Line(None, ((50, 10), (50, 80)))
    l2 = Line(None, ((10, 50), (80, 50)))
    assert_equals(l1.dot(l2), 0.0)

def test_line_cross():
    l1 = Line(None, ((50, 10), (50, 80)))
    l2 = Line(None, ((10, 50), (80, 50)))
    assert_equals(l1.cross(l2), -4900.0)

def test_line_get_y_intercept():
    l1 = Line(None, ((50, 10), (50, 80)))
    l2 = Line(None, ((10, 50), (80, 50)))    

    assert_equals(l2.get_y_intercept(), 50)
    assert_equals(l1.get_y_intercept(), float("-inf"))

def test_barcode():
    img = Image(BARCODE_IMAGE)
    barcode = img.find_barcode()[0]
    repr_str = "%s.%s at (%d,%d), read data: %s" % (
            barcode.__class__.__module__, barcode.__class__.__name__, barcode.x, barcode.y,
        barcode.data)
    assert_equals(barcode.__repr__(), repr_str)
    barcode.draw(color=(255, 0, 0), width=5)
    assert_equals(barcode.length(),[262.0])
    assert_almost_equals(barcode.get_area(), 68644.0)
    perform_diff([img], "test_barcode", 0.0)