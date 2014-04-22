from nose.tools import assert_equals, assert_almost_equals, \
                       assert_tuple_equal, assert_list_equal, \
                       assert_almost_equal

import cv2

from math import pi

from simplecv.image import Image
from simplecv.features.detection import Line, Barcode, Chessboard, Circle
from simplecv.tests.utils import perform_diff

BARCODE_IMAGE = "../data/sampleimages/barcode.png"
CHESSBOARD_IMAGE = "../data/sampleimages/CalibImage3.png"

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
    l3 = Line(img, ((300, 300), (400, 500)))
    assert l1.is_parallel(l2)
    assert_equals(l1.is_parallel(l3), False)

def test_line_perp():
    img = None
    l1 = Line(img, ((100, 200), (100, 400)))
    l2 = Line(img, ((200, 300), (400, 300)))
    assert l1.is_perpendicular(l2)
    assert l2.is_perpendicular(l1)

    l3 = Line(None, ((10, 10), (80, 80)))
    l4 = Line(None, ((80, 10), (10, 80)))
    l5 = Line(None, ((10, 10), (20, 30)))
    assert l3.is_perpendicular(l4)
    
    assert_equals(l1.is_perpendicular(l3), False)
    assert_equals(l3.is_perpendicular(l1), False)
    assert_equals(l3.is_perpendicular(l5), False)

def test_line_img_intersection():
    img = Image((512, 512))
    for x in range(200, 400):
        img[200, x] = (255.0, 255.0, 255.0)
    for y in range(200, 400):
        img[y, 200] = (255.0, 255.0, 255.0)

    l = Line(img, ((300, 100), (300, 500)))
    l1 = Line(img, ((100, 300), (500,300)))
    l2 = Line(img, ((200, 300), (300,200)))
    
    assert_equals([(300, 200)], l.img_intersections(img))
    assert_equals([(200, 300), (300, 200)], l2.img_intersections(img))
    assert_equals([(200, 300)], l1.img_intersections(img))

def test_line_get_angle():
    l1 = Line(None, ((50, 10), (50, 80)))
    l2 = Line(None, ((80, 50), (10, 50)))
    assert_equals(l1.get_angle(), 90.0)
    assert_equals(l2.get_angle(), 0.0)

def test_line_crop_to_image_edges():
    img = Image((101, 101))

    l = Line(img, ((-10, -5), (40, 40)))
    l_cr = l.crop_to_image_edges()
    assert_tuple_equal(((0, 4), (40, 40)), l_cr.end_points)

    l = Line(img, ((-5, -5), (140, 140)))
    l_cr = l.crop_to_image_edges()
    assert_tuple_equal(((0, 0), (100, 100)), l_cr.end_points)

    l = Line(img, ((40, 40), (140, 140)))
    l_cr = l.crop_to_image_edges()
    assert_tuple_equal(((40, 40), (100, 100)), l_cr.end_points)

    l = Line(img, ((105, -5), (50, 50)))
    l_cr = l.crop_to_image_edges()
    assert_tuple_equal(((50, 50), (100, 0)), l_cr.end_points)

    l = Line(img, ((105, -5), (-5, 105)))
    l_cr = l.crop_to_image_edges()
    assert_tuple_equal(((100, 0), (0, 100)), l_cr.end_points)

    l = Line(img, ((50, -50), (50, 90)))
    l_cr = l.crop_to_image_edges()
    assert_tuple_equal(((50, 0), (50, 90)), l_cr.end_points)

    l = Line(img, ((50, -50), (50, 150)))
    l_cr = l.crop_to_image_edges()
    assert_tuple_equal(((50, 0), (50, 100)), l_cr.end_points)

    l = Line(img, ((50, 10), (50, 150)))
    l_cr = l.crop_to_image_edges()
    assert_tuple_equal(((50, 10), (50, 100)), l_cr.end_points)

    l = Line(img, ((50, -50), (150, -50)))
    l_cr = l.crop_to_image_edges()
    assert_equals(None, l_cr)

    l = Line(img, ((50, 50), (150, 50)))
    l_cr = l.crop_to_image_edges()
    assert_tuple_equal(((50, 50), (100, 50)), l_cr.end_points)

    l = Line(img, ((20, 50), (80, 50)))
    l_cr = l.crop_to_image_edges()
    assert_tuple_equal(((20, 50), (80, 50)), l_cr.end_points)

    l = Line(img, ((-50, 50), (50, 50)))
    l_cr = l.crop_to_image_edges()
    assert_tuple_equal(((0, 50), (50, 50)), l_cr.end_points)

def test_line_extend_to_image_edges():
    img = Image((101, 101))

    l = Line(img, ((10, 10), (30, 30)))
    l_ext = l.extend_to_image_edges()
    assert_list_equal([(0, 0), (100, 100)], l_ext.end_points)

    l = Line(img, ((10, 10), (30, 10)))
    l_ext = l.extend_to_image_edges()
    assert_list_equal([(0, 10), (100, 10)], l_ext.end_points)

    l = Line(img, ((10, 10), (10, 30)))
    l_ext = l.extend_to_image_edges()
    assert_list_equal([(10, 0), (10, 100)], l_ext.end_points)

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

def test_chessboard():
    img = Image(CHESSBOARD_IMAGE)
    chessboard_patent = (8, 5)
    res, cor = cv2.findChessboardCorners(img.get_ndarray(), chessboard_patent)

    chessboard = Chessboard(img, chessboard_patent, cor)
    chessboard.get_area()
    chessboard.draw()

    perform_diff([img], "test_chessboard", 0.0)

def test_circle_draw():
    image = Image((200, 200))
    circ = Circle(image, 100, 100, 50)
    circ.draw(color=(255, 255, 255), width=3)
    perform_diff([image], "test_circle_draw", 0.0)

def test_circle_distance_from():
    image = Image((200, 200))
    circ = Circle(image, 100, 100, 50)

    assert_equals(circ.distance_from(), 0)
    assert_equals(circ.distance_from((0, 0)), ((100*100)+(100*100))**0.5)

def test_circle_mean_color():
    image = Image((201, 201))
    np_array = image.get_ndarray()
    np_array[:, :100] = (255, 0, 0)
    np_array[:, 101:] = (0, 0, 255)

    circ = Circle(image, 100, 100, 100)
    assert_almost_equal(circ.mean_color()[0], 126.68, 2)
    assert_equals(circ.mean_color()[1], 0.0)
    assert_almost_equal(circ.mean_color()[2], 126.68, 2)
    
def test_circle_properties():
    circ = Circle(None, 100, 100, 100)
    assert_almost_equal(circ.get_area(), 100*100*pi, 3)
    assert_almost_equal(circ.get_perimeter(), 100*2*pi, 3)
    assert_equals(circ.get_width(), 100*2)
    assert_equals(circ.get_height(), 100*2)
    assert_equals(circ.radius(), 100)
    assert_equals(circ.diameter(), 200)

def test_circle_crop():
    image = Image("simplecv")
    circ = Circle(image, 100, 100, 50)
    no_mask_image = circ.crop(no_mask=False)

    image = Image("simplecv")
    circ = Circle(image, 100, 100, 50)
    mask_image = circ.crop(no_mask=True)

    perform_diff([no_mask_image, mask_image], "test_circle_crop", 0.0)
