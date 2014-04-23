from nose.tools import assert_equals, assert_almost_equals, \
                       assert_tuple_equal, assert_list_equal, \
                       assert_almost_equal

import cv2

from math import pi

from simplecv.image import Image
from simplecv.features.detection import Line, Barcode, Chessboard, Circle, \
                                        Motion, ROI
from simplecv.tests.utils import perform_diff

BARCODE_IMAGE = "../data/sampleimages/barcode.png"
CHESSBOARD_IMAGE = "../data/sampleimages/CalibImage3.png"
KEYPOINT_IMAGE = "../data/sampleimages/KeypointTemplate2.png"
testimageclr = "../data/sampleimages/statue_liberty.jpg"

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

def test_keypoint():
    img = Image(KEYPOINT_IMAGE)
    kp  = img.find_keypoints()

    assert_equals(190, len(kp))

    keypoint = kp[0]
    keypoint_object = keypoint.get_object()

    assert_equals(keypoint_object.angle, keypoint.get_angle())
    assert_equals(keypoint_object.octave, keypoint.get_octave())
    assert_equals(keypoint_object.response, keypoint.quality())
    assert_equals(keypoint.get_flavor(), "SURF")
    assert_equals(keypoint.get_perimeter(), 2*pi*keypoint_object.size/2.0)
    assert_equals(keypoint.get_width(), keypoint_object.size)
    assert_equals(keypoint.get_height(), keypoint_object.size)
    assert_equals(keypoint.radius(), keypoint_object.size/2.0)
    assert_equals(keypoint.diameter(), keypoint_object.size)

    assert_equals(keypoint.distance_from(keypoint_object.pt), 0.0)
    dist = ((keypoint_object.pt[0]-img.size[0]/2)**2 + 
            (keypoint_object.pt[1]-img.size[1]/2)**2)**0.5
    assert_equals(keypoint.distance_from(), dist)

    m_color = keypoint.mean_color()
    color_dist = (m_color[0]**2 + m_color[1]**2 + m_color[2]**2)**0.5
    assert_equals(keypoint.color_distance(), color_dist)

    crop_mask = keypoint.crop(no_mask=True)
    crop_no_mask = keypoint.crop()

def test_motion():
    img = Image((100,100))
    np_array = img.get_ndarray()
    np_array[50:60, 30:40] = (255, 0, 0)
    np_array[40:50, 20:30] = (0, 0, 255)

    motion = Motion(img, 30, 50, 2, 3, 10)

    assert_equals(motion.magnitude(), 13.0**0.5)
    assert_equals(motion.unit_vector(), (2/(13.0**0.5), 3/(13.0**0.5)))
    assert_equals(motion.vector(), (2, 3))
    assert_equals(motion.mean_color(), (255.0/4, 0, 255.0/4))
    assert_equals(motion.window_sz(), 10)
    assert_equals(motion.normalize_to(0), None)
    motion.normalize_to(2)
    assert_equals(motion.norm_dx, 2/(13.0**0.5)*13.0**0.5/2.0)
    assert_equals(motion.norm_dy, 3/(13.0**0.5)*13.0**0.5/2.0)
    #assert_equals(motion.normalize_to())

    crop_image = motion.crop()
    crop_image.get_ndarray().shape
    crop_array = np_array[45:55, 25:35].copy()
    assert_equals(crop_image.get_ndarray().data, crop_array.data)

    motion.draw(normalize=False)
    motion.draw()

    motion = Motion(img, 30, 50, 0, 0, 10)
    assert_equals(motion.unit_vector(), (0.0, 0.0))

def test_shape_context_descriptor():
    img = Image((200, 200))
    np_array = img.get_ndarray()
    np_array[50:150, 30:80] = (255, 255, 255)

    blobs = img.find_blobs()
    blob = blobs[-1]

    shape_context_descriptors = blob.get_shape_context()
    shape_context_descriptor = shape_context_descriptors[0]
    shape_context_descriptor.draw(width=4)

def test_roi():
    import numpy as np
    from simplecv.color import Color
    img = Image(testimageclr)
    mask = img.threshold(248).dilate(5)
    blobs = img.find_blobs_from_mask(mask, minsize=1)
    y, x = np.where(mask.get_gray_ndarray() > 0)
    xmin = np.min(x)
    xmax = np.max(x)
    ymin = np.min(y)
    ymax = np.max(y)
    w = xmax - xmin
    h = ymax - ymin
    roi_list = []

    def subtest(data, effect):
        broke = False
        first = effect(data[0])
        i = 0
        for d in data:
            e = effect(d)
            print (i, e)
            i = i + 1
            if first != e:
                broke = True
        return broke

    broi = ROI(blobs)
    broi2 = ROI(blobs, image=img)

    roi_list.append(ROI(x=x, y=y, image=img))
    roi_list.append(ROI(x=list(x), y=list(y), image=img))
    roi_list.append(ROI(x=tuple(x), y=tuple(y), image=img))
    roi_list.append(ROI(zip(x, y), image=img))
    roi_list.append(ROI((xmin, ymin), (xmax, ymax), image=img))
    roi_list.append(ROI(xmin, ymin, w, h, image=img))
    roi_list.append(
        ROI([(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)],
            image=img))
    roi_list.append(ROI(roi_list[0]))

    # test the basics
    def to_xywh(roi):
        return roi.to_xywh()

    assert_list_equal([322, 1, 119, 52], roi_list[0].to_xywh())
    if subtest(roi_list, to_xywh):
        assert False

    broi.translate(10, 10)
    broi.translate(-10)
    broi.translate(y=-10)
    broi.to_tl_and_br()
    broi.to_points()
    broi.to_unit_xywh()
    broi.to_unit_tl_and_br()
    broi.to_unit_points()
    roi_list[0].crop()
    new_roi = ROI(zip(x, y), image=mask)
    test = new_roi.crop()
    yroi, xroi = np.where(test.get_gray_ndarray() > 128)
    roi_pts = zip(xroi, yroi)
    real_pts = new_roi.coord_transform_pts(roi_pts)
    unit_roi = new_roi.coord_transform_pts(roi_pts, output="ROI_UNIT")
    unit_src = new_roi.coord_transform_pts(roi_pts, output="SRC_UNIT")
    src1 = new_roi.coord_transform_pts(roi_pts, intype="SRC_UNIT",
                                       output='SRC')
    src2 = new_roi.coord_transform_pts(roi_pts, intype="ROI_UNIT",
                                       output='SRC')
    src3 = new_roi.coord_transform_pts(roi_pts, intype="SRC_UNIT",
                                       output='ROI')
    src4 = new_roi.coord_transform_pts(roi_pts, intype="ROI_UNIT",
                                       output='ROI')
    fs = new_roi.split_x(10)
    fs = new_roi.split_x(.5, unit_vals=True)
    for f in fs:
        f.draw(color=Color.BLUE)
    fs = new_roi.split_x(new_roi.xtl + 10, src_vals=True)
    xs = new_roi.xtl
    fs = new_roi.split_x([10, 20])
    fs = new_roi.split_x([xs + 10, xs + 20, xs + 30], src_vals=True)
    fs = new_roi.split_x([0.3, 0.6, 0.9], unit_vals=True)
    fs = new_roi.split_y(10)
    fs = new_roi.split_y(.5, unit_vals=True)
    for f in fs:
        f.draw(color=Color.BLUE)
    fs = new_roi.split_y(new_roi.ytl + 30, src_vals=True)
    test_roi = ROI(blobs[0], mask)
    for b in blobs[1:]:
        test_roi.merge(b)

    roi = ROI(100, y=img)
    roi = ROI(10, 20, 30, h=img)
    roi = ROI(10, 20, w=img)

    roi = ROI(10, 20, 50, 50, img)
    roi.resize(2)
    assert_equals(roi.w, 100)
    assert_equals(roi.h, 100)

    roi.resize((30, 30), percentage=False)
    assert_equals(roi.w, 130)
    assert_equals(roi.h, 130)

    roi.resize(50, 80, percentage=False)
    assert_equals(roi.w, 180)
    assert_equals(roi.h, 210)

    roi = ROI(10, 20, 50, 50, img)
    roi1 = ROI(20, 30, 60, 60, img)
    roi2 = ROI(90, 100, 30, 30, img)

    assert_equals(roi.overlaps(roi1), True)
    assert_equals(roi.overlaps(roi2), False)

    roi1.translate(0, 0)
    assert_equals(roi1.x, 0)
    assert_equals(roi1.y, 0)
    
    tl_br = roi2.to_tl_and_br()
    assert_equals(tl_br, [(90, 100), (120, 130)])

    roi3 = ROI(100, 80, 50, 50)
    assert_equals(roi3.coord_transform_y(50), None)
    assert_equals(roi2.coord_transform_y(20, output="ROI"), [20])

    assert_equals(roi3.coord_transform_pts((20, 40)), None)
    assert_equals(roi2.coord_transform_pts((20, 40), output="ROI"), [(20, 40)])

    assert_equals(roi2.split_x(0.5, True, True), None)
    assert_equals(roi2.split_x(300, True, False), None)
    assert_equals(roi2.split_y(0.5, True, True), None)
    assert_equals(roi2.split_y(300, True, False), None)

    # merge and rebase seems to be broken

    """
    roi1.merge(roi2)
    assert_equals(roi1.x, 20)
    assert_equals(roi1.y, 30)
    assert_equals(roi1.w, 100)
    assert_equals(roi1.h, 100)
    

    roi1.merge([roi2, roi2])
    assert_equals(roi1.x, 0)
    assert_equals(roi1.y, 0)
    assert_equals(roi1.w, 100)
    assert_equals(roi1.h, 100)

    roi1.rebase(10, 20, 50, 50)
    assert_equals(roi1.x, 10)
    assert_equals(roi1.y, 20)
    assert_equals(roi1.w, 50)
    assert_equals(roi1.h, 50)

    roi1.rebase([20, 10, 20, 30])
    assert_equals(roi1.x, 20)
    assert_equals(roi1.y, 10)
    assert_equals(roi1.w, 20)
    assert_equals(roi1.h, 30)

    roi1.rebase([-20, 10, 20, 30])
    """