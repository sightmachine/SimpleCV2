import pickle

from nose.tools import (nottest, assert_equals, assert_is_not_none,
                        assert_is, assert_true, assert_is_instance,
                        assert_false, assert_is_none)
import numpy as np
import cv2

from simplecv.color import ColorMap, Color
from simplecv.color_model import ColorModel
from simplecv.core.drawing.layer import DrawingLayer
from simplecv.dft import DFT
from simplecv.features.blob import Blob
from simplecv.features.contour import Contour
from simplecv.features.detection import Corner, Line, Circle, Chessboard,\
    Motion, KeyPoint, KeypointMatch, TemplateMatch, ShapeContextDescriptor, ROI
from simplecv.features.features import Feature, FeatureSet
from simplecv.font import Font
from simplecv.image import Image
from simplecv.image_set import ImageSet
from simplecv.linescan import LineScan
from simplecv.segmentation.color_segmentation import ColorSegmentation
from simplecv.segmentation.diff_segmentation import DiffSegmentation
from simplecv.segmentation.mog_segmentation import MOGSegmentation
from simplecv.segmentation.running_segmentation import RunningSegmentation


@nottest
def do_pickle(obj):
    s = pickle.dumps(obj)
    return pickle.loads(s)


def test_image():
    img = Image('simplecv')
    img.dl().line((0, 0), img.size_tuple)

    img2 = do_pickle(img)

    assert_equals(img._color_space, Image.BGR)
    assert_equals(img.data, img2.data)
    assert_equals(img.layers, [DrawingLayer([('set_default_alpha', (255,), {}),
                                             ('set_default_color', ((0, 0, 0),), {}),
                                             ('line', ((0, 0), (250, 250)), {})])])


def test_blob_contour():
    b = Blob()
    b.contour = Contour(np.array([[1, 1], [100, 300], [300, 100]]), blob=b)
    b.contour.holes = [np.array([[5, 5], [5, 10], [10, 5]]), np.array([[25, 25], [25, 45], [45, 25]])]

    # pickle only contour
    c2 = do_pickle(b.contour)
    assert_equals(b.contour.data, c2.data)
    assert_equals(len(c2.holes), 2)
    assert_equals(b.contour.holes[0].data, c2.holes[0].data)
    assert_equals(b.contour.holes[1].data, c2.holes[1].data)
    assert_is_not_none(c2.blob)
    assert_is(c2.blob.contour, c2)

    # pickle blob
    b2 = do_pickle(b)
    assert_equals(b2.contour.data, b.contour.data)
    assert_equals(len(b2.contour.holes), 2)
    assert_equals(b2.contour.holes[0].data, b.contour.holes[0].data)
    assert_equals(b2.contour.holes[1].data, b.contour.holes[1].data)
    assert_is_not_none(b2.contour.blob)
    assert_is(b2.contour.blob, b2)


def test_feature():
    i = Image('simplecv')
    f = Feature(i, 10, 20, [[10, 10], [10, 20], [20, 20]])

    f2 = do_pickle(f)

    assert_equals(f2.image.data, f.image.data)
    assert_equals(f2.x, 10)
    assert_equals(f2.y, 20)
    assert_equals(f2.points, [[10, 10], [10, 20], [20, 20]])


def test_corner():
    i = Image('simplecv')
    c = Corner(i, 10, 20)

    c2 = do_pickle(c)

    assert_equals(c2.image.data, c.image.data)
    assert_equals(c2.x, 10)
    assert_equals(c2.y, 20)
    assert_equals(c2.points, [(9, 19), (9, 21), (11, 21), (11, 19)])


def test_line():
    i = Image('simplecv')
    l = Line(i, [[10, 10], [10, 20]])

    l2 = do_pickle(l)

    assert_equals(l2.image.data, l.image.data)
    assert_equals(l2.x, 10)
    assert_equals(l2.y, 15)
    assert_equals(l2.points, [(10, 20), (10, 10), (10, 10), (10, 20)])


def test_circle():
    i = Image('simplecv')
    c = Circle(i, 10, 20, 5)

    c2 = do_pickle(c)

    assert_equals(c2.image.data, c.image.data)
    assert_equals(c2.x, 10)
    assert_equals(c2.y, 20)
    assert_equals(c2.r, 5)
    assert_equals(c2.points, [(5, 15), (15, 15), (15, 25), (5, 25)])


def test_chessboard():
    i = Image('simplecv')
    c = Chessboard(i, (2, 2), np.array([[[10, 10]], [[20, 20]], [[20, 10]], [[30, 20]]]))

    c2 = do_pickle(c)

    assert_equals(c2.image.data, c.image.data)
    assert_equals(c2.x, 20.0)
    assert_equals(c2.y, 15.0)
    assert_equals(c2.points, ([10, 10], [30, 20], [30, 20], [10, 10]))


def test_motion():
    i = Image('simplecv')
    m = Motion(i, 10, 20, 5, 7, 15)

    m2 = do_pickle(m)

    assert_equals(m2.image.data, m.image.data)
    assert_equals(m2.x, 10)
    assert_equals(m2.y, 20)
    assert_equals(m2.dx, 5)
    assert_equals(m2.dy, 7)
    assert_equals(m2.window, 15)
    assert_equals(m2.points, [(17, 27), (3, 27), (17, 27), (17, 13)])


def test_keypoint():
    i = Image('simplecv')
    cvkp = cv2.KeyPoint(3.0, 4.0, 1.0)
    kp = KeyPoint(i, cvkp, np.array([10, 20, 30, 50]), 'SURF')

    kp2 = do_pickle(kp)

    assert_equals(kp2.image.data, kp.image.data)
    assert_equals(kp2.descriptor.data, kp.descriptor.data)
    assert_equals(kp2.x, 3)
    assert_equals(kp2.y, 4)
    assert_equals(kp2.flavor, 'SURF')
    assert_equals(kp2.points, [(3.171010071662834, 4.469846310392954),
                               (3.32139380484327, 4.383022221559489),
                               (3.433012701892219, 4.25),
                               (3.492403876506104, 4.086824088833465),
                               (3.492403876506104, 3.913175911166535),
                               (3.433012701892219, 3.75),
                               (3.32139380484327, 3.616977778440511),
                               (3.1710100716628347, 3.530153689607046),
                               (3.0, 3.5),
                               (2.828989928337166, 3.5301536896070456),
                               (2.6786061951567306, 3.6169777784405106),
                               (2.566987298107781, 3.75),
                               (2.507596123493896, 3.913175911166535),
                               (2.507596123493896, 4.086824088833465),
                               (2.5669872981077804, 4.25),
                               (2.67860619515673, 4.3830222215594885),
                               (2.828989928337166, 4.469846310392954),
                               (3.0, 4.5)])


def test_keypoint_match():
    i = Image('simplecv')
    templ = i.invert()
    kpm = KeypointMatch(i, templ, ((1, 5), (2, 1), (2, 2), (1, 2)),
                        np.array([[10, 20, 30], [30, 40, 50], [40, 50, 60]]))

    kpm2 = do_pickle(kpm)

    assert_equals(kpm2.image.data, kpm.image.data)
    assert_equals(kpm2._template.data, kpm._template.data)
    assert_equals(kpm2._homography.data, kpm._homography.data)
    assert_equals(kpm2.x, 1)
    assert_equals(kpm2.y, 3)
    assert_equals(kpm2._min_rect, ((1, 5), (2, 1), (2, 2), (1, 2)))
    assert_equals(kpm2.points, [(1, 1), (1, 5), (2, 5), (2, 1)])


def test_template_match():
    i = Image('simplecv')
    templ = i.invert()
    tm = TemplateMatch(i, templ, (10, 40), 0.674)

    tm2 = do_pickle(tm)

    assert_equals(tm2.image.data, tm.image.data)
    assert_equals(tm2.template_image.data, tm.template_image.data)
    assert_equals(tm2.x, 10)
    assert_equals(tm2.y, 40)
    assert_equals(tm2.quality, 0.674)
    assert_equals(tm2.points, [(10, 40), (260, 40), (260, 290), (10, 290)])


def test_shape_context_descriptor():
    i = Image('simplecv')
    b = Blob()
    scd = ShapeContextDescriptor(i, (30, 23), np.array([1, 2, 3]), b)

    scd2 = do_pickle(scd)

    assert_equals(scd2.image.data, scd.image.data)
    assert_equals(scd2._descriptor.data, scd._descriptor.data)
    assert_is_not_none(scd2._source_blob)
    assert_equals(scd2.x, 30)
    assert_equals(scd2.y, 23)
    assert_equals(scd2.points, [(29, 22), (31, 22), (31, 24), (29, 24)])


def test_roi():
    i = Image('simplecv')
    roi = ROI(10, 55, 75, 25, i)

    roi2 = do_pickle(roi)

    assert_equals(roi2.image.data, roi.image.data)
    assert_equals(roi2.xtl, 10)
    assert_equals(roi2.ytl, 55)
    assert_equals(roi2.w, 75)
    assert_equals(roi2.h, 25)
    assert_equals(roi2.points, [(10, 55), (85, 55), (10, 80), (85, 80)])


def test_dft():
    orig_dft = DFT(size=(200, 300), channels=2, dia=100, type="gaussian",
              y_cutoff_high=200, x_cutoff_low=100)

    dft = do_pickle(orig_dft)

    assert_equals(dft.width, 200)
    assert_equals(dft.height, 300)
    assert_equals(dft.channels, 2)
    assert_equals(dft.get_dia(), 100)
    assert_equals(dft.get_type(), "gaussian")
    assert_equals(dft._numpy, None)
    assert_equals(dft._image, None)
    assert_equals(dft.get_order(), 0)
    assert_equals(dft._freqpass, "")
    assert_equals(dft._x_cutoff_low, 100)
    assert_equals(dft._y_cutoff_low, 0)
    assert_equals(dft._x_cutoff_high, 0)
    assert_equals(dft._y_cutoff_high, 200)


def test_image_set():
    img_set = ImageSet()
    scv_img = Image('simplecv')
    img_set.append(scv_img)
    lenna_img = Image('lenna')
    img_set.append(lenna_img)

    img_set2 = do_pickle(img_set)

    assert_equals(len(img_set2), 2)
    assert_equals(img_set2[0].data, scv_img.data)
    assert_equals(img_set2[1].data, lenna_img.data)


def test_feature_set():
    i = Image('simplecv')
    fs = FeatureSet()
    l = Line(i, [[10, 10], [10, 20]])
    fs.append(l)
    c = Circle(i, 10, 20, 5)
    fs.append(c)

    fs2 = do_pickle(fs)
    assert_equals(len(fs2), 2)

    assert_equals(fs2[0].image.data, l.image.data)
    assert_equals(fs2[0].x, 10)
    assert_equals(fs2[0].y, 15)
    assert_equals(fs2[0].points, [(10, 20), (10, 10), (10, 10), (10, 20)])

    assert_equals(fs2[1].image.data, c.image.data)
    assert_equals(fs2[1].x, 10)
    assert_equals(fs2[1].y, 20)
    assert_equals(fs2[1].r, 5)
    assert_equals(fs2[1].points, [(5, 15), (15, 15), (15, 25), (5, 25)])


def test_color_map():
    c = ColorMap((Color.RED, Color.BLUE, Color.WHITE), 10.5, 190.0)

    c2 = do_pickle(c)

    assert_equals(c2.color.tolist(), [[255, 0, 0], [0, 0, 255], [255, 255, 255]])
    assert_equals(c2.start_map, 10.5)
    assert_equals(c2.end_map, 190.0)
    assert_equals(c2.value_range, 179.5)
    assert_equals(c2.color_distance, 89.75)


def test_color_model():
    cm = ColorModel()
    cm.add(Color.RED)
    cm.add(Color.GREEN)

    cm2 = do_pickle(cm)

    assert_equals(cm2.data, {'\x00@\x00': 1, '\x00\x00\x7f': 1})
    assert_equals(cm2.bits, 1)
    assert_true(cm2.is_background)


def test_font():
    f = Font('ubuntu', 16)
    f2 = do_pickle(f)
    assert_equals(f2._font_face, 'ubuntu')
    assert_equals(f2._font_size, 16)


def test_linescan():
    line = np.array([[0, 0, 255], [0, 255, 255], [0, 255, 0],
                     [255, 0, 255], [255, 255, 255]])
    ls = LineScan(line)

    ls2 = do_pickle(ls)

    assert_equals(ls2, [[0, 0, 255], [0, 255, 255], [0, 255, 0],
                        [255, 0, 255], [255, 255, 255]])


def test_color_segmentation():
    cs = ColorSegmentation()
    cs2 = do_pickle(cs)

    assert_false(cs2.error)
    assert_is_instance(cs2.color_model, ColorModel)
    assert_is_none(cs2.cur_img)
    assert_is_none(cs2.truth_img)


def test_diff_segmentation():
    ds = DiffSegmentation(grayonly=True, threshold=(30, 40, 50))
    ds2 = do_pickle(ds)

    assert_false(ds2.error)
    assert_true(ds2.grayonly_mode)
    assert_equals(ds2.threshold, (30, 40, 50))
    assert_is_none(ds2.curr_img)
    assert_is_none(ds2.last_img)
    assert_is_none(ds2.diff_img)
    assert_is_none(ds2.color_img)


def test_mog_segmentation():
    mog = MOGSegmentation(history=200, mixtures=5, bg_ratio=0.3, noise_sigma=16,
                          learning_rate=0.7)
    mog2 = do_pickle(mog)

    assert_equals(mog2.history, 200)
    assert_equals(mog2.mixtures, 5)
    assert_equals(mog2.bg_ratio, 0.3)
    assert_equals(mog2.noise_sigma, 16)
    assert_equals(mog2.learning_rate, 0.7)


def test_running_segmentation():
    r = RunningSegmentation(alpha=0.5, thresh=(40, 20, 30))
    r2 = do_pickle(r)

    assert_equals(r2.alpha, 0.5)
    assert_equals(r2.thresh, (40, 20, 30))
