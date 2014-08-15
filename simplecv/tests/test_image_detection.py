import pickle

import cv2
import numpy as np
import math
from nose.tools import assert_equals, assert_is_instance, assert_greater \
                      , assert_less, assert_is_none, assert_is_not_none

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
CHESSBOARD_IMAGE = "../data/sampleimages/CalibImage3.png"
TEMPLATE_TEST_IMG = "../data/sampleimages/templatetest.png"
TEMPLATE_IMG = "../data/sampleimages/template.png"
testimageclr = "../data/sampleimages/statue_liberty.jpg"
circles = "../data/sampleimages/circles.png"

#alpha masking images
topImg = "../data/sampleimages/RatTop.png"


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
    perform_diff(result, name_stem, tolerance=0.5)
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

def test_find_skintone_blobs():
    img = Image('../data/sampleimages/04000.jpg')
    blobs = img.find_skintone_blobs()
    for b in blobs:
        assert_greater(b.area, 0)
        assert_greater(b.get_perimeter(), 0)
        assert_greater(b.avg_color[0], 0)
        assert_greater(b.avg_color[1], 0)
        assert_greater(b.avg_color[2], 0)
        assert_less(b.avg_color[0], 255)
        assert_less(b.avg_color[1], 255)
        assert_less(b.avg_color[2], 255)

    img = Image((100, 100))
    blobs = img.find_skintone_blobs()
    assert_equals(blobs, None)


def test_find_chessboard():
    img = Image(CHESSBOARD_IMAGE)
    feat = img.find_chessboard(subpixel=False)

    feat[0].draw()
    name_stem = "test_chessboard"
    perform_diff([img], name_stem, 0.0)

    img = Image(CHESSBOARD_IMAGE)
    feat = img.find_chessboard()
    assert_equals(feat, None)

    # Empty Image
    img = Image((100, 100))
    feat = img.find_chessboard()
    assert_equals(feat, None)

def test_find_template():
    results = []
    source = Image(TEMPLATE_TEST_IMG)
    source2 = source.copy()
    template = Image(TEMPLATE_IMG)

    fs = source.find_template(template, threshold=2)
    fs.draw()
    results.append(source)

    fs = source2.find_template(template, threshold=2, grayscale=False)
    fs.draw()
    results.append(source2)

    name_stem = "test_find_template"
    perform_diff(results, name_stem)

    # method = "SQR_DIFF"
    fs = source.find_template(template, threshold=3, method="SQR_DIFF")
    assert_is_not_none(fs)

    # method = "CCOEFF"
    fs = source.find_template(template, threshold=3, method="CCOEFF")
    assert_is_not_none(fs)

    # method = "CCOEFF_NORM"
    fs = source.find_template(template, threshold=3, method="CCOEFF_NORM", 
                              rawmatches=True)
    assert_is_not_none(fs)

    # method = "CCORR"
    fs = source.find_template(template, threshold=3, method="CCORR")
    assert_is_not_none(fs)

    # method = "CCORR_NORM"
    fs = source.find_template(template, threshold=3, method="CCORR_NORM",
                              rawmatches=True)
    assert_is_not_none(fs)

    # method = "UNKOWN"
    fs = source.find_template(template, threshold=3, method="UNKOWN")
    assert_is_none(fs)

    # None template
    template = None
    assert_is_none(source.find_template(template))

    # Template bigger than image
    template = source.resize(source.width+10, source.height)
    assert_is_none(source.find_template(template))

    template = source.resize(source.width, source.height + 10)
    assert_is_none(source.find_template(template))

def test_find_template_once():
    source = Image(TEMPLATE_TEST_IMG)
    template = Image(TEMPLATE_IMG)

    t = 2
    fs = source.find_template_once(template, threshold=t)
    assert len(fs) != 0

    fs = source.find_template_once(template, threshold=t, grayscale=False)
    assert len(fs) != 0

    fs = source.find_template_once(template, method='CCORR_NORM')
    assert len(fs) != 0

    fs = source.find_template_once(template, threshold=3, method="SQR_DIFF")
    assert_is_not_none(fs)

    # method = "CCOEFF"
    fs = source.find_template_once(template, threshold=3, method="CCOEFF")
    assert_is_not_none(fs)

    # method = "CCOEFF_NORM"
    fs = source.find_template_once(template, threshold=3, method="CCOEFF_NORM")
    assert_is_not_none(fs)    
    # method = "CCORR"
    fs = source.find_template_once(template, threshold=3, method="CCORR")
    assert_is_not_none(fs)

    # method = "CCORR_NORM"
    fs = source.find_template_once(template, threshold=3, method="CCORR_NORM")
    assert_is_not_none(fs)

    # method = "UNKOWN"
    fs = source.find_template_once(template, threshold=3, method="UNKOWN")
    assert_is_none(fs)

    # None template
    template = None
    assert_is_none(source.find_template_once(template))

    # Template bigger than image
    template = source.resize(source.width+10, source.height)
    assert_is_none(source.find_template_once(template))

    template = source.resize(source.width, source.height + 10)
    assert_is_none(source.find_template_once(template))

def test_find_circles():
    img = Image(circles)
    circs = img.find_circle(thresh=85)
    assert_equals(5, len(circs))
    circs.draw()
    assert circs[0] >= 1
    circs[0].coordinates()
    circs[0].get_width()
    circs[0].get_area()
    circs[0].get_perimeter()
    circs[0].get_height()
    circs[0].radius()
    circs[0].diameter()
    circs[0].color_distance()
    circs[0].mean_color()
    circs[0].distance_from(point=(0, 0))
    circs[0].draw()
    assert circs[0].crop()
    assert circs[0].crop(no_mask=True)

    results = [img]
    name_stem = "test_find_circle"
    perform_diff(results, name_stem)

    # find no circle
    img = Image((100, 100)) # Black Image
    assert_is_none(img.find_circle())

def test_find_keypoint_match():
    template = Image("../data/sampleimages/KeypointTemplate2.png")
    match0 = Image("../data/sampleimages/kptest0.png")
    match1 = Image("../data/sampleimages/kptest1.png")
    match2 = Image("../data/sampleimages/kptest2.png")

    fs0 = match0.find_keypoint_match(template)  # test zero
    fs1 = match1.find_keypoint_match(template, quality=300.00, min_dist=0.5,
                                     min_match=0.2)
    fs2 = match2.find_keypoint_match(template, quality=300.00, min_dist=0.5,
                                     min_match=0.2)

    for fs in [fs0, fs1, fs2]:
        assert fs
        assert_equals(1, len(fs))
        fs.draw()
        f = fs[0]
        f.draw_rect()
        f.draw()
        f.get_homography()
        f.get_min_rect()
        f.coordinates()
        f.crop()
        f.mean_color()

    match3 = Image("../data/sampleimages/aerospace.jpg")
    fs3 = match3.find_keypoint_match(template, quality=500.00, min_dist=0.2,
                                     min_match=0.1)
    assert fs3 is None

    # None template
    assert_is_none(match0.find_keypoint_match(None))

    # No keypoints found
    img = Image((100, 100)) # Black image
    assert_is_none(img.find_keypoint_match(template))

def test_find_keypoints():
    img = Image(testimage2)
    if cv2.__version__.startswith('$Rev:'):
        flavors = ['SURF', 'STAR', 'SIFT']  # supported in 2.3.1
    elif cv2.__version__ == '2.4.0' or cv2.__version__ == '2.4.1':
        flavors = ['SURF', 'STAR', 'FAST', 'MSER', 'ORB', 'BRISK', 'SIFT',
                   'Dense']
    else:
        flavors = ['SURF', 'STAR', 'FAST', 'MSER', 'ORB', 'BRISK', 'FREAK',
                   'SIFT', 'Dense']
    for flavor in flavors:
        try:
            print "trying to find " + flavor + " keypoints."
            kp = img.find_keypoints(flavor=flavor, min_quality=100)
        except:
            continue
        if kp is not None:
            print "Found: " + str(len(kp))
            for k in kp:
                k.get_object()
                k.get_descriptor()
                k.quality()
                k.get_octave()
                k.get_flavor()
                k.get_angle()
                k.coordinates()
                k.draw()
                k.distance_from()
                k.mean_color()
                k.get_area()
                k.get_perimeter()
                k.get_width()
                k.get_height()
                k.radius()
            kp[0].crop()
            kp.draw()
        else:
            print "Found None."
    results = [img]
    name_stem = "test_find_keypoints"
    perform_diff(results, name_stem)

    # UNKOWN flavor
    assert_is_none(img.find_keypoints(flavor="UNKOWN"))
    assert_is_none(img._get_raw_keypoints(flavor="UNKOWN")[0])

def test_find_motion():
    current1 = Image("../data/sampleimages/flow_simple1.png")
    prev = Image("../data/sampleimages/flow_simple2.png")

    fs = current1.find_motion(prev, window=7)
    assert_greater(len(fs), 0)
    fs[0].draw(color=Color.RED)
    img = fs[0].crop()
    color = fs[1].mean_color()
    wndw = fs[1].window_sz()
    for f in fs:
        f.vector()
        f.magnitude()

    current2 = Image("../data/sampleimages/flow_simple1.png")
    fs = current2.find_motion(prev, window=7)
    assert_greater(len(fs), 0)
    fs[0].draw(color=Color.RED)
    img = fs[0].crop()
    color = fs[1].mean_color()
    wndw = fs[1].window_sz()
    for f in fs:
        f.vector()
        f.magnitude()

    current3 = Image("../data/sampleimages/flow_simple1.png")
    fs = current3.find_motion(prev, window=7, aggregate=False)
    assert_greater(len(fs), 0)
    fs[0].draw(color=Color.RED)
    img = fs[0].crop()
    color = fs[1].mean_color()
    wndw = fs[1].window_sz()
    for f in fs:
        f.vector()
        f.magnitude()

    # different frame sizes
    current4 = current3.resize(current3.width/2, current3.height/2)
    assert_is_none(current4.find_motion(prev))

def test_find_blobs_from_palette():
    img = Image(testimageclr)
    img = img.scale(0.1)  # scale down the image to reduce test time
    p = img.get_palette()
    b1 = img.find_blobs_from_palette(p[0:5])
    b1.draw()
    assert_greater(len(b1), 0)

    p = img.get_palette(hue=True)
    b2 = img.find_blobs_from_palette(p[0:5])
    b2.draw()
    assert_greater(len(b2), 0)

    # dilate
    b3 = img.find_blobs_from_palette(p[0:5], dilate=1)
    b3.draw()
    assert_greater(len(b3), 0)

def test_smart_find_blobs():
    img = Image(topImg)
    mask = Image((img.width, img.height))
    mask.dl().circle((100, 100), 80, color=Color.MAYBE_BACKGROUND, filled=True)
    mask.dl().circle((100, 100), 60, color=Color.MAYBE_FOREGROUND, filled=True)
    mask.dl().circle((100, 100), 40, color=Color.FOREGROUND, filled=True)
    mask = mask.apply_layers()
    blobs = img.smart_find_blobs(mask=mask)
    blobs.draw()
    assert_equals(1, len(blobs))

    for t in range(2, 5):
        img = Image(topImg)
        blobs2 = img.smart_find_blobs(rect=(30, 30, 150, 185), thresh_level=t)
        assert_equals(1, len(blobs2))
        blobs2.draw()

    blobs2 = img.smart_find_blobs(rect=(30, 30, 150, 185),
                  thresh_level=1)
    assert_equals(len(blobs2), 0)

def test_find_blobs_from_mask():
    img = Image(testimage2)
    mask = img.binarize(inverted=True).invert()
    b1 = img.find_blobs_from_mask(mask)
    b2 = img.find_blobs()
    b1.draw()
    b2.draw()

    results = [img]
    name_stem = "test_find_blobs_from_mask"
    perform_diff(results, name_stem)

    assert len(b1) == len(b2)

    # different mask size
    mask = mask.resize(mask.width/2, mask.height/2)
    assert_is_none(img.find_blobs_from_mask(mask))

    # no blobs
    mask = Image(img.size) # Black mask
    blobs = img.find_blobs_from_mask(mask)
    assert_is_none(img.find_blobs_from_mask(mask))

def test_find_flood_fill_blobs():
    img = Image(testimage2)
    blobs = img.find_flood_fill_blobs((((10, 10), (20, 20), (50, 50))),
                                      tolerance=30)
    blobs.draw()
    name_stem = "test_find_flood_fill_blobs"
    results = [img]
    perform_diff(results, name_stem)


def test_find_grid_lines():
    img = Image("simplecv")
    img = img.grid((10, 10), (0, 255, 255))
    lines = img.find_grid_lines()
    assert lines
    lines.draw()
    result = [img]
    name_stem = "test_image_grid_lines"
    perform_diff(result, name_stem, 5)

    # no grid
    img = Image((100, 100))
    assert_is_none(img.find_grid_lines())

def test_match_sift_key_points():
    img = Image("lenna")
    skp, tkp = img.match_sift_key_points(img)
    assert_equals(len(skp), len(tkp))

    for i in range(len(skp)):
        assert_equals(skp[i].x, tkp[i].x)
        assert_equals(skp[i].y, tkp[i].y)

def test_find_features():
    img = Image('../data/sampleimages/mtest.png')
    h_features = img.find_features("harris", threshold=500)
    assert h_features
    s_features = img.find_features("szeliski", threshold=500)
    assert s_features

    # UNKOWN method
    assert_is_none(img.find_features("UNKOWN", threshold=500))

def test_find_keypoint_clusters():
    img = Image('simplecv')
    kpc = img.find_keypoint_clusters()
    assert_greater(len(kpc), 0)

    kpc1 = img.find_keypoint_clusters(flavor="corner")
    assert_greater(len(kpc1), 0)

    # no keypoints
    img1 = Image((100, 100))
    kpc2 = img1.find_keypoint_clusters(flavor="sift")
    assert_is_none(kpc2)

def test_get_freak_descriptor():
    if '$Rev' in cv2.__version__:
        return

    if int(cv2.__version__.replace('.', '0')) >= 20402:
        img = Image("lenna")
        flavors = ["SIFT", "SURF", "BRISK", "ORB", "STAR", "MSER", "FAST",
                   "Dense"]
        for flavor in flavors:
            f, d = img.get_freak_descriptor(flavor)
            assert_greater(len(f), 0)
            assert_equals(len(f), d.shape[0])
            assert_equals(64, d.shape[1])
"""
def test_image_fit_edge():
    np_array = np.zeros((32, 32), dtype=np.uint8)
    img = Image(array=np_array)

    #coords = img.bresenham_line((5, 6), (30, 25))
    linescan = img.get_line_scan(pt1=(5, 6), pt2=(28, 25))
    list1 = [255]*len(linescan)
    linescan = linescan + list1
    
    new_img = img.replace_line_scan(linescan)
    print new_img.get_ndarray()
    new_img.show()
    guess = [(5, 6), (28, 25)]
    print new_img.fit_edge(guess, window=2)
""" 

def test_image_fit_lines():
    np_array = np.zeros((32, 32), dtype=np.uint8)
    img = Image(array=np_array)

    linescan = img.get_line_scan(pt1=(5, 6), pt2=(23, 25))
    list1 = [255]*len(linescan)
    linescan = linescan + list1
    
    new_img = img.replace_line_scan(linescan)
    guess = [((4, 6), (22, 26))]
    line = new_img.fit_lines(guess, window=2)

    pt1 = line[0].end_points[0]
    pt2 = line[0].end_points[1]

    thresh1 = math.pow(pt1[0]-5, 2) + math.pow(pt1[1]-6,2)
    thresh2 = math.pow(pt2[0]-23, 2) + math.pow(pt2[1]-25,2)

    if thresh1 > 5 or thresh2 > 5:
        assert False

def test_image_fit_line_points():
    np_array = np.zeros((32, 32), dtype=np.uint8)
    img = Image(array=np_array)

    linescan = img.get_line_scan(pt1=(5, 6), pt2=(23, 25))
    list1 = [255]*len(linescan)
    linescan = linescan + list1
    
    new_img = img.replace_line_scan(linescan)
    guess = [((4, 6), (22, 26))]
    new_img.fit_line_points(guess, window=(2,2))


def test_edge_snap():
    img = Image('shapes.png', sample=True).edges()

    list1 = [(129, 32), (19, 88), (124, 135)]
    list2 = [(484, 294), (297, 437)]
    list3 = [(158, 357), (339, 82)]

    for l in list1, list2, list3:
        edge_lines = img.edge_snap(l)
        edge_lines.draw(color=Color.YELLOW, width=4)

    name_stem = "test_edge_snap"
    result = [img]
    perform_diff(result, name_stem)

    # non binary image
    img = Image('shapes.png', sample=True)
    assert_is_none(img.edge_snap(l))

def test_smart_rotate():
    img = Image('kptest2.png', sample=True)

    st1 = img.smart_rotate(auto=False, fixed=False).resize(500, 500)
    st2 = img.rotate(27, fixed=False).resize(500, 500)
    diff = np.average((st1 - st2).get_ndarray())
    assert diff <= 1.7
    if diff > 1.7:
        print diff
        assert False
    else:
        assert True

    # give empty image
    img = Image((100, 100))
    assert_equals(img.smart_rotate().get_ndarray().data, img.get_ndarray().data)
