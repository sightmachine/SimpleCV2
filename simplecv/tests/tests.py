# To run this test you need python nose tools installed
# Run test just use:
#   nosetest tests.py
#
# *Note: If you add additional test, please prefix the function name
# to the type of operation being performed.  For instance modifying an
# image, test_image_erode().  If you are looking for lines, then
# test_detection_lines().  This makes it easier to verify visually
# that all the correct test per operation exist

import os
import pickle
import tempfile

import cv2
import numpy as np
from nose.tools import assert_equals, assert_list_equal

from simplecv.base import logger
from simplecv.color import Color, ColorMap
from simplecv.drawing_layer import DrawingLayer
from simplecv.features.blobmaker import BlobMaker
from simplecv.features.detection import Corner, Line, ROI
from simplecv.features.facerecognizer import FaceRecognizer
from simplecv.features.features import FeatureSet
from simplecv.features.haar_cascade import HaarCascade
from simplecv.image import Image
from simplecv.image_set import ImageSet
from simplecv.linescan import LineScan
from simplecv.segmentation.color_segmentation import ColorSegmentation
from simplecv.segmentation.diff_segmentation import DiffSegmentation
from simplecv.segmentation.running_segmentation import RunningSegmentation

from simplecv.tests.utils import perform_diff

SHOW_WARNING_TESTS = True   # show that warnings are working - tests will pass
                            #  but warnings are generated.

#colors
black = Color.BLACK
white = Color.WHITE
red = Color.RED
green = Color.GREEN
blue = Color.BLUE

###############
# TODO -
# Examples of how to do profiling
# Examples of how to do a single test -
# UPDATE THE VISUAL TESTS WITH EXAMPLES.
# Fix exif data
# Turn off test warnings using decorators.
# Write a use the tests doc.

#images
barcode = "../data/sampleimages/barcode.png"
contour_hiearachy = "../data/sampleimages/contour_hiearachy.png"
testimage = "../data/sampleimages/9dots4lines.png"
testimage2 = "../data/sampleimages/aerospace.jpg"
whiteimage = "../data/sampleimages/white.png"
blackimage = "../data/sampleimages/black.png"
testimageclr = "../data/sampleimages/statue_liberty.jpg"
testbarcode = "../data/sampleimages/barcode.png"
testoutput = "../data/sampleimages/9d4l.jpg"
tmpimg = "../data/sampleimages/tmpimg.jpg"
greyscaleimage = "../data/sampleimages/greyscale.jpg"
logo = "../data/sampleimages/simplecv.png"
logo_inverted = "../data/sampleimages/simplecv_inverted.png"
ocrimage = "../data/sampleimages/ocr-test.png"
circles = "../data/sampleimages/circles.png"
webp = "../data/sampleimages/simplecv.webp"

#alpha masking images
topImg = "../data/sampleimages/RatTop.png"
bottomImg = "../data/sampleimages/RatBottom.png"
maskImg = "../data/sampleimages/RatMask.png"
alphaMaskImg = "../data/sampleimages/RatAlphaMask.png"
alphaSrcImg = "../data/sampleimages/GreenMaskSource.png"


def test_detection_find_corners():
    img = Image(testimage2)
    corners = img.find_corners(25)
    corners.draw()
    assert len(corners)
    result = [img]
    name_stem = "test_detection_find_corners"
    perform_diff(result, name_stem)


def test_image_histogram():
    img = Image(testimage2)
    h = img.histogram(25)

    for i in h:
        if type(i) != int:
            assert False


def test_detection_lines():
    img = Image(testimage2)
    lines = img.find_lines()
    lines.draw()
    result = [img]
    name_stem = "test_detection_lines"
    perform_diff(result, name_stem)

    if lines == 0 or lines is None:
        assert False


def test_detection_lines_standard():
    img = Image(testimage2)
    lines = img.find_lines(use_standard=True)
    lines.draw()
    result = [img]
    name_stem = "test_detection_lines_standard"
    perform_diff(result, name_stem)

    if lines == 0 or lines is None:
        assert False


# FIXME: Test should have assertions
def test_detection_feature_measures():
    img = Image(testimage2)

    fs = FeatureSet()
    fs.append(Corner(img, 5, 5))
    fs.append(Line(img, ((2, 2), (3, 3))))
    bm = BlobMaker()
    result = bm.extract(img)
    fs.extend(result)

    for f in fs:
        a = f.get_area()
        l = f.length()
        c = f.mean_color()
        d = f.color_distance()
        th = f.get_angle()
        pts = f.coordinates()
        dist = f.distance_from()  # distance from center of image

    fs2 = fs.sort_angle()
    fs3 = fs.sort_length()
    fs4 = fs.sort_color_distance()
    fs5 = fs.sort_area()
    fs1 = fs.sort_distance()


def test_detection_blobs_appx():
    img = Image("lenna")
    blobs = img.find_blobs()
    assert len(blobs) == 29
    blobs[-1].draw(color=Color.RED)
    blobs[-1].draw_appx(color=Color.BLUE)
    result = [img]

    img2 = Image("lenna")
    blobs = img2.find_blobs(appx_level=11)
    assert len(blobs) == 29
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
    assert len(blobs) == 5
    result.append(img)

    img = Image(contour_hiearachy)
    blobs = img.find_blobs()
    assert len(blobs) == 10
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
    blobs.draw(color=Color.RED)
    result = [img]
    name_stem = "test_detection_blobs_adaptive"
    perform_diff(result, name_stem)
    assert blobs is not None


def test_detection_blobs_smallimages():
    # Check if segfault occurs or not
    img = Image("../data/sampleimages/blobsegfaultimage.png")
    blobs = img.find_blobs()
    # if no segfault, pass


def test_detection_blobs_convexity_defects():
    if not hasattr(cv2, 'convexityDefects'):
        return

    img = Image('lenna')
    blobs = img.find_blobs()
    b = blobs[-1]
    feat = b.get_convexity_defects()
    points = b.get_convexity_defects(return_points=True)
    if len(feat) <= 0 or len(points) <= 0:
        assert False


def test_detection_barcode():
    try:
        import zbar
    except:
        return None

    img1 = Image(testimage)
    img2 = Image(testbarcode)

    if SHOW_WARNING_TESTS:
        nocode = img1.find_barcode()
        if nocode:  # we should find no barcode in our test image
            assert False
        code = img2.find_barcode()
        code.draw()
        if code.points:
            pass
        result = [img1, img2]
        name_stem = "test_detection_barcode"
        perform_diff(result, name_stem)


def test_detection_x():
    tmp_x = Image(testimage).find_lines().x()[0]

    if tmp_x > 0 and Image(testimage).size[0]:
        pass
    else:
        assert False


def test_detection_y():
    tmp_y = Image(testimage).find_lines().y()[0]

    if tmp_y > 0 and Image(testimage).size[0]:
        pass
    else:
        assert False


def test_detection_area():
    img = Image(testimage2)
    bm = BlobMaker()
    result = bm.extract(img)
    area_val = result[0].get_area()

    if area_val > 0:
        pass
    else:
        assert False


def test_detection_angle():
    angle_val = Image(testimage).find_lines().get_angle()[0]


def test_detection_length():
    img = Image(testimage)
    val = img.find_lines().length()

    if val is None:
        assert False
    if not isinstance(val, np.ndarray):
        assert False
    if len(val) < 0:
        assert False


def test_detection_sortangle():
    img = Image(testimage)
    val = img.find_lines().sort_angle()

    if val[0].x < val[1].x:
        pass
    else:
        assert False


def test_detection_sortarea():
    img = Image(testimage)
    bm = BlobMaker()
    result = bm.extract(img)
    val = result.sort_area()
    # FIXME: Find blobs may appear to be broken. Returning type none


def test_detection_sort_length():
    img = Image(testimage)
    val = img.find_lines().sort_length()
    # FIXME: Length is being returned as euclidean type,
    # believe we need a universal type, either Int or scvINT or something.


def test_feature_get_height():
    img_a = Image(logo)
    lines = img_a.find_lines(1)
    heights = lines.get_height()

    if len(heights) <= 0:
        assert False


def test_feature_get_width():
    img_a = Image(logo)
    lines = img_a.find_lines(1)
    widths = lines.get_width()

    if len(widths) <= 0:
        assert False


def test_feature_crop():
    img_a = Image(logo)

    lines = img_a.find_lines()

    cropped_images = lines.crop()

    if len(cropped_images) <= 0:
        assert False


def test_blob_maker():
    img = Image("../data/sampleimages/blockhead.png")
    blobber = BlobMaker()
    results = blobber.extract(img)
    print(len(results))
    if len(results) != 7:
        assert False


def test_blob_holes():
    img = Image("../data/sampleimages/blockhead.png")
    blobber = BlobMaker()
    blobs = blobber.extract(img)
    count = 0
    blobs.draw()
    results = [img]
    name_stem = "test_blob_holes"
    perform_diff(results, name_stem)

    for b in blobs:
        if b.hole_contour is not None:
            count += len(b.hole_contour)
    if count != 7:
        assert False

    for b in blobs:
        if len(b.convex_hull) < 3:
            assert False


def test_blob_data():
    # FIXME: Test should have assertion
    img = Image("../data/sampleimages/blockhead.png")
    blobber = BlobMaker()
    blobs = blobber.extract(img)
    for b in blobs:
        if b.area > 0:
            pass
        if b.get_perimeter() > 0:
            pass
        if sum(b.avg_color) > 0:
            pass
        if sum(b.bounding_box) > 0:
            pass
        if b.m00 is not 0 \
                and b.m01 is not 0 \
                and b.m10 is not 0 \
                and b.m11 is not 0 \
                and b.m20 is not 0 \
                and b.m02 is not 0 \
                and b.m21 is not 0 \
                and b.m12 is not 0:
            pass
        if sum(b.hu) > 0:
            pass


def test_blob_render():
    img = Image("../data/sampleimages/blockhead.png")
    blobber = BlobMaker()
    blobs = blobber.extract(img)
    dl = DrawingLayer((img.width, img.height))
    reimg = DrawingLayer((img.width, img.height))
    for b in blobs:
        b.draw(color=Color.RED, alpha=128)
        b.draw_holes(width=2, color=Color.BLUE)
        b.draw_hull(color=Color.ORANGE, width=2)
        b.draw(color=Color.RED, alpha=128, layer=dl)
        b.draw_holes(width=2, color=Color.BLUE, layer=dl)
        b.draw_hull(color=Color.ORANGE, width=2, layer=dl)
        b.draw_mask_to_layer(reimg)

    img.add_drawing_layer(dl)
    results = [img]
    name_stem = "test_blob_render"
    perform_diff(results, name_stem, tolerance=5.0)


def test_blob_methods():
    # FIXME: Test should have assertion
    img = Image("../data/sampleimages/blockhead.png")
    blobber = BlobMaker()
    blobs = blobber.extract(img)
    bl = (img.width, img.height)
    first = blobs[0]
    for b in blobs:
        b.get_width()
        b.get_height()
        b.get_area()
        b.get_max_x()
        b.get_min_x()
        b.get_max_y()
        b.get_min_y()
        b.min_rect_width()
        b.min_rect_height()
        b.min_rect_x()
        b.min_rect_y()
        b.get_contour()
        b.get_aspect_ratio()
        b.blob_image()
        b.blob_mask()
        b.get_hull_img()
        b.get_hull_mask()
        b.rectify_major_axis()
        b.blob_image()
        b.blob_mask()
        b.get_hull_img()
        b.get_hull_mask()
        b.get_angle()
        b.above(first)
        b.below(first)
        b.left(first)
        b.right(first)
        #b.contains(first)
        #b.overlaps(first)


def test_template_match():
    results = []
    source = Image("../data/sampleimages/templatetest.png")
    source2 = source.copy()
    template = Image("../data/sampleimages/template.png")

    fs = source.find_template(template, threshold=2)
    fs.draw()
    results.append(source)

    fs = source2.find_template(template, threshold=2, grayscale=False)
    fs.draw()
    results.append(source2)

    name_stem = "test_template_match"
    perform_diff(results, name_stem)


def test_template_match_once():
    source = Image("../data/sampleimages/templatetest.png")
    template = Image("../data/sampleimages/template.png")
    t = 2
    fs = source.find_template_once(template, threshold=t)
    assert len(fs) != 0

    fs = source.find_template_once(template, threshold=t, grayscale=False)
    assert len(fs) != 0

    fs = source.find_template_once(template, method='CCORR_NORM')
    assert len(fs) != 0


def test_segmentation_diff():
    segmentor = DiffSegmentation()
    i1 = Image("logo")
    i2 = Image("logo_inverted")
    segmentor.add_image(i1)
    segmentor.add_image(i2)
    blobs = segmentor.get_segmented_blobs()
    if blobs is None:
        assert False


def test_segmentation_running():
    segmentor = RunningSegmentation()
    i1 = Image("logo")
    i2 = Image("logo_inverted")
    segmentor.add_image(i1)
    segmentor.add_image(i2)
    blobs = segmentor.get_segmented_blobs()
    if blobs is None:
        assert False


def test_segmentation_color():
    segmentor = ColorSegmentation()
    i1 = Image("logo")
    i2 = Image("logo_inverted")
    segmentor.add_image(i1)
    segmentor.add_image(i2)
    blobs = segmentor.get_segmented_blobs()
    if blobs is None:
        assert False


def test_blit_regular():
    top = Image(topImg)
    bottom = Image(bottomImg)
    results = []
    results.append(bottom.blit(top))
    results.append(bottom.blit(top, pos=(-10, -10)))
    results.append(bottom.blit(top, pos=(-10, 10)))
    results.append(bottom.blit(top, pos=(10, -10)))
    results.append(bottom.blit(top, pos=(10, 10)))

    name_stem = "test_blit_regular"
    perform_diff(results, name_stem)


def test_blit_mask():
    top = Image(topImg)
    bottom = Image(bottomImg)
    mask = Image(maskImg)
    results = []
    results.append(bottom.blit(top, mask=mask))
    results.append(bottom.blit(top, mask=mask, pos=(-50, -50)))
    results.append(bottom.blit(top, mask=mask, pos=(-50, 50)))
    results.append(bottom.blit(top, mask=mask, pos=(50, -50)))
    results.append(bottom.blit(top, mask=mask, pos=(50, 50)))

    name_stem = "test_blit_mask"
    perform_diff(results, name_stem)


def test_blit_alpha():
    top = Image(topImg)
    bottom = Image(bottomImg)
    a = 0.5
    results = []
    results.append(bottom.blit(top, alpha=a))
    results.append(bottom.blit(top, alpha=a, pos=(-50, -50)))
    results.append(bottom.blit(top, alpha=a, pos=(-50, 50)))
    results.append(bottom.blit(top, alpha=a, pos=(50, -50)))
    results.append(bottom.blit(top, alpha=a, pos=(50, 50)))
    name_stem = "test_blit_alpha"
    perform_diff(results, name_stem)


def test_blit_alpha_mask():
    top = Image(topImg)
    bottom = Image(bottomImg)
    a_mask = Image(alphaMaskImg)
    results = []

    results.append(bottom.blit(top, alpha_mask=a_mask))
    results.append(bottom.blit(top, alpha_mask=a_mask, pos=(-10, -10)))
    results.append(bottom.blit(top, alpha_mask=a_mask, pos=(-10, 10)))
    results.append(bottom.blit(top, alpha_mask=a_mask, pos=(10, -10)))
    results.append(bottom.blit(top, alpha_mask=a_mask, pos=(10, 10)))

    name_stem = "test_blit_alpha_mask"
    perform_diff(results, name_stem)


def test_imageset():
    imgs = ImageSet()
    assert isinstance(imgs, ImageSet)


def test_hsv_conversion():
    px = Image((1, 1))
    px[0, 0] = Color.GREEN
    assert_list_equal(Color.hsv(Color.GREEN), px.to_hsv()[0, 0])


def test_white_balance():
    img = Image("../data/sampleimages/BadWB2.jpg")
    output = img.white_balance()
    output2 = img.white_balance(method="GrayWorld")
    results = [output, output2]
    name_stem = "test_white_balance"
    perform_diff(results, name_stem)


def test_hough_circles():
    img = Image(circles)
    circs = img.find_circle(thresh=100)
    assert_equals(5, len(circs))
    circs.draw()
    if circs[0] < 1:
        assert False
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
    name_stem = "test_hough_circle"
    perform_diff(results, name_stem)


def test_draw_rectangle():
    img = Image(testimage2)
    img.draw_rectangle(0, 0, 100, 100, color=Color.BLUE, width=0, alpha=0)
    img.draw_rectangle(1, 1, 100, 100, color=Color.BLUE, width=2, alpha=128)
    img.draw_rectangle(1, 1, 100, 100, color=Color.BLUE, width=1, alpha=128)
    img.draw_rectangle(2, 2, 100, 100, color=Color.BLUE, width=1, alpha=255)
    img.draw_rectangle(3, 3, 100, 100, color=Color.BLUE)
    img.draw_rectangle(4, 4, 100, 100, color=Color.BLUE, width=12)
    img.draw_rectangle(5, 5, 100, 100, color=Color.BLUE, width=-1)

    results = [img]
    name_stem = "test_draw_rectangle"
    perform_diff(results, name_stem)


def test_blob_min_rect():
    img = Image(testimageclr)
    blobs = img.find_blobs()
    for b in blobs:
        b.draw_min_rect(color=Color.BLUE, width=3, alpha=123)
    results = [img]
    name_stem = "test_blob_min_rect"
    perform_diff(results, name_stem)


def test_blob_rect():
    img = Image(testimageclr)
    blobs = img.find_blobs()
    for b in blobs:
        b.draw_rect(color=Color.BLUE, width=3, alpha=123)
    results = [img]
    name_stem = "test_blob_rect"
    perform_diff(results, name_stem)


def test_blob_pickle():
    img = Image(testimageclr)
    blobs = img.find_blobs()
    for b in blobs:
        p = pickle.dumps(b)
        ub = pickle.loads(p)
        assert (ub.mask - b.mask).mean_color() == 0


def test_blob_isa_methods():
    img1 = Image(circles)
    img2 = Image("../data/sampleimages/blockhead.png")
    blobs = img1.find_blobs().sort_area()
    t1 = blobs[-1].is_circle()
    f1 = blobs[-1].is_rectangle()
    blobs = img2.find_blobs().sort_area()
    f2 = blobs[-1].is_circle()
    t2 = blobs[-1].is_rectangle()
    if t1 and t2 and not f1 and not f2:
        pass
    else:
        assert False


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
            kp = img.find_keypoints(flavor=flavor)
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
                k.crop()
            kp.draw()
        else:
            print "Found None."
    results = [img]
    name_stem = "test_find_keypoints"
    perform_diff(results, name_stem)


def test_movement_feature():
    current1 = Image("../data/sampleimages/flow_simple1.png")
    prev = Image("../data/sampleimages/flow_simple2.png")

    fs = current1.find_motion(prev, window=7)
    if len(fs) > 0:
        fs.draw(color=Color.RED)
        img = fs[0].crop()
        color = fs[1].mean_color()
        wndw = fs[1].window_sz()
        for f in fs:
            f.vector()
            f.magnitude()
    else:
        assert False

    current2 = Image("../data/sampleimages/flow_simple1.png")
    fs = current2.find_motion(prev, window=7)
    if len(fs) > 0:
        fs.draw(color=Color.RED)
        img = fs[0].crop()
        color = fs[1].mean_color()
        wndw = fs[1].window_sz()
        for f in fs:
            f.vector()
            f.magnitude()
    else:
        assert False

    current3 = Image("../data/sampleimages/flow_simple1.png")
    fs = current3.find_motion(prev, window=7, aggregate=False)
    if len(fs) > 0:
        fs.draw(color=Color.RED)
        img = fs[0].crop()
        color = fs[1].mean_color()
        wndw = fs[1].window_sz()
        for f in fs:
            f.vector()
            f.magnitude()
    else:
        assert False

    results = [current1, current2, current3]
    name_stem = "test_movement_feature"
    #~ perform_diff(results,name_stem,tolerance=4.0)


def test_keypoint_extraction():
    img1 = Image("../data/sampleimages/KeypointTemplate2.png")
    img2 = Image("../data/sampleimages/KeypointTemplate2.png")
    img3 = Image("../data/sampleimages/KeypointTemplate2.png")
    img4 = Image("../data/sampleimages/KeypointTemplate2.png")

    kp1 = img1.find_keypoints()
    kp2 = img2.find_keypoints(highquality=True)
    kp3 = img3.find_keypoints(flavor="STAR")
    if not cv2.__version__.startswith("$Rev:"):
        kp4 = img4.find_keypoints(flavor="BRISK")
        kp4.draw()
        if len(kp4) == 0:
            assert False
    kp1.draw()
    kp2.draw()
    kp3.draw()

    #TODO: Fix FAST binding
    #~ kp4 = img.find_keypoints(flavor="FAST",min_quality=10)
    if len(kp1) == 190 \
            and len(kp2) == 190 \
            and len(kp3) == 37:  # ~ and len(kp4)==521):
        pass
    else:
        assert False
    results = [img1, img2, img3]
    name_stem = "test_keypoint_extraction"
    perform_diff(results, name_stem, tolerance=4.0)


def test_keypoint_match():
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
        assert fs is not None
        assert_equals(1, len(fs))
        fs.draw()
        f = fs[0]
        f.draw_rect()
        f.draw()
        f.get_homography()
        f.get_min_rect()
        f.coordinates()

    match3 = Image("../data/sampleimages/aerospace.jpg")
    fs3 = match3.find_keypoint_match(template, quality=500.00, min_dist=0.2,
                                     min_match=0.1)
    assert fs3 is None


def test_draw_keypoint_matches():
    template = Image("../data/sampleimages/KeypointTemplate2.png")
    match0 = Image("../data/sampleimages/kptest0.png")
    result = match0.draw_keypoint_matches(template, thresh=500.00,
                                          min_dist=0.15, width=1)
    assert_equals(template.width + match0.width, result.width)


def test_basic_palette():
    # FIXME: Test should have assertion
    img = Image(testimageclr)
    img = img.scale(0.1)  # scale down the image to reduce test time
    img.generate_palette(10, False)
    if img._palette is not None \
            and img._palette_members is not None \
            and img._palette_percentages is not None \
            and img._palette_bins == 10:
        img.generate_palette(20, True)
        if img._palette is not None \
                and img._palette_members is not None \
                and img._palette_percentages is not None \
                and img._palette_bins == 20:
            pass


def test_palettize():
    # FIXME: Test should have assertion
    img = Image(testimageclr)
    img = img.scale(0.1)  # scale down the image to reduce test time
    img2 = img.palettize(bins=20, hue=False)
    img3 = img.palettize(bins=3, hue=True)
    img4 = img.palettize(centroids=[Color.WHITE, Color.RED, Color.BLUE,
                                    Color.GREEN, Color.BLACK])
    img4 = img.palettize(hue=True, centroids=[0, 30, 60, 180])
    # UHG@! can't diff because of the kmeans initial conditions causes
    # things to bounce around... otherwise we need to set a friggin
    # huge tolerance

    #results = [img2,img3]
    #name_stem = "test_palettize"
    #perform_diff(results,name_stem)


def test_repalette():
    # FIXME: Test should have assertion
    img = Image(testimageclr)
    img = img.scale(0.1)  # scale down the image to reduce test time
    img2 = Image(bottomImg)
    img2 = img2.scale(0.1)  # scale down the image to reduce test time
    p = img.get_palette()
    img3 = img2.re_palette(p)
    p = img.get_palette(hue=True)
    img4 = img2.re_palette(p, hue=True)

    #results = [img3,img4]
    #name_stem = "test_repalette"
    #perform_diff(results,name_stem)


def test_draw_palette():
    # FIXME: Test should have assertion
    img = Image(testimageclr)
    img = img.scale(0.1)  # scale down the image to reduce test time
    img1 = img.draw_palette_colors()
    img2 = img.draw_palette_colors(horizontal=False)
    img3 = img.draw_palette_colors(size=(69, 420))
    img4 = img.draw_palette_colors(size=(69, 420), horizontal=False)
    img5 = img.draw_palette_colors(hue=True)
    img6 = img.draw_palette_colors(horizontal=False, hue=True)
    img7 = img.draw_palette_colors(size=(69, 420), hue=True)
    img8 = img.draw_palette_colors(size=(69, 420), horizontal=False, hue=True)


def test_palette_binarize():
    # FIXME: Test should have assertion
    img = Image(testimageclr)
    img = img.scale(0.1)  # scale down the image to reduce test time
    p = img.get_palette()
    img2 = img.binarize_from_palette(p[0:5])
    p = img.get_palette(hue=True)
    img2 = img.binarize_from_palette(p[0:5])


def test_palette_blobs():
    img = Image(testimageclr)
    img = img.scale(0.1)  # scale down the image to reduce test time
    p = img.get_palette()
    b1 = img.find_blobs_from_palette(p[0:5])
    b1.draw()
    assert len(b1) > 0

    p = img.get_palette(hue=True)
    b2 = img.find_blobs_from_palette(p[0:5])
    b2.draw()
    assert len(b2) > 0


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


def test_image_webp_save():
    #only run if webm suppport exist on system
    try:
        import webm
    except:
        if SHOW_WARNING_TESTS:
            logger.warning("Couldn't run the webp test as optional webm "
                           "library required")
    else:
        img = Image('simplecv')
        tf = tempfile.NamedTemporaryFile(suffix=".webp")
        if img.save(tf.name):
            pass
        else:
            assert False


def test_detection_spatial_relationships():
    img = Image(testimageclr)
    template = img.crop(200, 200, 50, 50)
    motion = img.embiggen((img.width + 10, img.height + 10), pos=(10, 10))
    motion = motion.crop(0, 0, img.width, img.height)
    blob_fs = img.find_blobs()
    line_fs = img.find_lines()
    corn_fs = img.find_corners()
    move_fs = img.find_motion(motion)
    move_fs = FeatureSet(move_fs[42:52])  # l337 s5p33d h4ck - okay not really
    temp_fs = img.find_template(template, threshold=1)
    a_circ = (img.width / 2, img.height / 2,
              np.min([img.width / 2, img.height / 2]))
    a_rect = (50, 50, 200, 200)
    a_point = (img.width / 2, img.height / 2)
    a_poly = [(0, 0), (img.width / 2, 0), (0, img.height / 2)]  # a triangle

    feats = [blob_fs, line_fs, corn_fs, temp_fs, move_fs]

    for f in feats:
        print str(len(f))

    for f in feats:
        for g in feats:
            sample = f[0]
            sample2 = f[1]
            print type(f[0])
            print type(g[0])

            g.above(sample)
            g.below(sample)
            g.left(sample)
            g.right(sample)
            g.overlaps(sample)
            g.inside(sample)
            g.outside(sample)

            g.inside(a_rect)
            g.outside(a_rect)

            g.inside(a_circ)
            g.outside(a_circ)

            g.inside(a_poly)
            g.outside(a_poly)

            g.above(a_point)
            g.below(a_point)
            g.left(a_point)
            g.right(a_point)


def test_get_raw_dft():
    img = Image("../data/sampleimages/RedDog2.jpg")
    raw3 = img.raw_dft_image()
    raw1 = img.raw_dft_image(grayscale=True)

    assert len(raw1) == 1
    assert raw1[0].shape[1] == img.width
    assert raw1[0].shape[0] == img.height
    assert raw1[0].dtype == np.float64

    assert len(raw3) == 3
    assert raw3[0].shape[1] == img.width
    assert raw3[0].shape[0] == img.height
    assert raw3[0].dtype == np.float64
    assert raw3[0].shape[2] == 2


def test_get_dft_log_magnitude():
    img = Image("../data/sampleimages/RedDog2.jpg")
    lm3 = img.get_dft_log_magnitude()
    lm1 = img.get_dft_log_magnitude(grayscale=True)

    results = [lm3, lm1]
    name_stem = "test_get_dft_log_magnitude"
    perform_diff(results, name_stem, tolerance=6.0)


def test_find_haar_features():
    img = Image("../data/sampleimages/orson_welles.jpg")
    img1 = img.copy()
    face = HaarCascade("face.xml")  # old HaarCascade
    f = img.find_haar_features(face)
    f2 = img1.find_haar_features("face_cv2.xml")  # new cv2 HaarCascade
    assert len(f) > 0
    assert len(f2) > 0
    f.draw()
    f2.draw()
    f[0].get_width()
    f[0].get_height()
    f[0].length()
    f[0].get_area()

    results = [img, img1]
    name_stem = "test_find_haar_features"
    perform_diff(results, name_stem)


def test_biblical_flood_fill():
    results = []
    img = Image(testimage2)
    b = img.find_blobs()
    results.append(img.flood_fill(b.coordinates(), tolerance=3,
                                  color=Color.RED))
    results.append(img.flood_fill(b.coordinates(), tolerance=(3, 3, 3),
                                  color=Color.BLUE))
    results.append(img.flood_fill(b.coordinates(), tolerance=(3, 3, 3),
                                  color=Color.GREEN, fixed_range=False))
    img.flood_fill((30, 30), lower=3, upper=5, color=Color.ORANGE)
    img.flood_fill((30, 30), lower=3, upper=(5, 5, 5), color=Color.ORANGE)
    img.flood_fill((30, 30), lower=(3, 3, 3), upper=5, color=Color.ORANGE)
    img.flood_fill((30, 30), lower=(3, 3, 3), upper=(5, 5, 5))
    img.flood_fill((30, 30), lower=(3, 3, 3), upper=(5, 5, 5),
                   color=np.array([255, 0, 0]))
    img.flood_fill((30, 30), lower=(3, 3, 3), upper=(5, 5, 5),
                   color=[255, 0, 0])

    name_stem = "test_biblical_flood_fill"
    perform_diff(results, name_stem)


def test_flood_fill_to_mask():
    img = Image(testimage2)
    b = img.find_blobs()
    imask = img.edges()
    omask = img.flood_fill_to_mask(b.coordinates(), tolerance=10)
    omask2 = img.flood_fill_to_mask(b.coordinates(), tolerance=(3, 3, 3),
                                    mask=imask)
    omask3 = img.flood_fill_to_mask(b.coordinates(), tolerance=(3, 3, 3),
                                    mask=imask, fixed_range=False)

    results = [omask, omask2, omask3]
    name_stem = "test_flood_fill_to_mask"
    perform_diff(results, name_stem)


def test_find_blobs_from_mask():
    img = Image(testimage2)
    mask = img.binarize().invert()
    b1 = img.find_blobs_from_mask(mask)
    b2 = img.find_blobs()
    b1.draw()
    b2.draw()

    results = [img]
    name_stem = "test_find_blobs_from_mask"
    perform_diff(results, name_stem)

    assert len(b1) == len(b2)


def test_image_slice():
    img = Image("../data/sampleimages/blockhead.png")
    i = img.find_lines()
    i2 = i[0:10]
    if type(i2) == list:
        assert False


def test_blob_spatial_relationships():
    img = Image("../data/sampleimages/spatial_relationships.png")
    # please see the image
    blobs = img.find_blobs(threshval=1)
    blobs = blobs.sort_area()
    print(len(blobs))

    center = blobs[-1]
    top = blobs[-2]
    right = blobs[-3]
    bottom = blobs[-4]
    left = blobs[-5]
    inside = blobs[-7]
    overlap = blobs[-6]

    assert top.above(center)
    assert bottom.below(center)
    assert right.right(center)
    assert left.left(center)
    assert center.contains(inside)
    assert not center.contains(left)
    assert center.overlaps(overlap)
    assert overlap.overlaps(center)

    my_tuple = (img.width / 2, img.height / 2)

    assert top.above(my_tuple)
    assert bottom.below(my_tuple)
    assert right.right(my_tuple)
    assert left.left(my_tuple)

    assert top.above(my_tuple)
    assert bottom.below(my_tuple)
    assert right.right(my_tuple)
    assert left.left(my_tuple)
    assert center.contains(my_tuple)

    my_npa = np.array([img.width / 2, img.height / 2])

    assert top.above(my_npa)
    assert bottom.below(my_npa)
    assert right.right(my_npa)
    assert left.left(my_npa)
    assert center.contains(my_npa)

    assert center.contains(inside)


def test_get_aspectratio():
    img = Image("../data/sampleimages/EdgeTest1.png")
    img2 = Image("../data/sampleimages/EdgeTest2.png")
    b = img.find_blobs()
    l = img2.find_lines()
    c = img2.find_circle(thresh=200)
    c2 = img2.find_corners()
    kp = img2.find_keypoints()
    bb = b.aspect_ratios()
    ll = l.aspect_ratios()
    cc = c.aspect_ratios()
    c22 = c2.aspect_ratios()
    kp2 = kp.aspect_ratios()

    if len(bb) > 0 and len(ll) > 0 \
            and len(cc) > 0 and len(c22) > 0\
            and len(kp2) > 0:
        pass
    else:
        assert False


def test_line_crop():
    img = Image("../data/sampleimages/EdgeTest2.png")
    l = img.find_lines().sort_area()
    l = l[-5:-1]
    results = []
    for ls in l:
        results.append(ls.crop())
    name_stem = "test_line_crop"
    perform_diff(results, name_stem, tolerance=3.0)


def test_get_corners():
    img = Image("../data/sampleimages/EdgeTest1.png")
    img2 = Image("../data/sampleimages/EdgeTest2.png")
    b = img.find_blobs()
    tl = b.top_left_corners()
    tr = b.top_right_corners()
    bl = b.bottom_left_corners()
    br = b.bottom_right_corners()

    l = img2.find_lines()
    tl2 = l.top_left_corners()
    tr2 = l.top_right_corners()
    bl2 = l.bottom_left_corners()
    br2 = l.bottom_right_corners()

    if tl is not None\
            and tr is not None \
            and bl is not None \
            and br is not None \
            and tl2 is not None \
            and tr2 is not None \
            and bl2 is not None \
            and br2 is not None:
        pass
    else:
        assert False


def test_save_kwargs():
    img = Image("lenna")
    l95 = "l95.jpg"
    l90 = "l90.jpg"
    l80 = "l80.jpg"
    l70 = "l70.jpg"

    img.save(l95, quality=95)
    img.save(l90, quality=90)
    img.save(l80, quality=80)
    img.save(l70, quality=75)

    s95 = os.stat(l95).st_size
    s90 = os.stat(l90).st_size
    s80 = os.stat(l80).st_size
    s70 = os.stat(l70).st_size

    if s70 < s80 and s80 < s90 and s90 < s95:
        pass
    else:
        assert False

    s95 = os.remove(l95)
    s90 = os.remove(l90)
    s80 = os.remove(l80)
    s70 = os.remove(l70)


def test_on_edge():
    img1 = "./../data/sampleimages/EdgeTest1.png"
    img2 = "./../data/sampleimages/EdgeTest2.png"
    img_a = Image(img1)
    img_b = Image(img2)
    img_c = Image(img2)
    img_d = Image(img2)
    img_e = Image(img2)

    blobs = img_a.find_blobs()
    circs = img_b.find_circle(thresh=200)
    corners = img_c.find_corners()
    kp = img_d.find_keypoints()
    lines = img_e.find_lines()

    rim = blobs.on_image_edge()
    inside = blobs.not_on_image_edge()
    rim.draw(color=Color.RED)
    inside.draw(color=Color.BLUE)

    rim = circs.on_image_edge()
    inside = circs.not_on_image_edge()
    rim.draw(color=Color.RED)
    inside.draw(color=Color.BLUE)

    #rim =  corners.on_image_edge()
    inside = corners.not_on_image_edge()
    #rim.draw(color=Color.RED)
    inside.draw(color=Color.BLUE)

    #rim =  kp.on_image_edge()
    inside = kp.not_on_image_edge()
    #rim.draw(color=Color.RED)
    inside.draw(color=Color.BLUE)

    rim = lines.on_image_edge()
    inside = lines.not_on_image_edge()
    rim.draw(color=Color.RED)
    inside.draw(color=Color.BLUE)

    results = [img_a, img_b, img_c, img_d, img_e]
    name_stem = "test_onEdge_Features"
    #~ perform_diff(results,name_stem,tolerance=8.0)


def test_feature_angles():
    img = Image("../data/sampleimages/rotation2.png")
    img2 = Image("../data/sampleimages/rotation.jpg")
    img3 = Image("../data/sampleimages/rotation.jpg")
    b = img.find_blobs()
    l = img2.find_lines()
    k = img3.find_keypoints()

    for bs in b:
        tl = bs.top_left_corner()
        img.draw_text(str(bs.get_angle()), tl[0], tl[1], color=Color.RED)

    for ls in l:
        tl = ls.top_left_corner()
        img2.draw_text(str(ls.get_angle()), tl[0], tl[1], color=Color.GREEN)

    for ks in k:
        tl = ks.top_left_corner()
        img3.draw_text(str(ks.get_angle()), tl[0], tl[1], color=Color.BLUE)

    results = [img, img2, img3]
    name_stem = "test_feature_angles"
    perform_diff(results, name_stem, tolerance=11.0)


def test_feature_angles_rotate():
    img = Image("../data/sampleimages/rotation2.png")
    blobs = img.find_blobs()
    assert_equals(13, len(blobs))

    for b in blobs:
        temp = b.crop()
        assert isinstance(temp, Image)
        derp = temp.rotate(b.get_angle(), fixed=False)
        derp.draw_text(str(b.get_angle()), 10, 10, color=Color.RED)
        b.rectify_major_axis()
        assert isinstance(b.blob_image(), Image)


def test_minrect_blobs():
    img = Image("../data/sampleimages/bolt.png")
    img = img.invert()
    results = []
    for i in range(-10, 10):
        ang = float(i * 18.00)
        print ang
        t = img.rotate(ang)
        b = t.find_blobs(threshval=128)
        b[-1].draw_min_rect(color=Color.RED, width=5)
        results.append(t)

    name_stem = "test_minrect_blobs"
    perform_diff(results, name_stem, tolerance=11.0)


def test_point_intersection():
    img = Image("simplecv")
    e = img.edges(0, 100)
    for x in range(25, 225, 25):
        a = (x, 25)
        b = (125, 125)
        pts = img.edge_intersections(a, b, width=1)
        e.draw_line(a, b, color=Color.RED)
        e.draw_circle(pts[0], 10, color=Color.GREEN)

    for x in range(25, 225, 25):
        a = (25, x)
        b = (125, 125)
        pts = img.edge_intersections(a, b, width=1)
        e.draw_line(a, b, color=Color.RED)
        e.draw_circle(pts[0], 10, color=Color.GREEN)

    for x in range(25, 225, 25):
        a = (x, 225)
        b = (125, 125)
        pts = img.edge_intersections(a, b, width=1)
        e.draw_line(a, b, color=Color.RED)
        e.draw_circle(pts[0], 10, color=Color.GREEN)

    for x in range(25, 225, 25):
        a = (225, x)
        b = (125, 125)
        pts = img.edge_intersections(a, b, width=1)
        e.draw_line(a, b, color=Color.RED)
        e.draw_circle(pts[0], 10, color=Color.GREEN)

    results = [e]
    name_stem = "test_point_intersection"
    perform_diff(results, name_stem, tolerance=6.0)


def test_find_skintone_blobs():
    # FIXME: Test should have assertion
    img = Image('../data/sampleimages/04000.jpg')

    blobs = img.find_skintone_blobs()
    for b in blobs:
        if b.area > 0:
            pass
        if b.get_perimeter() > 0:
            pass
        if b.avg_color[0] > 5 \
                and b.avg_color[1] > 140 \
                and b.avg_color[1] < 180 \
                and b.avg_color[2] > 77 \
                and b.avg_color[2] < 135:
            pass


def test_get_skintone_mask():
    img_set = []
    img_set.append(Image('../data/sampleimages/040000.jpg'))
    img_set.append(Image('../data/sampleimages/040001.jpg'))
    img_set.append(Image('../data/sampleimages/040002.jpg'))
    img_set.append(Image('../data/sampleimages/040003.jpg'))
    img_set.append(Image('../data/sampleimages/040004.jpg'))
    img_set.append(Image('../data/sampleimages/040005.jpg'))
    img_set.append(Image('../data/sampleimages/040006.jpg'))
    img_set.append(Image('../data/sampleimages/040007.jpg'))
    masks = [img.get_skintone_mask() for img in img_set]
    name_stem = 'test_skintone'
    masks.append(img_set[0].get_skintone_mask(dilate_iter=1))
    masks.append(img_set[0].get_skintone_mask(dilate_iter=2))
    masks.append(img_set[0].get_skintone_mask(dilate_iter=3))
    perform_diff(masks, name_stem)


def test_find_keypoints_all():
    img = Image(testimage2)
    methods = ["ORB", "SIFT", "SURF", "FAST", "STAR", "MSER", "Dense"]
    for i in methods:
        print i
        try:
            kp = img.find_keypoints(flavor=i)
        except:
            continue
        if kp is not None:
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
                k.crop()
            kp.draw()
        results = [img]
        name_stem = "test_find_keypoints"
        #~ perform_diff(results,name_stem,tolerance=8)


def test_image_new_crop():
    img = Image(logo)
    x = 5
    y = 6
    w = 10
    h = 20
    crop = img.crop((x, y, w, h))
    crop1 = img.crop([x, y, w, h])
    crop2 = img.crop((x, y), (x + w, y + h))
    crop3 = img.crop([(x, y), (x + w, y + h)])
    if SHOW_WARNING_TESTS:
        crop7 = img.crop((0, 0, -10, 10))
        crop8 = img.crop((-50, -50), (10, 10))
        crop9 = img.crop([(-3, -3), (10, 20)])
        crop10 = img.crop((-10, 10, 20, 20), centered=True)
        crop11 = img.crop([-10, -10, 20, 20])

    results = [crop, crop1, crop2, crop3]
    name_stem = "test_image_new_crop"
    perform_diff(results, name_stem)

    diff = crop - crop1
    assert_equals((0, 0, 0), diff.mean_color())


def test_image_temp_save():
    img1 = Image("lenna")
    img2 = Image(logo)
    path = []
    path.append(img1.save(temp=True))
    path.append(img2.save(temp=True))
    for i in path:
        if i is None:
            assert False


def test_image_set_average():
    iset = ImageSet()
    iset.append(Image("./../data/sampleimages/tracktest0.jpg"))
    iset.append(Image("./../data/sampleimages/tracktest1.jpg"))
    iset.append(Image("./../data/sampleimages/tracktest2.jpg"))
    iset.append(Image("./../data/sampleimages/tracktest3.jpg"))
    iset.append(Image("./../data/sampleimages/tracktest4.jpg"))
    iset.append(Image("./../data/sampleimages/tracktest5.jpg"))
    iset.append(Image("./../data/sampleimages/tracktest6.jpg"))
    iset.append(Image("./../data/sampleimages/tracktest7.jpg"))
    iset.append(Image("./../data/sampleimages/tracktest8.jpg"))
    iset.append(Image("./../data/sampleimages/tracktest9.jpg"))
    avg = iset.average()
    result = [avg]
    name_stem = "test_image_set_average"
    perform_diff(result, name_stem)


def test_save_to_gif():
    imgs = ImageSet()
    imgs.append(Image('../data/sampleimages/tracktest0.jpg'))
    imgs.append(Image('../data/sampleimages/tracktest1.jpg'))
    imgs.append(Image('../data/sampleimages/tracktest2.jpg'))
    imgs.append(Image('../data/sampleimages/tracktest3.jpg'))
    imgs.append(Image('../data/sampleimages/tracktest4.jpg'))
    imgs.append(Image('../data/sampleimages/tracktest5.jpg'))
    imgs.append(Image('../data/sampleimages/tracktest6.jpg'))
    imgs.append(Image('../data/sampleimages/tracktest7.jpg'))
    imgs.append(Image('../data/sampleimages/tracktest8.jpg'))
    imgs.append(Image('../data/sampleimages/tracktest9.jpg'))

    filename = "test_save_to_gif.gif"
    saved = imgs.save(filename)

    os.remove(filename)

    assert saved == len(imgs)


def test_sliceing_image_set():
    imgset = ImageSet("../data/sampleimages/")
    imgset = imgset[8::-2]
    if isinstance(imgset, ImageSet):
        assert True
    else:
        assert False


def test_builtin_rotations():
    img = Image('lenna')
    r1 = img - img.rotate180().rotate180()
    r2 = img - img.rotate90().rotate90().rotate90().rotate90()
    r3 = img - img.rotate_left().rotate_left().rotate_left().rotate_left()
    r4 = img - img.rotate_right().rotate_right().rotate_right().rotate_right()
    r5 = img - img.rotate270().rotate270().rotate270().rotate270()
    if r1.mean_color() == Color.BLACK \
            and r2.mean_color() == Color.BLACK \
            and r3.mean_color() == Color.BLACK \
            and r4.mean_color() == Color.BLACK \
            and r5.mean_color() == Color.BLACK:
        pass
    else:
        assert False


def test_histograms():
    img = Image('lenna')
    img.vertical_histogram()
    img.horizontal_histogram()

    img.vertical_histogram(bins=3)
    img.horizontal_histogram(bins=3)

    img.vertical_histogram(threshold=10)
    img.horizontal_histogram(threshold=255)

    img.vertical_histogram(normalize=True)
    img.horizontal_histogram(normalize=True)

    img.vertical_histogram(for_plot=True, normalize=True)
    img.horizontal_histogram(for_plot=True, normalize=True)


def test_blob_full_masks():
    img = Image('lenna')
    b = img.find_blobs()
    m1 = b[-1].get_full_masked_image()
    m2 = b[-1].get_full_hull_masked_image()
    m3 = b[-1].get_full_mask()
    m4 = b[-1].get_full_hull_mask()
    assert m1.width == img.width
    assert m2.width == img.width
    assert m3.width == img.width
    assert m4.width == img.width
    assert m1.height == img.height
    assert m2.height == img.height
    assert m3.height == img.height
    assert m4.height == img.height


def test_blob_edge_images():
    img = Image('lenna')
    b = img.find_blobs()
    m1 = b[-1].get_edge_image()
    assert isinstance(m1, Image)
    assert_equals(m1.size, img.size)
    m2 = b[-1].get_hull_edge_image()
    assert isinstance(m2, Image)
    assert_equals(m1.size, img.size)
    m3 = b[-1].get_full_edge_image()
    assert isinstance(m3, Image)
    assert_equals(m3.size, img.size)
    m4 = b[-1].get_full_hull_edge_image()
    assert isinstance(m4, Image)
    assert_equals(m4.size, img.size)


def test_line_scan():
    def lsstuff(ls):
        def a_line(x, m, b):
            return m * x + b

        ls2 = ls.smooth(degree=4)
        ls2 = ls2.normalize()
        ls2 = ls2.scale(value_range=[-1, 1])
        ls2 = ls2.derivative()
        ls2 = ls2.resample(100)
        ls2 = ls2.convolve([.25, 0.25, 0.25, 0.25])
        ls2.minima()
        ls2.maxima()
        ls2.local_minima()
        ls2.local_maxima()
        fft, f = ls2.fft()
        ls3 = ls2.ifft(fft)
        ls4 = ls3.fit_to_model(a_line)
        ls4.get_model_parameters(a_line)

    img = Image("lenna")
    ls = img.get_line_scan(x=128, channel=1)
    lsstuff(ls)
    ls = img.get_line_scan(y=128)
    lsstuff(ls)
    ls = img.get_line_scan(pt1=(0, 0), pt2=(128, 128), channel=2)
    lsstuff(ls)


def test_uncrop():
    img = Image('lenna')
    cropped_img = img.crop(10, 20, 250, 500)
    source_pts = cropped_img.uncrop([(2, 3), (56, 23), (24, 87)])
    if source_pts:
        pass


def test_grid():
    img = Image("simplecv")
    img1 = img.grid((10, 10), (0, 255, 0), 1)
    img2 = img.grid((20, 20), (255, 0, 255), 1)
    img3 = img.grid((20, 20), (255, 0, 255), 2)
    result = [img1, img2, img3]
    name_stem = "test_image_grid"
    perform_diff(result, name_stem, 12.0)


def test_remove_grid():
    img = Image("lenna")
    grid_image = img.grid()
    dlayer = grid_image.remove_grid()
    if dlayer is None:
        assert False
    dlayer1 = grid_image.remove_grid()
    if dlayer1 is not None:
        assert False


def test_cluster():
    img = Image("lenna")
    blobs = img.find_blobs()
    clusters1 = blobs.cluster(method="kmeans", k=5, properties=["color"])
    clusters2 = blobs.cluster(method="hierarchical")
    if clusters1 and clusters2:
        pass


def test_line_parallel():
    img = Image("lenna")
    l1 = Line(img, ((100, 200), (300, 400)))
    l2 = Line(img, ((200, 300), (400, 500)))
    if l1.is_parallel(l2):
        pass
    else:
        assert False


def test_line_perp():
    img = Image("lenna")
    l1 = Line(img, ((100, 200), (100, 400)))
    l2 = Line(img, ((200, 300), (400, 300)))
    if l1.is_perpendicular(l2):
        pass
    else:
        assert False


def test_line_img_intersection():
    img = Image((512, 512))
    for x in range(200, 400):
        img[200, x] = (255.0, 255.0, 255.0)
    l = Line(img, ((300, 100), (300, 500)))
    assert_equals([(300, 200)], l.img_intersections(img))


def test_line_crop_to_edges():
    img = Image((512, 512))
    l = Line(img, ((-10, -5), (400, 400)))
    l_cr = l.crop_to_image_edges()
    if l_cr.end_points == ((0, 5), (400, 400)):
        pass
    else:
        assert False


def test_line_extend_to_edges():
    img = Image((512, 512))
    l = Line(img, ((10, 10), (30, 30)))
    l_ext = l.extend_to_image_edges()
    if l_ext.end_points == [(0, 0), (511, 511)]:
        pass
    else:
        assert False


def test_find_grid_lines():
    img = Image("simplecv")
    img = img.grid((10, 10), (0, 255, 255))
    lines = img.find_grid_lines()
    assert lines
    lines.draw()
    result = [img]
    name_stem = "test_image_grid_lines"
    perform_diff(result, name_stem, 5)


def test_logical_and():
    img = Image("lenna")
    img1 = img.logical_and(img.invert())
    if not img1.get_ndarray().all():
        pass
    else:
        assert False


def test_logical_or():
    img = Image("lenna")
    img1 = img.logical_or(img.invert())
    if img1.get_ndarray().all():
        pass
    else:
        assert False


def test_logical_nand():
    img = Image("lenna")
    img1 = img.logical_nand(img.invert())
    if img1.get_ndarray().all():
        pass
    else:
        assert False


def test_logical_xor():
    img = Image("lenna")
    img1 = img.logical_xor(img.invert())
    if img1.get_ndarray().all():
        pass
    else:
        assert False


def test_match_sift_key_points():
    try:
        import cv2
    except ImportError:
        pass
        return
    if not "2.4.3" in cv2.__version__:
        pass
        return
    img = Image("lenna")
    skp, tkp = img.match_sift_key_points(img)
    if len(skp) == len(tkp):
        for i in range(len(skp)):
            if skp[i].x == tkp[i].x and skp[i].y == tkp[i].y:
                pass
            else:
                assert False
    else:
        assert False


def test_find_features():
    img = Image('../data/sampleimages/mtest.png')
    h_features = img.find_features("harris", threshold=500)
    s_features = img.find_features("szeliski", threshold=500)
    if h_features and s_features:
        pass
    else:
        assert False


def test_color_map():
    img = Image('../data/sampleimages/mtest.png')
    blobs = img.find_blobs()
    cm = ColorMap((Color.RED, Color.YELLOW, Color.BLUE), min(blobs.get_area()),
                  max(blobs.get_area()))
    for b in blobs:
        b.draw(cm[b.get_area()])
    result = [img]
    name_stem = "test_color_map"
    perform_diff(result, name_stem)


def test_minmax():
    img = Image('lenna')
    gray_img = img.to_gray()
    assert_equals(25, img.min_value())
    min, points = img.min_value(locations=True)
    assert_equals(25, min)
    for p in points:
        assert_equals(25, gray_img[p])

    assert_equals(245, img.max_value())
    max, points = img.max_value(locations=True)
    assert_equals(245, max)
    for p in points:
        assert_equals(245, gray_img[p])


def test_roi_feature():
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

    assert_list_equal([320, 0, 121, 53], roi_list[0].to_xywh())
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


def test_find_keypoint_clusters():
    img = Image('simplecv')
    kpc = img.find_keypoint_clusters()
    if len(kpc) <= 0:
        assert False


def test_replace_line_scan():
    img = Image("lenna")
    ls = img.get_line_scan(x=100)
    ls[50] = 0
    newimg = img.replace_line_scan(ls)
    assert_equals(0, newimg[50, 100])
    ls = img.get_line_scan(x=100, channel=1)
    ls[50] = 0
    new_img = img.replace_line_scan(ls)
    assert_equals(0, new_img[50, 100][1])


def test_running_average():
    img = Image('lenna')
    ls = img.get_line_scan(y=120)
    ra = ls.running_average(5)
    if ra[50] == sum(ls[48:53]) / 5:
        pass
    else:
        assert False


def line_scan_perform_diff(o_linescan, p_linescan, func, **kwargs):
    n_linescan = func(o_linescan, **kwargs)
    diff = sum([(i - j) for i, j in zip(p_linescan, n_linescan)])
    if diff > 10 or diff < -10:
        return False
    return True


def test_linescan_smooth():
    img = Image("lenna")
    l1 = img.get_line_scan(x=60)
    l2 = l1.smooth(degree=7)
    if line_scan_perform_diff(l1, l2, LineScan.smooth, degree=7):
        pass
    else:
        assert False


def test_linescan_normalize():
    img = Image("lenna")
    l1 = img.get_line_scan(x=90)
    l2 = l1.normalize()
    if line_scan_perform_diff(l1, l2, LineScan.normalize):
        pass
    else:
        assert False


def test_linescan_scale():
    img = Image("lenna")
    l1 = img.get_line_scan(y=90)
    l2 = l1.scale()
    if line_scan_perform_diff(l1, l2, LineScan.scale):
        pass
    else:
        assert False


def test_linescan_derivative():
    img = Image("lenna")
    l1 = img.get_line_scan(y=140)
    l2 = l1.derivative()
    if line_scan_perform_diff(l1, l2, LineScan.derivative):
        pass
    else:
        assert False


def test_linescan_resample():
    img = Image("lenna")
    l1 = img.get_line_scan(pt1=(300, 300), pt2=(450, 500))
    l2 = l1.resample(n=50)
    if line_scan_perform_diff(l1, l2, LineScan.resample, n=50):
        pass
    else:
        assert False


def test_linescan_fit_to_model():
    def a_line(x, m, b):
        return x * m + b

    img = Image("lenna")
    l1 = img.get_line_scan(y=200)
    l2 = l1.fit_to_model(a_line)
    if line_scan_perform_diff(l1, l2, LineScan.fit_to_model, f=a_line):
        pass
    else:
        assert False


def test_linescan_convolve():
    kernel = [0, 2, 0, 4, 0, 2, 0]
    img = Image("lenna")
    l1 = img.get_line_scan(x=400)
    l2 = l1.convolve(kernel)
    assert line_scan_perform_diff(l1, l2, LineScan.convolve, kernel=kernel)


def test_linescan_threshold():
    img = Image("lenna")
    l1 = img.get_line_scan(x=350)
    l2 = l1.threshold(threshold=200, invert=True)
    if line_scan_perform_diff(l1, l2, LineScan.threshold, threshold=200,
                              invert=True):
        pass
    else:
        assert False


def test_linescan_invert():
    img = Image("lenna")
    l1 = img.get_line_scan(y=200)
    l2 = l1.invert(max=40)
    if line_scan_perform_diff(l1, l2, LineScan.invert, max=40):
        pass
    else:
        assert False


def test_linescan_median():
    img = Image("lenna")
    l1 = img.get_line_scan(x=120)
    l2 = l1.median(sz=9)
    if line_scan_perform_diff(l1, l2, LineScan.median, sz=9):
        pass
    else:
        assert False


def test_linescan_median_filter():
    img = Image("lenna")
    l1 = img.get_line_scan(y=250)
    l2 = l1.median_filter(kernel_size=7)
    if line_scan_perform_diff(l1, l2, LineScan.median_filter, kernel_size=7):
        pass
    else:
        assert False


def test_linescan_detrend():
    img = Image("lenna")
    l1 = img.get_line_scan(y=90)
    l2 = l1.detrend()
    if line_scan_perform_diff(l1, l2, LineScan.detrend):
        pass
    else:
        assert False


def test_get_freak_descriptor():
    try:
        import cv2
    except ImportError:
        pass
    if '$Rev' in cv2.__version__:
        pass
    else:
        if int(cv2.__version__.replace('.', '0')) >= 20402:
            img = Image("lenna")
            flavors = ["SIFT", "SURF", "BRISK", "ORB", "STAR", "MSER", "FAST",
                       "Dense"]
            for flavor in flavors:
                f, d = img.get_freak_descriptor(flavor)
                if len(f) == 0:
                    assert False
                if d.shape[0] != len(f) and d.shape[1] != 64:
                    assert False


def test_gray_peaks():
    i = Image('lenna')
    peaks = i.gray_peaks()
    if peaks is None:
        assert False


def test_find_peaks():
    img = Image('lenna')
    ls = img.get_line_scan(x=150)
    peaks = ls.find_peaks()
    if peaks is None:
        assert False


def test_line_scan_sub():
    img = Image('lenna')
    ls = img.get_line_scan(x=200)
    ls1 = ls - ls
    if ls1[23] == 0:
        pass
    else:
        assert False


def test_line_scan_add():
    img = Image('lenna')
    ls = img.get_line_scan(x=20)
    l = ls + ls
    a = int(ls[20]) + int(ls[20])
    if a == l[20]:
        pass
    else:
        assert False


def test_line_scan_mul():
    img = Image('lenna')
    ls = img.get_line_scan(x=20)
    l = ls * ls
    a = int(ls[20]) * int(ls[20])
    if a == l[20]:
        pass
    else:
        assert False


def test_line_scan_div():
    img = Image('lenna')
    ls = img.get_line_scan(x=20)
    l = ls / ls
    a = int(ls[20]) / int(ls[20])
    if a == l[20]:
        pass
    else:
        assert False


def test_face_recognize():
    try:
        import cv2

        if hasattr(cv2, "createFisherFaceRecognizer"):
            f = FaceRecognizer()
            images1 = ["../data/sampleimages/ff1.jpg",
                       "../data/sampleimages/ff2.jpg",
                       "../data/sampleimages/ff3.jpg",
                       "../data/sampleimages/ff4.jpg",
                       "../data/sampleimages/ff5.jpg"]

            images2 = ["../data/sampleimages/fm1.jpg",
                       "../data/sampleimages/fm2.jpg",
                       "../data/sampleimages/fm3.jpg",
                       "../data/sampleimages/fm4.jpg",
                       "../data/sampleimages/fm5.jpg"]

            images3 = ["../data/sampleimages/fi1.jpg",
                       "../data/sampleimages/fi2.jpg",
                       "../data/sampleimages/fi3.jpg",
                       "../data/sampleimages/fi4.jpg"]

            imgset1 = []
            imgset2 = []
            imgset3 = []

            for img in images1:
                imgset1.append(Image(img))
            label1 = ["female"] * len(imgset1)

            for img in images2:
                imgset2.append(Image(img))
            label2 = ["male"] * len(imgset2)

            imgset = imgset1 + imgset2
            labels = label1 + label2
            imgset[4] = imgset[4].resize(400, 400)
            f.train(imgset, labels)

            for img in images3:
                imgset3.append(Image(img))
            imgset[2].resize(300, 300)
            label = []
            for img in imgset3:
                name, confidence = f.predict(img)
                label.append(name)

            if label == ["male", "male", "female", "female"]:
                pass
            else:
                assert False
    except ImportError:
        pass


def test_channel_mixer():
    i = Image('lenna')
    r = i.channel_mixer()
    g = i.channel_mixer(channel='g', weight=(100, 20, 30))
    b = i.channel_mixer(channel='b', weight=(30, 200, 10))
    if i != r and i != g and i != b:
        pass
    else:
        assert False


def test_prewitt():
    i = Image('lenna')
    p = i.prewitt()
    if i != p:
        pass
    else:
        assert False


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


def test_grayscalmatrix():
    img = Image("lenna")
    graymat = img.get_gray_ndarray()
    newimg = Image(graymat, color_space=Image.GRAY)
    from numpy import array_equal

    if not array_equal(img.get_gray_ndarray(), newimg.get_gray_ndarray()):
        assert False


def test_smart_rotate():
    img = Image('kptest2.png', sample=True)

    st1 = img.smart_rotate(auto=False, fixed=False).resize(500, 500)
    st2 = img.rotate(27, fixed=False).resize(500, 500)
    diff = np.average((st1 - st2).get_ndarray())
    if diff > 1.7:
        print diff
        assert False
    else:
        assert True


def test_get_normalized_hue_histogram():
    img = Image('lenna')
    a = img.get_normalized_hue_histogram((0, 0, 100, 100))
    b = img.get_normalized_hue_histogram()
    blobs = img.find_blobs()
    c = img.get_normalized_hue_histogram(blobs[-1])
    if a.shape == (180, 256) \
            and b.shape == (180, 256) \
            and c.shape == (180, 256):
        pass
    else:
        assert False


def test_back_project_hue_histogram():
    img = Image('lenna')
    img2 = Image('lyle')
    a = img2.get_normalized_hue_histogram()
    img_a = img.back_project_hue_histogram(a)
    img_b = img.back_project_hue_histogram((10, 10, 50, 50), smooth=False,
                                           full_color=True)
    img_c = img.back_project_hue_histogram(img2, threshold=1)
    result = [img_a, img_b, img_c]
    name_stem = "test_image_hist_back_proj"
    perform_diff(result, name_stem, 5)


def test_find_blobs_from_hue_histogram():
    img = Image('lenna')
    img2 = Image('lyle')
    a = img2.get_normalized_hue_histogram()
    a = img.find_blobs_from_hue_histogram(a)
    b = img.find_blobs_from_hue_histogram((10, 10, 50, 50), smooth=False)
    c = img.find_blobs_from_hue_histogram(img2, threshold=1)


def test_drawing_layer_to_svg():
    img = Image('lenna')
    dl = img.dl()
    dl.line((0, 0), (100, 100))
    svg = dl.get_svg()
    if svg == '<svg baseProfile="full" height="512" version="1.1" width="512"'\
              ' xmlns="http://www.w3.org/2000/svg" ' \
              'xmlns:ev="http://www.w3.org/2001/xml-events" ' \
              'xmlns:xlink="http://www.w3.org/1999/xlink"><defs />' \
              '<line x1="0" x2="100" y1="0" y2="100" /></svg>':
        pass
    else:
        assert False
