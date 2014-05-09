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
from nose.tools import assert_equals, assert_list_equal, assert_tuple_equal, \
    assert_true, assert_false, assert_greater, assert_less, \
    assert_is_instance, nottest, assert_is_none

from simplecv.base import logger
from simplecv.color import Color, ColorMap
from simplecv.drawing_layer import DrawingLayer
from simplecv.features.blobmaker import BlobMaker
from simplecv.features.detection import Line, ROI
from simplecv.features.facerecognizer import FaceRecognizer
from simplecv.features.features import FeatureSet
from simplecv.features.haar_cascade import HaarCascade
from simplecv.image import Image
from simplecv.image_set import ImageSet
from simplecv.linescan import LineScan
from simplecv.segmentation.color_segmentation import ColorSegmentation
from simplecv.segmentation.diff_segmentation import DiffSegmentation
from simplecv.segmentation.running_segmentation import RunningSegmentation
from simplecv.stream import JpegStreamer, VideoStream
from simplecv.display import Display

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


def test_segmentation_diff():
    segmentor = DiffSegmentation()
    i1 = Image("logo")
    i2 = Image("logo_inverted")
    segmentor.add_image(i1)
    segmentor.add_image(i2)
    blobs = segmentor.get_segmented_blobs()
    assert blobs


def test_segmentation_running():
    segmentor = RunningSegmentation()
    i1 = Image("logo")
    i2 = Image("logo_inverted")
    segmentor.add_image(i1)
    segmentor.add_image(i2)
    blobs = segmentor.get_segmented_blobs()
    assert blobs


def test_segmentation_color():
    segmentor = ColorSegmentation()
    i1 = Image("logo")
    i2 = Image("logo_inverted")
    segmentor.add_image(i1)
    segmentor.add_image(i2)
    blobs = segmentor.get_segmented_blobs()
    assert blobs


def test_imageset():
    imgs = ImageSet()
    assert isinstance(imgs, ImageSet)


def test_hsv_conversion():
    px = Image((1, 1))
    px[0, 0] = Color.GREEN
    assert_list_equal(Color.hsv(Color.GREEN), px.to_hsv()[0, 0])

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
        assert_equals(0, (ub.mask - b.mask).mean_color())


def test_blob_isa_methods():
    img1 = Image(circles)
    blobs = img1.find_blobs().sort_area()
    assert_true(blobs[-1].is_circle())
    assert_false(blobs[-1].is_rectangle())

    img2 = Image("../data/sampleimages/blockhead.png")
    blobs = img2.find_blobs().sort_area()
    assert_false(blobs[-1].is_circle())
    assert_true(blobs[-1].is_rectangle())


def test_keypoint_extraction():
    img1 = Image("../data/sampleimages/KeypointTemplate2.png")
    img2 = Image("../data/sampleimages/KeypointTemplate2.png")
    img3 = Image("../data/sampleimages/KeypointTemplate2.png")
    img4 = Image("../data/sampleimages/KeypointTemplate2.png")

    kp1 = img1.find_keypoints()
    assert_equals(190, len(kp1))
    kp1.draw()

    kp2 = img2.find_keypoints(highquality=True)
    assert_equals(190, len(kp2))
    kp2.draw()

    kp3 = img3.find_keypoints(flavor="STAR")
    assert_equals(37, len(kp3))
    kp3.draw()

    if not cv2.__version__.startswith("$Rev:"):
        kp4 = img4.find_keypoints(flavor="BRISK")
        kp4.draw()
        assert len(kp4) != 0
    # TODO: Fix FAST binding
    # kp4 = img.find_keypoints(flavor="FAST", min_quality=10)
    # assert_equals(521, len(kp4))

    results = [img1, img2, img3]
    name_stem = "test_keypoint_extraction"
    perform_diff(results, name_stem, tolerance=4.0)


def test_draw_keypoint_matches():
    template = Image("../data/sampleimages/KeypointTemplate2.png")
    match0 = Image("../data/sampleimages/kptest0.png")
    result = match0.draw_keypoint_matches(template, thresh=500.00,
                                          min_dist=0.15, width=1)
    assert_equals(template.width + match0.width, result.width)


def test_basic_palette():
    img = Image(testimageclr)
    img = img.scale(0.1)  # scale down the image to reduce test time
    img.generate_palette(10, False)
    assert_is_instance(img._palette, np.ndarray)
    assert_is_instance(img._palette_members, np.ndarray)
    assert_is_instance(img._palette_percentages, list)
    assert_equals(10, img._palette_bins)

    img.generate_palette(20, True)
    assert_is_instance(img._palette, np.ndarray)
    assert_is_instance(img._palette_members, np.ndarray)
    assert_is_instance(img._palette_percentages, list)
    assert_equals(20, img._palette_bins)


def test_palettize():
    img = Image(testimageclr)
    img = img.scale(0.1)  # scale down the image to reduce test time
    img2 = img.palettize(bins=20, hue=False)
    img3 = img.palettize(bins=3, hue=True)
    img4 = img.palettize(centroids=[Color.WHITE, Color.RED, Color.BLUE,
                                    Color.GREEN, Color.BLACK])
    img5 = img.palettize(hue=True, centroids=[0, 30, 60, 180])
    # UHG@! can't diff because of the kmeans initial conditions causes
    # things to bounce around... otherwise we need to set a friggin
    # huge tolerance
    assert all(map(lambda a: isinstance(a, Image), [img2, img3, img4, img5]))

def test_draw_palette():
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
    assert all(map(lambda a: isinstance(a, Image), [img1, img2, img3, img4,
                                                    img5, img5, img6, img7,
                                                    img8]))


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
        assert img.save(tf.name)


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

def test_image_slice():
    img = Image("../data/sampleimages/blockhead.png")
    i = img.find_lines()
    i2 = i[0:10]
    assert_is_instance(i2, FeatureSet)


def test_blob_spatial_relationships():
    img = Image("../data/sampleimages/spatial_relationships.png")
    # please see the image
    blobs = img.find_blobs(threshval=1)
    blobs = blobs.sort_area()

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
    assert_greater(len(b.aspect_ratios()), 0)
    assert_greater(len(l.aspect_ratios()), 0)
    assert_greater(len(c.aspect_ratios()), 0)
    assert_greater(len(c2.aspect_ratios()), 0)
    assert_greater(len(kp.aspect_ratios()), 0)


def test_get_corners():
    img = Image("../data/sampleimages/EdgeTest1.png")
    img2 = Image("../data/sampleimages/EdgeTest2.png")
    b = img.find_blobs()
    assert_is_instance(b.top_left_corners(), np.ndarray)
    assert_is_instance(b.top_right_corners(), np.ndarray)
    assert_is_instance(b.bottom_left_corners(), np.ndarray)
    assert_is_instance(b.bottom_right_corners(), np.ndarray)

    l = img2.find_lines()
    assert_is_instance(l.top_left_corners(), np.ndarray)
    assert_is_instance(l.top_right_corners(), np.ndarray)
    assert_is_instance(l.bottom_left_corners(), np.ndarray)
    assert_is_instance(l.bottom_right_corners(), np.ndarray)


def test_save_kwargs():
    img = Image("lenna")
    l95 = os.path.join(tempfile.gettempdir(), "l95.jpg")
    l90 = os.path.join(tempfile.gettempdir(), "l90.jpg")
    l80 = os.path.join(tempfile.gettempdir(), "l80.jpg")
    l70 = os.path.join(tempfile.gettempdir(), "l70.jpg")

    img.save(l95, quality=95)
    img.save(l90, quality=90)
    img.save(l80, quality=80)
    img.save(l70, quality=70)

    s95 = os.stat(l95).st_size
    s90 = os.stat(l90).st_size
    s80 = os.stat(l80).st_size
    s70 = os.stat(l70).st_size

    assert_greater(s80, s70)
    assert_greater(s90, s80)
    assert_greater(s95, s90)

    os.remove(l95)
    os.remove(l90)
    os.remove(l80)
    os.remove(l70)

    # invalid path
    img.save(path="/home/unkown_user/desktop/lena.jpg", temp=True)
    img.save(filename="lena")
    assert os.path.exists("lena.png")
    os.remove('lena.png')

    img.filename = "lena.png"
    img.save(verbose=True)
    assert os.path.exists("lena.png")
    os.remove('lena.png')

    img.filename = None
    img.filehandle = "lena.png"
    img.save()
    assert os.path.exists("lena.png")
    os.remove('lena.png')

    # JpegStreamer
    js = JpegStreamer()
    img.save(js)

    # VideoStreamer
    vs = VideoStream("video_stream_test.avi")
    img.save(vs)
    assert os.path.exists("video_stream_test.avi")
    os.remove("video_stream_test.avi")

    # Display
    d = Display()
    img.save(d)
    d.quit()

    # ipython notebook
    d = Display(displaytype="notebook")
    img.save(d)
    d.quit()

def test_delete_temp_files():
    img = Image("lenna")
    img_paths = []
    img_paths.append(img.save(temp=True, clean_temp=True))
    img_paths.append(img.save(temp=True, clean_temp=True))
    img_paths.append(img.save(temp=True, clean_temp=True))

    for path in img_paths:
        assert os.path.exists(path)
    
    img.delete_temp_files()

    for path in img_paths:
        assert not os.path.exists(path)

def test_insert_drawing_layer():
    img = Image("simplecv")
    dl1 = DrawingLayer((img.width, img.height))
    dl2 = DrawingLayer((img.width, img.height))
    img.insert_drawing_layer(dl2, 1)
    img.insert_drawing_layer(dl1, 2)
    assert_equals(len(img._layers), 2)
    assert_equals(img._layers[1], dl1)
    assert_equals(img._layers[0], dl2)

def test_add_drawing_layer():
    img = Image("simplecv")
    dl1 = DrawingLayer((img.width, img.height))
    dl2 = DrawingLayer((img.width, img.height))
    img.add_drawing_layer(dl1)
    img.add_drawing_layer(dl2)
    assert_is_none(img.add_drawing_layer())
    assert_is_none(img.add_drawing_layer(1))
    assert_equals(len(img._layers), 2)
    assert_equals(img._layers[0], dl1)
    assert_equals(img._layers[1], dl2)

def test_remove_drawing_layer():
    img = Image("simplecv")
    dl1 = DrawingLayer((img.width, img.height))
    dl2 = DrawingLayer((img.width, img.height))
    img.add_drawing_layer(dl1)
    img.add_drawing_layer(dl2)
    assert_is_none(img.remove_drawing_layer(3))
    assert_equals(img.remove_drawing_layer(), dl2)
    assert_equals(img.remove_drawing_layer(), dl1)
    assert_is_none(img.remove_drawing_layer())

def test_apply_layers():
    img = Image((100, 100))
    assert_equals(img, img.apply_layers())

    dl1 = DrawingLayer((img.width, img.height))
    dl1.rectangle((10, 10), (10, 10), color=(255, 0, 0))
    dl2 = DrawingLayer((img.width, img.height))
    dl2.rectangle((30, 30), (10, 10), color=(0, 255, 0))
    dl3 = DrawingLayer((img.width, img.height))
    dl3.rectangle((50, 50), (10, 10), color=(0, 0, 255))
    img.add_drawing_layer(dl1)
    img.add_drawing_layer(dl2)
    img.add_drawing_layer(dl3)
    new_img = img.apply_layers()
    assert_equals(new_img[10, 10], [255, 0, 0])
    assert_equals(new_img[30, 30], [0, 255, 0])
    assert_equals(new_img[50, 50], [0, 0, 255])

    img = Image((100, 100))
    img.add_drawing_layer(dl1)
    img.add_drawing_layer(dl2)
    img.add_drawing_layer(dl3)
    new_img = img.apply_layers([0, 2])

    assert_equals(new_img[10, 10], [255, 0, 0])
    assert_equals(new_img[30, 30], [0, 0, 0])
    assert_equals(new_img[50, 50], [0, 0, 255])

def test_get_drawing_layer():
    img = Image((100, 100))
    assert_equals(len(img._layers), 0)
    img.get_drawing_layer()
    assert_equals(len(img._layers), 1)
    dl1 = DrawingLayer((img.width, img.height))
    dl2 = DrawingLayer((img.width, img.height))
    dl3 = DrawingLayer((img.width, img.height))
    img.add_drawing_layer(dl1)
    img.add_drawing_layer(dl2)
    img.add_drawing_layer(dl3)
    
    assert_equals(img.get_drawing_layer(2), dl2)
    assert_equals(img.get_drawing_layer(), dl3)

def test_clear_layers():
    img = Image((100, 100))
    dl1 = DrawingLayer((img.width, img.height))
    dl2 = DrawingLayer((img.width*2, img.height))
    dl3 = DrawingLayer((img.width, img.height*2))
    img.add_drawing_layer(dl1)
    img.add_drawing_layer(dl2)
    img.add_drawing_layer(dl3)

    assert_equals(len(img._layers), 3)

    img.clear_layers()
    assert_equals(len(img._layers), 0)

def test_layers():
    img = Image((100, 100))
    dl1 = DrawingLayer((img.width, img.height))
    dl2 = DrawingLayer((img.width*2, img.height))
    dl3 = DrawingLayer((img.width, img.height*2))
    img.add_drawing_layer(dl1)
    img.add_drawing_layer(dl2)
    img.add_drawing_layer(dl3)

    assert_equals(len(img._layers), 3)
    assert_equals(img.layers(), [dl1, dl2, dl3])
    

def test_features_on_edge():
    img1 = "./../data/sampleimages/EdgeTest1.png"
    img2 = "./../data/sampleimages/EdgeTest2.png"

    img_a = Image(img1)
    blobs = img_a.find_blobs()
    rim = blobs.on_image_edge()
    inside = blobs.not_on_image_edge()
    rim.draw(color=Color.RED)
    inside.draw(color=Color.BLUE)

    img_b = Image(img2)
    circs = img_b.find_circle(thresh=200)
    rim = circs.on_image_edge()
    inside = circs.not_on_image_edge()
    rim.draw(color=Color.RED)
    inside.draw(color=Color.BLUE)

    img_c = Image(img2).copy()
    corners = img_c.find_corners()
    rim = corners.on_image_edge()
    inside = corners.not_on_image_edge()
    rim.draw(color=Color.RED)
    inside.draw(color=Color.BLUE)

    img_d = Image(img2).copy()
    kp = img_d.find_keypoints()
    rim = kp.on_image_edge()
    inside = kp.not_on_image_edge()
    rim.draw(color=Color.RED)
    inside.draw(color=Color.BLUE)

    img_e = Image(img2).copy()
    lines = img_e.find_lines()
    rim = lines.on_image_edge()
    inside = lines.not_on_image_edge()
    rim.draw(color=Color.RED)
    inside.draw(color=Color.BLUE)

    results = [img_a, img_b, img_c, img_d, img_e]
    name_stem = "test_features_on_edge"
    perform_diff(results, name_stem)


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
        assert_is_instance(temp, Image)
        derp = temp.rotate(b.get_angle(), fixed=False)
        derp.draw_text(str(b.get_angle()), 10, 10, color=Color.RED)
        b.rectify_major_axis()
        assert_is_instance(b.blob_image(), Image)


def test_minrect_blobs():
    img = Image("../data/sampleimages/bolt.png")
    img = img.invert()
    results = []
    for i in range(-10, 10):
        ang = float(i * 18.00)
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

def test_find_keypoints_all():
    img = Image(testimage2)
    methods = ["ORB", "SIFT", "SURF", "FAST", "STAR", "MSER", "Dense"]
    for i in methods:
        kp = img.find_keypoints(flavor=i)
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
        assert os.path.exists(i)


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

    filename = os.path.join(tempfile.gettempdir(), "test_save_to_gif.gif")
    saved = imgs.save(filename)
    os.remove(filename)
    assert_equals(saved, len(imgs))


def test_sliceing_image_set():
    imgset = ImageSet("../data/sampleimages/")
    imgset = imgset[8::-2]
    assert isinstance(imgset, ImageSet)


def test_builtin_rotations():
    img = Image('lenna')
    r1 = img - img.rotate180().rotate180()
    r2 = img - img.rotate90().rotate90().rotate90().rotate90()
    r3 = img - img.rotate_left().rotate_left().rotate_left().rotate_left()
    r4 = img - img.rotate_right().rotate_right().rotate_right().rotate_right()
    r5 = img - img.rotate270().rotate270().rotate270().rotate270()
    assert_equals(Color.BLACK, r1.mean_color())
    assert_equals(Color.BLACK, r2.mean_color())
    assert_equals(Color.BLACK, r3.mean_color())
    assert_equals(Color.BLACK, r4.mean_color())
    assert_equals(Color.BLACK, r5.mean_color())


def test_blob_full_masks():
    img = Image('lenna')
    b = img.find_blobs()
    m1 = b[-1].get_full_masked_image()
    m2 = b[-1].get_full_hull_masked_image()
    m3 = b[-1].get_full_mask()
    m4 = b[-1].get_full_hull_mask()
    assert_equals(m1.width, img.width)
    assert_equals(m2.width, img.width)
    assert_equals(m3.width, img.width)
    assert_equals(m4.width, img.width)
    assert_equals(m1.height, img.height)
    assert_equals(m2.height, img.height)
    assert_equals(m3.height, img.height)
    assert_equals(m4.height, img.height)


def test_blob_edge_images():
    img = Image('lenna')
    b = img.find_blobs()
    m1 = b[-1].get_edge_image()
    assert_is_instance(m1, Image)
    assert_equals(m1.size, img.size)
    m2 = b[-1].get_hull_edge_image()
    assert_is_instance(m2, Image)
    assert_equals(m1.size, img.size)
    m3 = b[-1].get_full_edge_image()
    assert_is_instance(m3, Image)
    assert_equals(m3.size, img.size)
    m4 = b[-1].get_full_hull_edge_image()
    assert_is_instance(m4, Image)
    assert_equals(m4.size, img.size)

def test_uncrop():
    img = Image('lenna')
    cropped_img = img.crop(10, 20, 250, 500)
    source_pts = cropped_img.uncrop([(2, 3), (56, 23), (24, 87)])
    assert source_pts


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
    assert dlayer
    dlayer1 = grid_image.remove_grid()
    assert dlayer1 is None


def test_cluster():
    img = Image("lenna")
    blobs = img.find_blobs()
    clusters1 = blobs.cluster(method="kmeans", k=5, properties=["color"])
    assert clusters1
    clusters2 = blobs.cluster(method="hierarchical")
    assert clusters2

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

def test_running_average():
    img = Image('lenna')
    ls = img.get_line_scan(y=120)
    ra = ls.running_average(5)
    assert_equals(sum(ls[48:53]) / 5, ra[50])


@nottest
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
    assert line_scan_perform_diff(l1, l2, LineScan.smooth, degree=7)


def test_linescan_normalize():
    img = Image("lenna")
    l1 = img.get_line_scan(x=90)
    l2 = l1.normalize()
    assert line_scan_perform_diff(l1, l2, LineScan.normalize)


def test_linescan_scale():
    img = Image("lenna")
    l1 = img.get_line_scan(y=90)
    l2 = l1.scale()
    assert line_scan_perform_diff(l1, l2, LineScan.scale)


def test_linescan_derivative():
    img = Image("lenna")
    l1 = img.get_line_scan(y=140)
    l2 = l1.derivative()
    assert line_scan_perform_diff(l1, l2, LineScan.derivative)


def test_linescan_resample():
    img = Image("lenna")
    l1 = img.get_line_scan(pt1=(300, 300), pt2=(450, 500))
    l2 = l1.resample(n=50)
    assert line_scan_perform_diff(l1, l2, LineScan.resample, n=50)


def test_linescan_fit_to_model():
    def a_line(x, m, b):
        return x * m + b

    img = Image("lenna")
    l1 = img.get_line_scan(y=200)
    l2 = l1.fit_to_model(a_line)
    assert line_scan_perform_diff(l1, l2, LineScan.fit_to_model, f=a_line)


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
    assert line_scan_perform_diff(l1, l2, LineScan.threshold, threshold=200,
                                  invert=True)


def test_linescan_invert():
    img = Image("lenna")
    l1 = img.get_line_scan(y=200)
    l2 = l1.invert(max=40)
    assert line_scan_perform_diff(l1, l2, LineScan.invert, max=40)


def test_linescan_median():
    img = Image("lenna")
    l1 = img.get_line_scan(x=120)
    l2 = l1.median(sz=9)
    assert line_scan_perform_diff(l1, l2, LineScan.median, sz=9)


def test_linescan_median_filter():
    img = Image("lenna")
    l1 = img.get_line_scan(y=250)
    l2 = l1.median_filter(kernel_size=7)
    assert line_scan_perform_diff(l1, l2, LineScan.median_filter,
                                  kernel_size=7)


def test_linescan_detrend():
    img = Image("lenna")
    l1 = img.get_line_scan(y=90)
    l2 = l1.detrend()
    assert line_scan_perform_diff(l1, l2, LineScan.detrend)

def test_gray_peaks():
    i = Image('lenna')
    peaks = i.gray_peaks()
    assert peaks is not None


def test_find_peaks():
    img = Image('lenna')
    ls = img.get_line_scan(x=150)
    peaks = ls.find_peaks()
    assert peaks is not None


def test_line_scan_sub():
    img = Image('lenna')
    ls = img.get_line_scan(x=200)
    ls1 = ls - ls
    assert_equals(0, ls1[23])


def test_line_scan_add():
    img = Image('lenna')
    ls = img.get_line_scan(x=20)
    l = ls + ls
    a = int(ls[20]) + int(ls[20])
    assert_equals(a, l[20])


def test_line_scan_mul():
    img = Image('lenna')
    ls = img.get_line_scan(x=20)
    l = ls * ls
    a = int(ls[20]) * int(ls[20])
    assert_equals(a, l[20])


def test_line_scan_div():
    img = Image('lenna')
    ls = img.get_line_scan(x=20)
    l = ls / ls
    a = int(ls[20]) / int(ls[20])
    assert_equals(a, l[20])


def test_face_recognize():
    if not hasattr(cv2, "createFisherFaceRecognizer"):
        return

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

    assert_list_equal(["male", "male", "female", "female"], label)

def test_prewitt():
    i = Image('lenna')
    p = i.prewitt()
    assert i != p


def test_grayscalmatrix():
    img = Image("lenna")
    graymat = img.get_gray_ndarray()
    newimg = Image(graymat, color_space=Image.GRAY)
    assert np.array_equal(img.get_gray_ndarray(), newimg.get_gray_ndarray())


def test_get_normalized_hue_histogram():
    img = Image('lenna')
    a = img.get_normalized_hue_histogram((0, 0, 100, 100))
    b = img.get_normalized_hue_histogram()
    blobs = img.find_blobs()
    c = img.get_normalized_hue_histogram(blobs[-1])
    assert_tuple_equal((180, 256), a.shape)
    assert_tuple_equal((180, 256), b.shape)
    assert_tuple_equal((180, 256), c.shape)


def test_find_blobs_from_hue_histogram():
    img = Image('lenna')
    img2 = Image('lyle')
    h = img2.get_normalized_hue_histogram()

    blobs = img.find_blobs_from_hue_histogram(h)
    assert_equals(75, len(blobs))
    blobs = img.find_blobs_from_hue_histogram((10, 10, 50, 50), smooth=False)
    assert_equals(44, len(blobs))
    blobs = img.find_blobs_from_hue_histogram(img2, threshold=1)
    assert_equals(75, len(blobs))


def test_drawing_layer_to_svg():
    img = Image('lenna')
    dl = img.dl()
    dl.line((0, 0), (100, 100))
    svg = dl.get_svg()
    result = '<svg baseProfile="full" height="512" version="1.1" width="512"'\
             ' xmlns="http://www.w3.org/2000/svg" ' \
             'xmlns:ev="http://www.w3.org/2001/xml-events" ' \
             'xmlns:xlink="http://www.w3.org/1999/xlink"><defs />' \
             '<line x1="0" x2="100" y1="0" y2="100" /></svg>'
    assert_equals(result, svg)
