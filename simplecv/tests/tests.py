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
    assert_is_instance, nottest, assert_is_none, assert_raises, assert_is_not_none

from simplecv.base import logger, ScvException
from simplecv.color import Color, ColorMap
from simplecv.core.drawing.layer import DrawingLayer
from simplecv.features.blob import Blob
from simplecv.features.detection import Line, Corner, Motion, KeyPoint, Circle, TemplateMatch
from simplecv.features.features import FeatureSet
from simplecv.image import Image
from simplecv.image_set import ImageSet
from simplecv.segmentation.color_segmentation import ColorSegmentation
from simplecv.segmentation.diff_segmentation import DiffSegmentation
from simplecv.segmentation.running_segmentation import RunningSegmentation

from simplecv.tests.utils import perform_diff, skipped

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
    lines = img_a.find(Line, 1)
    heights = lines.get_height()

    if len(heights) <= 0:
        assert False


def test_feature_get_width():
    img_a = Image(logo)
    lines = img_a.find(Line, 1)
    widths = lines.get_width()

    if len(widths) <= 0:
        assert False


def test_feature_crop():
    img_a = Image(logo)
    lines = img_a.find(Line)
    cropped_images = lines.crop()
    if len(cropped_images) <= 0:
        assert False


def test_blob_holes():
    img = Image("../data/sampleimages/blockhead.png")
    blobs = Blob.extract(img)
    blobs.draw()
    results = [img]
    name_stem = "test_blob_holes"
    perform_diff(results, name_stem)

    assert_equals(len(blobs), 7)

    assert_equals(len(blobs[0].contour.holes), 2)
    assert_equals(len(blobs[1].contour.holes), 1)
    assert_equals(len(blobs[2].contour.holes), 1)
    assert_equals(len(blobs[3].contour.holes), 1)
    assert_equals(len(blobs[4].contour.holes), 2)
    assert_equals(len(blobs[5].contour.holes), 0)
    assert_equals(len(blobs[6].contour.holes), 0)

    for b in blobs:
        assert len(b.convex_hull) > 3


def test_blob_render():
    img = Image("../data/sampleimages/blockhead.png")
    blobs = Blob.extract(img)
    dl = DrawingLayer()
    reimg = DrawingLayer()
    for b in blobs:
        b.draw(color=Color.RED, alpha=128, width=2)
        b.convex_hull.draw(color=Color.ORANGE, width=2)
        b.draw(color=Color.RED, alpha=128, width=2, layer=dl)
        b.convex_hull.draw(color=Color.ORANGE, width=2, layer=dl)
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
    files = os.listdir("../data/sampleimages/")
    files1 = []
    for f in files:
        if ".gif" not in f and ".mov" not in f:
            files1.append(os.path.abspath(
                          os.path.join("..", "data", "sampleimages", f)))

    imgset = ImageSet(files1)
    imgset1 = ImageSet("samples")
    imgset2 = ImageSet(os.path.abspath("../data/sampleimages"))

    imgset.sort(key=lambda x:x.filename)
    imgset1.sort(key=lambda x:x.filename)
    imgset2.sort(key=lambda x:x.filename)

    assert_equals(len(imgset), len(imgset1))
    assert_equals(len(imgset), len(imgset2))

    for i in range(len(imgset)):
        assert_equals(imgset[i].filename, imgset1[i].filename)
        assert_equals(imgset[i].filename, imgset2[i].filename)

"""
def test_imageset_download():
    imgset = ImageSet()
    imgset.download("simplecv", number=3, size="small")
    assert len(imgset) == 3

    imgset_thumb = ImageSet()
    imgset_thumb.download("simplcv", number=4, size="thumb")
    assert len(imgset) == 4
"""

def test_hsv_conversion():
    px = Image((1, 1))
    px[0, 0] = Color.GREEN
    assert_list_equal(Color.hsv(Color.GREEN), px.to_hsv()[0, 0].tolist())


def test_draw_rectangle():
    img = Image(testimage2)
    img.dl().rectangle((0, 0), (100, 100), color=Color.BLUE, width=0, alpha=0)
    img.dl().rectangle((1, 1), (100, 100), color=Color.BLUE, width=2, alpha=128)
    img.dl().rectangle((1, 1), (100, 100), color=Color.BLUE, width=1, alpha=128)
    img.dl().rectangle((2, 2), (100, 100), color=Color.BLUE, width=1, alpha=255)
    img.dl().rectangle((3, 3), (100, 100), color=Color.BLUE)
    img.dl().rectangle((4, 4), (100, 100), color=Color.BLUE, width=12)
    img.dl().rectangle((5, 5), (100, 100), color=Color.BLUE, filled=True)

    results = [img]
    name_stem = "test_draw_rectangle"
    perform_diff(results, name_stem)


def test_blob_min_rect():
    img = Image(testimageclr)
    blobs = img.find(Blob)
    for b in blobs:
        b.draw_min_rect(color=Color.BLUE, width=3, alpha=123)
    results = [img]
    name_stem = "test_blob_min_rect"
    perform_diff(results, name_stem)


def test_blob_rect():
    img = Image(testimageclr)
    blobs = img.find(Blob)
    for b in blobs:
        b.draw_rect(color=Color.BLUE, width=3, alpha=123)
    results = [img]
    name_stem = "test_blob_rect"
    perform_diff(results, name_stem)


def test_blob_pickle():
    img = Image(testimageclr)
    blobs = img.find(Blob)
    for b in blobs:
        p = pickle.dumps(b)
        ub = pickle.loads(p)


def test_blob_isa_methods():
    img1 = Image(circles)
    blobs = img1.find(Blob).sort_area()
    assert_true(blobs[3].is_circle(tolerance=0.1))
    assert_false(blobs[3].is_rectangle(tolerance=0.1))

    img2 = Image("../data/sampleimages/blockhead.png")
    blobs = img2.find(Blob).sort_area()
    assert_false(blobs[-1].is_circle())
    assert_true(blobs[-1].is_rectangle())


def test_keypoint_extraction():
    img1 = Image("../data/sampleimages/KeypointTemplate2.png")
    img2 = Image("../data/sampleimages/KeypointTemplate2.png")
    img3 = Image("../data/sampleimages/KeypointTemplate2.png")
    img4 = Image("../data/sampleimages/KeypointTemplate2.png")

    kp1 = img1.find(KeyPoint)
    assert_equals(190, len(kp1))
    kp1.draw()

    kp2 = img2.find(KeyPoint, highquality=True)
    assert_equals(190, len(kp2))
    kp2.draw()

    kp3 = img3.find(KeyPoint, flavor="STAR")
    assert_equals(37, len(kp3))
    kp3.draw()

    if not cv2.__version__.startswith("$Rev:"):
        kp4 = img4.find(KeyPoint, flavor="BRISK")
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
    blob_fs = img.find(Blob)
    line_fs = img.find(Line)
    corn_fs = img.find(Corner)
    move_fs = img.find(Motion, motion)
    move_fs = FeatureSet(move_fs[42:52])  # l337 s5p33d h4ck - okay not really
    temp_fs = img.find(TemplateMatch, template, threshold=1)
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
    i = img.find(Line)
    i2 = i[0:10]
    assert_is_instance(i2, FeatureSet)


def test_blob_spatial_relationships():
    img = Image("../data/sampleimages/spatial_relationships.png")
    # please see the image
    blobs = img.find(Blob, threshold=1)
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
    b = img.find(Blob)
    l = img2.find(Line)
    c = img2.find(Circle, threshold=200)
    c2 = img2.find(Corner)
    kp = img2.find(KeyPoint)
    assert_greater(len(b.aspect_ratios()), 0)
    assert_greater(len(l.aspect_ratios()), 0)
    assert_greater(len(c.aspect_ratios()), 0)
    assert_greater(len(c2.aspect_ratios()), 0)
    assert_greater(len(kp.aspect_ratios()), 0)


def test_get_corners():
    img = Image("../data/sampleimages/EdgeTest1.png")
    img2 = Image("../data/sampleimages/EdgeTest2.png")
    b = img.find(Blob)
    assert_is_instance(b.top_left_corners(), np.ndarray)
    assert_is_instance(b.top_right_corners(), np.ndarray)
    assert_is_instance(b.bottom_left_corners(), np.ndarray)
    assert_is_instance(b.bottom_right_corners(), np.ndarray)

    l = img2.find(Line)
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
    assert_is_none(img.save(path="/home/unkown_user/desktop/lena.jpg",
                            temp=True))
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
    """
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
    """

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
    dl1 = DrawingLayer()
    dl2 = DrawingLayer()
    img.layers.insert(1, dl2)
    img.layers.insert(2, dl1)
    assert_equals(len(img.layers), 2)
    assert_equals(img.layers[1], dl1)
    assert_equals(img.layers[0], dl2)

def test_add_drawing_layer():
    img = Image("simplecv")
    dl1 = DrawingLayer()
    dl2 = DrawingLayer()
    img.add_drawing_layer(dl1)
    img.add_drawing_layer(dl2)
    assert_raises(ScvException, img.add_drawing_layer, 'Not a drawing layer')
    assert_equals(len(img.layers), 2)
    assert_equals(img.layers[0], dl1)
    assert_equals(img.layers[1], dl2)

def test_remove_drawing_layer():
    img = Image("simplecv")
    dl1 = DrawingLayer()
    dl2 = DrawingLayer()
    img.add_drawing_layer(dl1)
    img.add_drawing_layer(dl2)
    del img.layers[1]
    assert_equals(img.layers.pop(), dl1)
    assert_equals(len(img.layers), 0)


def test_apply_layers():
    img = Image((100, 100))
    assert_equals(img.data, img.apply_layers().data)

    dl1 = DrawingLayer()
    dl1.rectangle((10, 10), (10, 10), color=(255, 0, 0), filled=True)
    dl2 = DrawingLayer()
    dl2.rectangle((30, 30), (10, 10), color=(0, 255, 0), filled=True)
    dl3 = DrawingLayer()
    dl3.rectangle((50, 50), (10, 10), color=(0, 0, 255), filled=True)
    img.add_drawing_layer(dl1)
    img.add_drawing_layer(dl2)
    img.add_drawing_layer(dl3)
    new_img = img.apply_layers()

    assert_true(new_img.is_bgr())
    assert_equals(new_img[15, 15].tolist(), [0, 0, 255])
    assert_equals(new_img[35, 35].tolist(), [0, 255, 0])
    assert_equals(new_img[55, 55].tolist(), [255, 0, 0])

    img = Image((100, 100))
    img.add_drawing_layer(dl1)
    img.add_drawing_layer(dl2)
    img.add_drawing_layer(dl3)
    new_img = img.apply_layers([0, 2])

    assert_true(new_img.is_bgr())
    assert_equals(new_img[15, 15].tolist(), [0, 0, 255])
    assert_equals(new_img[35, 35].tolist(), [0, 0, 0])
    assert_equals(new_img[55, 55].tolist(), [255, 0, 0])


def test_get_drawing_layer():
    img = Image((100, 100))
    assert_equals(len(img.layers), 0)
    img.get_drawing_layer()
    assert_equals(len(img.layers), 1)
    dl1 = DrawingLayer()
    dl2 = DrawingLayer()
    dl3 = DrawingLayer()
    img.add_drawing_layer(dl1)
    img.add_drawing_layer(dl2)
    img.add_drawing_layer(dl3)

    assert_equals(img.get_drawing_layer(2), dl2)
    assert_equals(img.get_drawing_layer(), dl3)


def test_clear_layers():
    img = Image((100, 100))
    dl1 = DrawingLayer()
    dl2 = DrawingLayer()
    dl3 = DrawingLayer()
    img.add_drawing_layer(dl1)
    img.add_drawing_layer(dl2)
    img.add_drawing_layer(dl3)

    assert_equals(len(img.layers), 3)

    img.clear_layers()
    assert_equals(len(img.layers), 0)


def test_layers():
    img = Image((100, 100))
    dl1 = DrawingLayer()
    dl2 = DrawingLayer()
    dl3 = DrawingLayer()
    img.add_drawing_layer(dl1)
    img.add_drawing_layer(dl2)
    img.add_drawing_layer(dl3)

    assert_equals(len(img.layers), 3)
    assert_equals(img.layers, [dl1, dl2, dl3])


def test_features_on_edge():
    img1 = "./../data/sampleimages/EdgeTest1.png"
    img2 = "./../data/sampleimages/EdgeTest2.png"

    img_a = Image(img1)
    blobs = img_a.find(Blob)
    rim = blobs.on_image_edge()
    inside = blobs.not_on_image_edge()
    rim.draw(color=Color.RED)
    inside.draw(color=Color.BLUE)

    img_b = Image(img2)
    circs = img_b.find(Circle, threshold=200)
    rim = circs.on_image_edge()
    inside = circs.not_on_image_edge()
    rim.draw(color=Color.RED)
    inside.draw(color=Color.BLUE)

    img_c = Image(img2).copy()
    corners = img_c.find(Corner)
    rim = corners.on_image_edge()
    inside = corners.not_on_image_edge()
    rim.draw(color=Color.RED)
    inside.draw(color=Color.BLUE)

    img_d = Image(img2).copy()
    kp = img_d.find(KeyPoint)
    rim = kp.on_image_edge()
    inside = kp.not_on_image_edge()
    rim.draw(color=Color.RED)
    inside.draw(color=Color.BLUE)

    img_e = Image(img2).copy()
    lines = img_e.find(Line)
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
    b = img.find(Blob)
    l = img2.find(Line)
    k = img3.find(KeyPoint)

    for bs in b:
        tl = bs.top_left_corner
        img.dl().text(str(bs.angle), tl, color=Color.RED)

    for ls in l:
        tl = ls.top_left_corner
        img2.dl().text(str(ls.angle), tl, color=Color.GREEN)

    for ks in k:
        tl = ks.top_left_corner
        img3.dl().text(str(ks.angle), tl, color=Color.BLUE)

    results = [img, img2, img3]
    name_stem = "test_feature_angles"
    perform_diff(results, name_stem, tolerance=11.0)


def test_feature_angles_rotate():
    img = Image("../data/sampleimages/rotation2.png")
    blobs = img.find(Blob)
    assert_equals(13, len(blobs))

    for b in blobs:
        temp = b.crop()
        assert_is_instance(temp, Image)
        derp = temp.rotate(b.angle, fixed=False)
        derp.dl().text(str(b.angle), (10, 10), color=Color.RED)
        b.rectify_major_axis()
        assert_is_instance(b.image, Image)


def test_minrect_blobs():
    img = Image("../data/sampleimages/bolt.png")
    img = img.invert()
    results = []
    for i in range(-10, 10):
        ang = float(i * 18.00)
        t = img.rotate(ang)
        b = t.find(Blob, threshold=128)
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
        e.dl().line(a, b, color=Color.RED)
        e.dl().circle(pts[0], 10, color=Color.GREEN)

    for x in range(25, 225, 25):
        a = (25, x)
        b = (125, 125)
        pts = img.edge_intersections(a, b, width=1)
        e.dl().line(a, b, color=Color.RED)
        e.dl().circle(pts[0], 10, color=Color.GREEN)

    for x in range(25, 225, 25):
        a = (x, 225)
        b = (125, 125)
        pts = img.edge_intersections(a, b, width=1)
        e.dl().line(a, b, color=Color.RED)
        e.dl().circle(pts[0], 10, color=Color.GREEN)

    for x in range(25, 225, 25):
        a = (225, x)
        b = (125, 125)
        pts = img.edge_intersections(a, b, width=1)
        e.dl().line(a, b, color=Color.RED)
        e.dl().circle(pts[0], 10, color=Color.GREEN)

    results = [e]
    name_stem = "test_point_intersection"
    perform_diff(results, name_stem, tolerance=6.0)


def test_find_keypoints_all():
    img = Image(testimage2)
    methods = ["ORB", "SIFT", "SURF", "FAST", "STAR", "MSER", "Dense"]
    for i in methods:
        kp = img.find(KeyPoint, flavor=i)
        if kp is not None:
            for k in kp:
                assert_is_not_none(k.object)
                assert_is_not_none(k.quality)
                assert_is_not_none(k.octave)
                assert_is_not_none(k.flavor)
                assert_is_not_none(k.angle)
                assert_is_not_none(k.coordinates)
                k.draw()
                assert_is_not_none(k.distance_from())
                assert_is_not_none(k.mean_color)
                assert_is_not_none(k.area)
                assert_is_not_none(k.perimeter)
                assert_is_not_none(k.width)
                assert_is_not_none(k.height)
                assert_is_not_none(k.radius)
                assert_is_not_none(k.crop())
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


def test_uncrop():
    img = Image('lenna')
    cropped_img = img.crop(10, 20, 250, 500)
    source_pts = cropped_img.uncrop([(2, 3), (56, 23), (24, 87)])
    assert source_pts


def test_grid():
    img = Image("simplecv")
    img1 = img.copy()
    img1.dl().grid(img1.size_tuple, (10, 10), (0, 255, 0), 1)
    img2 = img.copy()
    img2.dl().grid(img2.size_tuple, (20, 20), (255, 0, 255), 1)
    img3 = img.copy()
    img3.dl().grid(img3.size_tuple, (20, 20), (255, 0, 255), 2)
    result = [img1, img2, img3]
    name_stem = "test_image_grid"
    perform_diff(result, name_stem)


def test_cluster():
    img = Image("lenna")
    blobs = img.find(Blob)
    clusters1 = blobs.cluster(method="kmeans", k=5, properties=["color"])
    assert clusters1
    clusters2 = blobs.cluster(method="hierarchical")
    assert clusters2


def test_color_map():
    img = Image('../data/sampleimages/mtest.png')
    blobs = img.find(Blob)
    cm = ColorMap((Color.RED, Color.YELLOW, Color.BLUE), min(blobs.get_area()),
                  max(blobs.get_area()))
    for b in blobs:
        b.draw(cm[b.area])
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


def test_prewitt():
    i = Image('lenna')
    p = i.prewitt()
    assert i != p


def test_get_normalized_hue_histogram():
    img = Image('lenna')
    a = img.get_normalized_hue_histogram((0, 0, 100, 100))
    b = img.get_normalized_hue_histogram()
    blobs = img.find(Blob)
    c = img.get_normalized_hue_histogram(blobs[-1])
    assert_tuple_equal((180, 256), a.shape)
    assert_tuple_equal((180, 256), b.shape)
    assert_tuple_equal((180, 256), c.shape)


def test_find_blobs_from_hue_histogram():
    img = Image('lenna')
    img2 = Image('lyle')
    h = img2.get_normalized_hue_histogram()

    blobs = Blob.find_from_hue_histogram(img, h)
    assert_equals(75, len(blobs))
    blobs = Blob.find_from_hue_histogram(img, (10, 10, 50, 50), smooth=False)
    assert_equals(44, len(blobs))
    blobs = Blob.find_from_hue_histogram(img, img2, threshold=1)
    assert_equals(75, len(blobs))


def test_drawing_layer_to_svg():
    img = Image((10, 10))
    dl = img.dl()
    dl.line((0, 0), (100, 100))
    dl.lines(((100, 0), (100, 200), (200, 200)))
    dl.rectangle((0, 30), (20, 20))
    dl.rectangle_to_pts((70, 250), (80, 260))
    dl.centered_rectangle((100, 400), (40, 40))
    dl.polygon(((300, 50), (315, 50), (330, 40), (320, 45)))
    dl.circle((100, 200), 75)
    dl.ellipse((300, 300), (30, 60))
    dl.set_font_bold(True)
    dl.text('hello svg', pos=(50, 300))
    svg = img.apply_layers(renderer='svg')
    result = '<svg baseProfile="full" height="10" version="1.1" ' \
             'width="10" xmlns="http://www.w3.org/2000/svg" ' \
             'xmlns:ev="http://www.w3.org/2001/xml-events" ' \
             'xmlns:xlink="http://www.w3.org/1999/xlink"><defs />' \
             '<image height="10" width="10" x="0" ' \
             'xlink:href="data:image/png;base64,iVBORw0KGgoAAAANSUhE' \
             'UgAAAAoAAAAKCAYAAACNMs+9AAAAKklEQVQYGY3BAQEAAAABoPwfzQ' \
             'UV&#10;1CGoQ1CHoA5BHYI6BHUI6hDUIajDAKk6CgHUYcToAAAAAEl' \
             'FTkSuQmCC&#10;" y="0" />' \
             '<line stroke="rgb(0,0,0)" stroke-opacity="1.0" ' \
             'stroke-width="1" x1="0" x2="100" y1="0" y2="100" />' \
             '<line stroke="rgb(0,0,0)" stroke-opacity="1.0" ' \
             'stroke-width="1" x1="100" x2="100" y1="0" y2="200" />' \
             '<line stroke="rgb(0,0,0)" stroke-opacity="1.0" ' \
             'stroke-width="1" x1="100" x2="200" y1="200" y2="200" />' \
             '<rect fill-opacity="0" height="20" stroke="rgb(0,0,0)" ' \
             'stroke-opacity="1.0" stroke-width="1" width="20" x="0" y="30" />' \
             '<rect fill-opacity="0" height="10" stroke="rgb(0,0,0)" ' \
             'stroke-opacity="1.0" stroke-width="1" width="10" x="70" y="250" />' \
             '<rect fill-opacity="0" height="40" stroke="rgb(0,0,0)" ' \
             'stroke-opacity="1.0" stroke-width="1" width="40" x="80" y="380" />' \
             '<path d="M 300 50 L 315 50 L 330 40 L 320 45 Z" fill-opacity="0" ' \
             'fill-rule="evenodd" stroke="rgb(0,0,0)" stroke-opacity="1.0" stroke-width="1" />' \
             '<circle cx="100" cy="200" fill-opacity="0" r="75" stroke="rgb(0,0,0)" ' \
             'stroke-opacity="1.0" stroke-width="1" />' \
             '<ellipse cx="300" cy="300" fill-opacity="0" rx="15" ry="30" ' \
             'stroke="rgb(0,0,0)" stroke-opacity="1.0" stroke-width="1" />' \
             '<text fill="rgb(0,0,0)" fill-opacity="1.0" ' \
             'style="font-size: 11px;font-weight: bold;" ' \
             'x="50" y="300">hello svg</text>' \
             '</svg>'
    assert_equals(result, svg)


def test_draw():
    simg = Image("simplecv")
    img = Image((250, 250))
    img1 = Image((250, 250))

    lines = simg.find(Line)
    img.draw_features(lines, width=3)

    for line in lines:
        img1.draw_features(line, width=3)

    assert_equals(img.apply_layers().data,
                  img1.apply_layers().data)

    # incorrect params
    assert_is_none(img.draw_features(simg))
    assert_is_none(img.draw_features((100, 100)))


def test_draw_sift_key_point_match():
    template = Image("../data/sampleimages/KeypointTemplate2.png")
    match0 = Image("../data/sampleimages/kptest0.png")

    img = match0.draw_sift_key_point_match(template, distance=100, num=15)
    img = img.apply_layers()
    name_stem = "test_draw_key_point_matches"
    perform_diff([img], name_stem, 5.0)

    img = match0.draw_sift_key_point_match(template, distance=100)
    assert_is_none(match0.draw_sift_key_point_match(None))
