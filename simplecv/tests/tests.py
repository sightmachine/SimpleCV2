# /usr/bin/python
# To run this test you need python nose tools installed
# Run test just use:
#   nosetest tests.py
#
# *Note: If you add additional test, please prefix the function name
# to the type of operation being performed.  For instance modifying an
# image, test_image_erode().  If you are looking for lines, then
# test_detection_lines().  This makes it easier to verify visually
# that all the correct test per operation exist

from math import sqrt
import os
import pickle
import tempfile

from cv2 import cv
from nose.tools import nottest
import cv2
import numpy as np

from simplecv.base import logger, nparray_to_cvmat
from simplecv.camera import FrameSource
from simplecv.color import Color, ColorCurve, ColorMap
from simplecv.color_model import ColorModel
from simplecv.dft import DFT
from simplecv.drawing_layer import DrawingLayer
from simplecv.features.blobmaker import BlobMaker
from simplecv.features.detection import Corner, Line, ROI
from simplecv.features.facerecognizer import FaceRecognizer
from simplecv.features.features import FeatureSet
from simplecv.features.haar_cascade import HaarCascade
from simplecv.image_class import Image, ImageSet, ColorSpace
from simplecv.linescan import LineScan
from simplecv.segmentation.color_segmentation import ColorSegmentation
from simplecv.segmentation.diff_segmentation import DiffSegmentation
from simplecv.segmentation.running_segmentation import RunningSegmentation

VISUAL_TEST = False  # if TRUE we save the images - otherwise we DIFF against
                     # them - the default is False
SHOW_WARNING_TESTS = False  # show that warnings are working - tests will pass
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

#standards path
standard_path = "../data/test/standard/"


#Given a set of images, a path, and a tolerance do the image diff.
@nottest
def img_diffs(test_imgs, name_stem, tolerance, path):
    count = len(test_imgs)
    ret_val = False
    for idx in range(0, count):
        lhs = test_imgs[idx].apply_layers()  # this catches drawing methods
        if lhs.is_gray():
            lhs = lhs.to_bgr()
        fname = standard_path + name_stem + str(idx) + ".jpg"
        rhs = Image(fname)
        if lhs.size() == rhs.size():
            num_img_pixels = lhs.width * lhs.height * 3
            diff = cv2.absdiff(lhs.get_ndarray(), rhs.get_ndarray())
            diff_pixels = (diff > 0).astype(np.uint8)
            diff_pixels_sum = diff_pixels.sum()
            if diff_pixels_sum > 0:
                percent_diff_pixels = diff_pixels_sum / num_img_pixels
                print "{0:.2f}% difference".format(percent_diff_pixels * 100)
                lhs = Image((diff_pixels * (0, 0, 255)).astype(np.uint8))
                lhs.save(fname[:-4] + "_DIFF.png")
                rhs.save(fname[:-4] + "_RESULT.png")
                ret_val = True
    return ret_val


#Save a list of images to a standard path.
@nottest
def img_saves(test_imgs, name_stem, path=standard_path):
    count = len(test_imgs)
    for idx in range(0, count):
        fname = standard_path + name_stem + str(idx) + ".png"
        test_imgs[idx].save(fname)


#perform the actual image save and image diffs.
@nottest
def perform_diff(result, name_stem, tolerance=0.03, path=standard_path):
    if VISUAL_TEST:  # save the correct images for a visual test
        img_saves(result, name_stem, path)
    else:  # otherwise we test our output against the visual test
        assert not img_diffs(result, name_stem, tolerance, path)


def test_image_stretch():
    img = Image(greyscaleimage)
    stretched = img.stretch(100, 200)
    if stretched is None:
        assert False

    result = [stretched]
    name_stem = "test_stretch"
    perform_diff(result, name_stem)


def test_image_loadsave():
    img = Image(testimage)
    img.save(testoutput)
    if os.path.isfile(testoutput):
        os.remove(testoutput)
        pass
    else:
        assert False


def test_image_numpy_constructor():
    img = Image(testimage)
    grayimg = img.to_gray()

    chan3_array = np.array(img.get_ndarray())
    chan1_array = np.array(img.get_gray_ndarray())

    img2 = Image(chan3_array)
    grayimg2 = Image(chan1_array)

    if (img2[0, 0] == img[0, 0]).all() \
            and (grayimg2[0, 0] == grayimg[0, 0]).all():
        pass
    else:
        assert False


def test_image_bitmap():
    img1 = Image("lenna")
    img2 = Image("lenna")
    img2 = img2.smooth()
    result = [img1, img2]
    name_stem = "test_image_bitmap"
    perform_diff(result, name_stem)


# # Image Class Test

def test_image_scale():
    img = Image(testimage)
    thumb = img.scale(30, 30)
    if thumb is None:
        assert False
    result = [thumb]
    name_stem = "test_image_scale"
    perform_diff(result, name_stem)


def test_image_copy():
    img = Image(testimage2)
    copy = img.copy()

    assert (img[1, 1] == copy[1, 1]).all()
    assert img.size() == copy.size()

    result = [copy]
    name_stem = "test_image_copy"
    perform_diff(result, name_stem)


def test_image_getitem():
    img = Image(testimage)
    colors = img[1, 1]
    if colors[0] == 255 and colors[1] == 255 and colors[2] == 255:
        pass
    else:
        assert False


def test_image_getslice():
    img = Image(testimage)
    section = img[1:10, 1:10]
    if section is None:
        assert False


def test_image_setitem():
    img = Image(testimage)
    img[1, 1] = (0, 0, 0)
    newimg = Image(img.get_ndarray())
    colors = newimg[1, 1]
    if colors[0] == 0 and colors[1] == 0 and colors[2] == 0:
        pass
    else:
        assert False

    result = [newimg]
    name_stem = "test_image_setitem"
    perform_diff(result, name_stem)


def test_image_setslice():
    img = Image(testimage)
    img[1:10, 1:10] = (0, 0, 0)  # make a black box
    newimg = Image(img.get_ndarray())
    section = newimg[1:10, 1:10]
    for i in range(5):
        colors = section[i, 0]
        if colors[0] != 0 or colors[1] != 0 or colors[2] != 0:
            assert False
    result = [newimg]
    name_stem = "test_image_setslice"
    perform_diff(result, name_stem)


def test_detection_find_corners():
    img = Image(testimage2)
    corners = img.find_corners(25)
    corners.draw()
    if len(corners) == 0:
        assert False
    result = [img]
    name_stem = "test_detection_find_corners"
    perform_diff(result, name_stem)


def test_color_meancolor():
    a = np.arange(0, 256)
    b = a[::-1]
    c = np.copy(a) / 2
    a = a.reshape(16, 16)
    b = b.reshape(16, 16)
    c = c.reshape(16, 16)
    imgarr = np.dstack((a, b, c)).astype(np.uint8)
    img = Image(imgarr, color_space=ColorSpace.RGB)

    b, g, r = img.mean_color('BGR')
    print b, g, r
    if not (127 < r < 128 and 127 < g < 128 and 63 < b < 64):
        assert False

    r, g, b = img.mean_color('RGB')
    if not (127 < r < 128 and 127 < g < 128 and 63 < b < 64):
        assert False

    h, s, v = img.mean_color('HSV')
    if not (83 < h < 84 and 191 < s < 192 and 191 < v < 192):
        assert False

    x, y, z = img.mean_color('XYZ')
    if not (109 < x < 110 and 122 < y < 123 and 77 < z < 79):
        assert False

    gray = img.mean_color('Gray')
    if not (120 < gray < 121):
        assert False

    y, cr, cb = img.mean_color('YCrCb')
    if not (120 < y < 121 and 133 < cr < 134 and 96 < cb < 97):
        assert False

    h, l, s = img.mean_color('HLS')
    if not (84 < h < 85 and 117 < l < 118 and 160 < s < 161):
        assert False


def test_image_smooth():
    img = Image(testimage2)
    result = []
    result.append(img.smooth())
    result.append(img.smooth('bilateral', (3, 3), 4, 1))
    result.append(img.smooth('blur', (3, 3)))
    result.append(img.smooth('median', (3, 3)))
    result.append(img.smooth('gaussian', (5, 5), 0))
    result.append(img.smooth('bilateral', (3, 3), 4, 1, grayscale=False))
    result.append(img.smooth('blur', (3, 3), grayscale=True))
    result.append(img.smooth('median', (3, 3), grayscale=True))
    result.append(img.smooth('gaussian', (5, 5), 0, grayscale=True))
    name_stem = "test_image_smooth"
    perform_diff(result, name_stem)


def test_image_gamma_correct():
    img = Image(topImg)
    img2 = img.gamma_correct(1)
    img3 = img.gamma_correct(0.5)
    img4 = img.gamma_correct(2)
    result = []
    result.append(img3)
    result.append(img4)
    name_stem = "test_image_gammaCorrect"
    perform_diff(result, name_stem)
    if img3.mean_color() >= img2.mean_color() \
            and img4.mean_color() <= img2.mean_color():
        pass
    else:
        assert False


def test_image_binarize():
    img = Image(testimage2)
    binary = img.binarize()
    binary2 = img.binarize((60, 100, 200))
    hist = binary.histogram(20)
    hist2 = binary2.histogram(20)

    result = [binary, binary2]
    name_stem = "test_image_binarize"
    perform_diff(result, name_stem)

    if (hist[0] + hist[-1] == np.sum(hist) and hist2[0] + hist2[-1] == np.sum(
            hist2)):
        pass
    else:
        assert False


def test_image_binarize_adaptive():
    img = Image(testimage2)
    binary = img.binarize(-1)
    hist = binary.histogram(20)

    result = [binary]
    name_stem = "test_image_binarize_adaptive"
    perform_diff(result, name_stem)

    if hist[0] + hist[-1] == np.sum(hist):
        pass
    else:
        assert False


def test_image_invert():
    img = Image(testimage2)
    clr = img[1, 1]
    img = img.invert()

    result = [img]
    name_stem = "test_image_invert"
    perform_diff(result, name_stem)

    if clr[0] == (255 - img[1, 1][0]):
        pass
    else:
        assert False


def test_image_size():
    img = Image(testimage2)
    (width, height) = img.size()
    if type(width) == int and type(height) == int and width > 0 and height > 0:
        pass
    else:
        assert False


def test_image_drawing():
    img = Image(testimageclr)
    img.draw_circle((img.width / 2, img.height / 2), 10, thickness=3)
    img.draw_circle((img.width / 2, img.height / 2), 15, thickness=5,
                    color=Color.RED)
    img.draw_circle((img.width / 2, img.height / 2), 20)
    img.draw_line((5, 5), (5, 8))
    img.draw_line((5, 5), (10, 10), thickness=3)
    img.draw_line((0, 0), (img.width, img.height), thickness=3,
                  color=Color.BLUE)
    img.draw_rectangle(20, 20, 10, 5)
    img.draw_rectangle(22, 22, 10, 5, alpha=128)
    img.draw_rectangle(24, 24, 10, 15, width=-1, alpha=128)
    img.draw_rectangle(28, 28, 10, 15, width=3, alpha=128)
    result = [img]
    name_stem = "test_image_drawing"
    perform_diff(result, name_stem)


def test_image_draw():
    img = Image("lenna")
    newimg = Image("simplecv")
    lines = img.find_lines()
    newimg.draw(lines)
    lines.draw()
    result = [newimg, img]
    name_stem = "test_image_draw"
    perform_diff(result, name_stem, 5)


def test_image_splitchannels():
    img = Image(testimageclr)
    (r, g, b) = img.split_channels(True)
    (red, green, blue) = img.split_channels()
    result = [r, g, b, red, green, blue]
    name_stem = "test_image_splitchannels"
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
    blobs[-1].draw(color=Color.RED)
    blobs[-1].draw_appx(color=Color.BLUE)
    result = [img]

    img2 = Image("lenna")
    blobs = img2.find_blobs(appx_level=11)
    blobs[-1].draw(color=Color.RED)
    blobs[-1].draw_appx(color=Color.BLUE)
    result.append(img2)

    name_stem = "test_detection_blobs_appx"
    perform_diff(result, name_stem, 5.00)
    if blobs is None:
        assert False


def test_detection_blobs():
    img = Image(testbarcode)
    blobs = img.find_blobs()
    blobs.draw(color=Color.RED)
    result = [img]
    #TODO - WE NEED BETTER COVERAGE HERE
    name_stem = "test_detection_blobs"
    perform_diff(result, name_stem, 5.00)

    if blobs is None:
        assert False


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
    perform_diff(result, name_stem, 6.00)


def test_detection_blobs_adaptive():
    img = Image(testimage)
    blobs = img.find_blobs(-1, threshblocksize=99)
    blobs.draw(color=Color.RED)
    result = [img]
    name_stem = "test_detection_blobs_adaptive"
    perform_diff(result, name_stem, 5.00)

    if blobs is None:
        assert False


def test_detection_blobs_smallimages():
    # Check if segfault occurs or not
    img = Image("../data/sampleimages/blobsegfaultimage.png")
    blobs = img.find_blobs()
    # if no segfault, pass


def test_detection_blobs_convexity_defects():
    img = Image('lenna')
    blobs = img.find_blobs()
    b = blobs[-1]
    feat = b.get_convexity_defects()
    points = b.get_convexity_defects(return_points=True)
    if len(feat) <= 0 or len(points) <= 0:
        assert False
    pass


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

    if tmp_x > 0 and Image(testimage).size()[0]:
        pass
    else:
        assert False


def test_detection_y():
    tmp_y = Image(testimage).find_lines().y()[0]

    if tmp_y > 0 and Image(testimage).size()[0]:
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


def test_image():
    img = Image(testimage)
    if isinstance(img, Image):
        pass
    else:
        assert False


def test_color_colordistance():
    img = Image(blackimage)
    (r, g, b) = img.split_channels()
    avg = img.mean_color()

    c1 = Corner(img, 1, 1)
    c2 = Corner(img, 1, 2)
    if c1.color_distance(c2.mean_color()) != 0:
        assert False

    if c1.color_distance((0, 0, 0)) != 0:
        assert False

    if c1.color_distance((0, 0, 255)) != 255:
        assert False

    if c1.color_distance((255, 255, 255)) != sqrt(255 ** 2 * 3):
        assert False


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


#def test_distance_from():
#def test_sortColorDistance():
#def test_sortDistance():

def test_image_add():
    img_a = Image(blackimage)
    img_b = Image(whiteimage)
    img_c = img_a + img_b
    # FIXME: add assertion


def test_color_curve_hsl():
    # These are the weights
    y = np.array([[0, 0], [64, 128], [192, 128], [255, 255]])
    curve = ColorCurve(y)
    img = Image(testimage)
    img2 = img.apply_hls_curve(curve, curve, curve)
    img3 = img - img2

    result = [img2, img3]
    name_stem = "test_color_curve_hsl"
    perform_diff(result, name_stem)

    c = img3.mean_color()
    # there may be a bit of roundoff error
    if c[0] > 2.0 or c[1] > 2.0 or c[2] > 2.0:
        assert False


def test_color_curve_rgb():
    # These are the weights
    y = np.array([[0, 0], [64, 128], [192, 128], [255, 255]])
    curve = ColorCurve(y)
    img = Image(testimage)
    img2 = img.apply_rgb_curve(curve, curve, curve)
    img3 = img - img2

    result = [img2, img3]
    name_stem = "test_color_curve_rgb"
    perform_diff(result, name_stem)

    c = img3.mean_color()
    # there may be a bit of roundoff error
    if c[0] > 1.0 or c[1] > 1.0 or c[2] > 1.0:
        assert False


def test_color_curve_gray():
    # These are the weights
    y = np.array([[0, 0], [64, 128], [192, 128], [255, 255]])
    curve = ColorCurve(y)
    img = Image(testimage)
    gray = img.grayscale()
    img2 = img.apply_intensity_curve(curve)

    result = [img2]
    name_stem = "test_color_curve_gray"
    perform_diff(result, name_stem)

    g = gray.mean_color()
    i2 = img2.mean_color()
    if g[0] - i2[0] > 1:  # there may be a bit of roundoff error
        assert False


def test_image_dilate():
    img = Image(barcode)
    img2 = img.dilate(20)

    result = [img2]
    name_stem = "test_image_dilate"
    perform_diff(result, name_stem)
    c = img2.mean_color()

    if c[0] < 254 or c[1] < 254 or c[2] < 254:
        assert False


def test_image_erode():
    img = Image(barcode)
    img2 = img.erode(100)

    result = [img2]
    name_stem = "test_image_erode"
    perform_diff(result, name_stem)

    c = img2.mean_color()
    print(c)
    if c[0] > 0 or c[1] > 0 or c[2] > 0:
        assert False


def test_image_morph_open():
    img = Image(barcode)
    erode = img.erode()
    dilate = erode.dilate()
    result = img.morph_open()
    test = result - dilate
    c = test.mean_color()
    results = [result]
    name_stem = "test_image_morph_open"
    perform_diff(results, name_stem)

    if c[0] > 1 or c[1] > 1 or c[2] > 1:
        assert False


def test_image_morph_close():
    img = Image(barcode)
    dilate = img.dilate()
    erode = dilate.erode()
    result = img.morph_close()
    test = result - erode
    c = test.mean_color()

    results = [result]
    name_stem = "test_image_morph_close"
    perform_diff(results, name_stem)

    if c[0] > 1 or c[1] > 1 or c[2] > 1:
        assert False


def test_image_morph_grad():
    img = Image(barcode)
    dilate = img.dilate()
    erode = img.erode()
    dif = dilate - erode
    result = img.morph_gradient()
    test = result - dif
    c = test.mean_color()

    results = [result]
    name_stem = "test_image_morph_grad"
    perform_diff(results, name_stem)

    if c[0] > 1 or c[1] > 1 or c[2] > 1:
        assert False


def test_image_rotate_fixed():
    img = Image(testimage2)
    img2 = img.rotate(180, scale=1)
    img3 = img.flip_vertical()
    img4 = img3.flip_horizontal()
    img5 = img.rotate(70)
    img6 = img.rotate(70, scale=0.5)

    results = [img2, img3, img4, img5, img6]
    name_stem = "test_image_rotate_fixed"
    perform_diff(results, name_stem)

    test = img4 - img2
    c = test.mean_color()
    print(c)
    if c[0] > 5 or c[1] > 5 or c[2] > 5:
        assert False


def test_image_rotate_full():
    img = Image(testimage2)
    img2 = img.rotate(180, False, scale=1)

    results = [img2]
    name_stem = "test_image_rotate_full"
    perform_diff(results, name_stem)

    c1 = img.mean_color()
    c2 = img2.mean_color()
    if abs(c1[0] - c2[0]) > 5 \
            or abs(c1[1] - c2[1]) > 5 \
            or abs(c1[2] - c2[2]) > 5:
        assert False


def test_image_shear_warp():
    img = Image(testimage2)
    dst = ((img.width / 2, 0), (img.width - 1, img.height / 2),
           (img.width / 2, img.height - 1))
    s = img.shear(dst)

    color = s[0, 0]
    assert (color == (0, 0, 0)).all()

    dst = ((img.width * 0.05, img.height * 0.03),
           (img.width * 0.9, img.height * 0.1),
           (img.width * 0.8, img.height * 0.7),
           (img.width * 0.2, img.height * 0.9))
    w = img.warp(dst)

    results = [s, w]
    name_stem = "test_image_shear_warp"
    perform_diff(results, name_stem)

    color = s[0, 0]
    assert (color == (0, 0, 0)).all()


def test_image_affine():
    img = Image(testimage2)
    src = ((0, 0), (img.width - 1, 0), (img.width - 1, img.height - 1))
    dst = ((img.width / 2, 0), (img.width - 1, img.height / 2),
           (img.width / 2, img.height - 1))
    a_warp = cv2.getAffineTransform(np.array(src).astype(np.float32),
                                    np.array(dst).astype(np.float32))
    atrans = img.transform_affine(a_warp)

    a_warp2 = np.array(a_warp)
    atrans2 = img.transform_affine(a_warp2)

    test = atrans - atrans2
    c = test.mean_color()

    results = [atrans, atrans2]

    name_stem = "test_image_affine"
    perform_diff(results, name_stem)

    if c[0] > 1 or c[1] > 1 or c[2] > 1:
        assert False


def test_image_perspective():
    img = Image(testimage2)
    src = ((0, 0), (img.width - 1, 0), (img.width - 1, img.height - 1),
           (0, img.height - 1))
    dst = ((img.width * 0.05, img.height * 0.03),
           (img.width * 0.9, img.height * 0.1),
           (img.width * 0.8, img.height * 0.7),
           (img.width * 0.2, img.height * 0.9))
    src = np.array(src).astype(np.float32)
    dst = np.array(dst).astype(np.float32)

    p_warp = cv2.getPerspectiveTransform(src, dst)
    ptrans = img.transform_perspective(p_warp)

    p_warp2 = np.array(p_warp)
    ptrans2 = img.transform_perspective(p_warp2)

    test = ptrans - ptrans2
    c = test.mean_color()

    results = [ptrans, ptrans2]
    name_stem = "test_image_perspective"
    perform_diff(results, name_stem)
    if c[0] > 1 or c[1] > 1 or c[2] > 1:
        assert False


def test_image_horz_scanline():
    img = Image(logo)
    sl = img.get_horz_scanline(10)
    assert sl.shape[0] == img.width
    assert sl.shape[1] == 3
    assert len(sl.shape) == 2


def test_image_vert_scanline():
    img = Image(logo)
    sl = img.get_vert_scanline(10)
    assert sl.shape[0] == img.height
    assert sl.shape[1] == 3
    assert len(sl.shape) == 2


def test_image_horz_scanline_gray():
    img = Image(logo)
    sl = img.get_horz_scanline_gray(10)
    assert sl.shape[0] == img.width
    assert len(sl.shape) == 1


def test_image_vert_scanline_gray():
    img = Image(logo)
    sl = img.get_vert_scanline_gray(10)
    assert sl.shape[0] == img.width
    assert len(sl.shape) == 1


def test_image_get_pixel():
    img = Image(logo)
    px = img.get_pixel(0, 0)
    print(px)
    if px[0] != 0 or px[1] != 0 or px[2] != 0:
        assert False


def test_image_get_gray_pixel():
    img = Image(logo)
    px = img.get_gray_pixel(0, 0)
    if px != 0:
        assert False


def test_camera_calibration():
    fake_camera = FrameSource()
    path = "../data/sampleimages/CalibImage"
    ext = ".png"
    imgs = []
    for i in range(0, 10):
        fname = path + str(i) + ext
        img = Image(fname)
        imgs.append(img)

    fake_camera.calibrate(imgs)
    #we're just going to check that the function doesn't puke
    mat = fake_camera.get_camera_matrix()
    if not isinstance(mat, cv.cvmat):
        assert False
    #we're also going to test load in save in the same pass
    matname = "TestCalibration"
    if False == fake_camera.save_calibration(matname):
        assert False
    if False == fake_camera.load_calibration(matname):
        assert False


def test_camera_undistort():
    fake_camera = FrameSource()
    fake_camera.load_calibration("../data/test/StereoVision/Default")
    img = Image("../data/sampleimages/CalibImage0.png")
    img2 = fake_camera.undistort(img)

    results = [img2]
    name_stem = "test_camera_undistort"
    perform_diff(results, name_stem, tolerance=12)

    if not img2:  # right now just wait for this to return
        assert False


def test_image_crop():
    img = Image(logo)
    x = 5
    y = 6
    w = 10
    h = 20
    crop = img.crop(x, y, w, h)
    crop2 = img[x:(x + w), y:(y + h)]
    crop6 = img.crop(0, 0, 10, 10)
    # if( SHOW_WARNING_TESTS ):
    #     crop7 = img.crop(0,0,-10,10)
    #     crop8 = img.crop(-50,-50,10,10)
    #     crop3 = img.crop(-3,-3,10,20)
    #     crop4 = img.crop(-10,10,20,20,centered=True)
    #     crop5 = img.crop(-10,-10,20,20)

    tests = []
    tests.append(img.crop((50, 50), (10, 10)))  # 0
    tests.append(img.crop([10, 10, 40, 40]))  # 1
    tests.append(img.crop((10, 10, 40, 40)))  # 2
    tests.append(img.crop([50, 50], [10, 10]))  # 3
    tests.append(img.crop([10, 10], [50, 50]))  # 4

    roi = np.array([10, 10, 40, 40])
    pts1 = np.array([[50, 50], [10, 10]])
    pts2 = np.array([[10, 10], [50, 50]])
    pt1 = np.array([10, 10])
    pt2 = np.array([50, 50])

    tests.append(img.crop(roi))  # 5
    tests.append(img.crop(pts1))  # 6
    tests.append(img.crop(pts2))  # 7
    tests.append(img.crop(pt1, pt2))  # 8
    tests.append(img.crop(pt2, pt1))  # 9

    xs = [10, 10, 10, 20, 20, 20, 30, 30, 40, 40, 40, 50, 50, 50]
    ys = [10, 20, 50, 20, 30, 40, 30, 10, 40, 50, 10, 50, 10, 42]
    lots = zip(xs, ys)

    tests.append(img.crop(xs, ys))  # 10
    tests.append(img.crop(lots))  # 11
    tests.append(img.crop(np.array(xs), np.array(ys)))  # 12
    tests.append(img.crop(np.array(lots)))  # 14

    i = 0
    failed = False
    for img in tests:
        if img is None or img.width != 40 and img.height != 40:
            print "FAILED CROP TEST " + str(i) + " " + str(img)
            failed = True
        i = i + 1

    if failed:
        assert False
    results = [crop, crop2, crop6]
    name_stem = "test_image_crop"
    perform_diff(results, name_stem)

    diff = crop - crop2
    c = diff.mean_color()
    if c[0] > 0 or c[1] > 0 or c[2] > 0:
        assert False


def test_image_region_select():
    img = Image(logo)
    x1 = 0
    y1 = 0
    x2 = img.width
    y2 = img.height
    crop = img.region_select(x1, y1, x2, y2)

    results = [crop]
    name_stem = "test_image_region_select"
    perform_diff(results, name_stem)

    diff = crop - img
    c = diff.mean_color()
    if c[0] > 0 or c[1] > 0 or c[2] > 0:
        assert False


def test_image_subtract():
    img_a = Image(logo)
    img_b = Image(logo_inverted)
    img_c = img_a - img_b
    results = [img_c]
    name_stem = "test_image_subtract"
    perform_diff(results, name_stem)


def test_image_negative():
    img_a = Image(logo)
    img_b = -img_a
    results = [img_b]
    name_stem = "test_image_negative"
    perform_diff(results, name_stem)


def test_image_divide():
    img_a = Image(logo)
    img_b = Image(logo_inverted)

    img_c = img_a / img_b

    results = [img_c]
    name_stem = "test_image_divide"
    perform_diff(results, name_stem)


def test_image_and():
    img_a = Image(barcode)
    img_b = img_a.invert()

    img_c = img_a & img_b  # should be all black

    results = [img_c]
    name_stem = "test_image_and"
    perform_diff(results, name_stem)


def test_image_or():
    img_a = Image(barcode)
    img_b = img_a.invert()

    img_c = img_a | img_b  # should be all white

    results = [img_c]
    name_stem = "test_image_or"
    perform_diff(results, name_stem)


def test_image_edgemap():
    img_a = Image(logo)
    img_b = img_a._get_edge_map()
    #results = [imgB]
    #name_stem = "test_image_edgemap"
    #perform_diff(results,name_stem)


def test_color_colormap_build():
    cm = ColorModel()
    #cm.add(Image(logo))
    cm.add((127, 127, 127))
    if cm.contains((127, 127, 127)):
        cm.remove((127, 127, 127))
    else:
        assert False

    cm.remove((0, 0, 0))
    cm.remove((255, 255, 255))
    cm.add((0, 0, 0))
    cm.add([(0, 0, 0), (255, 255, 255)])
    cm.add([(255, 0, 0), (0, 255, 0)])
    img = cm.threshold(Image(testimage))
    c = img.mean_color()

    #if( c[0] > 1 or c[1] > 1 or c[2] > 1 ):
    #  assert False

    cm.save("temp.txt")
    cm2 = ColorModel()
    cm2.load("temp.txt")
    img = Image("logo")
    img2 = cm2.threshold(img)
    cm2.add((0, 0, 255))
    img3 = cm2.threshold(img)
    cm2.add((255, 255, 0))
    cm2.add((0, 255, 255))
    cm2.add((255, 0, 255))
    img4 = cm2.threshold(img)
    cm2.add(img)
    img5 = cm2.threshold(img)

    results = [img, img2, img3, img4, img5]
    name_stem = "test_color_colormap_build"
    perform_diff(results, name_stem)

    #c=img.mean_color()
    #if( c[0] > 1 or c[1] > 1 or c[2] > 1 ):
    #  assert False


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


def test_color_conversion_func_bgr():
    #we'll just go through the space to make sure nothing blows up
    img = Image(testimage)
    results = []
    results.append(img.to_bgr())
    results.append(img.to_rgb())
    results.append(img.to_hls())
    results.append(img.to_hsv())
    results.append(img.to_xyz())

    bgr = img.to_bgr()

    results.append(bgr.to_bgr())
    results.append(bgr.to_rgb())
    results.append(bgr.to_hls())
    results.append(bgr.to_hsv())
    results.append(bgr.to_xyz())

    name_stem = "test_color_conversion_func_bgr"
    perform_diff(results, name_stem, tolerance=4.0)


def test_color_conversion_func_rgb():
    img = Image(testimage)
    if not img.is_bgr():
        assert False
    rgb = img.to_rgb()

    foo = rgb.to_bgr()
    if not foo.is_bgr():
        assert False

    foo = rgb.to_rgb()
    if not foo.is_rgb():
        assert False

    foo = rgb.to_hls()
    if not foo.is_hls():
        assert False

    foo = rgb.to_hsv()
    if not foo.is_hsv():
        assert False

    foo = rgb.to_xyz()
    if not foo.is_xyz():
        assert False


def test_color_conversion_func_hsv():
    img = Image(testimage)
    hsv = img.to_hsv()
    results = [hsv]
    results.append(hsv.to_bgr())
    results.append(hsv.to_rgb())
    results.append(hsv.to_hls())
    results.append(hsv.to_hsv())
    results.append(hsv.to_xyz())
    name_stem = "test_color_conversion_func_hsv"
    perform_diff(results, name_stem, tolerance=4.0)


def test_color_conversion_func_hls():
    img = Image(testimage)

    hls = img.to_hls()
    results = [hls]

    results.append(hls.to_bgr())
    results.append(hls.to_rgb())
    results.append(hls.to_hls())
    results.append(hls.to_hsv())
    results.append(hls.to_xyz())

    name_stem = "test_color_conversion_func_hls"
    perform_diff(results, name_stem, tolerance=4.0)


def test_color_conversion_func_xyz():
    img = Image(testimage)

    xyz = img.to_xyz()
    results = [xyz]
    results.append(xyz.to_bgr())
    results.append(xyz.to_rgb())
    results.append(xyz.to_hls())
    results.append(xyz.to_hsv())
    results.append(xyz.to_xyz())

    name_stem = "test_color_conversion_func_xyz"
    perform_diff(results, name_stem, tolerance=8.0)


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
    perform_diff(results, name_stem, tolerance=3.0)

    for b in blobs:
        if b.hole_contour is not None:
            count += len(b.hole_contour)
    if count != 7:
        assert False


def test_blob_hull():
    img = Image("../data/sampleimages/blockhead.png")
    blobber = BlobMaker()
    blobs = blobber.extract(img)
    blobs.draw()

    results = [img]
    name_stem = "test_blob_holes"
    perform_diff(results, name_stem, tolerance=3.0)

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


def test_image_convolve():
    img = Image(testimageclr)
    kernel = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    img2 = img.convolve(kernel, center=(2, 2))

    results = [img2]
    name_stem = "test_image_convolve"
    perform_diff(results, name_stem)

    c = img.mean_color()
    d = img2.mean_color()
    e0 = abs(c[0] - d[0])
    e1 = abs(c[1] - d[1])
    e2 = abs(c[2] - d[2])
    if e0 > 1 or e1 > 1 or e2 > 1:
        assert False


def test_detection_ocr():
    img = Image(ocrimage)

    foundtext = img.read_text()
    print foundtext
    if len(foundtext) <= 1:
        assert False


def test_template_match():
    source = Image("../data/sampleimages/templatetest.png")
    template = Image("../data/sampleimages/template.png")
    t = 2
    fs = source.find_template(template, threshold=t)
    fs.draw()
    results = [source]
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


def test_template_match_rgb():
    source = Image("../data/sampleimages/templatetest.png")
    template = Image("../data/sampleimages/template.png")
    t = 2
    fs = source.find_template(template, threshold=t, grayscale=False)
    fs.draw()
    results = [source]
    name_stem = "test_template_match"
    perform_diff(results, name_stem)


def test_image_intergralimage():
    img = Image(logo)
    ii = img.integral_image()

    if len(ii) == 0:
        assert False


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


def test_embiggen():
    img = Image(logo)

    results = []
    w = int(img.width * 1.2)
    h = int(img.height * 1.2)

    results.append(img.embiggen(size=(w, h), color=Color.RED))
    results.append(img.embiggen(size=(w, h), color=Color.RED, pos=(30, 30)))

    results.append(img.embiggen(size=(w, h), color=Color.RED, pos=(-20, -20)))
    results.append(img.embiggen(size=(w, h), color=Color.RED, pos=(30, -20)))
    results.append(img.embiggen(size=(w, h), color=Color.RED, pos=(60, -20)))
    results.append(img.embiggen(size=(w, h), color=Color.RED, pos=(60, 30)))

    results.append(img.embiggen(size=(w, h), color=Color.RED, pos=(80, 80)))
    results.append(img.embiggen(size=(w, h), color=Color.RED, pos=(30, 80)))
    results.append(img.embiggen(size=(w, h), color=Color.RED, pos=(-20, 80)))
    results.append(img.embiggen(size=(w, h), color=Color.RED, pos=(-20, 30)))

    name_stem = "test_embiggen"
    perform_diff(results, name_stem)


def test_create_binary_mask():
    img2 = Image(logo)
    results = []
    results.append(
        img2.create_binary_mask(color1=(0, 100, 100), color2=(255, 200, 200)))
    results.append(
        img2.create_binary_mask(color1=(0, 0, 0), color2=(128, 128, 128)))
    results.append(
        img2.create_binary_mask(color1=(0, 0, 128), color2=(255, 255, 255)))

    name_stem = "test_createBinaryMask"
    perform_diff(results, name_stem)


def test_apply_binary_mask():
    img = Image(logo)
    mask = img.create_binary_mask(color1=(0, 128, 128), color2=(255, 255, 255))
    results = []
    results.append(img.apply_binary_mask(mask))
    results.append(img.apply_binary_mask(mask, bg_color=Color.RED))

    name_stem = "test_applyBinaryMask"
    perform_diff(results, name_stem, tolerance=3.0)


def test_apply_pixel_func():
    img = Image(logo)

    def myfunc(pixels):
        b, g, r = pixels
        return r, g, b

    img = img.apply_pixel_function(myfunc)
    name_stem = "test_apply_pixel_func"
    results = [img]
    perform_diff(results, name_stem)


def test_apply_side_by_side():
    img = Image(logo)
    img3 = Image(testimage2)

    # LB = little image big image
    # BL = big image little image
    # this is important to test all the possible cases.
    results = []

    results.append(img3.side_by_side(img, side='right', scale=False))
    results.append(img3.side_by_side(img, side='left', scale=False))
    results.append(img3.side_by_side(img, side='top', scale=False))
    results.append(img3.side_by_side(img, side='bottom', scale=False))

    results.append(img.side_by_side(img3, side='right', scale=False))
    results.append(img.side_by_side(img3, side='left', scale=False))
    results.append(img.side_by_side(img3, side='top', scale=False))
    results.append(img.side_by_side(img3, side='bottom', scale=False))

    results.append(img3.side_by_side(img, side='right', scale=True))
    results.append(img3.side_by_side(img, side='left', scale=True))
    results.append(img3.side_by_side(img, side='top', scale=True))
    results.append(img3.side_by_side(img, side='bottom', scale=True))

    results.append(img.side_by_side(img3, side='right', scale=True))
    results.append(img.side_by_side(img3, side='left', scale=True))
    results.append(img.side_by_side(img3, side='top', scale=True))
    results.append(img.side_by_side(img3, side='bottom', scale=True))

    name_stem = "test_apply_side_by_side"
    perform_diff(results, name_stem)


def test_resize():
    img = Image(logo)
    w = img.width
    h = img.height
    img2 = img.resize(w * 2, None)
    if img2.width != w * 2 or img2.height != h * 2:
        assert False

    img3 = img.resize(h=h * 2)

    if img3.width != w * 2 or img3.height != h * 2:
        assert False

    img4 = img.resize(h=h * 2, w=w * 2)

    if img4.width != w * 2 or img4.height != h * 2:
        assert False

    results = [img2, img3, img4]
    name_stem = "test_resize"
    perform_diff(results, name_stem)


def test_create_alpha_mask():
    alpha_mask = Image(alphaSrcImg)
    mask = alpha_mask.create_alpha_mask(hue=60)
    mask2 = alpha_mask.create_alpha_mask(hue_lb=59, hue_ub=61)
    top = Image(topImg)
    bottom = Image(bottomImg)
    bottom = bottom.blit(top, alpha_mask=mask2)
    results = [mask, mask2, bottom]
    name_stem = "test_create_alpha_mask"
    perform_diff(results, name_stem)


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

    if isinstance(imgs, ImageSet):
        pass
    else:
        assert False


def test_hsv_conversion():
    px = Image((1, 1))
    px[0, 0] = Color.GREEN
    if Color.hsv(Color.GREEN) == px.to_hsv()[0, 0]:
        pass
    else:
        assert False


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
    img2 = circs[0].crop()
    img3 = circs[0].crop(no_mask=True)

    results = [img, img2, img3]
    name_stem = "test_hough_circle"
    perform_diff(results, name_stem)

    if img2 is not None and img3 is not None:
        pass
    else:
        assert False


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
        if (ub.mask - b.mask).mean_color() != Color.BLACK:
            assert False


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
    #~ perform_diff(results,name_stem)


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
    try:
        import cv2
    except:
        pass
        return

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
    match3 = Image("../data/sampleimages/kptest2.png")
    match2 = Image("../data/sampleimages/aerospace.jpg")  # should be none

    fs0 = match0.find_keypoint_match(template)  # test zero
    fs1 = match1.find_keypoint_match(template, quality=300.00, min_dist=0.5,
                                     min_match=0.2)
    fs3 = match3.find_keypoint_match(template, quality=300.00, min_dist=0.5,
                                     min_match=0.2)
    print "This should fail"
    fs2 = match2.find_keypoint_match(template, quality=500.00, min_dist=0.2,
                                     min_match=0.1)
    if fs0 is not None and fs1 is not None and fs2 is None and fs3 is not None:
        fs0.draw()
        fs1.draw()
        fs3.draw()
        f = fs0[0]
        f.draw_rect()
        f.draw()
        f.get_homography()
        f.get_min_rect()
        f.x
        f.y
        f.coordinates()
    else:
        assert False

    results = [match0, match1, match2, match3]
    name_stem = "test_find_keypoint_match"
    perform_diff(results, name_stem)


def test_draw_keypoint_matches():
    template = Image("../data/sampleimages/KeypointTemplate2.png")
    match0 = Image("../data/sampleimages/kptest0.png")
    result = match0.draw_keypoint_matches(template, thresh=500.00,
                                          min_dist=0.15, width=1)
    results = [result]
    name_stem = "test_draw_keypoint_matches"
    perform_diff(results, name_stem, tolerance=4.0)


def test_basic_palette():
    # FIXME: Test should have assertion
    img = Image(testimageclr)
    img = img.scale(0.1)  # scale down the image to reduce test time
    img._generate_palette(10, False)
    if img._mPalette is not None\
            and img._mPaletteMembers is not None \
            and img._mPalettePercentages is not None \
            and img._mPaletteBins == 10:
        img._generate_palette(20, True)
        if img._mPalette is not None \
                and img._mPaletteMembers is not None \
                and img._mPalettePercentages is not None \
                and img._mPaletteBins == 20:
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

    p = img.get_palette(hue=True)
    b2 = img.find_blobs_from_palette(p[0:5])
    b2.draw()

    if len(b1) > 0 and len(b2) > 0:
        pass
    else:
        assert False


def test_skeletonize():
    img = Image(logo)
    s = img.skeletonize()
    s2 = img.skeletonize(10)

    results = [s, s2]
    name_stem = "test_skelotinze"
    perform_diff(results, name_stem)


def test_threshold():
    # FIXME: Test should have assertion
    img = Image(logo)
    for t in range(0, 255):
        img.threshold(t)


# FIXME: This test fails when all tests are executed
def test_smart_threshold():
    img = Image("../data/sampleimages/RatTop.png")
    mask = Image((img.width, img.height))
    mask.dl().circle((100, 100), 80, color=Color.MAYBE_BACKGROUND, filled=True)
    mask.dl().circle((100, 100), 60, color=Color.MAYBE_FOREGROUND, filled=True)
    mask.dl().circle((100, 100), 40, color=Color.FOREGROUND, filled=True)
    mask = mask.apply_layers()
    new_mask1 = img.smart_threshold(mask=mask)
    new_mask2 = img.smart_threshold(rect=(30, 30, 150, 185))

    results = [new_mask1, new_mask2]
    name_stem = "test_smart_threshold"
    perform_diff(results, name_stem)


def test_smart_find_blobs():
    img = Image("../data/sampleimages/RatTop.png")
    mask = Image((img.width, img.height))
    mask.dl().circle((100, 100), 80, color=Color.MAYBE_BACKGROUND, filled=True)
    mask.dl().circle((100, 100), 60, color=Color.MAYBE_FOREGROUND, filled=True)
    mask.dl().circle((100, 100), 40, color=Color.FOREGROUND, filled=True)
    mask = mask.apply_layers()
    blobs = img.smart_find_blobs(mask=mask)
    blobs.draw()
    results = [img]

    if len(blobs) < 1:
        assert False

    for t in range(2, 3):
        img = Image("../data/sampleimages/RatTop.png")
        blobs2 = img.smart_find_blobs(rect=(30, 30, 150, 185), thresh_level=t)
        if blobs2 is not None:
            blobs2.draw()
            results.append(img)

    name_stem = "test_smart_find_blobs"
    perform_diff(results, name_stem)


def test_image_webp_load():
    #only run if webm suppport exist on system
    try:
        import webm
    except:
        if SHOW_WARNING_TESTS:
            logger.warning("Couldn't run the webp test as optional webm "
                           "library required")
    else:
        img = Image(webp)

        if len(img.to_string()) <= 1:
            assert False


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


def test_get_exif_data():
    img = Image("../data/sampleimages/cat.jpg")
    img2 = Image(testimage)
    d1 = img.get_exif_data()
    d2 = img2.get_exif_data()
    if len(d1) > 0 and len(d2) == 0:
        pass
    else:
        assert False


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


def test_apply_dft_filter():
    img = Image("../data/sampleimages/RedDog2.jpg")
    flt = Image("../data/sampleimages/RedDogFlt.png")
    f1 = img.apply_dft_filter(flt)
    f2 = img.apply_dft_filter(flt, grayscale=True)
    results = [f1, f2]
    name_stem = "test_apply_dft_filter"
    perform_diff(results, name_stem)


def test_high_pass_filter():
    img = Image("../data/sampleimages/RedDog2.jpg")
    a = img.high_pass_filter(0.5)
    b = img.high_pass_filter(0.5, grayscale=True)
    c = img.high_pass_filter(0.5, y_cutoff=0.4)
    d = img.high_pass_filter(0.5, y_cutoff=0.4, grayscale=True)
    e = img.high_pass_filter([0.5, 0.4, 0.3])
    f = img.high_pass_filter([0.5, 0.4, 0.3], y_cutoff=[0.5, 0.4, 0.3])

    results = [a, b, c, d, e, f]
    name_stem = "test_HighPassFilter"
    perform_diff(results, name_stem)


def test_low_pass_filter():
    img = Image("../data/sampleimages/RedDog2.jpg")
    a = img.low_pass_filter(0.5)
    b = img.low_pass_filter(0.5, grayscale=True)
    c = img.low_pass_filter(0.5, y_cutoff=0.4)
    d = img.low_pass_filter(0.5, y_cutoff=0.4, grayscale=True)
    e = img.low_pass_filter([0.5, 0.4, 0.3])
    f = img.low_pass_filter([0.5, 0.4, 0.3], y_cutoff=[0.5, 0.4, 0.3])

    results = [a, b, c, d, e, f]
    name_stem = "test_LowPassFilter"
    perform_diff(results, name_stem)


def test_dft_gaussian():
    img = Image("../data/sampleimages/RedDog2.jpg")
    flt = DFT.create_gaussian_filter(dia=300, size=(300, 300), highpass=False)
    fltimg = img.filter(flt)
    fltimggray = img.filter(flt, grayscale=True)
    flt = DFT.create_gaussian_filter(dia=300, size=(300, 300), highpass=True)
    fltimg1 = img.filter(flt)
    fltimggray1 = img.filter(flt, grayscale=True)
    results = [fltimg, fltimggray, fltimg1, fltimggray1]
    name_stem = "test_DFT_gaussian"
    perform_diff(results, name_stem)


def test_dft_butterworth():
    img = Image("../data/sampleimages/RedDog2.jpg")
    flt = DFT.create_butterworth_filter(dia=300, size=(300, 300), order=3,
                                        highpass=False)
    fltimg = img.filter(flt)
    fltimggray = img.filter(flt, grayscale=True)
    flt = DFT.create_butterworth_filter(dia=100, size=(300, 300), order=3,
                                        highpass=True)
    fltimg1 = img.filter(flt)
    fltimggray1 = img.filter(flt, grayscale=True)
    results = [fltimg, fltimggray, fltimg1, fltimggray1]
    name_stem = "test_DFT_butterworth"
    perform_diff(results, name_stem)


def test_dft_lowpass():
    img = Image("../data/sampleimages/RedDog2.jpg")
    flt = DFT.create_lowpass_filter(x_cutoff=150, size=(600, 600))
    fltimg = img.filter(flt)
    fltimggray = img.filter(flt, grayscale=True)
    results = [fltimg, fltimggray]
    name_stem = "test_DFT_lowpass"
    perform_diff(results, name_stem, 20)


def test_dft_highpass():
    img = Image("../data/sampleimages/RedDog2.jpg")
    flt = DFT.create_lowpass_filter(x_cutoff=10, size=(600, 600))
    fltimg = img.filter(flt)
    fltimggray = img.filter(flt, grayscale=True)
    results = [fltimg, fltimggray]
    name_stem = "test_DFT_highpass"
    perform_diff(results, name_stem, 20)


def test_dft_notch():
    img = Image("../data/sampleimages/RedDog2.jpg")
    flt = DFT.create_notch_filter(dia1=500, size=(512, 512), ftype="lowpass")
    fltimg = img.filter(flt)
    fltimggray = img.filter(flt, grayscale=True)
    flt = DFT.create_notch_filter(dia1=300, size=(512, 512), ftype="highpass")
    fltimg1 = img.filter(flt)
    fltimggray1 = img.filter(flt, grayscale=True)
    results = [fltimg, fltimggray, fltimg1, fltimggray1]
    name_stem = "test_DFT_notch"
    perform_diff(results, name_stem, 20)


def test_find_haar_features():
    img = Image("../data/sampleimages/orson_welles.jpg")
    face = HaarCascade("face.xml")  # old HaarCascade
    f = img.find_haar_features(face)
    f2 = img.find_haar_features("face_cv2.xml")  # new cv2 HaarCascade
    if len(f) > 0 and len(f2) > 0:
        f.draw()
        f2.draw()
        f[0].get_width()
        f[0].get_height()
        f[0].draw()
        f[0].x
        f[0].y
        f[0].length()
        f[0].get_area()
        pass
    else:
        assert False

    results = [img]
    name_stem = "test_find_haar_features"
    perform_diff(results, name_stem)


def test_biblical_flood_fill():
    img = Image(testimage2)
    b = img.find_blobs()
    img.flood_fill(b.coordinates(), tolerance=3, color=Color.RED)
    img.flood_fill(b.coordinates(), tolerance=(3, 3, 3), color=Color.BLUE)
    img.flood_fill(b.coordinates(), tolerance=(3, 3, 3), color=Color.GREEN,
                   fixed_range=False)
    img.flood_fill((30, 30), lower=3, upper=5, color=Color.ORANGE)
    img.flood_fill((30, 30), lower=3, upper=(5, 5, 5), color=Color.ORANGE)
    img.flood_fill((30, 30), lower=(3, 3, 3), upper=5, color=Color.ORANGE)
    img.flood_fill((30, 30), lower=(3, 3, 3), upper=(5, 5, 5))
    img.flood_fill((30, 30), lower=(3, 3, 3), upper=(5, 5, 5),
                   color=np.array([255, 0, 0]))
    img.flood_fill((30, 30), lower=(3, 3, 3), upper=(5, 5, 5),
                   color=[255, 0, 0])

    results = [img]
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

    if len(b1) == len(b2):
        pass
    else:
        assert False


def test_band_pass_filter():
    img = Image("../data/sampleimages/RedDog2.jpg")
    a = img.band_pass_filter(0.1, 0.3)
    b = img.band_pass_filter(0.1, 0.3, grayscale=True)
    c = img.band_pass_filter(0.1, 0.3, y_cutoff_low=0.1, y_cutoff_high=0.3)
    d = img.band_pass_filter(0.1, 0.3, y_cutoff_low=0.1, y_cutoff_high=0.3,
                             grayscale=True)
    e = img.band_pass_filter([0.1, 0.2, 0.3], [0.5, 0.5, 0.5])
    f = img.band_pass_filter([0.1, 0.2, 0.3], [0.5, 0.5, 0.5],
                             y_cutoff_low=[0.1, 0.2, 0.3],
                             y_cutoff_high=[0.6, 0.6, 0.6])
    results = [a, b, c, d, e, f]
    name_stem = "test_band_pass_filter"
    perform_diff(results, name_stem)


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
    name_stem = "test_lineCrop"
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
    b = img.find_blobs()
    results = []

    for bs in b:
        temp = bs.crop()
        derp = temp.rotate(bs.get_angle(), fixed=False)
        derp.draw_text(str(bs.get_angle()), 10, 10, color=Color.RED)
        results.append(derp)
        bs.rectify_major_axis()
        results.append(bs.blob_image())

    name_stem = "test_feature_angles_rotate"
    perform_diff(results, name_stem, tolerance=7.0)


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


def test_pixelize():
    img = Image("../data/sampleimages/The1970s.png")
    img1 = img.pixelize(4)
    img2 = img.pixelize((5, 13))
    img3 = img.pixelize((img.width / 10, img.height))
    img4 = img.pixelize((img.width, img.height / 10))
    img5 = img.pixelize((12, 12), (200, 180, 250, 250))
    img6 = img.pixelize((12, 12), (600, 80, 250, 250))
    img7 = img.pixelize((12, 12), (600, 80, 250, 250), levels=4)
    img8 = img.pixelize((12, 12), levels=6)
    #img9 = img.pixelize(4, )
    #img10 = img.pixelize((5,13))
    #img11 = img.pixelize((img.width/10,img.height), mode=True)
    #img12 = img.pixelize((img.width,img.height/10), mode=True)
    #img13 = img.pixelize((12,12),(200,180,250,250), mode=True)
    #img14 = img.pixelize((12,12),(600,80,250,250), mode=True)
    #img15 = img.pixelize((12,12),(600,80,250,250),levels=4, mode=True)
    #img16 = img.pixelize((12,12),levels=6, mode=True)

    results = [img1, img2, img3, img4, img5, img6, img7, img8]
              # img9,img10,img11,img12,img13,img14,img15,img16]
    name_stem = "test_pixelize"
    perform_diff(results, name_stem, tolerance=6.0)


def test_hue_from_rgb():
    img = Image("lenna")
    img_hsv = img.to_hsv()
    h, s, r = img_hsv[100, 300]
    err = 2
    hue = Color.get_hue_from_rgb(img[100, 300])
    if hue > h - err and hue < h + err:
        pass
    else:
        assert False


def test_hue_from_bgr():
    img = Image("lenna")
    img_hsv = img.to_hsv()
    h, s, r = img_hsv[150, 400]
    err = 2
    color_tuple = tuple(reversed(img[150, 400]))
    hue = Color.get_hue_from_bgr(color_tuple)
    if hue > h - err and hue < h + err:
        pass
    else:
        assert False


def test_hue_to_rgb():
    r, g, b = Color.hue_to_rgb(0)
    if (r, g, b) == (255, 0, 0):
        pass
    else:
        assert False
    r, g, b = Color.hue_to_rgb(15)
    if (r, g, b) == (255, 128, 0):
        pass
    else:
        assert False
    r, g, b = Color.hue_to_rgb(30)
    if (r, g, b) == (255, 255, 0):
        pass
    else:
        assert False
    r, g, b = Color.hue_to_rgb(45)
    if (r, g, b) == (128, 255, 0):
        pass
    else:
        assert False
    r, g, b = Color.hue_to_rgb(60)
    if (r, g, b) == (0, 255, 0):
        pass
    else:
        assert False
    r, g, b = Color.hue_to_rgb(75)
    if (r, g, b) == (0, 255, 128):
        pass
    else:
        assert False
    r, g, b = Color.hue_to_rgb(90)
    if (r, g, b) == (0, 255, 255):
        pass
    else:
        assert False
    r, g, b = Color.hue_to_rgb(105)
    if (r, g, b) == (0, 128, 255):
        pass
    else:
        assert False
    r, g, b = Color.hue_to_rgb(120)
    if (r, g, b) == (0, 0, 255):
        pass
    else:
        assert False
    r, g, b = Color.hue_to_rgb(135)
    if (r, g, b) == (128, 0, 255):
        pass
    else:
        assert False
    r, g, b = Color.hue_to_rgb(150)
    if (r, g, b) == (255, 0, 255):
        pass
    else:
        assert False
    r, g, b = Color.hue_to_rgb(165)
    if (r, g, b) == (255, 0, 128):
        pass
    else:
        assert False


def test_hue_to_bgr():
    b, g, r = Color.hue_to_bgr(0)
    if (r, g, b) == (255, 0, 0):
        pass
    else:
        assert False
    b, g, r = Color.hue_to_bgr(15)
    if (r, g, b) == (255, 128, 0):
        pass
    else:
        assert False
    b, g, r = Color.hue_to_bgr(30)
    if (r, g, b) == (255, 255, 0):
        pass
    else:
        assert False
    b, g, r = Color.hue_to_bgr(45)
    if (r, g, b) == (128, 255, 0):
        pass
    else:
        assert False
    b, g, r = Color.hue_to_bgr(60)
    if (r, g, b) == (0, 255, 0):
        pass
    else:
        assert False
    b, g, r = Color.hue_to_bgr(75)
    if (r, g, b) == (0, 255, 128):
        pass
    else:
        assert False
    b, g, r = Color.hue_to_bgr(90)
    if (r, g, b) == (0, 255, 255):
        pass
    else:
        assert False
    b, g, r = Color.hue_to_bgr(105)
    if (r, g, b) == (0, 128, 255):
        pass
    else:
        assert False
    b, g, r = Color.hue_to_bgr(120)
    if (r, g, b) == (0, 0, 255):
        pass
    else:
        assert False
    b, g, r = Color.hue_to_bgr(135)
    if (r, g, b) == (128, 0, 255):
        pass
    else:
        assert False
    b, g, r = Color.hue_to_bgr(150)
    if (r, g, b) == (255, 0, 255):
        pass
    else:
        assert False
    b, g, r = Color.hue_to_bgr(165)
    if (r, g, b) == (255, 0, 128):
        pass
    else:
        assert False


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
    visual_test = True
    name_stem = 'test_skintone'
    perform_diff(masks, name_stem, tolerance=17)


def test_find_keypoints_all():
    try:
        import cv2
    except:
        pass
        return
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
    pass


def test_upload_flickr():
    try:
        import flickrapi
    except:
        if SHOW_WARNING_TESTS:
            logger.warning("Couldn't run the upload test as optional flickr "
                           "library required")
        pass
    else:
        img = Image('simplecv')
        api_key = None
        api_secret = None
        if api_key is None or api_secret is None:
            pass
        else:
            try:
                ret = img.upload('flickr', api_key, api_secret)
                if ret:
                    pass
                else:
                    assert False
            except:  # we will chock this up to key errors
                pass


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
        crop3 = img.crop([(-3, -3), (10, 20)])
        crop4 = img.crop((-10, 10, 20, 20), centered=True)
        crop5 = img.crop([-10, -10, 20, 20])

    results = [crop, crop1, crop2, crop3]
    name_stem = "test_image_new_crop"
    perform_diff(results, name_stem)

    diff = crop - crop1
    c = diff.mean_color()
    if c[0] > 0 or c[1] > 0 or c[2] > 0:
        assert False


def test_image_temp_save():
    img1 = Image("lenna")
    img2 = Image(logo)
    path = []
    path.append(img1.save(temp=True))
    path.append(img2.save(temp=True))
    for i in path:
        if i is None:
            assert False

    assert True


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


def test_upload_dropbox():
    try:
        import dropbox
    except:
        if SHOW_WARNING_TESTS:
            logger.warning("Couldn't run the upload test as optional dropbox "
                           "library required")
        pass
    else:
        img = Image('simplecv')
        api_key = ''
        api_secret = ''
        if api_key is None or api_secret is None:
            pass
        else:
            ret = img.upload('dropbox', api_key, api_secret)
            if ret:
                pass
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
    # FIXME: Test should have assertion
    img = Image('lenna')
    b = img.find_blobs()
    m1 = b[-1].get_edge_image()
    m2 = b[-1].get_hull_edge_image()
    m3 = b[-1].get_full_edge_image()
    m4 = b[-1].get_full_hull_edge_image()


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
    pass


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
        img[x, 200] = (255.0, 255.0, 255.0)
    l = Line(img, ((300, 100), (300, 500)))
    if l.img_intersections(img) == [(300, 200)]:
        pass
    else:
        assert False


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
    lines.draw()
    result = [img]
    name_stem = "test_image_gridLines"
    perform_diff(result, name_stem, 5)

    if lines == 0 or lines is None:
        assert False


def test_logical_and():
    img = Image("lenna")
    img1 = img.logical_and(img.invert())
    if not img1.get_numpy().all():
        pass
    else:
        assert False


def test_logical_or():
    img = Image("lenna")
    img1 = img.logical_or(img.invert())
    if img1.get_numpy().all():
        pass
    else:
        assert False


def test_logical_nand():
    img = Image("lenna")
    img1 = img.logical_nand(img.invert())
    if img1.get_numpy().all():
        pass
    else:
        assert False


def test_logical_xor():
    img = Image("lenna")
    img1 = img.logical_xor(img.invert())
    if img1.get_numpy().all():
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
    perform_diff(result, name_stem, 1.0)


def test_steganograpy():
    img = Image(logo)
    msg = 'How do I SimpleCV?'
    img.stega_encode(msg)
    img.save(logo)
    img2 = Image(logo)
    msg2 = img2.stega_decode()


def test_watershed():
    img = Image('../data/sampleimages/wshed.jpg')
    img1 = img.watershed()
    img2 = img.watershed(dilate=3, erode=2)
    img3 = img.watershed(mask=img.threshold(128), erode=1, dilate=1)
    my_mask = Image((img.width, img.height))
    my_mask = my_mask.flood_fill((0, 0), color=Color.WATERSHED_BG)
    mask = img.threshold(128)
    my_mask = (my_mask - (mask.dilate(2) + mask.erode(2)).to_bgr())
    img4 = img.watershed(mask=my_mask, use_my_mask=True)
    blobs = img.find_blobs_from_watershed(dilate=3, erode=2)
    blobs = img.find_blobs_from_watershed()
    blobs = img.find_blobs_from_watershed(mask=img.threshold(128), erode=1,
                                          dilate=1)
    blobs = img.find_blobs_from_watershed(mask=img.threshold(128), erode=1,
                                          dilate=1, invert=True)
    blobs = img.find_blobs_from_watershed(mask=my_mask, use_my_mask=True)
    result = [img1, img2, img3, img4]
    name_stem = "test_watershed"
    perform_diff(result, name_stem, 3.0)


def test_minmax():
    img = Image('../data/sampleimages/wshed.jpg')
    min = img.min_value()
    min, pts = img.min_value(locations=True)
    max = img.max_value()
    max, pts = img.max_value(locations=True)


def test_roi_feature():
    img = Image(testimageclr)
    mask = img.threshold(248).dilate(5)
    blobs = img.find_blobs_from_mask(mask, minsize=1)
    x, y = np.where(mask.get_gray_numpy() > 0)
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
    xroi, yroi = np.where(test.get_gray_numpy() > 128)
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
    if newimg[100, 50][0] == 0:
        pass
    else:
        assert False
    ls = img.get_line_scan(x=100, channel=1)
    ls[50] = 0
    new_img = img.replace_line_scan(ls)
    if new_img[100, 50][1] == 0:
        pass
    else:
        assert False


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


# def test_tvDenoising():
#     return  # this is way too slow.
#     try:
#         from skimage.filter import denoise_tv_chambolle
#
#         img = Image('lenna')
#         img1 = img.tv_denoising(gray=False, weight=20)
#         img2 = img.tv_denoising(weight=50, max_iter=250)
#         img3 = img.to_gray()
#         img3 = img3.tv_denoising(gray=True, weight=20)
#         img4 = img.tv_denoising(resize=0.5)
#         result = [img1, img2, img3, img4]
#         name_stem = "test_tvDenoising"
#         perform_diff(result, name_stem, 3)
#     except ImportError:
#         pass


# FIXME: the following tests should be merged
def test_motion_blur():
    i = Image('lenna')
    d = ('n', 's', 'e', 'w', 'ne', 'nw', 'se', 'sw')
    i0 = i.motion_blur(intensity=20, direction=d[0])
    i1 = i.motion_blur(intensity=20, direction=d[1])
    i2 = i.motion_blur(intensity=20, direction=d[2])
    i3 = i.motion_blur(intensity=20, direction=d[3])
    i4 = i.motion_blur(intensity=10, direction=d[4])
    i5 = i.motion_blur(intensity=10, direction=d[5])
    i6 = i.motion_blur(intensity=10, direction=d[6])
    i7 = i.motion_blur(intensity=10, direction=d[7])
    a = i.motion_blur(intensity=0)
    c = 0
    img = (i0, i1, i2, i3, i4, i5, i6, i7)
    for im in img:
        if im is not i:
            c += 1

    if c == 8 and a is i:
        pass
    else:
        assert False


def test_motion_blur2():
    image = Image('lenna')
    d = (-70, -45, -30, -10, 100, 150, 235, 420)
    p = (10, 20, 30, 40, 50, 60, 70, 80)
    img = []

    a = image.motion_blur2(0)
    for i in range(8):
        img += [image.motion_blur2(p[i], d[i])]
    c = 0
    for im in img:
        if im is not i:
            c += 1

    if c == 8 and a is image:
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

    for list_ in list1, list2, list3:
        edge_lines = img.edge_snap(list_)
        edge_lines.draw(color=Color.YELLOW, width=4)

    name_stem = "test_edgeSnap"
    result = [img]
    perform_diff(result, name_stem, 0.7)


def test_grayscalmatrix():
    img = Image("lenna")
    graymat = img.get_grayscale_matrix()
    newimg = Image(graymat, color_space=ColorSpace.GRAY)
    from numpy import array_equal

    if not array_equal(img.get_gray_numpy(), newimg.get_gray_numpy()):
        assert False


def test_get_lightness():
    img = Image('lenna')
    i = img.get_lightness()
    if int(i[27, 42][0]) == int((max(img[27, 42]) + min(img[27, 42])) / 2):
        pass
    else:
        assert False


def test_get_luminosity():
    img = Image('lenna')
    i = img.get_luminosity()
    a = np.array(img[27, 42], dtype=np.int)
    if int(i[27, 42][0]) == int(np.average(a, 0, (0.21, 0.71, 0.07))):
        pass
    else:
        assert False


def test_get_average():
    img = Image('lenna')
    i = img.get_average()
    if int(i[0, 0][0]) == int((img[0, 0][0]
                               + img[0, 0][1]
                               + img[0, 0][2]) / 3):
        pass
    else:
        assert False


def test_smart_rotate():
    img = Image('kptest2.png', sample=True)

    st1 = img.smart_rotate(auto=False, fixed=False).resize(500, 500)
    st2 = img.rotate(27, fixed=False).resize(500, 500)
    diff = np.average((st1 - st2).get_numpy())
    if diff > 1.7:
        print diff
        assert False
    else:
        assert True


def test_normalize():
    img = Image("lenna")
    img1 = img.normalize()
    img2 = img.normalize(min_cut=0, max_cut=0)
    result = [img1, img2]
    name_stem = "test_image_normalize"
    perform_diff(result, name_stem, 5)
    pass


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
    name_stem = "test_image_histBackProj"
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
