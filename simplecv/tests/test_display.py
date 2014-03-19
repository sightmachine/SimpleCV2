# To run this test you need python nose tools installed
# Run test just use:
#   nosetest test_display.py

import pickle

from cv2 import cv
import numpy as np

from simplecv.camera import FrameSource
from simplecv.color import Color, ColorCurve
from simplecv.color_model import ColorModel
from simplecv.drawing_layer import DrawingLayer
from simplecv.features.blobmaker import BlobMaker
from simplecv.features.haar_cascade import HaarCascade
from simplecv.image_class import Image


VISUAL_TEST = False  # if TRUE we save the images
                    # otherwise we DIFF against them - the default is False
SHOW_WARNING_TESTS = False  # show that warnings are working
                            # tests will pass but warnings are generated.

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

#track images
trackimgs = ["../data/sampleimages/tracktest0.jpg",
             "../data/sampleimages/tracktest1.jpg",
             "../data/sampleimages/tracktest2.jpg",
             "../data/sampleimages/tracktest3.jpg",
             "../data/sampleimages/tracktest4.jpg",
             "../data/sampleimages/tracktest5.jpg",
             "../data/sampleimages/tracktest6.jpg",
             "../data/sampleimages/tracktest7.jpg",
             "../data/sampleimages/tracktest8.jpg",
             "../data/sampleimages/tracktest9.jpg", ]


#Given a set of images, a path, and a tolerance do the image diff.
def img_diffs(test_imgs, name_stem, tolerance, path):
    count = len(test_imgs)
    for idx in range(0, count):
        lhs = test_imgs[idx].apply_layers()  # this catches drawing methods
        if lhs.is_gray():
            lhs = lhs.to_bgr()
        fname = standard_path + name_stem + str(idx) + ".jpg"
        rhs = Image(fname)
        if lhs.width == rhs.width and lhs.height == rhs.height:
            diff = (lhs - rhs)
            val = np.average(diff.get_ndarray())
            if val > tolerance:
                print val
                return True
    return False


#Save a list of images to a standard path.
def img_saves(test_imgs, name_stem, path=standard_path):
    count = len(test_imgs)
    for idx in range(0, count):
        fname = standard_path + name_stem + str(idx) + ".jpg"
        test_imgs[idx].save(fname)  # ,quality=95)


#perform the actual image save and image diffs.
def perform_diff(result, name_stem, tolerance=2.0, path=standard_path):
    if VISUAL_TEST:  # save the correct images for a visual test
        img_saves(result, name_stem, path)
    else:  # otherwise we test our output against the visual test
        if img_diffs(result, name_stem, tolerance, path):
            assert False
        else:
            pass


def test_image_stretch():
    img = Image(greyscaleimage)
    stretched = img.stretch(100, 200)
    if stretched is None:
        assert False

    result = [stretched]
    name_stem = "test_stretch"
    perform_diff(result, name_stem)


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

    if img[1, 1] != copy[1, 1] or img.size() != copy.size():
        assert False

    result = [copy]
    name_stem = "test_image_copy"
    perform_diff(result, name_stem)
    pass


def test_image_setitem():
    img = Image(testimage)
    img[1, 1] = (0, 0, 0)
    newimg = Image(img.get_bitmap())
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
    newimg = Image(img.get_bitmap())
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
    pass


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


def test_image_splitchannels():
    img = Image(testimageclr)
    (r, g, b) = img.split_channels(True)
    (red, green, blue) = img.split_channels()
    result = [r, g, b, red, green, blue]
    name_stem = "test_image_splitchannels"
    perform_diff(result, name_stem)
    pass


def test_detection_lines():
    img = Image(testimage2)
    lines = img.find_lines()
    lines.draw()
    result = [img]
    name_stem = "test_detection_lines"
    perform_diff(result, name_stem)

    if lines == 0 or lines is None:
        assert False


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


def test_color_curve_hsl():
    y = np.array(
        [[0, 0], [64, 128], [192, 128], [255, 255]])  # These are the weights
    curve = ColorCurve(y)
    img = Image(testimage)
    img2 = img.apply_hls_curve(curve, curve, curve)
    img3 = img - img2

    result = [img2, img3]
    name_stem = "test_color_curve_hsl"
    perform_diff(result, name_stem)

    c = img3.mean_color()
    if c[0] > 2.0 or c[1] > 2.0 or c[2] > 2.0:
        # there may be a bit of roundoff error
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
    if c[0] > 1.0 or c[1] > 1.0 or c[2] > 1.0:
        # there may be a bit of roundoff error
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
    img2 = img.rotate(180, "full", scale=1)

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
    if color != (0, 0, 0):
        assert False

    dst = ((img.width * 0.05, img.height * 0.03),
           (img.width * 0.9, img.height * 0.1),
           (img.width * 0.8, img.height * 0.7),
           (img.width * 0.2, img.height * 0.9))
    w = img.warp(dst)

    results = [s, w]
    name_stem = "test_image_shear_warp"
    perform_diff(results, name_stem)

    color = s[0, 0]
    if color != (0, 0, 0):
        assert False


def test_image_affine():
    img = Image(testimage2)
    src = ((0, 0), (img.width - 1, 0), (img.width - 1, img.height - 1))
    dst = ((img.width / 2, 0), (img.width - 1, img.height / 2),
           (img.width / 2, img.height - 1))
    a_warp = cv.CreateMat(2, 3, cv.CV_32FC1)
    cv.GetAffineTransform(src, dst, a_warp)
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
    p_warp = cv.CreateMat(3, 3, cv.CV_32FC1)
    cv.GetPerspectiveTransform(src, dst, p_warp)
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


# FIXME: This test intermittently fails
def test_camera_undistort():
    fake_camera = FrameSource()
    fake_camera.load_calibration("TestCalibration")
    img = Image("../data/sampleimages/CalibImage0.png")
    img2 = fake_camera.undistort(img)

    results = [img2]
    name_stem = "test_camera_undistort"
    perform_diff(results, name_stem)

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
    if SHOW_WARNING_TESTS:
        crop7 = img.crop(0, 0, -10, 10)
        crop8 = img.crop(-50, -50, 10, 10)
        crop3 = img.crop(-3, -3, 10, 20)
        crop4 = img.crop(-10, 10, 20, 20, centered=True)
        crop5 = img.crop(-10, -10, 20, 20)

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


def test_template_match():
    source = Image("../data/sampleimages/templatetest.png")
    template = Image("../data/sampleimages/template.png")
    t = 2
    fs = source.find_template(template, threshold=t)
    fs.draw()
    results = [source]
    name_stem = "test_template_match"
    perform_diff(results, name_stem)


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

    name_stem = "test_create_binary_mask"
    perform_diff(results, name_stem)


def test_apply_binary_mask():
    img = Image(logo)
    mask = img.create_binary_mask(color1=(0, 128, 128), color2=(255, 255, 255))
    results = []
    results.append(img.apply_binary_mask(mask))
    results.append(img.apply_binary_mask(mask, bg_color=Color.RED))

    name_stem = "test_apply_binary_mask"
    perform_diff(results, name_stem, tolerance=3.0)


def test_apply_pixel_func():
    img = Image(logo)

    def myfunc((r, g, b)):
        return b, g, r

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
    try:
        import cv2
    except:
        pass
        return
    img = Image(testimage2)
    kp = img.find_keypoints()
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
    fs = current2.find_motion(prev, window=7, method='HS')
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
    fs = current3.find_motion(prev, window=7, method='LK', aggregate=False)
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
    perform_diff(results, name_stem, tolerance=4.0)


def test_keypoint_extraction():
    try:
        import cv2
    except:
        pass
        return

    img1 = Image("../data/sampleimages/KeypointTemplate2.png")
    img2 = Image("../data/sampleimages/KeypointTemplate2.png")
    img3 = Image("../data/sampleimages/KeypointTemplate2.png")

    kp1 = img1.find_keypoints()
    kp2 = img2.find_keypoints(highquality=True)
    kp3 = img3.find_keypoints(flavor="STAR")
    kp1.draw()
    kp2.draw()
    kp3.draw()
    #TODO: Fix FAST binding
    #~ kp4 = img.find_keypoints(flavor="FAST",min_quality=10)
    if len(kp1) == 190 and len(kp2) == 190 and len(kp3) == 37:
         #~ and len(kp4)==521:
        pass
    else:
        assert False
    results = [img1, img2, img3]
    name_stem = "test_keypoint_extraction"
    perform_diff(results, name_stem, tolerance=3.0)


def test_keypoint_match():
    try:
        import cv2
    except:
        pass
        return

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
        #f.mean_color()
        f.crop()
        f.x
        f.y
        f.coordinates()
    else:
        assert False

    results = [match0, match1, match2, match3]
    name_stem = "test_find_keypoint_match"
    perform_diff(results, name_stem)


def test_draw_keypoint_matches():
    try:
        import cv2
    except:
        pass
        return
    template = Image("../data/sampleimages/KeypointTemplate2.png")
    match0 = Image("../data/sampleimages/kptest0.png")
    result = match0.draw_keypoint_matches(template, thresh=500.00,
                                          min_dist=0.15, width=1)

    results = [result]
    name_stem = "test_draw_keypoint_matches"
    perform_diff(results, name_stem, tolerance=4.0)


def test_skeletonize():
    img = Image(logo)
    s = img.skeletonize()
    s2 = img.skeletonize(10)

    results = [s, s2]
    name_stem = "test_skelotinze"
    perform_diff(results, name_stem)


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


def test_get_dft_log_magnitude():
    img = Image("../data/sampleimages/RedDog2.jpg")
    lm3 = img.get_dft_log_magnitude()
    lm1 = img.get_dft_log_magnitude(grayscale=True)

    results = [lm3, lm1]
    name_stem = "test_get_dft_log_magnitude"
    perform_diff(results, name_stem)


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
    name_stem = "test_high_pass_filter"
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
    name_stem = "test_low_pass_filter"
    perform_diff(results, name_stem)


def test_find_haar_features():
    img = Image("../data/sampleimages/orson_welles.jpg")
    face = HaarCascade("face.xml")
    f = img.find_haar_features(face)
    f2 = img.find_haar_features("face.xml")
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


def test_line_crop():
    img = Image("../data/sampleimages/EdgeTest2.png")
    l = img.find_lines().sort_area()
    l = l[-5:-1]
    results = []
    for ls in l:
        results.append(ls.crop())
    name_stem = "test_lineCrop"
    perform_diff(results, name_stem, tolerance=3.0)


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
    perform_diff(results, name_stem, tolerance=7.0)


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
    perform_diff(results, name_stem, tolerance=9.0)


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

    results = [img1, img2, img3, img4, img5, img6, img7,
               img8]  # img9,img10,img11,img12,img13,img14,img15,img16]
    name_stem = "test_pixelize"
    perform_diff(results, name_stem, tolerance=6.0)


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


def test_sobel():
    img = Image("lenna")
    s = img.sobel()
    name_stem = "test_sobel"
    s = [s]
    perform_diff(s, name_stem)


def test_image_new_smooth():
    img = Image(testimage2)
    result = []
    result.append(img.median_filter())
    result.append(img.median_filter((3, 3)))
    result.append(img.median_filter((5, 5), grayscale=True))
    result.append(img.bilateral_filter())
    result.append(
        img.bilateral_filter(diameter=14, sigma_color=20, sigma_space=34))
    result.append(img.bilateral_filter(grayscale=True))
    result.append(img.blur())
    result.append(img.blur((5, 5)))
    result.append(img.blur((3, 5), grayscale=True))
    result.append(img.gaussian_blur())
    result.append(img.gaussian_blur((3, 7), sigma_x=10, sigma_y=12))
    result.append(
        img.gaussian_blur((7, 9), sigma_x=10, sigma_y=12, grayscale=True))
    name_stem = "test_image_new_smooth"
    perform_diff(result, name_stem)


def test_camshift():
    ts = []
    bb = (195, 160, 49, 46)
    imgs = [Image(img) for img in trackimgs]
    ts = imgs[0].track("camshift", ts, imgs[1:], bb)
    if ts:
        pass
    else:
        assert False


def test_lk():
    ts = []
    bb = (195, 160, 49, 46)
    imgs = [Image(img) for img in trackimgs]
    ts = imgs[0].track("LK", ts, imgs[1:], bb)
    if ts:
        pass
    else:
        assert False
