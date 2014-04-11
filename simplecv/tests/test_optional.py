import os
import tempfile

from nose.tools import assert_equals

from simplecv.base import logger
from simplecv.color import Color
from simplecv.image import Image
from simplecv.core.camera.screen_camera import ScreenCamera
from simplecv.tests.utils import perform_diff

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


def test_detection_barcode():
    try:
        import zbar
    except:
        return

    img1 = Image(testimage)
    img2 = Image(testbarcode)

    nocode = img1.find_barcode()
    if nocode:  # we should find no barcode in our test image
        assert False
    code = img2.find_barcode()
    code.draw()
    result = [img1, img2]
    name_stem = "test_detection_barcode"
    perform_diff(result, name_stem)


def test_detection_ocr():
    img = Image(ocrimage)

    foundtext = img.read_text()
    assert len(foundtext) > 1


def test_image_webp_load():
    #only run if webm suppport exist on system
    try:
        import webm
    except:
        logger.warning("Couldn't run the webp test as optional webm "
                       "library required")
        return

    img = Image(webp)
    assert len(img.to_string()) > 1


def test_image_webp_save():
    #only run if webm suppport exist on system
    try:
        import webm
    except ImportError:
        logger.warning("Couldn't run the webp test as optional "
                       "webm library required")
        return

    img = Image('simplecv')
    tf = tempfile.NamedTemporaryFile(suffix=".webp")
    assert img.save(tf.name)


def test_screenshot():
    try:
        import pyscreenshot
    except ImportError:
        logger.warning("Couldn't run the pyscreenshot test. "
                       "Install pyscreenshot library")
        return

    sc = ScreenCamera()
    res = sc.get_resolution()
    img = sc.get_image()
    crop = (res[0]/4, res[1]/4, res[0]/2, res[1]/2)
    sc.set_roi(crop)
    crop_img = sc.get_image()
    assert img
    assert crop_img


def test_tv_denoising():
    try:
        from skimage.filter import denoise_tv_chambolle
    except ImportError:
        return

    img = Image('lenna')
    img1 = img.tv_denoising(gray=False, weight=20)
    img2 = img.tv_denoising(weight=50, max_iter=250)
    img3 = img.to_gray()
    img3 = img3.tv_denoising(gray=True, weight=20)
    img4 = img.tv_denoising(resize=0.5)
    result = [img1, img2, img3, img4]
    name_stem = "test_tvDenoising"
    perform_diff(result, name_stem, 3)


def test_steganograpy():
    try:
        import stepic
    except ImportError:
        return

    tmp_file = os.path.join(tempfile.gettempdir(), 'simplecv_tmp.png')
    img = Image(logo)
    msg = 'How do I SimpleCV?'
    img = img.stega_encode(msg)
    assert img
    img.save(tmp_file)
    img2 = Image(tmp_file)
    msg2 = img2.stega_decode()
    assert_equals(msg, msg2)
