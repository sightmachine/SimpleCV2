# /usr/bin/python
# To run this test you need python nose tools installed
# Run test just use:
#   nosetest test_optional.py
#
import tempfile

from nose.tools import with_setup, nottest

from simplecv.base import logger
from simplecv.color import Color
from simplecv.image_class import Image
from simplecv.camera import ScreenCamera


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


#These function names are required by nose test, please leave them as is
def setup_context():
    img = Image(testimage)


def destroy_context():
    img = ""


@with_setup(setup_context, destroy_context)
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
        # FIXME: no function perform_diff
        perform_diff(result, name_stem)
    else:
        pass


def test_detection_ocr():
    img = Image(ocrimage)

    foundtext = img.read_text()
    print foundtext
    if(len(foundtext) <= 1):
        assert False
    else:
        pass


def test_image_webp_load():
    #only run if webm suppport exist on system
    try:
        import webm
    except:
        if SHOW_WARNING_TESTS:
            logger.warning("Couldn't run the webp test as optional webm "
                           "library required")
        pass

    else:
        img = Image(webp)

        if len(img.to_string()) <= 1:
            assert False

        else:
            pass


def test_image_webp_save():
    #only run if webm suppport exist on system
    try:
        import webm
    except ImportError:
        if SHOW_WARNING_TESTS:
            logger.warning("Couldn't run the webp test as optional "
                           "webm library required")
        pass

    else:
        img = Image('simplecv')
        tf = tempfile.NamedTemporaryFile(suffix=".webp")
        if img.save(tf.name):
            pass
        else:
            assert False


def test_screenshot():
    try:
        import pyscreenshot
    except ImportError:
        if SHOW_WARNING_TESTS:
            logger.warning("Couldn't run the pyscreenshot test. "
                           "Install pyscreenshot library")
        pass

    else:
        sc = ScreenCamera()
        res = sc.get_resolution()
        img = sc.get_image()
        crop = (res[0]/4, res[1]/4, res[0]/2, res[1]/2)
        sc.set_roi(crop)
        cropImg = sc.get_image()
        if img and cropImg:
            assert True
        else:
            assert False
