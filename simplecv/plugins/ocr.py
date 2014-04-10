from cStringIO import StringIO

OCR_ENABLED = True
try:
    import tesseract
except ImportError:
    OCR_ENABLED = False

from simplecv.core.image import convert
from simplecv.core.image import image_method


@image_method
def read_text(img):
    """
    **SUMMARY**

    This function will return any text it can find using OCR on the
    image.

    Please note that it does not handle rotation well, so if you need
    it in your application try to rotate and/or crop the area so that
    the text would be the same way a document is read

    **RETURNS**

    A String

    **EXAMPLE**

    >>> img = Imgae("somethingwithtext.png")
    >>> text = img.read_text()
    >>> print text

    **NOTE**

    If you're having run-time problems I feel bad for your son,
    I've got 99 problems but dependencies ain't one:

    http://code.google.com/p/tesseract-ocr/
    http://code.google.com/p/python-tesseract/

    """

    if not OCR_ENABLED:
        return "Please install the correct OCR library required - " \
               "http://code.google.com/p/tesseract-ocr/ " \
               "http://code.google.com/p/python-tesseract/"

    api = tesseract.TessBaseAPI()
    api.SetOutputName("outputName")
    api.Init(".", "eng", tesseract.OEM_DEFAULT)
    api.SetPageSegMode(tesseract.PSM_AUTO)

    jpgdata = StringIO()
    convert.to_pil_image(img).save(jpgdata, "jpeg")
    jpgdata.seek(0)
    stringbuffer = jpgdata.read()
    result = tesseract.ProcessPagesBuffer(stringbuffer, len(stringbuffer),
                                          api)
    return result
