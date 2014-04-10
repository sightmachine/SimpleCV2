import warnings
try:
    from PIL import Image as PilImage
except:
    import Image as PilImage

from simplecv.base import logger
from simplecv.factory import Factory
from simplecv.core.image import image_method


@image_method
def stega_encode(img, message):
    """
    **SUMMARY**

    A simple steganography tool for hidding messages in images.
    **PARAMETERS**

    * *message* -A message string that you would like to encode.

    **RETURNS**

    Your message encoded in the returning image.

    **EXAMPLE**

    >>>> img = Image('lenna')
    >>>> img2 = img.stega_encode("HELLO WORLD!")
    >>>> img2.save("TopSecretImg.png")
    >>>> img3 = Image("TopSecretImg.png")
    >>>> img3.stega_decode()

    **NOTES**

    More here:
    http://en.wikipedia.org/wiki/Steganography
    You will need to install stepic:
    http://domnit.org/stepic/doc/pydoc/stepic.html

    You may need to monkey with jpeg compression
    as it seems to degrade the encoded message.

    PNG sees to work quite well.

    """

    try:
        import stepic
    except ImportError:
        logger.warning("stepic library required")
        return None
    warnings.simplefilter("ignore")
    pil_img = PilImage.frombuffer("RGB", img.size, img.to_rgb().to_string())
    stepic.encode_inplace(pil_img, message)
    ret_val = Factory.Image(pil_img)
    return ret_val.flip_vertical()


@image_method
def stega_decode(img):
    """
    **SUMMARY**

    A simple steganography tool for hidding and finding
    messages in images.

    **RETURNS**

    Your message decoded in the image.

    **EXAMPLE**

    >>>> img = Image('lenna')
    >>>> img2 = img.stega_encode("HELLO WORLD!")
    >>>> img2.save("TopSecretImg.png")
    >>>> img3 = Image("TopSecretImg.png")
    >>>> img3.stega_decode()

    **NOTES**

    More here:
    http://en.wikipedia.org/wiki/Steganography
    You will need to install stepic:
    http://domnit.org/stepic/doc/pydoc/stepic.html

    You may need to monkey with jpeg compression
    as it seems to degrade the encoded message.

    PNG sees to work quite well.

    """
    try:
        import stepic
    except ImportError:
        logger.warning("stepic library required")
        return None
    warnings.simplefilter("ignore")
    pil_img = PilImage.frombuffer("RGB", img.size(), img.to_rgb().to_string())
    result = stepic.decode(pil_img)
    return result
