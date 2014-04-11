import os

ZXING_ENABLED = True
try:
    import zxing
except ImportError:
    ZXING_ENABLED = False

from simplecv.base import logger
from simplecv.features.features import FeatureSet
from simplecv.factory import Factory
from simplecv.core.image import image_method

_barcode_reader = None


@image_method
def find_barcode(img, do_zlib=True, zxing_path=""):
    """
    **SUMMARY**

    This function requires zbar and the zbar python wrapper
    to be installed or zxing and the zxing python library.

    **ZBAR**

    To install please visit:
    http://zbar.sourceforge.net/

    On Ubuntu Linux 12.04 or greater:
    sudo apt-get install python-zbar


    **ZXING**

    If you have the python-zxing library installed, you can find 2d and 1d
    barcodes in your image.  These are returned as Barcode feature objects
    in a FeatureSet.  The single parameter is the ZXing_path along with
    setting the do_zlib flag to False. You do not need the parameter if you
    don't have the ZXING_LIBRARY env parameter set.

    You can clone python-zxing at:

    http://github.com/oostendo/python-zxing

    **INSTALLING ZEBRA CROSSING**

    * Download the latest version of zebra crossing from:
     http://code.google.com/p/zxing/

    * unpack the zip file where ever you see fit

      >>> cd zxing-x.x, where x.x is the version number of zebra crossing
      >>> ant -f core/build.xml
      >>> ant -f javase/build.xml

      This should build the library, but double check the readme

    * Get our helper library

      >>> git clone git://github.com/oostendo/python-zxing.git
      >>> cd python-zxing
      >>> python setup.py install

    * Our library does not have a setup file. You will need to add
       it to your path variables. On OSX/Linux use a text editor to modify
       your shell file (e.g. .bashrc)

      export ZXING_LIBRARY=<FULL PATH OF ZXING LIBRARY - (i.e. step 2)>
      for example:

      export ZXING_LIBRARY=/my/install/path/zxing-x.x/

      On windows you will need to add these same variables to the system
      variable, e.g.

      http://www.computerhope.com/issues/ch000549.htm

    * On OSX/Linux source your shell rc file (e.g. source .bashrc). Windows
     users may need to restart.

    * Go grab some barcodes!

    .. Warning::
      Users on OSX may see the following error:

      RuntimeWarning: tmpnam is a potential security risk to your program

      We are working to resolve this issue. For normal use this should not
      be a problem.

    **Returns**

    A :py:class:`FeatureSet` of :py:class:`Barcode` objects. If no barcodes
     are detected the method returns None.

    **EXAMPLE**

    >>> bc = cam.getImage()
    >>> barcodes = img.findBarcodes()
    >>> for b in barcodes:
    >>>     b.draw()

    **SEE ALSO**

    :py:class:`FeatureSet`
    :py:class:`Barcode`

    """
    if do_zlib:
        try:
            import zbar
        except:
            logger.warning('The zbar library is not installed, please '
                           'install to read barcodes')
            return None

        #configure zbar
        scanner = zbar.ImageScanner()
        scanner.parse_config('enable')
        raw = img.get_pil().convert('L').tostring()
        width = img.width
        height = img.height

        # wrap image data
        image = zbar.Image(width, height, 'Y800', raw)

        # scan the image for barcodes
        scanner.scan(image)
        barcode = None
        # extract results
        for symbol in image:
            # do something useful with results
            barcode = symbol
    else:
        if not ZXING_ENABLED:
            logger.warn("Zebra Crossing (ZXing) Library not installed. "
                        "Please see the release notes.")
            return None

        global _barcode_reader
        if not _barcode_reader:
            if not zxing_path:
                _barcode_reader = zxing.BarCodeReader()
            else:
                _barcode_reader = zxing.BarCodeReader(zxing_path)

        tmp_filename = os.tmpnam() + ".png"
        img.save(tmp_filename)
        barcode = _barcode_reader.decode(tmp_filename)
        os.unlink(tmp_filename)

    if barcode:
        f = Factory.Barcode(img, barcode)
        return FeatureSet([f])
    else:
        return None
