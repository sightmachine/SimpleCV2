from simplecv.base import logger
from simplecv.core.camera.frame_source import FrameSource
from simplecv.factory import Factory

_SANE_INIT = False


class Scanner(FrameSource):
    """
    **SUMMARY**

    The Scanner lets you use any supported SANE-compatable
    scanner as a SimpleCV camera
    List of supported devices:
    http://www.sane-project.org/sane-supported-devices.html

    Requires the PySANE wrapper for libsane.  The sane scanner object
    is available for direct manipulation at Scanner.device

    This scanner object is heavily modified from
    https://bitbucket.org/DavidVilla/pysane

    Constructor takes an index (default 0) and a list of SANE options
    (default is color mode).

    **EXAMPLE**

    >>> scan = Scanner(0, { "mode": "gray" })
    >>> preview = scan.get_preview()
    >>> stuff = preview.find_blobs(minsize = 1000)
    >>> topleft = (np.min(stuff.x()), np.min(stuff.y()))
    >>> bottomright = (np.max(stuff.x()), np.max(stuff.y()))
    >>> scan.set_roi(topleft, bottomright)
    >>> scan.set_property("resolution", 1200) #set high resolution
    >>> scan.set_property("mode", "color")
    >>> img = scan.get_image()
    >>> scan.set_roi() #reset region of interest
    >>> img.show()

    """

    def __init__(self, id=0, properties={"mode": "color"}):
        super(Scanner, self).__init__()
        self.usbid = None
        self.manufacturer = None
        self.model = None
        self.kind = None
        self.device = None
        self.max_x = None
        self.max_y = None
        global _SANE_INIT
        import sane

        if not _SANE_INIT:
            try:
                sane.init()
                _SANE_INIT = True
            except Exception:
                logger.warn("Initializing pysane failed.")
                return

        devices = sane.get_devices()
        if not len(devices):
            logger.warn("Did not find a sane-compatable device")
            return

        self.usbid, self.manufacturer, self.model, self.kind = devices[id]

        self.device = sane.open(self.usbid)
        self.max_x = self.device.br_x
        self.max_y = self.device.br_y  # save our extents for later

        for name, value in properties.items():
            setattr(self.device, name, value)

    def get_image(self):
        """
        **SUMMARY**

        Retrieve an Image-object from the scanner.  Any ROI set with
        setROI() is taken into account.
        **RETURNS**

        A SimpleCV Image.  Note that whatever the scanner mode is,
        SimpleCV will return a 3-channel, 8-bit image.

        **EXAMPLES**
        >>> scan = Scanner()
        >>> scan.get_image().show()
        """
        return Factory.Image(self.device.scan())

    def get_preview(self):
        """
        **SUMMARY**

        Retrieve a preview-quality Image-object from the scanner.
        **RETURNS**

        A SimpleCV Image.  Note that whatever the scanner mode is,
        SimpleCV will return a 3-channel, 8-bit image.

        **EXAMPLES**
        >>> scan = Scanner()
        >>> scan.get_preview().show()
        """
        self.preview = True
        img = Factory.Image(self.device.scan())
        self.preview = False
        return img

    def get_all_properties(self):
        """
        **SUMMARY**

        Return a list of all properties and values from the scanner
        **RETURNS**

        Dictionary of active options and values.  Inactive options appear
        as "None"

        **EXAMPLES**
        >>> scan = Scanner()
        >>> print scan.get_all_properties()
        """
        props = {}
        for prop in self.device.optlist:
            val = None
            if hasattr(self.device, prop):
                val = getattr(self.device, prop)
            props[prop] = val

        return props

    def print_properties(self):

        """
        **SUMMARY**

        Print detailed information about the SANE device properties
        **RETURNS**

        Nothing

        **EXAMPLES**
        >>> scan = Scanner()
        >>> scan.print_properties()
        """
        for prop in self.device.optlist:
            try:
                print self.device[prop]
            except Exception:
                pass

    def get_property(self, prop):
        """
        **SUMMARY**
        Returns a single property value from the SANE device
        equivalent to Scanner.device.PROPERTY

        **RETURNS**
        Value for option or None if missing/inactive

        **EXAMPLES**
        >>> scan = Scanner()
        >>> print scan.get_property('mode')
        color
        """
        if hasattr(self.device, prop):
            return getattr(self.device, prop)
        return None

    def set_roi(self, topleft=(0, 0), bottomright=(-1, -1)):
        """
        **SUMMARY**
        Sets an ROI for the scanner in the current resolution.  The
        two parameters, topleft and bottomright, will default to the
        device extents, so the ROI can be reset by calling setROI with
        no parameters.

        The ROI is set by SANE in resolution independent units (default
        MM) so resolution can be changed after ROI has been set.

        **RETURNS**
        None

        **EXAMPLES**
        >>> scan = Scanner()
        >>> scan.set_roi((50, 50), (100,100))
        >>> scan.get_image().show() # a very small crop on the scanner


        """
        self.device.tl_x = self.px2mm(topleft[0])
        self.device.tl_y = self.px2mm(topleft[1])
        if bottomright[0] == -1:
            self.device.br_x = self.max_x
        else:
            self.device.br_x = self.px2mm(bottomright[0])

        if bottomright[1] == -1:
            self.device.br_y = self.max_y
        else:
            self.device.br_y = self.px2mm(bottomright[1])

    def set_property(self, prop, val):
        """
        **SUMMARY**
        Assigns a property value from the SANE device
        equivalent to Scanner.device.PROPERTY = VALUE

        **RETURNS**
        None

        **EXAMPLES**
        >>> scan = Scanner()
        >>> print scan.get_property('mode')
        color
        >>> scan.set_property("mode", "gray")
        """
        setattr(self.device, prop, val)

    def px2mm(self, pixels=1):
        """
        **SUMMARY**
        Helper function to convert native scanner resolution to millimeter
        units

        **RETURNS**
        Float value

        **EXAMPLES**
        >>> scan = Scanner()
        >>> scan.px2mm(scan.device.resolution) #return DPI in DPMM
        """
        return float(pixels * 25.4 / float(self.device.resolution))
