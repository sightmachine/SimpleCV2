import ctypes as ct

import cv2
import numpy as np

from simplecv.base import logger
from simplecv.core.camera.camera import Camera
from simplecv.factory import Factory

ARAVIS_ENABLED = True
try:
    from gi.repository import Aravis
except ImportError:
    ARAVIS_ENABLED = False


class GigECamera(Camera):
    """
        GigE Camera driver via Aravis
    """
    plist = [
        'available_pixel_formats',
        'available_pixel_formats_as_display_names',
        'available_pixel_formats_as_strings',
        'binning',
        'device_id',
        'exposure_time',
        'exposure_time_bounds',
        'frame_rate',
        'frame_rate_bounds',
        'gain',
        'gain_bounds',
        'height_bounds',
        'model_name',
        'payload',
        'pixel_format',
        'pixel_format_as_string',
        'region',
        'sensor_size',
        'trigger_source',
        'vendor_name',
        'width_bounds'
    ]

    #def __init__(self, camera_id=None, properties={}, threaded=False):
    def __init__(self, properties={}, threaded=False):
        if not ARAVIS_ENABLED:
            print "GigE is supported by the Aravis library, download and \
                   build from https://github.com/sightmachine/aravis"
            print "Note that you need to set GI_TYPELIB_PATH=$GI_TYPELIB_PATH:\
                   (PATH_TO_ARAVIS)/src for the GObject Introspection"
            return

        self._cam = Aravis.Camera.new(None)

        self._pixel_mode = "RGB"
        if properties.get("mode", False):
            self._pixel_mode = properties.pop("mode")

        if self._pixel_mode == "gray":
            self._cam.set_pixel_format(Aravis.PIXEL_FORMAT_MONO_8)
        else:
            #we'll use bayer (basler cams)
            self._cam.set_pixel_format(Aravis.PIXEL_FORMAT_BAYER_BG_8)
            #TODO, deal with other pixel formats

        if properties.get("roi", False):
            roi = properties['roi']
            self._cam.set_region(*roi)
            #TODO, check sensor size

        if properties.get("width", False):
            #TODO, set internal function to scale results of getimage
            pass

        if properties.get("framerate", False):
            self._cam.set_frame_rate(properties['framerate'])

        self._stream = self._cam.create_stream(None, None)

        payload = self._cam.get_payload()
        self._stream.push_buffer(Aravis.Buffer.new_allocate(payload))
        [_, _, width, height] = self._cam.get_region()
        self._height, self._width = height, width

    def get_image(self):
        if not ARAVIS_ENABLED:
            logger.warn("Initializing failed, Aravis library not found.")
            return

        camera = self._cam
        camera.start_acquisition()
        buff = self._stream.pop_buffer()
        self.capture_time = buff.timestamp_ns / 1000000.0
        img = np.fromstring(ct.string_at(buff.data_address(), buff.size),
                            dtype=np.uint8).reshape(self._height, self._width)
        rgb = cv2.cvtColor(img, cv2.COLOR_BAYER_BG2BGR)
        self._stream.push_buffer(buff)
        camera.stop_acquisition()
        #TODO, we should handle software triggering
        #(separate capture and get image events)

        return Factory.Image(rgb)

    def get_property_list(self):
        return self.plist

    def get_property(self, name=None):
        '''
        This function get's the properties availble to the camera

        Usage:
        > camera.get_property('region')
        > (0, 0, 128, 128)

        Available Properties:
        see function camera.get_property_list()
        '''

        if name is None:
            print "You need to provide a property, available properties are:"
            print ""
            for prop in self.get_property_list():
                print prop
            return

        stringval = "get_{}".format(name)
        try:
            return getattr(self._cam, stringval)()
        except Exception:
            print 'Property {} does not appear to exist'.format(name)
            return None

    def set_property(self, name=None, *args):
        '''
        This function sets the property available to the camera

        Usage:
        > camera.set_property('region',(256,256))

        Available Properties:
        see function camera.get_property_list()
        '''

        if name is None:
            print "You need to provide a property, available properties are:"
            print ""
            self.get_all_properties()
            return

        if len(args) <= 0:
            print "You must provide a value to set"
            return

        stringval = "set_{}".format(name)
        try:
            #FIXME - may be setattr should be used?
            return getattr(self._cam, stringval)(*args)
        except Exception:
            print 'Property {} does not appear to exist or value\
                   is not in correct format'.format(name)
            return None

    def get_all_properties(self):
        '''
        This function just prints out all the properties available to the
        camera
        '''
        for prop in self.get_property_list():
            print "{}: {}".format(prop, self.get_property(prop))
