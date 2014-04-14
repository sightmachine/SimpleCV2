from collections import deque
import time
import threading
import ctypes as ct

from simplecv.base import SYSTEM
from simplecv.factory import Factory
from simplecv.core.camera.frame_source import FrameSource

try:
    from PIL import Image as PilImage
except:
    import Image as PilImage


class AVTCameraThread(threading.Thread):

    def __init__(self, camera):
        super(AVTCameraThread, self).__init__()
        self.running = True
        self.verbose = False
        self.logger = None
        self.framerate = 0
        self._stop = threading.Event()
        self.camera = camera
        self.lock = threading.Lock()
        self.name = 'Thread-Camera-ID-' + str(self.camera.uniqueid)

    def run(self):
        counter = 0
        timestamp = time.time()

        while self.running:
            self.lock.acquire()
            self.camera.runCommand("AcquisitionStart")
            frame = self.camera._get_frame(1000)

            if frame:
                img = Factory.Image(PilImage.fromstring(
                    self.camera.imgformat,
                    (self.camera.width, self.camera.height),
                    frame.ImageBuffer[:int(frame.ImageBufferSize)]))
            self.camera._buffer.appendleft(img)

            self.camera.runCommand("AcquisitionStop")
            self.lock.release()
            counter += 1
            time.sleep(0.01)

            if time.time() - timestamp >= 1:
                self.camera.framerate = counter
                counter = 0
                timestamp = time.time()

    def stop(self):
        self._stop.set()

    def stopped(self):
        return self._stop.isSet()


AVT_CAMERA_ERRORS = [
    ("ePvErrSuccess", "No error"),
    ("ePvErrCameraFault", "Unexpected camera fault"),
    ("ePvErrInternalFault", "Unexpected fault in PvApi or driver"),
    ("ePvErrBadHandle", "Camera handle is invalid"),
    ("ePvErrBadParameter", "Bad parameter to API call"),
    ("ePvErrBadSequence", "Sequence of API calls is incorrect"),
    ("ePvErrNotFound", "Camera or attribute not found"),
    ("ePvErrAccessDenied", "Camera cannot be opened in the specified mode"),
    ("ePvErrUnplugged", "Camera was unplugged"),
    ("ePvErrInvalidSetup", "Setup is invalid (an attribute is invalid)"),
    ("ePvErrResources", "System/network resources or memory not available"),
    ("ePvErrBandwidth", "1394 bandwidth not available"),
    ("ePvErrQueueFull", "Too many frames on queue"),
    ("ePvErrBufferTooSmall", "Frame buffer is too small"),
    ("ePvErrCancelled", "Frame cancelled by user"),
    ("ePvErrDataLost", "The data for the frame was lost"),
    ("ePvErrDataMissing", "Some data in the frame is missing"),
    ("ePvErrTimeout", "Timeout during wait"),
    ("ePvErrOutOfRange", "Attribute value is out of the expected range"),
    ("ePvErrWrongType", "Attribute is not this type (wrong access function)"),
    ("ePvErrForbidden", "Attribute write forbidden at this time"),
    ("ePvErrUnavailable", "Attribute is not available at this time"),
    ("ePvErrFirewall", "A firewall is blocking the traffic (Windows only)"),
]


def pverr(errcode):
    if errcode:
        raise Exception(": ".join(AVT_CAMERA_ERRORS[errcode]))


class AVTCamera(FrameSource):
    """
    **SUMMARY**
    AVTCamera is a ctypes wrapper for the Prosilica/Allied Vision cameras,
    such as the "manta" series.

    These require the PvAVT binary driver from Allied Vision:
    http://www.alliedvisiontec.com/us/products/1108.html

    Note that as of time of writing the new VIMBA driver is not available
    for Mac/Linux - so this uses the legacy PvAVT drive

    Props to Cixelyn, whos py-avt-pvapi module showed how to get much
    of this working https://bitbucket.org/Cixelyn/py-avt-pvapi

    All camera properties are directly from the PvAVT manual -- if not
    specified it will default to whatever the camera state is.  Cameras
    can either by

    **EXAMPLE**
    >>> cam = AVTCamera(0, {"width": 656, "height": 492})
    >>>
    >>> img = cam.get_image()
    >>> img.show()
    """

    _buffersize = 10  # Number of images to keep

    _properties = {
        "AcqEndTriggerEvent": ("Enum", "R/W"),
        "AcqEndTriggerMode": ("Enum", "R/W"),
        "AcqRecTriggerEvent": ("Enum", "R/W"),
        "AcqRecTriggerMode": ("Enum", "R/W"),
        "AcqStartTriggerEvent": ("Enum", "R/W"),
        "AcqStartTriggerMode": ("Enum", "R/W"),
        "FrameRate": ("Float32", "R/W"),
        "FrameStartTriggerDelay": ("Uint32", "R/W"),
        "FrameStartTriggerEvent": ("Enum", "R/W"),
        "FrameStartTriggerMode": ("Enum", "R/W"),
        "FrameStartTriggerOverlap": ("Enum", "R/W"),
        "AcquisitionFrameCount": ("Uint32", "R/W"),
        "AcquisitionMode": ("Enum", "R/W"),
        "RecorderPreEventCount": ("Uint32", "R/W"),
        "ConfigFileIndex": ("Enum", "R/W"),
        "ConfigFilePowerup": ("Enum", "R/W"),
        "DSPSubregionBottom": ("Uint32", "R/W"),
        "DSPSubregionLeft": ("Uint32", "R/W"),
        "DSPSubregionRight": ("Uint32", "R/W"),
        "DSPSubregionTop": ("Uint32", "R/W"),
        "DefectMaskColumnEnable": ("Enum", "R/W"),
        "ExposureAutoAdjustTol": ("Uint32", "R/W"),
        "ExposureAutoAlg": ("Enum", "R/W"),
        "ExposureAutoMax": ("Uint32", "R/W"),
        "ExposureAutoMin": ("Uint32", "R/W"),
        "ExposureAutoOutliers": ("Uint32", "R/W"),
        "ExposureAutoRate": ("Uint32", "R/W"),
        "ExposureAutoTarget": ("Uint32", "R/W"),
        "ExposureMode": ("Enum", "R/W"),
        "ExposureValue": ("Uint32", "R/W"),
        "GainAutoAdjustTol": ("Uint32", "R/W"),
        "GainAutoMax": ("Uint32", "R/W"),
        "GainAutoMin": ("Uint32", "R/W"),
        "GainAutoOutliers": ("Uint32", "R/W"),
        "GainAutoRate": ("Uint32", "R/W"),
        "GainAutoTarget": ("Uint32", "R/W"),
        "GainMode": ("Enum", "R/W"),
        "GainValue": ("Uint32", "R/W"),
        "LensDriveCommand": ("Enum", "R/W"),
        "LensDriveDuration": ("Uint32", "R/W"),
        "LensVoltage": ("Uint32", "R/V"),
        "LensVoltageControl": ("Uint32", "R/W"),
        "IrisAutoTarget": ("Uint32", "R/W"),
        "IrisMode": ("Enum", "R/W"),
        "IrisVideoLevel": ("Uint32", "R/W"),
        "IrisVideoLevelMax": ("Uint32", "R/W"),
        "IrisVideoLevelMin": ("Uint32", "R/W"),
        "VsubValue": ("Uint32", "R/C"),
        "WhitebalAutoAdjustTol": ("Uint32", "R/W"),
        "WhitebalAutoRate": ("Uint32", "R/W"),
        "WhitebalMode": ("Enum", "R/W"),
        "WhitebalValueRed": ("Uint32", "R/W"),
        "WhitebalValueBlue": ("Uint32", "R/W"),
        "EventAcquisitionStart": ("Uint32", "R/C 40000"),
        "EventAcquisitionEnd": ("Uint32", "R/C 40001"),
        "EventFrameTrigger": ("Uint32", "R/C 40002"),
        "EventExposureEnd": ("Uint32", "R/C 40003"),
        "EventAcquisitionRecordTrigger": ("Uint32", "R/C 40004"),
        "EventSyncIn1Rise": ("Uint32", "R/C 40010"),
        "EventSyncIn1Fall": ("Uint32", "R/C 40011"),
        "EventSyncIn2Rise": ("Uint32", "R/C 40012"),
        "EventSyncIn2Fall": ("Uint32", "R/C 40013"),
        "EventSyncIn3Rise": ("Uint32", "R/C 40014"),
        "EventSyncIn3Fall": ("Uint32", "R/C 40015"),
        "EventSyncIn4Rise": ("Uint32", "R/C 40016"),
        "EventSyncIn4Fall": ("Uint32", "R/C 40017"),
        "EventOverflow": ("Uint32", "R/C 65534"),
        "EventError": ("Uint32", "R/C"),
        "EventNotification": ("Enum", "R/W"),
        "EventSelector": ("Enum", "R/W"),
        "EventsEnable1": ("Uint32", "R/W"),
        "BandwidthCtrlMode": ("Enum", "R/W"),
        "ChunkModeActive": ("Boolean", "R/W"),
        "NonImagePayloadSize": ("Unit32", "R/V"),
        "PayloadSize": ("Unit32", "R/V"),
        "StreamBytesPerSecond": ("Uint32", "R/W"),
        "StreamFrameRateConstrain": ("Boolean", "R/W"),
        "StreamHoldCapacity": ("Uint32", "R/V"),
        "StreamHoldEnable": ("Enum", "R/W"),
        "TimeStampFrequency": ("Uint32", "R/C"),
        "TimeStampValueHi": ("Uint32", "R/V"),
        "TimeStampValueLo": ("Uint32", "R/V"),
        "Height": ("Uint32", "R/W"),
        "RegionX": ("Uint32", "R/W"),
        "RegionY": ("Uint32", "R/W"),
        "Width": ("Uint32", "R/W"),
        "PixelFormat": ("Enum", "R/W"),
        "TotalBytesPerFrame": ("Uint32", "R/V"),
        "BinningX": ("Uint32", "R/W"),
        "BinningY": ("Uint32", "R/W"),
        "CameraName": ("String", "R/W"),
        "DeviceFirmwareVersion": ("String", "R/C"),
        "DeviceModelName": ("String", "R/W"),
        "DevicePartNumber": ("String", "R/C"),
        "DeviceSerialNumber": ("String", "R/C"),
        "DeviceVendorName": ("String", "R/C"),
        "FirmwareVerBuild": ("Uint32", "R/C"),
        "FirmwareVerMajor": ("Uint32", "R/C"),
        "FirmwareVerMinor": ("Uint32", "R/C"),
        "PartClass": ("Uint32", "R/C"),
        "PartNumber": ("Uint32", "R/C"),
        "PartRevision": ("String", "R/C"),
        "PartVersion": ("String", "R/C"),
        "SerialNumber": ("String", "R/C"),
        "SensorBits": ("Uint32", "R/C"),
        "SensorHeight": ("Uint32", "R/C"),
        "SensorType": ("Enum", "R/C"),
        "SensorWidth": ("Uint32", "R/C"),
        "UniqueID": ("Uint32", "R/C"),
        "Strobe1ControlledDuration": ("Enum", "R/W"),
        "Strobe1Delay": ("Uint32", "R/W"),
        "Strobe1Duration": ("Uint32", "R/W"),
        "Strobe1Mode": ("Enum", "R/W"),
        "SyncIn1GlitchFilter": ("Uint32", "R/W"),
        "SyncInLevels": ("Uint32", "R/V"),
        "SyncOut1Invert": ("Enum", "R/W"),
        "SyncOut1Mode": ("Enum", "R/W"),
        "SyncOutGpoLevels": ("Uint32", "R/W"),
        "DeviceEthAddress": ("String", "R/C"),
        "HostEthAddress": ("String", "R/C"),
        "DeviceIPAddress": ("String", "R/C"),
        "HostIPAddress": ("String", "R/C"),
        "GvcpRetries": ("Uint32", "R/W"),
        "GvspLookbackWindow": ("Uint32", "R/W"),
        "GvspResentPercent": ("Float32", "R/W"),
        "GvspRetries": ("Uint32", "R/W"),
        "GvspSocketBufferCount": ("Enum", "R/W"),
        "GvspTimeout": ("Uint32", "R/W"),
        "HeartbeatInterval": ("Uint32", "R/W"),
        "HeartbeatTimeout": ("Uint32", "R/W"),
        "MulticastEnable": ("Enum", "R/W"),
        "MulticastIPAddress": ("String", "R/W"),
        "PacketSize": ("Uint32", "R/W"),
        "StatDriverType": ("Enum", "R/V"),
        "StatFilterVersion": ("String", "R/C"),
        "StatFrameRate": ("Float32", "R/V"),
        "StatFramesCompleted": ("Uint32", "R/V"),
        "StatFramesDropped": ("Uint32", "R/V"),
        "StatPacketsErroneous": ("Uint32", "R/V"),
        "StatPacketsMissed": ("Uint32", "R/V"),
        "StatPacketsReceived": ("Uint32", "R/V"),
        "StatPacketsRequested": ("Uint32", "R/V"),
        "StatPacketResent": ("Uint32", "R/V")
    }

    class AVTCameraInfo(ct.Structure):
        """
        AVTCameraInfo is an internal ctypes.Structure-derived class which
        contains metadata about cameras on the local network.

        Properties include:
        * UniqueId
        * CameraName
        * ModelName
        * PartNumber
        * SerialNumber
        * FirmwareVersion
        * PermittedAccess
        * InterfaceId
        * InterfaceType
        """
        _fields_ = [
            ("StructVer", ct.c_ulong),
            ("UniqueId", ct.c_ulong),
            ("CameraName", ct.c_char * 32),
            ("ModelName", ct.c_char * 32),
            ("PartNumber", ct.c_char * 32),
            ("SerialNumber", ct.c_char * 32),
            ("FirmwareVersion", ct.c_char * 32),
            ("PermittedAccess", ct.c_long),
            ("InterfaceId", ct.c_ulong),
            ("InterfaceType", ct.c_int)
        ]

        def __repr__(self):
            return "<SimpleCV.Camera.AVTCameraInfo " \
                   "- UniqueId: %s>" % self.UniqueId

    class AVTFrame(ct.Structure):
        _fields_ = [
            ("image_buffer", ct.POINTER(ct.c_char)),
            ("image_buffer_size", ct.c_ulong),
            ("ancillary_buffer", ct.c_int),
            ("ancillary_buffer_size", ct.c_int),
            ("Context", ct.c_int * 4),
            ("_reserved1", ct.c_ulong * 8),

            ("Status", ct.c_int),
            ("ImageSize", ct.c_ulong),
            ("AncillarySize", ct.c_ulong),
            ("Width", ct.c_ulong),
            ("Height", ct.c_ulong),
            ("RegionX", ct.c_ulong),
            ("RegionY", ct.c_ulong),
            ("Format", ct.c_int),
            ("BitDepth", ct.c_ulong),
            ("BayerPattern", ct.c_int),
            ("FrameCount", ct.c_ulong),
            ("TimestampLo", ct.c_ulong),
            ("TimestampHi", ct.c_ulong),
            ("_reserved2", ct.c_ulong * 32)
        ]

        def __init__(self, buffersize):
            self.image_buffer = ct.create_string_buffer(buffersize)
            self.image_buffer_size = ct.c_ulong(buffersize)
            self.ancillary_buffer = 0
            self.ancillary_buffer_size = 0
            self.img = None
            #self.hasImage = False
            self.frame = None

    # FIXME: __del__ prevents garbage collection of AVTCamera objects
    def __del__(self):
        #This function should disconnect from the AVT Camera
        pverr(self.dll.PvCameraClose(self.handle))

    def __init__(self, camera_id=-1, properties={}, threaded=False):
        super(AVTCamera, self).__init__()

        self._buffer = None  # Buffer to store images
        # in the rolling image buffer for threads
        self._lastimage = None  # Last image loaded into memory
        self._thread = None
        self._framerate = 0
        self.threaded = False
        self._pvinfo = {}

        if SYSTEM == "Windows":
            self.dll = ct.windll.LoadLibrary("PvAPI.dll")
        elif SYSTEM == "Darwin":
            self.dll = ct.CDLL("libPvAPI.dylib", ct.RTLD_GLOBAL)
        else:
            self.dll = ct.CDLL("libPvAPI.so")

        if not self._pvinfo.get("initialized", False):
            self.dll.PvInitialize()
            self._pvinfo['initialized'] = True
        #initialize.  Note that we rely on listAllCameras being the next
        #call, since it blocks on cameras initializing

        camlist = self.list_all_cameras()

        if not len(camlist):
            raise Exception("Couldn't find any cameras with the PvAVT "
                            "driver. Use SampleViewer to confirm you have one "
                            "connected.")

        if camera_id < 9000:  # camera was passed as an index reference
            if camera_id == -1:  # accept -1 for "first camera"
                camera_id = 0

            camera_id = camlist[camera_id].UniqueId

        camera_id = long(camera_id)
        self.handle = ct.c_uint()
        init_count = 0
        #wait until camera is availble:
        while self.dll.PvCameraOpen(camera_id, 0, ct.byref(self.handle)) != 0:
            if init_count > 4:  # Try to connect 5 times before giving up
                raise Exception('Could not connect to camera, please '
                                'verify with SampleViewer you can connect')
            init_count += 1
            time.sleep(1)  # sleep and retry to connect to camera in a second

        pverr(self.dll.PvCaptureStart(self.handle))
        self.uniqueid = camera_id

        self.set_property("AcquisitionMode", "SingleFrame")
        self.set_property("FrameStartTriggerMode", "Freerun")

        if properties.get("mode", "RGB") == 'gray':
            self.set_property("PixelFormat", "Mono8")
        else:
            self.set_property("PixelFormat", "Rgb24")

        #give some compatablity with other cameras
        if properties.get("mode", ""):
            properties.pop("mode")

        if properties.get("height", ""):
            properties["Height"] = properties["height"]
            properties.pop("height")

        if properties.get("width", ""):
            properties["Width"] = properties["width"]
            properties.pop("width")

        for prop in properties:
            self.set_property(prop, properties[prop])

        if threaded:
            self._thread = AVTCameraThread(self)
            self._thread.daemon = True
            self._buffer = deque(maxlen=self._buffersize)
            self._thread.start()
            self.threaded = True

        self._refresh_frame_stats()

    def restart(self):
        """
        This tries to restart the camera thread
        """
        self._thread.stop()
        self._thread = AVTCameraThread(self)
        self._thread.daemon = True
        self._buffer = deque(maxlen=self._buffersize)
        self._thread.start()

    def list_all_cameras(self):
        """
        **SUMMARY**
        List all cameras attached to the host

        **RETURNS**
        List of AVTCameraInfo objects, otherwise empty list

        """
        camlist = (self.AVTCameraInfo * 100)()
        starttime = time.time()
        while int(camlist[0].UniqueId) == 0 and time.time() - starttime < 10:
            self.dll.PvCameraListEx(ct.byref(camlist), 100,
                                    None, ct.sizeof(self.AVTCameraInfo))
            time.sleep(0.1)  # keep checking for cameras until timeout

        return [cam for cam in camlist if cam.UniqueId != 0]

    def run_command(self, command):
        """
        **SUMMARY**
        Runs a PvAVT Command on the camera

        Valid Commands include:
        * FrameStartTriggerSoftware
        * AcquisitionAbort
        * AcquisitionStart
        * AcquisitionStop
        * ConfigFileLoad
        * ConfigFileSave
        * TimeStampReset
        * TimeStampValueLatch

        **RETURNS**

        0 on success

        **EXAMPLE**
        >>>c = AVTCamera()
        >>>c.run_command("TimeStampReset")
        """
        return self.dll.PvCommandRun(self.handle, command)

    def get_property(self, name):
        """
        **SUMMARY**
        This retrieves the value of the AVT Camera attribute

        There are around 140 properties for the AVT Camera, so reference the
        AVT Camera and Driver Attributes pdf that is provided with
        the driver for detailed information

        Note that the error codes are currently ignored, so empty values
        may be returned.

        **EXAMPLE**
        >>>c = AVTCamera()
        >>>print c.get_property("ExposureValue")
        """
        valtype, _ = self._properties.get(name, (None, None))

        if not valtype:
            return None

        val = ''
        err = 0
        if valtype == "Enum":
            val = ct.create_string_buffer(100)
            vallen = ct.c_long()
            err = self.dll.PvAttrEnumGet(self.handle, name, val,
                                         100, ct.byref(vallen))
            val = str(val[:vallen.value])
        elif valtype == "Uint32":
            val = ct.c_uint()
            err = self.dll.PvAttrUint32Get(self.handle, name, ct.byref(val))
            val = int(val.value)
        elif valtype == "Float32":
            val = ct.c_float()
            err = self.dll.PvAttrFloat32Get(self.handle, name, ct.byref(val))
            val = float(val.value)
        elif valtype == "String":
            val = ct.create_string_buffer(100)
            vallen = ct.c_long()
            err = self.dll.PvAttrStringGet(self.handle, name, val,
                                           100, ct.byref(vallen))
            val = str(val[:vallen.value])
        elif valtype == "Boolean":
            val = ct.c_bool()
            err = self.dll.PvAttrBooleanGet(self.handle, name, ct.byref(val))
            val = bool(val.value)

        #TODO, handle error codes

        return val

    #TODO, implement the PvAttrRange* functions
    #def get_property_range(self, name)

    def get_all_properties(self):
        """
        **SUMMARY**
        This returns a dict with the name and current value of the
        documented PvAVT attributes

        CAVEAT: it addresses each of the properties individually, so
        this may take time to run if there's network latency

        **EXAMPLE**
        >>>c = AVTCamera(0)
        >>>props = c.get_all_properties()
        >>>print props['ExposureValue']

        """
        props = {}
        for name in self._properties.keys():
            props[name] = self.get_property(name)
        return props

    def set_property(self, name, value, skip_buffer_size_check=False):
        """
        **SUMMARY**
        This sets the value of the AVT Camera attribute.

        There are around 140 properties for the AVT Camera, so reference the
        AVT Camera and Driver Attributes pdf that is provided with
        the driver for detailed information

        By default, we will also refresh the height/width and bytes per
        frame we're expecting -- you can manually bypass this if you want speed

        Returns the raw PvAVT error code (0 = success)

        **Example**
        >>>c = AVTCamera()
        >>>c.set_property("ExposureValue", 30000)
        >>>c.get_image().show()
        """
        valtype, _ = self._properties.get(name, (None, None))

        if not valtype:
            return None

        if valtype == "Uint32":
            err = self.dll.PvAttrUint32Set(self.handle, name,
                                           ct.c_uint(int(value)))
        elif valtype == "Float32":
            err = self.dll.PvAttrFloat32Set(self.handle, name,
                                            ct.c_float(float(value)))
        elif valtype == "Enum":
            err = self.dll.PvAttrEnumSet(self.handle, name, str(value))
        elif valtype == "String":
            err = self.dll.PvAttrStringSet(self.handle, name, str(value))
        elif valtype == "Boolean":
            err = self.dll.PvAttrBooleanSet(self.handle, name,
                                            ct.c_bool(bool(value)))

        #just to be safe, re-cache the camera metadata
        if not skip_buffer_size_check:
            self._refresh_frame_stats()

        return err

    def get_image(self):
        """
        **SUMMARY**
        Extract an Image from the Camera, returning the value. No matter
        what the image characteristics on the camera, the Image returned
        will be RGB 8 bit depth, if camera is in greyscale mode it will
        be 3 identical channels.

        **EXAMPLE**
        >>>c = AVTCamera()
        >>>c.get_image().show()
        """

        if self.threaded:
            self._thread.lock.acquire()
            try:
                img = self._buffer.pop()
                self._lastimage = img
            except IndexError:
                img = self._lastimage
            self._thread.lock.release()

        else:
            self.run_command("AcquisitionStart")
            frame = self._get_frame()
            img = Factory.Image(PilImage.fromstring(
                self.imgformat, (self.width, self.height),
                frame.image_buffer[:int(frame.image_buffer_size)]))
            self.run_command("AcquisitionStop")

        return img

    def setup_async_mode(self):
        self.set_property('AcquisitionMode', 'SingleFrame')
        self.set_property('FrameStartTriggerMode', 'Software')

    def setup_sync_mode(self):
        self.set_property('AcquisitionMode', 'Continuous')
        self.set_property('FrameStartTriggerMode', 'FreeRun')

    def unbuffer(self):
        img = Factory.Image(PilImage.fromstring(
            self.imgformat, (self.width, self.height),
            self.frame.ImageBuffer[:int(self.frame.ImageBufferSize)]))

        return img

    def _refresh_frame_stats(self):
        self.width = self.get_property("Width")
        self.height = self.get_property("Height")
        self.buffersize = self.get_property("TotalBytesPerFrame")
        self.pixelformat = self.get_property("PixelFormat")
        self.imgformat = 'RGB'
        if self.pixelformat == 'Mono8':
            self.imgformat = 'L'

    def _get_frame(self, timeout=2000):
        #return the AVTFrame object from the camera, timeout in ms
        #need to multiply by bitdepth
        try:
            frame = self.AVTFrame(self.buffersize)
            pverr(self.dll.PvCaptureQueueFrame(self.handle,
                                               ct.byref(frame), None))
            try:
                pverr(self.dll.PvCaptureWaitForFrameDone(self.handle,
                                                         ct.byref(frame),
                                                         timeout))
            except Exception, e:
                print "Exception waiting for frame: ", e
                raise e

        except Exception, e:
            print "Exception aquiring frame: ", e
            raise e

        return frame
