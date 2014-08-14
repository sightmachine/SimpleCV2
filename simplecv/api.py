""" These imports are used is simplecv.shell
    also can be used to import all classes: from simplecv.api import *
"""
from simplecv.color import Color, ColorMap
from simplecv.core.camera.avt_camera import AVTCamera
from simplecv.core.camera.camera import Camera
from simplecv.core.camera.digital_camera import DigitalCamera
from simplecv.core.camera.frame_source import FrameSource
from simplecv.core.camera.gige_camera import GigECamera
from simplecv.core.camera.scanner import Scanner
from simplecv.core.camera.screen_camera import ScreenCamera
from simplecv.core.camera.stereo_camera import StereoCamera, StereoImage
from simplecv.core.camera.virtual_camera import VirtualCamera
from simplecv.factory import Factory
from simplecv.features.blobmaker import BlobMaker
from simplecv.features.detection import Line, ROI, Corner, Line, \
    Circle, KeyPoint, KeypointMatch, Motion, TemplateMatch, \
    ShapeContextDescriptor
from simplecv.features.features import FeatureSet, Feature
from simplecv.image import Image
from simplecv.image_set import ImageSet
from simplecv.display import Display
