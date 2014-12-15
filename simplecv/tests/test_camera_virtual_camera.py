import os

from nose.tools import assert_equals

from simplecv.core.camera.virtual_camera import VirtualCamera
from simplecv import DATA_DIR

MOV_PATH = os.path.join(DATA_DIR, 'sampleimages/ball.mov')
SIMPLECV_PATH = os.path.join(DATA_DIR, 'sampleimages/simplecv.png')


def test_camera_virtualcamera_video():
    vc = VirtualCamera(MOV_PATH, "video", 10)

    assert_equals(vc.get_frame_number(), 9)
    assert_equals(vc.get_current_play_time(), 300)

    img = vc.get_image()

    vc.rewind(start=0)
    assert_equals(vc.get_frame_number(), 0)

    vc.skip_frames(10)
    assert_equals(vc.get_frame_number(), 9)

    img1 = vc.get_image()

    vc.skip_frames(100)
    img2 = vc.get_frame(10)

    assert_equals(img.data, img1.data)
    assert_equals(img.data, img2.data)


def test_camera_virtualcamera_image():
    vc = VirtualCamera(SIMPLECV_PATH, "image")
    img = vc.get_image()
    vc.skip_frames(100)
    img1 = vc.get_image()
    vc.rewind()
    img2 = vc.get_image()

    assert_equals(img.data, img1.data)
    assert_equals(img.data, img2.data)
