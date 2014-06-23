from simplecv.core.camera.virtual_camera import VirtualCamera
from simplecv.core.image import Image
from nose.tools import assert_equals

def test_camera_VirtualCamera_video():
    vc = VirtualCamera("../data/sampleimages/ball.mov", "video", 10)
    
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

    assert_equals(img.get_ndarray().data, img1.get_ndarray().data)
    assert_equals(img.get_ndarray().data, img2.get_ndarray().data)

def test_camera_VirtualCamera_image():
    vc = VirtualCamera("../data/sampleimages/simplecv.png", "image")
    img = vc.get_image()
    vc.skip_frames(100)
    img1 = vc.get_image()
    vc.rewind()
    img2 = vc.get_image()

    assert_equals(img.get_ndarray().data, img1.get_ndarray().data)
    assert_equals(img.get_ndarray().data, img2.get_ndarray().data)
