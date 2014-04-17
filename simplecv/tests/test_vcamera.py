import os
import tempfile

from nose.tools import nottest, assert_is_instance, assert_tuple_equal

from simplecv import DATA_DIR
from simplecv.core.camera.virtual_camera import VirtualCamera
from simplecv.image_set import ImageSet
from simplecv.image import Image


testimage = os.path.join(DATA_DIR, 'sampleimages/9dots4lines.png')
testvideo = os.path.join(DATA_DIR, 'sampleimages/ball.mov')
testoutput = os.path.join(tempfile.gettempdir(), 'vc.jpg')


@nottest
def do_full_vcam_coverage_test(vcam):
    maxf = 100
    for i in range(0, maxf):
        img = vcam.get_image()
        vcam.get_frame_number()
        if img is None:
            break

    vcam.rewind()

    for i in range(0, maxf):
        vcam.skip_frames(2)
        img = vcam.get_image()
        if img is None:
            break


def test_camera_constructor():
    mycam = VirtualCamera(testimage, "image")
    props = mycam.get_all_properties()

    for i in props.keys():
        print str(i) + ": " + str(props[i]) + "\n"


def test_camera_image():
    mycam = VirtualCamera(testimage, "image")
    do_full_vcam_coverage_test(mycam)


def test_camera_video():
    mycam = VirtualCamera(testvideo, "video")
    img = mycam.get_image()
    img.save(testoutput)

    assert_is_instance(img, Image)
    assert_tuple_equal((320, 240), img.size)
    do_full_vcam_coverage_test(mycam)


def test_camera_iset():
    iset = ImageSet(os.path.join(DATA_DIR, 'test/animation'))
    mycam = VirtualCamera(iset, "imageset")
    img = mycam.get_image()

    assert_is_instance(img, Image)
    do_full_vcam_coverage_test(mycam)


def test_camera_iset_directory():
    path = os.path.join(DATA_DIR, 'test/animation')
    mycam = VirtualCamera(path, "imageset")
    img = mycam.get_image()

    assert_is_instance(img, Image)
    do_full_vcam_coverage_test(mycam)
