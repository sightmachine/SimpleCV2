import os
import tempfile

from nose.tools import nottest

from simplecv.camera import VirtualCamera
from simplecv.image_set import ImageSet


testimage = "../data/sampleimages/9dots4lines.png"
testvideo = "../data/sampleimages/ball.mov"
testoutput = os.path.join(tempfile.gettempdir(), 'vc.jpg')


@nottest
def do_full_vcam_coverage_test(vcam):
    run = True
    count = 0
    maxf = 1000
    while run:
        img = vcam.get_image()
        vcam.get_frame_number()
        count = count + 1
        if img is None or count > maxf:
            run = False
    vcam.rewind()
    run = True
    while run:
        vcam.skip_frames(2)
        img = vcam.get_image()
        count = count + 1
        if img is None or count > maxf:
            run = False
    return True


def test_camera_constructor():
    mycam = VirtualCamera(testimage, "image")
    props = mycam.get_all_properties()

    for i in props.keys():
        print str(i) + ": " + str(props[i]) + "\n"


def test_camera_image():
    mycam = VirtualCamera(testimage, "image")

    assert do_full_vcam_coverage_test(mycam)


def test_camera_video():
    mycam = VirtualCamera(testvideo, "video")
    img = mycam.get_image()
    img.save(testoutput)

    assert img is not None
    assert img.size == (320, 240)
    assert do_full_vcam_coverage_test(mycam)


def test_camera_iset():
    iset = ImageSet('../data/test/standard/')
    mycam = VirtualCamera(iset, "imageset")
    img = mycam.get_image()

    assert img is not None
    assert do_full_vcam_coverage_test(mycam)


def test_camera_iset_directory():
    iset = '../data/test/standard/'
    mycam = VirtualCamera(iset, "imageset")
    img = mycam.get_image()

    assert img is not None
    assert do_full_vcam_coverage_test(mycam)
