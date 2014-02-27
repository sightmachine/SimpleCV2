#!/usr/bin/python

import os, sys
from SimpleCV import *
from nose.tools import with_setup


testimage = "../data/sampleimages/9dots4lines.png"
testvideo = "../data/sampleimages/ball.mov"
testoutput = "../data/test/standard/vc.jpg"

def doFullVCamCoverageTest(vcam):
    run = True
    count = 0
    maxf = 1000
    while run:
        img = vcam.getImage()
        vcam.getFrameNumber()
        count = count + 1
        if( img is None or count > maxf ):
            run = False
    vcam.rewind()
    run = True
    while run:
        vcam.skipFrames(2)
        img = vcam.getImage()  
        count = count + 1
        if( img is None or count > maxf ):
            run = False
    return True
    

def test_camera_constructor():
    mycam = VirtualCamera(testimage, "image")
    props = mycam.getAllProperties()

    for i in props.keys():
        print str(i) + ": " + str(props[i]) + "\n"
        
    pass

def test_camera_image():
    mycam = VirtualCamera(testimage, "image")
    if(doFullVCamCoverageTest(mycam)):
        pass
    else:
        assert False

def test_camera_video():
    mycam = VirtualCamera(testvideo, "video")
    img = mycam.getImage()
    img.save(testoutput)
    assert img.size() == (320, 240)
    if(doFullVCamCoverageTest(mycam)):
        pass
    else:
        assert False

def test_camera_iset():
    iset = ImageSet('../data/test/standard/')
    mycam = VirtualCamera(iset, "imageset")
    img = mycam.getImage()
    if(doFullVCamCoverageTest(mycam)):
        pass
    else:
        assert False

def test_camera_iset_directory():
    iset = '../data/test/standard/'
    mycam = VirtualCamera(iset, "imageset")
    img = mycam.getImage()
    if(doFullVCamCoverageTest(mycam)):
        pass
    else:
        assert False
