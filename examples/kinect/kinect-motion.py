#!/usr/bin/python

from simplecv import Kinect, Image, pg, np, time
from simplecv.display import Display

d = Display(flags = pg.FULLSCREEN)
#create video streams

cam = Kinect()
#initialize the camera

depth = cam.getDepth().stretch(0,200)
while True:
    new_depth = cam.getDepth().stretch(0,200)
    img = cam.getImage()
    diff_1 = new_depth - depth
    diff_2 = depth - new_depth
    diff = diff_1 + diff_2
    img_filter = diff.binarize(0)

    motion_img = img - img_filter
    motion_img_open = motion_img.morph_open()
    motion_img_open.show()
    depth = new_depth
