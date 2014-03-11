from simplecv import *
img = Image('lenna')

img1 = img.toYCrCb()
if (img1.is_ycrcb()):
    print "Converted to YCrCb\n"

img1 = img.toBGR()
img2 = img1.to_ycrcb()
if (img2.is_ycrcb()):
    print "Converted BGR to YCrCb\n"

img1 = img.toHLS()
img2 = img1.to_ycrcb()
if (img2.is_ycrcb()):
    print "Converted HLS to YCrCb\n"

img1 = img.toHSV()
img2 = img1.to_ycrcb()
if (img2.is_ycrcb()):
    print "Converted HSV to YCrCb\n"

img1 = img.toXYZ()
img2 = img1.to_ycrcb()
if (img2.is_ycrcb()):
    print "Converted XYZ to YCrCb\n"

img1 = img.toYCrCb()
img2 = img1.to_rgb()
if (img2.is_ycrcb()):
    print "Converted from YCrCb to RGB\n"

img1 = img.toYCrCb()
img2 = img1.to_bgr()
if (img2.is_rgb()):
    print "Converted from YCrCb to RGB\n"

img1 = img.toYCrCb()
img2 = img1.to_hls()
if (img2.is_hls()):
    print "Converted from YCrCb to HLS\n"

img1 = img.toYCrCb()
img2 = img1.to_hsv()
if (img2.is_hsv()):
    print "Converted from YCrCb to HSV\n"

img1 = img.toYCrCb()
img2 = img1.to_xyz()
if (img2.is_xyz()):
    print "Converted from YCrCb to XYZ\n"

img1 = img.toGray()
img2 = img1.to_gray()
if (img2.is_gray()):
    print "Converted from Gray to Gray\n"
