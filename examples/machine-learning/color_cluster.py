'''
This program trys to extract the color pallette from an image
it could be used in machine learning as a color classifier
'''
print __doc__

from simplecv import *
disp = Display((640,528))
cam = Camera()
count = 0
pal = None
while disp.isNotDone():
    img = cam.getImage()
    if count%10 == 0:
        temp = img.scale(.3)
        p = temp.get_palette()
        pal = temp.draw_palette_colors(size=(640,48))
    result = img.re_palette(p)
    result = result.side_by_side(pal,side='bottom')
    result.save(disp)
    count = count + 1
