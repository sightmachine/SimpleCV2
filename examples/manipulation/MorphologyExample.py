#!/usr/bin/python
'''
This example rotates through a few of the image morphology examples

for more information see:
http://en.wikipedia.org/wiki/Mathematical_morphology
'''
print __doc__
from simplecv import *

display = Display(resolution = (800, 600)) #create a new display to draw images on
cam = Camera() #initialize the camera
max_threshold = 1 # this is used for the edge detection
threshold_step = 1 # this is the amount to adjust the threshold by each time the display is updated
threshold = max_threshold
example = 1


while display.isNotDone():
    image = cam.getImage().flip_horizontal() # get image (or frame) from camera

    # This just automatically cycles through threshold levels
    if(threshold >= 20):
        threshold = max_threshold
        if(example >= 4):
            example = 1
        else:
            example = example + 1
    else:
        threshold = threshold + threshold_step


    if(example == 1):
        image = image.erode(threshold)
        text = "Erode Morphology Example: img.erode(" + str(threshold) + ")"

    elif(example == 2):
        image = image.dilate(threshold)
        text = "Dilate Morphology Example: img.dilate(" + str(threshold) + ")"

    elif(example == 3):
        image = image.morph_open()
        text = "Open Morphology Example: img.morph_open()"

    elif(example == 4):
        image = image.morph_close()
        text = "Close Morphology Example: img.morph_close()"

    elif(example == 5):
        image = image.morph_gradient()
        text = "Gradient Morphology Example: img.morph_gradient()"

    image.draw_text(text, 10, 10, color=Color.RED, fontsize=30)
    image.show()
