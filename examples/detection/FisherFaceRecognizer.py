from simplecv import *
import time
"""
This is an example of HOW-TO use FaceRecognizer to recognize gender
of the person.
"""


def identifyGender():
    f = FaceRecognizer()
    cam = Camera()
    img = cam.getImage()
    cascade = LAUNCH_PATH + "/data/Features/HaarCascades/face.xml"
    feat = img.find_haar_features(cascade)
    if feat:
        crop_image = feat.sort_area()[-1].crop()
        feat.sort_area()[-1].draw()

    f.load(LAUNCH_PATH + "/data/Features/FaceRecognizer/GenderData.xml")
    w, h = f.image_size
    crop_image = crop_image.resize(w, h)
    label, confidence = f.predict(crop_image)
    print label
    if label == 0:
        img.draw_text("Female", fontsize=48)

    else:
        img.draw_text("Male", fontsize=48)
    img.show()
    time.sleep(4)

identifyGender()
