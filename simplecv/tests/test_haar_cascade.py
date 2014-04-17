import os
import tempfile

from nose.tools import assert_equals

from simplecv.image import Image
from simplecv.features.haar_cascade import HaarCascade
from simplecv import DATA_DIR

FACECASCADE = 'face.xml'

testimage = "../data/sampleimages/orson_welles.jpg"
testoutput = os.path.join(tempfile.gettempdir(), 'orson_welles_face.jpg')

testneighbor_in = "../data/sampleimages/04000.jpg"
testneighbor_out = "../data/sampleimages/04000_face.jpg"


def test_haarcascade():
    img = Image(testimage)
    faces = img.find_haar_features(FACECASCADE)

    if faces:
        faces.draw()
        img.save(testoutput)
    else:
        assert False

    cascade = HaarCascade(FACECASCADE, "face_cascade")
    assert_equals(cascade.get_name(), "face_cascade")
    
    fhandle = os.path.join(DATA_DIR, "Features", "HaarCascades", "face.xml")
    assert_equals(cascade.get_fhandle(), fhandle)

    cascade.set_name("eye_cascade")
    assert_equals(cascade.get_name(), "eye_cascade")

    new_fhandle = os.path.join(DATA_DIR, "Features", "HaarCascades", "eye.xml") 
    cascade.load(new_fhandle)
    assert_equals(cascade.get_fhandle(), new_fhandle)

    emptycascade = HaarCascade()

def test_minneighbors(img_in=testneighbor_in, img_out=testneighbor_out):
    img = Image(img_in)
    faces = img.find_haar_features(FACECASCADE, min_neighbors=20)
    if faces:
        faces.draw()
        img.save(img_out)
    assert len(faces) <= 1, "Haar Cascade is potentially ignoring the " \
                            "'HIGH' min_neighbors of 20"
