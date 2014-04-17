from nose.tools import assert_list_equal, assert_equal
from simplecv.features.facerecognizer import FaceRecognizer
from simplecv.image import Image
from simplecv.base import LAUNCH_PATH
import os

def test_facerecognizer():

    
    images3 = ["../data/sampleimages/fi1.jpg",
               "../data/sampleimages/fi2.jpg",
               "../data/sampleimages/fi3.jpg",
               "../data/sampleimages/fi4.jpg"]

    
    imgset3 = []

    for img in images1:
        imgset1.append(Image(img))
    label1 = ["female"] * len(imgset1)

    for img in images2:
        imgset2.append(Image(img))
    label2 = ["male"] * len(imgset2)

    imgset = imgset1 + imgset2
    labels = label1 + label2
    imgset[4] = imgset[4].resize(400, 400)
    f.train(imgset, labels)

    for img in images3:
        imgset3.append(Image(img))
    imgset[2].resize(300, 300)
    label = []
    for img in imgset3:
        name, confidence = f.predict(img)
        label.append(name)

    assert_list_equal(["male", "male", "female", "female"], label)

def test_facerecognizer_train():
    images1 = ["../data/sampleimages/ff1.jpg",
               "../data/sampleimages/ff2.jpg",
               "../data/sampleimages/ff3.jpg",
               "../data/sampleimages/ff4.jpg",
               "../data/sampleimages/ff5.jpg"]

    images2 = ["../data/sampleimages/fm1.jpg",
               "../data/sampleimages/fm2.jpg",
               "../data/sampleimages/fm3.jpg",
               "../data/sampleimages/fm4.jpg",
               "../data/sampleimages/fm5.jpg"]

    images3 = ["../data/sampleimages/fi1.jpg",
               "../data/sampleimages/fi2.jpg",
               "../data/sampleimages/fi3.jpg",
               "../data/sampleimages/fi4.jpg"]

    imgset1 = []
    imgset2 = []
    imgset3 = []
    label = []

    for img in images1:
        imgset1.append(Image(img))
    label1 = ["female"] * len(imgset1)

    for img in images2:
        imgset2.append(Image(img))
    label2 = ["male"] * len(imgset2)

    imgset = imgset1 + imgset2
    labels = label1 + label2
    imgset[4] = imgset[4].resize(400, 400)

    for img in images3:
        imgset3.append(Image(img))

    f = FaceRecognizer()
    trained = f.train(csvfile="../data/test/standard/test_facerecognizer_train_data.csv",
            delimiter=",")

    for img in imgset3:
        name, confidence = f.predict(img)
        label.append(name)

    assert_equal(trained, True)
    assert_list_equal(["male", "male", "female", "female"], label)

    fr1 = FaceRecognizer()
    trained = fr1.train(csvfile="no_such_file.csv")
    assert_equal(trained, False)

    fr2 = FaceRecognizer()
    trained = fr2.train(imgset, labels)
    assert_equal(trained, True)
    
    label = []
    for img in imgset3:
        name, confidence = fr2.predict(img)
        label.append(name)
    assert_list_equal(["male", "male", "female", "female"], label)

    fr3 = FaceRecognizer()
    trained = fr3.train(imgset1, label1)
    assert_equal(trained, False)

    fr4 = FaceRecognizer()
    trained = fr4.train(imgset, label2)
    assert_equal(trained, False)

    prediction = fr4.predict(imgset3[0])
    assert_equal(prediction, None)

def test_facerecognizer_load():
    f = FaceRecognizer()
    filename = os.path.join(LAUNCH_PATH, "data", "Features", "FaceRecognizer",
                            "GenderData.xml")

    trained = f.load(filename)
    images3 = ["../data/sampleimages/ff1.jpg",
               "../data/sampleimages/ff5.jpg",
               "../data/sampleimages/fm3.jpg",
               "../data/sampleimages/fm4.jpg"]

    imgset3 = []
    label = []

    for img in images3:
        imgset3.append(Image(img))

    label = []
    for img in imgset3:
        name, confidence = f.predict(img)
        label.append(name)

    assert_list_equal([0, 0, 1, 1], label)

    fr1 = FaceRecognizer()
    trained = fr1.load("no_such_file.xml")
    assert_equal(trained, False)

    prediction = fr1.predict(imgset3[0])
    assert_equal(prediction, None)

def test_facerecognizer_save():
    images1 = ["../data/sampleimages/ff1.jpg",
               "../data/sampleimages/ff2.jpg",
               "../data/sampleimages/ff3.jpg",
               "../data/sampleimages/ff4.jpg",
               "../data/sampleimages/ff5.jpg"]

    images2 = ["../data/sampleimages/fm1.jpg",
               "../data/sampleimages/fm2.jpg",
               "../data/sampleimages/fm3.jpg",
               "../data/sampleimages/fm4.jpg",
               "../data/sampleimages/fm5.jpg"]

    images3 = ["../data/sampleimages/fi1.jpg",
               "../data/sampleimages/fi2.jpg",
               "../data/sampleimages/fi3.jpg",
               "../data/sampleimages/fi4.jpg"]

    imgset1 = []
    imgset2 = []

    for img in images1:
        imgset1.append(Image(img))
    label1 = ["female"] * len(imgset1)

    for img in images2:
        imgset2.append(Image(img))
    label2 = ["male"] * len(imgset2)

    imgset = imgset1 + imgset2
    labels = label1 + label2

    f = FaceRecognizer()
    trained = f.train(imgset, labels)

    filename = os.path.join(LAUNCH_PATH, "tests", "gendertrain.xml")
    if (trained):
        saved = f.save(filename)
        assert_equal(saved, True)

        if not os.path.exists(os.path.abspath(filename)):
            assert False

        os.remove(filename)