from nose.tools import assert_equals, assert_is_none, assert_is_not_none

from simplecv.image import Image
from simplecv.image_set import ImageSet
from simplecv.tracking.mf_tracker import mfTracker
from simplecv.tracking.surf_tracker import surfTracker
from simplecv.tracking.track_class import Track
import time

iset = []
iset.append(Image("./../data/sampleimages/tracktest0.jpg"))
iset.append(Image("./../data/sampleimages/tracktest1.jpg"))
iset.append(Image("./../data/sampleimages/tracktest2.jpg"))
iset.append(Image("./../data/sampleimages/tracktest3.jpg"))
iset.append(Image("./../data/sampleimages/tracktest4.jpg"))
iset.append(Image("./../data/sampleimages/tracktest5.jpg"))
iset.append(Image("./../data/sampleimages/tracktest6.jpg"))
iset.append(Image("./../data/sampleimages/tracktest7.jpg"))
iset.append(Image("./../data/sampleimages/tracktest8.jpg"))
iset.append(Image("./../data/sampleimages/tracktest9.jpg"))


def test_tracking_mf_tracker():
    bb = (190, 152, 70, 70)
    ts = []

    for i in range(len(iset) - 1):
        img = iset[i+1]
        pImg = iset[i]
        ts = mfTracker(img, bb, ts, pImg, numM=15,
                       numN=15, winsize=25, winsize_lk=25, margin=10)
        assert_is_not_none(ts)
        assert_is_not_none(ts.shift)
        assert_equals(ts.getImage(), img)
        assert_is_not_none(ts.getBB())


def test_tracking_surf_tracker():
    bb = (120, 150, 130, 80)
    ts = []

    for i in range(len(iset)):
        img = iset[i]
        pImg = iset[i]
        surfts = surfTracker(img, bb, ts, eps_val=0.69,
                       min_samples=4, distnace=200)
        ts.append(surfts)
    assert_is_not_none(ts)
    assert_is_not_none(surfts.getTrackedPoints())
    assert_is_not_none(surfts.getDetector())
    assert_is_not_none(surfts.getDescriptor())
    assert_is_not_none(surfts.getImageKeyPoints())
    assert_is_not_none(surfts.getImageDescriptor())
    assert_is_not_none(surfts.getTemplateKeyPoints())
    assert_is_not_none(surfts.getTemplateDescriptor())
    assert_is_not_none(surfts.getTemplateImage())

def test_tracking_camshift_tracker():
    ts = []
    bb = (195, 160, 49, 46)
    imgs = iset
    ts = imgs[0].track("camshift", ts, imgs[1:], bb)
    assert ts


def test_tracking_lk_tracker():
    ts = []
    bb = (195, 160, 49, 46)
    imgs = iset
    ts = imgs[0].track("LK", ts, imgs[1:], bb)
    assert ts


def test_tracking_Track_test():
    img = Image("simplecv")
    bb = [20, 40, 60, 50]

    class track_test(Track):
        def __init__(self, img, bb):
            self = Track.__init__(self, img, bb)

    def meanc(img):
        return img.mean_color()

    tr = track_test(img, bb)

    assert_equals(tr.getCenter(), (50, 65))
    assert_equals(tr.get_area(), 3000)
    assert_equals(tr.getImage(), img)
    assert_equals(tr.getBB(), bb)
    assert_equals(tr.processTrack(meanc), img.mean_color())
    tr.draw()
    tr.drawBB()
    tr.showCoordinates()
    tr.showSizeRatio()
    tr.showPixelVelocity()
    tr.showPixelVelocityRT()
    tr.drawPredicted()
    tr.showPredictedCoordinates()
    tr.showCorrectedCoordinates()
    tr.drawCorrected()