import numpy as np
import cv2
from nose.tools import assert_equals


from simplecv.tests.utils import perform_diff, create_test_image
from simplecv.image import Image
from simplecv.color import Color
from simplecv.core.image.transform import MAX_DIMENSION

testimage = "../data/sampleimages/9dots4lines.png"
testimage2 = "../data/sampleimages/aerospace.jpg"
topimg = "../data/sampleimages/RatTop.png"


def test_image_flip_vertical():
    img = create_test_image()
    img = img.flip_vertical()
    flip_array = np.array([[[255, 0, 0], [255, 255, 255]],
                           [[0, 0, 255], [0, 255, 0]]], dtype=np.uint8)
    assert_equals(flip_array.data, img.get_ndarray().data)


def test_image_flip_horizontal():
    img = create_test_image()
    img = img.flip_horizontal()
    flip_array = np.array([[[0, 255, 0], [0, 0, 255]],
                           [[255, 255, 255], [255, 0, 0]]], dtype=np.uint8)
    assert_equals(flip_array.data, img.get_ndarray().data)


def test_image_resize():
    img = Image(source=testimage)
    thumb = img.resize(30, 30)
    if thumb is None:
        assert False
    result = [thumb]
    name_stem = "test_image_scale"
    perform_diff(result, name_stem)


def test_image_rotate_fixed():
    img = Image(source=testimage2)
    img2 = img.rotate(180, scale=1)
    img3 = img.flip_vertical()
    img4 = img3.flip_horizontal()
    img5 = img.rotate(70)
    img6 = img.rotate(70, scale=0.5)

    results = [img2, img3, img4, img5, img6]
    name_stem = "test_image_rotate_fixed"
    perform_diff(results, name_stem)


def test_image_rotate_full():
    img = Image(source=testimage2)
    img2 = img.rotate(135, False, scale=1)

    results = [img2]
    name_stem = "test_image_rotate_full"
    perform_diff(results, name_stem)


def test_image_shear_warp():
    img = Image(source=testimage2)
    dst = ((img.width / 2, 0), (img.width - 1, img.height / 2),
           (img.width / 2, img.height - 1))
    s = img.shear(dst)

    color = s[0, 0]
    assert color == [0, 0, 0]

    dst = ((img.width * 0.05, img.height * 0.03),
           (img.width * 0.9, img.height * 0.1),
           (img.width * 0.8, img.height * 0.7),
           (img.width * 0.2, img.height * 0.9))
    w = img.warp(dst)

    results = [s, w]
    name_stem = "test_image_shear_warp"
    perform_diff(results, name_stem)


def test_image_affine():
    img = Image(source=testimage2)
    src = ((0, 0), (img.width - 1, 0), (img.width - 1, img.height - 1))
    dst = ((img.width / 2, 0), (img.width - 1, img.height / 2),
           (img.width / 2, img.height - 1))
    a_warp = cv2.getAffineTransform(np.array(src).astype(np.float32),
                                    np.array(dst).astype(np.float32))
    atrans = img.transform_affine(a_warp)

    a_warp2 = np.array(a_warp)
    atrans2 = img.transform_affine(a_warp2)

    results = [atrans, atrans2]
    name_stem = "test_image_affine"
    perform_diff(results, name_stem)


def test_image_perspective():
    img = Image(source=testimage2)
    src = ((0, 0), (img.width - 1, 0), (img.width - 1, img.height - 1),
           (0, img.height - 1))
    dst = ((img.width * 0.05, img.height * 0.03),
           (img.width * 0.9, img.height * 0.1),
           (img.width * 0.8, img.height * 0.7),
           (img.width * 0.2, img.height * 0.9))
    src = np.array(src).astype(np.float32)
    dst = np.array(dst).astype(np.float32)

    p_warp = cv2.getPerspectiveTransform(src, dst)
    ptrans = img.transform_perspective(p_warp)

    p_warp2 = np.array(p_warp)
    ptrans2 = img.transform_perspective(p_warp2)

    results = [ptrans, ptrans2]
    name_stem = "test_image_perspective"
    perform_diff(results, name_stem)


def test_image_crop():
    img = Image(source='simplecv')
    x = 5
    y = 6
    w = 10
    h = 20
    crop = img.crop(x, y, w, h)
    crop2 = img[y:(y + h), x:(x + w)]
    crop6 = img.crop(0, 0, 10, 10)

    crop7 = img.crop(0, 0, -10, 10)
    crop8 = img.crop(-50, -50, 10, 10)
    crop3 = img.crop(-3, -3, 10, 20)
    crop4 = img.crop(-10, 10, 20, 20, centered=True)
    crop5 = img.crop(-10, -10, 20, 20)

    tests = []
    tests.append(img.crop((50, 50), (10, 10)))  # 0
    tests.append(img.crop([10, 10, 40, 40]))  # 1
    tests.append(img.crop((10, 10, 40, 40)))  # 2
    tests.append(img.crop([50, 50], [10, 10]))  # 3
    tests.append(img.crop([10, 10], [50, 50]))  # 4

    roi = np.array([10, 10, 40, 40])
    pts1 = np.array([[50, 50], [10, 10]])
    pts2 = np.array([[10, 10], [50, 50]])
    pt1 = np.array([10, 10])
    pt2 = np.array([50, 50])

    tests.append(img.crop(roi))  # 5
    tests.append(img.crop(pts1))  # 6
    tests.append(img.crop(pts2))  # 7
    tests.append(img.crop(pt1, pt2))  # 8
    tests.append(img.crop(pt2, pt1))  # 9

    xs = [10, 10, 10, 20, 20, 20, 30, 30, 40, 40, 40, 50, 50, 50]
    ys = [10, 20, 50, 20, 30, 40, 30, 10, 40, 50, 10, 50, 10, 42]
    lots = zip(xs, ys)

    tests.append(img.crop(xs, ys))  # 10
    tests.append(img.crop(lots))  # 11
    tests.append(img.crop(np.array(xs), np.array(ys)))  # 12
    tests.append(img.crop(np.array(lots)))  # 14

    i = 0
    failed = False
    for img in tests:
        if img is None or img.width != 40 and img.height != 40:
            print "FAILED CROP TEST " + str(i) + " " + str(img)
            failed = True
        i = i + 1
    assert not failed

    results = [crop, crop2, crop6]
    name_stem = "test_image_crop"
    perform_diff(results, name_stem)


def test_image_region_select():
    img = Image(source='simplecv')
    x1 = 0
    y1 = 0
    x2 = img.width
    y2 = img.height
    crop = img.region_select(x1, y1, x2, y2)

    results = [crop]
    name_stem = "test_image_region_select"
    perform_diff(results, name_stem)


def test_embiggen():
    img = Image(source='simplecv')

    results = []
    w = int(img.width * 1.2)
    h = int(img.height * 1.2)

    results.append(img.embiggen(size=(w, h), color=Color.RED))
    results.append(img.embiggen(size=(w, h), color=Color.RED, pos=(30, 30)))
    results.append(img.embiggen(size=(w, h), color=Color.RED, pos=(-20, -20)))
    results.append(img.embiggen(size=(w, h), color=Color.RED, pos=(30, -20)))
    results.append(img.embiggen(size=(w, h), color=Color.RED, pos=(60, -20)))
    results.append(img.embiggen(size=(w, h), color=Color.RED, pos=(60, 30)))
    results.append(img.embiggen(size=(w, h), color=Color.RED, pos=(80, 80)))
    results.append(img.embiggen(size=(w, h), color=Color.RED, pos=(30, 80)))
    results.append(img.embiggen(size=(w, h), color=Color.RED, pos=(-20, 80)))
    results.append(img.embiggen(size=(w, h), color=Color.RED, pos=(-20, 30)))
    name_stem = "test_embiggen"
    perform_diff(results, name_stem)


def test_apply_side_by_side():
    img = Image(source='simplecv')
    img3 = Image(source=testimage2)

    # LB = little image big image
    # BL = big image little image
    # this is important to test all the possible cases.
    results = []

    results.append(img3.side_by_side(img, side='right', scale=False))
    results.append(img3.side_by_side(img, side='left', scale=False))
    results.append(img3.side_by_side(img, side='top', scale=False))
    results.append(img3.side_by_side(img, side='bottom', scale=False))

    results.append(img.side_by_side(img3, side='right', scale=False))
    results.append(img.side_by_side(img3, side='left', scale=False))
    results.append(img.side_by_side(img3, side='top', scale=False))
    results.append(img.side_by_side(img3, side='bottom', scale=False))

    results.append(img3.side_by_side(img, side='right', scale=True))
    results.append(img3.side_by_side(img, side='left', scale=True))
    results.append(img3.side_by_side(img, side='top', scale=True))
    results.append(img3.side_by_side(img, side='bottom', scale=True))

    results.append(img.side_by_side(img3, side='right', scale=True))
    results.append(img.side_by_side(img3, side='left', scale=True))
    results.append(img.side_by_side(img3, side='top', scale=True))
    results.append(img.side_by_side(img3, side='bottom', scale=True))

    name_stem = "test_apply_side_by_side"
    perform_diff(results, name_stem)


def test_resize():
    img = Image(source=topimg)
    w = img.width
    h = img.height

    img2 = img.resize(w * 2)
    assert_equals(w * 2, img2.width)
    assert_equals(h * 2, img2.height)

    img3 = img.resize(h=h * 2)
    assert_equals(w * 2, img3.width)
    assert_equals(h * 2, img3.height)

    img4 = img.resize(h=h * 2, w=w * 2)
    assert_equals(w * 2, img4.width)
    assert_equals(h * 2, img4.height)

    results = [img2, img3, img4]
    name_stem = "test_image_resize"
    perform_diff(results, name_stem)

    img5 = img.resize(h=MAX_DIMENSION + 1)
    assert img5 is None
