import numpy as np
import cv2
from nose.tools import assert_equals, assert_is_none

from simplecv.color import Color
from simplecv.core.image.transform import MAX_DIMENSION
from simplecv.features.detection import Line
from simplecv.image import Image
from simplecv.tests.utils import perform_diff, create_test_image

testimage = "../data/sampleimages/9dots4lines.png"
testimage2 = "../data/sampleimages/aerospace.jpg"

#alpha masking images
topImg = "../data/sampleimages/RatTop.png"
bottomImg = "../data/sampleimages/RatBottom.png"
maskImg = "../data/sampleimages/RatMask.png"
alphaMaskImg = "../data/sampleimages/RatAlphaMask.png"
alphaSrcImg = "../data/sampleimages/GreenMaskSource.png"



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

    # smart crop
    img = Image("simplecv")
    crop_img = img.crop(50, 100, 500, 500, smart=True)
    np_arr = img.get_ndarray()[100:, 50:].copy()
    assert_equals(np_arr.data, crop_img.get_ndarray().data)

    # feature crop
    lines = img.find(Line)
    crop_img = img.crop(lines[0])

    # tuple and list
    np_arr = img.get_ndarray()[10:60, :50].copy()
    crop_img = img.crop(((0, 10), (20, 10), (50, 50), (0, 60)))
    assert_equals(np_arr.data, crop_img.get_ndarray().data)
    crop_img = img.crop([(0, 10), (20, 10), (50, 50), (0, 60)])
    assert_equals(np_arr.data, crop_img.get_ndarray().data)
    crop_img = img.crop((0, 10, 50, 50))
    assert_equals(np_arr.data, crop_img.get_ndarray().data)
    
    # invalid tuple/list
    assert_is_none(img.crop(((0, 10), (20, 10, 20),
                  (50, 50, 30), (0, 60, 40))))
    assert_is_none(img.crop(x=((0, 10), 2, 3, 4, 5), y=(0, 1, 2, 3, 4, 5)))
    assert_is_none(img.crop(x=[((0, 10), 0), 1, 2, 3, 4, 5]))
    assert_is_none(img.crop(x=[(0, 10, 20), (10, 20, 30)]))
    assert_is_none(img.crop(x=(0, 10, 20), y=(20, 30, 40)))
    assert_is_none(img.crop(x=50))

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

    # invalid params
    assert_is_none(img.region_select(0, 10, 10, 10))
    assert_is_none(img.region_select(10, 10, 0, 10))
    assert_is_none(img.region_select(0, 10, img.width+10, 10))
    assert_is_none(img.region_select(0, 10, 10, img.height+20))


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

    # non bgr/gray image
    assert_is_none(img.to_hsv().embiggen(size=(w, h), color=Color.RED))
    assert_is_none(img.embiggen(size=1.2, pos=(1000, 1000)))
    assert_is_none(img.embiggen(size=0.5))


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
    img = Image(source=topImg)
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

    # w/h None
    assert_is_none(img.resize())

def test_image_scale():
    img = Image("simplecv")
    img1 = img.scale(0.5)
    img2 = img.scale(5.0)

    assert_equals(img1.size, (img.width/2, img.height/2))
    assert_equals(img2.size, (img.width*5, img.height*5))

    # large/small scale ratio
    assert_equals(img, img.scale(1000))
    assert_equals(img, img.scale(0.0001))

def test_image_transpose():
    img = Image("lenna")
    new_img = img.resize(w=256, h=512)
    t_img = new_img.transpose()
    assert_equals(t_img.size, (512, 256))

def test_image_split():
    img = Image("lenna")
    splits = img.split(8, 4)
    assert_equals(len(splits), 4)
    assert_equals(len(splits[0]), 8)

    np_array = img.get_ndarray()

    row = 0
    col = 0
    for split in splits:
        col = 0
        for split_img in split:
            assert_equals(split_img.size, (64, 128))
            np_arr = np_array[row:row+128, col:col+64].copy()
            assert_equals(split_img.get_ndarray().data, np_arr.data)
            col += 64
        row += 128

def test_image_adaptive_scale():
    img = Image("simplecv")
    w, h = img.size

    new_img = img.adaptive_scale(img.size) # no resize
    assert_equals(new_img, img)
    new_img = img.adaptive_scale((img.width/2, img.height/2))
    
    new_img = img.adaptive_scale((img.width/3, img.height/4))
    assert_equals(new_img.size, (img.width/3, img.height/4))

    new_img = img.adaptive_scale((img.width*1.1, img.height*1.2))
    assert_equals(new_img.size, (img.width*1.1, img.height*1.2))

    new_img = img.adaptive_scale((img.width*1.1, img.height*0.3))
    assert_equals(new_img.size, (img.width*1.1, img.height*0.3))

    new_img = img.adaptive_scale((img.width*0.5, img.height*1.2))
    assert_equals(new_img.size, (img.width*0.5, img.height*1.2))

    new_img = img.adaptive_scale((img.width/3, img.height/4), fit=False)
    assert_equals(new_img.size, (img.width/3, img.height/4))

    new_img = img.adaptive_scale((img.width*1.1, img.height*1.2), fit=False)
    assert_equals(new_img.size, (img.width*1.1, img.height*1.2))

    new_img = img.adaptive_scale((img.width*1.1, img.height*0.3), fit=False)
    assert_equals(new_img.size, (img.width*1.1, img.height*0.3))

    new_img = img.adaptive_scale((img.width*0.5, img.height*1.2), fit=False)
    assert_equals(new_img.size, (img.width*0.5, img.height*1.2))

def test_blit_regular():
    top = Image(topImg)
    bottom = Image(bottomImg)
    results = []
    results.append(bottom.blit(top))
    results.append(bottom.blit(top, pos=(-10, -10)))
    results.append(bottom.blit(top, pos=(-10, 10)))
    results.append(bottom.blit(top, pos=(10, -10)))
    results.append(bottom.blit(top, pos=(10, 10)))

    name_stem = "test_blit_regular"
    perform_diff(results, name_stem)


def test_blit_mask():
    top = Image(topImg)
    bottom = Image(bottomImg)
    mask = Image(maskImg)
    results = []
    results.append(bottom.blit(top, mask=mask))
    results.append(bottom.blit(top, mask=mask, pos=(-50, -50)))
    results.append(bottom.blit(top, mask=mask, pos=(-50, 50)))
    results.append(bottom.blit(top, mask=mask, pos=(50, -50)))
    results.append(bottom.blit(top, mask=mask, pos=(50, 50)))

    name_stem = "test_blit_mask"
    perform_diff(results, name_stem)

    # mask size mismatch
    assert_is_none(bottom.blit(top, mask=mask.resize(mask.width/2)))


def test_blit_alpha():
    top = Image(topImg)
    bottom = Image(bottomImg)
    a = 0.5
    results = []
    results.append(bottom.blit(top, alpha=a))
    results.append(bottom.blit(top, alpha=a, pos=(-50, -50)))
    results.append(bottom.blit(top, alpha=a, pos=(-50, 50)))
    results.append(bottom.blit(top, alpha=a, pos=(50, -50)))
    results.append(bottom.blit(top, alpha=a, pos=(50, 50)))
    name_stem = "test_blit_alpha"
    perform_diff(results, name_stem)


def test_blit_alpha_mask():
    top = Image(topImg)
    bottom = Image(bottomImg)
    a_mask = Image(alphaMaskImg)
    results = []

    results.append(bottom.blit(top, alpha_mask=a_mask))
    results.append(bottom.blit(top, alpha_mask=a_mask, pos=(-10, -10)))
    results.append(bottom.blit(top, alpha_mask=a_mask, pos=(-10, 10)))
    results.append(bottom.blit(top, alpha_mask=a_mask, pos=(10, -10)))
    results.append(bottom.blit(top, alpha_mask=a_mask, pos=(10, 10)))

    name_stem = "test_blit_alpha_mask"
    perform_diff(results, name_stem)

    # alphas mask size mismatch
    assert_is_none(bottom.blit(top, alpha_mask=a_mask.resize(a_mask.width/2)))