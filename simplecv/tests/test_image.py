import cv2
from nose.tools import assert_equals, nottest, raises
import numpy as np

from simplecv.image_class import Image, ColorSpace

LENNA_PATH = '../data/sampleimages/lenna.png'
WEBP_IMAGE_PATH = '../data/sampleimages/simplecv.webp'


@nottest
def create_test_array():
    """ Returns array 2 x 2 pixels, 8 bit and BGR color space
        pixels are colored so:
        RED, GREEN
        BLUE, WHITE
    """
    return np.array([[[0, 0, 255], [0, 255, 0]],       # RED,  GREEN
                     [[255, 0, 0], [255, 255, 255]]],  # BLUE, WHITE
                    dtype=np.uint8)


@nottest
def create_test_image():
    bgr_array = create_test_array()
    return Image(bgr_array, color_space=ColorSpace.BGR)


def test_image_init_path_to_png():
    img1 = Image("lenna")
    assert_equals((512, 512), img1.size())
    assert_equals(ColorSpace.BGR, img1.get_color_space())
    assert img1.is_bgr()

    img_ndarray = img1.get_ndarray()

    assert isinstance(img_ndarray, np.ndarray)
    assert_equals(3, len(img_ndarray.shape))

    assert not img1.is_rgb()
    assert not img1.is_hsv()
    assert not img1.is_hls()
    assert not img1.is_ycrcb()
    assert not img1.is_xyz()
    assert not img1.is_gray()


def test_image_init_path_to_webp():
    img = Image(WEBP_IMAGE_PATH)

    assert_equals((250, 250), img.size())
    assert img.is_rgb()
    img_ndarray = img.get_ndarray()
    assert isinstance(img_ndarray, np.ndarray)
    assert_equals(3, len(img_ndarray.shape))
    assert_equals(np.uint8, img.dtype)


@raises(Exception)
def test_image_init_bad_path():
    Image('/bad/path/to/image.png')


def test_image_init_ndarray_color():
    color_ndarray = cv2.imread(LENNA_PATH)
    img1 = Image(color_ndarray, color_space=ColorSpace.BGR)

    assert_equals((512, 512), img1.size())
    assert_equals(ColorSpace.BGR, img1.get_color_space())
    assert img1.is_bgr()

    img_ndarray = img1.get_ndarray()

    assert isinstance(img_ndarray, np.ndarray)
    assert_equals(3, len(img_ndarray.shape))


def test_image_init_ndarray_grayscale():
    ndarray = cv2.imread(LENNA_PATH)
    gray_ndarray = cv2.cvtColor(ndarray, cv2.COLOR_BGR2GRAY)

    img1 = Image(gray_ndarray)
    assert_equals((512, 512), img1.size())
    assert_equals(ColorSpace.GRAY, img1.get_color_space())
    assert img1.is_gray()

    img_ndarray = img1.get_ndarray()

    assert isinstance(img_ndarray, np.ndarray)
    assert_equals(2, len(img_ndarray.shape))


def test_image_numpy_constructor():
    img = Image(LENNA_PATH)
    grayimg = img.to_gray()

    chan3_array = np.array(img.get_ndarray())
    chan1_array = np.array(img.get_gray_ndarray())

    img2 = Image(chan3_array)
    grayimg2 = Image(chan1_array)

    assert img2[0, 0] == img[0, 0]
    assert grayimg2[0, 0] == grayimg[0, 0]


def test_image_loadsave():
    img = Image(testimage)
    img.save(testoutput)
    if os.path.isfile(testoutput):
        os.remove(testoutput)
    else:
        assert False


def test_image_init_tuple_bgr():
    img1 = Image([5, 10], color_space=ColorSpace.BGR)
    assert img1.is_bgr()
    assert_equals((5, 10), img1.size())
    assert_equals(np.zeros((5, 10, 3), np.uint8).data, img1.get_ndarray().data)


def test_image_init_tuple_gray():
    img1 = Image([5, 10], color_space=ColorSpace.GRAY)
    assert img1.is_gray()
    assert_equals((5, 10), img1.size())
    assert_equals(np.zeros((5, 10), np.uint8).data, img1.get_ndarray().data)


def test_image_convert_bgr_to_bgr():
    bgr_array = create_test_array()
    result_bgr_array = Image.convert(bgr_array, ColorSpace.BGR, ColorSpace.BGR)
    assert bgr_array is not result_bgr_array
    assert_equals(bgr_array.data, result_bgr_array.data)


def test_image_bgr_to_rbg_to_bgr():
    bgr_img = create_test_image()
    rgb_img = bgr_img.to_rgb()
    rgb_array = np.array([[[255, 0, 0], [0, 255, 0]],
                          [[0, 0, 255], [255, 255, 255]]], dtype=np.uint8)
    assert_equals(rgb_array.data, rgb_img.get_ndarray().data)
    assert rgb_img.is_rgb()

    new_bgr_img = rgb_img.to_bgr()

    assert_equals(bgr_img.get_ndarray().data, new_bgr_img.get_ndarray().data)
    assert new_bgr_img.is_bgr()


def test_image_bgr_to_gray():
    bgr_img = create_test_image()
    gray_img = bgr_img.to_gray()
    gray_array = np.array([[76, 150],
                           [29, 255]], dtype=np.uint8)
    assert_equals(gray_array.data, gray_img.get_ndarray().data)
    assert gray_img.is_gray()


def test_image_bgr_to_hsv():
    bgr_img = create_test_image()
    hsv_img = bgr_img.to_hsv()
    hsv_array = np.array([[[0, 255, 255], [60, 255, 255]],
                          [[120, 255, 255], [0, 0, 255]]], dtype=np.uint8)
    assert_equals(hsv_array.data, hsv_img.get_ndarray().data)
    assert hsv_img.is_hsv()


def test_image_bgr_to_ycrcb():
    bgr_img = create_test_image()
    ycrcb_img = bgr_img.to_ycrcb()
    ycrcb_array = np.array([[[76, 255, 85], [150, 21, 43]],
                            [[29, 107, 255], [255, 128, 128]]], dtype=np.uint8)
    assert_equals(ycrcb_array.data, ycrcb_img.get_ndarray().data)
    assert ycrcb_img.is_ycrcb()


def test_image_bgr_to_zyx():
    bgr_img = create_test_image()
    xyz_img = bgr_img.to_xyz()
    xyz_array = np.array([[[105, 54, 5], [91, 182, 30]],
                          [[46, 18, 242], [242, 255, 255]]], dtype=np.uint8)
    assert_equals(xyz_array.data, xyz_img.get_ndarray().data)
    assert xyz_img.is_xyz()


def test_image_bgr_to_hls():
    bgr_img = create_test_image()
    hls_img = bgr_img.to_hls()
    hls_array = np.array([[[0, 128, 255], [60, 128, 255]],
                          [[120, 128, 255], [0, 255, 0]]], dtype=np.uint8)
    assert_equals(hls_array.data, hls_img.get_ndarray().data)
    assert hls_img.is_hls()


def test_image_hsv_to_gray():
    hsv_array = np.array([[[0, 255, 255], [60, 255, 255]],
                          [[120, 255, 255], [0, 0, 255]]], dtype=np.uint8)
    hsv_img = Image(hsv_array, color_space=ColorSpace.HSV)
    gray_array = np.array([[76, 150],
                           [29, 255]], dtype=np.uint8)
    gray_img = hsv_img.to_gray()

    assert_equals(gray_array.data, gray_img.get_ndarray().data)
    assert gray_img.is_gray()


def test_image_copy():
    img = create_test_image()
    copy_img = img.copy()

    assert img is not copy_img
    assert_equals(img.size(), copy_img.size())
    assert_equals(img.get_ndarray().data, copy_img.get_ndarray().data)
    assert_equals(img.get_color_space(), copy_img.get_color_space())


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


def test_image_operator_wrong_size():
    array1 = np.ones((2, 2, 3), dtype=np.uint8) * 5
    img1 = Image(array1)
    array2 = np.ones((2, 1, 3), dtype=np.uint8) * 5
    img2 = Image(array2)

    img = img1 - img2
    assert_equals(None, img)
    img = img1 + img2
    assert_equals(None, img)
    img = img1 & img2
    assert_equals(None, img)
    img = img1 | img2
    assert_equals(None, img)
    img = img1 / img2
    assert_equals(None, img)
    img = img1 * img2
    assert_equals(None, img)
    img = img1.max(img2)
    assert_equals(None, img)
    img = img1.min(img2)
    assert_equals(None, img)


def test_image_sub_image():
    array1 = np.ones((2, 2, 3), dtype=np.uint8) * 5
    img1 = Image(array1)
    array2 = np.ones((2, 2, 3), dtype=np.uint8) * 5
    img2 = Image(array2)
    array = np.zeros((2, 2, 3), dtype=np.uint8)

    img = img1 - img2
    assert_equals(array.data, img.get_ndarray().data)


def test_image_sub_int():
    array1 = np.ones((2, 2, 3), dtype=np.uint8) * 5
    img1 = Image(array1)
    array = np.zeros((2, 2, 3), dtype=np.uint8)

    img = img1 - 5
    assert_equals(array.data, img.get_ndarray().data)


def test_image_add_image():
    array1 = np.ones((2, 2, 3), dtype=np.uint8) * 5
    img1 = Image(array1)
    array2 = np.ones((2, 2, 3), dtype=np.uint8) * 5
    img2 = Image(array2)
    array = np.ones((2, 2, 3), dtype=np.uint8) * 10

    img = img1 + img2
    assert_equals(array.data, img.get_ndarray().data)


def test_image_add_int():
    array1 = np.ones((2, 2, 3), dtype=np.uint8) * 5
    img1 = Image(array1)

    array = np.ones((2, 2, 3), dtype=np.uint8) * 10

    img = img1 + 5
    assert_equals(array.data, img.get_ndarray().data)


def test_image_and_image():
    array1 = np.ones((2, 2, 3), dtype=np.uint8) * 5
    img1 = Image(array1)
    array2 = np.ones((2, 2, 3), dtype=np.uint8) * 3
    img2 = Image(array2)
    array = np.ones((2, 2, 3), dtype=np.uint8)

    img = img1 & img2
    assert_equals(array.data, img.get_ndarray().data)


def test_image_and_int():
    array1 = np.ones((2, 2, 3), dtype=np.uint8) * 5
    img1 = Image(array1)
    array = np.ones((2, 2, 3), dtype=np.uint8)

    img = img1 & 3
    assert_equals(array.data, img.get_ndarray().data)


def test_image_or_image():
    array1 = np.ones((2, 2, 3), dtype=np.uint8) * 5
    img1 = Image(array1)
    array2 = np.ones((2, 2, 3), dtype=np.uint8) * 3
    img2 = Image(array2)
    array = np.ones((2, 2, 3), dtype=np.uint8) * 7

    img = img1 | img2
    assert_equals(array.data, img.get_ndarray().data)


def test_image_or_int():
    array1 = np.ones((2, 2, 3), dtype=np.uint8) * 5
    img1 = Image(array1)
    array = np.ones((2, 2, 3), dtype=np.uint8) * 7

    img = img1 | 3
    assert_equals(array.data, img.get_ndarray().data)


def test_image_div_image():
    array1 = np.ones((2, 2, 3), dtype=np.uint8) * 50
    img1 = Image(array1)
    array2 = np.ones((2, 2, 3), dtype=np.uint8) * 2
    img2 = Image(array2)
    array = np.ones((2, 2, 3), dtype=np.uint8) * 25

    img = img1 / img2
    assert_equals(array.data, img.get_ndarray().data)


def test_image_div_int_float():
    array1 = np.ones((2, 2, 3), dtype=np.uint8) * 50
    img1 = Image(array1)
    array = np.ones((2, 2, 3), dtype=np.uint8) * 25

    img = img1 / 2
    assert_equals(array.data, img.get_ndarray().data)

    array = np.ones((2, 2, 3), dtype=np.uint8) * 20
    img = img1 / 2.5
    assert_equals(array.data, img.get_ndarray().data)


def test_image_multiply_image():
    array1 = np.ones((2, 2, 3), dtype=np.uint8) * 50
    img1 = Image(array1)
    array2 = np.ones((2, 2, 3), dtype=np.uint8) * 2
    img2 = Image(array2)
    array = np.ones((2, 2, 3), dtype=np.uint8) * 100

    img = img1 * img2
    assert_equals(array.data, img.get_ndarray().data)


def test_image_multiply_int_float():
    array1 = np.ones((2, 2, 3), dtype=np.uint8) * 50
    img1 = Image(array1)
    array = np.ones((2, 2, 3), dtype=np.uint8) * 100

    img = img1 * 2
    assert_equals(array.data, img.get_ndarray().data)

    array = np.ones((2, 2, 3), dtype=np.uint8) * 125
    img = img1 * 2.5
    assert_equals(array.data, img.get_ndarray().data)


@raises(ValueError)
def test_image_pow_image():
    array1 = np.ones((2, 2, 3), dtype=np.uint8) * 50
    img1 = Image(array1)
    array2 = np.ones((2, 2, 3), dtype=np.uint8) * 2
    img2 = Image(array2)
    img = img1 ** img2


def test_image_pow_int_float():
    array1 = np.ones((2, 2, 3), dtype=np.uint8) * 5
    img1 = Image(array1)
    array = np.ones((2, 2, 3), dtype=np.uint8) * 255

    img = img1 ** 20
    assert_equals(array.data, img.get_ndarray().data)

    img = img1 ** 20
    assert_equals(array.data, img.get_ndarray().data)


def test_image_neg_invert():
    array1 = np.ones((2, 2, 3), dtype=np.uint8) * 5
    img1 = Image(array1)
    array = np.ones((2, 2, 3), dtype=np.uint8) * 250

    img = ~img1
    assert_equals(array.data, img.get_ndarray().data)
    img = -img1
    assert_equals(array.data, img.get_ndarray().data)


def test_image_max_int():
    array1 = np.ones((2, 2, 3), dtype=np.uint8) * 5
    img1 = Image(array1)
    array = np.ones((2, 2, 3), dtype=np.uint8) * 20

    img = img1.max(20)
    assert_equals(array.data, img.get_ndarray().data)


def test_image_max_image():
    array1 = np.ones((2, 2, 3), dtype=np.uint8) * 2
    img1 = Image(array1)
    array2 = np.ones((2, 2, 3), dtype=np.uint8) * 3
    img2 = Image(array2)
    array = np.ones((2, 2, 3), dtype=np.uint8) * 3

    img = img1.max(img2)
    assert_equals(array.data, img.get_ndarray().data)


def test_image_min_int():
    array1 = np.ones((2, 2, 3), dtype=np.uint8) * 5
    img1 = Image(array1)
    array = np.ones((2, 2, 3), dtype=np.uint8) * 5

    img = img1.min(20)
    assert_equals(array.data, img.get_ndarray().data)


def test_image_min_image():
    array1 = np.ones((2, 2, 3), dtype=np.uint8) * 2
    img1 = Image(array1)
    array2 = np.ones((2, 2, 3), dtype=np.uint8) * 3
    img2 = Image(array2)
    array = np.ones((2, 2, 3), dtype=np.uint8) * 2

    img = img1.min(img2)
    assert_equals(array.data, img.get_ndarray().data)


def test_image_clear():
    bgr_img = create_test_image()
    bgr_img.clear()
    clear_array = np.zeros((bgr_img.width, bgr_img.height, 3), dtype=np.uint8)

    assert_equals(clear_array.data, bgr_img.get_ndarray().data)


def test_image_gray_clear():
    gray_img = create_test_image().to_gray()
    gray_img.clear()
    clear_array = np.zeros((gray_img.width, gray_img.height), dtype=np.uint8)

    assert_equals(clear_array.data, gray_img.get_ndarray().data)


def test_image_getitem():
    array = np.arange(27, dtype=np.uint8).reshape((3, 3, 3))
    img = Image(array)

    assert_equals([0, 1, 2], img[0, 0])
    assert_equals([3, 4, 5], img[0, 1])
    assert_equals([9, 10, 11], img[1, 0])
    assert_equals([12, 13, 14], img[1, 1])

    assert_equals(array[:, :].tolist(), img[:, :].get_ndarray().tolist())
    assert_equals(array[1:2, 1:2].tolist(),
                  img[1:2, 1:2].get_ndarray().tolist())


def test_image_setitem():
    array = np.arange(27, dtype=np.uint8).reshape((3, 3, 3))
    img = Image(array)

    img[1, 2] = [255, 255, 255]
    assert_equals([255, 255, 255], img[1, 2])

    img[0:2, 0:2] = [50, 50, 50]
    array = np.ones((2, 2, 3), dtype=np.uint8) * 50
    assert_equals(array.tolist(), img[0:2, 0:2].get_ndarray().tolist())
