import os
import tempfile

from mock import patch
from nose.tools import assert_equals, raises
import cv2
import numpy as np

from simplecv import DATA_DIR
from simplecv.color import Color
from simplecv.image import Image
from simplecv.tests.utils import (perform_diff, create_test_image,
                                  create_test_array)

LENNA_PATH = os.path.join(DATA_DIR, 'sampleimages/lenna.png')
WEBP_IMAGE_PATH = os.path.join(DATA_DIR, 'sampleimages/simplecv.webp')
TESTIMAGECLR = os.path.join(DATA_DIR, 'sampleimages/statue_liberty.jpg')
TESTIMAGE = os.path.join(DATA_DIR, 'sampleimages/9dots4lines.png')


def test_image_init_path_to_png():
    img1 = Image(source=LENNA_PATH)
    assert_equals((512, 512), img1.size)
    assert_equals(Image.BGR, img1.color_space)
    assert img1.is_color_space(Image.BGR)

    img_ndarray = img1.get_ndarray()

    assert isinstance(img_ndarray, np.ndarray)
    assert_equals(3, len(img_ndarray.shape))

    assert not img1.is_color_space(Image.RGB)
    assert not img1.is_color_space(Image.HSV)
    assert not img1.is_color_space(Image.HLS)
    assert not img1.is_color_space(Image.YCR_CB)
    assert not img1.is_color_space(Image.XYZ)
    assert not img1.is_color_space(Image.GRAY)


def test_image_repr():
    img = Image((10, 10))
    repr_str = img.__repr__()
    assert 'simplecv.Image Object' in repr_str
    assert 'size:(10, 10)' in repr_str
    assert 'dtype: uint8' in repr_str
    assert 'channels: 3' in repr_str
    assert 'filename: (None)' in repr_str
    assert 'dtype: uint8' in repr_str
    assert 'at memory location: ' in repr_str

    # Test for some filename
    img = Image(LENNA_PATH)
    repr_str = img.__repr__()
    assert 'simplecv.Image Object' in repr_str
    assert 'size:(512, 512)' in repr_str
    assert 'dtype: uint8' in repr_str
    assert 'channels: 3' in repr_str
    assert 'filename: (None)' not in repr_str
    assert LENNA_PATH in repr_str
    assert 'dtype: uint8' in repr_str
    assert 'at memory location: ' in repr_str


def test_image_init_sample_png():
    img1 = Image(source="lenna")
    assert_equals((512, 512), img1.size)
    assert_equals(Image.BGR, img1.color_space)
    assert img1.is_color_space(Image.BGR)


def test_sample_images():
    img = Image('lenna')
    assert img.filename
    assert 'lenna.png' in img.filename
    assert img.is_color_space(Image.BGR)
    assert_equals(np.uint8, img.dtype)

    img = Image('simplecv')
    assert img.filename
    assert 'simplecv.png' in img.filename
    assert img.is_color_space(Image.BGR)
    assert_equals(np.uint8, img.dtype)

    img = Image('inverted')
    assert img.filename
    assert 'simplecv_inverted.png' in img.filename
    assert img.is_color_space(Image.BGR)
    assert_equals(np.uint8, img.dtype)

    img = Image('lyle')
    assert img.filename
    assert 'LyleJune1973.png' in img.filename
    assert img.is_color_space(Image.BGR)
    assert_equals(np.uint8, img.dtype)

    img = Image('lyle')
    assert img.filename
    assert 'LyleJune1973.png' in img.filename
    assert img.is_color_space(Image.BGR)
    assert_equals(np.uint8, img.dtype)

    img = Image("parity")
    assert img.filename
    assert img.is_color_space(Image.BGR)
    assert_equals(np.uint8, img.dtype)


def test_image_init_path_to_webp():
    img = Image(source=WEBP_IMAGE_PATH)

    assert_equals((250, 250), img.size)
    assert img.is_color_space(Image.RGB)
    img_ndarray = img.get_ndarray()
    assert isinstance(img_ndarray, np.ndarray)
    assert_equals(3, len(img_ndarray.shape))
    assert_equals(np.uint8, img.dtype)


@raises(Exception)
def test_image_init_bad_path():
    Image('/bad/path/to/image.png')


@patch('urllib2.urlopen')
def test_image_init_url(urlopen_mock):
    with open(LENNA_PATH) as f:
        urlopen_mock.return_value = f
        img = Image("http://someserver.com/lenna.png")
        assert_equals((512, 512), img.size)
        assert_equals(np.uint8, img.dtype)
        assert img.is_color_space(Image.BGR)


def test_image_init_ndarray_color():
    color_ndarray = cv2.imread(LENNA_PATH)
    img1 = Image(array=color_ndarray, color_space=Image.BGR)

    assert_equals((512, 512), img1.size)
    assert_equals(Image.BGR, img1.color_space)
    assert img1.is_color_space(Image.BGR)

    img_ndarray = img1.get_ndarray()

    assert isinstance(img_ndarray, np.ndarray)
    assert_equals(3, len(img_ndarray.shape))


def test_image_init_ndarray_grayscale():
    ndarray = cv2.imread(LENNA_PATH)
    gray_ndarray = cv2.cvtColor(ndarray, cv2.COLOR_BGR2GRAY)

    img1 = Image(array=gray_ndarray)
    assert_equals((512, 512), img1.size)
    assert_equals(Image.GRAY, img1.color_space)
    assert img1.is_color_space(Image.GRAY)

    img_ndarray = img1.get_ndarray()

    assert isinstance(img_ndarray, np.ndarray)
    assert_equals(2, len(img_ndarray.shape))


def test_image_numpy_constructor():
    img = Image(source=LENNA_PATH)
    grayimg = img.to_color_space(Image.GRAY)

    chan3_array = np.array(img.get_ndarray())
    chan1_array = np.array(img.get_gray_ndarray())

    img2 = Image(array=chan3_array, color_space=Image.BGR)
    grayimg2 = Image(array=chan1_array)

    assert img2[0, 0] == img[0, 0]
    assert grayimg2[0, 0] == grayimg[0, 0]


def test_image_save():
    temp_file = os.path.join(tempfile.gettempdir(), 'temp_image.png')
    img = create_test_image()
    img.save(temp_file)
    assert os.path.isfile(temp_file)
    os.remove(temp_file)


def test_image_init_tuple_bgr():
    img1 = Image(source=[5, 10], color_space=Image.BGR)
    assert img1.is_bgr()
    assert_equals((5, 10), img1.size)
    assert_equals(np.zeros((5, 10, 3), np.uint8).data, img1.get_ndarray().data)


def test_image_init_tuple_gray():
    img1 = Image(source=[5, 10], color_space=Image.GRAY)
    assert img1.is_gray()
    assert_equals((5, 10), img1.size)
    assert_equals(np.zeros((5, 10), np.uint8).data, img1.get_ndarray().data)


def test_image_convert_bgr_to_bgr():
    bgr_array = create_test_array()
    result_bgr_array = Image.convert(bgr_array, Image.BGR, Image.BGR)
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
    hsv_img = Image(array=hsv_array, color_space=Image.HSV)
    gray_array = np.array([[76, 150],
                           [29, 255]], dtype=np.uint8)
    gray_img = hsv_img.to_gray()

    assert_equals(gray_array.data, gray_img.get_ndarray().data)
    assert gray_img.is_gray()


def test_image_copy():
    img = create_test_image()
    copy_img = img.copy()

    assert img is not copy_img
    assert_equals(img.size, copy_img.size)
    assert_equals(img.get_ndarray().data, copy_img.get_ndarray().data)
    assert_equals(img.color_space, copy_img.color_space)


def test_image_operator_wrong_size():
    array1 = np.ones((2, 2, 3), dtype=np.uint8) * 5
    img1 = Image(array=array1, color_space=Image.BGR)
    array2 = np.ones((2, 1, 3), dtype=np.uint8) * 5
    img2 = Image(array=array2, color_space=Image.BGR)

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
    # img = img1.max(img2)
    # assert_equals(None, img)
    # img = img1.min(img2)
    # assert_equals(None, img)


def test_image_sub_image():
    array1 = np.ones((2, 2, 3), dtype=np.uint8) * 5
    img1 = Image(array=array1)
    array2 = np.ones((2, 2, 3), dtype=np.uint8) * 5
    img2 = Image(array=array2)
    array = np.zeros((2, 2, 3), dtype=np.uint8)

    img = img1 - img2
    assert_equals(array.data, img.get_ndarray().data)
    assert isinstance(img, Image)


def test_image_sub_int():
    array1 = np.ones((2, 2, 3), dtype=np.uint8) * 5
    img1 = Image(array=array1)
    array = np.zeros((2, 2, 3), dtype=np.uint8)

    img = img1 - 5
    assert_equals(array.data, img.get_ndarray().data)
    assert isinstance(img, Image)


def test_image_add_image():
    array1 = np.ones((2, 2, 3), dtype=np.uint8) * 5
    img1 = Image(array=array1)
    array2 = np.ones((2, 2, 3), dtype=np.uint8) * 5
    img2 = Image(array=array2)
    array = np.ones((2, 2, 3), dtype=np.uint8) * 10

    img = img1 + img2
    assert_equals(array.data, img.get_ndarray().data)
    assert isinstance(img, Image)


def test_image_add_int():
    array1 = np.ones((2, 2, 3), dtype=np.uint8) * 5
    img1 = Image(array=array1)

    array = np.ones((2, 2, 3), dtype=np.uint8) * 10

    img = img1 + 5
    assert_equals(array.data, img.get_ndarray().data)
    assert isinstance(img, Image)


def test_image_and_image():
    array1 = np.ones((2, 2, 3), dtype=np.uint8) * 5
    img1 = Image(array=array1)
    array2 = np.ones((2, 2, 3), dtype=np.uint8) * 3
    img2 = Image(array=array2)
    array = np.ones((2, 2, 3), dtype=np.uint8)

    img = img1 & img2
    assert_equals(array.data, img.get_ndarray().data)
    assert isinstance(img, Image)


def test_image_and_int():
    array1 = np.ones((2, 2, 3), dtype=np.uint8) * 5
    img1 = Image(array=array1)
    array = np.ones((2, 2, 3), dtype=np.uint8)

    img = img1 & 3
    assert_equals(array.data, img.get_ndarray().data)
    assert isinstance(img, Image)


def test_image_or_image():
    array1 = np.ones((2, 2, 3), dtype=np.uint8) * 5
    img1 = Image(array=array1)
    array2 = np.ones((2, 2, 3), dtype=np.uint8) * 3
    img2 = Image(array=array2)
    array = np.ones((2, 2, 3), dtype=np.uint8) * 7

    img = img1 | img2
    assert_equals(array.data, img.get_ndarray().data)
    assert isinstance(img, Image)


def test_image_or_int():
    array1 = np.ones((2, 2, 3), dtype=np.uint8) * 5
    img1 = Image(array=array1)
    array = np.ones((2, 2, 3), dtype=np.uint8) * 7

    img = img1 | 3
    assert_equals(array.data, img.get_ndarray().data)
    assert isinstance(img, Image)


def test_image_div_image():
    array1 = np.ones((2, 2, 3), dtype=np.uint8) * 50
    img1 = Image(array=array1)
    array2 = np.ones((2, 2, 3), dtype=np.uint8) * 2
    img2 = Image(array=array2)
    array = np.ones((2, 2, 3), dtype=np.uint8) * 25

    img = img1 / img2
    assert_equals(array.data, img.get_ndarray().data)
    assert isinstance(img, Image)


def test_image_div_int_float():
    array1 = np.ones((2, 2, 3), dtype=np.uint8) * 50
    img1 = Image(array=array1)
    array = np.ones((2, 2, 3), dtype=np.uint8) * 25

    img = img1 / 2
    assert_equals(array.data, img.get_ndarray().data)
    assert isinstance(img, Image)

    array = np.ones((2, 2, 3), dtype=np.uint8) * 20
    img = img1 / 2.5
    assert_equals(array.data, img.get_ndarray().data)
    assert isinstance(img, Image)


def test_image_multiply_image():
    array1 = np.ones((2, 2, 3), dtype=np.uint8) * 50
    img1 = Image(array=array1)
    array2 = np.ones((2, 2, 3), dtype=np.uint8) * 2
    img2 = Image(array=array2)
    array = np.ones((2, 2, 3), dtype=np.uint8) * 100

    img = img1 * img2
    assert_equals(array.data, img.get_ndarray().data)
    assert isinstance(img, Image)


def test_image_multiply_int_float():
    array1 = np.ones((2, 2, 3), dtype=np.uint8) * 50
    img1 = Image(array=array1)
    array = np.ones((2, 2, 3), dtype=np.uint8) * 100

    img = img1 * 2
    assert_equals(array.data, img.get_ndarray().data)
    assert isinstance(img, Image)

    array = np.ones((2, 2, 3), dtype=np.uint8) * 125
    img = img1 * 2.5
    assert_equals(array.data, img.get_ndarray().data)
    assert isinstance(img, Image)


@raises(ValueError)
def test_image_pow_image():
    array1 = np.ones((2, 2, 3), dtype=np.uint8) * 50
    img1 = Image(array=array1)
    array2 = np.ones((2, 2, 3), dtype=np.uint8) * 2
    img2 = Image(array=array2)
    img = img1 ** img2


def test_image_pow_int_float():
    array1 = np.ones((2, 2, 3), dtype=np.uint8) * 5
    img1 = Image(array=array1)
    array = np.ones((2, 2, 3), dtype=np.uint8) * 255

    img = img1 ** 20
    assert_equals(array.data, img.get_ndarray().data)
    assert isinstance(img, Image)

    img = img1 ** 20
    assert_equals(array.data, img.get_ndarray().data)
    assert isinstance(img, Image)


def test_image_neg_invert():
    array1 = np.ones((2, 2, 3), dtype=np.uint8) * 5
    img1 = Image(array=array1)
    array = np.ones((2, 2, 3), dtype=np.uint8) * 250

    img = ~img1
    assert_equals(array.data, img.get_ndarray().data)
    assert isinstance(img, Image)
    img = -img1
    assert_equals(array.data, img.get_ndarray().data)
    assert isinstance(img, Image)


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
    img = Image(array=array)

    assert_equals([0, 1, 2], img[0, 0])
    assert_equals([3, 4, 5], img[0, 1])
    assert_equals([9, 10, 11], img[1, 0])
    assert_equals([12, 13, 14], img[1, 1])

    assert_equals(array[:, :].tolist(), img[:, :].get_ndarray().tolist())
    assert_equals(array[1:2, 1:2].tolist(),
                  img[1:2, 1:2].get_ndarray().tolist())


def test_image_setitem():
    array = np.arange(27, dtype=np.uint8).reshape((3, 3, 3))
    img = Image(array=array)

    img[1, 2] = [255, 255, 255]
    assert_equals([255, 255, 255], img[1, 2])

    img[0:2, 0:2] = [50, 50, 50]
    array = np.ones((2, 2, 3), dtype=np.uint8) * 50
    assert_equals(array.tolist(), img[0:2, 0:2].get_ndarray().tolist())


def test_image_split_merge_channels():
    img = create_test_image()
    b, g, r = img.split_channels()
    assert_equals([[0, 0], [255, 255]], b.get_ndarray().tolist())
    assert_equals([[0, 255], [0, 255]], g.get_ndarray().tolist())
    assert_equals([[255, 0], [0, 255]], r.get_ndarray().tolist())

    img1 = img.merge_channels(b, g, r)
    assert_equals(img1.get_ndarray().tolist(), img.get_ndarray().tolist())


def test_image_drawing():
    img = Image(source=TESTIMAGECLR)
    img.draw_circle((img.width / 2, img.height / 2), 10, thickness=3)
    img.draw_circle((img.width / 2, img.height / 2), 15, thickness=5,
                    color=Color.RED)
    img.draw_circle((img.width / 2, img.height / 2), 20)
    img.draw_line((5, 5), (5, 8))
    img.draw_line((5, 5), (10, 10), thickness=3)
    img.draw_line((0, 0), (img.width, img.height), thickness=3,
                  color=Color.BLUE)
    img.draw_rectangle(20, 20, 10, 5)
    img.draw_rectangle(22, 22, 10, 5, alpha=128)
    img.draw_rectangle(24, 24, 10, 15, width=-1, alpha=128)
    img.draw_rectangle(28, 28, 10, 15, width=3, alpha=128)
    result = [img]
    name_stem = "test_image_drawing"
    perform_diff(result, name_stem)


def test_image_draw():
    img = Image(source="lenna")
    newimg = Image(source="simplecv")
    lines = img.find_lines()
    newimg.draw(lines)
    lines.draw()
    result = [newimg, img]
    name_stem = "test_image_draw"
    perform_diff(result, name_stem)


def test_color_conversion_func_bgr():
    #we'll just go through the space to make sure nothing blows up
    img = Image(source=TESTIMAGE)
    results = []
    results.append(img.to_bgr())
    results.append(img.to_rgb())
    results.append(img.to_hls())
    results.append(img.to_hsv())
    results.append(img.to_xyz())
    results.append(img.to_ycrcb())
    bgr = img.to_bgr()

    results.append(bgr.to_bgr())
    results.append(bgr.to_rgb())
    results.append(bgr.to_hls())
    results.append(bgr.to_hsv())
    results.append(bgr.to_xyz())
    results.append(bgr.to_ycrcb())

    name_stem = "test_color_conversion_func_bgr"
    perform_diff(results, name_stem)


def test_color_conversion_func_rgb():
    img = Image(source=TESTIMAGE)
    assert img.is_bgr()
    rgb = img.to_rgb()
    assert rgb.is_rgb()
    assert rgb.to_bgr().is_bgr()
    assert rgb.to_rgb().is_rgb()
    assert rgb.to_hls().is_hls()
    assert rgb.to_hsv().is_hsv()
    assert rgb.to_xyz().is_xyz()
    assert rgb.to_ycrcb().is_ycrcb()


def test_color_conversion_func_hsv():
    img = Image(source=TESTIMAGE)
    hsv = img.to_hsv()
    results = [hsv]
    results.append(hsv.to_bgr())
    results.append(hsv.to_rgb())
    results.append(hsv.to_hls())
    results.append(hsv.to_hsv())
    results.append(hsv.to_xyz())
    results.append(hsv.to_ycrcb())
    name_stem = "test_color_conversion_func_hsv"
    perform_diff(results, name_stem)


def test_color_conversion_func_hls():
    img = Image(source=TESTIMAGE)
    hls = img.to_hls()
    results = [hls]
    results.append(hls.to_bgr())
    results.append(hls.to_rgb())
    results.append(hls.to_hls())
    results.append(hls.to_hsv())
    results.append(hls.to_xyz())
    results.append(hls.to_ycrcb())
    name_stem = "test_color_conversion_func_hls"
    perform_diff(results, name_stem)


def test_color_conversion_func_xyz():
    img = Image(source=TESTIMAGE)
    xyz = img.to_xyz()
    results = [xyz]
    results.append(xyz.to_bgr())
    results.append(xyz.to_rgb())
    results.append(xyz.to_hls())
    results.append(xyz.to_hsv())
    results.append(xyz.to_xyz())
    results.append(xyz.to_ycrcb())
    name_stem = "test_color_conversion_func_xyz"
    perform_diff(results, name_stem)


def test_color_conversion_func_ycrcb():
    img = Image(source=TESTIMAGE)
    ycrcb = img.to_ycrcb()
    results = [ycrcb]
    results.append(ycrcb.to_bgr())
    results.append(ycrcb.to_rgb())
    results.append(ycrcb.to_hls())
    results.append(ycrcb.to_hsv())
    results.append(ycrcb.to_xyz())
    results.append(ycrcb.to_ycrcb())
    name_stem = "test_color_conversion_func_ycrcb"
    perform_diff(results, name_stem)


def test_get_exif_data():
    img = Image("../data/sampleimages/cat.jpg")
    d1 = img.get_exif_data()
    assert_equals(37, len(d1))
    assert_equals('sRGB', d1['EXIF ColorSpace'].printable)
    assert_equals('JPEG (old-style)', d1['Thumbnail Compression'].printable)
    img2 = Image(TESTIMAGE)
    d2 = img2.get_exif_data()
    assert_equals(0, len(d2))
