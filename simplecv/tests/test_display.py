# To run this test you need python nose tools installed
# Run test just use:
#   nosetest test_display.py

from simplecv.color import Color
from simplecv.image import Image
from simplecv.tests.utils import perform_diff

#colors
black = Color.BLACK
white = Color.WHITE
red = Color.RED
green = Color.GREEN
blue = Color.BLUE

#images
testimage2 = "../data/sampleimages/aerospace.jpg"

#track images
trackimgs = ["../data/sampleimages/tracktest0.jpg",
             "../data/sampleimages/tracktest1.jpg",
             "../data/sampleimages/tracktest2.jpg",
             "../data/sampleimages/tracktest3.jpg",
             "../data/sampleimages/tracktest4.jpg",
             "../data/sampleimages/tracktest5.jpg",
             "../data/sampleimages/tracktest6.jpg",
             "../data/sampleimages/tracktest7.jpg",
             "../data/sampleimages/tracktest8.jpg",
             "../data/sampleimages/tracktest9.jpg", ]


def test_sobel():
    img = Image("lenna")
    s = img.sobel()
    name_stem = "test_sobel"
    s = [s]
    perform_diff(s, name_stem)


def test_image_new_smooth():
    img = Image(testimage2)
    result = []
    result.append(img.median_filter())
    result.append(img.median_filter((3, 3)))
    result.append(img.median_filter((5, 5), grayscale=True))
    result.append(img.bilateral_filter())
    result.append(
        img.bilateral_filter(diameter=14, sigma_color=20, sigma_space=34))
    result.append(img.bilateral_filter(grayscale=True))
    result.append(img.blur())
    result.append(img.blur((5, 5)))
    result.append(img.blur((3, 5), grayscale=True))
    result.append(img.gaussian_blur())
    result.append(img.gaussian_blur((3, 7), sigma_x=10, sigma_y=12))
    result.append(
        img.gaussian_blur((7, 9), sigma_x=10, sigma_y=12, grayscale=True))
    name_stem = "test_image_new_smooth"
    perform_diff(result, name_stem)


def test_camshift():
    ts = []
    bb = (195, 160, 49, 46)
    imgs = [Image(img) for img in trackimgs]
    ts = imgs[0].track("camshift", ts, imgs[1:], bb)
    assert ts


def test_lk():
    ts = []
    bb = (195, 160, 49, 46)
    imgs = [Image(img) for img in trackimgs]
    ts = imgs[0].track("LK", ts, imgs[1:], bb)
    assert ts
