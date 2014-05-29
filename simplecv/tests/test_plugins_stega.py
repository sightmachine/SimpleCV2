from nose.tools import assert_equals, assert_not_equals, assert_almost_equals
from simplecv.core.image import operation
from simplecv.plugins import stega
from simplecv.image import Image

from simplecv.tests.utils import perform_diff

def test_stega_stega_encode():
	img = Image("../data/sampleimages/lenna.png")
	img1 = stega.stega_encode(img, "Hello World!")
	result = [img1]
	name_stem = "test_stega_stega_encode"
	perform_diff(result, name_stem, 0.0)

def test_stega_stega_decode():
	img = Image("../data/sampleimages/lenna_stega_encoded.png")
	result = img.stega_decode()
	assert_equals(result, "Hello World!")

