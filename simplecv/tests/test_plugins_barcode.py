from nose.tools import assert_equals, assert_not_equals, assert_almost_equals
from simplecv.core.image import operation
from simplecv.plugins import barcode

from simplecv.tests.utils import perform_diff, perform_diff_blobs

from simplecv.image import Image

def test_barcode_find_barcode():
	img = Image("../data/sampleimages/barcode.png")
	featureset = barcode.find_barcode(img)
	f = featureset[0]
	img_crop = f.crop()
	result = [img_crop]
	name_stem = "test_barcode_find_barcode"
	perform_diff(result, name_stem, 0.0)