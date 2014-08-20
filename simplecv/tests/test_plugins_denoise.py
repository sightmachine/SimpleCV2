from nose.tools import assert_equals, assert_almost_equals
from simplecv.core.image import operation
from simplecv.plugins import denoise
from simplecv.tests.utils import perform_diff, perform_diff_blobs, skipped

from simplecv.image import Image

@skipped  # FIXME
def test_denoise_tv_denoising():
    img = Image("../data/sampleimages/lena_noisy.png")
    img2 = Image("../data/sampleimages/cameraman_noisy.png", color_space=Image.GRAY)
    retval = denoise.tv_denoising(img)
    retval1 = denoise.tv_denoising(img, resize=0.5) 
    retval2 = denoise.tv_denoising(img2, gray=True, weight=100, eps=0.002)

    result = [retval, retval1, retval2]
    name_stem = "test_plugins_denoise"
    perform_diff(result, name_stem, 0.0)
