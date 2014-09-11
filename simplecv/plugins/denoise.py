from simplecv.base import logger
from simplecv.factory import Factory
from simplecv.core.image import image_method
import numpy as np

@image_method
def tv_denoising(img, gray=False, weight=50, eps=0.0002, max_iter=200,
                 resize=1):
    """
    **SUMMARY**

    Performs Total Variation Denoising, this filter tries to minimize the
    total-variation of the image.

    see : http://en.wikipedia.org/wiki/Total_variation_denoising

    **Parameters**

    * *gray* - Boolean value which identifies the colorspace of
        the input image. If set to True, filter uses gray scale values,
        otherwise colorspace is used.

    * *weight* - Denoising weight, it controls the extent of denoising.

    * *eps* - Stopping criteria for the algorithm. If the relative
      difference of the cost function becomes less than this value, the
      algorithm stops.

    * *max_iter* - Determines the maximum number of iterations the
      algorithm goes through for optimizing.

    * *resize* - Parameter to scale up/down the image. If set to
        1 filter is applied on the original image. This parameter is
        mostly to speed up the filter.

    **NOTE**

    This function requires Scikit-image library to be installed!
    To install scikit-image library run::

        sudo pip install -U scikit-image

    Read More: http://scikit-image.org/

    """

    try:
        from skimage.filter import denoise_tv_chambolle
    except ImportError:
        logger.warn('Scikit-image Library not installed!')
        return None

    img = img.copy()

    if resize <= 0:
        logger.warn('Enter a valid resize value')
        return None

    if resize != 1:
        img = img.resize(int(img.width * resize), int(img.height * resize))

    if gray is True:
        img = img.to_gray()
        multichannel = False
    elif gray is False:
        multichannel = True
    else:
        logger.warn('gray value not valid')
        return None

    denoise_mat = denoise_tv_chambolle(img, weight, eps, max_iter,
                                       multichannel)
    ret_val = img * denoise_mat

    ret_val = Factory.Image(ret_val.astype(np.uint8))
    if resize != 1:
        return ret_val.resize(int(ret_val.width / resize),
                              int(ret_val.width / resize))
    else:
        return ret_val
