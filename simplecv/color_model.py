# SimpleCV Color Model Library
#
# This library is used to model the color of foreground and background objects
from pickle import load, dump

import numpy as np

from simplecv.base import logger
from simplecv.image import Image


class ColorModel(object):
    """
    **SUMMARY**

    The color model is used to model the color of foreground and background
    objects by using a a training set of images.

    You can create the color model with any number of "training" images, or
    add images to the model with add() and remove().  Then for your data
    images, you can useThresholdImage() to return a segmented picture.

    """
    #TODO: Discretize the colorspace into smaller intervals,eg r=[0-7][8-15]
    # etc
    #TODO: Work in HSV space

    def __init__(self, data=None, is_background=True):
        self.is_background = is_background
        self.data = {}
        self.bits = 1

        if data:
            try:
                [self.add(d) for d in data]
            except TypeError:
                self.add(data)

    def _make_canonical(self, data):
        """
        Turn input types in a common form used by the rest of the class -- a
        4-bit shifted list of unique colors
        """

        #first cast everything to a numpy array
        if isinstance(data, Image):
            ret = data.reshape(-1, 3)
        elif isinstance(data, list):
            temp = []
            for dtl in data:  # do the bgr conversion
                t = (dtl[2], dtl[1], dtl[0])
                temp.append(t)
            ret = np.array(temp, dtype=np.uint8)
        elif isinstance(data, tuple):
            ret = np.array((data[2], data[1], data[0]), dtype=np.uint8)
        elif isinstance(data, np.ndarray):
            ret = data
        else:
            logger.warning("ColorModel: color is not in an accepted format!")
            return None

        rshft = np.right_shift(ret, self.bits)  # right shift 4 bits

        if len(rshft.shape) > 1:
            uniques = np.unique(
                rshft.view([('', rshft.dtype)] * rshft.shape[1])
            ).view(rshft.dtype).reshape(-1, 3)
        else:
            uniques = [rshft]
        #create a unique set of colors.  I had to look this one up

        #create a dict of encoded strings
        return dict.fromkeys(map(np.ndarray.tostring, uniques), 1)

    def reset(self):
        """
        **SUMMARY**
        Resets the color model. I.e. clears it out the stored values.


        **RETURNS**

        Nothing.

        **EXAMPLE**

        >>> cm = ColorModel()
        >>> cm.add(Image("lenna"))
        >>> cm.reset()

        """
        self.data = {}

    def add(self, data):
        """
        **SUMMARY**

        Add an image, array, or tuple to the color model.

        **PARAMETERS**

        * *data* - An image, array, or tupple of values to the color model.

        **RETURNS**

        Nothings.

        **EXAMPLE**

        >>> cm = ColorModel()
        >>> cm.add(Image("lenna"))
        >>> cm.reset()

        """
        self.data.update(self._make_canonical(data))

    def remove(self, data):
        """
        **SUMMARY**

        Remove an image, array, or tuple from the model.

        **PARAMETERS**

        * *data* - An image, array, or tupple of value.

        **RETURNS**

        Nothings.

        **EXAMPLE**

        >>> cm = ColorModel()
        >>> cm.add(Image("lenna"))
        >>> cm.remove(Color.BLACK)

        """
        self.data = dict.fromkeys(set(self.data) ^
                                  set(self._make_canonical(data)), 1)

    def threshold(self, img):
        """
        **SUMMARY**

        Perform a threshold operation on the given image. This involves
        iterating over the image and comparing each pixel to the model. If the
        pixel is in the model it is set to be either the foreground (white) or
        background (black) based on the setting of mIsBackground.

        **PARAMETERS**

        * *img* - the image to perform the threshold on.

        **RETURNS**

        The thresholded image.

        **EXAMPLE**

        >>> cm = ColorModel()
        >>> cm.add(Color.RED)
        >>> cm.add(Color.BLUE)
        >>> result = cm.threshold(Image("lenna")
        >>> result.show()

        """
        a = 0
        b = 255
        if not self.is_background:
            a, b = b, a

        # bitshift down and reshape to Nx3
        rshft = np.right_shift(img, self.bits).reshape(-1, 3)
        # map to True/False based on the model
        mapped = np.array(map(self.data.has_key,
                              map(np.ndarray.tostring, rshft)))
        # replace True and False with fg and bg
        thresh = np.where(mapped, a, b)
        return Image(thresh.reshape(img.width, img.height).astype(np.uint8))

    def contains(self, color):
        """
        **SUMMARY**

        Return true if a particular color is in our color model.

        **PARAMETERS**

        * *color* - A three value color tupple.

        **RETURNS**

        Returns True if the color is in the model, False otherwise.

        **EXAMPLE**

        >>> cm = ColorModel()
        >>> cm.add(Color.RED)
        >>> cm.add(Color.BLUE)
        >>> if cm.contains(Color.RED)
        >>>   print "Yo - we gots red y'all."


       """
        # reverse the color, cast to uint8, right shift
        # convert to string, check dict
        color_name = np.right_shift(np.cast['uint8'](color[::-1]),
                                    self.bits).tostring()
        return color_name in self.data

    def set_is_foreground(self):
        """
        **SUMMARY**

        Set our model as being foreground imagery. I.e. things in
        the model are the foreground and will be marked as white
        during the threhsold operation.

        **RETURNS**

        Nothing.

        """
        self.is_background = False

    def set_is_background(self):
        """
        **SUMMARY**

        Set our model as being background imagery. I.e. things in
        the model are the background and will be marked as black
        during the threhsold operation.


        **RETURNS**

        Nothing.

        """
        self.is_background = True

    def load(self, filename):
        """
        **SUMMARY**

        Load the color model from the specified file.

        **TO DO**

        This should be converted to pickle.

        **PARAMETERS**

        * *filename* - The file name and path to load the data from.

        **RETURNS**

        Nothing.

        **EXAMPLE**

        >>> cm = ColorModel()
        >>> cm.load("myColors.txt")
        >>> cm.add(Color.RED)
        >>> cm.add(Color.BLUE)
        >>> cm.save("mymodel")

        """
        self.data = load(open(filename))

    def save(self, filename):
        """
        **SUMMARY**

        Save a color model file.

        **PARAMETERS**

        * *filename* - The file name and path to save the data to.

        **RETURNS**

        Nothing.

        **EXAMPLE**

        >>> cm = ColorModel()
        >>> cm.add(Color.RED)
        >>> cm.add(Color.BLUE)
        >>> cm.save("mymodel.txt")

        **TO DO**

        This should be converted to pickle.

        """
        dump(self.data, open(filename, "wb"))
