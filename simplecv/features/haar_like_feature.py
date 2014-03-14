"""
HaarLikeFeature class
"""


class HaarLikeFeature(object):
    """
    Create a single Haar feature and optionally set the regions that define
    the Haar feature and its name. The formal of the feature is

    The format is [[[TL],[BR],SIGN],[[TL],[BR],SIGN].....]
    Where TR and BL are the unit coorinates for the top right and bottom
    left coodinates.

    For example
    [[[0,0],[0.5,0.5],1],[[0.5.0],[1.0,1.0],-1]]

    Takes the right side of the image and subtracts from the left hand side
    of the image.
    """

    def __init__(self, name=None, regions=None):
        self.name = name
        self.regions = regions

    def set_regions(self, regions):
        """
        Set the list of regions. The regions are square coordinates on a unit
        sized image followed by the sign of a region.

        The format is [[[TL],[BR],SIGN],[[TL],[BR],SIGN].....]
        Where TR and BL are the unit coorinates for the top right and bottom
        left coodinates.

        For example
        [[[0,0],[0.5,0.5],1],[[0.5.0],[1.0,1.0],-1]]

        Takes the right side of the image and subtracts from the left hand side
        of the image.
        """
        self.regions = regions

    def set_name(self, name):
        """
        Set the name of this feature, the name must be unique.
        """
        self.name = name

    def apply(self, int_img):
        """
        This method takes in an integral image and applies the haar-cascade
        to the image, and returns the result.
        """
        w = int_img.shape[0] - 1
        h = int_img.shape[1] - 1
        accumulator = 0
        for i in range(len(self.regions)):
            # using the integral image
            # a = Lower Right Hand Corner
            # b = upper right hand corner
            # c = lower left hand corner
            # d = upper left hand corner
            # sum = a - b - c + d
            # regions are in
            # (left, top, right, bottom, sign) format
            left = self.regions[i][0]  # left (all are unit length)
            top = self.regions[i][1]  # top
            right = self.regions[i][2]  # right
            bottom = self.regions[i][3]  # bottom
            sign = self.regions[i][4]  # sign
            x_a = int(w * right)
            y_a = int(h * bottom)
            x_b = int(w * right)
            y_b = int(h * top)
            x_c = int(w * left)
            y_c = int(h * bottom)
            x_d = int(w * left)
            y_d = int(h * top)
            accumulator += sign * (int_img[x_a, y_a] - int_img[x_b, y_b]
                                   - int_img[x_c, y_c] + int_img[x_d, y_d])
        return accumulator

    def write_to_file(self, ofile):
        """
        Write the Haar cascade to a human readable file. ofile is an open file
        pointer.
        """
        ofile.write(self.name)
        ofile.write(" " + str(len(self.regions)) + "\n")
        for i in range(len(self.regions)):
            temp = self.regions[i]
            for j in range(len(temp)):
                ofile.write(str(temp[j]) + ' ')
            ofile.write('\n')
        ofile.write('\n')
