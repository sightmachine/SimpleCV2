# SimpleCV Font Library
#
# This library is used to add fonts to images

from PIL import ImageFont as pilImageFont


class Font(object):
    """
    The Font class allows you to create a font object to be
    used in drawing or writing to images.
    There are some defaults available, to see them, just type
    Font.print_fonts()
    """

    _font_path = "simplecv/data/fonts/"
    _extension = ".ttf"
    _font_face = "ubuntu"
    _font_size = 16
    _font = None

    # These fonts were downloaded from Google at:
    # http://www.http://www.google.com/webfonts
    _fonts = [
        "ubuntu",
        "astloch",
        "carter_one",
        "kranky",
        "la_belle_aurore",
        "monofett",
        "reenie_beanie",
        "shadows_Into_light",
        "special_elite",
        "unifrakturmaguntia",
        "vt323",
        "wallpoet",
        "wire_one"
    ]

    def __init__(self, font_face="ubuntu", font_size=16):
        """
        This creates a new font object, it uses ubuntu as the default font
        To give it a custom font you can just pass the absolute path
        to the truetype font file.
        """

        self.set_size(font_size)
        self.set_font(font_face)

    def get_font(self):
        """
        Get the font from the object to be used in drawing

        Returns: PIL Image Font
        """

        return self._font

    def set_font(self, new_font='ubuntu'):
        """
        Set the name of the font listed in the font family
        if the font isn't listed in the font family then pass it the absolute
        path of the truetype font file.
        Example: Font.set_font("/home/simplecv/my_font.ttf")
        """

        if isinstance(new_font, basestring):
            print "Please pass a string"
            return None

        if new_font in self._fonts:
            self._font_face = new_font
            font_to_use = self._font_path + self._font_face + "/" + \
                self._font_face + self._extension
        else:
            self._font_face = new_font
            font_to_use = new_font

        self._font = pilImageFont.truetype(font_to_use, self._font_size)

    def set_size(self, size):
        """
        Set the font point size. i.e. 16pt
        """

        #FIXME: delete print...
        print type(size)
        if isinstance(size, int):
            self._font_size = size
        else:
            print "please provide an integer"

    def get_size(self):
        """
        Gets the size of the current font

        Returns: Integer
        """

        return self._font_size

    def get_fonts(self):
        """
        This returns the list of fonts built into simplecv
        """

        return self._fonts

    def print_fonts(self):
        """
        This prints a list of fonts built into simplecv
        """

        # FIXME: change print -> some logging?
        for font in self._fonts:
            print font
