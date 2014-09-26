from simplecv.color import Color


class RendererBase(object):

    def __init__(self):
        self.default_alpha = 255
        self.default_color = Color.BLACK
        self.font_color = 0
        self.font_size = 18
        self.font_name = None
        self.font_bold = False
        self.font_italic = False
        self.font_underline = False

    def set_default_alpha(self, alpha):
        """
        This method sets the default alpha value for all methods called on this
        layer. The default value starts out at 255 which is completely
         transparent.
        """
        self.default_alpha = alpha

    def set_default_color(self, color):
        """
        This method sets the default rendering color.

        Parameters:
            color - Color object or Color Tuple
        """
        self.default_color = color

    def select_font(self, name):
        """
        This method attempts to set the font from a font file. It is advisable
        to use one of the fonts listed by the list_fonts() method. The input
        is a string with the font name.
        """
        self.font_name = name

    def set_font_size(self, size):
        """
        This method sets the font size roughly in points. A size of 10 is
        almost too small to read. A size of 20 is roughly 10 pixels high and a
        good choice.

        Parameters:
            sz = Int
        """
        self.font_size = size

    def set_font_bold(self, value):
        """
        This method sets and unsets the current font to be bold.
        """
        self.font_bold = value

    def set_font_italic(self, value):
        """
        This method sets and unsets the current font to be italic.
        """
        self.font_italic = value

    def set_font_underline(self, value):
        """
        This method sets and unsets the current font to be underlined
        """
        self.font_underline = value
