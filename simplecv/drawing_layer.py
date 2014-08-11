# SimpleCV Stream Library
#
# This library is used for drawing and text rendering
from numpy import ones, uint8
import pygame as pg
from pygame import gfxdraw
import svgwrite

from simplecv.color import Color
from simplecv.core.pluginsystem import apply_plugins

#DOCS
#TESTS
#IMAGE AGNOSTIC
#RESIZE
#ADD IMAGE INTERFACE


@apply_plugins
class DrawingLayer(object):
    """
    DrawingLayer gives you a way to mark up Image classes without changing
    the image data itself. This class wraps pygame's Surface class and
    provides basic drawing and text rendering functions


    Example:
    image = Image("/path/to/image.png")
    image2 = Image("/path/to/image2.png")
    image.dl().blit(image2) #write image 2 on top of image
    """

    def __init__(self, (width, height)):
        #pg.init()
        if not pg.font.get_init():
            pg.font.init()

        self.svg = svgwrite.Drawing(size=(width, height))
        self.width = width
        self.height = height
        self.surface = pg.Surface((width, height), flags=pg.SRCALPHA)
        self.default_alpha = 255
        # This is used to track the changed value in alpha
        self.alpha_delta = 1
        self.clear_color = pg.Color(0, 0, 0, 0)

        self.surface.fill(self.clear_color)
        self.default_color = Color.BLACK

        self.font_color = 0
        self.font_size = 18
        self.font_name = None
        self.font_bold = False
        self.font_italic = False
        self.font_underline = False
        self.font = pg.font.Font(self.font_name, self.font_size)

    def __repr__(self):
        return "<SimpleCV.DrawingLayer Object size (%d, %d)>" % (
            self.width, self.height)

    def set_default_alpha(self, alpha):
        """
        This method sets the default alpha value for all methods called on this
        layer. The default value starts out at 255 which is completely
         transparent.
        """
        if 0 <= alpha <= 255:
            self.default_alpha = alpha

    def get_default_alpha(self):
        """
        Returns the default alpha value.
        """
        return self.default_alpha

    def set_layer_alpha(self, alpha):
        """
        This method sets the alpha value of the entire layer in a single
        pass. This is helpful for merging layers with transparency.
        """
        self.surface.set_alpha(alpha)
        # Get access to the alpha band of the image.
        pixels_alpha = pg.surfarray.pixels_alpha(self.surface)
        # Do a floating point multiply, by alpha 100, on each alpha value.
        # Then truncate the values (convert to integer) and copy back into
        # the surface.
        pixels_alpha[...] = (ones(pixels_alpha.shape) * alpha).astype(uint8)

        # Unlock the surface.

        self.alpha_delta = alpha / 255.0  # update the changed state

    def get_svg(self):
        return self.svg.tostring()

    def _get_surface(self):
        return self.surface

    def _csv_rgb_to_pg_color(self, color, alpha=-1):
        if alpha == -1:
            alpha = self.default_alpha

        if color == Color.DEFAULT:
            color = self.default_color
        ret_value = pg.Color(color[0], color[1], color[2], alpha)
        return ret_value

    def set_default_color(self, color):
        """
        This method sets the default rendering color.

        Parameters:
            color - Color object or Color Tuple
        """
        self.default_color = color

    def line(self, start, stop, color=Color.DEFAULT, width=1, antialias=True,
             alpha=-1):
        """
        Draw a single line from the (x,y) tuple start to the (x,y) tuple stop.
        Optional parameters:

        color - The object's color as a simple CVColor object, if no value is
                specified the default is used.

        alpha - The alpha blending for the object. If this value is -1 then the
                layer default value is used. A value of 255 means opaque,
                while 0 means transparent.

        width - The line width in pixels.

        antialias - Draw an antialiased object of width one.

        Parameters:
            start - Tuple
            stop - Tuple
            color - Color object or Color Tuple
            width - Int
            antialias - Boolean
            alpha - Int

        """
        if antialias and width == 1:
            pg.draw.aaline(self.surface,
                           self._csv_rgb_to_pg_color(color, alpha),
                           start, stop, width)
        else:
            pg.draw.line(self.surface,
                         self._csv_rgb_to_pg_color(color, alpha),
                         start, stop, width)

        start_int = tuple(int(x) for x in start)
        stop_int = tuple(int(x) for x in stop)
        self.svg.add(self.svg.line(start=start_int, end=stop_int))

    def lines(self, points, color=Color.DEFAULT, antialias=True, alpha=-1,
              width=1):
        """
        Draw a set of lines from the list of (x,y) tuples points. Lines are
        draw between each successive pair of points.

        Optional parameters:

        color - The object's color as a simple CVColor object, if no value is
                specified the default is used.

        alpha - The alpha blending for the object. If this value is -1 then the
                layer default value is used. A value of 255 means opaque,
                while 0 means transparent.

        width - The line width in pixels.

        antialias - Draw an antialiased object of width one.

        Parameters:
            points - Tuple
            color - Color object or Color Tuple
            antialias - Boolean
            alpha - Int
            width - Int

        """
        if antialias and width == 1:
            pg.draw.aalines(self.surface,
                            self._csv_rgb_to_pg_color(color, alpha),
                            0, points, width)
        else:
            pg.draw.lines(self.surface,
                          self._csv_rgb_to_pg_color(color, alpha),
                          0, points, width)

        last_point = points[0]
        for point in points[1:]:
            last_int = tuple(int(x) for x in last_point)
            curr_int = tuple(int(x) for x in point)
            self.svg.add(self.svg.line(start=last_int, end=curr_int))
            last_point = point

    #need two points(TR,BL), center+W+H, and TR+W+H
    def rectangle(self, top_left, dimensions, color=Color.DEFAULT, width=1,
                  filled=False, alpha=-1):
        """
        Draw a rectangle given the top_left the (x,y) coordinate of the top
        left corner and dimensions (w,h) tge width and height

        color - The object's color as a simple CVColor object, if no value is
                specified the default is used.

        alpha - The alpha blending for the object. If this value is -1 then the
                layer default value is used. A value of 255 means opaque,
                while 0 means transparent.

        w -     The line width in pixels. This does not work if antialiasing
                is enabled.

        filled -The rectangle is filled in
        """
        if filled:
            width = 0
        rect = pg.Rect((top_left[0], top_left[1]),
                       (dimensions[0], dimensions[1]))
        pg.draw.rect(self.surface, self._csv_rgb_to_pg_color(color, alpha),
                     rect, width)

        tl_int = tuple(int(x) for x in top_left)
        dim_int = tuple(int(x) for x in dimensions)
        self.svg.add(self.svg.rect(insert=tl_int, size=dim_int))

    def rectangle_to_pts(self, pt0, pt1, color=Color.DEFAULT, width=1,
                         filled=False, alpha=-1):
        """
        Draw a rectangle given two (x,y) points

        color - The object's color as a simple CVColor object, if no value is
                specified the default is used.

        alpha - The alpha blending for the object. If this value is -1 then the
                layer default value is used. A value of 255 means opaque,
                while 0 means transparent.

        w -     The line width in pixels. This does not work if antialiasing is
                enabled.

        filled -The rectangle is filled in
        """

        if pt0[0] > pt1[0]:
            w = pt0[0] - pt1[0]
            x = pt1[0]
        else:
            w = pt1[0] - pt0[0]
            x = pt0[0]
        if pt0[1] > pt1[1]:
            h = pt0[1] - pt1[1]
            y = pt1[1]
        else:
            h = pt1[1] - pt0[1]
            y = pt0[1]
        if filled:
            width = 0
        rect = pg.Rect((x, y), (w, h))
        pg.draw.rect(self.surface, self._csv_rgb_to_pg_color(color, alpha),
                     rect, width)

        self.svg.add(
            self.svg.rect(insert=(int(x), int(y)), size=(int(w), int(h))))

    def centered_rectangle(self, center, dimensions, color=Color.DEFAULT,
                           width=1, filled=False, alpha=-1):
        """
        Draw a rectangle given the center (x,y) of the rectangle and dimensions
        (width, height)

        color - The object's color as a simple CVColor object, if no value is
                specified the default is used.

        alpha - The alpha blending for the object. If this value is -1 then the
                layer default value is used. A value of 255 means opaque, while
                0 means transparent.

        w -     The line width in pixels. This does not work if antialiasing is
                enabled.

        filled -The rectangle is filled in


        parameters:
            center - Tuple
            dimenions - Tuple
            color - Color object or Color Tuple
            width - Int
            filled - Boolean
            alpha - Int

        """
        if filled:
            width = 0
        xtl = center[0] - (dimensions[0] / 2)
        ytl = center[1] - (dimensions[1] / 2)
        rect = pg.Rect(xtl, ytl, dimensions[0], dimensions[1])
        pg.draw.rect(self.surface, self._csv_rgb_to_pg_color(color, alpha),
                     rect, width)

        dim_int = tuple(int(x) for x in dimensions)
        self.svg.add(
            self.svg.rect(insert=(int(xtl), int(ytl)), size=dim_int))

    def polygon(self, points, color=Color.DEFAULT, width=1, filled=False,
                antialias=True, alpha=-1):
        """
        Draw a polygon from a list of (x,y)

        color - The object's color as a simple CVColor object, if no value is
                specified the default is used.

        alpha - The alpha blending for the object. If this value is -1 then the
                layer default value is used. A value of 255 means opaque,
                while 0 means transparent.

        width - The
        width in pixels. This does not work if antialiasing is enabled.

        filled -The object is filled in

        antialias - Draw the edges of the object antialiased. Note this does
        not work when the object is filled.
        """
        if filled:
            width = 0
        if not filled:
            if antialias and width == 1:
                pg.draw.aalines(self.surface,
                                self._csv_rgb_to_pg_color(color, alpha), True,
                                points, width)
            else:
                pg.draw.lines(self.surface,
                              self._csv_rgb_to_pg_color(color, alpha), True,
                              points, width)
        else:
            pg.draw.polygon(self.surface, self._csv_rgb_to_pg_color(color,
                                                                    alpha),
                            points, width)
        return None

    def circle(self, center, radius, color=Color.DEFAULT, width=1,
               filled=False, alpha=-1, antialias=True):
        """
        Draw a circle given a location and a radius.

        color - The object's color as a simple CVColor object, if no value is
                specified the default is used.

        alpha - The alpha blending for the object. If this value is -1 then the
                layer default value is used. A value of 255 means opaque,
                while 0 means transparent.

        width - The line width in pixels. This does not work if antialiasing is
                enabled.

        filled -The object is filled in

        Parameters:
            center - Tuple
            radius - Int
            color - Color object or Color Tuple
            width - Int
            filled - Boolean
            alpha - Int
            antialias - Int
        """
        if filled:
            width = 0
        if antialias is False or width > 1 or filled:
            pg.draw.circle(self.surface,
                           self._csv_rgb_to_pg_color(color, alpha),
                           center, int(radius), int(width))
        else:
            pg.gfxdraw.aacircle(self.surface, int(center[0]), int(center[1]),
                                int(radius),
                                self._csv_rgb_to_pg_color(color, alpha))

        cen_int = tuple(int(x) for x in center)
        self.svg.add(self.svg.circle(center=cen_int, r=radius))

        return None

    def ellipse(self, center, dimensions, color=Color.DEFAULT, width=1,
                filled=False, alpha=-1):
        """
        Draw an ellipse given a location and a dimensions.

        color - The object's color as a simple CVColor object, if no value is
                specified the default is used.

        alpha - The alpha blending for the object. If this value is -1 then the
                layer default value is used. A value of 255 means opaque,
                while 0 means transparent.

        width - The line width in pixels. This does not work if antialiasing is
                enabled.

        filled -The object is filled in

        Parameters:
            center - Tuple
            dimensions - Tuple
            color - Color object or Color tuple
            width - Int
            filled - Boolean
            alpha - Int
        """
        if filled:
            width = 0
        rect = pg.Rect(center[0] - (dimensions[0] / 2),
                       center[1] - (dimensions[1] / 2), dimensions[0],
                       dimensions[1])
        pg.draw.ellipse(self.surface,
                        self._csv_rgb_to_pg_color(color, alpha), rect, width)

        cen_int = tuple(int(x) for x in center)
        dim_int = tuple(int(x) for x in dimensions)
        self.svg.add(self.svg.ellipse(center=cen_int, r=dim_int))

        return None

    def bezier(self, points, steps, color=Color.DEFAULT, alpha=-1):
        """
        Draw a bezier curve based on a control point and the a number of stapes

        color - The object's color as a simple CVColor object, if no value is
                specified the default is used.

        alpha - The alpha blending for the object. If this value is -1 then the
                layer default value is used. A value of 255 means opaque,
                while 0 means transparent

        Parameters:
            points - list
            steps - Int
            color - Color object or Color Tuple
            alpha - Int


        """
        pg.gfxdraw.bezier(self.surface, points, steps,
                          self._csv_rgb_to_pg_color(color, alpha))
        return None

    def set_font_bold(self, do_bold):
        """
        This method sets and unsets the current font to be bold.
        """
        self.font_bold = do_bold
        self.font.set_bold(do_bold)
        return None

    def set_font_italic(self, do_italic):
        """
        This method sets and unsets the current font to be italic.
        """
        self.font_italic = do_italic
        self.font.set_italic(do_italic)
        return None

    def set_font_underline(self, do_underline):
        """
        This method sets and unsets the current font to be underlined
        """
        self.font_underline = do_underline
        self.font.set_underline(do_underline)
        return None

    def select_font(self, font_name):
        """
        This method attempts to set the font from a font file. It is advisable
        to use one of the fonts listed by the list_fonts() method. The input
        is a string with the font name.
        """
        full_name = pg.font.match_font(font_name)
        self.font_name = full_name
        self.font = pg.font.Font(self.font_name, self.font_size)
        return None

    @staticmethod
    def list_fonts():
        """
        This method returns a list of strings corresponding to the fonts
        available on the current system.
        """
        return pg.font.get_fonts()

    def set_font_size(self, size):
        """
        This method sets the font size roughly in points. A size of 10 is
        almost too small to read. A size of 20 is roughly 10 pixels high and a
        good choice.

        Parameters:
            sz = Int
        """
        self.font_size = size
        self.font = pg.font.Font(self.font_name, self.font_size)
        return None

    def text(self, text, location, color=Color.DEFAULT, alpha=-1):
        """
        Write the a text string at a given location

        text -  A text string to print.

        location-The location to place the top right corner of the text

        color - The object's color as a simple CVColor object, if no value is
                specified the default is used.

        alpha - The alpha blending for the object. If this value is -1 then the
                layer default value is used. A value of 255 means opaque,
                while 0 means transparent.

        Parameters:
            text - String
            location - Tuple
            color - Color object or Color tuple
            alpha - Int

        """
        if len(text) < 0:
            return None
        tsurface = self.font.render(text, True,
                                    self._csv_rgb_to_pg_color(color, alpha))
        if alpha == -1:
            alpha = self.default_alpha
        #this is going to be slow, dumb no native support.
        #see http://www.mail-archive.com/pygame-users@seul.org/msg04323.html
        # Get access to the alpha band of the image.
        pixels_alpha = pg.surfarray.pixels_alpha(tsurface)
        # Do a floating point multiply, by alpha 100, on each alpha value.
        # Then truncate the values (convert to integer)
        # and copy back into the surface.
        pixels_alpha[...] = (pixels_alpha * (alpha / 255.0)).astype(uint8)
        # Unlock the surface.
        del pixels_alpha
        self.surface.blit(tsurface, location)

        font_style = "font-size: {}px;".format(
            self.font_size - 7)  # Adjust for web
        if self.font_bold:
            font_style += "font-weight: bold;"
        if self.font_italic:
            font_style += "font-style: italic;"
        if self.font_underline:
            font_style += "text-decoration: underline;"
        if self.font_name:
            font_style += "font-family: \"{}\";".format(self.font_name)
        altered_location = (location[0],
                            location[1] + self.text_dimensions(text)[1])
        alt_int = tuple(int(x) for x in altered_location)
        self.svg.add(self.svg.text(text, insert=alt_int, style=font_style))
        return None

    def text_dimensions(self, text):
        """
        The text_dimensions function takes a string and returns the dimensions
        (width, height) of this text being rendered on the screen.
        """
        tsurface = self.font.render(text, True,
                                    self._csv_rgb_to_pg_color(Color.WHITE,
                                                              255))
        return tsurface.get_width(), tsurface.get_height()

    def ez_view_text(self, text, location, fgcolor=Color.WHITE,
                     bgcolor=Color.BLACK):
        """
        ez_view_text works just like text but it sets both the foreground and
        background color and overwrites the image pixels. Use this method to
        make easily viewable text on a dynamic video stream.

        fgcolor - The color of the text.

        bgcolor - The background color for the text are.
        """
        if len(text) < 0:
            return
        alpha = 255
        tsurface = self.font.render(text, True,
                                    self._csv_rgb_to_pg_color(fgcolor, alpha),
                                    self._csv_rgb_to_pg_color(bgcolor, alpha))
        self.surface.blit(tsurface, location)

    def sprite(self, img, pos=(0, 0), scale=1.0, rot=0.0, alpha=255):
        """
        sprite draws a sprite (a second small image) onto the current layer.
        The sprite can be loaded directly from a supported image file like a
        gif, jpg, bmp, or png, or loaded as a surface or SCV image.

        pos - the (x,y) position of the upper left hand corner of the sprite

        scale - a scale multiplier as a float value. E.g. 1.1 makes the sprite
                10% bigger

        rot = a rotation angle in degrees

        alpha = an alpha value 255=opaque 0=transparent.
        """

        if not pg.display.get_init():
            pg.display.init()

        if img.__class__.__name__ == 'str':
            image = pg.image.load(img, "RGB")
        elif img.__class__.__name__ == 'Image':
            image = img.get_pg_surface()
        else:
            image = img  # we assume we have a surface
        image = image.convert(self.surface)
        if rot != 0.00:
            image = pg.transform.rotate(image, rot)
        if scale != 1.0:
            image = pg.transform.scale(image,
                                       (int(image.get_width() * scale),
                                        int(image.get_height() * scale)))
        pixels_alpha = pg.surfarray.pixels_alpha(image)
        pixels_alpha[...] = (pixels_alpha * (alpha / 255.0)).astype(uint8)
        self.surface.blit(image, pos)

    def blit(self, img, coordinates=(0, 0)):
        """
        Blit one image onto the drawing layer at upper left coordinates

        Parameters:
            img - Image
            coordinates - Tuple

        """

        #can we set a color mode so we can do a little bit of masking here?
        self.surface.blit(img.get_pg_surface(), coordinates)

    def replace_overlay(self, overlay):
        """
        This method allows you to set the surface manually.

        Parameters:
            overlay - Pygame Surface
        """
        self.surface = overlay

    #get rid of all drawing
    def clear(self):
        """
        This method removes all of the drawing on this layer (i.e. the layer is
        erased completely)
        """
        self.surface = pg.Surface((int(self.width), int(self.height)),
                                  flags=pg.SRCALPHA)

    def render_to_surface(self, surf):
        """
        Blit this layer to another surface.

        Parameters:
            surf - Pygame Surface
        """
        surf.blit(self.surface, (0, 0))
        return surf

    def render_to_other_layer(self, other_layer):
        """
        Add this layer to another layer.

        Parameters:
            other_layer - Pygame Surface
        """
        other_layer.surface.blit(self.surface, (0, 0))
