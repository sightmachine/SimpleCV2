import numpy as np
import pygame as pg
from pygame import gfxdraw

from simplecv.base import ScvException
from simplecv.color import Color
from simplecv.core.drawing.renderer.rendererbase import RendererBase
from simplecv.factory import Factory


class PyGameRenderer(RendererBase):

    def __init__(self):
        super(PyGameRenderer, self).__init__()

        if not pg.font.get_init():
            pg.font.init()

        self.width = 0
        self.height = 0
        self.clear_color = pg.Color(0, 0, 0, 0)
        self.surface = None

    def render(self, layer, image):
        self.width = image.width
        self.height = image.height
        self.surface = pg.Surface((self.width, self.height), flags=pg.SRCALPHA)
        self.surface.fill(self.clear_color)

        # perform operations
        for operation, args, kwargs in layer:
            if not hasattr(self, operation):
                raise ScvException('No such operation in PyGameRenderer: {}'.format(operation))

            getattr(self, operation)(*args, **kwargs)

        # blit surface
        img_surf = image.get_pg_surface().copy()
        img_surf.blit(self.surface, (0, 0))
        return Factory.Image(source=img_surf)

    def _csv_rgb_to_pg_color(self, color, alpha=-1):
        if alpha == -1:
            alpha = self.default_alpha

        if color is None:
            color = self.default_color
        return pg.Color(color[0], color[1], color[2], alpha)

    @staticmethod
    def list_fonts():
        """
        This method returns a list of strings corresponding to the fonts
        available on the current system.
        """
        return pg.font.get_fonts()

    def _create_font(self):
        if self.font_name is not None:
            self.font_name = pg.font.match_font(self.font_name)
        font = pg.font.Font(self.font_name, self.font_size)
        font.set_underline(self.font_underline)
        font.set_bold(self.font_bold)
        font.set_italic(self.font_italic)
        return font

    def line(self, start, stop, color=None, width=1, antialias=True,
             alpha=-1):
        width = int(width)
        start = (int(start[0]), int(start[1]))
        stop = (int(stop[0]), int(stop[1]))
        if antialias and width == 1:
            pg.draw.aaline(self.surface,
                           self._csv_rgb_to_pg_color(color, alpha),
                           start, stop, width)
        else:
            pg.draw.line(self.surface,
                         self._csv_rgb_to_pg_color(color, alpha),
                         start, stop, width)

    def lines(self, points, color=None, antialias=True, alpha=-1,
              width=1):
        if antialias and width == 1:
            pg.draw.aalines(self.surface,
                            self._csv_rgb_to_pg_color(color, alpha),
                            0, points, width)
        else:
            pg.draw.lines(self.surface,
                          self._csv_rgb_to_pg_color(color, alpha),
                          0, points, width)

    def rectangle(self, top_left, dimensions, color=None, width=1,
                  filled=False, alpha=-1):
        if width < 0:
            filled = True
        if filled:
            width = 0
        rect = pg.Rect((int(top_left[0]), int(top_left[1])),
                       (int(dimensions[0]), int(dimensions[1])))
        pg.draw.rect(self.surface, self._csv_rgb_to_pg_color(color, alpha),
                     rect, int(width))

    def polygon(self, points, color=None, width=1, filled=False,
                antialias=True, alpha=-1):
        if width < 0:
            filled = True
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

    def circle(self, center, radius, color=None, width=1,
               filled=False, alpha=-1, antialias=True):
        width = int(width)
        radius = int(radius)
        center = int(center[0]), int(center[1])
        if width < 0:
            filled = True
        if filled:
            width = 0
        if antialias is False or width > 1 or filled:
            pg.draw.circle(self.surface,
                           self._csv_rgb_to_pg_color(color, alpha),
                           center, radius, width)
        else:
            pg.gfxdraw.aacircle(self.surface, center[0], center[1],
                                radius,
                                self._csv_rgb_to_pg_color(color, alpha))

    def ellipse(self, center, dimensions, color=None, width=1,
                filled=False, alpha=-1):
        if width < 0:
            filled = True
        if filled:
            width = 0
        rect = pg.Rect(int(center[0] - (dimensions[0] / 2)),
                       int(center[1] - (dimensions[1] / 2)),
                       int(dimensions[0]), int(dimensions[1]))
        pg.draw.ellipse(self.surface,
                        self._csv_rgb_to_pg_color(color, alpha),
                        rect, int(width))

    def bezier(self, points, steps, color=None, alpha=-1):
        pg.gfxdraw.bezier(self.surface, points, steps,
                          self._csv_rgb_to_pg_color(color, alpha))

    def text(self, text, pos, color=None, alpha=-1):
        font = self._create_font()
        tsurface = font.render(text, True,
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
        pixels_alpha[...] = (pixels_alpha * (alpha / 255.0)).astype(np.uint8)
        # Unlock the surface.
        del pixels_alpha
        self.surface.blit(tsurface, pos)

    def ez_view_text(self, text, pos, fgcolor=Color.WHITE, bgcolor=Color.BLACK):
        alpha = 255
        font = self._create_font()
        tsurface = font.render(text, True,
                               self._csv_rgb_to_pg_color(fgcolor, alpha),
                               self._csv_rgb_to_pg_color(bgcolor, alpha))
        self.surface.blit(tsurface, pos)

    def sprite(self, img, pos=(0, 0), scale=1.0, rot=0.0, alpha=255):
        if not pg.display.get_init():
            pg.display.init()

        if isinstance(img, basestring):
            image = pg.image.load(img, "RGB")
        elif isinstance(img, Factory.Image):
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
        pixels_alpha[...] = (pixels_alpha * (alpha / 255.0)).astype(np.uint8)
        self.surface.blit(image, pos)

    def blit(self, img, pos=(0, 0)):
        #can we set a color mode so we can do a little bit of masking here?
        self.surface.blit(img.get_pg_surface(), pos)
