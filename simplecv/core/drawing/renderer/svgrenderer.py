import base64
import cv2
import svgwrite as sw

from simplecv.base import ScvException
from simplecv.color import Color
from simplecv.core.drawing.renderer.rendererbase import RendererBase


class SvgRenderer(RendererBase):

    def __init__(self):
        super(SvgRenderer, self).__init__()
        self.width = 0
        self.height = 0
        self.svg = None

    def render(self, layer, image):
        self.width = image.width
        self.height = image.height
        self.svg = sw.Drawing(size=(self.width, self.height))

        # add image to svg first
        self.blit(image)

        # perform operations
        for operation, args, kwargs in layer:
            if not hasattr(self, operation):
                raise ScvException('No such operation in SvgRenderer: {}'.format(operation))

            getattr(self, operation)(*args, **kwargs)

        return self.svg.tostring()

    def _to_svg_color(self, color):
        if color is None:
            color = self.default_color
        return sw.rgb(*color)

    def _to_svg_alpha(self, alpha):
        if alpha == -1:
            alpha = self.default_alpha
        return alpha / 255.0

    def _create_font_style(self):
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
        return font_style

    def line(self, start, stop, color=None, width=1, antialias=True,
             alpha=-1):
        color = self._to_svg_color(color)
        alpha = self._to_svg_alpha(alpha)
        self.svg.add(self.svg.line(start=map(int, start),
                                   end=map(int, stop),
                                   stroke=color,
                                   stroke_width=int(width),
                                   stroke_opacity=alpha))

    def lines(self, points, color=None, antialias=True, alpha=-1,
              width=1):
        lines = zip(points[:-1], points[1:])
        for start, stop in lines:
            self.line(start, stop, color=color, antialias=antialias, alpha=alpha)

    def rectangle(self, top_left, dimensions, color=None, width=1,
                  filled=False, alpha=-1):
        if width < 0:
            filled = True
        if filled:
            width = 0
        color = self._to_svg_color(color)
        alpha = self._to_svg_alpha(alpha)
        if filled:
            self.svg.add(self.svg.rect(insert=map(int, top_left),
                                       size=map(int, dimensions),
                                       fill=color,
                                       fill_opacity=alpha))
        else:
            self.svg.add(self.svg.rect(insert=map(int, top_left),
                                       size=map(int, dimensions),
                                       fill_opacity=0,
                                       stroke=color,
                                       stroke_width=int(width),
                                       stroke_opacity=alpha))

    def polygon(self, points, color=None, width=1, filled=False,
                antialias=True, alpha=-1):
        if width < 0:
            filled = True
        if filled:
            width = 0
        color = self._to_svg_color(color)
        alpha = self._to_svg_alpha(alpha)
        if filled:
            self.svg.add(self.svg.polygon(points=points,
                                          fill=color,
                                          fill_opacity=alpha))
        else:
            self.svg.add(self.svg.polygon(points=points,
                                          fill_opacity=0,
                                          stroke=color,
                                          stroke_width=int(width),
                                          stroke_opacity=alpha))

    def circle(self, center, radius, color=None, width=1,
               filled=False, alpha=-1, antialias=True):
        if width < 0:
            filled = True
        if filled:
            width = 0
        color = self._to_svg_color(color)
        alpha = self._to_svg_alpha(alpha)
        if filled:
            self.svg.add(self.svg.circle(center=map(int, center),
                                         r=int(radius),
                                         fill=color,
                                         fill_opacity=alpha))
        else:
            self.svg.add(self.svg.circle(center=map(int, center),
                                         r=int(radius),
                                         fill_opacity=0,
                                         stroke=color,
                                         stroke_width=int(width),
                                         stroke_opacity=alpha))

    def ellipse(self, center, dimensions, color=None, width=1,
                filled=False, alpha=-1):
        if width < 0:
            filled = True
        if filled:
            width = 0
        color = self._to_svg_color(color)
        alpha = self._to_svg_alpha(alpha)
        if filled:
            self.svg.add(self.svg.ellipse(center=map(int, center),
                                          r=map(int, dimensions),
                                          fill=color,
                                          fill_opacity=alpha))
        else:
            self.svg.add(self.svg.ellipse(center=map(int, center),
                                          r=map(int, dimensions),
                                          fill_opacity=0,
                                          stroke=color,
                                          stroke_width=int(width),
                                          stroke_opacity=alpha))

    def bezier(self, points, steps, color=None, alpha=-1):
        bezier = [('M', points[0][0], points[0][1])] + \
                 [('T', x, y) for x, y in points[1:]]
        color = self._to_svg_color(color)
        alpha = self._to_svg_alpha(alpha)
        self.svg.add(self.svg.path(bezier,
                                   fill_opacity=0,
                                   stroke=color,
                                   stroke_width=1,
                                   stroke_opacity=alpha))

    def text(self, text, pos, color=None, alpha=-1):
        font_style = self._create_font_style()
        color = self._to_svg_color(color)
        alpha = self._to_svg_alpha(alpha)
        self.svg.add(self.svg.text(text,
                                   insert=map(int, pos),
                                   style=font_style,
                                   fill=color,
                                   fill_opacity=alpha))

    def ez_view_text(self, text, pos, fgcolor=Color.WHITE, bgcolor=Color.BLACK):
        # just use text method
        self.text(text=text, pos=pos, color=fgcolor)

    def sprite(self, img, pos=(0, 0), scale=1.0, rot=0.0, alpha=255):
        alpha = self._to_svg_alpha(alpha)
        data = base64.encodestring(cv2.imencode('.png', img)[1].tostring())
        data = 'data:image/png;base64,' + data
        insert_pos = map(lambda a: a / scale, pos)
        sprite = sw.image.Image(data, insert=insert_pos, size=img.size_tuple,
                           opacity=alpha)
        sprite.scale(scale)
        rotate_pos = (pos[0] + (img.width * scale / 2),
                      pos[1] + (img.height * scale / 2))
        sprite.rotate(rot, rotate_pos)
        self.svg.add(sprite)

    def blit(self, img, pos=(0, 0)):
        data = base64.encodestring(cv2.imencode('.png', img)[1].tostring())
        data = 'data:image/png;base64,' + data
        self.svg.add(sw.image.Image(data, insert=pos, size=img.size_tuple))
