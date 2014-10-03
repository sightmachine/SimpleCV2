import cv2
import numpy as np

from simplecv.base import ScvException
from simplecv.color import Color
from simplecv.core.drawing.renderer.rendererbase import RendererBase


class OpenCvRenderer(RendererBase):

    def __init__(self):
        super(OpenCvRenderer, self).__init__()
        self.image = None

    def render(self, layer, image):
        self.image = image.to_bgr()

        # perform operations
        for operation, args, kwargs in layer:
            if not hasattr(self, operation):
                raise ScvException('No such operation in SvgRenderer: {}'.format(operation))

            getattr(self, operation)(*args, **kwargs)

        return self.image

    def _to_cv_color(self, color):
        if color is None:
            color = self.default_color
        return color[::-1]

    def _to_cv_alpha(self, alpha):
        if alpha == -1:
            alpha = self.default_alpha
        return alpha / 255.0

    def _to_int_tuple(self, (x, y)):
        return (int(x), int(y))

    def line(self, start, stop, color=None, width=1, antialias=True,
             alpha=-1):
        color = self._to_cv_color(color)
        alpha = self._to_cv_alpha(alpha)
        line_type = cv2.CV_AA if antialias else 8
        start = self._to_int_tuple(start)
        stop = self._to_int_tuple(stop)
        if alpha != 1:
            overlay = self.image.copy()
            cv2.line(overlay, start, stop,color=color,
                     thickness=int(width), lineType=line_type)
            cv2.addWeighted(overlay, alpha, self.image,
                            1 - alpha, 0, self.image)
        else:
            cv2.line(self.image, start, stop, color=color,
                     thickness=int(width), lineType=line_type)

    def lines(self, points, color=None, antialias=True, alpha=-1,
              width=1):
        color = self._to_cv_color(color)
        alpha = self._to_cv_alpha(alpha)
        line_type = cv2.CV_AA if antialias else 8
        if alpha != 1:
            overlay = self.image.copy()
            cv2.polylines(overlay, np.int32([points]), isClosed=False,
                          color=color, thickness=width, lineType=line_type)
            cv2.addWeighted(overlay, alpha, self.image,
                            1 - alpha, 0, self.image)
        else:
            cv2.polylines(self.image, np.int32([points]), isClosed=False,
                          color=color, thickness=width, lineType=line_type)

    def rectangle(self, top_left, dimensions, color=None, width=1,
                  filled=False, alpha=-1, antialias=True):
        if filled:
            width = -1
        color = self._to_cv_color(color)
        alpha = self._to_cv_alpha(alpha)
        line_type = cv2.CV_AA if antialias else 8
        top_left = self._to_int_tuple(top_left)
        dimensions = self._to_int_tuple(dimensions)
        bottom_right = tuple(map(lambda (a, b): a + b, zip(top_left, dimensions)))
        if alpha != 1:
            overlay = self.image.copy()
            cv2.rectangle(overlay, top_left, bottom_right, color=color,
                          thickness=int(width), lineType=line_type)
            cv2.addWeighted(overlay, alpha, self.image,
                            1 - alpha, 0, self.image)
        else:
            cv2.rectangle(self.image, top_left, bottom_right, color=color,
                          thickness=int(width), lineType=line_type)

    def polygon(self, points, holes=(), color=None, width=1, filled=False,
                antialias=True, alpha=-1):
        if width < 0:
            filled = True
        if filled:
            width = -1
        color = self._to_cv_color(color)
        line_type = cv2.CV_AA if antialias else 8
        if filled:
            alpha = self.default_alpha if alpha == -1 else alpha
            mask = self.image.get_empty(1)
            cv2.fillPoly(mask, np.int32([points]),
                         color=alpha, lineType=line_type)
            for hole in holes:
                cv2.fillPoly(mask, np.int32([hole]),
                             color=0, lineType=line_type)
            self.blit_color_alpha(self.image, color, mask)
        else:
            alpha = self._to_cv_alpha(alpha)
            if alpha != 1:
                overlay = self.image.copy()
                cv2.polylines(overlay, np.int32([points]), isClosed=True,
                              color=color, thickness=width,
                              lineType=line_type)
                for hole in holes:
                    cv2.polylines(overlay, np.int32([hole]), isClosed=True,
                                  color=color, thickness=width,
                                  lineType=line_type)
                cv2.addWeighted(overlay, alpha, self.image,
                                1 - alpha, 0, self.image)
            else:
                cv2.polylines(self.image, np.int32([points]), isClosed=True,
                              color=color, thickness=width,
                              lineType=line_type)
                for hole in holes:
                    cv2.polylines(self.image, np.int32([hole]), isClosed=True,
                                  color=color, thickness=width,
                                  lineType=line_type)

    def circle(self, center, radius, color=None, width=1,
               filled=False, alpha=-1, antialias=True):
        if width < 0:
            filled = True
        if filled:
            width = -1
        color = self._to_cv_color(color)
        alpha = self._to_cv_alpha(alpha)
        line_type = cv2.CV_AA if antialias else 8
        center = self._to_int_tuple(center)
        if alpha != 1:
            overlay = self.image.copy()
            cv2.circle(overlay, center, int(radius), color=color,
                       thickness=int(width), lineType=line_type)
            cv2.addWeighted(overlay, alpha, self.image,
                            1 - alpha, 0, self.image)
        else:
            cv2.circle(self.image, center, int(radius), color=color,
                       thickness=int(width), lineType=line_type)

    def ellipse(self, center, dimensions, color=None, width=1,
                filled=False, alpha=-1, antialias=True):
        if width < 0:
            filled = True
        if filled:
            width = -1
        color = self._to_cv_color(color)
        alpha = self._to_cv_alpha(alpha)
        line_type = cv2.CV_AA if antialias else 8
        center = self._to_int_tuple(center)
        dimensions = int(dimensions[0] / 2), int(dimensions[1] / 2)
        if alpha != 1:
            overlay = self.image.copy()
            cv2.ellipse(overlay, center, dimensions,
                        0, 0, 360, color=color,
                        lineType=line_type, thickness=width)
            cv2.addWeighted(overlay, alpha, self.image,
                            1 - alpha, 0, self.image)
        else:
            cv2.ellipse(self.image, center, dimensions,
                        0, 0, 360, color=color,
                        lineType=line_type, thickness=width)

    def bezier(self, points, steps, color=None, alpha=-1):
        raise NotImplementedError

    def text(self, text, pos, color=None, alpha=-1):
        color = self._to_cv_color(color)
        alpha = self._to_cv_alpha(alpha)
        pos = self._to_int_tuple(pos)
        text_size = self.font_size / 45.0
        if alpha != 1:
            overlay = self.image.copy()
            cv2.putText(overlay, text, pos, cv2.FONT_HERSHEY_TRIPLEX,
                        text_size, color, thickness=1, lineType=cv2.CV_AA)
            cv2.addWeighted(overlay, alpha, self.image,
                            1 - alpha, 0, self.image)
        else:
            cv2.putText(self.image, text, pos, cv2.FONT_HERSHEY_TRIPLEX,
                        text_size, color, thickness=1, lineType=cv2.CV_AA)

    def ez_view_text(self, text, pos, fgcolor=Color.WHITE, bgcolor=Color.BLACK):
        fgcolor = self._to_cv_color(fgcolor)
        bgcolor = self._to_cv_color(bgcolor)
        text_size = self.font_size / 45.0
        pos = self._to_int_tuple(pos)
        font = cv2.FONT_HERSHEY_TRIPLEX
        txsz, h = cv2.getTextSize(text, font, text_size, thickness=1)
        cv2.rectangle(self.image, (pos[0], pos[1] + h),
                      (pos[0] + txsz[0], pos[1] - (h + txsz[1])),
                      color=bgcolor, thickness=-1, lineType=cv2.CV_AA)
        cv2.putText(self.image, text, pos, font, text_size,
                    fgcolor, thickness=1, lineType=cv2.CV_AA)

    def sprite(self, img, pos=(0, 0), scale=1.0, rot=0.0, alpha=-1):
        pos = self._to_int_tuple(pos)
        alpha = self._to_cv_alpha(alpha)
        sprite = img.to_bgra().scale(scale).rotate(rot, fixed=False)
        x, y = (pos[0] - sprite.shape[0] / 2,
                pos[1] - sprite.shape[1] / 2)
        if alpha != 1:
            sprite[:, :, 3] = cv2.multiply(sprite[:, :, 3], alpha)
        img_roi = self.image[y:y + sprite.shape[0], x:x + sprite.shape[1]]
        self.blit_with_alpha(img_roi, sprite, sprite[:, :, 3])

    def blit(self, img, pos=(0, 0)):
        x, y = pos
        self.image[y:y + img.shape[0], x:x + img.shape[1]] = img

    @staticmethod
    def blit_with_alpha(img1, img2, alpha_mask):
        alpha_ratio = cv2.divide(alpha_mask, 255.0, dtype=cv2.CV_32F)
        for c in range(0, 3):
              img1[:,:,c] = cv2.add(cv2.multiply(img1[:,:,c],
                                                 cv2.subtract(1, alpha_ratio, dtype=cv2.CV_32F),
                                                 dtype=cv2.CV_32F),
                                    cv2.multiply(img2[:,:,c], alpha_ratio, dtype=cv2.CV_32F))

    @staticmethod
    def blit_color_alpha(img1, color, alpha_mask):
        alpha_ratio = cv2.divide(alpha_mask, 255.0, dtype=cv2.CV_32F)
        for c, value in enumerate(color):
              img1[:,:,c] = cv2.add(cv2.multiply(img1[:,:,c],
                                                 cv2.subtract(1, alpha_ratio, dtype=cv2.CV_32F),
                                                 dtype=cv2.CV_32F),
                                    cv2.multiply(value, alpha_ratio, dtype=cv2.CV_32F))