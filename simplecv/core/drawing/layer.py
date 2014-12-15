import functools

from simplecv.color import Color


def register_operation(func):
    """ Decorator to register operation within the drawing layer
    """
    @functools.wraps(func)
    def wrapper(dl, *args, **kwargs):
        func(dl, *args, **kwargs)  # run func to perform type checking
        dl.append((func.__name__, args, kwargs))  # add operation to layer
    return wrapper


class DrawingLayer(list):
    """
    DrawingLayer gives you a way to mark up Image classes without changing
    the image data itself.
    """
    def __init__(self, seq=None):
        if seq is None:
            list.__init__(self)
        else:
            for value in seq:
                if (len(value) != 3
                        or not hasattr(self, value[0])  # first should be name of method
                        or not isinstance(value[1], tuple)  # tuple of args
                        or not isinstance(value[2], dict)):  # dict of kwargs
                    raise ValueError('seq is not a DrawingLayer')
            list.__init__(self, seq)

        if len(self) == 0:
            # Set default values to renderer first
            self.set_default_alpha(255)
            self.set_default_color(Color.BLACK)

    def __repr__(self):
        return 'simplecv.DrawingLayer({})'.format(list.__repr__(self))

    def __add__(self, other):
        if isinstance(other, DrawingLayer):
            return DrawingLayer(list.__add__(self, other))
        else:
            raise ValueError('DrawingLayer.__add__: other operand should be '
                             'a DrawingLayer instance')

    def __getitem__(self, item):
        result = list.__getitem__(self, item)
        if isinstance(item, slice):
            return DrawingLayer(result)
        else:
            return result

    def clear(self):
        """
        This method removes all of the drawing on this layer (i.e. the layer is
        erased completely)
        """
        del self[:]

    def add_opeartion(self, operation, *args, **kwargs):
        self.append((operation, args, kwargs))

    def contains_drawing_operations(self):
        for operation, args, kwargs in self:
            if not (operation.startswith('set_')
                    or operation == 'select_font'):
                return True
        return False

    @register_operation
    def set_default_alpha(self, alpha):
        """
        This method sets the default alpha value for all methods called on this
        layer. The default value starts out at 255 which is completely
         transparent.
        """
        if not 0 <= alpha <= 255:
            raise ValueError('Alpha should be from 0 to 255')

    @register_operation
    def set_default_color(self, color):
        """
        This method sets the default rendering color.

        Parameters:
            color - Color object or Color Tuple
        """
        pass

    @register_operation
    def select_font(self, name):
        """
        This method attempts to set the font from a font file. It is advisable
        to use one of the fonts listed by the list_fonts() method. The input
        is a string with the font name.
        """
        pass

    @register_operation
    def set_font_size(self, size):
        """
        This method sets the font size roughly in points. A size of 10 is
        almost too small to read. A size of 20 is roughly 10 pixels high and a
        good choice.

        Parameters:
            sz = Int
        """
        pass

    @register_operation
    def set_font_bold(self, value):
        """
        This method sets and unsets the current font to be bold.
        """
        pass

    @register_operation
    def set_font_italic(self, value):
        """
        This method sets and unsets the current font to be italic.
        """
        pass

    @register_operation
    def set_font_underline(self, value):
        """
        This method sets and unsets the current font to be underlined
        """
        pass

    @register_operation
    def line(self, start, stop, color=None, width=1, antialias=True,
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
        pass

    @register_operation
    def lines(self, points, color=None, antialias=True, alpha=-1,
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
        if len(points) < 2:
           raise ValueError('must be more than 2 points')

    @register_operation
    def rectangle(self, top_left, dimensions, color=None, width=1,
                  filled=False, alpha=-1, antialias=True):
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
        pass

    @register_operation
    def polygon(self, points, holes=(), color=None, width=1, filled=False,
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
        if len(points) < 2:
           raise ValueError('must be more than 2 points')

    @register_operation
    def circle(self, center, radius, color=None, width=1,
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
        pass

    @register_operation
    def ellipse(self, center, dimensions, color=None, width=1,
                filled=False, alpha=-1, antialias=True):
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
        pass

    @register_operation
    def bezier(self, points, steps, color=None, alpha=-1):
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
        pass

    @register_operation
    def text(self, text, pos, color=None, alpha=-1):
        """
        Write the a text string at a given location

        text -  A text string to print.

        pos  -  The location to place the top right corner of the text

        color - The object's color as a simple CVColor object, if no value is
                specified the default is used.

        font_name - name of the font

        underline - makes font to be underlined

        italic - makes font to be italic

        bold - makes font to be bold

        size  - the font size

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
            raise ValueError('text should not be empty')

    @register_operation
    def ez_view_text(self, text, pos, fgcolor=Color.WHITE, bgcolor=Color.BLACK):
        """
        ez_view_text works just like text but it sets both the foreground and
        background color and overwrites the image pixels. Use this method to
        make easily viewable text on a dynamic video stream.

        fgcolor - The color of the text.

        bgcolor - The background color for the text are.
        """
        if len(text) < 0:
            raise ValueError('text should not be empty')

    @register_operation
    def sprite(self, img, pos=(0, 0), scale=1.0, rot=0.0, alpha=-1):
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
        pass

    @register_operation
    def blit(self, img, pos=(0, 0)):
        """
        Blit one image onto the drawing layer at upper left coordinates

        Parameters:
            img - Image
            coordinates - Tuple

        """
        pass

    def rectangle_to_pts(self, pt0, pt1, color=None, width=1,
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
        self.rectangle(top_left=(x, y), dimensions=(w, h), color=color,
                       width=width, filled=filled, alpha=alpha)

    def centered_rectangle(self, center, dimensions, color=None,
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
        xtl = center[0] - (dimensions[0] / 2)
        ytl = center[1] - (dimensions[1] / 2)
        self.rectangle(top_left=(xtl, ytl), dimensions=dimensions, color=color,
                       width=width, filled=filled, alpha=alpha)

    def grid(self, size, dimensions=(10, 10), color=(0, 0, 0), width=1,
             antialias=True, alpha=-1):
        """
        **SUMMARY**

        Draw a grid on the layer

        **PARAMETERS**

        * *size* - size of the grid
        * *dimensions* - No of rows and cols as an (rows,xols) tuple or list.
        * *color* - Grid's color as a tuple or list.
        * *width* - The grid line width in pixels.
        * *antialias* - Draw an antialiased object
        * *aplha* - The alpha blending for the object. If this value is -1 then
          the layer default value is used. A value of 255 means opaque, while
          0 means transparent.

        **RETURNS**

        Returns the index of the drawing layer of the grid

        **EXAMPLE**

        >>>> img = Image('something.png')
        >>>> img.dl().grid(img.size_tuple, [20, 20], (255, 0, 0))
        """
        w, h = size
        try:
            step_row = w / dimensions[0]
            step_col = h / dimensions[1]
        except ZeroDivisionError:
            raise ValueError('dimensions contains zero')

        for i in range(step_row, w, step_row):
            self.line((0, i), (w, i), color, width, antialias, alpha)

        for j in range(step_col, h, step_col):
            self.line((j, 0), (j, h), color, width, antialias, alpha)
