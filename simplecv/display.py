# SimpleCV Display Library
#
# This library is used to draw images on

import os
import pygame as pg
import numpy as np

from simplecv import DATA_DIR
from simplecv.factory import Factory

PYGAME_INITIALIZED = False


class Display(object):
    """
    **SUMMARY**

    WindowStream opens a window (Pygame Display Surface) to which you can write
    images. The default resolution is 640, 480 -- but you can also specify 0,0
    which will maximize the display. Flags are pygame constants, including:


    By default display will attempt to scale the input image to fit
    neatly on the screen with minimal distorition. This means that
    if the aspect ratio matches the screen it will scale cleanly.
    If your image does not match the screen aspect ratio we will
    scale it to fit nicely while maintining its natural aspect ratio.

    Because SimpleCV performs this scaling there are two sets of input mouse
    coordinates, the (mouse_x,mouse_y) which scale to the image, and
    (mouse_raw_x, mouse_raw_y) which do are the actual screen coordinates.

    * pygame.FULLSCREEN: create a fullscreen display.
    * pygame.DOUBLEBUF: recommended for HWSURFACE or OPENGL.
    * pygame.HWSURFACE: hardware accelerated, only in FULLSCREEN.
    * pygame.OPENGL: create an opengl renderable display.
    * pygame.RESIZABLE: display window should be sizeable.
    * pygame.NOFRAME: display window will have no border or controls.

    Display should be used in a while loop with the is_done() method, which
    checks events and sets the following internal state controls:

    * mouse_x: the x position of the mouse cursor on the input image.
    * mouse_y: the y position of the mouse curson on the input image.
    * mouse_raw_x: The position of the mouse on the screen.
    * mouse_raw_y: The position of the mouse on the screen.

    **NOTES**

    The mouse position on the screen is not the mouse position on the image.
    If you are trying to draw on the image or take in coordinates use mouse_x
    and mouse_y as these values are scaled along with the image.

    * mouse_left: the state of the left button.
    * mouse_right: the state of the right button.
    * mouse_middle: the state of the middle button.
    * mouse_wheel_up: scroll wheel has been moved up.
    * mouse_wheel_down: the wheel has been clicked
                        towards the bottom of the mouse

    **EXAMPLE**

    >>> # create a new display to draw images on
    >>> display = Display(resolution=(800, 600))
    >>> cam = Camera()  # initialize the camera
    >>> done = False # setup boolean to stop the program
    >>> while not display.is_done():
    >>> # get image, flip it so it looks mirrored, save to display
    >>> cam.get_image().flip_horizontal().save(display)
    >>> # Let the program sleep for 1ms so the computer can do other things
    >>> time.sleep(0.01)
    >>> if display.mouse_left:
    >>>     display.done = True

    """

    def __repr__(self):
        return "<simplecv.Display Object resolution:(%s), \
                Image Resolution: (%d, %d) at memory location: (%s)>" \
                % (self.resolution, self.imgw, self.imgh, hex(id(self)))

    def __init__(self, resolution=(640, 480), flags=0, title="SimpleCV",
                 displaytype='standard', headless=False):
        """
        **SUMMARY**

        This is the generic display object. You are able to set the display
        type. The standard display type will pop up a window. The notebook
        display type is to be used in conjunction with IPython Notebooks
        this is so it is web based.  If you have IPython Notebooks installed
        you just need to start IPython Notebooks and open in your browser.

        **PARAMETERS**

        * *resolution* - the size of the diplay in pixels.
        * *flags* - ???
        * *title* - the title bar on the display.
        * *displaytype* - The type of display. Options are as follows:

          * 'standard' - A pygame window.
          * 'notebook' - Ipython Web Notebook output

        * *headless* - If False we ignore healess mode.
                       If true all rendering is suspended.

        **EXAMPLE**

        Once in IPython you can do the following:

        >>> from simplecv.display import Display
        >>> from simplecv.image import Image
        >>> disp = Display(displaytype='notebook')
        >>> img = Image('simplecv')
        >>> img.save(disp)

        """
        self.resolution = None
        self.sourceresolution = ''
        self.sourceoffset = ''
        self.screen = ''
        self.done = False
        self.mouse_x = 0  # These are the scaled mouse values,
        self.mouse_y = 0  # if you want to do image manipulation use these.
        self.mouse_left = 0
        self.mouse_middle = 0
        self.mouse_right = 0
        self.mouse_wheel_up = 0
        self.mouse_wheel_down = 0
        self.imgw = 0
        self.imgh = 0

        global PYGAME_INITIALIZED

        if headless:
            os.environ["SDL_VIDEODRIVER"] = "dummy"

        if not PYGAME_INITIALIZED:
            if not displaytype == 'notebook':
                pg.init()
            PYGAME_INITIALIZED = True
        self.xscale = 1.0
        self.yscale = 1.0
        self.xoffset = 0
        self.yoffset = 0
        self.last_left_button = 0
        self.last_right_button = 0
        self.left_button_down = None
        self.left_button_up = None
        self.right_button_down = None
        self.right_button_up = None
        self.pressed = None
        self.displaytype = displaytype
        # NOTE: NO PYGAME CALLS SHOULD BE MADE IN INIT AS THEY KILL
        # THE DISPLAY IN IPYTHON NOTEBOOKS

        # Raw x and y are the actual position on the screen
        # versus the position on the image.
        self.mouse_raw_x = 0
        self.mouse_raw_y = 0

        self.resolution = resolution
        if not displaytype == 'notebook':
            self.screen = pg.display.set_mode(resolution, flags)
        scv_png = 'simplecv.png'
        # checks if simplecv.png exists
        if os.path.isfile(os.path.join(DATA_DIR, 'sampleimages', scv_png)):
            scv_logo = Factory.Image("simplecv").scale(32, 32)
            pg.display.set_icon(scv_logo.get_pg_surface())
        if flags != pg.FULLSCREEN and flags != pg.NOFRAME:
            pg.display.set_caption(title)

    def left_button_up_position(self):
        """
        **SUMMARY**

        Returns the position where the left mouse button went up.

        .. warning::
          You must call :py:meth:`check_events` or :py:meth:`is_done`
          in your main display loop for this method to work.

        **RETURNS**

        An (x,y) mouse postion tuple where the mouse went up.

        **EXAMPLE**

        >>> disp = Display((600, 800))
        >>> cam = Camera()
        >>> while disp.is_not_done():
        >>>   img = cam.get_image()
        >>>   dwn = disp.left_button_down_position()
        >>>   up = disp.left_button_up_position()
        >>>   if up is not None and dwn is not None:
        >>>     bb = disp.points_to_bounding_box(up, dwn)
        >>>     img.draw_rectangle(bb[0], bb[1], bb[2], bb[3])
        >>>   img.save(disp)

        **SEE ALSO**

        :py:meth:`right_button_up_postion`
        :py:meth:`left_button_down_postion`
        :py:meth:`right_button_down_postion`
        :py:meth:`points_to_bounding_box`

        """
        return self.left_button_up

    def left_button_down_position(self):
        """
        **SUMMARY**

        Returns the position where the left mouse button went down.

        .. warning::
          You must call :py:meth:`check_events` or :py:meth:`is_done`
          in your main display loop for this method to work.


        **RETURNS**

        An (x,y) mouse postion tuple where the mouse went up.

        **EXAMPLE**

        >>> disp = Display((600, 800))
        >>> cam = Camera()
        >>> while disp.is_not_done():
        >>>   img = cam.get_image()
        >>>   dwn = disp.left_button_down_position()
        >>>   up = disp.left_button_up_position()
        >>>   if up is not None and dwn is not None:
        >>>     bb = disp.points_to_bounding_box(up, dwn)
        >>>     img.draw_rectangle(bb[0], bb[1], bb[2], bb[3])
        >>>   img.save(disp)

        **SEE ALSO**

        :py:meth:`left_button_up_postion`
        :py:meth:`right_button_up_postion`
        :py:meth:`right_button_down_postion`
        :py:meth:`points_to_bounding_box`
        :py:meth:`check_events`

        """
        return self.left_button_down

    def right_button_up_position(self):
        """
        **SUMMARY**

        Returns the position where the right mouse button went up.

        .. warning::
          You must call :py:meth:`check_events` or :py:meth:`is_done`
          in your main display loop for this method to work.


        **RETURNS**

        An (x,y) mouse postion tuple where the mouse went up.

        **EXAMPLE**

        >>> disp = Display((600, 800))
        >>> cam = Camera()
        >>> while disp.is_not_done():
        >>>   img = cam.get_image()
        >>>   dwn = disp.right_button_down_position()
        >>>   up = disp.right_button_up_position()
        >>>   if up is not None and dwn is not None:
        >>>     bb = disp.points_to_bounding_box(up, dwn)
        >>>     img.draw_rectangle(bb[0], bb[1], bb[2], bb[3])
        >>>   img.save(disp)


        >>> disp = Display((600, 800))
        >>> dwn = disp.right_button_down_position()
        >>> up = disp.right_button_up_position()
        >>> bb = disp.points_to_bounding_box(up, dwn)
        >>> #draw bb

        **SEE ALSO**

        :py:meth:`left_button_up_postion`
        :py:meth:`left_button_down_postion`
        :py:meth:`right_button_down_postion`
        :py:meth:`points_to_bounding_box`
        :py:meth:`check_events`

        """
        return self.right_button_up

    def right_button_down_position(self):
        """
        **SUMMARY**

        Returns the position where the right mouse button went down.

        .. warning::
          You must call :py:meth:`check_events` or :py:meth:`is_done`
          in your main display loop for this method to work.

        **RETURNS**

        An (x,y) mouse postion tuple where the mopuse went down.

        **EXAMPLE**

        >>> disp = Display((600, 800))
        >>> cam = Camera()
        >>> while disp.is_not_done():
        >>>   img = cam.get_image()
        >>>   dwn = disp.right_button_down_position()
        >>>   up = disp.right_button_up_position()
        >>>   if up is not None and dwn is not None:
        >>>     bb = disp.points_to_bounding_box(up, dwn)
        >>>     img.draw_rectangle(bb[0], bb[1], bb[2], bb[3])
        >>>   img.save(disp)

        **SEE ALSO**

        :py:meth:`left_button_up_postion`
        :py:meth:`left_button_down_postion`
        :py:meth:`right_button_down_postion`
        :py:meth:`points_to_bounding_box`
        :py:meth:`check_events`

        """
        return self.right_button_down

    @staticmethod
    def points_to_bounding_box(pt0, pt1):
        """
        **SUMMARY**

        Given two screen cooridnates return the bounding box in x,y,w,h format.
        This is helpful for drawing regions on the display.

        **RETURNS**

        The bounding box from two coordinates as a ( x,y,w,h) tuple.

        **EXAMPLE**

        >>> disp = Display((600, 800))
        >>> cam = Camera()
        >>> while disp.is_not_done():
        >>>   img = cam.get_image()
        >>>   dwn = disp.left_button_down_position()
        >>>   up = disp.left_button_up_position()
        >>>   if up is not None and dwn is not None:
        >>>     bb = disp.points_to_bounding_box(up, dwn)
        >>>     img.draw_rectangle(bb[0], bb[1], bb[2], bb[3])
        >>>   img.save(disp)


        **SEE ALSO**

        :py:meth:`left_button_up_postion`
        :py:meth:`left_button_down_postion`
        :py:meth:`right_button_down_postion`
        :py:meth:`right_button_up_postion`
        :py:meth:`check_events`
        """
        xmax = np.max((pt0[0], pt1[0]))
        xmin = np.min((pt0[0], pt1[0]))
        ymax = np.max((pt0[1], pt1[1]))
        ymin = np.min((pt0[1], pt1[1]))
        return xmin, ymin, xmax-xmin, ymax-ymin

    def write_frame(self, img, fit=True):
        """
        **SUMMARY**

        write_frame copies the given Image object to the display,
        you can also use Image.save()

        Write frame trys to fit the image to the display with the minimum
        ammount of distortion possible. When fit=True write frame will decide
        how to scale the image such that the aspect ratio is maintained and
        the smallest amount of distorition possible is completed.
        This means the axis that has the minimum scaling needed will be shrunk
        or enlarged to match the display.


        **PARAMETERS**

        * *img* -  the SimpleCV image to save to the display.
        * *fit* - When fit=False write frame will crop and
                  center the image as best it can.
          If the image is too big it is cropped and centered. If it is too
          small it is centered. If it is too big along one axis that axis is
          cropped and the other axis is centered if necessary.


        **RETURNS**

        Nothing.

        **EXAMPLE**

        >>> img = Image("lenna")
        >>> disp = Display((512, 512))
        >>> disp.write_frame(img)


        """
        # Grrrrr we're going to need to re-write this functionality
        # So if the image is the right size do nothing
        # if the image has a 'nice' scale factor we should scale it
        # e.g. 800x600=>640x480
        # if( fit )
        #   if one axis is too big -> scale it down to fit
        #   if both axes are too big and they don't match eg 800x800 img
        #      and 640x480 screen => scale to 400x400 and center
        #   if both axis are too small -> scale the biggest to fill
        #   if one axis is too small and one axis is alright
        #      we center along the too small axis
        # else(!fit)
        #   if one / both axis is too big - crop it
        #   if one / both too small - center along axis
        #
        # this is getting a little long. Probably needs to be refactored.
        wndw_ar = float(self.resolution[0])/float(self.resolution[1])
        img_ar = float(img.width)/float(img.height)
        self.sourceresolution = img.size
        self.sourceoffset = (0, 0)
        self.imgw = img.width
        self.imgh = img.height
        self.xscale = 1.0
        self.yscale = 1.0
        self.xoffset = 0
        self.yoffset = 0
        if img.size == self.resolution:  # we have to resize
            pgsurf = img.get_pg_surface()
            self.screen.blit(pgsurf, pgsurf.get_rect())
            pg.display.flip()
        elif img_ar == wndw_ar:
            self.xscale = (float(img.width)/float(self.resolution[0]))
            self.yscale = (float(img.height)/float(self.resolution[1]))
            img = img.scale(self.resolution[0], self.resolution[1])
            pgsurf = img.get_pg_surface()
            self.screen.blit(pgsurf, pgsurf.get_rect())
            pg.display.flip()
        elif fit:
            #scale factors
            wscale = (float(img.width)/float(self.resolution[0]))
            hscale = (float(img.height)/float(self.resolution[1]))

            if wscale > 1:  # we're shrinking what is the percent reduction
                wscale = 1 - (1.0/wscale)
            else:  # we need to grow the image by a percentage
                wscale = 1.0-wscale

            if hscale > 1:
                hscale = 1 - (1.0/hscale)
            else:
                hscale = 1.0 - hscale

            if wscale == 0:  # if we can get away with not scaling do that
                targetx = 0
                targety = (self.resolution[1]-img.height)/2
                targetw = img.width
                targeth = img.height
                pgsurf = img.get_pg_surface()
            elif hscale == 0:  # if we can get away with not scaling do that
                targetx = (self.resolution[0]-img.width)/2
                targety = 0
                targetw = img.width
                targeth = img.height
                pgsurf = img.get_pg_surface()
            elif wscale < hscale:  # the width has less distortion
                sfactor = float(self.resolution[0])/float(img.width)
                targetw = int(float(img.width)*sfactor)
                targeth = int(float(img.height)*sfactor)
                if targetw > self.resolution[0] \
                        or targeth > self.resolution[1]:
                    # aw shucks that still didn't work do the other way instead
                    sfactor = float(self.resolution[1])/float(img.height)
                    targetw = int(float(img.width)*sfactor)
                    targeth = int(float(img.height)*sfactor)
                    targetx = (self.resolution[0]-targetw)/2
                    targety = 0
                else:
                    targetx = 0
                    targety = (self.resolution[1]-targeth)/2
                img = img.scale(targetw, targeth)
                pgsurf = img.get_pg_surface()
            else:  # the height has more distortion
                sfactor = float(self.resolution[1])/float(img.height)
                targetw = int(float(img.width)*sfactor)
                targeth = int(float(img.height)*sfactor)
                if targetw > self.resolution[0] \
                        or targeth > self.resolution[1]:
                    # aw shucks that still didn't work do the other way instead
                    sfactor = float(self.resolution[0])/float(img.width)
                    targetw = int(float(img.width)*sfactor)
                    targeth = int(float(img.height)*sfactor)
                    targetx = 0
                    targety = (self.resolution[1]-targeth)/2
                else:
                    targetx = (self.resolution[0]-targetw)/2
                    targety = 0
                img = img.scale(targetw, targeth)
                pgsurf = img.get_pg_surface()
            # clear out the screen so everything is clean
            black = pg.Surface((self.resolution[0], self.resolution[1]))
            black.fill((0, 0, 0))
            self.screen.blit(black, black.get_rect())
            self.screen.blit(pgsurf, (targetx, targety))
            self.sourceoffset = (targetx, targety)
            pg.display.flip()
            self.xoffset = targetx
            self.yoffset = targety
            self.xscale = (float(self.imgw)/float(targetw))
            self.yscale = (float(self.imgh)/float(targeth))
        else:  # we're going to crop instead
            # self.do_clamp = False
            targetx = 0
            targety = 0
            cornerx = 0
            cornery = 0
            pgsurf = img.get_pg_surface()
            if img.width <= self.resolution[0] and\
               img.height <= self.resolution[1]:  # center a too small image
                # we're too small just center the thing
                targetx = (self.resolution[0]/2)-(img.width/2)
                targety = (self.resolution[1]/2)-(img.height/2)
                cornerx = targetx
                cornery = targety
                pgsurf = img.get_pg_surface()
            elif img.width > self.resolution[0] \
                    and img.height > self.resolution[1]:
                # crop too big on both axes
                targetw = self.resolution[0]
                targeth = self.resolution[1]
                targetx = 0
                targety = 0
                x = (img.width - self.resolution[0])/2
                y = (img.height - self.resolution[1])/2
                cornerx = -1*x
                cornery = -1*y
                img = img.crop(x, y, targetw, targeth)
                pgsurf = img.get_pg_surface()
            elif img.width < self.resolution[0] \
                    and img.height >= self.resolution[1]:  # height too big
                # crop along the y dimension and center along the x dimension
                targetw = img.width
                targeth = self.resolution[1]
                targetx = (self.resolution[0]-img.width)/2
                targety = 0
                x = 0
                y = (img.height - self.resolution[1])/2
                cornerx = targetx
                cornery = -1 * y
                img = img.crop(x, y, targetw, targeth)
                pgsurf = img.get_pg_surface()
            elif img.width > self.resolution[0] \
                    and img.height <= self.resolution[1]:  # width too big
                # crop along the y dimension and center along the x dimension
                targetw = self.resolution[0]
                targeth = img.height
                targetx = 0
                targety = (self.resolution[1]-img.height)/2
                x = (img.width - self.resolution[0])/2
                y = 0
                cornerx = -1 * x
                cornery = targety
                img = img.crop(x, y, targetw, targeth)
                pgsurf = img.get_pg_surface()
            self.xoffset = cornerx
            self.yoffset = cornery
            black = pg.Surface((self.resolution[0], self.resolution[1]))
            black.fill((0, 0, 0))
            self.screen.blit(black, black.get_rect())
            self.screen.blit(pgsurf, (targetx, targety))
            pg.display.flip()

    def _set_button_state(self, state, button):
        if button == 1:
            self.mouse_left = state
        if button == 2:
            self.mouse_middle = state
        if button == 3:
            self.mouse_right = state
        if button == 4:
            self.mouse_wheel_up = 1
        if button == 5:
            self.mouse_wheel_down = 1

    def check_events(self, return_strings=False):
        """

        **SUMMARY**

        CheckEvents checks the pygame event queue and sets the internal display
        values based on any new generated events.

        .. warning::
          This method must be called (or :py:meth:`is_done`
                                      or :py:meth:`is_not_done`)
            to perform mouse event checking.

        **PARAMETERS**

        return_strings - pygame returns an enumerated int by default,
                         when this is set to true we return a list of strings.

        **RETURNS**

        A list of key down events. Parse them with pg.K_<lowercase_letter>

        """
        self.mouse_wheel_up = self.mouse_wheel_down = 0
        self.last_left_button = self.mouse_left
        self.last_right_button = self.mouse_right
        self.left_button_down = None
        self.left_button_up = None
        self.right_button_down = None
        self.right_button_up = None
        key = []
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                self.done = True
            if event.type == pg.MOUSEMOTION:
                self.mouse_raw_x = event.pos[0]
                self.mouse_raw_y = event.pos[1]
                x = int((event.pos[0] - self.xoffset)*self.xscale)
                y = int((event.pos[1] - self.yoffset)*self.yscale)
                (self.mouse_x, self.mouse_y) = self._clamp(x, y)
                self.mouse_left, self.mouse_middle, self.mouse_right = \
                    event.buttons
            if event.type == pg.MOUSEBUTTONUP:

                self._set_button_state(0, event.button)

            if event.type == pg.MOUSEBUTTONDOWN:
                self._set_button_state(1, event.button)
            if event.type == pg.KEYDOWN:
                if return_strings:
                    key.append(pg.key.name(event.key))
                else:
                    key.append(event.key)

        self.pressed = pg.key.get_pressed()

        if self.last_left_button == 0 and self.mouse_left == 1:
            self.left_button_down = (self.mouse_x, self.mouse_y)
        if self.last_left_button == 1 and self.mouse_left == 0:
            self.left_button_up = (self.mouse_x, self.mouse_y)

        if self.last_right_button == 0 and self.mouse_right == 1:
            self.right_button_down = (self.mouse_x, self.mouse_y)
        if self.last_right_button == 1 and self.mouse_right == 0:
            self.right_button_up = (self.mouse_x, self.mouse_y)

        #If ESC pressed, end the display
        if self.pressed[pg.K_ESCAPE] == 1:
            self.done = True

        return key

    def is_done(self):
        """
        **SUMMARY**

        Checks the event queue and returns True if a quit event has been
        issued.

        **RETURNS**

        True on a quit event, False otherwise.

        **EXAMPLE**

        >>> disp = Display()
        >>> cam = Camera()
        >>> while not disp.is_done():
        >>>   img = cam.get_image()
        >>>   img.save(disp)

        """
        self.check_events()
        return self.done

    def is_not_done(self):
        """
        **SUMMARY**

        Checks the event queue and returns False as long as the quit
        event hasn't been issued.

        **RETURNS**

        False on a quit event, True otherwise.

        **EXAMPLE**

        >>> disp = Display()
        >>> cam = Camera()
        >>> while disp.is_not_done():
        >>>   img = cam.get_image()
        >>>   img.save(disp)

        """
        return not self.is_done()

    def _clamp(self, x, y):
        """
        clamp all values between zero and the image width
        """
        r_x = x
        r_y = y
        if x > self.imgw:
            r_x = self.imgw
        if x < 0:
            r_x = 0

        if y > self.imgh:
            r_y = self.imgh
        if y < 0:
            r_y = 0
        return r_x, r_y

    @classmethod
    def quit(cls):
        """
        quit the pygame instance

        Example:
        >>> img = Image("simplecv")
        >>> d = img.show()
        >>> time.sleep(5)
        >>> d.quit()
        """
        pg.display.quit()
        pg.quit()
