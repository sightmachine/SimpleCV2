import cv2


class Window(object):

    def __init__(self, name='simplecv', flags=cv2.WINDOW_AUTOSIZE):
        super(Window, self).__init__()
        self.trackbars = {}
        self.name = name
        cv2.namedWindow(self.name, flags)

    def show(self, image):
        cv2.imshow(self.name, image)

    def close(self):
        cv2.destroyWindow(self.name)

    def wait_key(self, delay=0):
        return cv2.waitKey(delay)

    def on_mouse(self, event, x, y, mouse_key, data=None):
        """ Callback for mouse events

            event - int - see cv2.EVENT_* constants
            x, y - int, int - position of the cursor
            mouse_key - int - mouse key
        """
        pass

    def on_key(self, key):
        """ Callback for keyboard

            key - int - key code
        """
        pass

    def on_trackbar(self, name, value):
        """ Callback for track bars
        """
        pass

    def event_loop(self):
        cv2.setMouseCallback(self.name, self.on_mouse)

        key = None
        while key != 27:  # ESC key
            key = cv2.waitKey()
            self.on_key(key)

    def add_trackbar(self, name, value=0, max=255):
        if name in self.trackbars:
            raise ValueError('Trackbar with the same name already exists')
        self.trackbars[name] = TrackBar(name, self,
                                        value=value, max=max)


class TrackBar(object):

    def __init__(self, name, window, value=0, max=255):
        super(TrackBar, self).__init__()
        self.name = name
        self.window = window
        cv2.createTrackbar(self.name, self.window.name,
                           value, max, self.on_change)

    @property
    def value(self):
        return cv2.getTrackbarPos(self.name, self.window.name)

    @value.setter
    def value(self, value):
        cv2.setTrackbarPos(self.name, self.window.name, value)

    def on_change(self, value):
        self.window.on_trackbar(self.name, value)
