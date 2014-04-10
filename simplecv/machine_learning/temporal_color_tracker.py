import time

import numpy as np

from simplecv.core.camera.camera import Camera
from simplecv.core.camera.virtual_camera import VirtualCamera
from simplecv.color import Color
from simplecv.features.detection import ROI
from simplecv.image_set import ImageSet
from simplecv.linescan import LineScan


class TemporalColorTracker(object):
    """
    **SUMMARY**

    The temporal color tracker attempts to find and periodic color
    signal in an roi or arbitrary function. Once the temporal tracker is
    trained it will return a count object every time the signal is detected.
    This class is usefull for counting periodically occuring events, for
    example, waves on a beach or the second hand on a clock.

    """

    def __init__(self):
        self._rtData = LineScan([])  # the deployed data
        self._steadyState = None  # mu/signal for the ss behavior
        self._extractor = None
        self._roi = None
        self._window = None
        self._template = None
        self._cutoff = None
        self._bestKey = None
        self._isPeak = False
        self.corr_templates = None
        self.peaks = {}
        self.valleys = {}
        self.do_peaks = {}
        self.corr_std_mult = 3.0
        self.count = 0

    def train(self, src, roi=None, extractor=None, do_corr=False,
              maxframes=1000,
              sswndw=0.05, pkwndw=30, pkdelta=3, corr_std_mult=2.0,
              forcechannel=None, verbose=True):
        """
        **SUMMARY**

        To train the TemporalColorTracker you provide it with a video, camera,
        or ImageSet and either an region of interest (ROI) or a function of the
        form:

        (R,G,B) = MyFunction(Image)

        This function takes in an image and returns a tuple of RGB balues for
        the frame. The TemoralColroTracker will then attempt to find the
        maximum peaks in the data and create a model for the peaks.

        **PARAMETERS**

        * *src* - An image source, either a camera, a virtual camera (like a
         video) or an ImageSet.
        * *roi* - An ROI object that tells the tracker where to look in the
         frame.
        * *extractor* - A function with the following signature:
          (R,G,B) = Extract(Image)
        * *do_corr* - Do correlation use correlation to confirm that the signal
         is present.
        * *maxframes* - The maximum number of frames to use for training.
        * *sswndw* - SteadyState window, this is the size of the window to look
         for a steady state, i.e a region where the signal is not changing.
        * *pkwndw* - The window size to look for peaks/valleys in the signal.
         This is roughly the period of the signal.
        * *pkdelta* - The minimum difference between the steady state to look
         for peaks.
        * *corr_std_mult* - The maximum correlation standard deviation of the
         training set to use when looking for a signal. This is the knob to
         dial in when using correlation to confirm the event happened.
        * *forcechannel* - A string that is the channel to use. Options are:
           * 'r' - Red Channel
           * 'g' - Green Channel
           * 'b' - Blue Channel
           * 'h' - Hue Channel
           * 'i' - Intensity Channel
          By default this module will look at the signal with the highest
          peak/valley swings. You can manually override this behavior.
        * *verbose* - Print debug info after training.

        **RETURNS**

        Nothing, will raise an exception if no signal is found.

        **EXAMPLE**

        A really simple example

        >>>> cam = Camera(1)
        >>>> tct = TemporalColorTracker()
        >>>> img = cam.get_image()
        >>>> roi = ROI(img.width * 0.45, img.height * 0.45, img.width * 0.1,
        ....           img.height * 0.1, img)
        >>>> tct.train(cam, roi=roi, maxframes=250)
        >>>> disp = Display((800, 600))
        >>>> while disp.isNotDone():
        >>>>     img = cam.get_image()
        >>>>     result = tct.recognize(img)
        >>>>     roi = ROI(img.width * 0.45, img.height * 0.45,
        ....               img.width * 0.1, img.height * 0.1, img)
        >>>>     roi.draw(width=3)
        >>>>     img.draw_text(str(result), 20, 20, color=Color.RED,
        ....                   fontsize=32)
        >>>>     img = img.apply_layers()
        >>>>     img.save(disp)

        """
        if roi is None and extractor is None:
            raise Exception('Need to provide an ROI or an extractor')
        self.do_corr = do_corr
        self.corr_std_mult = corr_std_mult
        self._extractor = extractor  # function that returns a RGB values
        self._roi = roi
        self._extract(src, maxframes, verbose)
        self._find_steady_state(window_sz_prct=sswndw)
        self._find_peaks(pkwndw, pkdelta)
        self._extract_signal_info(forcechannel)
        self._build_signal_profile()
        if verbose:
            for key in self.data.keys():
                print 30 * '-'
                print "Channel: {0}".format(key)
                print "Data Points: {0}".format(len(self.data[key]))
                print "Steady State: {0}+/-{1}".format(
                    self._steadyState[key][0], self._steadyState[key][1])
                print "Peaks: {0}".format(self.peaks[key])
                print "Valleys: {0}".format(self.valleys[key])
                print "Use Peaks: {0}".format(self.do_peaks[key])
            print 30 * '-'
            print "BEST SIGNAL: {0}".format(self._bestKey)
            print "BEST WINDOW: {0}".format(self._window)
            print "BEST CUTOFF: {0}".format(self._cutoff)

    def _get_data_from_img(self, img):
        """
        Get the data from the image
        """
        mc = None
        if self._extractor:
            mc = self._extractor(img)
        else:
            temp = self._roi.reassign(img)
            mc = temp.mean_color()
        self.data['r'].append(mc[0])
        self.data['g'].append(mc[1])
        self.data['b'].append(mc[2])
        # NEED TO CHECK THAT THIS REALLY RGB
        self.data['i'].append(Color.get_lightness(mc))
        self.data['h'].append(Color.get_hue_from_rgb(mc))
        #return [mc[0],mc[1],mc[2],gray,Color.rgbToHue(mc)]

    def _extract(self, src, maxframes, verbose):
        # get the full dataset and append it to the data vector dictionary.
        self.data = {'r': [], 'g': [], 'b': [], 'i': [], 'h': []}
        if isinstance(src, ImageSet):
            src = VirtualCamera(src, st='imageset')  # this could cause a bug
        elif isinstance(src, (VirtualCamera, Camera)):
            count = 0
            for i in range(0, maxframes):
                img = src.get_image()
                count = count + 1
                if verbose:
                    print "Got Frame {0}".format(count)
                if isinstance(src, Camera):
                    time.sleep(0.05)  # let the camera sleep
                if img is None:
                    break
                else:
                    self._get_data_from_img(img)

        else:
            raise Exception('Not a valid training source')

    def _find_steady_state(self, window_sz_prct=0.05):
        # slide a window across each of the signals
        # find where the std dev of the window is minimal
        # this is the steady state (e.g. where the
        # assembly line has nothing moving)
        # save the mean and sd of this value
        # as a tuple in the steadyStateDict
        self._steadyState = {}
        for key in self.data.keys():
            wndw_sz = int(np.floor(window_sz_prct * len(self.data[key])))
            signal = self.data[key]
            # slide the window and get the std
            data = [np.std(signal[i:i + wndw_sz]) for i in
                    range(0, len(signal) - wndw_sz)]
            # find the first spot where sd is minimal
            index = np.where(data == np.min(data))[0][0]
            # find the mean for the window
            mean = np.mean(signal[index:index + wndw_sz])
            self._steadyState[key] = (mean, data[index])

    def _find_peaks(self, pk_wndw, pk_delta):
        """
        Find the peaks and valleys in the data
        """
        self.peaks = {}
        self.valleys = {}
        for key in self.data.keys():
            ls = LineScan(self.data[key])
            # need to automagically adjust the window
            # to make sure we get a minimum number of
            # of peaks, maybe let the user guess a min?
            self.peaks[key] = ls.find_peaks(pk_wndw, pk_delta)
            self.valleys[key] = ls.find_valleys(pk_wndw, pk_delta)

    def _extract_signal_info(self, forcechannel):
        """
        Find the difference between the peaks and valleys
        """
        self.pd = {}
        self.vd = {}
        self.do_peaks = {}
        best_spread = 0.00
        best_do_peaks = None
        best_key = None
        for key in self.data.keys():
            #Look at which signal has a bigger distance from
            #the steady state behavior
            if len(self.peaks[key]) > 0:
                peak_mean = np.mean(np.array(self.peaks[key])[:, 1])
                self.pd[key] = np.abs(self._steadyState[key][0] - peak_mean)
            else:
                self.pd[key] = 0.00

            if len(self.valleys[key]) > 0:
                valley_mean = np.mean(np.array(self.valleys[key])[:, 1])
                self.vd[key] = np.abs(self._steadyState[key][0] - valley_mean)
            else:
                self.vd[key] = 0.00

            self.do_peaks[key] = False
            best = self.vd[key]
            if self.pd[key] > self.vd[key]:
                best = self.pd[key]
                self.do_peaks[key] = True
            if best > best_spread:
                best_spread = best
                best_do_peaks = self.do_peaks[key]
                best_key = key
        # Now we know which signal has the most spread
        # and what direction we are looking for.
        if forcechannel is not None:
            if forcechannel in self.data:
                self._bestKey = forcechannel
            else:
                raise Exception('That is not a valid data channel')
        else:
            self._bestKey = best_key

    def _build_signal_profile(self):
        key = self._bestKey
        self._window = None
        peaks = None
        if self.do_peaks[key]:
            self._isPeak = True
            peaks = self.peaks[key]
            # We're just going to do halfway
            self._cutoff = self._steadyState[key][0] + (self.pd[key] / 2.0)
        else:
            self._isPeak = False
            peaks = self.valleys[key]
            self._cutoff = self._steadyState[key][0] - (self.vd[key] / 2.0)
        if len(peaks) > 1:
            p2p = np.array(peaks[1:]) - np.array(peaks[:-1])
            p2p_mean = int(np.mean(p2p))
            p2ps = int(np.std(p2p))
            p2p_mean = p2p_mean + 2 * p2ps
            # constrain it to be an od window
            if int(p2p_mean) % 2 == 1:
                p2p_mean = p2p_mean + 1
            self._window = p2p_mean
        else:
            raise Exception("Can't find enough peaks")
        if self.do_corr and self._window is not None:
            self._do_corr()

            #NEED TO ERROR OUT ON NOT ENOUGH POINTS

    def _do_corr(self):
        key = self._bestKey
        # build an average signal for the peaks and valleys
        # centered at the peak. The go and find the correlation
        # value of each peak/valley with the average signal
        self.corr_templates = []
        halfwndw = self._window / 2

        if self._isPeak:
            plist = self.peaks[key]
        else:
            plist = self.valleys[key]

        for peak in plist:
            center = peak[0]
            lb = center - halfwndw
            ub = center + halfwndw
            # ignore signals that fall of the end of the data
            if lb > 0 and ub < len(self.data[key]):
                self.corr_templates.append(np.array(self.data[key][lb:ub]))
        if len(self.corr_templates) < 1:
            raise Exception(
                'Could not find a coherrent signal for correlation.')

        sig = np.copy(self.corr_templates[0])  # little np gotcha
        for peak in self.corr_templates[1:]:
            sig += peak
        self._template = sig / len(self.corr_templates)
        self._template /= np.max(self._template)
        corrvals = [np.correlate(peak / np.max(peak), self._template) for peak
                    in self.corr_templates]
        print corrvals
        self.corr_thresh = (np.mean(corrvals), np.std(corrvals))

    def _get_best_value(self, img):
        """
        Extract the data from the live signal
        """
        if self._extractor:
            mc = self._extractor(img)
        else:
            temp = self._roi.reassign(img)
            mc = temp.mean_color()
        if self._bestKey == 'r':
            return mc[0]
        elif self._bestKey == 'g':
            return mc[1]
        elif self._bestKey == 'b':
            return mc[2]
        elif self._bestKey == 'i':
            return Color.get_lightness(mc)
        elif self._bestKey == 'h':
            return Color.get_hue_from_rgb(mc)

    def _update_buffer(self, v):
        """
        Keep a buffer of the running data and process it to determine if there
        is a peak.
        """
        self._rtData.append(v)
        wndwcenter = int(np.floor(self._window / 2.0))
        # pop the end of the buffer
        if len(self._rtData) > self._window:
            self._rtData = self._rtData[1:]
            if self._isPeak:
                lm = self._rtData.find_peaks()
                for l in lm:
                    if l[0] == wndwcenter and l[1] > self._cutoff:
                        if self.do_corr:
                            corrval = np.correlate(self._rtData.normalize(),
                                                   self._template)
                            thresh = self.corr_thresh[0] - \
                                self.corr_std_mult * self.corr_thresh[1]
                            if corrval[0] > thresh:
                                self.count += 1
                        else:
                            self.count += 1
            else:
                lm = self._rtData.find_valleys()
                for l in lm:
                    if l[0] == wndwcenter and l[1] < self._cutoff:
                        if self.do_corr:
                            corrval = np.correlate(self._rtData.normalize(),
                                                   self._template)
                            thresh = self.corr_thresh[0] - \
                                self.corr_std_mult * self.corr_thresh[1]
                            if corrval[0] > thresh:
                                self.count += 1
                        else:
                            self.count += 1
        return self.count

    def recognize(self, img):
        """

        **SUMMARY***

        This method is used to do the real time signal analysis. Pass the
        method an image from the stream and it will return the event count.
        Note that due to buffering the signal may lag the actual video by up
        to a few seconds.

        **PARAMETERS**

        * *img* - The image in the stream to test.

        **RETURNS**

        Returns an int that is the count of the number of times the event has
        occurred.

        **EXAMPLE**

        >>>> cam = Camera(1)
        >>>> tct = TemporalColorTracker()
        >>>> img = cam.get_image()
        >>>> roi = ROI(img.width * 0.45, img.height * 0.45, img.width * 0.1,
                       img.height * 0.1, img)
        >>>> tct.train(cam, roi=roi, maxframes=250)
        >>>> disp = Display((800, 600))
        >>>> while disp.isNotDone():
        >>>>     img = cam.getImage()
        >>>>     result = tct.recognize(img)
        >>>>     roi = ROI(img.width * 0.45, img.height * 0.45,
                           img.width * 0.1, img.height * 0.1, img)
        >>>>     roi.draw(width=3)
        >>>>     img.draw_text(str(result), 20, 20, color=Color.RED,
                              fontsize=32)
        >>>>     img = img.apply_layers()
        >>>>     img.save(disp)

        **TODO**

        Return True/False if the event occurs.
        """
        if self._bestKey is None:
            raise Exception('The TemporalColorTracker has not been trained.')
        v = self._get_best_value(img)
        return self._update_buffer(v)
