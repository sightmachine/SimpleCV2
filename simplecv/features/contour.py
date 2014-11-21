import math
import operator

import numpy as np
import scipy
import scipy.spatial.distance as spsd
# import pandas as pd

from simplecv.base import lazyproperty, PicklabeNdarray
from simplecv.factory import Factory
from simplecv.color import Color


class Contour(PicklabeNdarray):

    __survive_pickling__ = ['blob', 'holes']

    def __new__(subtype, data, dtype=None, blob=None, holes=None):
        subarr = np.array(data, dtype=dtype).view(subtype)
        subarr.blob = blob
        subarr.holes = holes if holes is not None else []
        return subarr

    def __array_finalize__(self, obj):
        if obj is not None:
            self.blob = getattr(obj, 'blob', None)
            self.holes = getattr(obj, 'holes', [])
        else:
            self.blob = None
            self.holes = []

    @lazyproperty
    def min_x(self):
        return float(np.min(self[:, 0]))

    @lazyproperty
    def min_y(self):
        return float(np.min(self[:, 1]))

    @lazyproperty
    def max_x(self):
        return float(np.max(self[:, 0]))

    @lazyproperty
    def max_y(self):
        return float(np.max(self[:, 1]))

    @property
    def width(self):
        return self.max_x - self.min_x

    @property
    def height(self):
        return self.max_y - self.min_y

    #geez, how many times has this been written
    def rotate(self, angle, point=(0, 0)):
        angle = math.radians(angle)
        rot_mat = np.array([[np.cos(angle), -np.sin(angle)],
                            [np.sin(angle), np.cos(angle)]])
        point = np.array(point)
        new_contour = np.dot((self - point), rot_mat) + point

        # rotate holes
        new_contour.holes = []
        for hole in self.holes:
            new_contour.holes.append(np.dot((hole - point), rot_mat) + point)

        return new_contour

    def draw(self, color=Color.GREEN, alpha=255, width=1, layer=None):
        """
        Draw the contour to either the source image or to the
        specified layer given by layer.

        * *color* - The color to render the blob's convex hull as an RGB
         triplet.
        * *alpha* - The alpha value of the rendered blob.
        * *width* - The width of the drawn blob in pixels, if w=-1 then the
         polygon is filled.
        * *layer* - if layer is not None, the blob is rendered to the layer
         versus the source image.
        """
        self._draw(self, self.holes, color=color,
                   alpha=alpha, width=width, layer=layer)

    @staticmethod
    def _draw(contour, holes=(), color=Color.GREEN, alpha=255, width=1, layer=None):
        if contour.blob is None:
            return

        if layer is None:
            layer = contour.blob.image.dl()

        layer.polygon(points=contour, color=color, holes=holes,
                      width=width, antialias=True, alpha=alpha)

    def to_image(self, color=Color.WHITE, filled=False, full=False, masked=False):
        width = -1 if filled else 1

        if self.blob is not None and self.blob.image is not None:
            img = self.blob.image
        else:
            img = None

        if full:
            if img is None:
                img = Factory.Image((self.max_x, self.max_y))
            if masked:
                mask = self.to_mask(filled=filled, full=True).to_gray() != 0
                masked_img = img.get_empty()
                masked_img[mask] = img[mask]
                return masked_img
            else:
                self.draw(color=color, width=width, layer=img.dl())
                return img.apply_layers()
        else:
            if img is None:
                img = Factory.Image((self.width, self.height))
            else:
                roi = (self.min_x, self.min_y, self.width, self.height)
                img = img.crop(*roi)
            if masked:
                mask = self.to_mask(filled=filled, full=False).to_gray() != 0
                masked_img = img.get_empty()

                masked_img[mask] = img[mask]
                return masked_img
            else:
                min_coord = np.array([self.min_x, self.min_y])
                contour = self - min_coord
                holes = [hole - min_coord for hole in self.holes]
                self._draw(contour, holes, color=color, width=width, layer=img.dl())
                return img.apply_layers()

    def to_mask(self, filled=True, full=False):
        width = -1 if filled else 1

        if full:
            if self.blob is not None and self.blob.image is not None:
                img = self.blob.image.get_empty()
            else:
                img = Factory.Image((self.max_x, self.max_y))
            self.draw(color=Color.WHITE, width=width, layer=img.dl())
        else:
            img = Factory.Image((self.width, self.height))
            min_coord = np.array([self.min_x, self.min_y])
            contour = self - min_coord
            holes = [hole - min_coord for hole in self.holes]
            self._draw(contour, holes,
                       color=Color.WHITE, width=width, layer=img.dl())

        return img.apply_layers()

    def to_array(self):
        return [(float(data[0]), float(data[1])) for data in self]

    def to_tuples(self):
        dtype = str(self.dtype)
        return np.array(self.view(dtype + "," + dtype).reshape(-1))

    def sort_from_point_on_axis(self, pointaxis, axis=0):
        return self[np.argsort(np.abs(self[:, axis] - pointaxis))]

    def nearest_to_point(self, point):
        return self[np.argmin(spsd.cdist(self, [point]))]

    def nearest_to_points(self, points):
        return self[np.argmin(spsd.cdist(self, points)[:, 0])]

    def extract_between_on_axis(self, pt1, pt2, axis=0):
        segment = np.array([pt1, pt2])
        return self[(self[:, axis] >= np.min(segment[:, axis])) & (self[:, axis] <= np.max(segment[:, axis]))]

    #adapted from
    #http://stackoverflow.com/questions/14834693/approximating-a-polygon-with-a-circle/14835559#14835559
    def fit_radius(self):
        from scipy.optimize import minimize

        x_m = np.average(self[:, 0])
        y_m = np.average(self[:, 1])
        r_m = spsd.cdist([[x_m, y_m]], self).mean()
        r_m /= 2

        # Best fit a circle to these points
        def err((w, v, r)):
            pts = [np.linalg.norm([x - w, y - v]) - r for x, y in self]
            return (np.array(pts) ** 2).sum()

        xf, yf, rf = scipy.optimize.fmin(err, [x_m, y_m, r_m])
        #res = scipy.optimize.minimize(err,[x_m,y_m,r_m])
        #if not res.success:
        #    print res.message
        #xf, yf, rf = res.x
        return (xf, yf), rf

    def slope_intercept(self, axis=0):
        x, y = self[:, axis], self[:, abs(axis - 1)]
        #note that we invert x,y if axis = 1

        return np.polyfit(x, y, 1)
        #fit a polynomial

    #if axis = 0, predict Y from X, otherwise predict X from Y
    def fit_line(self, axis=0):
        x, y = self[:, axis], self[:, abs(axis - 1)]
        #note that we invert x,y if axis = 1

        mx, c = np.polyfit(x, y, 1)
        #fit a polynomial

        startpoint = [x[0], mx * x[0] + c]
        endpoint = [x[-1], mx * x[-1] + c]
        #take the start and end values of our input

        if axis:
            startpoint = list(reversed(startpoint))
            endpoint = list(reversed(endpoint))
            #reverse back if we used axis = 1
        return startpoint, endpoint

    #for loops make me cringe -- has to be a better way
    def segmentize(self, points, minlength=2, isclosed=False):
        firstindex = self.index_of(points[0])
        lastindex = firstindex
        segments = []

        if not isclosed:
            for p in points[1:]:
                index = self.index_of(p)
                indexes = sorted([lastindex, index])
                seg = self[indexes[0]:indexes[1]]
                if len(seg) > minlength:
                    segments.append(seg)
                lastindex = index
            return segments

        pindexes = sorted([self.index_of(p) for p in points])
        start = np.argmin(pindexes)
        if start:
            pindexes = np.concatenate([pindexes[start:], pindexes[:start]])  #shift so we start at lowest number

        firstindex = pindexes[0]
        lastindex = firstindex
        for index in pindexes[1:]:
            indexes = sorted([lastindex, index])
            seg = self[indexes[0]:indexes[1]]
            if len(seg) > minlength:
                segments.append(seg)
            lastindex = index

        lastseg = Contour(np.concatenate([self[lastindex:], self[:firstindex + 1]]))
        if len(lastseg) > minlength:
            segments.append(lastseg)

        return segments

    def sort_on_axis(self, axis=0):
        return self[np.argsort(self[:, axis])]

    def find_flat_point(self, smooth_window=7, axis=0, start_index=0, offset=0, flat_threshold=0.01, win_type="boxcar"):
        smoothed_contour_derivative = pd.rolling_window(self[1:, axis] - self[:-1, axis], smooth_window,
                                                        win_type=win_type, center=True)

        flat_spot_indexes = np.where(np.abs(smoothed_contour_derivative) < flat_threshold)[
                                0] + 1  #we lost one in the derivative
        flat_spot_indexes = flat_spot_indexes[flat_spot_indexes > start_index]

        if len(flat_spot_indexes):
            return self[flat_spot_indexes[0] - offset]
        else:  #return end of contour
            return None

    def find_inflection_point(self, window=5, win_type="triang", center=True, reverse=False):
        if reverse is True:
            contour = self[::-1]
        else:
            contour = self

        dy = contour[-1][1] - contour[0][1]
        dx = contour[-1][0] - contour[0][0]
        angle = np.arctan2(dy, dx) * 180 / np.pi
        c2 = contour.rotate(angle)
        window = 5
        index = np.argmin(pd.rolling_window(c2[:, 1], window, win_type=win_type, center=center)[window / 2:-window / 2])
        return contour[index + window / 2]

    def render_points(self, points, color=(0, 255, 0)):
        points = np.array(points)
        x = points[:, 0] - self.min_x
        y = points[:, 1] - self.min_y

        img = self.to_image()
        for p in zip(x, y):
            img.dl().circle(p, radius=3, color=color)

        return img.apply_layers()

    def find_point_along_axis(self, axis_distance, axis=0, point=None, smoothing_window=7):

        if point == None:
            point = self[0]

        if axis_distance > 0:
            point_distances = self[
                pd.rolling_window(self[:, axis], smoothing_window, win_type="boxcar", center=True) > point[
                    axis] + axis_distance]
        else:
            point_distances = self[
                pd.rolling_window(self[:, axis], smoothing_window, win_type="boxcar", center=True) < point[
                    axis] + axis_distance]

        if len(point_distances):
            return point_distances[0]
        else:
            return None

    def first_point_at_axis_position(self, axis_value, thresh=0.1, axis=0, smooth_window=7):
        axis_distances = np.abs(
            pd.rolling_window(self[:, axis], smooth_window, win_type="boxcar", center=True) - axis_value)
        #compute the distance of the smoothed curve from our desired point

        close_points = self[axis_distances < thresh]
        #find the list of points that fall within our threshold

        if len(close_points):
            return close_points[0]  #return the first one
        else:
            return None

    #there has got to be a more efficient way to do this
    #we introduce a tolerance in case we're dealing with points as floats
    #maybe using pdist/cdist
    def index_of(self, point, tolerance=0.01):
        distances = self.distances(point)

        within_tol = distances[distances < tolerance]
        if len(within_tol):
            return np.argmin(distances)
        else:
            return None

    def width_at_point(self, pointaxis, axis=0, direction=0):
        if direction == 0:
            section = self[self[:, axis] < pointaxis]
        if direction == 1:
            section = self[self[:, axis] > pointaxis]

        if axis == 0:
            return section.height
        else:
            return section.width

    def angle_to_axis(self, axis=1, anglerange=(-10, 10), binfactor=2, startstep=1, minstep=0.04, offset=(0, 0)):
        anglerange = list(anglerange)
        bins = float(np.max(self[:, axis]) - np.min(self[:, axis])) / binfactor
        rotmap = {}
        maxangle = 0

        #there is probably a much more efficient way to do this
        #rotate part and histogram along axis, find peak value
        #search around that space
        while startstep > minstep:

            for angle in np.arange(anglerange[0], anglerange[1], startstep):
                if not rotmap.get(angle, False):
                    rotmap[angle] = np.max(np.histogram(self.rotate(angle, point=offset)[:, axis], bins=bins)[0])

            maxangle = max(rotmap.iteritems(), key=operator.itemgetter(1))[0]

            newstep = float(startstep) / (.5 * abs(anglerange[0] - anglerange[1]) / float(startstep))
            anglerange[0] = float(maxangle - startstep)
            anglerange[1] = float(maxangle + startstep)
            startstep = newstep

        if abs(maxangle) < minstep:
            maxangle = 0
        return maxangle

    #for stability, we may lock a point on an axis, and take the mean value of surrounding points
    def mean_value_point_on_axis(self, point, axis, window=5):
        nearby = self.sort_from_point_on_axis(point[axis], axis=axis)[:window]

        if axis == 1:
            return [np.mean(nearby[:, 0]), point[1]]
        else:
            return [point[0], np.mean(nearby[:, 1])]

    def find_slopes(self):
        deltas = self[:-1] - self[1:]
        slopes = deltas[:, 1].astype(float) / deltas[:, 0].astype(float)
        return slopes

    # uses pandas as PD
    def find_line_departure(self, axis=0, lineend=None, window=0, threshold=2):
        if not lineend:
            lineend = len(self) / 2

        window = int(window)

        points = Contour(self.tolist())

        slopex, interceptx = self[:lineend].slope_intercept(0)
        slopey, intercepty = self[:lineend].slope_intercept(1)

        slope = slopex
        if axis == 1 or np.isnan(slope) or np.abs(slopey) < np.abs(slopex):
            slope = 1 / slopey

        angle_direction = 0

        if points[0][axis] > points[lineend][axis]:
            angle_direction += 180

        rotateangle = angle_direction + np.arctan(slope) * 180 / np.pi
        rotated_points = points.rotate(rotateangle, point=points[0])
        rotated_points[:, 1] -= points[0][1]

        if window:
            smoothed_values = pd.rolling_window(rotated_points[:, 1], window, win_type="triang", center=True)
            smoothed = Contour(zip(rotated_points[:, 0], smoothed_values))
        else:
            smoothed = rotated_points

        #go backwards, and add half a window length
        distant = np.where(np.abs(smoothed[lineend:, 1][::-1]) < threshold)[0]
        if len(distant):
            return self[-distant[0] - window / 2]
        return self[-1]

    #make sure we have a value for every integer along an axis
    def interpolate_on_axis(self, axis=0):
        if axis == 0:
            sort_index = 1
        else:
            sort_index = 0

        interp_points = []
        for index, p in enumerate(self):
            interp_points.append(p)

            #if we would skip a value, interpolate between

            if index < len(self) - 1 and abs(p[sort_index] - self[index + 1][sort_index]) > 1:
                dp = self[index + 1]
                positions = sorted((int(p[sort_index]), int(dp[sort_index])))

                steps = range(*positions)

                vals = np.interp(steps, sorted([p[sort_index], dp[sort_index]]), [p[axis], dp[axis]])
                interps = zip(steps, vals)

                #check sign to see if we're going "backwards":
                if p[sort_index] > dp[sort_index]:
                    interps = interps[::-1]  #reverse order on self

                for i, s in enumerate(steps[1:]):
                    interp = [0.0, 0.0]
                    interp[sort_index] = s
                    interp[axis] = vals[i + 1]
                    interp_points.append(interp)

        return Contour(interp_points)

    # uses pandas as PD
    def debump_on_axis(self, axis=0, slopethresh=1, smooth_window=3, win_type="triang"):
        smoothed = pd.rolling_window(self[:, axis], smooth_window, win_type=win_type, center=True)
        slopes = smoothed[1:] - smoothed[:-1]
        secondder = slopes[1:] - slopes[:-1]  #second derivative of the smoothed line

        bump_indexes = np.where(np.abs(secondder) > slopethresh)[0] + 2
        withoutbump = Contour(np.delete(self, bump_indexes, axis=0))

        return withoutbump.interpolate_on_axis(axis)

    def denoise_on_axis(self, axis=0, direction=0, debump=True):
        if axis == 0:
            sort_index = 1
        else:
            sort_index = 0
        #unfortunately, sometimes we skip steps -- we need to make sure we have a point on each value of our axis of interest
        #seems like there should be a much more efficient way to do this

        interp_points = self.interpolate_on_axis(axis)
        dtype = np.result_type(interp_points)

        if axis == 1:
            sorted_points = np.sort(interp_points.view("{0},{0}".format(dtype.name)), order=['f0', 'f1'], axis=0).view(
                dtype)
        else:
            sorted_points = np.sort(interp_points.view("{0},{0}".format(dtype.name)), order=['f1', 'f0'], axis=0).view(
                dtype)

        if direction:
            sorted_points = sorted_points[::-1]

        denoised_points_index = np.unique(sorted_points[:, sort_index], return_index=1)[1]
        denoised_points = Contour(sorted_points[denoised_points_index], 'float32')

        if debump:
            return denoised_points.debump_on_axis(axis)

        return denoised_points

    def point_distances(self, closed=True):
        if closed:
            return np.array(np.sqrt(((self - np.roll(self, 1, axis=0)) ** 2).sum(axis=-1)))
        else:
            return np.array(np.sqrt(((self[:-1] - self[1:]) ** 2).sum(axis=-1)))

    def point_distances_on_axis(self, axis=0, closed=True):
        if closed:
            return np.abs(np.array(self[:, axis] - np.roll(self[:, axis])))
        else:
            return np.abs(np.array(self[:-1, axis] - self[1:, axis]))

    def distances(self, point):
        return spsd.cdist(self, [point])[:, 0]

    #TODO return angles 0 deg = 12 o clock
    def angles(self, point):
        return np.array(np.arctan2(point[0] - self[:, 0], point[1] - self[:, 1])) * 180.0 / np.pi

    def areas(self, point):
        point_distances = self.distances(point)
        border_distances = self.point_distances(point)
        thetas = self.angles(point)
        pd_roll = np.roll(point_distances, 1, axis=0)
        heron_s = (point_distances + pd_roll + border_distances) / 2.0
        return np.sqrt(heron_s * (heron_s - point_distances) * (heron_s - pd_roll) * (heron_s - border_distances))

    def find_symmetry_axis(self, bins=360, point=None):
        centroid = point
        if not centroid:
            centroid = np.mean(self[:, 0]), np.mean(self[:, 1])

        thetas = self.angles(centroid)
        distances = self.distances(centroid)
        step = 360.0 / bins
        theta_bins = np.arange(-180.0, 180.0, step)
        thetabin_indexes = np.digitize(thetas, theta_bins)

        distance_means = []
        for bin in range(bins):
            thetadist = distances[thetabin_indexes == bin + 1]
            if len(thetadist):
                distance_means.append(np.mean(thetadist))
            else:
                distance_means.append(np.nan)

        #cribbed from http://stackoverflow.com/questions/6518811/interpolate-nan-values-in-a-numpy-array
        def nan_helper(y):
            return np.isnan(y), lambda z: z.nonzero()[0]

        distance_means = np.array(distance_means)
        nans, x = nan_helper(distance_means)
        distance_means[nans] = np.interp(x(nans), x(~nans), distance_means[~nans])

        testlen = bins / 2
        distance_correlations = []
        for symtest in range(bins):
            distance_correlations.append(np.corrcoef(np.roll(distance_means, symtest)[:testlen],
                                                     np.roll(distance_means, symtest)[::-1][:testlen])[0][1])

        return theta_bins[np.nanargmax(distance_correlations)]