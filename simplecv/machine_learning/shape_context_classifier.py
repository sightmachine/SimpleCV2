import math
import sys
import warnings

import numpy as np


class ShapeContextClassifier(object):
    """
    Classify an object based on shape context
    """

    def __init__(self, images, labels):
        """
        Create a shape context classifier.

        * *images* - a list of input binary images where the things
          to be detected are white.

        * *labels* - the names of each class of objects.
        """
        # The below import has been done in init since it throws
        # "Need scikits learn installed" for $import simplecv
        try:
            from sklearn import neighbors
        except ImportError:
            raise ImportError("Need scikits learn installed")

        self.img_map = {}
        self.pt_map = {}
        self.desc_map = {}
        self.knn_map = {}
        self.blob_count = {}
        self.labels = labels
        self.images = images

        warnings.simplefilter("ignore")
        for i in range(0, len(images)):
            print "precomputing " + images[i].filename
            self.img_map[labels[i]] = images[i]

            pts, desc, count = self._image_to_feature_vector(images[i])
            self.blob_count[labels[i]] = count
            self.pt_map[labels[i]] = pts
            self.desc_map[labels[i]] = desc
            knn = neighbors.KNeighborsClassifier()
            knn.fit(desc, range(0, len(pts)))
            self.knn_map[labels[i]] = knn

    def _image_to_feature_vector(self, img):
        """
        generate a list of points, SC descriptors, and the count of points
        """
        #IMAGES MUST BE WHITE ON BLACK!
        fulllist = []
        raw_descriptors = []
        blobs = img.find_blobs(minsize=50)
        count = 0
        if blobs is not None:
            count = len(blobs)
            for b in blobs:
                fulllist += b._filter_sc_points()
                raw_descriptors = blobs[0]._generate_sc(fulllist)
        return fulllist, raw_descriptors, count

    def _get_match(self, model_scd, test_scd):
        correspondence, distance = self._do_matching(model_scd, test_scd)
        return self._match_quality(distance)

    def _do_matching(self, model_name, test_scd):
        # my_pts = len(test_scd)
        # ot_pts = len(self.pt_map[model_name])
        # some magic metric that keeps features
        # with a lot of points from dominating
        # this could be moved to after the sum
        # metric = 1.0 + np.log10(np.max([my_pts, ot_pts])
        #                                / np.min([my_pts, ot_pts]))
        other_idx = []
        distance = []

        warnings.simplefilter("ignore")

        for sample in test_scd:
            best = self.knn_map[model_name].predict(sample)
            idx = best[0]  # this is where we can play with k
            scd = self.desc_map[model_name][idx]
            temp = np.sqrt(np.sum(((sample - scd) ** 2)))
            #temp = 0.5*np.sum((sample-scd)**2)/np.sum((sample+scd))
            if math.isnan(temp):
                temp = sys.maxint
            distance.append(temp)
        return [other_idx, distance]

    def _match_quality(self, distances):
        #distances = np.array(distances)
        #sd = np.std(distances)
        #x = np.mean(distances)
        #min = np.min(distances)
        # not sure trimmed mean is perfect
        # realistically we should have some bimodal dist
        # and we want to throw away stuff with awful matches
        # so long as the number of points is not a huge
        # chunk of our points.
        #tmean = sps.tmean(distances,(min,x+sd))
        tmean = np.mean(distances)
        std = np.std(distances)
        return tmean, std

    def _build_match_dict(self, image, count_blobs):
        # we may want to base the count on the num best
        # matchesber of large blobs
        points, descriptors, count = self._image_to_feature_vector(image)
        match_dict = {}
        match_std = {}
        for key, value in self.desc_map.items():
            # only do matching for similar number of blobs
            if count_blobs and self.blob_count[key] == count:
                #need to hold on to correspondences
                correspondence, distances = self._do_matching(key, descriptors)
                result, std = self._match_quality(distances)
                match_dict[key] = result
                match_std[key] = std
            elif not count_blobs:
                correspondence, distances = self._do_matching(key, descriptors)
                result, std = self._match_quality(distances)
                match_dict[key] = result
                match_std[key] = std

        return points, descriptors, count, match_dict, match_std

    def classify(self, image, blob_filter=True):
        """
        Classify an input image.

        * *image* - the input binary image.
        * *blob_filter* - Do a first pass where you only match objects
          that have the same number of blobs - speeds up computation
          and match quality.
        """
        points, descriptors, count, match_dict, match_std = \
            self._build_match_dict(image, blob_filter)
        best = sys.maxint
        best_name = "No Match"
        for k, v in match_dict.items():
            if v < best:
                best = v
                best_name = k

        return best_name, best, match_dict, match_std

    def get_top_nmatches(self, image, n=3, blob_filter=True):
        """
        Classify an input image and return the top n results.

        * *image* - the input binary image.
        * *n* - the number of results to return.
        * *blob_filter* - Do a first pass where you only match objects
          that have the same number of blobs - speeds up computation
          and match quality.
        """
        n = np.clip(n, 1, len(self.labels))
        points, descriptors, count, match_dict, match_std = \
            self._build_match_dict(image, blob_filter)
        best_matches = list(sorted(match_dict, key=match_dict.__getitem__))
        ret_list = []
        for k in best_matches:
            ret_list.append((k, match_dict[k]))
        return ret_list[0:n], match_dict, match_std
