import glob
import os
import warnings

import cv2
import numpy as np
import scipy.cluster.vq as cluster
import scipy.spatial.distance as spsd

from simplecv.base import IMAGE_FORMATS
from simplecv.image_class import Image, ColorSpace


class BOFFeatureExtractor(object):
    """
    For a discussion of bag of features please see:
    http://en.wikipedia.org/wiki/Bag_of_words_model_in_computer_vision

    Initialize the bag of features extractor. This assumes you don't have
    the feature codebook pre-computed.
    patchsz = the dimensions of each codebook patch
    numcodes = the number of different patches in the codebook.
    imglayout = the shape of the resulting image in terms of patches
    padding = the pixel padding of each patch in the resulting image.

    """

    def __init__(self, patchsz=(11, 11), numcodes=128, imglayout=(8, 16),
                 padding=0):

        self.padding = padding
        self.layout = imglayout
        self.patch_size = patchsz
        self.num_codes = numcodes
        self.codebook_img = None
        self.codebook = None

    def generate(self, imgdirs, numcodes=128, size=(11, 11), imgs_per_dir=50,
                 img_layout=(8, 16), padding=0, verbose=True):
        """
        This method builds the bag of features codebook from a list of
        directories with images in them. Each directory should be broken down
        by image class.

        * imgdirs: This list of directories.
        * patchsz: the dimensions of each codebook patch
        * numcodes: the number of different patches in the codebook.
        * imglayout: the shape of the resulting image in terms of patches -
         this must match the size of numcodes.
         I.e. numcodes == img_layout[0] * img_layout[1]
        * padding:the pixel padding of each patch in the resulting image.
        * imgs_per_dir: this method can use a specified number of images per
         directory
        * verbose: print output


        Once the method has completed it will save the results to a local file
        using the file name codebook.png


        WARNING:

            THIS METHOD WILL TAKE FOREVER
        """
        if numcodes != img_layout[0] * img_layout[1]:
            warnings.warn("Numcodes must match the size of image layout.")
            return None

        self.padding = padding
        self.layout = img_layout
        self.num_codes = numcodes
        self.patch_size = size
        raw_features = np.zeros(
            size[0] * size[1])  # fakeout numpy so we can use vstack
        for path in imgdirs:
            files = []
            for ext in IMAGE_FORMATS:
                files.extend(glob.glob(os.path.join(path, ext)))
            nimgs = min(len(files), imgs_per_dir)
            for i in range(nimgs):
                infile = files[i]
                if verbose:
                    print path + " " + str(i) + " of " + str(imgs_per_dir)
                    print "Opening file: " + infile
                img = Image(infile)
                new_feat = self._get_patches(img, size)
                if verbose:
                    print "     Got " + str(len(new_feat)) + " features."
                raw_features = np.vstack((raw_features, new_feat))
                del img
        # pop the fake value we put on the top
        raw_features = raw_features[1:, :]
        if verbose:
            print "=================================="
            print "Got " + str(len(raw_features)) + " features "
            print "Doing K-Means .... this will take a long time"
        self.codebook = self._make_codebook(raw_features, self.num_codes)
        self.codebook_img = self._codebook_to_img(self.codebook,
                                                  self.patch_size,
                                                  self.num_codes,
                                                  self.layout,
                                                  self.padding)
        self.codebook_img.save('codebook.png')

    def extract_patches(self, img, size=(11, 11)):
        """
        Get patches from a single images. This is an external access method.
        The user will need to maintain the list of features. See the generate
        method as a guide to doing this by hand. Size is the image patch size.
        """
        return self._get_patches(img, size)

    def make_codebook(self, feature_stack, ncodes=128):
        """
        This method will return the centroids of the k-means analysis of a
        large number of images. Ncodes is the number of centroids to find.
        """
        return self._make_codebook(feature_stack, ncodes)

    @staticmethod
    def _make_codebook(data, ncodes=128):
        """
        Do the k-means ... this is slow as as shit
        """
        [centroids, _] = cluster.kmeans2(data, ncodes, minit='points')
        return centroids

    # FIXME: unused count arg. delete?
    @staticmethod
    def _img_to_codebook(img, patchsize, count, patch_arrangement,
                         spacersz):
        """
        img = the image
        patchsize = the patch size (ususally 11x11)
        count = total codes
        patch_arrangement = how are the patches grided in the image
        (eg 128 = (8x16) 256=(16x16) )
        spacersz = the number of pixels between patches
        """
        img = img.to_hls()
        width = patchsize[0]
        height = patchsize[1]
        lmat = img.get_ndarray()[:, :, 1]
        length = width * height
        ret_value = np.zeros(length)
        for widx in range(patch_arrangement[0]):
            for hidx in range(patch_arrangement[1]):
                x = (widx * patchsize[0]) + ((widx + 1) * spacersz)
                y = (hidx * patchsize[1]) + ((hidx + 1) * spacersz)
                patch = lmat[y:y + height, x:x + width]
                ret_value = np.vstack(
                    (ret_value, np.array(patch[:, :]).reshape(length)))
        ret_value = ret_value[1:, :]
        return ret_value

    # FIXME: unused count arg. delete?
    @staticmethod
    def _codebook_to_img(cbook, patchsize, count, patch_arrangement, spacersz):
        """
        cbook = the codebook
        patchsize = the patch size (ususally 11x11)
        count = total codes
        patch_arrangement = how are the patches grided in the image
        (eg 128 = (8x16) 256=(16x16) )
        spacersz = the number of pixels between patches
        """
        width = (patchsize[0] * patch_arrangement[0]) + (
            (patch_arrangement[0] + 1) * spacersz)
        height = (patchsize[1] * patch_arrangement[1]) + (
            (patch_arrangement[1] + 1) * spacersz)
        bm = np.zeros((height, width), np.uint8)
        img = Image(bm, color_space=ColorSpace.GRAY)
        count = 0
        for widx in range(patch_arrangement[0]):
            for hidx in range(patch_arrangement[1]):
                x = (widx * patchsize[0]) + ((widx + 1) * spacersz)
                y = (hidx * patchsize[1]) + ((hidx + 1) * spacersz)
                temp = Image(cbook[count, :].reshape(patchsize[0],
                                                     patchsize[1]))
                img.blit(temp, pos=(x, y))
                count += 1
        return img

    def _get_patches(self, img, size=None):
        if size is None:
            size = self.patch_size
        img2 = img.to_hls()
        lmat = img2.get_ndarray()[:, :, 1]
        wsteps = img2.width / size[0]
        hsteps = img2.height / size[1]
        width = size[0]
        height = size[1]
        length = width * height
        ret_value = np.zeros(length)
        for widx in range(wsteps):
            for hidx in range(hsteps):
                x = (widx * size[0])
                y = (hidx * size[1])
                patch = cv2.equalizeHist(lmat[y:y + height, x:x + width])
                ret_value = np.vstack(
                    (ret_value, np.array(patch[:, :]).reshape(length)))
        # pop the fake value we put on top of the stack
        ret_value = ret_value[1:, :]
        return ret_value

    def load(self, datafile):
        """
        Load a codebook from file using the datafile. The datafile
        should point to a local image for the source patch image.
        """
        with open(datafile, 'r') as dfile:
            lines = dfile.readlines()
            self.num_codes = int(lines[1])
            self.patch_size = int(lines[2]), int(lines[3])
            self.padding = int(lines[4])
            self.layout = int(lines[5]), int(lines[6])
            data_dir = os.path.dirname(datafile)
            self.codebook_img = Image(os.path.join(data_dir, lines[7].strip()))
            self.codebook = self._img_to_codebook(self.codebook_img,
                                                  self.patch_size,
                                                  self.num_codes,
                                                  self.layout,
                                                  self.padding)

    def save(self, imgfname, datafname):
        """
        Save the bag of features codebook and data set to a local file.
        """
        my_file = open(datafname, 'w')
        my_file.write("BOF Codebook Data\n")
        my_file.write(str(self.num_codes) + "\n")
        my_file.write(str(self.patch_size[0]) + "\n")
        my_file.write(str(self.patch_size[1]) + "\n")
        my_file.write(str(self.padding) + "\n")
        my_file.write(str(self.layout[0]) + "\n")
        my_file.write(str(self.layout[1]) + "\n")
        my_file.write(imgfname + "\n")
        my_file.close()
        if self.codebook_img is None:
            self._codebook_to_img(self.codebook, self.patch_size,
                                  self.num_codes, self.layout, self.padding)
        self.codebook_img.save(imgfname)
        return

    def __getstate__(self):
        if self.codebook_img is None:
            self._codebook_to_img(self.codebook, self.patch_size,
                                  self.num_codes, self.layout, self.padding)
        mydict = self.__dict__.copy()
        del mydict['codebook']
        return mydict

    def __setstate__(self, mydict):
        self.__dict__ = mydict
        self.codebook = self._img_to_codebook(self.codebook_img,
                                              self.patch_size,
                                              self.num_codes,
                                              self.layout,
                                              self.padding)

    def extract(self, img):
        """
        This method extracts a bag of features histogram for the input image
        using the provided codebook. The result are the bin counts for each
        codebook code.
        """
        data = self._get_patches(img)
        codes = np.argmin(spsd.cdist(data, self.codebook), axis=1)
        [ret_value, _] = np.histogram(codes, self.num_codes, normed=True,
                                      range=(0, self.num_codes - 1))
        return ret_value

    def reconstruct(self, img):
        """
        This is a "just for fun" method as a sanity check for the BOF codeook.
        The method takes in an image, extracts each codebook code, and replaces
        the image at the position with the code.
        """
        ret_value = img.get_empty(1)
        data = self._get_patches(img)
        p = spsd.cdist(data, self.codebook)
        codes = np.argmin(p, axis=1)
        count = 0
        wsteps = img.width / self.patch_size[0]
        hsteps = img.height / self.patch_size[1]

        ret_value = Image(ret_value, color_space=ColorSpace.GRAY)
        for widx in range(wsteps):
            for hidx in range(hsteps):
                x = (widx * self.patch_size[0])
                y = (hidx * self.patch_size[1])
                p = codes[count]
                temp = Image(self.codebook[p, :].reshape(self.patch_size[0],
                                                         self.patch_size[1]))
                ret_value = ret_value.blit(temp, pos=(x, y))
                count += 1
        return ret_value

    def get_field_names(self):
        """
        This method gives the names of each field in the feature vector in the
        order in which they are returned. For example, 'xpos' or 'width'
        """
        ret_value = []
        for widx in range(self.layout[0]):
            for hidx in range(self.layout[1]):
                temp = "CB_R" + str(widx) + "_C" + str(hidx)
                ret_value.append(temp)
        return ret_value

    def get_num_fields(self):
        """
        This method returns the total number of fields in the feature vector.
        """
        return self.num_codes
