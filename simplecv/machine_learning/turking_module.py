import os
import glob
import pickle

from simplecv.color import Color
from simplecv.image_set import ImageSet


class TurkingModule(object):
    """
    This class is a helper utility for automatically turking
    image data for supervsed learning. This class helps you
    run through a bunch of images and sort them into a bunch
    of classes or categories.

    You provide a path to
    the images that need to be turked/sorted, your class labels, and
    what keys you want to bind to the classes. The turker will
    load all the images, optionally process them, and then display
    them. To sort the images you just push a key mapped to your class
    and the class tosses them into a directory and labels them.
    The class can optionally pickle your data for you to use later.
    """

    def __init__(self, source_paths, out_path, class_list, key_bindings,
                 preprocess=None, postprocess=None):
        """
        **SUMMARY**

        Init sets up the turking module.

        **PARAMETERS**

        * *source_path* - A list of the path(s) with the images to be turked.
        * *out_path* - the output path, a directory will be created for each
         class.
        * *classes* - the names of the classes you are turking as a list of
         strings.
        * *key_bindings* - the keys to bind to each class when turking.
        * *preprocess* - a preprocess function. It should take in an image and
         return a list of images.
        * *postprocess* a post-process step. The signature should be image in
         and image out.


        **EXAMPLE**
        >>> def get_blobs(img):
        >>>     blobs = img.find_blobs()
        >>>     return [b.mMask for b in blobs]

        >>> def scale_inv(img):
        >>>     return img.resize(100,100).invert()

        >>> turker = TurkingModule(['./data/'], ['./turked/'],
        ...                        ['apple', 'banana', 'cherry'],
        ...                        ['a', 'b', 'c'],
        ...                        preprocess=get_blobs, postprocess=scale_inv)
        >>> turker.turk()
        >>> # ~~~ stuff ~~~
        >>> turker.save('./derp.pkl')

        ** TODO **

        TODO: Make it so you just pickle the data and don't have to save each
        file

        """
        #if( not os.access(out_path,os.W_OK) ):
        #    print "Output path is not writeable."
        #    raise Exception("Output path is not writeable.")

        self.key_bindings = key_bindings
        self.classes = class_list
        self.count_map = {}
        self.class_map = {}
        self.directory_map = {}
        self.out_path = out_path
        self.key_map = {}
        if len(class_list) != len(key_bindings):
            raise Exception("Must have a key for each class.")
        for key, cls in zip(key_bindings, class_list):
            self.key_map[key] = cls
        # this should work

        if preprocess is None:
            def fake_pre_process(img):
                return [img]
            preprocess = fake_pre_process
        self.preprocess = preprocess

        if postprocess is None:
            def fake_post_process(img):
                return img
            postprocess = fake_post_process
        self.postprocess = postprocess

        self.src_imgs = ImageSet()

        if isinstance(source_paths, ImageSet):
            self.src_imgs = source_paths
        else:
            for isp in source_paths:
                print "Loading " + isp
                img_set = ImageSet(isp)
                print "Loaded " + str(len(img_set))
                self.src_imgs += img_set

        if not os.path.exists(out_path):
            os.mkdir(out_path)
        for class_item in class_list:
            outdir = out_path + class_item + '/'
            self.directory_map[class_item] = outdir
            if not os.path.exists(outdir):
                os.mkdir(outdir)

        for class_item in class_list:
            searchstr = self.directory_map[class_item] + '*.png'
            self.count_map[class_item] = len(glob.glob(searchstr))
            self.class_map[class_item] = ImageSet(
                self.directory_map[class_item])

    def _save_it(self, img, class_type):
        img.clear_layers()
        path = self.out_path + class_type + "/" + class_type + str(
            self.count_map[class_type]) + ".png"
        print "Saving: " + path
        img = self.postprocess(img)
        self.class_map[class_type].append(img)
        img.save(path)
        self.count_map[class_type] += 1

    def get_class(self, class_name):
        """
        **SUMMARY**

        Returns the image set that has been turked for the given class.

        **PARAMETERS**

        * *className* - the class name as a string.

        **RETURNS**

        An image set on success, None on failure.

        **EXAMPLE**
        >>> def get_blobs(img):
        >>>     blobs = img.find_blobs()
        >>>     return [b.mMask for b in blobs]

        >>> def scale_inv(img):
        >>>     return img.resize(100,100).invert()

        >>> turker = TurkingModule(['./data/'], ['./turked/'],
        ...                        ['apple', 'banana', 'cherry'],
        ...                        ['a', 'b', 'c'],
        ...                        preprocess=get_blobs, postprocess=scale_inv)
        >>> iset = turker.get_class('cats')
        >>> iset.show()
        """
        return self.class_map.get(class_name)

    def _draw_controls(self, img, font_size, color, spacing):
        img.dl().seet_font_size(font_size)
        img.dl().text("space - skip", (10, spacing), color)
        img.dl().text("esc - exit", (10, 2 * spacing), color)

        y_crd = 3 * spacing
        for k, cls in self.key_map.items():
            txt = k + " - " + cls
            img.dl().text(txt, (10, y_crd), color)
            y_crd += spacing
        return img

    def turk(self, save_original=False, disp_size=(800, 600), show_keys=True,
             font_size=16, color=Color.RED, spacing=10):
        """
        **SUMMARY**

        This function does the turning of the data. The method goes through
        each image, applies the preprocessing (which can return multiple
        images), displays each image with an optional display of the key
        mapping. The user than selects the key that describes the class of the
        image. The image is then post processed and saved to the directory.
        The escape key kills the turking, the space key skips an image.

        **PARAMETERS**

        * *save_original* - if true save the original image versus the
         preprocessed image.
        * *disp_size* - size of the display to create.
        * *show_keys* - Show the key mapping for the turking. Note that on
         small images this may not render correctly.
        * *font_size* - the font size for the turking display.
        * *color* - the font color.
        * *spacing* - the spacing between each line of text on the display.

        **RETURNS**

        Nothing but stores each image in the directory. The image sets are also
        available via the get_class method.

        **EXAMPLE**

        >>> def get_blobs(img):
        >>>     blobs = img.find_blobs()
        >>>     return [b.mMask for b in blobs]

        >>> def scale_inv(img):
        >>>     return img.resize(100,100).invert()

        >>> turker = TurkingModule(['./data/'], ['./turked/'],
        ...                        ['apple', 'banana', 'cherry'],
        ...                        ['a', 'b', 'c'],
        ...                        preprocess=get_blobs, postprocess=scale_inv)
        >>> turker.turk()
        >>> # ~~~ stuff ~~~
        >>> turker.save('./derp.pkl')

        ** TODO **
        TODO: fix the display so that it renders correctly no matter what the
        image size.
        TODO: Make it so you can stop and start turking at any given spot in
        the process
        """
        # FIXME: Rewrite without using of PygameDisplay
        return

        disp = PygameDisplay(disp_size)
        for img in self.src_imgs:
            print img.filename
            samples = self.preprocess(img)
            for sample in samples:
                if show_keys:
                    sample = self._draw_controls(sample, font_size, color,
                                                 spacing)

                sample.save(disp)
                got_key = False
                while not got_key:
                    keys = disp.check_events(True)
                    for k in keys:
                        if k in self.key_map:
                            if save_original:
                                self._save_it(img, self.key_map[k])
                            else:
                                self._save_it(sample, self.key_map[k])
                            got_key = True
                        if k == 'space':
                            got_key = True  # skip
                        if k == 'escape':
                            return

    def save(self, fname):
        """
        **SUMMARY**

        Pickle the relevant data from the turking.

        ** PARAMETERS **

        * *fname* - the file fame.
        """
        save_this = [self.classes, self.directory_map, self.class_map,
                     self.count_map]
        pickle.dump(save_this, open(fname, "wb"))

        # TODO: eventually we should allow the user to randomly
        # split up the data set and then save it.
        # def splitTruthTest(self)
