"""
This class is encapsulates almost everything needed to train, test, and deploy
a multiclass decision tree / forest image classifier. Training data should
be stored in separate directories for each class. This class uses the feature
extractor base class to  convert images into a feature vector. The basic
workflow is as follows.
1. Get data.
2. Setup Feature Extractors (roll your own or use the ones I have written).
3. Train the classifier.
4. Test the classifier.
5. Tweak parameters as necessary.
6. Repeat until you reach the desired accuracy.
7. Save the classifier.
8. Deploy using the classify method.
"""
import glob
import os
import pickle

from simplecv.base import logger, IMAGE_FORMATS
from simplecv.color import Color
from simplecv.drawing_layer import DrawingLayer
from simplecv.image_class import Image


ORANGE_ENABLED = True
try:
    try:
        import orange
    except ImportError:
        import Orange

    import orngTest  # for cross validation
    import orngStat
    import orngEnsemble  # for bagging / boosting

except ImportError:
    ORANGE_ENABLED = False


class TreeClassifier:
    """
    This method encapsulates a number of tree-based machine learning approaches
    and associated meta algorithms.

    Decision trees:
    http://en.wikipedia.org/wiki/Decision_trees

    boosted adpative decision trees
    http://en.wikipedia.org/wiki/Adaboost

    random forrests
    http://en.wikipedia.org/wiki/Random_forest

    bagging (bootstrap aggregating)
    http://en.wikipedia.org/wiki/Bootstrap_aggregating
    """

    TREE_TYPE_DICT = {
        "Tree": 0,  # A vanilla classification tree
        "Bagged": 1,  # Bootstrap aggregating aka bagging - make new
                      # data sets and test on them
        "Forest": 2,  # Lots of little trees
        "Boosted": 3  # Highly optimized trees.
    }

    FOREST_FLAVOR_DICT = {
        "NTrees": 100,       # number of trees in our forest
        "NAttributes": None  # number of attributes per split
                             # sqrt(features) is default
    }

    BOOSTED_FLAVOR_DICT = {
        "NClassifiers": 10,  # number of learners
    }

    BAGGED_FLAVOR_DICT = {
        "NClassifiers": 10,  # numbers of classifiers / tree splits
    }

    def __init__(self, feature_extractors=None, flavor='Tree',
                 flavor_dict=None):
        """
        dist = distance algorithm
        k = number of nearest neighbors
        """
        if not ORANGE_ENABLED:
            logger.warning("I'm sorry, but you need the orange machine "
                           "learning library installed to use this")
            return

        if not feature_extractors:
            feature_extractors = []

        self.class_names = []
        self.data_set_raw = []
        self.data_set_orange = []
        self.classifier = None
        self.learner = None
        self.tree = None
        self.feature_extractors = None
        self.orange_domain = None
        self.flavor_params = None
        self.flavor = self.TREE_TYPE_DICT[flavor]
        if flavor_dict is None:
            if self.flavor == self.TREE_TYPE_DICT["Bagged"]:
                self.flavor_params = self.BAGGED_FLAVOR_DICT
            elif self.flavor == self.TREE_TYPE_DICT["Forest"]:
                # mmmm tastes like pinecones and squirrels
                self.flavor_params = self.FOREST_FLAVOR_DICT
            elif self.flavor == self.TREE_TYPE_DICT["Boosted"]:
                self.flavor_params = self.BOOSTED_FLAVOR_DICT
        else:
            self.flavor_params = flavor_dict
        self.feature_extractors = feature_extractors

    def load(cls, fname):
        """
        Load the classifier from file
        """
        return pickle.load(file(fname))

    load = classmethod(load)

    def save(self, fname):
        """
        Save the classifier to file
        """
        output = open(fname, 'wb')
        pickle.dump(self, output, 2)  # use two otherwise it w
        output.close()

    def __getstate__(self):
        mydict = self.__dict__.copy()
        self.data_set_orange = None
        del mydict['data_set_orange']
        self.orange_domain = None
        del mydict['orange_domain']
        self.learner = None
        del mydict['learner']
        self.tree = None
        del mydict['tree']
        return mydict

    def __setstate__(self, mydict):
        self.__dict__ = mydict
        col_names = []
        for extractor in self.feature_extractors:
            col_names.extend(extractor.get_field_names())
        self.orange_domain = orange.Domain(
            map(orange.FloatVariable, col_names),
            orange.EnumVariable("type", values=self.class_names))
        self.data_set_orange = orange.ExampleTable(self.orange_domain,
                                                   self.data_set_raw)
        if self.flavor == 0:
            self.learner = orange.TreeLearner()
            self.classifier = self.learner(self.data_set_orange)
        elif self.flavor == 1:  # bagged
            self.tree = orange.TreeLearner()
            self.learner = orngEnsemble.BaggedLearner(self.tree,
                                                      t=self.flavor_params[
                                                      "NClassifiers"])
            self.classifier = self.learner(self.data_set_orange)
        elif self.flavor == 2:  # forest
            self.tree = orange.TreeLearner()
            self.learner = orngEnsemble.RandomForestLearner(
                trees=self.flavor_params["NTrees"],
                attributes=self.flavor_params["NAttributes"])
            self.classifier = self.learner(self.data_set_orange)
        elif self.flavor == 3:  # boosted
            self.tree = orange.TreeLearner()
            self.learner = orngEnsemble.BoostedLearner(self.tree,
                                                       t=self.flavor_params[
                                                       "NClassifiers"])
            self.classifier = self.learner(self.data_set_orange)

    def classify(self, image):
        """
        Classify a single image. Takes in an image and returns the string
        of the classification.

        Make sure you haved loaded the feauture extractors and the training
        data.

        """
        feature_vector = []
        for extractor in self.feature_extractors:  # get the features
            feats = extractor.extract(image)
            if feats is not None:
                feature_vector.extend(feats)
        feature_vector.extend([self.class_names[0]])
        test = orange.ExampleTable(self.orange_domain, [feature_vector])
        c = self.classifier(test[0])  # classify
        return str(c)  # return to class name

    def set_feature_extractors(self, extractors):
        """
        Add a list of feature extractors to the classifier. These feature
        extractors
        must match the ones used to train the classifier. If the classifier is
        already trained then this method will require that you retrain the
        data.
        """
        self.feature_extractors = extractors
        return None

    def _train_path(self, path, class_name, subset, disp, verbose):
        count = 0
        files = []
        for ext in IMAGE_FORMATS:
            files.extend(glob.glob(os.path.join(path, ext)))
        if subset > 0:
            nfiles = min(subset, len(files))
        else:
            nfiles = len(files)
        bad_feat = False
        for i in range(nfiles):
            infile = files[i]
            if verbose:
                print "Opening file: " + infile
            img = Image(infile)
            feature_vector = []
            for extractor in self.feature_extractors:
                feats = extractor.extract(img)
                if feats is not None:
                    feature_vector.extend(feats)
                else:
                    bad_feat = True

            if bad_feat:
                bad_feat = False
                continue

            feature_vector.extend([class_name])
            self.data_set_raw.append(feature_vector)
            text = 'Training: ' + class_name
            self._write_text(disp, img, text, Color.WHITE)
            count = count + 1
            del img
        return count

    def _train_image_set(self, imageset, class_name, subset, disp, verbose):
        count = 0
        bad_feat = False
        if subset > 0:
            imageset = imageset[0:subset]
        for img in imageset:
            if verbose:
                print "Opening file: " + img.filename
            feature_vector = []
            for extractor in self.feature_extractors:
                feats = extractor.extract(img)
                if feats is not None:
                    feature_vector.extend(feats)
                else:
                    bad_feat = True

            if bad_feat:
                bad_feat = False
                continue

            feature_vector.extend([class_name])
            self.data_set_raw.append(feature_vector)
            text = 'Training: ' + class_name
            self._write_text(disp, img, text, Color.WHITE)
            count = count + 1
            del img
        return count

    def train(self, images, class_names, disp=None, subset=-1, savedata=None,
              verbose=True):
        """
        Train the classifier.
        images paramater can take in a list of paths or a list of imagesets
        images - the order of the paths or imagesets must be in the same order
        as the class type

        - Note all image classes must be in seperate directories
        - The class names must also align to the directories or imagesets

        disp - if display is a display we show images and class label,
        otherwise nothing is done.

        subset - if subset = -1 we use the whole dataset. If subset = # then we
        use min(#images,subset)

        savedata - if save data is None nothing is saved. If savedata is a file
        name we save the data to a tab delimited file.

        verbose - print confusion matrix and file names
        returns [%Correct %Incorrect Confusion_Matrix]
        """
        #if( (self.flavor == 1 or self.flavor == 3) and len(class_names) > 2):
        #   logger.warning("Boosting / Bagging only works for binary
        #  classification tasks!!!")

        count = 0
        self.class_names = class_names
        # for each class, get all of the data in the path and train
        for i in range(len(class_names)):
            if isinstance(images[i], str):
                count = count + self._train_path(images[i], class_names[i],
                                                 subset, disp, verbose)
            else:
                count = count + self._train_image_set(images[i],
                                                      class_names[i],
                                                      subset, disp, verbose)

        col_names = []
        for extractor in self.feature_extractors:
            col_names.extend(extractor.get_field_names())

        if count <= 0:
            logger.warning("No features extracted - bailing")
            return None

        self.orange_domain = orange.Domain(
            map(orange.FloatVariable, col_names),
            orange.EnumVariable("type", values=self.class_names))
        self.data_set_orange = orange.ExampleTable(self.orange_domain,
                                                   self.data_set_raw)
        if savedata is not None:
            orange.saveTabDelimited(savedata, self.data_set_orange)

        if self.flavor == 0:
            self.learner = orange.TreeLearner()
            self.classifier = self.learner(self.data_set_orange)
        elif self.flavor == 1:  # bagged
            self.tree = orange.TreeLearner()
            self.learner = orngEnsemble.BaggedLearner(self.tree,
                                                      t=self.flavor_params[
                                                      "NClassifiers"])
            self.classifier = self.learner(self.data_set_orange)
        elif self.flavor == 2:  # forest
            self.tree = orange.TreeLearner()
            self.learner = orngEnsemble.RandomForestLearner(
                trees=self.flavor_params["NTrees"],
                attributes=self.flavor_params["NAttributes"])
            self.classifier = self.learner(self.data_set_orange)
        elif self.flavor == 3:  # boosted
            self.tree = orange.TreeLearner()
            self.learner = orngEnsemble.BoostedLearner(self.tree,
                                                       t=self.flavor_params[
                                                       "NClassifiers"])
            self.classifier = self.learner(self.data_set_orange)

        correct = 0
        incorrect = 0
        for i in range(count):
            c = self.classifier(self.data_set_orange[i])
            test = self.data_set_orange[i].getclass()
            if verbose:
                print "original", test, "classified as", c
            if test == c:
                correct = correct + 1
            else:
                incorrect = incorrect + 1

        good = 100 * (float(correct) / float(count))
        bad = 100 * (float(incorrect) / float(count))

        confusion = 0
        if len(self.class_names) > 2:
            cross_validator = orngTest.learnAndTestOnLearnData(
                [self.learner], self.data_set_orange)
            confusion = orngStat.confusionMatrices(cross_validator)[0]

        if verbose:
            print("Correct: " + str(good))
            print("Incorrect: " + str(bad))
            if confusion != 0:
                classes = self.data_set_orange.domain.classVar.values
                print "\t" + "\t".join(classes)
                for className, classConfusions in zip(classes, confusion):
                    print ("%s" + ("\t%i" * len(classes))) % (
                        (className, ) + tuple(classConfusions))

        if self.flavor == 0:
            self._print_tree(self.classifier)

        return [good, bad, confusion]

    def test(self, images, class_names, disp=None, subset=-1, savedata=None,
             verbose=True):
        """
        Test the classifier.
        images paramater can take in a list of paths or a list of imagesets
        images - the order of the paths or imagesets must be in the same order
        as the class type

        - Note all image classes must be in seperate directories
        - The class names must also align to the directories or imagesets

        disp - if display is a display we show images and class label,
        otherwise nothing is done.

        subset - if subset = -1 we use the whole dataset. If subset = # then we
        use min(#images,subset)

        savedata - if save data is None nothing is saved. If savedata is a file
        name we save the data to a tab delimited file.

        verbose - print confusion matrix and file names
        returns [%Correct %Incorrect Confusion_Matrix]
        """
        count = 0
        correct = 0
        self.class_names = class_names
        col_names = []
        for extractor in self.feature_extractors:
            col_names.extend(extractor.get_field_names())
            if self.orange_domain is None:
                self.orange_domain = orange.Domain(
                    map(orange.FloatVariable, col_names),
                    orange.EnumVariable("type", values=self.class_names))

        dataset = []
        for i in range(len(class_names)):
            if isinstance(images[i], str):
                [dataset, cnt, crct] = self._test_path(images[i],
                                                       class_names[i],
                                                       dataset, subset, disp,
                                                       verbose)
                count = count + cnt
                correct = correct + crct
            else:
                [dataset, cnt, crct] = self._test_image_set(images[i],
                                                            class_names[i],
                                                            dataset, subset,
                                                            disp, verbose)
                count = count + cnt
                correct = correct + crct

        test_data = orange.ExampleTable(self.orange_domain, dataset)

        if savedata is not None:
            orange.saveTabDelimited(savedata, test_data)

        confusion = 0
        if len(self.class_names) > 2:
            cross_validator = orngTest.learnAndTestOnTestData(
                [self.learner], self.data_set_orange, test_data)
            confusion = orngStat.confusionMatrices(cross_validator)[0]

        good = 100 * (float(correct) / float(count))
        bad = 100 * (float(count - correct) / float(count))
        if verbose:
            print("Correct: " + str(good))
            print("Incorrect: " + str(bad))
            if confusion != 0:
                classes = self.data_set_orange.domain.classVar.values
                print "\t" + "\t".join(classes)
                for className, classConfusions in zip(classes, confusion):
                    print ("%s" + ("\t%i" * len(classes))) % (
                        (className, ) + tuple(classConfusions))
        return [good, bad, confusion]

    def _test_path(self, path, class_name, dataset, subset, disp, verbose):
        count = 0
        correct = 0
        bad_feat = False
        files = []
        for ext in IMAGE_FORMATS:
            files.extend(glob.glob(os.path.join(path, ext)))
        if subset > 0:
            nfiles = min(subset, len(files))
        else:
            nfiles = len(files)
        for i in range(nfiles):
            infile = files[i]
            if verbose:
                print "Opening file: " + infile
            img = Image(infile)
            feature_vector = []
            for extractor in self.feature_extractors:
                feats = extractor.extract(img)
                if feats is not None:
                    feature_vector.extend(feats)
                else:
                    bad_feat = True
            if bad_feat:
                del img
                bad_feat = False
                continue
            feature_vector.extend([class_name])
            dataset.append(feature_vector)
            test = orange.ExampleTable(self.orange_domain, [feature_vector])
            c = self.classifier(test[0])
            test_class = test[0].getclass()
            if test_class == c:
                text = "Classified as " + str(c)
                self._write_text(disp, img, text, Color.GREEN)
                correct = correct + 1
            else:
                text = "Mislassified as " + str(c)
                self._write_text(disp, img, text, Color.RED)
            count = count + 1
            del img

        return [dataset, count, correct]

    def _test_image_set(self, imageset, class_name, dataset, subset, disp,
                        verbose):
        count = 0
        correct = 0
        bad_feat = False
        if subset > 0:
            imageset = imageset[0:subset]
        for img in imageset:
            if verbose:
                print "Opening file: " + img.filename
            feature_vector = []
            for extractor in self.feature_extractors:
                feats = extractor.extract(img)
                if feats is not None:
                    feature_vector.extend(feats)
                else:
                    bad_feat = True
            if bad_feat:
                del img
                bad_feat = False
                continue
            feature_vector.extend([class_name])
            dataset.append(feature_vector)
            test = orange.ExampleTable(self.orange_domain, [feature_vector])
            c = self.classifier(test[0])
            test_class = test[0].getclass()
            if test_class == c:
                text = "Classified as " + str(c)
                self._write_text(disp, img, text, Color.GREEN)
                correct = correct + 1
            else:
                text = "Mislassified as " + str(c)
                self._write_text(disp, img, text, Color.RED)
            count = count + 1
            del img

        return [dataset, count, correct]

    def _write_text(self, disp, img, txt, color):
        if disp is not None:
            txt = ' ' + txt + ' '
            img = img.adaptive_scale(disp.resolution)
            layer = DrawingLayer((img.width, img.height))
            layer.set_font_size(60)
            layer.ez_view_text(txt, (20, 20), fgcolor=color)
            img.add_drawing_layer(layer)
            img.apply_layers()
            img.save(disp)

    def _print_tree(self, x):
        #adapted from the orange documentation
        if type(x) == orange.TreeClassifier:
            self._print_tree0(x.tree, 0)
        elif type(x) == orange.TreeNode:
            self._print_tree0(x, 0)
        else:
            raise TypeError("invalid parameter")

    def _print_tree0(self, node, level):
        #adapted from the orange documentation
        if not node:
            print " " * level + "<null node>"
            return

        if node.branchSelector:
            node_desc = node.branchSelector.classVar.name
            node_cont = node.distribution
            print "\n" + "   " * level + "%s (%s)" % (node_desc, node_cont),
            for i in range(len(node.branches)):
                print "\n" + "   " * level + ": %s" % node.branchDescriptions[
                    i],
                self._print_tree0(node.branches[i], level + 1)
        else:
            node_cont = node.distribution
            major_class = node.nodeClassifier.defaultValue
            print "--> %s (%s) " % (major_class, node_cont)
