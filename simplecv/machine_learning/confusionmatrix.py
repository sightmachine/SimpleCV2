import numpy as np


# FIXME: No docs
class ConfusionMatrix():

    def __init__(self, classlist):
        self.classlist = classlist
        self.class_count = len(classlist)
        self.confusion_matrix = np.zeros([self.class_count, self.class_count])
        self.correct_count = 0
        self.incorrect_count = 0
        self.total_count = 0
        self.name_map = {}
        idx = 0
        for obj in classlist:
            self.name_map[obj] = idx
            idx = idx + 1

    def add_data_point(self, truth_name, test_name):
        self.confusion_matrix[self.name_map[truth_name]][
            self.name_map[test_name]] += 1
        if truth_name == test_name:
            self.correct_count += 1
        else:
            self.incorrect_count += 1

        self.total_count += 1

    def get_correct_percent(self):
        if self.total_count > 0 and self.correct_count:
            return np.around(
                float(self.correct_count) / float(self.total_count), 4)
        else:
            return 0.00

    def get_incorrect_percent(self):
        if self.total_count > 0 and self.correct_count:
            return np.around(
                float(self.incorrect_count) / float(self.total_count), 4)
        else:
            return 0.00

    def get_class_correct_percent(self, class_name):
        total = float(
            np.sum(self.confusion_matrix[:, self.name_map[class_name]]))
        correct = float(
            self.confusion_matrix[self.name_map[class_name],
                                  self.name_map[class_name]])
        if correct == 0 or total == 0:
            return 0
        else:
            return np.around(correct / total, 2)

    def get_class_incorrect_percent(self, class_name):
        total = float(
            np.sum(self.confusion_matrix[:, self.name_map[class_name]]))
        correct = float(
            self.confusion_matrix[self.name_map[class_name],
                                  self.name_map[class_name]])
        incorrect = total - correct
        if incorrect == 0 or total == 0:
            return 0
        else:
            return np.around(incorrect / total, 2)

    def get_class_correct(self, class_name):
        correct = self.confusion_matrix[
            self.name_map[class_name], self.name_map[class_name]]
        return correct

    def get_class_incorrect(self, class_name):
        total = np.sum(self.confusion_matrix[:, self.name_map[class_name]])
        correct = self.confusion_matrix[
            self.name_map[class_name], self.name_map[class_name]]
        incorrect = total - correct
        return incorrect

    def get_class_count(self, class_name):
        return np.sum(self.confusion_matrix[:, self.name_map[class_name]])

    def get_misclassified_count(self, class_name):
        # if we're class A, this returns the number of class B, C ...
        # that were classified as A
        count = np.sum(self.confusion_matrix[self.name_map[class_name], :])
        correct = self.confusion_matrix[
            [self.name_map[class_name]], self.name_map[class_name]]
        total = count - correct
        return int(total[0])

    def to_string(self, pad_sz=7):
        ret_val = 50 * '#'
        ret_val += "\n"
        ret_val += "Total Data Points " + str(self.total_count) + "\n"
        ret_val += "Correct Data Points " + str(self.correct_count) + "\n"
        ret_val += "Incorrect Data Points " + str(self.incorrect_count) + "\n"
        ret_val += "\n"
        ret_val += "Correct " + str(
            100.00 * self.get_correct_percent()) + "%\n"
        ret_val += "Incorrect " + str(
            100.00 * self.get_incorrect_percent()) + "% \n"
        ret_val += 50 * '#'
        ret_val += '\n'
        wrd_len = 0
        sz = pad_sz
        for c in self.classlist:
            if len(c) > wrd_len:
                wrd_len = len(c)

        top = (wrd_len + 1) * " "
        for c in self.classlist:
            top = top + c[0:np.min([len(c), sz])].rjust(sz, " ") + "|"
        ret_val += top + "\n"
        for i in range(0, len(self.classlist)):
            line = self.classlist[i].rjust(wrd_len, " ") + "|"
            nums = self.confusion_matrix[i]
            for n in nums:
                line += str(n).rjust(sz, " ") + "|"
            ret_val += line + "\n"
        ret_val += 50 * '#'
        ret_val += "\n"
        return ret_val
