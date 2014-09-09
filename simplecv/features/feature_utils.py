"""
So this is a place holder for some routines that should live in
featureset if we can make it specific to a type of features
"""
import numpy as np


def get_parallel_sets(line_fs, parallel_threshold=2):
    result = []
    size = len(line_fs)
    #construct the pairwise cross product ignoring dupes
    for i in range(0, size):
        for j in range(0, size):
            if j <= i:
                result.append(np.Inf)
            else:
                result.append(np.abs(line_fs[i].cross(line_fs[j])))

    result = np.array(result)
    # reshape it
    result = result.reshape(size, size)
    # find the lines that are less than our thresh
    line1, line2 = np.where(result < parallel_threshold)
    idxs = zip(line1, line2)
    result = []
    # now construct the line pairs
    for idx in idxs:
        result.append((line_fs[idx[0]], line_fs[idx[1]]))
    return result


def parallel_distance(line1, line2):
    # FIXME: need to be added
    pass
