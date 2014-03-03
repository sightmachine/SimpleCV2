"""
So this is a place holder for some routines that should live in
featureset if we can make it specific to a type of features
"""
import numpy as np


def get_parallel_sets(line_fs, parallel_thresh=2):
    result = []
    sz = len(line_fs)
    #construct the pairwise cross product ignoring dupes
    for i in range(0, sz):
        for j in range(0, sz):
            if j <= i:
                result.append(np.Inf)
            else:
                result.append(np.abs(line_fs[i].cross(line_fs[j])))

    result = np.array(result)
    # reshape it
    result = result.reshape(sz, sz)
    # find the lines that are less than our thresh
    l1, l2 = np.where(result < parallel_thresh)
    idxs = zip(l1, l2)
    result = []
    # now construct the line pairs
    for idx in idxs:
        result.append((line_fs[idx[0]], line_fs[idx[1]]))
    return result


def parallel_distance(line1, line2):
    pass
