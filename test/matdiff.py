#!/usr/bin/env python
"""
Simple diff of Matlab .mat files.  These files can contain modification
timestamps in their headers, so regular `diff` won't work.

Arrays are compared using numpy.allclose after converting NaN values
using numpy.nan_to_num().

Can compare two directories with .mat files that have the same filenames,
or two .mat files.  This is useful for verifying that code modifications
didn't change the computations' results.
"""
import numpy as np
import os
import sys
from scipy.io import loadmat


def mat_files_equal(a_filename, b_filename):
    a = loadmat(a_filename)
    b = loadmat(b_filename)
    if a.keys() != b.keys():
        print "Files have different keys"
        return False
    else:
        for key in a.keys():
            if key == "__header__":
                # Headers are allowed to differ, since they could have
                # different creation timestamps.
                continue
            elif isinstance(a[key], np.ndarray):
                # nan is unequal to anything, so let's replace it:
                if not np.allclose(np.nan_to_num(a[key]),
                                   np.nan_to_num(b[key])):
                    print "Unequal arrays for key '%s'" % key
                    return False
            elif a[key] != b[key]:
                print "Unequal scalars for key '%s'" % key
                return False
        return True


def assert_mat_files_equal(a, b):
    if not mat_files_equal(a, b):
        print "Files %s and %s are different" % (a, b)
        exit(-1)


if __name__ == "__main__":
    a = sys.argv[1]
    b = sys.argv[2]
    if os.path.isdir(a) and os.path.isdir(b):
        for filename in os.listdir(a):
            assert_mat_files_equal(os.path.join(a, filename),
                                   os.path.join(b, filename))
    elif os.path.isfile(a) and os.path.isfile(b):
        assert_mat_files_equal(a, b)
    else:
        print "Must compare two files or two directories"
        sys.exit(-1)
