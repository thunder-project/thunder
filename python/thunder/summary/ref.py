# compute summary statistics
#
# example:
# pyspark ref.py local data/fish.txt raw results mean
#

import os
import argparse
from numpy import median, std
from thunder.util.dataio import *
from pyspark import SparkContext


def ref(data, mode):

    # get z ordering
    zinds = data.filter(lambda (k, x): (k[0] == 1) & (k[1] == 1)).map(lambda (k, x): k[2])

    # compute summary statistics
    if mode == 'median':
        refout = data.map(lambda (k, x): median(x))
    if mode == 'mean':
        refout = data.map(lambda (k, x): mean(x))
    if mode == 'std':
        refout = data.map(lambda (k, x): std(x))

    return refout, zinds


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="compute summary statistics on time series data")
    parser.add_argument("master", type=str)
    parser.add_argument("dataFile", type=str)
    parser.add_argument("dataMode", choices=("raw", "dff", "sub"), help="form of data preprocessing")
    parser.add_argument("outputDir", type=str)
    parser.add_argument("mode", choices=("mean", "median", "std"),
                        help="desired summary statistic")

    args = parser.parse_args()
    sc = SparkContext(args.master, "ref")

    lines = sc.textFile(args.dataFile)
    data = parse(lines, "raw", "xyz").cache()

    refout, zinds = ref(data, args.mode)

    outputDir = args.outputDir + "-ref",

    outputDir = args.outputDir + "-ref"
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)

    saveout(refout, outputDir, "ref" + args.mode, "matlab")
    saveout(zinds, outputDir, "zinds", "matlab")