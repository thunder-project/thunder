# ref <master> <dataFile> <outputDir> <mode>
#
# compute summary statistics
#
# example:
# pyspark ref.py local data/fish.txt results mean
#

import sys
import os
import argparse
from numpy import median, std
from thunder.util.dataio import *
from pyspark import SparkContext


def ref(sc, dataFile, outputDir, mode):

    if not os.path.exists(outputDir):
        os.makedirs(outputDir)

    # parse data
    lines = sc.textFile(dataFile)
    data = parse(lines, "raw", "xyz").cache()

    # get z ordering
    zinds = data.filter(lambda (k, x): (k[0] == 1) & (k[1] == 1)).map(lambda (k, x): k[2])
    saveout(zinds, outputDir, "zinds", "matlab")

    # compute summary statistics
    if mode == 'median':
        refout = data.map(lambda (k, x): median(x))
    if mode == 'mean':
        refout = data.map(lambda (k, x): mean(x))
    if mode == 'std':
        refout = data.map(lambda (k, x): std(x))

    saveout(refout, outputDir, "ref" + mode, "matlab")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="compute summary statistics on time series data")
    parser.add_argument("master", type=str)
    parser.add_argument("dataFile", type=str)
    parser.add_argument("outputDir", type=str)
    parser.add_argument("mode", choices=("mean", "median", "std"),
                        help="desired summary statistic")

    args = parser.parse_args()
    sc = SparkContext(args.master, "ref")

    ref(sc, args.dataFile, args.outputDir + "-ref", args.mode)
