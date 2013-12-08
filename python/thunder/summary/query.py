# query <master> <inputFile> <inds> <outputDir>
#
# query a data set by averaging together values with given indices
#

import sys
import os
import argparse
from numpy import *
from scipy.linalg import *
from scipy.io import loadmat
from thunder.util.dataio import *
from pyspark import SparkContext


def query(sc, dataFile, outputDir, indsFile):

    if not os.path.exists(outputDir):
        os.makedirs(outputDir)

    # parse data
    lines = sc.textFile(dataFile)
    data = parse(lines, "raw", "linear").cache()

    # loop over indices, averaging time series
    inds = loadmat(indsFile)['inds'][0]
    nInds = len(inds)
    ts = zeros((len(data.first()[1]), nInds))
    for i in range(0, nInds):
        indsTmp = inds[i]
        n = len(indsTmp)
        ts[:, i] = data.filter(lambda k, x: k in indsTmp).map(lambda k, x: x).reduce(lambda x, y: x + y) / n
    saveout(ts, outputDir, "ts", "matlab")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="query time series data by averaging values for given indices")
    parser.add_argument("master", type=str)
    parser.add_argument("dataFile", type=str)
    parser.add_argument("outputDir", type=str)
    parser.add_argument("indsFile", type=str)

    args = parser.parse_args()
    sc = SparkContext(args.master, "query")
    outputDir = args.outputDir + "-query"

    query(sc, args.dataFile, args.outputDir, args.indsFile)