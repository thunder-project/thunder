# query a data set by averaging together values with given indices
#
# example:
# query.py local data/fish.txt raw data/summary/fish_inds.mat ~/spark-test 88 76

import os
import argparse
from numpy import *
from scipy.linalg import *
from scipy.io import loadmat
from thunder.util.dataio import *
from pyspark import SparkContext


def query(data, indsFile):

    # loop over indices, averaging time series
    inds = loadmat(indsFile)['inds'][0]
    nInds = len(inds)
    ts = zeros((len(data.first()[1]), nInds))
    for i in range(0, nInds):
        indsTmp = inds[i]
        n = len(indsTmp)
        ts[:, i] = data.filter(lambda (k, x): k in indsTmp).map(lambda (k, x): x).reduce(lambda x, y: x + y) / n

    return ts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="query time series data by averaging values for given indices")
    parser.add_argument("master", type=str)
    parser.add_argument("dataFile", type=str)
    parser.add_argument("dataMode", choices=("raw", "dff", "sub"), help="form of data preprocessing")
    parser.add_argument("indsFile", type=str)
    parser.add_argument("outputDir", type=str)
    parser.add_argument("mxX", type=int)
    parser.add_argument("mxY", type=int)

    args = parser.parse_args()
    egg = glob.glob(os.environ['THUNDER_EGG'] + "*.egg")
    sc = SparkContext(args.master, "query", pyFiles=egg)

    # TODO: once top is implemented in pyspark, use instead of specifying mxX and mxY
    lines = sc.textFile(args.dataFile)
    data = parse(lines, "dff", "linear", None, [args.mxX, args.mxY]).cache()

    ts = query(data, args.indsFile)

    outputDir = args.outputDir + "-query"
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)

    saveout(ts, outputDir, "ts", "matlab")
