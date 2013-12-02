# shotgun <master> <dataFile> <modelFile> <outputDir> <lambda>
# 
# use the "shotgun" approach for L1 regularized regression
# parallelizing over features
# algorithm by Bradley et al., 2011, ICML
#
# lambda - learning rate
#
# example:
# shotgun.py local data/shotgun.txt data/regression/shotgun results 10
#

import argparse
import os
from numpy import *
from scipy.linalg import *
from scipy.sparse import *
from thunder.util.dataio import *
from thunder.regression.util import *
from pyspark import SparkContext


def updateFeature(x,y,Ab,b,lam):
    AA = dot(x,x)
    Ay = dot(x,y)
    d_j = Ay - dot(x,Ab) + AA*b
    if d_j < -lam :
        new_value = (d_j + lam)/AA
    elif d_j > lam:
        new_value = (d_j - lam)/AA
    else :
        new_value = 0
    return float(new_value)


def shotgun(sc, dataFile, modelFile, outputDir, lam):
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)

    # parse data
    lines = sc.textFile(dataFile)
    A = parse(lines, "raw", "linear", None, [1, 1]).cache()

    # parse model
    model = RegressionModel.load(modelFile, "shotgun")

    # get constants
    d = A.count()
    n = len(A.first()[1])

    # initialize sparse weight vector
    b = csc_matrix((d, 1))

    # initialize product Ab
    Ab = zeros((n, 1))

    iIter = 1
    nIter = 50
    deltaCheck = 10 ^ 2
    tol = 10 ** -6

    while (iIter < nIter) & (deltaCheck > tol):
        update = A.map(lambda (k, x): (k, updateFeature(x, model.y, Ab, b[k, 0], lam))).filter(
            lambda (k, x): x != b[k, 0]).collect()
        nUpdate = len(update)

        b = b.todok()
        diff = zeros((nUpdate, 1))
        for i in range(nUpdate):
            key = update[i][0]
            value = update[i][1]
            diff[i] = abs(value - b[key, 0])
            b[key, 0] = value
        b = b.tocsc()

        deltaCheck = amax(diff)

        Ab = A.map(lambda (k, x): x * b[k, 0]).reduce(lambda x, y: x + y)

        iIter = iIter + 1

    saveout(b.todense(), outputDir, "b", "matlab")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("master", type=str)
    parser.add_argument("dataFile", type=str)
    parser.add_argument("modelFile", type=str)
    parser.add_argument("outputDir", type=str)
    parser.add_argument("lam", type=double, help="lambda")

    args = parser.parse_args()
    sc = SparkContext(args.master, "shotgun")
    outputDir = args.outputDir + "-shotgun"

    shotgun(sc, args.dataFile, args.modelFile, outputDir, args.lam)
