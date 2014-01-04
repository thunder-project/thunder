# use the "shotgun" approach for L1 regularized regression
# parallelizing over features
# algorithm by Bradley et al., 2011, ICML
#
# example:
# shotgun.py local data/shotgun.txt raw data/regression/shotgun results 10


import argparse
import os
from numpy import *
from scipy.linalg import *
from scipy.sparse import *
from thunder.util.dataio import *
from thunder.regression.util import *
from pyspark import SparkContext


def updateFeature(x, y, Ab, b, lam):
    AA = dot(x, x)
    Ay = dot(x, y)
    d_j = Ay - dot(x, Ab) + AA*b
    if d_j < -lam:
        new_value = (d_j + lam)/AA
    elif d_j > lam:
        new_value = (d_j - lam)/AA
    else:
        new_value = 0
    return float(new_value)


def shotgun(data, modelFile, lam):

    # parse model
    model = RegressionModel.load(modelFile, "shotgun")

    # get constants
    d = data.count()
    n = len(data.first()[1])

    # initialize sparse weight vector
    b = csc_matrix((d, 1))

    # initialize product Ab
    Ab = zeros((n, 1))

    iIter = 1
    nIter = 50
    deltaCheck = 10 ^ 2
    tol = 10 ** -6

    while (iIter < nIter) & (deltaCheck > tol):
        update = data.map(lambda (k, x): (k, updateFeature(x, model.y, Ab, b[k, 0], lam))).filter(
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

        Ab = data.map(lambda (k, x): x * b[k, 0]).reduce(lambda x, y: x + y)

        iIter += 1

    return b.todense()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("master", type=str)
    parser.add_argument("dataFile", type=str)
    parser.add_argument("dataMode", choices=("raw", "dff", "sub"), help="form of data preprocessing")
    parser.add_argument("modelFile", type=str)
    parser.add_argument("outputDir", type=str)
    parser.add_argument("lam", type=double, help="lambda")

    args = parser.parse_args()
    egg = glob.glob(os.environ['THUNDER_EGG'] + "*.egg")
    sc = SparkContext(args.master, "shotgun", pyFiles=egg)
    lines = sc.textFile(args.dataFile)
    data = parse(lines, args.dataMode, "linear", None, [1, 1]).cache()

    b = shotgun(data, args.modelFile, args.lam)

    outputDir = args.outputDir + "-shotgun"
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)

    saveout(b, outputDir, "b", "matlab")
