# crosscorr <master> <dataFile> <modelFile> <outputDir> <lag>
#
# cross correlate time series
#
# maximum lag
#
# example:
# crosscorr.py local data/fish.txt data/regression/fish_correlation results 5
#

import argparse
import os
from numpy import *
from scipy.linalg import *
from thunder.util.dataio import *
from thunder.regression.util import *
from thunder.factorization.util import *
from pyspark import SparkContext


def crosscorr(data, modelFile, lag):

    # create model
    model = RegressionModel.load(modelFile, "crosscorr", lag)

    # do cross correlation
    betas = model.fit(data).cache()

    r = data.map(lambda x: norm(x - mean(x)))

    if lag is not 0:
        # do PCA
        comps, latent, scores = svd1(betas, 2)
        return betas, r, comps, latent, scores
    else:
        return betas, r


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="fit a regression model")
    parser.add_argument("master", type=str)
    parser.add_argument("dataFile", type=str)
    parser.add_argument("dataMode", choices=("raw", "dff", "sub"), help="form of data preprocessing")
    parser.add_argument("modelFile", type=str)
    parser.add_argument("outputDir", type=str)
    parser.add_argument("lag", type=int)

    args = parser.parse_args()
    egg = glob.glob(os.environ['THUNDER_EGG'] + "*.egg")
    sc = SparkContext(args.master, "crosscorr", pyFiles=egg)
    lines = sc.textFile(args.dataFile)
    data = parse(lines, args.dataMode).cache()

    outputDir = args.outputDir + "-crosscorr"
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)

    if args.lag is not 0:
        betas, r, comps, latent, scores = crosscorr(data, args.modelFile, args.lag)
        saveout(comps, outputDir, "comps", "matlab")
        saveout(latent, outputDir, "latent", "matlab")
        saveout(scores, outputDir, "scores", "matlab", 2)
    else:
        betas, r = crosscorr(data, args.modelFile, args.lag)

    saveout(r, outputDir, "r", "matlab")
    saveout(betas, outputDir, "betas", "matlab", args.lag*2 + 1)
