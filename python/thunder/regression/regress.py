# fit a regression model
#
# example:
# regress.py local data/fish.txt raw data/regression/fish_linear results linear


import argparse
import os
from numpy import *
from scipy.linalg import *
from thunder.util.dataio import *
from thunder.regression.util import *
from thunder.factorization.util import *
from pyspark import SparkContext


def regress(data, modelFile, regressMode):

    # create model
    model = RegressionModel.load(modelFile, regressMode)

    # do regression
    betas = model.fit(data).cache()

    # get statistics
    stats = betas.map(lambda x: x[1])

    # do PCA
    comps, latent, scores = svd1(betas.map(lambda x: x[0]), 2)

    # compute trajectories from raw data
    traj = model.fit(data, comps)

    # get simple measure of response strength
    r = data.map(lambda x: norm(x - mean(x)))

    return betas, stats, comps, latent, scores, traj, r


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="fit a regression model")
    parser.add_argument("master", type=str)
    parser.add_argument("dataFile", type=str)
    parser.add_argument("dataMode", choices=("raw", "dff", "sub"), help="form of data preprocessing")
    parser.add_argument("modelFile", type=str)
    parser.add_argument("outputDir", type=str)
    parser.add_argument("regressMode", choices=("mean", "linear", "bilinear"),
                        help="the form of regression")

    args = parser.parse_args()
    egg = os.path.join(os.path.dirname(__file__), "../../dist/Thunder-1.0-py2.7.egg")
    sc = SparkContext(args.master, "regress", pyFiles=[egg])
    lines = sc.textFile(args.dataFile)
    data = parse(lines, args.dataMode).cache()

    betas, stats, comps, latent, scores, traj, r = regress(data, args.modelFile, args.regressMode)

    outputDir = args.outputDir + "-regress"
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)

    saveout(stats, outputDir, "stats", "matlab")
    saveout(comps, outputDir, "comps", "matlab")
    saveout(latent, outputDir, "latent", "matlab")
    saveout(scores, outputDir, "scores", "matlab", 2)
    saveout(traj, outputDir, "traj", "matlab")
    saveout(r, outputDir, "r", "matlab")