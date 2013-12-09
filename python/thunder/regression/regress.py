# regress <master> <dataFile> <modelFile> <outputDir> <regressMode>
#
# fit a regression model
#
# regressMode - form of regression (mean, linear, bilinear)
#
# example:
# regress.py local data/fish.txt data/regression/fish_linear results linear
# 

import argparse
import os
from numpy import *
from scipy.linalg import *
from thunder.util.dataio import *
from thunder.regression.util import *
from thunder.factorization.util import *
from pyspark import SparkContext


def regress(sc, dataFile, modelFile, outputDir, regressMode):
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)

    # parse data
    lines = sc.textFile(dataFile)
    data = parse(lines, "dff")
    data.cache()

    # create model
    model = RegressionModel.load(modelFile, regressMode)

    # do regression
    betas = model.fit(data).cache()

    # get statistics
    stats = betas.map(lambda x: x[1])
    saveout(stats, outputDir, "stats", "matlab")

    # do PCA
    comps, latent, scores = svd1(betas.map(lambda x: x[0]), 2)
    saveout(comps, outputDir, "comps", "matlab")
    saveout(latent, outputDir, "latent", "matlab")
    saveout(scores, outputDir, "scores", "matlab", 2)

    # compute trajectories from raw data
    traj = model.fit(data, comps)
    saveout(traj, outputDir, "traj", "matlab")

    # get simple measure of response strength
    r = data.map(lambda x: norm(x - mean(x)))
    saveout(r, outputDir, "r", "matlab")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="fit a regression model")
    parser.add_argument("master", type=str)
    parser.add_argument("dataFile", type=str)
    parser.add_argument("modelFile", type=str)
    parser.add_argument("outputDir", type=str)
    parser.add_argument("regressMode", choices=("mean", "linear", "bilinear"),
                        help="the form of regression")

    args = parser.parse_args()
    sc = SparkContext(args.master, "regress")

    regress(sc, args.dataFile, args.modelFile, args.outputDir + "-regress", args.regressMode)