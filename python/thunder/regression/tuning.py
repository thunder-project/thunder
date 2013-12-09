# tuning <master> <dataFile> <modelFile> <outputDir> <regressMode> <tuningMode>"
#
# fit a parametric tuning curve to regression results
#
# regressMode - form of regression (mean, linear, bilinear)
# tuningMode - parametric tuning curve (circular, gaussian)
#
# example:
# tuning.py local data/fish.txt data/regression/fish_bilinear results bilinear circular
# 

import argparse
import os
from numpy import *
from thunder.util.dataio import *
from thunder.regression.util import *
from thunder.factorization.util import *
from pyspark import SparkContext


def tuning(sc, dataFile, modelFile, outputDir, regressMode, tuningMode):
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)

    # parse data
    lines = sc.textFile(dataFile)
    data = parse(lines, "dff")

    # create models
    regressionModel = RegressionModel.load(modelFile, regressMode)
    tuningModel = TuningModel.load(modelFile, tuningMode)

    # do regression
    betas = regressionModel.fit(data).cache()

    # get statistics
    stats = betas.map(lambda x: x[1])
    saveout(stats, outputDir, "stats", "matlab")

    # get tuning curves
    params = tuningModel.fit(betas.map(lambda x: x[0]))
    saveout(params, outputDir, "params", "matlab")

    if regressMode == "bilinear":
        comps, latent, scores = svd1(betas.map(lambda x: x[2]), 2)
        saveout(comps, outputDir, "comps", "matlab")
        saveout(latent, outputDir, "latent", "matlab")
        saveout(scores, outputDir, "scores", "matlab", 2)

    # get simple measure of response strength
    r = data.map(lambda x: norm(x - mean(x)))
    saveout(r, outputDir, "r", "matlab")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="fit a parametric tuning curve to regression results")
    parser.add_argument("master", type=str)
    parser.add_argument("dataFile", type=str)
    parser.add_argument("modelFile", type=str)
    parser.add_argument("outputDir", type=str)
    parser.add_argument("regressMode", choices=("mean", "linear", "bilinear"),
                        help="the form of regression")
    parser.add_argument("tuningMode", choices=("circular", "gaussian"),
                        help="parametric tuning curve")

    args = parser.parse_args()
    sc = SparkContext(args.master, "tuning")

    tuning(sc, args.dataFile, args.modelFile, args.outputDir + "-tuning", args.regressMode, args.tuningMode)