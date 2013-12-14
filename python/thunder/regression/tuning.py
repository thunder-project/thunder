# fit a parametric tuning curve to regression results
#
# example:
# tuning.py local data/fish.txt raw data/regression/fish_bilinear results bilinear circular

import argparse
import os
from numpy import *
from thunder.util.dataio import *
from thunder.regression.util import *
from thunder.factorization.util import *
from pyspark import SparkContext


def tuning(data, modelFile, regressMode, tuningMode):

    # create models
    regressionModel = RegressionModel.load(modelFile, regressMode)
    tuningModel = TuningModel.load(modelFile, tuningMode)

    # do regression
    betas = regressionModel.fit(data).cache()

    # get statistics
    stats = betas.map(lambda x: x[1])

    # get tuning curves
    params = tuningModel.fit(betas.map(lambda x: x[0]))

    # get simple measure of response strength
    r = data.map(lambda x: norm(x - mean(x)))

    if regressMode == "bilinear":
        comps, latent, scores = svd1(betas.map(lambda x: x[2]), 2)
        return params, stats, r, comps, latent, scores
    else:
        return params, stats, r

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="fit a parametric tuning curve to regression results")
    parser.add_argument("master", type=str)
    parser.add_argument("dataFile", type=str)
    parser.add_argument("dataMode", choices=("raw", "dff", "sub"), help="form of data preprocessing")
    parser.add_argument("modelFile", type=str)
    parser.add_argument("outputDir", type=str)
    parser.add_argument("regressMode", choices=("mean", "linear", "bilinear"),
                        help="the form of regression")
    parser.add_argument("tuningMode", choices=("circular", "gaussian"),
                        help="parametric tuning curve")

    args = parser.parse_args()
    sc = SparkContext(args.master, "tuning")
    lines = sc.textFile(args.dataFile)
    data = parse(lines, "dff").cache()

    outputDir = args.outputDir + "-tuning"
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)

    if args.regressMode == "bilinear":
        params, stats, r, comps, latent, scores = tuning(data, args.modelFile, args.regressMode, args.tuningMode)
        saveout(params, outputDir, "params", "matlab")
        saveout(stats, outputDir, "stats", "matlab")
        saveout(r, outputDir, "r", "matlab")
        saveout(comps, outputDir, "comps", "matlab")
        saveout(scores, outputDir, "scores", "matlab", 2)
        saveout(latent, outputDir, "latent", "matlab")

    if args.regressMode == "linear":
        params, stats, r = tuning(data, args.modelFile, args.regressMode, args.tuningMode)
        saveout(params, outputDir, "params", "matlab")
        saveout(stats, outputDir, "stats", "matlab")
        saveout(r, outputDir, "r", "matlab")
