"""
Standalone app for mass-unvariate regression
"""

import os
import argparse
import glob
from thunder.regression import RegressionModel
from thunder.utils import load
from thunder.utils import save
from pyspark import SparkContext


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="fit a regression model")
    parser.add_argument("datafile", type=str)
    parser.add_argument("modelfile", type=str)
    parser.add_argument("outputdir", type=str)
    parser.add_argument("regressmode", choices=("mean", "linear", "bilinear"), help="form of regression")
    parser.add_argument("--preprocess", choices=("raw", "dff", "sub", "dff-highpass", "dff-percentile"
                        "dff-detrendnonlin", "dff-detrend-percentile"), default="raw", required=False)

    args = parser.parse_args()

    sc = SparkContext(appName="regress")

    data = load(sc, args.datafile, args.preprocess)
    stats, betas, resid = RegressionModel.load(args.modelfile, args.regressmode).fit(data)

    outputdir = args.outputdir + "-regress"
    save(stats, outputdir, "stats", "matlab")
    save(betas, outputdir, "betas", "matlab")
