"""
Standalone app for mass-unvariate tuning analyses
"""

import os
import argparse
import glob
from thunder.regression import RegressionModel, TuningModel
from thunder.utils import load
from thunder.utils import save
from pyspark import SparkContext


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="fit a parametric tuning curve to regression results")
    parser.add_argument("datafile", type=str)
    parser.add_argument("tuningmodelfile", type=str)
    parser.add_argument("outputdir", type=str)
    parser.add_argument("tuningmode", choices=("circular", "gaussian"), help="form of tuning curve")
    parser.add_argument("--regressmodelfile", type=str)
    parser.add_argument("--regressmode", choices=("linear", "bilinear"), help="form of regression")
    parser.add_argument("--preprocess", choices=("raw", "dff", "sub", "dff-highpass", "dff-percentile"
                        "dff-detrendnonlin", "dff-detrend-percentile"), default="raw", required=False)

    args = parser.parse_args()
    
    sc = SparkContext(appName="tuning")

    data = load(sc, args.datafile, args.preprocess)
    tuningmodel = TuningModel.load(args.tuningmodelfile, args.tuningmode)
    if args.regressmodelfile is not None:
        # use regression results
        regressmodel = RegressionModel.load(args.regressmodelfile, args.regressmode)
        betas, stats, resid = regressmodel.fit(data)
        params = tuningmodel.fit(betas)
    else:
        # use data
        params = tuningmodel.fit(data)

    outputdir = args.outputdir + "-tuning"
    save(params, outputdir, "params", "matlab")
