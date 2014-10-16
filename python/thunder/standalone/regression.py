"""
Example standalone app for mass-univariate regression
"""

import argparse
from thunder import ThunderContext, RegressionModel, export


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="fit a regression model")
    parser.add_argument("datafile", type=str)
    parser.add_argument("modelfile", type=str)
    parser.add_argument("outputdir", type=str)
    parser.add_argument("regressmode", choices=("mean", "linear", "bilinear"), help="form of regression")

    args = parser.parse_args()

    tsc = ThunderContext.start(appName="regress")

    data = tsc.loadText(args.datafile, args.preprocess)
    result = RegressionModel.load(args.modelfile, args.regressmode).fit(data)

    outputdir = args.outputdir + "-regress"
    export(result.select('stats'), outputdir, "stats", "matlab")
    export(result.select('betas'), outputdir, "betas", "matlab")