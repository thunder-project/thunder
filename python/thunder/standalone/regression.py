"""
Example standalone app for mass-univariate regression
"""

import argparse
from thunder.utils.context import ThunderContext
from thunder.regression import RegressionModel
from thunder.utils.save import save


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="fit a regression model")
    parser.add_argument("datafile", type=str)
    parser.add_argument("modelfile", type=str)
    parser.add_argument("outputdir", type=str)
    parser.add_argument("regressmode", choices=("mean", "linear", "bilinear"), help="form of regression")
    parser.add_argument("--preprocess", choices=("raw", "dff", "sub", "dff-highpass", "dff-percentile"
                        "dff-detrendnonlin", "dff-detrend-percentile"), default="raw", required=False)

    args = parser.parse_args()

    tsc = ThunderContext.start(appName="regress")

    data = tsc.loadText(args.datafile, args.preprocess)
    result = RegressionModel.load(args.modelfile, args.regressmode).fit(data)

    outputdir = args.outputdir + "-regress"
    save(result.select('stats'), outputdir, "stats", "matlab")
    save(result.select('betas'), outputdir, "betas", "matlab")