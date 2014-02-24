import os
import argparse
import glob
from thunder.regression.util import RegressionModel
from thunder.util.load import load
from thunder.util.save import save
from pyspark import SparkContext


def regress(data, modelfile, regressmode):
    """Perform mass univariate regression

    :param data: RDD of data points as key value pairs
    :param modelfile: model parameters (string with file location, array, or tuple)
    :param regressmode: form of regression ("linear" or "bilinear")

    :return stats: statistics of the fit
    :return betas: regression coefficients
    """
    # create model
    model = RegressionModel.load(modelfile, regressmode)

    # do regression
    betas, stats, resid = model.fit(data)

    return stats, betas


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="fit a regression model")
    parser.add_argument("master", type=str)
    parser.add_argument("datafile", type=str)
    parser.add_argument("modelfile", type=str)
    parser.add_argument("outputdir", type=str)
    parser.add_argument("regressmode", choices=("linear", "bilinear"), help="form of regression")
    parser.add_argument("--preprocess", choices=("raw", "dff", "sub"), default="raw", required=False)

    args = parser.parse_args()
    egg = glob.glob(os.path.join(os.environ['THUNDER_EGG'], "*.egg"))
    sc = SparkContext(args.master, "regress", pyFiles=egg)
    data = load(sc, args.datafile, args.preprocess)

    stats, betas = regress(data, args.modelfile, args.regressmode)

    outputdir = args.outputdir + "-regress"
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)

    save(stats.cache(), outputdir, "stats", "matlab")
    save(betas.cache(), outputdir, "comps", "matlab")
