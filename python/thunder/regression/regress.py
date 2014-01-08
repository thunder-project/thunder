import os
import argparse
import glob
from thunder.regression.util import RegressionModel
from thunder.factorization.util import svd
from thunder.util.dataio import parse, saveout
from pyspark import SparkContext


def regress(data, modelfile, regressmode):
    """perform mass univariate regression,
    followed by principal components analysis
    to reduce dimensionality

    arguments:
    data - RDD of data points
    modelfile - model parameters (string with file location, array, or tuple)
    regressmode - form of regression ("linear" or "bilinear")

    returns:
    stats - statistics of the fit
    comps, latent, scores, traj - results of principal components analysis
    """
    # create model
    model = RegressionModel.load(modelfile, regressmode)

    # do regression
    betas, stats, resid = model.fit(data)

    # do principal components analysis
    scores, latent, comps = svd(betas, 2)

    # compute trajectories from raw data
    traj = model.fit(data, comps)

    return stats, comps, latent, scores, traj


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="fit a regression model")
    parser.add_argument("master", type=str)
    parser.add_argument("datafile", type=str)
    parser.add_argument("modelfile", type=str)
    parser.add_argument("outputdir", type=str)
    parser.add_argument("regressmode", choices=("linear", "bilinear"), help="form of regression")
    parser.add_argument("--preprocess", choices=("raw", "dff", "sub"), default="raw", required=False)

    args = parser.parse_args()
    egg = glob.glob(os.environ['THUNDER_EGG'] + "*.egg")
    sc = SparkContext(args.master, "regress", pyFiles=egg)
    lines = sc.textFile(args.datafile)
    data = parse(lines, args.preprocess).cache()

    stats, comps, latent, scores, traj = regress(data, args.modelfile, args.regressmode)

    outputdir = args.outputdir + "-regress"
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)

    saveout(stats, outputdir, "stats", "matlab")
    saveout(comps, outputdir, "comps", "matlab")
    saveout(latent, outputdir, "latent", "matlab")
    saveout(scores, outputdir, "scores", "matlab", 2)
    saveout(traj, outputdir, "traj", "matlab")