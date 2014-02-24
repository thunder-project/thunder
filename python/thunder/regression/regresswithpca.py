import os
import argparse
import glob
from thunder.regression.util import RegressionModel
from thunder.factorization.util import svd
from thunder.util.load import load
from thunder.util.save import save
from pyspark import SparkContext


def regresswithpca(data, modelfile, regressmode):
    """Perform univariate regression,
    followed by principal components analysis
    to reduce dimensionality

    :param data: RDD of data points as key value pairs
    :param modelfile: model parameters (string with file location, array, or tuple)
    :param regressmode: form of regression ("linear" or "bilinear")

    :return stats: statistics of the fit
    :return comps: compoents from PCA
    :return scores: scores from PCA
    :return latent: latent variances from PCA
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
    egg = glob.glob(os.path.join(os.environ['THUNDER_EGG'], "*.egg"))
    sc = SparkContext(args.master, "regress", pyFiles=egg)
    data = load(sc, args.datafile, args.preprocess).cache()

    stats, comps, latent, scores, traj = regress(data, args.modelfile, args.regressmode)

    outputdir = args.outputdir + "-regress"
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)

    save(stats, outputdir, "stats", "matlab")
    save(comps, outputdir, "comps", "matlab")
    save(latent, outputdir, "latent", "matlab")
    save(scores, outputdir, "scores", "matlab")
    save(traj, outputdir, "traj", "matlab")