import os
import argparse
import glob
from thunder.regression.util import RegressionModel
from thunder.factorization import PCA
from thunder.util.load import load
from thunder.util.save import save
from pyspark import SparkContext


def regresswithpca(data, modelfile, regressmode, k=2):
    """Perform univariate regression,
    followed by principal components analysis
    to reduce dimensionality

    :param data: RDD of data points as key value pairs
    :param modelfile: model parameters (string with file location, array, or tuple)
    :param regressmode: form of regression ("linear" or "bilinear")
    :param k: number of principal components to compute

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
    pca = PCA(k=k)
    pca.fit(betas)

    # compute trajectories from raw data
    traj = model.fit(data, pca.comps)

    return stats, pca.comps, pca.latent, pca.scores, traj


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="fit a regression model")
    parser.add_argument("master", type=str)
    parser.add_argument("datafile", type=str)
    parser.add_argument("modelfile", type=str)
    parser.add_argument("outputdir", type=str)
    parser.add_argument("regressmode", choices=("mean", "linear", "bilinear"), help="form of regression")
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--preprocess", choices=("raw", "dff", "dff-highpass", "sub"), default="raw", required=False)

    args = parser.parse_args()
    
    sc = SparkContext(args.master, "regresswithpca")

    if args.master != "local":
        egg = glob.glob(os.path.join(os.environ['THUNDER_EGG'], "*.egg"))
        sc.addPyFile(egg[0])

    data = load(sc, args.datafile, args.preprocess)

    stats, comps, latent, scores, traj = regresswithpca(data, args.modelfile, args.regressmode, args.k)

    outputdir = args.outputdir + "-regress"

    save(stats, outputdir, "stats", "matlab")
    save(comps, outputdir, "comps", "matlab")
    save(latent, outputdir, "latent", "matlab")
    save(scores, outputdir, "scores", "matlab")
    save(traj, outputdir, "traj", "matlab")