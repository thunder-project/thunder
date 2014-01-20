import os
import argparse
import glob
from thunder.sigprocessing.util import SigProcessingMethod
from thunder.factorization.util import svd
from thunder.util.parse import parse
from thunder.util.saveout import saveout
from pyspark import SparkContext


def crosscorr(data, sigfile, lag):
    """cross-correlate data points
    (typically time series data)
    against a signal over the specified lags

    arguments:
    data - RDD of data points
    sigfile - signal to correlate with (string with file location or array)
    lag - maximum lag (result will be 2*lag + 1)

    returns:
    betas - cross-correlations at different time lags
    scores, latent, comps - result of applying pca if lag > 0
    """

    # compute cross correlations
    method = SigProcessingMethod.load("crosscorr", sigfile=sigfile, lag=lag)
    betas = method.calc(data)

    if lag is not 0:
        # do PCA
        scores, latent, comps = svd(betas, 2)
        return betas, scores, latent, comps
    else:
        return betas


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="fit a regression model")
    parser.add_argument("master", type=str)
    parser.add_argument("datafile", type=str)
    parser.add_argument("sigfile", type=str)
    parser.add_argument("outputdir", type=str)
    parser.add_argument("lag", type=int)
    parser.add_argument("--preprocess", choices=("raw", "dff", "sub"), default="raw", required=False)

    args = parser.parse_args()
    egg = glob.glob(os.environ['THUNDER_EGG'] + "*.egg")
    sc = SparkContext(args.master, "crosscorr", pyFiles=egg)
    lines = sc.textFile(args.datafile)
    data = parse(lines, args.preprocess).cache()

    outputdir = args.outputdir + "-crosscorr"
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)

    # post-process data with pca if lag greater than 0
    if args.lag is not 0:
        betas, scores, latent, comps = crosscorr(data, args.sigfile, args.lag)
        saveout(comps, outputdir, "comps", "matlab")
        saveout(latent, outputdir, "latent", "matlab")
        saveout(scores, outputdir, "scores", "matlab", nout=2)
    else:
        betas = crosscorr(data, args.sigfile, args.lag)
        saveout(betas, outputdir, "stats", "matlab")
