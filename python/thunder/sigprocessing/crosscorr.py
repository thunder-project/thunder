import os
import argparse
import glob
from thunder.sigprocessing.util import SigProcessingMethod
from thunder.factorization.util import svd
from thunder.util.load import load
from thunder.util.save import save
from pyspark import SparkContext


def crosscorr(data, sigfile, lag):
    """Cross-correlate data points
    (typically time series data)
    against a signal over the specified lags

    :param data: RDD of data points as key value pairs
    :param sigfile: signal to correlate with (string with file location or array)
    :param lag: maximum lag (result will be length 2*lag + 1)

    :return betas: cross-correlations at different time lags
    :return scores: scores from PCA (if lag > 0)
    :return latent: scores from PCA (if lag > 0)
    :return comps: components from PCA (if lag > 0)
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
    parser.add_argument("--preprocess", choices=("raw", "dff", "dff-highpass", "sub"), default="raw", required=False)

    args = parser.parse_args()
    egg = glob.glob(os.path.join(os.environ['THUNDER_EGG'], "*.egg"))
    sc = SparkContext(args.master, "crosscorr", pyFiles=egg)
    data = load(sc, args.datafile, args.preprocess).cache()

    outputdir = args.outputdir + "-crosscorr"
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)

    # post-process data with pca if lag greater than 0
    if args.lag is not 0:
        betas, scores, latent, comps = crosscorr(data, args.sigfile, args.lag)
        save(comps, outputdir, "comps", "matlab")
        save(latent, outputdir, "latent", "matlab")
        save(scores, outputdir, "scores", "matlab")
    else:
        betas = crosscorr(data, args.sigfile, args.lag)
        save(betas, outputdir, "stats", "matlab")
