import os
import argparse
import glob
from numpy import zeros
from thunder.sigprocessing.util import SigProcessingMethod
from thunder.util.parse import parse
from thunder.util.saveout import saveout
from pyspark import SparkContext


def query(data, indsfile):
    """query data by averaging together
    data points with the given indices

    arguments:
    data - RDD of data points (pairs of (int,array))
    sigfile - indsfile (string with file location or array)
    lag - maximum lag (result will be 2*lag + 1)

    returns:
    betas - cross-correlations at different time lags
    scores, latent, comps - result of applying pca if lag > 0
    """
    # load indices
    method = SigProcessingMethod.load("query", indsfile=indsfile)

    # loop over indices, averaging time series
    ts = zeros((method.n, len(data.first()[1])))
    for i in range(0, method.n):
        ts[i, :] = data.filter(lambda (k, x): k in method.inds[i]).map(
            lambda (k, x): x).mean()

    return ts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="query time series data by averaging values for given indices")
    parser.add_argument("master", type=str)
    parser.add_argument("datafile", type=str)
    parser.add_argument("indsfile", type=str)
    parser.add_argument("outputdir", type=str)
    parser.add_argument("--preprocess", choices=("raw", "dff", "sub"), default="raw", required=False)

    args = parser.parse_args()
    egg = glob.glob(os.environ['THUNDER_EGG'] + "*.egg")
    sc = SparkContext(args.master, "query", pyFiles=egg)

    lines = sc.textFile(args.datafile)
    data = parse(lines, args.preprocess, nkeys=3, keepkeys="linear").cache()

    ts = query(data, args.indsfile)

    outputdir = args.outputdir + "-query"
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)

    saveout(ts, outputdir, "ts", "matlab")
