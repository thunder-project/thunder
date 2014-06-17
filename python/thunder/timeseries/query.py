"""
Class and standalone app for querying
"""

import os
import argparse
import glob
from numpy import zeros
from scipy.io import loadmat
from thunder.io import load, getdims, subtoind
from thunder.io import save
from pyspark import SparkContext


class Query(object):
    """Class for querying time series data by averaging together
    data points with the specified indices

    Parameters
    ----------
    indsfile : str, or array-like (2D)
        Array of indices, each an array-like of integer indicies
    """

    def __init__(self, indsfile):
        if type(indsfile) is str:
            inds = loadmat(indsfile)['inds'][0]
        else:
            inds = indsfile
        self.inds = inds
        self.n = len(inds)

    def calc(self, data):
        """Calculate averages. Keys (tuples) are converted
        into linear indices based on their dimensions

        Parameters
        ----------
        data : RDD of (tuple, array) pairs, each array of shape (ncols,)
            Data to compute averages from

        Returns
        -------
        ts : array, shape (n, ncols)
        """

        dims = getdims(data)
        data = subtoind(data, dims.max)

        # loop over indices, averaging time series
        ts = zeros((self.n, len(data.first()[1])))
        for i in range(0, self.n):
            indsb = data.context.broadcast(self.inds[i])
            ts[i, :] = data.filter(lambda (k, _): k in indsb.value).map(
                lambda (k, x): x).sum() / len(self.inds[i])

        return ts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="query time series data by averaging values for given indices")
    parser.add_argument("master", type=str)
    parser.add_argument("datafile", type=str)
    parser.add_argument("indsfile", type=str)
    parser.add_argument("outputdir", type=str)
    parser.add_argument("--preprocess", choices=("raw", "dff", "dff-highpass", "sub"), default="raw", required=False)

    args = parser.parse_args()

    sc = SparkContext(args.master, "query")

    if args.master != "local":
        egg = glob.glob(os.path.join(os.environ['THUNDER_EGG'], "*.egg"))
        sc.addPyFile(egg[0])

    data = load(sc, args.datafile, args.preprocess).cache()
    ts = Query(args.indsfile).calc(data)

    outputdir = args.outputdir + "-query"
    save(ts, outputdir, "ts", "matlab")
