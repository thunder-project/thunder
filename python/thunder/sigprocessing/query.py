import os
import argparse
import glob
from numpy import zeros
from thunder.sigprocessing.util import SigProcessingMethod
from thunder.util.load import load, subtoind, getdims
from thunder.util.save import save
from pyspark import SparkContext


def query(data, indsfile):
    """Query data by averaging together
    data points with the given indices

    :param data: RDD of data points as key value pairs
    :param indsfile: string with file location or array

    :return ts: array with averages
    """
    # load indices
    method = SigProcessingMethod.load("query", indsfile=indsfile)

    # convert to linear indexing
    dims = getdims(data)
    data = subtoind(data, dims)

    # loop over indices, averaging time series
    ts = zeros((method.n, len(data.first()[1])))
    for i in range(0, method.n):
        ts[i, :] = data.filter(lambda (k, _): k in method.inds[i]).map(
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

    data = load(sc, args.datafile, args.preprocess).cache()

    ts = query(data, args.indsfile)

    outputdir = args.outputdir + "-query"
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)

    save(ts, outputdir, "ts", "matlab")
