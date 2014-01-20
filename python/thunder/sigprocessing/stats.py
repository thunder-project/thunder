import os
import argparse
import glob
from thunder.sigprocessing.util import SigProcessingMethod
from thunder.util.parse import parse
from thunder.util.saveout import saveout
from pyspark import SparkContext


def stats(data, statistic):
    """compute summary statistics on every data point

    arguments:
    data - RDD of data points
    mode - which statistic to compute ("median", "mean", "std", "norm")

    returns:
    vals - RDD of statistics
    """

    method = SigProcessingMethod.load("stats", statistic=statistic)
    vals = method.calc(data)

    return vals

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="compute summary statistics on time series data")
    parser.add_argument("master", type=str)
    parser.add_argument("datafile", type=str)
    parser.add_argument("outputdir", type=str)
    parser.add_argument("mode", choices=("mean", "median", "std", "norm"),
                        help="which summary statistic")
    parser.add_argument("--preprocess", choices=("raw", "dff", "sub"), default="raw", required=False)

    args = parser.parse_args()
    egg = glob.glob(os.environ['THUNDER_EGG'] + "*.egg")
    sc = SparkContext(args.master, "ref", pyFiles=egg)

    lines = sc.textFile(args.datafile)
    data = parse(lines, args.preprocess).cache()

    vals = stats(data, args.mode)

    outputdir = args.outputdir + "-stats",

    outputdir = args.outputdir + "-stats"
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)

    saveout(vals, outputdir, "stats_" + args.mode, "matlab")