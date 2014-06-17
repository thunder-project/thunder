"""
Standalone app for calculating time series statistics
"""

import os
import argparse
import glob
from thunder.timeseries import Stats
from thunder.io import load
from thunder.io import save
from pyspark import SparkContext


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="compute summary statistics on time series data")
    parser.add_argument("master", type=str)
    parser.add_argument("datafile", type=str)
    parser.add_argument("outputdir", type=str)
    parser.add_argument("mode", choices=("mean", "median", "std", "norm"), help="which summary statistic")
    parser.add_argument("--preprocess", choices=("raw", "dff", "dff-highpass", "sub"), default="raw", required=False)

    args = parser.parse_args()

    sc = SparkContext(args.master, "stats")

    if args.master != "local":
        egg = glob.glob(os.path.join(os.environ['THUNDER_EGG'], "*.egg"))
        sc.addPyFile(egg[0])

    data = load(sc, args.datafile, args.preprocess).cache()
    vals = Stats(args.mode).calc(data)

    outputdir = args.outputdir + "-stats"
    save(vals, outputdir, "stats_" + args.mode, "matlab")