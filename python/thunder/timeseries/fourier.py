"""
Standalone app for Fourier analysis
"""

import os
import argparse
import glob
from pyspark import SparkContext
from thunder.timeseries import Fourier
from thunder.utils import load
from thunder.utils import save


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="compute a fourier transform on each time series")
    parser.add_argument("datafile", type=str)
    parser.add_argument("outputdir", type=str)
    parser.add_argument("freq", type=int)
    parser.add_argument("--preprocess", choices=("raw", "dff", "sub", "dff-highpass", "dff-percentile"
                        "dff-detrendnonlin", "dff-detrend-percentile"), default="raw", required=False)

    args = parser.parse_args()

    sc = SparkContext(appName="fourier")

    data = load(sc, args.datafile, args.preprocess).cache()
    out = Fourier(freq=args.freq).calc(data)

    outputdir = args.outputdir + "-fourier"
    save(out, outputdir, "fourier", "matlab")
