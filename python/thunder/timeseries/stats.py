"""
Class and standalone app for calculating series statistics
"""

import argparse
from thunder.utils.context import ThunderContext
from thunder.utils import save


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="compute summary statistics on time series data")
    parser.add_argument("datafile", type=str)
    parser.add_argument("outputdir", type=str)
    parser.add_argument("mode", type=str)
    parser.add_argument("--preprocess", choices=("raw", "dff", "sub", "dff-highpass", "dff-percentile"
                        "dff-detrendnonlin", "dff-detrend-percentile"), default="raw", required=False)

    args = parser.parse_args()

    tsc = ThunderContext.start(appName="stats")

    data = tsc.loadText(args.datafile, args.preprocess).cache()
    vals = data.seriesStat(args.mode)

    outputdir = args.outputdir + "-stats"
    save(vals, outputdir, "stats_" + args.mode, "matlab")