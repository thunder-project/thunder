"""
Example standalone app for calculating series statistics
"""

import optparse
from thunder import ThunderContext


if __name__ == "__main__":
    parser = optparse.OptionParser(description="compute summary statistics on time series data",
                                   usage="%prog datafile outputdir mode [options]")
    parser.add_option("--preprocess", action="store_true", default=False)

    opts, args = parser.parse_args()
    try:
        datafile = args[0]
        outputdir = args[1]
        mode = args[2]
    except IndexError:
        parser.print_usage()
        raise Exception("too few arguments")

    tsc = ThunderContext.start(appName="stats")

    data = tsc.loadSeries(datafile).cache()
    vals = data.seriesStat(mode)

    outputdir += "-stats"
    tsc.export(vals, outputdir, "stats_" + mode, "matlab")