"""
Example standalone app for mass-univariate regression
"""

import optparse
from thunder import ThunderContext, RegressionModel


if __name__ == "__main__":
    parser = optparse.OptionParser(description="fit a regression model",
                                   usage="%prog datafile modelfile outputdir [options]")
    parser.add_option("--regressmode", choices=("mean", "linear", "bilinear"),
                      default="linear", help="form of regression")

    opts, args = parser.parse_args()
    try:
        datafile = args[0]
        modelfile = args[1]
        outputdir = args[2]
    except IndexError:
        parser.print_usage()
        raise Exception("too few arguments")

    tsc = ThunderContext.start(appName="regress")

    data = tsc.loadText(datafile)
    result = RegressionModel.load(modelfile, opts.regressmode).fit(data)

    outputdir += "-regress"
    tsc.export(result.select('stats'), outputdir, "stats", "matlab")
    tsc.export(result.select('betas'), outputdir, "betas", "matlab")