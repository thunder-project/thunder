"""
Example standalone app for independent component analysis
"""

import optparse
from thunder import ThunderContext, ICA, export


if __name__ == "__main__":
    parser = optparse.OptionParser(description="do independent components analysis",
                                   usage="%prog datafile outputdir k c [options]")
    parser.add_option("--svdmethod", choices=("direct", "em"), default="direct")
    parser.add_option("--maxiter", type=float, default=100)
    parser.add_option("--tol", type=float, default=0.000001)
    parser.add_option("--seed", type=int, default=0)

    opts, args = parser.parse_args()
    try:
        datafile = args[0]
        outputdir = args[1]
        k = int(args[2])
        c = int(args[3])
    except IndexError:
        parser.print_usage()
        raise Exception("too few arguments")

    tsc = ThunderContext.start(appName="ica")

    data = tsc.loadSeries(datafile).cache()
    model = ICA(k=k, c=c, svdmethod=opts.svdmethod, maxiter=opts.maxiter, tol=opts.tol, seed=opts.seed)
    result = model.fit(data)

    outputdir += "-ica"
    export(result.a, outputdir, "a", "matlab")
    export(result.sigs, outputdir, "sigs", "matlab")
