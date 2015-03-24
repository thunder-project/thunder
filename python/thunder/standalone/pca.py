"""
Example standalone app for principal component analysis
"""

import optparse
from thunder import ThunderContext, PCA


if __name__ == "__main__":
    parser = optparse.OptionParser(description="do principal components analysis",
                                   usage="%prog datafile outputdir k [options]")
    parser.add_option("--svdmethod", choices=("direct", "em"), default="direct")

    opts, args = parser.parse_args()
    try:
        datafile = args[0]
        outputdir = args[1]
        k = int(args[2])
    except IndexError:
        parser.print_usage()
        raise Exception("too few arguments")

    tsc = ThunderContext.start(appName="pca")

    data = tsc.loadSeries(datafile).cache()

    model = PCA(k, opts.svdmethod)
    model.fit(data)

    outputdir += "-pca"
    tsc.export(model.comps, outputdir, "comps", "matlab")
    tsc.export(model.latent, outputdir, "latent", "matlab")
    tsc.export(model.scores, outputdir, "scores", "matlab")