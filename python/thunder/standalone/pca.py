"""
Example standalone app for principal component analysis
"""

import argparse
from thunder import ThunderContext, PCA, export


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="do principal components analysis")
    parser.add_argument("datafile", type=str)
    parser.add_argument("outputdir", type=str)
    parser.add_argument("k", type=int)
    parser.add_argument("--svdmethod", choices=("direct", "em"), default="direct", required=False)

    args = parser.parse_args()

    tsc = ThunderContext.start(appName="pca")

    data = tsc.loadSeries(args.datafile).cache()

    model = PCA(args.k, args.svdmethod)
    model.fit(data)

    outputdir = args.outputdir + "-pca"
    export(model.comps, outputdir, "comps", "matlab")
    export(model.latent, outputdir, "latent", "matlab")
    export(model.scores, outputdir, "scores", "matlab")