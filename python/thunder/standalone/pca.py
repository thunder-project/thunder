"""
Example standalone app for principal component analysis
"""

import argparse
from thunder.factorization import PCA
from thunder.utils.context import ThunderContext
from thunder.utils.save import save


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="do principal components analysis")
    parser.add_argument("datafile", type=str)
    parser.add_argument("outputdir", type=str)
    parser.add_argument("k", type=int)
    parser.add_argument("--svdmethod", choices=("direct", "em"), default="direct", required=False)
    parser.add_argument("--preprocess", choices=("raw", "dff", "dff-highpass", "sub"), default="raw", required=False)

    args = parser.parse_args()

    tsc = ThunderContext.start(appName="pca")

    data = tsc.loadText(args.datafile, args.preprocess).cache()
    result = PCA(args.k, args.svdmethod).fit(data)

    outputdir = args.outputdir + "-pca"
    save(result.comps, outputdir, "comps", "matlab")
    save(result.latent, outputdir, "latent", "matlab")
    save(result.scores, outputdir, "scores", "matlab")