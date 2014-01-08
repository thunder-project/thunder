import os
import argparse
import glob
from thunder.util.dataio import parse, saveout
from thunder.factorization.util import svd
from pyspark import SparkContext


def pca(data, k, svdmethod="direct"):
    """perform principal components analysis
    using the svd

    arguments:
    data - RDD of data points
    k - number of principal components to recover
    method - which svd algorithm to use (default = "direct")

    returns:
    comps - the k principal components (as array)
    latent - the latent values
    scores - the k scores (as RDD)
    """
    scores, latent, comps = svd(data, k, meansubtract=0, method=svdmethod)
    return scores, latent, comps

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="do principal components analysis")
    parser.add_argument("master", type=str)
    parser.add_argument("datafile", type=str)
    parser.add_argument("outputdir", type=str)
    parser.add_argument("k", type=int)
    parser.add_argument("--svdmethod", choices=("direct", "em"), default="direct", required=False)
    parser.add_argument("--preprocess", choices=("raw", "dff", "sub"), default="raw", required=False)

    args = parser.parse_args()
    egg = glob.glob(os.environ['THUNDER_EGG'] + "*.egg")
    sc = SparkContext(args.master, "pca", pyFiles=egg)
    lines = sc.textFile(args.datafile)
    data = parse(lines, args.preprocess).cache()

    scores, latent, comps = pca(data, args.k, args.svdmethod)

    outputdir = args.outputdir + "-pca"
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)

    saveout(comps, outputdir, "comps", "matlab")
    saveout(latent, outputdir, "latent", "matlab")
    saveout(scores, outputdir, "scores", "matlab", args.k)