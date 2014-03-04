import os
import argparse
import glob
from thunder.util.load import load
from thunder.util.save import save
from thunder.factorization.util import svd
from pyspark import SparkContext


def pca(data, k, svdmethod="direct"):
    """Perform principal components analysis
    using the singular value decomposition

    :param data: RDD of data points as key value pairs
    :param k: number of principal components to recover
    :param svdmethod: which svd algorithm to use (default = "direct")

    :return comps: the k principal components (as array)
    :return latent: the latent values
    :return scores: the k scores (as RDD)
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
    parser.add_argument("--preprocess", choices=("raw", "dff", "dff-highpass", "sub"), default="raw", required=False)

    args = parser.parse_args()
    egg = glob.glob(os.path.join(os.environ['THUNDER_EGG'], "*.egg"))
    sc = SparkContext(args.master, "pca", pyFiles=egg)
    data = load(sc, args.datafile, args.preprocess).cache()

    scores, latent, comps = pca(data, args.k, args.svdmethod)

    outputdir = args.outputdir + "-pca"
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)

    save(comps, outputdir, "comps", "matlab")
    save(latent, outputdir, "latent", "matlab")
    save(scores, outputdir, "scores", "matlab")