# performs principal components analysis
#
# example:
# pca.py local data/iris.txt raw results 2


import os
import argparse
from thunder.util.dataio import parse, saveout
from thunder.factorization.util import svd1, svd3, svd4
from pyspark import SparkContext


def pca(data, k):
    comps, latent, scores = svd4(data, k, 0)
    return comps, latent, scores

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="do principal components analysis")
    parser.add_argument("master", type=str)
    parser.add_argument("dataFile", type=str)
    parser.add_argument("dataMode", choices=("raw", "dff", "sub"), help="form of data preprocessing")
    parser.add_argument("outputDir", type=str)
    parser.add_argument("k", type=int)

    args = parser.parse_args()
    sc = SparkContext(args.master, "pca")
    lines = sc.textFile(args.dataFile)
    data = parse(lines, args.dataMode).cache()

    comps, latent, scores = pca(data, args.k)

    outputDir = args.outputDir + "-pca"
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)

    saveout(comps, outputDir, "comps", "matlab")
    saveout(latent, outputDir, "latent", "matlab")
    saveout(scores, outputDir, "scores", "matlab", args.k)