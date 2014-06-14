import os
import argparse
import glob
from thunder.util.load import load
from thunder.util.save import save
from thunder.factorization import SVD
from thunder.util.matrices import RowMatrix
from pyspark import SparkContext


class PCA(object):
    """Perform principal components analysis
    using the singular value decomposition

    :param data: RDD of data points as key value pairs
    :param k: number of principal components to recover
    :param svdmethod: which svd algorithm to use (default = "direct")

    :return comps: the k principal components (as array)
    :return latent: the latent values
    :return scores: the k scores (as RDD)
    """

    def __init__(self, k=3, svdmethod='direct'):
        self.k = k
        self.svdmethod = svdmethod

    def fit(self, data):

        if type(data) is not RowMatrix:
            data = RowMatrix(data)

        data.center(0)
        svd = SVD(k=self.k, method=self.svdmethod)
        svd.calc(data)

        self.scores = svd.u
        self.latent = svd.s
        self.comps = svd.v


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="do principal components analysis")
    parser.add_argument("master", type=str)
    parser.add_argument("datafile", type=str)
    parser.add_argument("outputdir", type=str)
    parser.add_argument("k", type=int)
    parser.add_argument("--svdmethod", choices=("direct", "em"), default="direct", required=False)
    parser.add_argument("--preprocess", choices=("raw", "dff", "dff-highpass", "sub"), default="raw", required=False)

    args = parser.parse_args()

    sc = SparkContext(args.master, "pca")

    if args.master != "local":
        egg = glob.glob(os.path.join(os.environ['THUNDER_EGG'], "*.egg"))
        sc.addPyFile(egg[0])

    data = load(sc, args.datafile, args.preprocess).cache()

    pca = PCA(k=args.k, svdmethod=args.svdmethod)
    pca.fit(data)

    outputdir = args.outputdir + "-pca"

    save(pca.comps, outputdir, "comps", "matlab")
    save(pca.latent, outputdir, "latent", "matlab")
    save(pca.scores, outputdir, "scores", "matlab")