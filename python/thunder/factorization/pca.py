"""
Class and standalone app for Principal Component Analysis
"""

import os
import argparse
import glob
import numpy as np
import matplotlib.pyplot as plt
import mpld3
from mpld3 import plugins
from thunder.io import load
from thunder.io import save
from thunder.factorization import SVD
from thunder.util.matrices import RowMatrix
from thunder.viz.plugins import LinkedView
from pyspark import SparkContext


class PCA(object):
    """Perform principal components analysis
    using the singular value decomposition

    Parameters
    ----------
    k : int
        Number of principal components to estimate

    svdmethod : str, optional, default = "direct"
        Which method to use for performing the SVD

    Attributes
    ----------
    `comps` : array, shape (k, ncols)
        The k principal components

    `latent` : array, shape (k,)
        The latent values

    `scores` : RDD of nrows (tuple, array) pairs, each of shape (k,)
        The scores (i.e. the representation of the data in PC space)
    """

    def __init__(self, k=3, svdmethod='direct'):
        self.k = k
        self.svdmethod = svdmethod

    def fit(self, data):
        """Estimate principal components

        Parameters
        ----------
        data : RDD of (tuple, array) pairs, or RowMatrix
        """

        if type(data) is not RowMatrix:
            data = RowMatrix(data)

        data.center(0)
        svd = SVD(k=self.k, method=self.svdmethod)
        svd.calc(data)

        self.scores = svd.u
        self.latent = svd.s
        self.comps = svd.v

        return self

    def plot(self):

        fig, ax = plt.subplots(2)

        # scatter periods and amplitudes
        pts = self.scores.map(lambda (k, v): v).collect()
        points = ax[1].scatter(map(lambda x: x[0], pts), map(lambda x: x[1], pts), s=100, alpha=0.5)

        # create the line object
        x = np.linspace(0, np.shape(self.comps)[1], 10)
        lines = ax[0].plot(x, 0 * x, '-w', lw=4, alpha=0.5)
        ax[0].set_ylim(-0.25, 0.25)
        ax[0].set_title("Hover over points to see principal components")

        result = self.scores.collect()
        linedata = map(lambda x: map(lambda x: list(x), zip(range(0, 10), (x[1][0] * self.comps[0, :] + x[1][1] * self.comps[1, :]).tolist())), result)
        plugins.connect(fig, LinkedView(points, lines[0], linedata))

        return mpld3.fig_to_html(fig)


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
    result = PCA(args.k, args.svdmethod).fit(data)

    outputdir = args.outputdir + "-pca"
    save(result.comps, outputdir, "comps", "matlab")
    save(result.latent, outputdir, "latent", "matlab")
    save(result.scores, outputdir, "scores", "matlab")