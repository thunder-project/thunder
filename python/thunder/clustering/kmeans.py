"""
Classes and standalone app for KMeans clustering
"""

import argparse
from numpy import sum, array, argmin, corrcoef, random, ndarray
from scipy.spatial.distance import cdist
from matplotlib import pyplot
import colorsys
import mpld3
from mpld3 import plugins
from pyspark import SparkContext
from thunder.utils import load
from thunder.utils import save
from thunder.viz.plugins import HiddenAxes
from thunder.viz.plots import scatter
from thunder.viz.colorize import Colorize
from thunder.utils.load import isrdd


class KMeansModel(object):
    """Class for an estimated KMeans model

    Parameters
    ----------
    centers : array
        Cluster centers

    Attributes
    ----------
    centers : array
        Cluster centers

    colors : array
        Unique color labels for each cluster
    """
    def __init__(self, centers):

        self.centers = centers
        n = len(self.centers)
        hsv_tuples = [(x*1.0/n, 0.5, 0.5) for x in range(n)]
        self.colors = map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples)

    def calc(self, data, func):
        """Base function for making clustering predictions"""

        # small optimization to avoid serializing full model
        centers = self.centers

        if isrdd(data):
            return data.mapValues(lambda x: func(centers, x))

        elif isinstance(data, list):
            return map(lambda x: func(centers, x), data)

        elif isinstance(data, ndarray):
            if data.ndims == 1:
                return func(centers, data)
            else:
                return map(lambda x: func(centers, x), data)

    def predict(self, data):
        """Predict the cluster that all data points belong to, and the similarity

        Parameters
        ----------
        data : RDD of (tuple, array) pairs, a list of arrays, or a single array
            The data to predict cluster assignments on

        Returns
        -------
        closest : RDD of (tuple, array) pairs, list of arrays, or a single array
            For each data point, ggives the closest center to that point
        """

        closestpoint = lambda centers, p: argmin(cdist(centers, array([p])))
        return self.calc(data, closestpoint)

    def similarity(self, data):
        """Estimate similarity between each data point and the cluster it belongs to

        Parameters
        ----------
        data : RDD of (tuple, array) pairs, a list of arrays, or a single array
            The data to estimate similarities on

        Returns
        -------
        similarities : RDD of (tuple, array) pairs, list of arrays, or a single array
            For each data point, gives the similarity to its nearest cluster
        """

        similarity = lambda centers, p: corrcoef(centers[argmin(cdist(centers, array([p])))], p)[0,1]
        return self.calc(data, similarity)

    def plot(self, data, notebook=False, show=True, savename=None):

        fig = pyplot.figure()
        ncenters = len(self.centers)

        colorizer = Colorize()
        colorizer.get = lambda x: self.colors[int(self.predict(x)[0])]

        # plot time series of each center
        # TODO move into a time series plotting function in viz.plots
        for i, center in enumerate(self.centers):
            ax = pyplot.subplot2grid((ncenters, 3), (i, 0))
            ax.plot(center, color=self.colors[i], linewidth=5)
            fig.add_axes(ax)

        # make a scatter plot of the data
        ax2 = pyplot.subplot2grid((ncenters, 3), (0, 1), rowspan=ncenters, colspan=2)
        ax2, h2 = scatter(data, colormap=colorizer, ax=ax2)
        fig.add_axes(ax2)

        plugins.connect(fig, HiddenAxes())

        if show and notebook is False:
            mpld3.show()

        if savename is not None:
            mpld3.save_html(fig, savename)

        elif show is False:
            return mpld3.fig_to_html(fig)


class KMeans(object):
    """Class for KMeans clustering

    Parameters
    ----------
    k : int
        Number of clusters to find

    maxiter : int, optional, default = 20
        Maximum number of iterations to use

    tol : float, optional, default = 0.001
        Change tolerance for stopping algorithm
    """
    def __init__(self, k, maxiter=20, tol=0.001, init="random"):
        self.k = k
        self.maxiter = maxiter
        self.tol = tol
        self.init = init
        if not (init == "random" or init == "sample"):
            raise Exception("init must be random or sample")

    def train(self, data):
        """Train the clustering model using the standard
        k-means algorithm

        Parameters
        ----------
        data :  RDD of (tuple, array) pairs
            The data to cluster

        Returns
        -------
        centers : KMeansModel
            The estimated cluster centers
        """

        if self.init == "sample":
            samples = data.takeSample(False, max(self.k, 1000))[0:self.k]
            centers = array(map(lambda (_, v): v, samples))

        if self.init == "random":
            d = len(data.first()[1])
            centers = random.randn(self.k, d)

        tempdist = 1.0
        iter = 0

        while (tempdist > self.tol) & (iter < self.maxiter):

            closest = data.map(lambda (_, v): v).map(lambda p: (argmin(cdist(centers, array([p]))), (p, 1)))
            pointstats = closest.reduceByKey(lambda (x1, y1), (x2, y2): (x1 + x2, y1 + y2))
            newpoints = pointstats.map(lambda (x, (y, z)): (x, y / z)).collect()
            tempdist = sum(sum((centers[x] - y) ** 2) for (x, y) in newpoints)

            for (i, j) in newpoints:
                centers[i] = j

            iter += 1

        return KMeansModel(centers)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="do kmeans clustering")
    parser.add_argument("datafile", type=str)
    parser.add_argument("outputdir", type=str)
    parser.add_argument("k", type=int)
    parser.add_argument("--maxiter", type=float, default=20, required=False)
    parser.add_argument("--tol", type=float, default=0.001, required=False)
    parser.add_argument("--preprocess", choices=("raw", "dff", "dff-highpass", "sub"), default="raw", required=False)

    args = parser.parse_args()

    sc = SparkContext(appName="kmeans")

    data = load(sc, args.datafile, args.preprocess).cache()
    model = KMeans(k=args.k, maxiter=args.maxiter, tol=args.tol).train(data)
    labels = model.predict(data)

    outputdir = args.outputdir + "-kmeans"
    save(model.centers, outputdir, "centers", "matlab")
    save(labels, outputdir, "labels", "matlab")
