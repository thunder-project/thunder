"""
Classes and standalone app for KMeans clustering
"""

import os
import argparse
import glob
from numpy import sum, array, argmin, corrcoef
from scipy.spatial.distance import cdist
from thunder.io import load
from thunder.io import save
from pyspark import SparkContext
from thunder.io.load import isrdd


class KMeansModel(object):
    """Class for an estimated KMeans model

    Parameters
    ----------
    centers : array
        The cluster centers
    """
    def __init__(self, centers):
        self.centers = centers

    def predict(self, data):
        """Predict the cluster that all data points belong to, and the similarity

        Parameters
        ----------
        data : RDD of (tuple, array) pairs
            The data

        Returns
        -------
        closest : RDD of (tuple, array) pairs
            For each data point, gives an array with the closest center for each data point,
            and the correlation with that center
        """
        if isrdd(data):
            return data.mapValues(lambda x: KMeans.similarity(x, self.centers))
        else:
            return map(lambda x: KMeans.similarity(x, self.centers), data)



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
    def __init__(self, k, maxiter=20, tol=0.001):
        self.k = k
        self.maxiter = maxiter
        self.tol = tol

    @staticmethod
    def closestpoint(p, centers):
        """Return the index of the closest point in centers to p"""
        return argmin(cdist(centers, array([p])))

    @staticmethod
    def similarity(p, centers):
        ind = argmin(cdist(centers, array([p])))
        corr = corrcoef(centers[ind], p)[0,1]
        return array([ind, corr])

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

        centers = array(map(lambda (_, v): v, data.takeSample(False, self.k)))
        tempdist = 1.0
        iter = 0

        while (tempdist > self.tol) & (iter < self.maxiter):
            closest = data.map(lambda (_, v): v).map(lambda p: (KMeans.closestpoint(p, centers), (p, 1)))
            pointstats = closest.reduceByKey(lambda (x1, y1), (x2, y2): (x1 + x2, y1 + y2))
            newpoints = pointstats.map(lambda (x, (y, z)): (x, y / z)).collect()
            tempdist = sum(sum((centers[x] - y) ** 2) for (x, y) in newpoints)

            for (i, j) in newpoints:
                centers[i] = j

            iter += 1

        return KMeansModel(centers)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="do kmeans clustering")
    parser.add_argument("master", type=str)
    parser.add_argument("datafile", type=str)
    parser.add_argument("outputdir", type=str)
    parser.add_argument("k", type=int)
    parser.add_argument("--maxiter", type=float, default=20, required=False)
    parser.add_argument("--tol", type=float, default=0.001, required=False)
    parser.add_argument("--preprocess", choices=("raw", "dff", "dff-highpass", "sub"), default="raw", required=False)

    args = parser.parse_args()

    sc = SparkContext(args.master, "kmeans")

    if args.master != "local":
        egg = glob.glob(os.path.join(os.environ['THUNDER_EGG'], "*.egg"))
        sc.addPyFile(egg[0])

    data = load(sc, args.datafile, args.preprocess).cache()
    model = KMeans(k=args.k, maxiter=args.maxiter, tol=args.tol).train(data)
    labels = model.predict(data)

    outputdir = args.outputdir + "-kmeans"
    save(model.centers, outputdir, "centers", "matlab")
    save(labels, outputdir, "labels", "matlab")
