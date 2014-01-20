import os
import argparse
import glob
from numpy import sum
from thunder.util.parse import parse
from thunder.util.saveout import saveout
from pyspark import SparkContext


def closestpoint(p, centers):
    """return the index of the closest point in centers to p"""

    bestindex = 0
    closest = float("+inf")
    for i in range(len(centers)):
        tempdist = sum((p - centers[i]) ** 2)
        if tempdist < closest:
            closest = tempdist
            bestindex = i
    return bestindex


def kmeans(data, k, maxiter=20, tol=0.001):
    """perform kmeans clustering

    arguments:
    data - RDD of data points
    k - number of clusters
    maxiter - maximum number of iterations (default = 20)
    tol - change tolerance for stopping algorithm (default = 0.001)

    returns:
    labels - RDD with labels for each data point
    centers - array of cluster centroids
    """
    centers = data.take(k)

    tempdist = 1.0
    iter = 0

    while (tempdist > tol) & (iter < maxiter):
        closest = data.map(lambda p: (closestpoint(p, centers), (p, 1)))
        pointstats = closest.reduceByKey(lambda (x1, y1), (x2, y2): (x1 + x2, y1 + y2))
        newpoints = pointstats.map(lambda (x, (y, z)): (x, y / z)).collect()
        tempdist = sum(sum((centers[x] - y) ** 2) for (x, y) in newpoints)

        for (i, j) in newpoints:
            centers[i] = j

        iter += 1

    labels = data.map(lambda p: closestpoint(p, centers))

    return labels, centers

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="do kmeans clustering")
    parser.add_argument("master", type=str)
    parser.add_argument("datafile", type=str)
    parser.add_argument("outputdir", type=str)
    parser.add_argument("k", type=int)
    parser.add_argument("--maxiter", type=float, default=20, required=False)
    parser.add_argument("--tol", type=float, default=0.001, required=False)
    parser.add_argument("--preprocess", choices=("raw", "dff", "sub"), default="raw", required=False)

    args = parser.parse_args()
    egg = glob.glob(os.environ['THUNDER_EGG'] + "*.egg")
    sc = SparkContext(args.master, "kmeans", pyFiles=egg)

    lines = sc.textFile(args.datafile)
    data = parse(lines, args.preprocess).cache()

    labels, centers = kmeans(data, k=args.k, maxiter=args.maxiter, tol=args.tol)

    outputdir = args.outputdir + "-kmeans"
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)
    saveout(labels, outputdir, "labels", "matlab")
    saveout(centers, outputdir, "centers", "matlab")