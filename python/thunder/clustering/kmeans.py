# perform kmeans clustering
#
# example:
# pyspark kmeans.py local data/iris.txt results 2 euclidean

import os
import argparse
from numpy import dot, mean
from numpy.linalg import norm
from thunder.util.dataio import saveout, parse
from pyspark import SparkContext


def closestPoint(p, centers, dist):
    bestIndex = 0
    closest = float("+inf")
    for i in range(len(centers)):
        if dist == 'euclidean':
            tempDist = sum((p - centers[i]) ** 2)
        if dist == 'corr':
            tempDist = max(1 - dot(p, centers[i]), 0)
        if tempDist < closest:
            closest = tempDist
            bestIndex = i
    return bestIndex, closest


def kmeans(data, k, dist):

    if dist == 'corr':
        data = data.map(lambda x: (x - mean(x)) / norm(x)).cache()

    centers = data.take(k)

    convergeDist = 0.001
    tempDist = 1.0
    iteration = 0
    mxIteration = 100

    while (tempDist > convergeDist) & (iteration < mxIteration):
        if dist == 'corr':
            centers = map(lambda x: x / norm(x), centers)
        closest = data.map(lambda p: (closestPoint(p, centers, dist)[0], (p, 1)))
        pointStats = closest.reduceByKey(lambda (x1, y1), (x2, y2): (x1 + x2, y1 + y2))
        newPoints = pointStats.map(lambda (x, (y, z)): (x, y / z)).collect()
        tempDist = sum(sum((centers[x] - y) ** 2) for (x, y) in newPoints)

        for (i, j) in newPoints:
            centers[i] = j

        iteration += 1

    if dist == 'corr':
        centers = map(lambda x: x / norm(x), centers)

    labels = data.map(lambda p: closestPoint(p, centers, dist)[0])
    dists = data.map(lambda p: closestPoint(p, centers, dist)[1])
    normDists = data.map(lambda p: closestPoint((p - mean(p))/norm(p), centers, 'corr')[1])

    return labels, centers, dists, normDists

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="do kmeans clustering")
    parser.add_argument("master", type=str)
    parser.add_argument("dataFile", type=str)
    parser.add_argument("dataMode", choices=("raw", "dff", "sub"), help="form of data preprocessing")
    parser.add_argument("outputDir", type=str)
    parser.add_argument("k", type=int)
    parser.add_argument("dist", choices=("euclidean", "correlation"),
                        help="distance metric for kmeans")

    args = parser.parse_args()
    egg = glob.glob(os.environ['THUNDER_EGG'] + "*.egg")
    sc = SparkContext(args.master, "kmeans", pyFiles=egg)

    lines = sc.textFile(args.dataFile)
    data = parse(lines, args.dataMode).cache()

    labels, centers, dists, normDists = kmeans(data, args.k, args.dist)

    outputDir = args.outputDir + "-kmeans"
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)
    saveout(labels, outputDir, "labels", "matlab")
    saveout(dists, outputDir, "dists", "matlab")
    saveout(centers, outputDir, "centers", "matlab")
    saveout(normDists, outputDir, "normDists", "matlab")