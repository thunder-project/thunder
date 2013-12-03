# kmeans <master> <inputFile> <outputFile> <k> <dist>
#
# perform kmeans, based on example included with Spark
#
# k - number of clusters to find
# dist - distance function (euclidean or correlation)
#
# example:
#
# pyspark kmeans.py local data/iris.txt results 2 euclidean
#

import sys
import os
from numpy import dot, mean
from numpy.linalg import norm
from thunder.util.dataio import saveout, parse
from pyspark import SparkContext

argsIn = sys.argv[1:]
if len(sys.argv) < 5:
    print >> sys.stderr, "usage: kmeans <master> <dataFile> <outputDir> <k> <dist>"
    exit(-1)


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

sc = SparkContext(argsIn[0], "kmeans")
dataFile = str(argsIn[1])
outputDir = str(argsIn[2]) + "-kmeans"
k = int(argsIn[3])
dist = str(argsIn[4])
if not os.path.exists(outputDir):
    os.makedirs(outputDir)

# load data
lines = sc.textFile(dataFile)
data = parse(lines, "raw").cache()

if dist == 'corr':
    data = data.map(lambda x: (x - mean(x)) / norm(x)).cache()

centers = data.take(k)

if dist == 'corr':
    centers = map(lambda x: x - mean(x), centers)

convergeDist = 0.001
tempDist = 1.0
iteration = 0
mxIteration = 100

while (tempDist > convergeDist) & (iteration < mxIteration):

    if dist == 'corr':
        kPointsNorm = map(lambda x: x / norm(x), centers)
        closest = data.map(lambda p: (closestPoint(p, kPointsNorm, dist)[0], (p, 1)))
    if dist == 'euclidean':
        closest = data.map(lambda p: (closestPoint(p, centers, dist)[0], (p, 1)))

    pointStats = closest.reduceByKey(lambda (x1, y1), (x2, y2): (x1 + x2, y1 + y2))
    newPoints = pointStats.map(lambda (x, (y, z)): (x, y / z)).collect()
    tempDist = sum(sum((centers[x] - y) ** 2) for (x, y) in newPoints)

    for (i, j) in newPoints:
        centers[i] = j

    iteration += 1

if dist == 'corr':
    centers = map(lambda x: x / norm(x), centers)

labels = data.map(lambda p: closestPoint(p, centers, dist)[0]).collect()
dists = data.map(lambda p: closestPoint(p, centers, dist)[1]).collect()
saveout(labels, outputDir, "labels", "matlab")
saveout(dists, outputDir, "dists", "matlab")
saveout(centers, outputDir, "centers", "matlab")

if dist == 'euclidean':
    centers = map(lambda x: x / norm(x), centers)
    normDists = data.map(lambda p: closestPoint((p - mean(p))/norm(p), centers, 'corr')[1]).collect()
    saveout(normDists, outputDir, "normDists", "matlab")
