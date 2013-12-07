# pca <master> <dataFile> <outputDir> <k>
#
# performs PCA using the SVD
#
# k - number of principal components to return
#
# example:
# pca.py local data/iris.txt results 2
#

import sys
import os
from thunder.util.dataio import parse, saveout
from thunder.factorization.util import svd1, svd3
from pyspark import SparkContext

argsIn = sys.argv[1:]
if len(argsIn) < 4:
    print >> sys.stderr, "usage: pca <master> <dataFile> <outputDir> <k> <nPartitions>"
    exit(-1)

# parse inputs
sc = SparkContext(argsIn[0], "pca")
dataFile = str(argsIn[1])
outputDir = str(argsIn[2]) + "-pca"
k = int(argsIn[3])

if not os.path.exists(outputDir):
    os.makedirs(outputDir)

# load data
lines = sc.textFile(dataFile)
data = parse(lines, "raw").cache()

# do pca
comps, latent, scores = svd3(data, k, 0)
#saveout(comps, outputDir, "comps", "matlab")
#saveout(latent, outputDir, "latent", "matlab")
#saveout(scores, outputDir, "scores", "matlab", k)
