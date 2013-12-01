# pca <master> <dataFile> <outputDir> <k>
# 
# performs PCA using the SVD
#
# k - number of principal components to return
#

import sys
import os
from numpy import *
from scipy.linalg import *
from thunder.util.dataio import *
from thunder.factorization.util import *
from pyspark import SparkContext

argsIn = sys.argv[1:]

if len(sys.argv) < 4:
    print >> sys.stderr, \
        "(pca) usage: pca <master> <dataFile> <outputDir> <k>"
    exit(-1)

# parse inputs
sc = SparkContext(sys.argv[1], "pca")
dataFile = str(sys.argv[2])
outputDir = str(sys.argv[3]) + "-pca"
k = int(sys.argv[4])

if not os.path.exists(outputDir) : os.makedirs(outputDir)

# load data
lines = sc.textFile(dataFile)
data = parse(lines, "dff").cache()
n = data.count()

# do pca
comps,latent,scores = svd1(data,k)
saveout(comps,outputDir,"comps","matlab")
saveout(latent,outputDir,"latent","matlab")
saveout(scores,outputDir,"scores","matlab")
