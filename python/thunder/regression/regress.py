# regress <master> <dataFile> <modelFile> <outputDir> <regressMode>
#
# fit a regression model
#
# regressMode - form of regression (mean, linear, bilinear)
#
# example:
# regress.py local data/fish.txt data/regression/fish_linear results linear
#

import sys
import os
from numpy import *
from scipy.linalg import *
from thunder.util.dataio import *
from thunder.regression.util import *
from thunder.factorization.util import *
from pyspark import SparkContext

argsIn = sys.argv[1:]
if len(argsIn) < 5:
    print >> sys.stderr, "usage: regress <master> <dataFile> <modelFile> <outputDir> <regressMode>"
    exit(-1)

# parse inputs
sc = SparkContext(argsIn[0], "regress")
dataFile = str(argsIn[1])
modelFile = str(argsIn[2])
outputDir = str(argsIn[3]) + "-regress"
regressMode = str(argsIn[4])
if not os.path.exists(outputDir) : os.makedirs(outputDir)

# parse data
lines = sc.textFile(dataFile)
data = parse(lines, "dff").cache()

# create model
model = regressionModel(modelFile,regressMode)

# do regression
betas = regressionFit(data,model).cache()

# get statistics
stats = betas.map(lambda x : x[1])
saveout(stats,outputDir,"stats","matlab")

# do PCA
comps,latent,scores = svd1(betas.map(lambda x : x[0]),2)
saveout(comps,outputDir,"comps","matlab")
saveout(latent,outputDir,"latent","matlab")
saveout(scores,outputDir,"scores","matlab",2)

# compute trajectories from raw data
traj = regressionFit(data,model,comps)
saveout(traj,outputDir,"traj","matlab")

# get simple measure of response strength
r = data.map(lambda x : norm(x-mean(x)))
saveout(r,outputDir,"r","matlab")
