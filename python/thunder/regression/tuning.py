# tuning <master> <dataFile> <modelFile> <outputDir> <regressMode> <tuningMode>"
#
# fit a parametric tuning curve to regression results
#
# regressMode - form of regression (mean, linear, bilinear)
# tuningMode - parametric tuning curve (circular, gaussian)

import sys
import os
from numpy import *
from thunder.util.dataio import *
from thunder.regression.util import *
from thunder.factorization.util import *
from pyspark import SparkContext

argsIn = sys.argv[1:]
if len(argsIn) < 6:
    print >> sys.stderr, \
    "usage: tuning <master> <dataFile> <modelFile> <outputDir> <regressMode> <tuningMode>"
    exit(-1)

# parse inputs
sc = SparkContext(argsIn[0], "regress")
dataFile = str(argsIn[1])
modelFile = str(argsIn[2])
outputDir = str(argsIn[3]) + "-tuning"
regressMode = str(argsIn[4])
tuningMode = str(argsIn[5])
if not os.path.exists(outputDir) :
    os.makedirs(outputDir)

# parse data
lines = sc.textFile(dataFile)
data = parse(lines, "dff")

# create models
model1 = regressionModel(modelFile,regressMode)
model2 = tuningModel(modelFile,tuningMode)

# do regression
betas = regressionFit(data,model1).cache()

# get statistics
stats = betas.map(lambda x : x[1])
saveout(stats,outputDir,"stats","matlab")

# get tuning curves
params = tuningFit(betas.map(lambda x : x[0]),model2)
saveout(params,outputDir,"params","matlab")

if regressMode == "bilinear" :
    comps,latent,scores = svd1(betas.map(lambda x : x[2]),2)
    saveout(comps,outputDir,"comps","matlab")
    saveout(latent,outputDir,"latent","matlab")
    saveout(scores,outputDir,"scores","matlab",2)

# get simple measure of response strength
r = data.map(lambda x : norm(x-mean(x)))
saveout(r,outputDir,"r","matlab")

