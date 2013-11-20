# regression <master> <dataFile> <modelFile> <outputDir> <mode> <opts>

import sys
import os
from numpy import *
from scipy.linalg import *
from pyspark import SparkContext
from thunder.util.dataio import *
from thunder.regression.util import *
from thunder.factorization.util import *

argsIn = sys.argv[1:]
if len(argsIn) < 5:
  print >> sys.stderr, \
  "(regression) usage: regression <master> <dataFile> <modelFile> <outputDir> <mode>"
  exit(-1)

# parse inputs
sc = SparkContext(argsIn[0], "regression")
dataFile = str(argsIn[1])
modelFile = str(argsIn[2])
outputDir = str(argsIn[3]) + "-regression"
mode = str(argsIn[4])
if not os.path.exists(outputDir) : os.makedirs(outputDir)

# parse data
lines = sc.textFile(dataFile)
data = parse(lines, "raw").cache()

# create model
model = regressionModel(modelFile,mode)

# do regression
betas = regressionFit(data,model).cache()

# get statistics
stats = betas.map(lambda x : x[1])
saveout(stats,outputDir,"stats","matlab")

# do PCA
comps,latent,scores = svd1(betas.map(lambda x : x[0]),2)

# write results
saveout(comps,outputDir,"comps","matlab")
saveout(latent,outputDir,"latent","matlab")
saveout(scores,outputDir,"scores","matlab")

# compute trajectories from raw data
traj = regressionFit(data,model,comps)
saveout(traj,outputDir,"traj","matlab")

# get simple measure of response strength
r = data.map(lambda x : norm(x-mean(x)))
saveout(r,outputDir,"r","matlab")


# # get statistics using randomization
# if outputMode == 'stats'
# 	stats = Y.map(lambda y : getRegression(y,model)).collect()
# 	savemat(outputFile+"/"+"stats.mat",mdict={'stats':stats},oned_as='column',do_compression='true')

# # get norms of coefficients to make a contrast map
# if outputMode == 'norm' :
# 	B = Y.map(lambda y : (y,getNorm(y,model)))
# 	n = B.count()
# 	m = len(Y.first())
# 	traj = zeros((2,m))
# 	for ic in range(0,2) :
# 		traj[ic,:] = B.filter(lambda (y,b) : (b[ic] - b[1-ic])>0.01).map(lambda (y,b) : y * b[ic]).reduce(lambda x,y : x + y) / n
# 	norms = B.map(lambda (y,b) : float16(b)).collect()
# 	savemat(outputFile+"/"+"traj.mat",mdict={'traj':traj},oned_as='column',do_compression='true')
# 	savemat(outputFile+"/"+"norms.mat",mdict={'norms':norms},oned_as='column',do_compression='true')

# def getNorm(y,model) : 
# 	b = getRegression(y,model)
# 	n = zeros((model.nG,))
# 	for ig in range(0,model.nG) :
# 		n[ig] = norm(b[model.g==ig])
# 	return n

