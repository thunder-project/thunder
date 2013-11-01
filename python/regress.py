# regress <master> <inputFile_Y> <inputFile_X> <outputFile> <analMode> <outputMode>
# 
# time series regression on a data matrix
# each row is (x,y,z,timeseries)
# inputs are signals to regress against
# can process results either by doing dimensionality reduction
# or by fitting a parametric model
#

import sys
import os
from numpy import *
from scipy.linalg import *
from scipy.io import * 
from scipy.stats import vonmises
from pyspark import SparkContext
import logging

argsIn = sys.argv[1:]

if len(argsIn) < 6:
  print >> sys.stderr, \
  "(regress) usage: regress <master> <inputFile_Y> <inputFile_X> <outputFile> <regressMode> <outputMode> <opts>"
  exit(-1)

def parseVector(line,mode="raw",xyz=0,inds=None):
	vec = [float(x) for x in line.split(' ')]
	ts = array(vec[3:]) # get tseries
	if inds is not None :
		ts = ts[inds[0]:inds[1]]
	if mode == "dff" :
		meanVal = mean(ts)
		ts = (ts - meanVal) / (meanVal + 0.1)
	if xyz == 1 :
		return ((int(vec[0]),int(vec[1]),int(vec[2])),ts)
	else :
		return ts

def getRegression(y,model) :
	if model.regressMode == 'mean' :
		b = dot(model.X,y)
		return b
	if model.regressMode == 'linear' :
		b = dot(model.Xhat,y)[1:]
		# subtract mean separately for each group of regressors
		for ig in range(0,len(unique(model.g))) :
			ginds = model.g==ig
			b[ginds] = b[ginds] - mean(b[ginds])
		return b
	if model.regressMode == 'bilinear' :
		b1 = dot(model.X1hat,y)
		b1 = b1 - min(b1)
		b1hat = dot(transpose(model.X1),b1)
		X3 = X2 * b1hat
		X3 = concatenate((ones((1,shape(X3)[1])),X3))
		X3hat = dot(inv(dot(X3,transpose(X3))),X3)
		b2 = dot(X3hat,y)
		b = b2[1:]
		if model.outputMode == 'tuning' :
			return b
		if model.outputMode == 'pca' :
			return (b1 - mean(b1))


def getTuning(y,model) :
	if model.tuningMode == 'circular' :
		z = norm(y-mean(y))
		y = y - min(y)
		y = y/sum(y)
		r = inner(y,exp(1j*model.s))
		mu = angle(r)
		v = absolute(r)/sum(y)
		n = len(y)
		if v < 0.53 :
			k = 2*v + (v**3) + 5*(v**5)/6
		elif (v>=0.53) & (v<0.85) :
			k = -.4 + 1.39*v + 0.43/(1-v)
		else :
			k = 1/(v**3 - 4*(v**2) + 3*v)
		return (z,mu,k)

# parse inputs
sc = SparkContext(argsIn[0], "regress")
inputFile_Y = str(argsIn[1])
inputFile_X = str(argsIn[2])
outputFile = str(argsIn[3]) + "-regress"
regressMode = str(argsIn[4])
outputMode = str(argsIn[5])
opts = str(argsIn[6])

if not os.path.exists(outputFile) :
	os.makedirs(outputFile)
logging.basicConfig(filename=outputFile+'/'+'stdout.log',level=logging.INFO,format='%(asctime)s %(message)s',datefmt='%m/%d/%Y %I:%M:%S %p')

# parse data
logging.info("(regress) loading data")
lines_Y = sc.textFile(inputFile_Y)
Y = lines_Y.map(lambda y : parseVector(y,"dff")).cache()

# parse model
class model : pass
model.regressMode = regressMode
model.outputMode = outputMode
if regressMode == 'mean' :
	X = loadmat(inputFile_X + "_X.mat")['X']
	X = X.astype(float)
	model.X = X
if regressMode == 'linear' :
	X = loadmat(inputFile_X + "_X.mat")['X']
	X = X.astype(float)
	g = loadmat(inputFile_X + "_g.mat")['g']
	g = g.astype(float)[0]
	Xhat = dot(inv(dot(X,transpose(X))),X)
	model.X = X
	model.Xhat = Xhat
	model.g = g
if regressMode == 'bilinear' :
	X1 = loadmat(inputFile_X + "_X1.mat")['X1']
	X2 = loadmat(inputFile_X + "_X2.mat")['X2']
	X1hat = dot(inv(dot(X1,transpose(X1))),X1)
	model.X1 = X1
	model.X2 = X2
	model.X1hat = X1hat
if outputMode == 'tuning' :
	s = loadmat(inputFile_X + "_s.mat")['s']
	model.s = s
	model.tuningMode = opts
if outputMode == 'pca' :
	model.k = int(opts)

# compute parameter estimates
B = Y.map(lambda y : getRegression(y,model))

# process outputs using pca
if outputMode == 'pca' :
	k = model.k
	n = B.count()
	cov = B.map(lambda b : outer(b,b)).reduce(lambda x,y : (x + y)) / n
	w, v = eig(cov)
	w = real(w)
	v = real(v)
	inds = argsort(w)[::-1]
	comps = transpose(v[:,inds[0:k]])
	savemat(outputFile+"/"+"comps.mat",mdict={'comps':comps},oned_as='column',do_compression='true')
	latent = w[inds[0:k]]
	for ik in range(0,k) :
		scores = Y.map(lambda y : float16(inner(getRegression(y,model),comps[ik,:])))
		savemat(outputFile+"/"+"scores-"+str(ik)+".mat",mdict={'scores':scores.collect()},oned_as='column',do_compression='true')
	traj = Y.map(lambda y : outer(y,inner(getRegression(y,model),comps))).reduce(lambda x,y : x + y) / n
	savemat(outputFile+"/"+"traj.mat",mdict={'traj':traj},oned_as='column',do_compression='true')

# process output with a parametric tuning curve
if outputMode == 'tuning' :
	if model.tuningMode == 'circular' :
		P = B.map(lambda b : float16(getTuning(b,model))).collect()
		#nOut = len(P.first())
		savemat(outputFile+"/"+"P.mat",mdict={'P':P},oned_as='column',do_compression='true')
		#for ip in range(0,nOut) :
		#	p = P.map(lambda p : float16(p[ip])).collect()
		#	savemat(outputFile+"/"+"p-"+str(ip)+".mat",mdict={'p':p},oned_as='column',do_compression='true')


























