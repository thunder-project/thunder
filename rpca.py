# rpca <master> <inputFile> <outputFile> <slices> 

import sys
import os
from numpy import *
from scipy.linalg import *
from pyspark import SparkContext

if len(sys.argv) < 5:
  print >> sys.stderr, \
    "(rpca) usage: rpca <master> <inputFile> <outputFile> <slices>"
  exit(-1)

def parseVector(line):
    return array([float(x) for x in line.split(' ')])

def shrinkVec(x,thresh):
	tmp = abs(x)-thresh
	tmp[tmp<0] = 0
	return tmp

def svdThreshold(RDD,thresh,n):
	cov = RDD.map(lambda x : outer(x,x)).reduce(lambda x,y : (x + y))
	w, v = eig(cov)
	sw = sqrt(w)
	inds = (sw-thresh)>0
	d = (1/sw[inds]) * shrinkVec(sw[inds],thresh)
	vthresh = real(dot(dot(v[:,inds],diag(d)),transpose(v[:,inds])))
	return RDD.map(lambda x : dot(x,vthresh))

def shrinkage(RDD,thresh):
	return RDD.map(lambda x : sign(x) * shrinkVec(x,thresh))

# parse inputs
sc = SparkContext(sys.argv[1], "rpca")
inputFile = str(sys.argv[2])
outputFile = str(sys.argv[3])
slices = int(sys.argv[4])
if not os.path.exists(outputFile):
    os.makedirs(outputFile)

# load data
lines = sc.textFile(inputFile,slices)
data = lines.map(parseVector)
n = data.count()
m = len(data.first())

# mean subtraction
#meanVec = data.reduce(lambda x,y : x+y) / n
#sub = data.map(lambda x : x - meanVec)

# create broadcast variables
print "(rpca) broadcasting variables"
M = array(data)
L = zeros((n,m))
S = zeros((n,m))
Y = zeros((n,m))
Mb = sc.broadcast(M)
Lb = sc.broadcast(L)
Sb = sc.broadcast(S)
Yb = sc.broadcast(Y)

mu = float(12)
lam = 1/sqrt(n)

iterNum = 0
iterMax = 50

while iterNum < iterMax:
	print "(rpca) starting iteration " + str(iterNum)
	iterNum += 1
	MSY = sc.parallelize(range(n),slices).map(
		lambda x : Mb.value[x,:] - Sb.value[x,:] + (1/mu)*Yb.value[x,:])
	L = svdThreshold(MSY,1/mu,n).collect()
	Lb = sc.broadcast(array(L))
	MLY = sc.parallelize(range(n),slices).map(
		lambda x : Mb.value[x,:] - Lb.value[x,:] + (1/mu)*Yb.value[x,:])
	S = shrinkage(MLY,lam/mu).collect()
	Sb = sc.broadcast(array(S))
	Y = sc.parallelize(range(n),slices).map(
		lambda x : Yb.value[x,:] + mu*(Mb.value[x,:] - Lb.value[x,:] - Sb.value[x,:])).collect()
	Yb = sc.broadcast(array(Y))

savetxt(outputFile+"/"+"out-L-"+outputFile+".txt",Lb.value,fmt='%.8f')
savetxt(outputFile+"/"+"out-S-"+outputFile+".txt",Sb.value,fmt='%.8f')

