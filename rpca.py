# rpca <master> <inputFile> <outputFile> <slices> 

import sys

from numpy import *
from scipy.linalg import *
from pyspark import SparkContext

if len(sys.argv) < 7:
  print >> sys.stderr, \
    "(pca) usage: rpca <master> <inputFile> <outputFile> <slices>"
  exit(-1)

def parseVector(line):
    return array([float(x) for x in line.split(' ')])

def svdThreshold(RDD,thresh):
	cov = RDD.map(lambda x : outer(x,x)).reduce(lambda x,y : (x + y)) / (n - 1)
	w, v = eig(cov)
	sw = sqrt(w)
	inds = find((sw-r)>0)
	d = 1/(sw(inds)) * max(sw(inds)-r,0)
	vthresh = dot(dot(v[:,inds],diag(d)),transpose(v[:,inds]))
	return RDD.map(lambda x : dot(x,vthresh))

def shrinkage(RDD,thresh):
	return RDD.map(lambda x : sign(x) * max(abs(x)-thresh,0))

# parse inputs
sc = SparkContext(sys.argv[1], "pca")
inputFile = str(sys.argv[2])
outputFile = str(sys.argv[3])
slices = int(sys.argv[4])
if not os.path.exists(outputFile):
    os.makedirs(outputFile)

# load data
lines = sc.textFile(inputFile,slices)
data = lines.map(parseVector).cache()
n = data.count()
m = len(data.first())

# mean subtraction
meanVec = data.reduce(lambda x,y : x+y) / n
sub = data.map(lambda x : x - meanVec)

# create variables
M = sub.collect()
L = zeros((n,m))
S = zeros((n,m))
Y = zeros((n,m))
Mb = sc.broadcast(M)
Lb = sc.broadcast(L)
Sb = sc.broadcast(S)
Yb = sc.broadcast(Y)

mu = 12
lam = 1/sqrt(n)

iterNum = 0
iterMax = 10

while iterNum < iterMax:
	iterNum += 1
	MSY = sc.parallelize(range(n), slices).map(
		lambda x : Mb.value[x,:] - Sb.value[x,:] + (1/mu)*Yb.value[x,:])
	L = svdThreshold(MSY).collect()
	Lb = sc.broadcast(L)
	MLY = sc.parallelize(range(n), slices).map(
		lambda x : Mb.value[x,:] - Lb.value[x,:] + (1/mu)*Yb.value[x,:])
	S = shrinkage(MLY).collect()
	Sb = sc.broadcast(S)
	Y = sc.parallelize(range(n), slices).map(
		lambda x : Yb.value[x,:] + mu*(Mb.value[x,:]-Lb.value[x,:]-Sb.value[x,:])).collect()
	Yb = sc.broadcast(Y)

# print S, L

