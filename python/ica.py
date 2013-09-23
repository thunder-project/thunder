# ica <master> <inputFile> <outputFile> <k> <c>
# 
# perform ica on a data matrix
# each row is (x,y,z,timeseries)
#

import sys
import os
from numpy import *
from scipy.linalg import *
from scipy.io import * 
from pyspark import SparkContext
import logging

if len(sys.argv) < 6:
  print >> sys.stderr, \
  "(ica) usage: ica <master> <inputFile> <outputFile> <k> <c>"
  exit(-1)

def parseVector(line):
	vec = [float(x) for x in line.split(' ')]
	ts = array(vec[3:]) # get tseries, drop x,y,z coords
	med = median(ts)
	ts = (ts - med) / (med + 0.1) # convert to dff
	return ts

# parse inputs
sc = SparkContext(sys.argv[1], "ica")
inputFile = str(sys.argv[2]);
k = int(sys.argv[4])
c = int(sys.argv[5])
outputFile = str(sys.argv[3])+"-ica-pcs-"+str(k)+"-ics-"+str(c)
if not os.path.exists(outputFile):
    os.makedirs(outputFile)
logging.basicConfig(filename=outputFile+'/'+'stdout.log',level=logging.INFO,format='%(asctime)s %(message)s',datefmt='%m/%d/%Y %I:%M:%S %p')

# load data
logging.info('(ica) loading data')
lines = sc.textFile(inputFile)

# compute covariance matrix
data = lines.map(parseVector).cache()
n = data.count()
m = len(data.first())
logging.info('(ica) mean subtraction')
meanVec = data.reduce(lambda x,y : x+y) / n
sub = data.map(lambda x : x)
logging.info('(ica) computing covariance')
cov = sub.map(lambda x : outer(x,x)).reduce(lambda x,y : (x + y)) / n

# do eigenvector decomposition
logging.info('(ica) doing eigendecomposition')
w, v = eig(cov)
inds = argsort(w)[::-1]
kEigVecs = v[:,inds[0:k]]
kEigVals = w[inds[0:k]]

# whiten data
logging.info('(ica) whitening data')
whtMat = real(dot(inv(diag(sqrt(kEigVals))),transpose(kEigVecs)))
unwhtMat = real(dot(kEigVecs,diag(sqrt(kEigVals))))
wht = sub.map(lambda x : dot(whtMat,x))

# do multiple independent component extraction
logging.info('(ica) starting iterative ica')
B = orth(random.randn(k,c))
Bold = zeros((k,c))
iterNum = 0
minAbsCos = 0
termTol = 0.000001
iterMax = 1000
errVec = zeros(iterMax)

while (iterNum < iterMax) & ((1 - minAbsCos) > termTol):
	iterNum += 1
	logging.info('(ica) iteration ' + str(iterNum))
	# update rule for pow3 nonlinearity (todo: add other nonlins)
	B = wht.map(lambda x : outer(x,dot(x,B) ** 3)).reduce(lambda x,y : x + y) / n - 3 * B
	# orthognalize
	B = dot(B,real(sqrtm(inv(dot(transpose(B),B)))))
	# evaluate error
	minAbsCos = min(abs(diag(dot(transpose(B),Bold))))
	# store results
	Bold = B
	errVec[iterNum-1] = (1 - minAbsCos)

# get unmixing matrix
W = dot(transpose(B),whtMat)

# save output files
logging.info('(ica) finished after ' + str(iterNum) + ' iterations')
logging.info('(ica) writing output...')
savemat(outputFile+"/"+"W-comps-"+str(c)+".mat",mdict={'W':W},oned_as='column',do_compression='true')

for ic in range(0,c):
	sigs = sub.map(lambda x : dot(W[ic,:],x))
	savemat(outputFile+"/"+"sig-"+str(ic)+"-comps-"+str(c)+".mat",mdict={'sigs':sigs.collect()},oned_as='column',do_compression='true')



