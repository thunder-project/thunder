# empca <master> <inputFile> <outputFile> <k>
# 
# uses iterative EM to do pca on a data matrix
# algorithm by Sam Roweis (NIPS, 1997)
# input is a text file
# format should be rows of ' ' separated values
# - example: space (rows) x time (cols)
# - rows should be whichever dim is larger
# 'k' is number of pcs to return
# writes pcs and eigenvalues to text

import sys
import os
from numpy import *
from scipy.linalg import *
from pyspark import SparkContext
import logging

if len(sys.argv) < 5:
  print >> sys.stderr, \
    "(empca) usage: empca <master> <inputFile> <outputFile> <k>"
  exit(-1)

def parseVector(line):
    return array([float(x) for x in line.split(' ')])

def outerProd(x):
	return outer(x,x)

# parse inputs
sc = SparkContext(sys.argv[1], "empca")
inputFile = str(sys.argv[2])
k = int(sys.argv[4])
outputFile = "empca-"+str(sys.argv[3])
if not os.path.exists(outputFile):
    os.makedirs(outputFile)
logging.basicConfig(filename=outputFile+'/'+'stdout.log',level=logging.INFO,format='%(asctime)s %(message)s',datefmt='%m/%d/%Y %I:%M:%S %p')

# initialize data and mean subtract
logging.info('(empca) loading data')
lines = sc.textFile(inputFile)
data = lines.map(parseVector)
n = data.count()
d = len(data.first())
logging.info('(empca) mean subtraction')
meanVec = data.reduce(lambda x,y : x+y) / n
sub = data.map(lambda x : x - meanVec).cache()

# initialize pcs
C = random.rand(k,d)
nIter = 20

# iteratively estimate subspace
for iter in range(nIter):
	logging.info('(empca) doing iteration ' + str(iter))
	Cold = C
	Cinv = dot(transpose(C),inv(dot(C,transpose(C))))
	Cinvb = sc.broadcast(Cinv)
	XX = sub.map(
		lambda x : outerProd(dot(x,Cinvb.value))).reduce(
		lambda x,y : x + y)
	XXinv = inv(XX)
	XXinvb = sc.broadcast(XXinv)
	C = sub.map(lambda x : outer(x,dot(dot(x,Cinvb.value),XXinvb.value))).reduce(
		lambda x,y: x + y)
	C = transpose(C)
	error = sum((C-Cold) ** 2)
	logging.info('(empca) change is ' + str(error))

logging.info('(empca) finished after ' + str(iter) + ' iterations')

# find ordered orthogonal basis for the subspace
logging.info('(empca) orthogalizing result')
C = transpose(orth(transpose(C)))
Cb = sc.broadcast(C)
cov = sub.map(lambda x : outerProd(dot(x,transpose(Cb.value)))).reduce(
	lambda x,y : x + y) / n
w, v = eig(cov)
w = real(w)
v = real(v)
inds = argsort(w)[::-1]
evecs = dot(transpose(v[:,inds[0:k]]),C)
evals = w[inds[0:k]]

# save the results
logging.info('(empca) saving to text')
savetxt(outputFile+"/"+"evecs-pcs-"+str(k)+".txt",evecs,fmt='%.8f')
savetxt(outputFile+"/"+"evals-pcs-"+str(k)+".txt",evals,fmt='%.8f')


