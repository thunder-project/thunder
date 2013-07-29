import sys
import os
from numpy import *
from scipy.linalg import *
from scipy.sparse import *
from pyspark import SparkContext
import logging

if len(sys.argv) < 6:
  print >> sys.stderr, \
  "(shotgun) usage: shotgun <master> <inputFile_A> <inputFile_y> <outputFile> <lambda>"
  exit(-1)

def parseVector(line):
	vec = [float(x) for x in line.split(' ')]
	return (int(vec[0]),array(vec[1:]))

def updateFeature(x,y,Ab,b,lam):
	AA = dot(x,x)
	Ay = dot(x,y)
	d_j = Ay - dot(x,Ab) + AA*b
	if d_j < -lam :
		new_value = (d_j + lam)/AA
	elif d_j > lam:
		new_value = (d_j - lam)/AA
	else :
		new_value = 0
	return float(new_value)

# parse inputs
sc = SparkContext(sys.argv[1], "shotgun")
inputFile_A = str(sys.argv[2])
inputFile_y = str(sys.argv[3])
outputFile = str(sys.argv[4])
lam = double(sys.argv[5])
if not os.path.exists(outputFile):
    os.makedirs(outputFile)
logging.basicConfig(filename=outputFile+'/'+'stdout.log',level=logging.INFO,format='%(asctime)s %(message)s',datefmt='%m/%d/%Y %I:%M:%S %p')

# parse data
logging.info("(shotgun) loading data")
lines_A = sc.textFile(inputFile_A)
lines_y = sc.textFile(inputFile_y)
y = array([float(x) for x in lines_y.collect()[0].split(' ')])
A = lines_A.map(parseVector).cache()
d = A.count()
n = len(A.first()[1])

# initialize sparse weight vector
b = csr_matrix((d,1))
bOld = csr_matrix((d,1))
Ab = zeros((n,1))
# precompute constants (d x 1)
#logging.info("(shotgun) precomputing vectors")
#Ay = A.map(lambda (k,x) : dot(x,y)).collect()
#AA = A.map(lambda (k,x) : dot(x,x)).collect()

iIter = 1
nIter = 50
deltaCheck = 10^2
tol = 10 ** -6

logging.info("(shotgun) beginning iterative estimation...")

while (iIter < nIter) & (deltaCheck > tol):
	logging.info("(shotgun) starting iteration " + str(iIter))
	logging.info("(shotgun) updating features")
	update = A.map(lambda (k,x) : (k,updateFeature(x,y,Ab,b[k,0],lam))).filter(lambda (k,x) : x != b[k,0]).collect()
	nUpdate = len(update)
	diff = zeros((nUpdate,1))
	for i in range(nUpdate):
		key = update[i][0]
		value = update[i][1]
		bOld[key,0] = b[key,0]
		b[key,0] = value
		diff[i] = abs(b[key,0] - bOld[key,0])

	deltaCheck = amax(diff)
	logging.info("(shotgun) features updated: " + str(nUpdate))
	logging.info("(shotgun) change in b: " + str(deltaCheck))
	
	logging.info("(shotgun) updating Ab")
	Ab = A.filter(lambda (k,x) : b[k,0] != 0).map(lambda (k,x) : x*b[k,0]).reduce(lambda x,y : x+y)	

	iIter = iIter + 1

logging.info("(shotgun) finised after " + str(iIter) + " iterations")

savetxt(outputFile+"/"+"b.txt",b.todense(),fmt='%.8f')

