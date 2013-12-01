# query <master> <inputFile> <inds> <outputDir>
# 
# query a data set by averaging together points
#

import sys
import os
from numpy import *
from scipy.linalg import *
from pyspark import SparkContext

argsIn = sys.argv[1:]

if len(sys.argv) < 4:
    print >> sys.stderr, \
    "(query) usage: query <master> <inputFile> <inds> <outputDir>"
    exit(-1)

# parse inputs
sc = SparkContext(sys.argv[1], "query")
inputFile = str(sys.argv[2])
indsFile = str(sys.argv[3])
outputDir = str(sys.argv[4]) + "-query"

lines = sc.textFile(dataFile)
data = parse(lines, "raw", "linear").cache()

inds = loadmat(indsFile)['inds'][0]

if len(inds) == 1 :
    indsTmp = inds[0]
    n = len(indsTmp)
    ts = data.filter(lambda (k,x) : k in indsTmp).map(lambda (k,x) : x).reduce(lambda x,y :x+y) / n
    saveout(ts,outputDir,"ts","matlab")

else :
    nInds = len(inds)
    ts = zeros((len(data.first()[1]),nInds))
    for i in range(0,nInds) :
        indsTmp = inds[i]
        n = len(indsTmp)
        ts[:,i] = data.filter(lambda (k,x) : k in indsTmp).map(lambda (k,x) : x).reduce(lambda x,y :x+y) / n
    saveout(ts,outputDir,"ts","matlab")




