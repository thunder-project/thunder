# ref <master> <dataFile> <outputDir> <mode>
# 
# compute summary statistics
#

import sys
import os
from numpy import *
import thunder
from thunder.util.dataio import *
from pyspark import SparkContext

argsIn = sys.argv[1:]

if len(argsIn) < 4:
  print >> sys.stderr, \
  "(ref) usage: ref <master> <dataFile> <outputDir> <mode>"
  exit(-1)

# parse inputs
sc = SparkContext(argsIn[0], "ref")
dataFile = str(argsIn[1])
outputDir = str(argsIn[2])
mode = str(argsIn[3])
if not os.path.exists(outputDir) : os.makedirs(outputDir)

# parse data
lines = sc.textFile(dataFile)
data = parse(lines, "raw", "xyz").cache()

# get z ordering
zinds = data.filter(lambda (k,x) : (k[0] == 1) & (k[1] == 1)).map(lambda (k,x) : k[2])
saveout(zinds,outputDir,"zinds","matlab")

# compute ref
if mode == 'med':
	ref = data.map(lambda (k,x) : median(x))
if mode == 'mean':
	ref = data.map(lambda (k,x) : mean(x))
if mode == 'std':
	ref = data.map(lambda (k,x) : std((x - mean(x))/(mean(x)+0.1)))
if mode == 'perc':
	ref = data.map(lambda (k,x) : percentile(x,90))

saveout(ref.map(float16).collect(),outputDir,"ref"+mode,"matlab")