import sys
import os
from numpy import *
from scipy.linalg import *
from scipy.io import * 
from pyspark import SparkContext
import logging

if len(sys.argv) < 5:
  print >> sys.stderr, \
  "(trigger) usage: trigger <master> <inputFile_X> <inputFile_t> <outputFile>"
  exit(-1)

def parseVector(line):
	vec = [float(x) for x in line.split(' ')]
	ts = array(vec[3:]) # get tseries
	#med = median(ts)
	#ts = (ts - med) / (med) # convert to dff
	ts = ts/5000;
	return ts

# parse inputs
sc = SparkContext(sys.argv[1], "trigger")
inputFile_X = str(sys.argv[2])
inputFile_t = str(sys.argv[3])
outputFile = str(sys.argv[4]) + "-trigger"
if not os.path.exists(outputFile):
    os.makedirs(outputFile)
logging.basicConfig(filename=outputFile+'/'+'stdout.log',level=logging.INFO,format='%(asctime)s %(message)s',datefmt='%m/%d/%Y %I:%M:%S %p')

# parse data
logging.info("(trigger) loading data")
lines_X = sc.textFile(inputFile_X) # the data
X = lines_X.map(parseVector).cache()
t = loadmat(inputFile_t)['trigInds'] # the triggers

# compute triggered movie
if min(shape(t)) == 1 :
	for it in unique(t):
		logging.info('(trigger) getting triggered response at frame ' + str(it))
		resp = X.map(lambda x : mean(x[t[0]==it]))
		logging.info('(trigger) saving results...')
		savemat(outputFile+"/"+"resp-frame-"+str(int(it))+".mat",mdict={'resp':resp.collect()},oned_as='column',do_compression='true')
else :
	for it in range(shape(t)[0])
		logging.info('(trigger) getting triggered response at frame ' + str(it))
		resp = X.map(lambda x : dot(t,x))
		logging.info('(trigger) saving results...')
		savemat(outputFile+"/"+"resp-frame-"+str(int(it))+".mat",mdict={'resp':resp.collect()},oned_as='column',do_compression='true')