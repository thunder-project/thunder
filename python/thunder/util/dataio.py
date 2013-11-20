# utilities for loading and saving data

from scipy.io import * 
from numpy import *
import pyspark

def parseVector(line, filter="raw", inds=None):

	vec = [float(x) for x in line.split(' ')]
	ts = array(vec[3:]) # get tseries
	return ts
	
def saveout(data, outputDir, outputFile, outputFormat) :

	if outputFormat == "matlab" :
		dtype = type(data) 
		if (dtype == pyspark.rdd.RDD) | (dtype == pyspark.rdd.PipelinedRDD) :
			savemat(outputDir+"/"+outputFile+".mat",mdict={outputFile : data.map(float16).collect()},oned_as='column',do_compression='true')
		else :
			savemat(outputDir+"/"+outputFile+".mat",mdict={outputFile : data},oned_as='column',do_compression='true')
