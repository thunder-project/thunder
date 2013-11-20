# utilities for loading and saving data

from scipy.io import * 
from numpy import *
#import pyspark


def parseVector(line, filter="raw", inds=None):

	vec = [float(x) for x in line.split(' ')]
	ts = array(vec[3:]) # get tseries
	if filter == "dff" : # convert to dff
		meanVal = mean(ts)
		ts = (ts - meanVal) / (meanVal + 0.1)
	if inds is not None :
		if inds == "xyz" :
			return ((int(vec[0]),int(vec[1]),int(vec[2])),ts)
		if inds == "linear" :
			k = int(vec[0]) + int((vec[1] - 1)*1650)
			return (k,ts)
	else :
		return ts
	
def saveout(data, outputDir, outputFile, outputFormat) :

	if outputFormat == "matlab" :
		dtype = type(data) 
		#
		savemat(outputDir+"/"+outputFile+".mat",mdict={outputFile : data},oned_as='column',do_compression='true')
