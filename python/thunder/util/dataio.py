"""
utilities for loading and saving data
"""

from scipy.io import savemat
from numpy import array, mean, float16
import pyspark


def parse(data, filter="raw", inds=None, trange=None, xy=None):
    """methods for parsing data

    arguments:
    data - RDD of raw data points (lines of text, numbers separated by spaces)
    filter - how to filter the data ("raw", "dff", "sub")
    inds - whether to keep inds and in what form ("xyz", "linear")
    trange - only include a specified range of time points
    xy - array or tuple of max x and y values for conversion to linear indexing

    TODO: add a loader for small helper matrices, text or matlab format
    """

    def parsevector(line, filter="raw", inds=None, trange=None, xy=None):

        vec = [float(x) for x in line.split(' ')]
        ts = array(vec[3:])  # get tseries

        if filter == "dff":  # convert to dff
            meanval = mean(ts)
            ts = (ts - meanval) / (meanval + 0.1)

        # TODO: add soft normalization: compute the max, divide the max plus a constant
        if filter == "sub":  # subtracts the mean
            ts = (ts - mean(ts))

        if trange is not None:  # sub selects a range of indices
            ts = ts[trange[0]:trange[1]]

        if inds is not None:  # keep xyz keys
            if inds == "xyz":
                return (int(vec[0]), int(vec[1]), int(vec[2])), ts
            # TODO: once top is implemented in pyspark, use to get xy bounds
            if inds == "linear":
                k = int(vec[0]) + int((vec[1] - 1)*xy[0] + int((vec[2]-1)*xy[1])) - 1
                return k, ts

        else:
            return ts

    return data.map(lambda x: parsevector(x, filter, inds, trange, xy))


def saveout(data, outputdir, outputfile, outputformat, nout=1):
    """methods for saving data

    arguments:
    data - RDD or array
    outputdir - location to save data to
    outputfile - file name to save data to
    outputform - format for data ("matlab")
    nout - number of entries per element in RDD

    TODO: add option for writing out images
    TODO: add option for writing JSON (with a variety of modes)
    """

    if outputformat == "matlab":
        dtype = type(data)

        if (dtype == pyspark.rdd.RDD) | (dtype == pyspark.rdd.PipelinedRDD):
            if nout > 1:
                for iOut in range(0, nout):
                    result = data.map(lambda x: float16(x[iOut])).collect()
                    savemat(outputdir+"/"+outputfile+"-"+str(iOut)+".mat",
                            mdict={outputfile+str(iOut): result}, oned_as='column', do_compression='true')
            else:
                result = data.map(float16).collect()
                savemat(outputdir+"/"+outputfile+".mat",
                        mdict={outputfile: result}, oned_as='column', do_compression='true')

        else:
            savemat(outputdir+"/"+outputfile+".mat",
                    mdict={outputfile: data}, oned_as='column', do_compression='true')
