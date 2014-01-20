"""
utilities for saving data
"""

import os
from scipy.io import savemat
from math import isnan
from numpy import sum, shape, reshape, prod, maximum, minimum, float16, uint8, savetxt
from PIL import Image
import pyspark


def arraytoim(mat, filename, dims):
    """write a numpy array to a png image
    given specified dimensions
    if mat is 3d, will separately write each image
    along the 3rd dimension

    arguments:
    mat - numpy array, 2d or 3d, dtype must be uint8
    filename - base filename for writing
    dims - dimensions of the image (2-tuple)
    """
    datadims = shape(mat)
    depth = prod(datadims)/prod(dims)
    if depth > 1:
        mat = reshape(mat, (dims[0], dims[1], depth))
        for z in range(0, depth):
            cdata = mat[:, :, z]
            Image.fromarray(cdata).save(filename+"-"+str(z)+".png")
    else:
        cdata = reshape(mat, (dims[0], dims[1]))
        Image.fromarray(cdata).save(filename+".png")


def rescale(data):
    """rescale data to lie between 0 and 255 and convert to uint8
    for strictly positive data, subtract the min and divide by max
    otherwise, divide by the maximum absolute value and center
    if each element of data has multiple entries,
    they will be rescaled separately

    arguments:
    data - RDD of doubles or numpy arrays
    """
    data = data.map(lambda x: 0 if isnan(x) else x)
    mnvals = data.reduce(minimum)
    mxvals = data.reduce(maximum)
    if sum(mnvals < 0) == 0:
        data = data.map(lambda x: uint8(255 * (x - mnvals)/(mxvals - mnvals)))
    else:
        mxvals = maximum(abs(mxvals), abs(mnvals))
        data = data.map(lambda x: uint8(255 * ((x / (2 * mxvals)) + 0.5)))
    return data


def saveout(data, outputdir, outputfile, outputformat, nout=None, dims=None):
    """methods for saving data

    arguments:
    data - RDD or array
    outputdir - location to save data to
    outputfile - file name to save data to
    outputform - format for data ("matlab", "text", or "image")

    opt arguments:
    nout - number of entries per element in RDD
    dims - image dimensions (width, height)
    """

    filename = os.path.join(outputdir, outputfile)

    if (outputformat == "matlab") | (outputformat == "text"):
        dtype = type(data)

        if (dtype == pyspark.rdd.RDD) | (dtype == pyspark.rdd.PipelinedRDD):
            if nout > 1:
                for iout in range(0, nout):
                    result = data.map(lambda x: float16(x[iout])).collect()
                    if outputformat == "matlab":
                        savemat(filename+"-"+str(iout)+".mat",
                                mdict={outputfile+str(iout): result}, oned_as='column', do_compression='true')
                    if outputformat == "text":
                        savetxt(filename+"-"+str(iout)+".txt", result, fmt="%.6f")
            else:
                result = data.map(float16).collect()
                if outputformat == "matlab":
                    savemat(filename+".mat", mdict={outputfile: result}, oned_as='column', do_compression='true')
                if outputformat == "text":
                    savetxt(filename+".txt", result, fmt="%.6f")

        else:
            if outputformat == "matlab":
                savemat(filename+".mat", mdict={outputfile: data}, oned_as='column', do_compression='true')
            if outputformat == "text":
                savetxt(filename+".txt", data, fmt="%.6f")

    if outputformat == "image":

        data = rescale(data)
        dtype = type(data)

        if (dtype == pyspark.rdd.RDD) | (dtype == pyspark.rdd.PipelinedRDD):
            if nout > 1:
                for iout in range(0, nout):
                    result = data.map(lambda x: x[iout]).collect()
                    arraytoim(result, filename+"-"+str(iout), dims)
            else:
                result = data.collect()
                arraytoim(result, filename, dims)
        else:
            arraytoim(data, filename, dims)



