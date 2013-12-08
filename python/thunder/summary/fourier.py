# fourier <master> <dataFile> <outputFile> <freq>
#
# computes the amplitude and phase of time series data
#
# example:
# fourier.py local data/fish.txt results 12
#

import argparse
import sys
import os
from numpy import mean, array, angle, abs, sqrt, zeros, fix, size, pi
from numpy.fft import fft
from thunder.util.dataio import *
from pyspark import SparkContext


def getFourierTransform(vec, freq):
    vec = vec - mean(vec)
    nframes = len(vec)
    ft = fft(vec)
    ft = ft[0:int(fix(nframes/2))]
    ampFT = 2*abs(ft)/nframes
    amp = ampFT[freq]
    co = zeros(size(amp))
    sumAmp = sqrt(sum(ampFT**2))
    co = amp / sumAmp
    ph = -(pi/2) - angle(ft[freq])
    if ph < 0:
        ph += pi * 2
    return array([co, ph])


def fourier(sc, dataFile, outputDir, freq):

    if not os.path.exists(outputDir):
        os.makedirs(outputDir)

    # load data
    lines = sc.textFile(dataFile)
    data = parse(lines, "dff")

    # do fourier analysis on each time series
    out = data.map(lambda x: getFourierTransform(x, freq)).cache()

    # save results
    co = out.map(lambda x: x[0])
    ph = out.map(lambda x: x[1])

    saveout(co, outputDir, "co", "matlab")
    saveout(ph, outputDir, "ph", "matlab")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="compute a fourier transform on each time series")
    parser.add_argument("master", type=str)
    parser.add_argument("dataFile", type=str)
    parser.add_argument("outputDir", type=str)
    parser.add_argument("freq", type=int)

    args = parser.parse_args()
    sc = SparkContext(args.master, "fourier")
    outputDir = args.outputDir + "-fourier"

    fourier(sc, args.dataFile, args.outputDir, args.freq)
