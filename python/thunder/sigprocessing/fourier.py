import os
import argparse
import glob
from thunder.sigprocessing.util import SigProcessingMethod
from thunder.util.parse import parse
from thunder.util.saveout import saveout
from pyspark import SparkContext


def fourier(data, freq):
    """compute fourier transform of data points
    (typically time series data)

    arguments:
    data - RDD of data points
    freq - frequency (number of cycles - 1)

    returns:
    co - RDD of coherence (normalized amplitude)
    ph - RDD of phase
    """

    method = SigProcessingMethod.load("fourier", freq=freq)
    out = method.calc(data).cache()

    co = out.map(lambda x: x[0])
    ph = out.map(lambda x: x[1])

    return co, ph

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="compute a fourier transform on each time series")
    parser.add_argument("master", type=str)
    parser.add_argument("datafile", type=str)
    parser.add_argument("outputdir", type=str)
    parser.add_argument("freq", type=int)
    parser.add_argument("--preprocess", choices=("raw", "dff", "sub"), default="raw", required=False)

    args = parser.parse_args()
    egg = glob.glob(os.environ['THUNDER_EGG'] + "*.egg")
    sc = SparkContext(args.master, "fourier", pyFiles=egg)

    lines = sc.textFile(args.datafile)
    data = parse(lines, "dff")

    co, ph = fourier(data, args.freq)

    outputdir = args.outputdir + "-fourier"
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)
    saveout(co, outputdir, "co", "matlab")
    saveout(ph, outputdir, "ph", "matlab")
