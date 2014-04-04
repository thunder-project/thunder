import os
import argparse
import glob
from thunder.sigprocessing.util import SigProcessingMethod
from thunder.util.load import load
from thunder.util.save import save
from pyspark import SparkContext


def fourier(data, freq):
    """Compute fourier transform of data points
    (typically time series data)

    :param data: RDD of data points as key value pairs
    :param freq: frequency (number of cycles)

    :return: co: RDD of coherence (normalized amplitude)
    :return: ph: RDD of phase
    """

    method = SigProcessingMethod.load("fourier", freq=freq)
    out = method.calc(data)

    co = out.mapValues(lambda x: x[0])
    ph = out.mapValues(lambda x: x[1])

    return co, ph

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="compute a fourier transform on each time series")
    parser.add_argument("master", type=str)
    parser.add_argument("datafile", type=str)
    parser.add_argument("outputdir", type=str)
    parser.add_argument("freq", type=int)
    parser.add_argument("--preprocess", choices=("raw", "dff", "dff-highpass", "sub"), default="raw", required=False)

    args = parser.parse_args()

    sc = SparkContext(args.master, "fourier")

    if args.master != "local":
        egg = glob.glob(os.path.join(os.environ['THUNDER_EGG'], "*.egg"))
        sc.addPyFile(egg[0])

    data = load(sc, args.datafile, args.preprocess).cache()

    co, ph = fourier(data, args.freq)

    outputdir = args.outputdir + "-fourier"

    save(co, outputdir, "co", "matlab")
    save(ph, outputdir, "ph", "matlab")
