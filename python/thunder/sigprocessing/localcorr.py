import os
import argparse
import glob
from numpy import corrcoef
from thunder.util.parse import parse
from thunder.util.saveout import saveout
from pyspark import SparkContext


def clip(val, mn, mx):
    """clip a value below by mn and above by mx"""

    if val < mn:
        return mn
    if val > mx:
        return mx
    else:
        return val


def maptoneighborhood(ind, ts, sz, mn_x, mx_x, mn_y, mx_y):
    """create a list of key value pairs with multiple shifted copies
    of the time series ts over a region specified by sz
    """
    rng_x = range(-sz, sz+1, 1)
    rng_y = range(-sz, sz+1, 1)
    out = list()
    for x in rng_x:
        for y in rng_y:
            new_x = clip(ind[0] + x, mn_x, mx_x)
            new_y = clip(ind[1] + y, mn_y, mx_y)
            newind = (new_x, new_y, ind[2])
            out.append((newind, ts))
    return out


def localcorr(data, sz):
    """compute correlation between every data point
    and the average of a local neighborhood in x and y
    (typically time series data)

    arguments:
    data - RDD of data points (pairs of ((int,int), array) or ((int,int,int), array))
    sz - neighborhood size (total neighborhood is a 2*sz+1 square)

    returns:
    corr - RDD of correlations
    """

    # get boundaries
    xs = data.map(lambda (k, _): k[0])
    ys = data.map(lambda (k, _): k[1])
    mx_x = xs.reduce(max)
    mn_x = xs.reduce(min)
    mx_y = ys.reduce(max)
    mn_y = ys.reduce(min)

    # flat map to key value pairs where the key is neighborhood identifier and value is time series
    neighbors = data.flatMap(lambda (k, v): maptoneighborhood(k, v, sz, mn_x, mx_x, mn_y, mx_y))

    # printing here seems to fix a hang later, possibly a PySpark bug
    print(neighbors.first())

    # reduce by key to get the average time series for each neighborhood
    means = neighbors.reduceByKey(lambda x, y: x + y).map(lambda (k, v): (k, v / ((2*sz+1)**2)))

    # join with the original time series data to compute correlations
    result = data.join(means)

    # get correlations and sort by key so result is in the right order
    corr = result.map(lambda (k, v): (k, corrcoef(v[0], v[1])[0, 1])).sortByKey().map(
        lambda (k, v): v)

    return corr


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="correlate time series with neighbors")
    parser.add_argument("master", type=str)
    parser.add_argument("datafile", type=str)
    parser.add_argument("outputdir", type=str)
    parser.add_argument("sz", type=int)
    parser.add_argument("--preprocess", choices=("raw", "dff", "sub"), default="raw", required=False)

    args = parser.parse_args()
    egg = glob.glob(os.environ['THUNDER_EGG'] + "*.egg")
    sc = SparkContext(args.master, "localcorr", pyFiles=egg)

    lines = sc.textFile(args.datafile)
    data = parse(lines, args.preprocess, nkeys=3, keepkeys="true").cache()

    corrs = localcorr(data, args.sz)

    outputdir = args.outputdir + "-localcorr"
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)

    saveout(corrs, outputdir, "corr", "matlab")