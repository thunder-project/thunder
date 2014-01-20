import os
import argparse
import glob
from numpy import random, sqrt, zeros, real, dot, outer, diag, transpose
from scipy.linalg import sqrtm, inv, orth
from thunder.util.parse import parse
from thunder.util.saveout import saveout
from thunder.factorization.util import svd
from pyspark import SparkContext


def ica(data, k, c, svdmethod="direct", maxiter=100, tol=0.000001, seed=0):
    """perform independent components analysis

    arguments:
    data - RDD of data points
    k - number of principal components to use
    c - number of independent components to find
    maxiter - maximum number of iterations (default = 100)
    tol - tolerance for change in estimate (default = 0.000001)

    returns:
    w - the mixing matrix
    sigs - the independent components
    """
    # get count
    n = data.count()

    # reduce dimensionality
    scores, latent, comps = svd(data, k, meansubtract=0, method=svdmethod)

    # whiten data
    whtmat = real(dot(inv(diag(latent/sqrt(n))), comps))
    unwhtmat = real(dot(transpose(comps), diag(latent/sqrt(n))))
    wht = data.map(lambda x: dot(whtmat, x))

    # do multiple independent component extraction
    if seed != 0:
        random.seed(seed)
    b = orth(random.randn(k, c))
    b_old = zeros((k, c))
    iter = 0
    minabscos = 0
    errvec = zeros(maxiter)

    while (iter < maxiter) & ((1 - minabscos) > tol):
        iter += 1
        # update rule for pow3 non-linearity (TODO: add others)
        b = wht.map(lambda x: outer(x, dot(x, b) ** 3)).sum() / n - 3 * b
        # make orthogonal
        b = dot(b, real(sqrtm(inv(dot(transpose(b), b)))))
        # evaluate error
        minabscos = min(abs(diag(dot(transpose(b), b_old))))
        # store results
        b_old = b
        errvec[iter-1] = (1 - minabscos)

    # get un-mixing matrix
    w = dot(transpose(b), whtmat)

    # get components
    sigs = data.map(lambda x: dot(w, x))

    return w, sigs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="do independent components analysis")
    parser.add_argument("master", type=str)
    parser.add_argument("datafile", type=str)
    parser.add_argument("outputdir", type=str)
    parser.add_argument("k", type=int)
    parser.add_argument("c", type=int)
    parser.add_argument("--svdmethod", choices=("direct", "em"), default="direct", required=False)
    parser.add_argument("--maxiter", type=float, default=100, required=False)
    parser.add_argument("--tol", type=float, default=0.000001, required=False)
    parser.add_argument("--preprocess", choices=("raw", "dff", "sub"), default="raw", required=False)
    parser.add_argument("--seed", type=int, default=0, required=False)

    args = parser.parse_args()
    egg = glob.glob(os.environ['THUNDER_EGG'] + "*.egg")
    sc = SparkContext(args.master, "ica", pyFiles=egg)
    lines = sc.textFile(args.datafile)
    data = parse(lines, args.preprocess).cache()

    w, sigs = ica(data, args.k, args.c, svdmethod=args.svdmethod, maxiter=args.maxiter, tol=args.tol, seed=args.seed)

    outputdir = args.outputdir + "-ica"
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)

    saveout(w, outputdir, "w", "matlab")
    saveout(sigs, outputdir, "sigs", "matlab", nout=args.c)
