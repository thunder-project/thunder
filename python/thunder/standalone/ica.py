"""
Example standalone app for independent component analysis
"""

import argparse
from thunder.factorization import ICA
from thunder.utils.context import ThunderContext
from thunder.utils.save import save


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="do independent components analysis")
    parser.add_argument("datafile", type=str)
    parser.add_argument("outputdir", type=str)
    parser.add_argument("k", type=int)
    parser.add_argument("c", type=int)
    parser.add_argument("--svdmethod", choices=("direct", "em"), default="direct", required=False)
    parser.add_argument("--maxiter", type=float, default=100, required=False)
    parser.add_argument("--tol", type=float, default=0.000001, required=False)
    parser.add_argument("--seed", type=int, default=0, required=False)
    parser.add_argument("--preprocess", choices=("raw", "dff", "sub", "dff-highpass", "dff-percentile"
                        "dff-detrendnonlin", "dff-detrend-percentile"), default="raw", required=False)

    args = parser.parse_args()

    tsc = ThunderContext.start(appName="ica")

    data = tsc.loadText(args.datafile, args.preprocess).cache()
    result = ICA(args.k, args.c, args.svdmethod, args.maxiter, args.tol, args.seed).fit(data)

    outputdir = args.outputdir + "-ica"
    save(result.w, outputdir, "w", "matlab")
    save(result.sigs, outputdir, "sigs", "matlab")
