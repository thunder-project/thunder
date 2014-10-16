"""
Example standalone app for independent component analysis
"""

import argparse
from thunder import ThunderContext, ICA, export


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

    args = parser.parse_args()

    tsc = ThunderContext.start(appName="ica")

    data = tsc.loadSeries(args.datafile).cache()
    model = ICA(k=args.k, c=args.c, svdmethod=args.svdmethod, maxiter=args.maxiter, tol=args.tol, seed=args.seed)
    result = model.fit(data)

    outputdir = args.outputdir + "-ica"
    export(result.a, outputdir, "a", "matlab")
    export(result.sigs, outputdir, "sigs", "matlab")
