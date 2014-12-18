"""
Example standalone app for non-negative factorization
"""

import argparse
from thunder import ThunderContext, NMF, export


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="do non-negative matrix factorization")
    parser.add_argument("datafile", type=str)
    parser.add_argument("outputdir", type=str)
    parser.add_argument("k", type=int)
    parser.add_argument("--nmfmethod", choices="als", default="als", required=False)
    parser.add_argument("--maxiter", type=float, default=20, required=False)
    parser.add_argument("--tol", type=float, default=0.001, required=False)
    parser.add_argument("--w_hist", type=bool, default=False, required=False)
    parser.add_argument("--recon_hist", type=bool, default=False, required=False)

    args = parser.parse_args()

    tsc = ThunderContext.start(appName="nmf")

    data = tsc.loadSeries(args.datafile).cache()
    nmf = NMF(k=args.k, method=args.nmfmethod, maxIter=args.maxiter, tol=args.tol, wHist=args.w_hist,
              reconHist=args.recon_hist)
    nmf.fit(data)

    outputdir = args.outputdir + "-nmf"
    export(nmf.w, outputdir, "w", "matlab")
    export(nmf.h, outputdir, "h", "matlab")
    if args.w_hist:
        export(nmf.wConvergence, outputdir, "w_convergence", "matlab")
    if args.recon_hist:
        export(nmf.reconErr, outputdir, "rec_err", "matlab")
