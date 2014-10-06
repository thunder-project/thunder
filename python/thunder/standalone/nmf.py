"""
Example standalone app for non-negative factorization
"""

import argparse
from thunder.factorization import NMF
from thunder.utils.context import ThunderContext
from thunder.utils.save import save


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
    parser.add_argument("--preprocess", choices=("raw", "dff", "sub", "dff-highpass", "dff-percentile"
                        "dff-detrendnonlin", "dff-detrend-percentile"), default="raw", required=False)

    args = parser.parse_args()

    tsc = ThunderContext.start(appName="nmf")

    data = tsc.loadText(args.datafile, args.preprocess).cache()
    nmf = NMF(k=args.k, method=args.nmfmethod, maxiter=args.maxiter, tol=args.tol, w_hist=args.w_hist,
              recon_hist=args.recon_hist)
    nmf.fit(data)

    outputdir = args.outputdir + "-nmf"
    save(nmf.w, outputdir, "w", "matlab")
    save(nmf.h, outputdir, "h", "matlab")
    if args.w_hist:
        save(nmf.w_convergence, outputdir, "w_convergence", "matlab")
    if args.recon_hist:
        save(nmf.recon_err, outputdir, "rec_err", "matlab")
