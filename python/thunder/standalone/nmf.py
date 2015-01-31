"""
Example standalone app for non-negative factorization
"""

import optparse
from thunder import ThunderContext, NMF, export


if __name__ == "__main__":
    parser = optparse.OptionParser(description="do non-negative matrix factorization",
                                   usage="%prog datafile outputdir k [options]")
    parser.add_option("--nmfmethod", choices=["als"], default="als")
    parser.add_option("--maxiter", type=float, default=20)
    parser.add_option("--tol", type=float, default=0.001)
    parser.add_option("--w_hist", action="store_true", default=False)
    parser.add_option("--recon_hist", action="store_true", default=False)

    opts, args = parser.parse_args()
    try:
        datafile = args[0]
        outputdir = args[1]
        k = int(args[2])
    except IndexError:
        parser.print_usage()
        raise Exception("too few arguments")

    tsc = ThunderContext.start(appName="nmf")

    data = tsc.loadSeries(datafile).cache()
    nmf = NMF(k=k, method=opts.nmfmethod, maxiter=opts.maxiter, tol=opts.tol, w_hist=opts.w_hist,
              recon_hist=opts.recon_hist)
    nmf.fit(data)

    outputdir += "-nmf"
    export(nmf.w, outputdir, "w", "matlab")
    export(nmf.h, outputdir, "h", "matlab")
    if opts.w_hist:
        export(nmf.w_convergence, outputdir, "w_convergence", "matlab")
    if opts.recon_hist:
        export(nmf.recon_err, outputdir, "rec_err", "matlab")
