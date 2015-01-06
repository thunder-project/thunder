"""
Example standalone app for mass-unvariate regression combined with PCA
"""

import optparse
from thunder import ThunderContext, RegressionModel, PCA, export


if __name__ == "__main__":
    parser = optparse.OptionParser(description="fit a regression model",
                                   usage="%prog datafile modelfile outputdir [options]")
    parser.add_option("--regressmode", choices=("mean", "linear", "bilinear"), help="form of regression")
    parser.add_option("--k", type=int, default=2)

    opts, args = parser.parse_args()
    try:
        datafile = args[0]
        modelfile = args[1]
        outputdir = args[2]
    except IndexError:
        parser.print_usage()
        raise Exception("too few arguments")

    tsc = ThunderContext.start(appName="regresswithpca")

    data = tsc.loadSeries(datafile)
    model = RegressionModel.load(modelfile, opts.regressmode)  # do regression
    betas, stats, resid = model.fit(data)
    pca = PCA(opts.k).fit(betas)  # do PCA
    traj = model.fit(data, pca.comps)  # get trajectories

    outputdir += "-regress"
    export(pca.comps, outputdir, "comps", "matlab")
    export(pca.latent, outputdir, "latent", "matlab")
    export(pca.scores, outputdir, "scores", "matlab")
    export(traj, outputdir, "traj", "matlab")