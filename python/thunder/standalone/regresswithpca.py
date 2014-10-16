"""
Example standalone app for mass-unvariate regression combined with PCA
"""

import argparse
from thunder import ThunderContext, RegressionModel, PCA, export


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="fit a regression model")
    parser.add_argument("datafile", type=str)
    parser.add_argument("modelfile", type=str)
    parser.add_argument("outputdir", type=str)
    parser.add_argument("regressmode", choices=("mean", "linear", "bilinear"), help="form of regression")
    parser.add_argument("--k", type=int, default=2)

    args = parser.parse_args()

    tsc = ThunderContext.start(appName="regresswithpca")

    data = tsc.loadSeries(args.datafile)
    model = RegressionModel.load(args.modelfile, args.regressmode)  # do regression
    betas, stats, resid = model.fit(data)
    pca = PCA(args.k).fit(betas)  # do PCA
    traj = model.fit(data, pca.comps)  # get trajectories

    outputdir = args.outputdir + "-regress"
    export(pca.comps, outputdir, "comps", "matlab")
    export(pca.latent, outputdir, "latent", "matlab")
    export(pca.scores, outputdir, "scores", "matlab")
    export(traj, outputdir, "traj", "matlab")