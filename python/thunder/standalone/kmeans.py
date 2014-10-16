"""
Example standalone app for kmeans clustering
"""

import argparse
from thunder import ThunderContext, KMeans, export

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="do kmeans clustering")
    parser.add_argument("datafile", type=str)
    parser.add_argument("outputdir", type=str)
    parser.add_argument("k", type=int)
    parser.add_argument("--maxiter", type=float, default=20, required=False)
    parser.add_argument("--tol", type=float, default=0.001, required=False)

    args = parser.parse_args()

    tsc = ThunderContext.start(appName="kmeans")

    data = tsc.loadSeries(args.datafile).cache()
    model = KMeans(k=args.k, maxIterations=args.maxiter).fit(data)
    labels = model.predict(data)

    outputdir = args.outputdir + "-kmeans"
    export(model.centers, outputdir, "centers", "matlab")
    export(labels, outputdir, "labels", "matlab")
