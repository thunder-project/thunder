"""
Example standalone app for kmeans clustering
"""

import optparse
from thunder import ThunderContext, KMeans

if __name__ == "__main__":
    parser = optparse.OptionParser(description="do kmeans clustering",
                                   usage="%prog datafile outputdir k [options]")
    parser.add_option("--maxiter", type=float, default=20)
    parser.add_option("--tol", type=float, default=0.001)
    opts, args = parser.parse_args()
    try:
        datafile = args[0]
        outputdir = args[1]
        k = int(args[2])
    except IndexError:
        parser.print_usage()
        raise Exception("too few arguments")

    tsc = ThunderContext.start(appName="kmeans")

    data = tsc.loadSeries(datafile).cache()
    model = KMeans(k=k, maxIterations=opts.maxiter).fit(data)
    labels = model.predict(data)

    outputdir += "-kmeans"
    tsc.export(model.centers, outputdir, "centers", "matlab")
    tsc.export(labels, outputdir, "labels", "matlab")
