import os
import argparse
import glob
from thunder.classification.util import MassUnivariateClassifier
from thunder.util.load import load
from thunder.util.save import save
from pyspark import SparkContext


def classify(data, params, classifymode, featureset=None, cv=0):
    """Perform mass univariate classification

    :param data: RDD of data points as key value pairs
    :param params: string with file location, or dictionary of parameters for classification
    :param classifymode: form of classifier ("naivebayes")
    :param featureset: set of features to use for classification (default=None)
    :param cv: number of cross validation folds (default=0, for no cv)

    :return perf: performance
    """
    # create classifier
    clf = MassUnivariateClassifier.load(params, classifymode, cv)

    # do classification
    perf = clf.classify(data, featureset)

    return perf


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="fit a regression model")
    parser.add_argument("master", type=str)
    parser.add_argument("datafile", type=str)
    parser.add_argument("paramfile", type=str)
    parser.add_argument("outputdir", type=str)
    parser.add_argument("classifymode", choices="naivebayes", help="form of classifier")
    parser.add_argument("--preprocess", choices=("raw", "dff", "dff-highpass", "sub"), default="raw", required=False)

    args = parser.parse_args()

    sc = SparkContext(args.master, "classify")

    if args.master != "local":
        egg = glob.glob(os.path.join(os.environ['THUNDER_EGG'], "*.egg"))
        sc.addPyFile(egg[0])

    data = load(sc, args.datafile, args.preprocess)

    perf = classify(data, args.paramfile, args.classifymode)

    outputdir = args.outputdir + "-classify"

    save(perf, outputdir, "perf", "matlab")
