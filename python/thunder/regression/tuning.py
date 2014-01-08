import os
import argparse
import glob
from thunder.regression.util import RegressionModel, TuningModel
from thunder.util.dataio import parse, saveout
from pyspark import SparkContext


def tuning(data, tuningmodelfile, tuningmode, regressmodelfile=None, regressmode=None):
    """estimate parameters of a tuning curve model,
    optionally preceeded by regression

    arguments:
    data - RDD of data points
    tuningmodelfile - model parameters (string with file location, array, or tuple)
    tuningmode - form of tuning ("gaussian" or "circular")
    regressmodelfile - model parameters (default=None)
    regressmode - form of regression ("linear" or "bilinear") (default=None)

    returns:
    params - tuning curve parameters
    """
    # create tuning model
    tuningmodel = TuningModel.load(tuningmodelfile, tuningmode)

    # get tuning curves
    if regressmodelfile is not None:
        # use regression results
        regressmodel = RegressionModel.load(regressmodelfile, regressmode)
        betas, stats, resid = regressmodel.fit(data)
        params = tuningmodel.fit(betas)
    else:
        # use data
        params = tuningmodel.fit(data)

    return params

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="fit a parametric tuning curve to regression results")
    parser.add_argument("master", type=str)
    parser.add_argument("datafile", type=str)
    parser.add_argument("tuningmodelfile", type=str)
    parser.add_argument("outputdir", type=str)
    parser.add_argument("tuningmode", choices=("circular", "gaussian"), help="form of tuning curve")
    parser.add_argument("--preprocess", choices=("raw", "dff", "sub"), default="raw", required=False)
    parser.add_argument("--regressmodelfile", type=str)
    parser.add_argument("--regressmode", choices=("linear", "bilinear"), help="form of regression")

    args = parser.parse_args()
    egg = glob.glob(os.environ['THUNDER_EGG'] + "*.egg")
    sc = SparkContext(args.master, "tuning", pyFiles=egg)
    lines = sc.textFile(args.datafile)
    data = parse(lines, args.preprocess).cache()

    params = tuning(data, args.tuningmodelfile, args.tuningmode, args.regressmodelfile, args.regressmode)

    outputdir = args.outputdir + "-tuning"
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)

    saveout(params, outputdir, "params", "matlab")
