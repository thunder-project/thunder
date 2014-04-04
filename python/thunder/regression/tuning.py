import os
import argparse
import glob
from thunder.regression.util import RegressionModel, TuningModel
from thunder.util.load import load
from thunder.util.save import save
from pyspark import SparkContext


def tuning(data, tuningmodelfile, tuningmode, regressmodelfile=None, regressmode=None):
    """Estimate parameters of a tuning curve model,
    optionally preceeded by regression

    :param data: RDD of data points as key value pairs
    :param tuningmodelfile: model parameters for tuning (string with file location, array, or tuple)
    :param: tuningmode: form of tuning ("gaussian" or "circular")
    :param regressmodelfile: model parameters for regression (default=None)
    :param regressmode: form of regression ("linear" or "bilinear") (default=None)

    :return params: tuning curve parameters
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
    parser.add_argument("--preprocess", choices=("raw", "dff", "dff-highpass", "sub"), default="raw", required=False)
    parser.add_argument("--regressmodelfile", type=str)
    parser.add_argument("--regressmode", choices=("linear", "bilinear"), help="form of regression")

    args = parser.parse_args()
    
    sc = SparkContext(args.master, "tuning")

    if args.master != "local":
        egg = glob.glob(os.path.join(os.environ['THUNDER_EGG'], "*.egg"))
        sc.addPyFile(egg[0])

    data = load(sc, args.datafile, args.preprocess).cache()

    params = tuning(data, args.tuningmodelfile, args.tuningmode, args.regressmodelfile, args.regressmode)

    outputdir = args.outputdir + "-tuning"

    save(params, outputdir, "params", "matlab")
