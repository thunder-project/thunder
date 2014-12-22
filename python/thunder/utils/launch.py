"""Functions used in the Thunder launch scripts in python/bin"""
import glob
import os
import subprocess

import thunder


def getSparkHome():
    sparkhome = os.getenv("SPARK_HOME")
    if sparkhome is None:
        raise Exception("The environment variable SPARK_HOME must be set to the Spark installation directory")
    return sparkhome


def getMasterURI(optsDict):
    if "master" in optsDict:
        master = optsDict["master"]
    elif "MASTER" in os.environ:
        master = os.getenv("MASTER")
    elif isEC2():
        master = getEC2Master()
    else:
        master = "local"
    return master


def isEC2():
    return os.path.isfile('/root/spark-ec2/cluster-url')


def getEC2Master():
    """Returns the cluster master URI, read from /root/spark-ec2/cluster-url.

    This file is expected to exist on EC2 clusters. An exception will be thrown if the file is missing.
    """
    with open('/root/spark-ec2/cluster-url', 'r') as f:
        master = f.read().strip()
    return master


def findThunderEgg():
    # get directory
    calldir = os.path.dirname(os.path.realpath(__file__))
    distdir = os.path.join(calldir, '..', '..', 'dist')

    # check for egg
    egg = glob.glob(os.path.join(distdir, "thunder_python-" + str(thunder.__version__) + "*.egg"))
    if len(egg) == 1:
        return egg[0]
    if not egg:
        return None
    raise Exception("Multiple Thunder .egg files found - please run 'python setup.py clean bdist_egg' to rebuild " +
                    "the appropriate version. Found: %s" % egg)


def buildThunderEgg():
    calldir = os.path.dirname(os.path.realpath(__file__))
    pythondir = os.path.join(calldir, '..', '..')
    subprocess.check_call(["python", 'setup.py', 'clean', 'bdist_egg'], cwd=pythondir)
    return findThunderEgg()


def findThunderJar():
    thunderdir = os.path.dirname(os.path.realpath(thunder.__file__))
    thunderJar = os.path.join(thunderdir, 'lib', 'thunder_2.10-'+str(thunder.__version__)+'.jar')
    if not os.path.isfile(thunderJar):
        raise Exception("Thunder jar file not found at '%s'. Does Thunder need to be rebuilt?")
    return thunderJar


def getCommaSeparatedOptionsList(childOptionsFlag, commaSeparated, additionalDefault=None):
    lst = commaSeparated.split(",") if commaSeparated else []
    if additionalDefault:
        lst.append(additionalDefault)
    if lst:
        return [childOptionsFlag, ",".join(lst)]
    return []


def transformArguments(args):
    """Modifies command line arguments passed to the thunder launch scripts
    by adding required jar and egg files (if not already present).

    The passed arguments will be modified as follows and returned in a new list:
    1. the first element of args will be dropped (assumption is that this is sys.argv[0],
    the program name, which the caller is responsible for replacing.

    Parameters
    ----------
    args: sequence of strings
        Sequence of string arguments, as would obtained from sys.argv

    Returns
    -------
        Sequence of string arguments
    """

    # first pass through arguments, looking for a few specific flags
    opts = {}
    passthruArgs = []
    if len(args) > 1:
        argIter = iter(args[1:])
        try:
            arg = argIter.next()
            larg = arg.lower().lstrip("-")
            if larg in frozenset(["master", "py-files", "jars"]):
                opts[larg] = argIter.next()
            else:
                passthruArgs.append(arg)
        except StopIteration:
            pass

    master = getMasterURI(opts)
    if "local" not in master:
        thunderEgg = findThunderEgg()
        if not thunderEgg:
            thunderEgg = buildThunderEgg()
            if not thunderEgg:
                raise Exception("Unable to find or rebuild thunder .egg file.")
    else:
        thunderEgg = None

    thunderJar = findThunderJar()

    pyFiles = getCommaSeparatedOptionsList("--py-files", opts.get("py-files", []), thunderEgg)
    jars = getCommaSeparatedOptionsList("--jars", opts.get("jars", []), thunderJar)

    retVals = ["--master", master]
    retVals.extend(pyFiles)
    retVals.extend(jars)
    retVals.extend(passthruArgs)

    return passthruArgs
