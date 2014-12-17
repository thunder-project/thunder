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


def getMasterURI():
    master = os.getenv("MASTER")
    if (not master) and isEC2():
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


def findPy4J(sparkHome):
    py4jglob = os.path.join(sparkHome, "python", "lib", "py4j-*-src.zip")
    try:
        return glob.iglob(py4jglob).next()
    except StopIteration:
        raise Exception("No py4j jar found at '"+py4jglob+"'; is SPARK_HOME set correctly?")


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


def getOptionsList(childOptionsFlag, optionVal):
    if optionVal:
        return [childOptionsFlag, str(optionVal)]
    return []


def getCommaSeparatedOptionsList(childOptionsFlag, commaSeparated, additionalDefault=None):
    lst = commaSeparated.split(",") if commaSeparated else []
    if additionalDefault:
        lst.append(additionalDefault)
    if lst:
        return [childOptionsFlag, ",".join(lst)]
    return []


def optionNameToAttribute(optName):
    return optName.lstrip("-").replace("-", "_")


def addOptionsToParser(optionParser, sparkHome):
    from optparse import OptionGroup
    optionParser.add_option("--master", default=getMasterURI(),
                            help="spark://host:port, mesos://host:port, yarn, or local (default: '%default').")
    # currently only 'client' deploy-mode is supported with python apps - and Thunder is python-only.
    # optionParser.add_option("--deploy-mode", default="client", type="choice", choices=["client", "cluster"],
    #                         help="Whether to deploy your driver on the worker nodes (cluster) or locally as an " +
    #                              "external client (client) (default: '%default').")
    # 'name' is hardcoded to 'PySparkShell' in Spark's shell.py; this param will have no effect if launching the shell:
    optionParser.add_option("--name", default="",
                            help="A name for your application (Has no effect in thunder-shell).")
    optionParser.add_option("--py-files", default="",
                            help="Comma-separated list of additional .zip, .egg, or .py files to place " +
                                 "on the PYTHONPATH for Python apps. The Thunder .egg file is automatically included.")
    optionParser.add_option("--files", default="",
                            help="Comma-separated list of files to be placed in the working directory of each executor.")
    optionParser.add_option("--jars", default="",
                            help="Comma-separated list of local jars to include on the driver and executor classpaths." +
                                 "The Thunder jar file is automatically included.")
    uncommonGroup = OptionGroup(optionParser, "Less Common Options")
    uncommonGroup.add_option(
        "--conf", action="append", type="string", nargs=2, default=[],
        help="Arbitrary Spark configuration property, specified as KEY VALUE (whitespace-separated).")
    uncommonGroup.add_option(
        "--properties-file", default=os.path.join(sparkHome, "conf", "spark-defaults.conf"),
        help="Properties file from which to load Spark properties (default: '%default').")
    # the driver-memory default is changed from the Spark default, which is only 512M, since we
    # expect to be doing lots of large collect() calls in Thunder:
    uncommonGroup.add_option(
        "--driver-memory", default="20G",
        help="Memory for driver (e.g. 1000M, 2G)(default: '%default').")
    uncommonGroup.add_option(
        "--driver-java-options", default="",
        help="Extra Java options to pass to the driver.")
    uncommonGroup.add_option(
        "--driver-library-path", default="",
        help="Extra library path entries to pass to the driver.")
    uncommonGroup.add_option(
        "--driver-class-path", default="",
        help="Extra class path entries to pass to the driver. Note that jars added with --jars are automatically " +
             "included in the classpath.")
    optionParser.add_option_group(uncommonGroup)
    # standaloneClusterGroup = OptionGroup(optionParser, "Spark standalone with cluster deploy mode only")
    # standaloneClusterGroup.add_option("--driver-cores", default=1, type="int",
    #                                   help="Cores for driver (default: %default).")
    # standaloneClusterGroup.add_option("--supervise", default=False, action="store_true",
    #                                   help="If given, restarts the driver on failure.")
    # optionParser.add_option_group(standaloneClusterGroup)
    standaloneMesosGroup = OptionGroup(optionParser, "Spark standalone and Mesos only")
    standaloneMesosGroup.add_option("--total-executor-cores", default=-1, type="int",
                                    help="Total cores for all executors.")
    optionParser.add_option_group(standaloneMesosGroup)
    yarnGroup = OptionGroup(optionParser, "YARN-only")
    yarnGroup.add_option("--executor-cores", default=1, type="int",
                         help="Number of cores per executor (default: %default).")
    yarnGroup.add_option("--queue", default="default",
                         help="The YARN queue to submit to (default: '%default').")
    yarnGroup.add_option("--num-executors", default=2, type="int",
                         help="Number of executors to launch (default: %default).")
    yarnGroup.add_option("--archives", default="",
                         help="Comma separated list of archives to be extracted into the working directory " +
                              "of each executor.")
    optionParser.add_option_group(yarnGroup)


def parseOptionsIntoChildProcessArguments(opts):
    childArgs = []

    if not ("local" in opts.master):
        thunderEgg = findThunderEgg()
        if not thunderEgg:
            thunderEgg = buildThunderEgg()
            if not thunderEgg:
                raise Exception("Unable to find or rebuild thunder .egg file.")
    else:
        thunderEgg = None

    thunderJar = findThunderJar()

    childArgs.extend(getOptionsList("--master", opts.master))
    # childArgs.extend(getOptionsList("--deploy-mode", opts.deploy_mode))
    childArgs.extend(getCommaSeparatedOptionsList("--py-files", opts.py_files, thunderEgg))
    childArgs.extend(getCommaSeparatedOptionsList("--jars", opts.jars, thunderJar))

    simpleOptions = ["--name", "--files", "--properties-file", "--driver-memory",
                     "--driver-java-options", "--driver-library-path", "--driver-class-path"]
    for simpleOpt in simpleOptions:
        optVal = getattr(opts, optionNameToAttribute(simpleOpt), None)
        if optVal:
            childArgs.extend(getOptionsList(simpleOpt, optVal))

    if opts.conf:
        for key, value in opts.conf:
            childArgs.extend(["--conf", key + "=" + value])

    # standalone in cluster deploy mode options
    # if opts.master.lower().startswith("spark") and opts.deploy_mode.lower() == "cluster":
    #     childArgs.extend(getOptionsList("--driver-cores", opts.driver_cores))
    #     if opts.supervise:
    #         childArgs.append("--supervise")

    # standalone or mesos options
    if opts.master.lower().startswith("spark") or opts.master.lower().startswith("mesos"):
        if opts.total_executor_cores > 0:
            childArgs.extend(["--total-executor-cores", str(opts.total_executor_cores)])

    if opts.master.lower().startswith("yarn"):
        yarnSimpleOptions = ["--executor-cores", "--queue", "--num-executors"]
        for simpleOpt in yarnSimpleOptions:
            optVal = getattr(opts, optionNameToAttribute(simpleOpt), None)
            if optVal:
                childArgs.extend(getOptionsList(simpleOpt, optVal))
        childArgs.extend(getCommaSeparatedOptionsList("--archives", opts.archives))

    return childArgs