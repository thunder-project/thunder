#!/usr/bin/env python
try:
    import thunder
except ImportError:
    thunder = None
    raise Exception("Unable to import Thunder. Please make sure that the Thunder installation directory is listed in " +
                    "the PYTHONPATH environment variable.")

from thunder.utils.launchutils import *


def main():
    SPARK_HOME = getSparkHome()

    from optparse import OptionParser  # use optparse instead of argparse for python2.6 compat
    parser = OptionParser(usage="%prog [submit options]",
                          version=thunder.__version__)
    addOptionsToParser(parser, SPARK_HOME)
    opts, args = parser.parse_args()

    childArgs = parseOptionsIntoChildProcessArguments(opts)

    sparkSubmit = os.path.join(SPARK_HOME, 'bin', 'pyspark')
    childArgs = [sparkSubmit] + childArgs

    # add python script
    os.environ['PYTHONSTARTUP'] = os.path.join(os.path.dirname(os.path.realpath(thunder.__file__)), 'utils', 'shell.py')

    subprocess.call(childArgs, env=os.environ)

if __name__ == "__main__":
    main()