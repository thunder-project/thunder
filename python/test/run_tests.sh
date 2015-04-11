#!/usr/bin/env bash

# assumes run from python/test directory


if [ -z "$SPARK_HOME" ]; then
    echo 'You need to set $SPARK_HOME to run these tests.' >&2
    exit 1
fi

PYFORJ=`ls -1 $SPARK_HOME/python/lib/py4j-*-src.zip | head -1`
export THUNDER_JAR=`find .. -name "thunder*.jar" | head -1`
if [ -z "$THUNDER_JAR" ]; then
    echo 'Thunder jar not found; you may need to build the scala dependencies with "sbt clean package" first in order to run the rdd tests'
    # exit 1
fi

export SPARK_CLASSPATH=$THUNDER_JAR

export PYTHONPATH=$PYTHONPATH:$SPARK_HOME/python:../
export PYTHONPATH=$PYTHONPATH:$PYFORJ

export PYTHONWARNINGS="ignore"

nosetests $@ --verbosity 2 --rednose --nologcapture