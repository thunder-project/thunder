#!/bin/bash

cd thunder/python
./setup.py bdist_egg
cd ../..

echo "export PYTHONPATH=/root/thunder/python/" >> /root/.bash_profile
echo "alias pyspark="/root/spark/bin/pyspark"" >> /root/.bash_profile
echo "export THUNDER_EGG=/root/thunder/python/dist/" >> /root/.bash_profile