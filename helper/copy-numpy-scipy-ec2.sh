#!/bin/bash

SLAVES=`cat /root/spark-ec2/slaves`

echo "installing numpy and scipy on master"
sudo yum -y -q install numpy scipy

echo "installing numpy and scipy on slaves"
for slave in $SLAVES; do
    echo "installing numpy on $slave"
    ssh -t -t $slave sudo yum -y -q install numpy scipy
done
