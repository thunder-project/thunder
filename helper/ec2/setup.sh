#!/bin/bash

SLAVES=`cat /root/spark-ec2/slaves`

echo "Installing dependencies on master"
sudo yum -y -q install numpy scipy python-imaging

echo "Installing dependencies on slaves"
for slave in $SLAVES; do
    echo "Installing numpy on $slave"
    ssh -t -t $slave sudo yum -y -q install numpy scipy
done


