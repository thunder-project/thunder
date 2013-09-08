#!/usr/bin/env python

#
# test a streaming app by dumping files from one directory
# into another, at a specified rate
#
# <streaming_test> srcPath targetPath waitTime
#
# example:
# data/streaming_test.py /groups/ahrens/ahrenslab/Misha/forJeremy_SparkStreamingSample/ /nobackup/freeman/buffer/ 1
#

import sys, os, time, glob;

srcPath = str(sys.argv[1])
targetPath = str(sys.argv[2])
waitTime = float(sys.argv[3])
files = glob.glob(srcPath+"*")
count = 1
for f in files:
	cmd = "scp " + f + " " + targetPath 
	os.system(cmd)
	print('writing file ' +str(count))
	count = count + 1
	time.sleep(waitTime)

