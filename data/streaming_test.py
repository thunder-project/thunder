#!/usr/bin/env python

#
# test a streaming app by dumping files from one directory
# into another, at a specified rate
#

import sys, os, time, glob;

srcPath = str(sys.argv[1])
targetPath = str(sys.argv[2])
waitTime = float(sys.argv[3])
files = glob.glob(srcPath+"*")
for f in files:
	cmd = "scp " + f + " " + targetPath 
	os.system(cmd)
	time.sleep(waitTime)

