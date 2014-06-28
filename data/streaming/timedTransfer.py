#!/usr/bin/env python

#
# test a streaming app by dumping files from one directory
# into another, at a specified rate
#
# <streaming_test> srcPath targetPath waitTime
#
#

import sys, os, time, glob;

srcPath = str(sys.argv[1])
targetPath = str(sys.argv[2])
targetPathTmp = os.path.dirname(targetPath) + "_tmp"

# create temporary folder
if os.path.exists(targetPathTmp):
	os.rmdir(targetPathTmp)
os.mkdir(targetPathTmp)

# delete existing files in target folder
files = sorted(glob.glob(targetPath+"*.bin"))
for f in files:
	os.remove(f)

waitTime = float(sys.argv[3])
files = sorted(glob.glob(srcPath+"*.bin"))
count = 1
for f in files:
	cmd_cp = "scp " + f + " " + targetPathTmp 
	os.system(cmd_cp)
	print('writing file ' + str(count))
	tmpfile = os.path.join(targetPathTmp, os.path.basename(f))
	cmd_mv = "mv " + tmpfile + " " + targetPath
	os.system(cmd_mv)
	count = count + 1
	time.sleep(waitTime)

os.rmdir(targetPathTmp)