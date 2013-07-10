# cca <master> <inputFile> <outputFile> <slices> <label1> <label2> <k> <c>
# 
# compute canonical correlations on two data sets
# input is a local text file or a file in HDFS
# format should be rows of ' ' separated values
# first entry in each row indicates which dataset
# - example: space (rows) x time (cols)
# - rows should be whichever dim is larger
# 'label1' and 'label2' specify the datasets
# 'k' is number of principal components for dim reduction
# 'c' is the number of canonical correlations to return
# writes components to text (in input space)


import sys
import os
from numpy import *
from scipy.linalg import *
from pyspark import SparkContext

if len(sys.argv) < 9:
  print >> sys.stderr, \
  "(ica) usage: cca <master> <inputFile> <outputFile> <slices> <label1> <label2> <k> <c>"
  exit(-1)

def parseVector(line):
    return array([float(x) for x in line.split(' ')])

# parse inputs
sc = SparkContext(sys.argv[1], "cca")
inputFile = str(sys.argv[2]);
outputFile = str(sys.argv[3])
slices = int(sys.argv[4])
label1 = int(sys.argv[5])
label2 = int(sys.argv[6])
k = int(sys.argv[7])
c = int(sys.argv[8])
if not os.path.exists(outputFile):
    os.makedirs(outputFile)

# load data and split according to label
print("(cca) loading data...")
lines = sc.textFile(inputFile)
data = lines.map(parseVector)
data1 = data.filter(lambda x : x[0] == label1).cache()
data2 = data.filter(lambda x : x[0] == label2).cache()

# get dims
n1 = data1.count()
n2 = data2.count()
m1 = len(data1.first())-1
m2 = len(data2.first())-1

# remove label
data1 = data1.map(lambda x : x[1:m1+1])
data2 = data2.map(lambda x : x[1:m2+1])

# remove means
print("(cca) mean subtraction")
data1mean = data1.reduce(lambda x,y : x+y) / n1
data1sub = data1.map(lambda x : x - data1mean)
data2mean = data2.reduce(lambda x,y : x+y) / n2
data2sub = data2.map(lambda x : x - data2mean)

# do dimensionality reduction
print("(cca) reducing dimensionality")
cov1 = data1sub.map(lambda x : outer(x,x)).reduce(lambda x,y : (x + y)) / (n1 - 1)
w1, v1 = eig(cov1)
v1 = v1[:,argsort(w1)[::-1]];
cov2 = data2sub.map(lambda x : outer(x,x)).reduce(lambda x,y : (x + y)) / (n1 - 1)
w2, v2 = eig(cov2)
v2 = v2[:,argsort(w2)[::-1]];

# mean subtract inputs to cca
x1 = v1[:,0:k]
x2 = v2[:,0:k]
x1 = x1-mean(x1,axis=0)
x2 = x2-mean(x2,axis=0)

# do cca
print("(cca) computing canonical correlations")
q1,r1,p1 = qr(x1,mode='economic',pivoting=True)
q2,r2,p2 = qr(x2,mode='economic',pivoting=True)
l,d,m = svd(dot(transpose(q1),q2))
A = lstsq(r1,l * sqrt(n1-1))[0]
B = lstsq(r2,transpose(m) * sqrt(n2-1))[0]
A = A[argsort(p1)[::1],:]
B = B[argsort(p2)[::1],:]

# write output
print("(cca) writing results...")
for ic in range(0,c):
	time1 = dot(v1[:,0:k],A[:,ic])
	out1 = data1sub.map(lambda x : inner(x,dot(v1[:,0:k],A[:,ic])))
	savetxt(outputFile+"/"+"out-label-"+str(label1)+"-cc-"+str(ic)+"-"+outputFile+".txt",out1.collect(),fmt='%.8f')
	savetxt(outputFile+"/"+"out-label-"+str(label1)+"-time-"+str(ic)+"-"+outputFile+".txt",time1,fmt='%.8f')
	time2 = dot(v2[:,0:k],B[:,ic])
	out2 = data2sub.map(lambda x : inner(x,dot(v2[:,0:k],B[:,ic])))
	savetxt(outputFile+"/"+"out-label-"+str(label2)+"-cc-"+str(ic)+"-"+outputFile+".txt",out2.collect(),fmt='%.8f')
	savetxt(outputFile+"/"+"out-label-"+str(label2)+"-time-"+str(ic)+"-"+outputFile+".txt",time2,fmt='%.8f')


