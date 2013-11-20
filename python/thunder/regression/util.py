from scipy.io import * 
from numpy import *
from scipy.linalg import *

def regressionModel(modelFile,regressionMode) :
	
	class model : pass

	if regressionMode == 'mean' :
		X = loadmat(modelFile + "_X.mat")['X']
		X = X.astype(float)
		model.X = X

	if regressionMode == 'linear' :
		X = loadmat(modelFile + "_X.mat")['X']
		X = concatenate((ones((1,shape(X)[1])),X))
		X = X.astype(float)
		g = loadmat(modelFile + "_g.mat")['g']
		g = g.astype(float)[0]
		Xhat = dot(inv(dot(X,transpose(X))),X)
		model.X = X
		model.Xhat = Xhat
		model.g = g
		model.nG = len(unique(model.g))

	if regressionMode == 'bilinear' :
		X1 = loadmat(modelFile + "_X1.mat")['X1']
		X2 = loadmat(modelFile + "_X2.mat")['X2']
		X1hat = dot(inv(dot(X1,transpose(X1))),X1)
		model.X1 = X1
		model.X2 = X2
		model.X1hat = X1hat

	model.regressionMode = regressionMode

	return model

def tuningModel(modelFile,tuningMode) :

	class model : pass

	s = loadmat(modelFile + "_s.mat")['s']
	model.s = s
	model.tuningMode = tuningMode

	return model

def regressionFit(y,model) :

	if model.regressionMode == 'mean' :
		b = dot(model.X,y)
		return b

	if model.regressionMode == 'linear' :
		b = dot(model.Xhat,y)
		predic = dot(b,model.X)
		sse = sum((predic-y) ** 2)
		sst = sum((y-mean(y)) ** 2)
		r2 = 1 - sse/sst
		return (b[1:],r2)

		# if model.outputMode == 'stats'
		# 	r2shuffle = zeros((model.nRnd,)) 
		# 	X = copy(model.X)
		# 	m = shape(X)[1]
		# 	for iShuf in range(0,model.nRnd) :
		# 		for ix in range(0,shape(X)[0]) :
		# 			shift = int(round(random.rand(1)*m))
		# 			X[ix,:] = roll(X[ix,:],shift)
		# 		b = lstsq(transpose(X),y)[0]
		# 		predic = dot(b,X)
		# 		sse = sum((predic-y) ** 2)
		# 		r2shuffle[iShuf] = 1 - sse/sst
		# 	p = sum(r2shuffle > r2) / model.nRnd
		# 	return p

	if model.regressionMode == 'bilinear' :
		b1 = dot(model.X1hat,y)
		b1 = b1 - min(b1)
		b1hat = dot(transpose(model.X1),b1)
		if sum(b1hat) == 0 :
			b1hat = b1hat + 0.001
		X3 = model.X2 * b1hat
		X3 = concatenate((ones((1,shape(X3)[1])),X3))
		X3hat = dot(inv(dot(X3,transpose(X3))),X3)
		b2 = dot(X3hat,y)
		predic = dot(b2,X3)
		sse = sum((predic-y) ** 2)
		sst = sum((y-mean(y)) ** 2)
		r2 = 1 - sse/sst

		return (b1,b2[1:],r2)

def tuningFit(y,model) :
	
	if model.tuningMode == 'circular' :
		y = y - min(y)
		y = y/sum(y)
		r = inner(y,exp(1j*model.s))
		mu = angle(r)
		v = absolute(r)/sum(y)
		if v < 0.53 :
			k = 2*v + (v**3) + 5*(v**5)/6
		elif (v>=0.53) & (v<0.85) :
			k = -.4 + 1.39*v + 0.43/(1-v)
		else :
			k = 1/(v**3 - 4*(v**2) + 3*v)
		return (mu,k)

	if model.tuningMode == 'gaussian' :
		y[y<0] = 0
		y = y/sum(y)
		mu = dot(model.s,y)
		sigma = dot(y,(model.s-mu)**2)
		return (mu,sigma)
