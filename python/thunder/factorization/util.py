from numpy import *
from scipy.linalg import *

def svd1(data,k,meanSubtract=1) :

	n = data.count()
	if meanSubtract == 1 :
		cov = data.map(lambda x : outer(x-mean(x),x-mean(x))).reduce(lambda x,y : (x + y)) / n
	else :
		cov = data.map(lambda x : outer(x,x)).reduce(lambda x,y : (x + y)) / n
	w, v = eig(cov)
	w = real(w)
	v = real(v)
	inds = argsort(w)[::-1]
	latent = w[inds[0:k]]
	comps = transpose(v[:,inds[0:k]])
	if meanSubtract == 1 :
		scores = data.map(lambda x : inner(x-mean(x),comps))
	else :
		scores = data.map(lambda x : inner(x,comps))

	return comps, latent, scores

# TODO: svd with alternating least squares when d is large
#def svd2(data,k,meanSubtract=1) :
