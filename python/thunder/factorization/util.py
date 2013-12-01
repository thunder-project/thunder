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
# def svd2(data,k,meanSubtract=1) :

def svd2(data,k,meanSubtract=1) :
	def randomVector(ind,seed,k) :
		random.seed(ind*100000*seed)
		r = random.rand(1,k)
		return r

	m = len(data.first()[1])

	v = data.map(lambda x,y : randomVector(x,seed1,k))
	u = random.randn(k,m)
	nIter = 5
	iIter = 1
	# fixed initialization
	#var u = factory2D.make(sc.textFile("data/h0.txt").map(parseLine _).toArray())
	#var w = sc.textFile("data/w0.txt").map(parseVector2 _)
	while (iter < nIter) :
		# goal is to solve R = VU subject to U,V > 0
		# by iteratively updating U and V with least squares and clipping
    
		# precompute inv(V' * V)
		vinv = inv(v.map(lambda x : outer(x,x)).reduce(lambda x,y : (x+y)))

		# update U using least squares row-wise with inv(V' * V) * V * R (same as pinv(V) * R)
		u = data.join(v.map(lambda x : dot(vinv,x))).mapValues(lambda x,y : outer(x,y)).reduce(lambda x,y: x + y)

		# precompute pinv(U)
		uinv = tranpose(inv(transpose(u)))

		# update V using least squares row-wise with R * pinv(U)
		v = data.mapValues(lambda x : dot(transpose(uinv),x))
    
		iter += 1

