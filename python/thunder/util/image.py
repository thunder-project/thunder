# utilities for image processing

from thunder.util.signal import clip

def mapToNeighborhood(ind,ts,sz,mxX,mxY):
	# create a list of key value pairs with multiple shifted copies
	# of the time series ts
	rngX = range(-sz,sz+1,1)
	rngY = range(-sz,sz+1,1)
	out = list()
	for x in rngX :
		for y in rngY :
			newX = clip(ind[0] + x,1,mxX)
			newY = clip(ind[1] + y,1,mxY)
			newind = (newX, newY, ind[2])
			out.append((newind,ts))
	return out