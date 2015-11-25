from numpy import vstack

def elementwiseMean(arys):
    from numpy import mean
    combined = vstack([ary.ravel() for ary in arys])
    meanAry = mean(combined, axis=0)
    return meanAry.reshape(arys[0].shape)


def elementwiseVar(arys):
    from numpy import var
    combined = vstack([ary.ravel() for ary in arys])
    varAry = var(combined, axis=0)
    return varAry.reshape(arys[0].shape)


def elementwiseStdev(arys):
    from numpy import std
    combined = vstack([ary.ravel() for ary in arys])
    stdAry = std(combined, axis=0)
    return stdAry.reshape(arys[0].shape)


