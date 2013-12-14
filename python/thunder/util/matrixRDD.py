"""
Class for doing matrix operations on RDDs of (int, ndarray) pairs
"""

import sys
from numpy import dot, isclose, outer, shape, ndarray, mean, add, subtract, multiply, zeros, std, divide
from pyspark.accumulators import AccumulatorParam


def MatrixSumIteratorSelf(iterator):
    yield sum(outer(x, x) for x in iterator)


def MatrixSumIteratorOther(iterator):
    yield sum(outer(x, y) for x, y in iterator)


class MatrixAccumulatorParam(AccumulatorParam):
    def zero(self, value):
        return zeros(shape(value))

    def addInPlace(self, val1, val2):
        val1 += val2
        return val1


class MatrixRDD(object):
    def __init__(self, rdd, n=None, d=None):
        self.rdd = rdd
        if n is None:
            self.n = rdd.count()
        else:
            self.n = n
        if d is None:
            vec = rdd.first()[1]
            if type(vec) is ndarray:
                self.d = len(vec)
            else:
                self.d = 1
        else:
            self.d = d

    def collect(self):
        """
        Collect the rows of the matrix
        """
        return self.rdd.map(lambda (k, v): v).collect()

    def first(self):
        """
        Get the first row of the matrix
        """
        return self.rdd.first()[1]

    def cov(self, axis=None):
        """
        Compute a covariance matrix
        Optional argument: "axis" for mean subtraction, 0 (rows) or 1 (columns)
        """
        if axis is None:
            return self.outer() / self.n
        else:
            return self.center(axis).outer() / self.n

    def outer(self, method="reduce"):
        """
        Compute outer product of the MatrixRDD with itself
        Method: "reduce" (use a reducer) or "accum" (use an accumulator)
        """
        if method is "reduce":
            return self.rdd.map(lambda (k, v): v).mapPartitions(MatrixSumIteratorSelf).sum()
        if method is "accum":
            global mat
            mat = self.rdd.context.accumulator(zeros((self.d, self.d)), MatrixAccumulatorParam())

            def outerSum(x):
                global mat
                mat += outer(x, x)
            self.rdd.map(lambda (k, v): v).foreach(outerSum)
            return mat.value
        else:
            raise Exception("method must be reduce or accum")

    def times(self, other, method="reduce"):
        """
        Multiply a MatrixRDD by another matrix
        other: MatrixRDD, scalar, or numpy array
        method: "reduce" (use a reducer) or "accum" (use an accumulator)
        """
        dtype = type(other)
        if dtype == MatrixRDD:
            if self.n != other.n:
                raise Exception(
                    "cannot multiply shapes ("+str(self.n)+","+str(self.d)+") and ("+str(other.n)+","+str(other.d)+")")
            else:
                if method is "reduce":
                    return self.rdd.join(other.rdd).map(lambda (k, v): v).mapPartitions(MatrixSumIteratorOther).sum()
                if method is "accum":
                    global mat
                    mat = self.rdd.context.accumulator(zeros((self.d, other.d)), MatrixAccumulatorParam())

                    def outerSum(x):
                        global mat
                        mat += outer(x[0], x[1])
                    self.rdd.join(other.rdd).map(lambda (k, v): v).foreach(outerSum)
                    return mat.value
                else:
                    raise Exception("method must be reduce or accum")
        else:
            # TODO: check size of array, broadcast if too big
            if dtype == ndarray:
                dims = shape(other)
                if (len(dims) == 1 and sum(isclose(dims, self.d) == 0)) or (len(dims) == 2 and dims[0] != self.d):
                    raise Exception(
                        "cannot multiply shapes ("+str(self.n)+","+str(self.d)+") and " + str(dims))
                if len(dims) == 0:
                    new_d = 1
                else:
                    new_d = dims[0]
            return MatrixRDD(self.rdd.mapValues(lambda x: dot(x, other)), self.n, new_d)

    def elementwise(self, other, op):
        """
        Apply elementwise operation to a MatrixRDD
        other: MatrixRDD, scalar, or numpy array
        op: binary operator, e.g. add, subtract
        """
        dtype = type(other)
        if dtype is MatrixRDD:
            if (self.n is not other.n) | (self.d is not other.d):
                print >> sys.stderr, \
                    "cannot do elementwise operation for shapes ("+self.n+","+self.d+") and ("+other.n+","+other.d+")"
            else:
                return MatrixRDD(self.rdd.join(other.rdd).mapValues(lambda (x, y): op(x, y)), self.n, self.d)
        else:
            if dtype is ndarray:
                dims = shape(other)
                if len(dims) > 1 or dims[0] is not self.d:
                    raise Exception(
                        "cannot do elementwise operation for shapes ("+str(self.n)+","+str(self.d)+") and " + str(dims))
            return MatrixRDD(self.rdd.mapValues(lambda x: op(x, other)), self.n, self.d)

    def plus(self, other):
        """
        Elementwise addition (see elementwise)
        """
        return MatrixRDD.elementwise(self, other, add)

    def minus(self, other):
        """
        Elementwise subtraction (see elementwise)
        """
        return MatrixRDD.elementwise(self, other, subtract)

    def dottimes(self, other):
        """
        Elementwise multiplcation (see elementwise)
        """
        return MatrixRDD.elementwise(self, other, multiply)

    def dotdivide(self, other):
        """
        Elementwise division (see elementwise)
        """
        return MatrixRDD.elementwise(self, other, divide)

    def center(self, axis=0):
        """
        Center a MatrixRDD by mean subtraction
        axis: center rows (0) or columns (1)
        """
        if axis is 0:
            return MatrixRDD(self.rdd.mapValues(lambda x: x - mean(x)), self.n, self.d)
        if axis is 1:
            meanVec = self.rdd.map(lambda (k, v): v).mean()
            return self.minus(meanVec)
        else:
            raise Exception("axis must be 0 or 1")

    def zscore(self, axis=0):
        """
        zscore a MatrixRDD by mean subtraction and division by standard deviation
        axis: center rows (0) or columns (1)
        """
        if axis is 0:
            return MatrixRDD(self.rdd.mapValues(lambda x: (x - mean(x))/std(x)), self.n, self.d)
        if axis is 1:
            meanVec = self.rdd.map(lambda (k, v): v).mean()
            stdVec = self.rdd.map(lambda (k, v): v).std()
            return self.minus(meanVec).dotdivide(stdVec)
        else:
            raise Exception("axis must be 0 or 1")


    # TODO: right divide and left divide

    # TODO: common operation is multiplying an RDD by its transpose times a matrix, how do this cleanly?