"""
class for doing matrix operations on RDDs of (int, ndarray) pairs
(experimental!)

TODO: right divide and left divide
TODO: common operation is multiplying an RDD by its transpose times a matrix, how do this cleanly?
TODO: test using these in the various analyses packages (especially thunder.factorization)
"""

import sys
from numpy import dot, allclose, outer, shape, ndarray, mean, add, subtract, multiply, zeros, std, divide
from pyspark.accumulators import AccumulatorParam


def matrixsum_iterator_self(iterator):
    yield sum(outer(x, x) for x in iterator)


def matrixsum_iterator_other(iterator):
    yield sum(outer(x, y) for x, y in iterator)


class MatrixAccumulatorParam(AccumulatorParam):
    def zero(self, value):
        return zeros(shape(value))

    def addInPlace(self, val1, val2):
        val1 += val2
        return val1


class RowMatrix(object):
    def __init__(self, rdd, nrows=None, ncols=None):
        self.rdd = rdd
        if nrows is None:
            self.nrows = rdd.count()
        else:
            self.nrows = nrows
        if ncols is None:
            vec = rdd.first()[1]
            if type(vec) is ndarray:
                self.ncols = len(vec)
            else:
                self.ncols = 1
        else:
            self.ncols = ncols

    def collect(self):
        """
        collect the rows of the matrix
        """
        return self.rdd.map(lambda (k, v): v).collect()

    def first(self):
        """
        get the first row of the matrix
        """
        return self.rdd.first()[1]

    def cov(self, axis=None):
        """
        compute a covariance matrix

        arguments:
        axis - axis for mean subtraction, 0 (rows) or 1 (columns)
        """
        if axis is None:
            return self.outer() / self.nrows
        else:
            return self.center(axis).outer() / self.nrows

    def outer(self, method="reduce"):
        """
        compute outer product of the MatrixRDD with itself

        arguments:
        method - "reduce" (use a reducer) or "accum" (use an accumulator)
        """
        if method is "reduce":
            return self.rdd.map(lambda (k, v): v).mapPartitions(matrixsum_iterator_self).sum()

        if method is "accum":
            global mat
            mat = self.rdd.context.accumulator(zeros((self.ncols, self.ncols)), MatrixAccumulatorParam())

            def outerSum(x):
                global mat
                mat += outer(x, x)
            self.rdd.map(lambda (k, v): v).foreach(outerSum)
            print(mat.value)
            return mat.value

        if method is "aggregate":

            def seqop(x, v):
                return x + outer(v, v)

            def combop(x, y):
                x += y
                return x

            return self.rdd.map(lambda (_, v): v).aggregate(zeros((self.ncols, self.ncols)), seqop, combop)

        else:
            raise Exception("method must be reduce or accum")

    def times(self, other, method="reduce"):
        """
        Multiply a MatrixRDD by another matrix

        arguments:
        other - MatrixRDD, scalar, or numpy array
        method - "reduce" (use a reducer) or "accum" (use an accumulator)
        """
        dtype = type(other)
        if dtype == RowMatrix:
            if self.nrows != other.nrows:
                raise Exception(
                    "cannot multiply shapes ("+str(self.nrows)+","+str(self.ncols)+") and ("+str(other.nrows)+","+str(other.ncols)+")")
            else:
                if method is "reduce":
                    return self.rdd.join(other.rdd).map(lambda (k, v): v).mapPartitions(matrixsum_iterator_other).sum()
                if method is "accum":
                    global mat
                    mat = self.rdd.context.accumulator(zeros((self.ncols, other.ncols)), MatrixAccumulatorParam())

                    def outersum(x):
                        global mat
                        mat += outer(x[0], x[1])
                    self.rdd.join(other.rdd).map(lambda (k, v): v).foreach(outersum)
                    return mat.value
                else:
                    raise Exception("method must be reduce or accum")
        else:
            # TODO: check size of array, broadcast if too big
            if dtype == ndarray:
                dims = shape(other)
                if (len(dims) == 1 and sum(allclose(dims, self.ncols) == 0)) or (len(dims) == 2 and dims[0] != self.ncols):
                    raise Exception(
                        "cannot multiply shapes ("+str(self.nrows)+","+str(self.ncols)+") and " + str(dims))
                if len(dims) == 0:
                    new_d = 1
                else:
                    new_d = dims[0]
            return RowMatrix(self.rdd.mapValues(lambda x: dot(x, other)), self.nrows, new_d)

    def elementwise(self, other, op):
        """
        apply elementwise operation to a MatrixRDD

        arguments:
        other - MatrixRDD, scalar, or numpy array
        op - binary operator, e.g. add, subtract
        """
        dtype = type(other)
        if dtype is RowMatrix:
            if (self.nrows is not other.nrows) | (self.ncols is not other.ncols):
                print >> sys.stderr, \
                    "cannot do elementwise op for shapes ("+self.nrows+","+self.ncols+") and ("+other.nrows+","+other.ncols+")"
            else:
                return RowMatrix(self.rdd.join(other.rdd).mapValues(lambda (x, y): op(x, y)), self.nrows, self.ncols)
        else:
            if dtype is ndarray:
                dims = shape(other)
                if len(dims) > 1 or dims[0] is not self.ncols:
                    raise Exception(
                        "cannot do elementwise operation for shapes ("+str(self.nrows)+","+str(self.ncols)+") and " + str(dims))
            return RowMatrix(self.rdd.mapValues(lambda x: op(x, other)), self.nrows, self.ncols)

    def plus(self, other):
        """
        elementwise addition (see elementwise)
        """
        return RowMatrix.elementwise(self, other, add)

    def minus(self, other):
        """
        elementwise subtraction (see elementwise)
        """
        return RowMatrix.elementwise(self, other, subtract)

    def dottimes(self, other):
        """
        elementwise multiplcation (see elementwise)
        """
        return RowMatrix.elementwise(self, other, multiply)

    def dotdivide(self, other):
        """
        elementwise division (see elementwise)
        """
        return RowMatrix.elementwise(self, other, divide)

    def center(self, axis=0):
        """
        center a MatrixRDD by mean subtraction

        arguments:
        axis - center rows (0) or columns (1)
        """
        if axis is 0:
            return RowMatrix(self.rdd.mapValues(lambda x: x - mean(x)), self.nrows, self.ncols)
        if axis is 1:
            meanVec = self.rdd.map(lambda (k, v): v).mean()
            return self.minus(meanVec)
        else:
            raise Exception("axis must be 0 or 1")

    def zscore(self, axis=0):
        """
        zscore a MatrixRDD by mean subtraction and division by standard deviation

        arguments:
        axis - center rows (0) or columns (1)
        """
        if axis is 0:
            return RowMatrix(self.rdd.mapValues(lambda x: (x - mean(x))/std(x)), self.nrows, self.ncols)
        if axis is 1:
            meanvec = self.rdd.map(lambda (k, v): v).mean()
            stdvec = self.rdd.map(lambda (k, v): v).std()
            return self.minus(meanvec).dotdivide(stdvec)
        else:
            raise Exception("axis must be 0 or 1")

