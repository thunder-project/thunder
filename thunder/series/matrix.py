from numpy import dot, outer, shape, ndarray, add, subtract, multiply, zeros, divide, arange

from ..series.series import Series


class Matrix(Series):
    """
    Collection of rows of a dense matrix.
    """
    @property
    def _constructor(self):
        return Matrix

    @property
    def nrows(self):
        return self.shape[0]

    @property
    def ncols(self):
        return self.shape[1]

    def cov(self, axis=None):
        """
        Compute covariance of a distributed matrix.

        Parameters
        ----------
        axis : int, optional, default = None
            Axis for performing mean subtraction, None (no subtraction), 0 (rows) or 1 (columns)
        """
        if axis is None:
            return self.gramian() / self.nrows
        else:
            return self.center(axis).gramian() / self.nrows

    def gramian(self, method="accum"):
        """
        Compute gramian of a distributed matrix.

        The product of the matrix with its transpose, i.e. A^T * A

        Parameters
        ----------
        method : string, optional, default = "accum"
            Method to use for summation
        """

        from pyspark.accumulators import AccumulatorParam

        class MatrixAccumulator(AccumulatorParam):
            def zero(self, value):
                return zeros(shape(value))

            def addInPlace(self, val1, val2):
                val1 += val2
                return val1

        if method is "reduce":
            return self.rdd.map(lambda (k, v): v).mapPartitions(outer_iterator).sum()

        elif method is "accum":
            global mat
            mat = self.rdd.context.accumulator(zeros((self.ncols, self.ncols)), MatrixAccumulator())

            def outer_sum(x):
                global mat
                mat += outer(x, x)

            self.rdd.map(lambda (k, v): v).foreach(outer_sum)
            return mat.value

        else:
            raise Exception("Method must be reduce or accum")

    def times(self, other):
        """
        Multiply a Matrix by another Matrix.

        Other matrix can be either another Matrix or a local matrix.
        NOTE: A.times(B) computes A^T * B
        NOTE: If multiplying two Matrices, they must have the same
        number of partitions and number of records per partition,
        e.g. because one was created through a map of the other,
        see zip

        Parameters
        ----------
        other : Matrix, scalar, or numpy array
            Matrix to multiply with

        method : string, optional, default = "reduce"
            Method to use for summation
        """
        dtype = type(other)
        if dtype == Matrix:
            if self.nrows != other.nrows:
                raise Exception(
                    "Cannot multiply shapes ("
                    + str(self.nrows) + "," + str(self.ncols) + ") and (" +
                    str(other.nrows) + "," + str(other.ncols) + ")")
            else:
                return self.rdd.zip(other.rdd).map(lambda ((k1, x), (k2, y)): (x, y))\
                    .mapPartitions(outer_iterator_other).sum()
        else:
            dims = shape(other)
            if dims[0] != self.ncols:
                raise Exception(
                    "Cannot multiply shapes ("
                    +str(self.nrows)+","+str(self.ncols)+") and " + str(dims))
            if len(dims) == 0:
                new_d = 1
            else:
                new_d = dims[1]
            other_b = self.rdd.context.broadcast(other)
            newindex = arange(0, new_d)
            newrdd = self.rdd.mapValues(lambda x: dot(x, other_b.value))
            return self._constructor(newrdd, index=newindex).__finalize__(self)

    def element_wise(self, other, op):
        """
        Apply an elementwise operation to distributed matrices.

        Can be applied to two RowMatrices,
        or between a Matrix and a local array.
        NOTE: For two RowMatrices, must have the same partitions
        and number of records per iteration (e.g. because
        one was created through a map on the other, see zip)

        Parameters
        ----------
        other : Matrix, scalar, or numpy array
            Matrix to combine with element-wise

        op : function
            Binary operator to use for elementwise operations, e.g. add, subtract
        """
        dtype = type(other)
        if dtype is Matrix:
            if (self.nrows is not other.nrows) | (self.ncols is not other.ncols):
                raise Exception(
                    "Cannot do elementwise op for shapes ("
                    + self.nrows + "," + self.ncols + ") and ("
                    + other.nrows + "," + other.ncols + ")")
            else:
                func = lambda ((k1, x), (k2, y)): (k1, add(x, y))
                return self._constructor(self.rdd.zip(other.rdd).map(func)).__finalize__(self)
        else:
            if dtype is ndarray:
                dims = shape(other)
                if len(dims) > 1 or dims[0] is not self.ncols:
                    raise Exception(
                        "Cannot do elementwise operation for shapes ("
                        + str(self.nrows) + "," + str(self.ncols) + ") and " + str(dims))
            func = lambda x: op(x, other)
            return self._constructor(self.rdd.mapValues(func)).__finalize__(self)

    def plus(self, other):
        """
        Elementwise addition of distributed matrices.

        See also
        --------
        elementwise

        """
        return Matrix.element_wise(self, other, add)

    def minus(self, other):
        """
        Elementwise division of distributed matrices.

        See also
        --------
        elementwise
        """
        return Matrix.element_wise(self, other, subtract)

    def dottimes(self, other):
        """
        Elementwise division of distributed matrices.

        See also
        --------
        elementwise
        """
        return Matrix.element_wise(self, other, multiply)

    def dotdivide(self, other):
        """
        Elementwise division of distributed matrices.

        See also
        --------
        elementwise
        """
        return Matrix.element_wise(self, other, divide)


def outer_iterator(iterator):
    yield sum(outer(x, x) for x in iterator)


def outer_iterator_other(iterator):
    yield sum(outer(x, y) for x, y in iterator)




