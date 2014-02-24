package thunder.regression

import cern.colt.matrix.DoubleFactory2D
import cern.colt.matrix.DoubleFactory1D
import cern.colt.matrix.linalg.Algebra.DEFAULT.{inverse, mult}
import scala.math.{pow, max}

class SharedLinearRegressionModel(var features: Array[Array[Double]]) extends Serializable {

  /** Number of features and data points. */
  val d = features.size // features
  val n = features(0).size // data points

  /** make n x (d + 1) matrix of labels with column of ones appended. */
  val x = DoubleFactory2D.dense.make(n, d + 1)
  for (i <- 0 until n) {
    x.set(i, 0, 1)
  }
  for (i <- 0 until n ; j <- 1 until d + 1) {
    x.set(i, j, features(j - 1)(i))
  }

  /** Pre compute pseudo inverse for efficiency. */
  val xhat = inverse(x)

  /** Compute r2 given data point and prediction. */
  def getR2(point: Array[Double], predict: Array[Double]) = {
    val sse = point.zip(predict).map{case (ix, iy) => ix - iy}.map(x => pow(x, 2)).sum
    val sst = point.map(x => x - (point.sum / point.length)).map(x => pow(x, 2)).sum
    1.0 - (sse / sst)
  }

  /** Compute features weighted by response. */
  def weights(point: Array[Double]): Array[Double] = {
    val pointPos = point.map(x => max(x, 0))
    features.map(x => x.zip(pointPos).map{case (ix, iy) => ix * iy}.sum / pointPos.sum )
  }

  /** Fit a model for each data point using the shared features. */
  def fit(point: Array[Double]): LinearRegressionModel = {
    val y = DoubleFactory1D.dense.make(point.length)
    for (i <- 0 until point.length) {
      y.set(i, point(i))
    }
    val b = mult(xhat, y)
    val predict = mult(x, b).toArray
    val r2 = getR2(point, predict)
    val intercept = b.toArray()(0)
    val weights = b.toArray.drop(1)
    new LinearRegressionModel(weights, intercept, r2)
  }

}