package thunder.regression

import org.apache.spark.mllib.regression.{RegressionModel, GeneralizedLinearModel}
import org.jblas.DoubleMatrix

/**
 * Regression model trained using LinearRegression.
 * Extends the version from ML Lib by adding r2.
 *
 * @param weights Weights computed for every feature.
 * @param intercept Intercept computed for this model.
 * @param r2 r2 for the fitted model
 */
class LinearRegressionModel(
                             override val weights: Array[Double],
                             override val intercept: Double,
                             val r2: Double)
  extends GeneralizedLinearModel(weights, intercept)
  with RegressionModel with Serializable {

  override def predictPoint(dataMatrix: DoubleMatrix, weightMatrix: DoubleMatrix,
                            intercept: Double) = {
    dataMatrix.dot(weightMatrix) + intercept
  }
}