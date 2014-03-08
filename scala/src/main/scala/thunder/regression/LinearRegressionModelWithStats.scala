package thunder.regression

import org.apache.spark.mllib.regression.{RegressionModel, GeneralizedLinearModel}
import org.jblas.DoubleMatrix

/**
 * Regression model trained using LinearRegression.
 * Extends the version from ML Lib by adding
 * summary statistics (response-weighted features and r2).
 *
 * @param weights Weights computed for every feature.
 * @param intercept Intercept computed for this model.
 * @param r2 r2 for the fitted model
 * @param tuning Response-weighted features
 */
class LinearRegressionModelWithStats(
                             override val weights: Array[Double],
                             override val intercept: Double,
                             val r2: Double,
                             val tuning: Array[Double])
  extends GeneralizedLinearModel(weights, intercept)
  with RegressionModel with Serializable {

  override def predictPoint(dataMatrix: DoubleMatrix, weightMatrix: DoubleMatrix,
                            intercept: Double) = {
    dataMatrix.dot(weightMatrix) + intercept
  }
}