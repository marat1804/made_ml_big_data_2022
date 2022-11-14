package model

import breeze.linalg.{DenseMatrix, DenseVector}

class LinearRegression {
  var weights: DenseVector[Double] = DenseVector.zeros[Double](size = 0)

  def fit(X: DenseMatrix[Double], y: DenseVector[Double]): Unit = {
    val a = X.t * X
    val b = X.t * y
    weights = a \ b
  }

  def predict(X: DenseMatrix[Double]): DenseVector[Double] = {
    X * weights
  }
}
