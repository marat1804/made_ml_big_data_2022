import model.LinearRegression

import java.io.File
import breeze.linalg.{DenseMatrix, DenseVector, csvread, csvwrite, lowerTriangular}
import utils.{crossValidation, getLogger, meanAbsoluteError}

object Main {
  def main(args: Array[String]): Unit = {
    if (args.length != 3) {
      println("Provide path for train, test")
      return
    }
    val logger = getLogger(name = "Linear Regression", outputFile = "log.txt")

    logger.info(s"Train path - ${args(0)}")
    logger.info(s"Test path - ${args(1)}")
    logger.info(s"Result path - ${args(2)}")

    val train: DenseMatrix[Double] = csvread(new File(args(0)), separator = ',', skipLines = 1)
    logger.info(s"Read train data with size (${train.rows}, ${train.cols})")

    logger.info("Init Linear Regression")
    val model = new LinearRegression()
    logger.info("Reading train data")
    val xTrain: DenseMatrix[Double] = train(::, 0 to -2)
    val yTrain: DenseVector[Double] = train(::, -1)

    logger.info("Start cross validation of model")
    crossValidation(model, train, 5, logger)

    logger.info("Training Linear Regression")
    model.fit(xTrain, yTrain)
    val maeTrain: Double = meanAbsoluteError(yTrain, model.predict(xTrain))
    logger.info(s"MAE-score on train data - ${maeTrain}")

    logger.info("Reading test data")
    val test: DenseMatrix[Double] = csvread(new File(args(1)), separator = ',', skipLines = 1)
    logger.info(s"Read test data with size (${test.rows}, ${test.cols})")
    val testPredictions: DenseVector[Double] = model.predict(test)

    logger.info("Predicted test values")
    csvwrite(new File(args(2)), testPredictions.asDenseMatrix.t)
    logger.info(s"Saved results in ${args(2)}")
    logger.info("Finish scoring with Linear Regression")
  }
}
