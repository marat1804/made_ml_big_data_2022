import breeze.linalg.{DenseMatrix, DenseVector, lowerTriangular}
import breeze.numerics.abs
import breeze.stats.mean
import model.LinearRegression

import java.util.logging.{FileHandler, Logger, SimpleFormatter}


package object utils {
  def meanAbsoluteError(yTrue: DenseVector[Double], yPred: DenseVector[Double]): Double = {
    mean(abs(yTrue - yPred))
  }

  def crossValidation(model: LinearRegression, data: DenseMatrix[Double], numFolds: Int, logger: Logger): Unit = {
    val stepOfValidation = data.rows / numFolds
    logger.info(s"Step of validation is ${stepOfValidation}")
    for (i <- 0 until  numFolds) {
      val trainIndex: IndexedSeq[Int] = IndexedSeq(0, i * stepOfValidation - 1) ++
        IndexedSeq((i + 1) * stepOfValidation + 1, data.rows - 1)
      val validationIndex: IndexedSeq[Int] = IndexedSeq(i * stepOfValidation, (i + 1) * stepOfValidation)
      val train: DenseMatrix[Double] = data(trainIndex, ::).toDenseMatrix
      val validation: DenseMatrix[Double] = data(validationIndex, ::).toDenseMatrix
      model.fit(train(::, 0 to -2), train(::, -1))
      val mae: Double = meanAbsoluteError(validation(::, -1), model.predict(validation(::, 0 to -2)))
      logger.info(s"Fold $i - MAE-score: $mae")
    }
  }

  def getLogger(name: String, outputFile: String): Logger = {
    System.setProperty(
      "java.util.logging.SimpleFormatter.format",
      "[%1$tF %1$tT] [%4$-7s] %5$s %n"
    )

    val logger = Logger.getLogger(name)
    val handler = new FileHandler(outputFile)
    val formatter = new SimpleFormatter()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger
  }
}
