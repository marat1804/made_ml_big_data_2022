package org.apache.spark.ml.made

import breeze.linalg.{*, DenseMatrix, DenseVector}
import breeze.numerics.{rint, sigmoid}
import com.google.common.io.Files
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.DataFrame
import org.scalatest.flatspec._
import org.scalatest.matchers._


class LogisticRegressionTest extends AnyFlatSpec with should.Matchers with WithSpark {
  val delta: Double = 0.01
  var weights: DenseVector[Double] = LogisticRegressionTest._weights
  val bias: Double = LogisticRegressionTest._bias
  val y_true: DenseVector[Double] = LogisticRegressionTest._y
  val df: DataFrame = LogisticRegressionTest._df
  val x: DenseMatrix[Double] = LogisticRegressionTest._X

  private def validateModel(data: DataFrame): Unit = {
    val y_pred = data.collect().map(_.getAs[Double](1))

    y_pred.length should be (100000)

    for (i <- y_pred.indices) {
      y_pred(i) should be (y_true(i))
    }
  }

  private def validateEstimator(model: LogisticRegressionModel) = {
    model.weights.size should be (weights.size + 1)
    model.weights(0) should be (weights(0) +- delta)
    model.weights(1) should be (weights(1) +- delta)
    model.weights(2) should be (weights(2) +- delta)
    model.weights(3) should be (bias +- delta)
  }

  "Estimator" should "create working model" in {
    val estimator = new LogisticRegression()
      .setInputCol("features")
      .setOutputCol("label")

    val model = estimator.fit(df)

    validateEstimator(model)
  }

  "Model" should "make right predictions of target variable" in {
    val model: LogisticRegressionModel = new LogisticRegressionModel(
      weights = Vectors.fromBreeze(
        DenseVector.vertcat(weights, DenseVector[Double](bias))
      )
    ).setInputCol("features").setOutputCol("label")

    validateModel(model.transform(df))
  }

  "Estimator" should "work after re-read" in {
    val pipeline = new Pipeline().setStages(Array(
      new LogisticRegression()
        .setInputCol("features")
        .setOutputCol("label")
    ))

    val tempDir = Files.createTempDir()

    pipeline.write.overwrite().save(tempDir.getAbsolutePath)

    val model = Pipeline.load(tempDir.getAbsolutePath)
      .fit(df)
      .stages(0)
      .asInstanceOf[LogisticRegressionModel]

    validateEstimator(model)
  }

  "Model" should "work after re-read" in {
    val pipeline = new Pipeline().setStages(Array(
      new LogisticRegression()
        .setInputCol("features")
        .setOutputCol("label")
    ))

    val model = pipeline.fit(df)

    val tempDir = Files.createTempDir()

    print(tempDir.getAbsolutePath)

    model.write.overwrite().save(tempDir.getAbsolutePath)

    val reRead: PipelineModel = PipelineModel.load(tempDir.getAbsolutePath)

    validateModel(reRead.transform(df))
  }
}

object LogisticRegressionTest extends WithSpark {
  lazy val _X: DenseMatrix[Double] = DenseMatrix.rand[Double](100000, 3)
  lazy val _weights: DenseVector[Double] = DenseVector(2.7, -2.3, 0.3)
  lazy val _bias: Double = 1.0
  lazy val _y: DenseVector[Double] = sigmoid(_X * _weights + _bias)
  lazy val _df: DataFrame = createDataFrame(_X, _y)

  def createDataFrame(X: DenseMatrix[Double], y: DenseVector[Double]): DataFrame = {
    import sqlc.implicits._

    lazy val data: DenseMatrix[Double] = DenseMatrix.horzcat(X, y.asDenseMatrix.t)

    lazy val df = data(*, ::).iterator
      .map(x => (x(0), x(1), x(2), x(3)))
      .toSeq
      .toDF("x1", "x2", "x3", "label")

    lazy val assembler = new VectorAssembler()
      .setInputCols(Array("x1", "x2", "x3"))
      .setOutputCol("features")

    lazy val _df: DataFrame = assembler
      .transform(df)
      .select("features", "label")

    _df
  }
}
