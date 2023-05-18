package org.apache.spark.ml.made

import org.apache.spark.ml.{Estimator, Model}
import breeze.linalg
import breeze.numerics.{sigmoid, rint}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.{Vector, VectorUDT, Vectors}
import org.apache.spark.ml.param.{DoubleParam, IntParam, ParamMap}
import org.apache.spark.ml.param.shared.{HasInputCol, HasOutputCol}
import org.apache.spark.sql.types.StructType
import org.apache.spark.ml.util._
import org.apache.spark.mllib
import org.apache.spark.mllib.stat.MultivariateOnlineSummarizer
import org.apache.spark.sql.{DataFrame, Dataset, Encoder}
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.functions.lit


trait LogisticRegressionParameters extends HasInputCol with HasOutputCol {
  def setInputCol(value: String): this.type = set(inputCol, value)

  def setOutputCol(value: String): this.type = set(outputCol, value)

  private val maxSteps = new IntParam(this, "maxSteps", "Maximum number of steps in GD")
  private val lr = new DoubleParam(this, "lr", "Step size on each iteration")
  private val regParam = new DoubleParam(this, "regParam", "Regularization coefficient")

  def setMaxSteps(value: Int): this.type = set(maxSteps, value)
  def getMaxSteps: Int = $(maxSteps)
  setDefault(maxSteps -> 1000)

  def setLearningRate(value: Double): this.type = set(lr, value)
  def getLearningRate: Double = $(lr)
  setDefault(lr -> 0.75)

  protected def validateAndTransformSchema(schema: StructType): StructType = {
    SchemaUtils.checkColumnType(schema, getInputCol, new VectorUDT())

    if (schema.fieldNames.contains($(outputCol))) {
      SchemaUtils.checkColumnType(schema, getOutputCol, new VectorUDT())
      schema
    } else {
      SchemaUtils.appendColumn(schema, schema(getInputCol).copy(name = getOutputCol))
    }
  }
}

class LogisticRegression(override val uid: String) extends Estimator[LogisticRegressionModel] with LogisticRegressionParameters
  with DefaultParamsWritable {
  def this() = this(Identifiable.randomUID("logReg"))

  override def fit(inputData: Dataset[_]): LogisticRegressionModel = {
    implicit val encoder: Encoder[Vector] = ExpressionEncoder()

    val assembler: VectorAssembler = new VectorAssembler()
      .setInputCols(Array($(inputCol), "ones", $(outputCol)))
      .setOutputCol("features_label")

    val vectors: Dataset[Vector] = assembler
      .transform(inputData.withColumn("ones", lit(1)))
      .select("features_label")
      .as[Vector]

    val numFeatures: Int = MetadataUtils.getNumFeatures(inputData, $(inputCol))
    var weights: linalg.DenseVector[Double] = linalg.DenseVector.rand[Double](numFeatures + 1)

    for (_ <- 0 until  getMaxSteps) {
      val summary = vectors.rdd.mapPartitions((data: Iterator[Vector]) => {
        val summarizer = new MultivariateOnlineSummarizer()
        data.foreach(v => {
          val X = v.asBreeze(0 until weights.size).toDenseVector
          val y = v.asBreeze(weights.size)
          val y_pred = sigmoid(X.dot(weights))
          val grad = X * (y_pred - y)
          summarizer.add(mllib.linalg.Vectors.fromBreeze(grad))
        })
        Iterator(summarizer)
      }).reduce(_ merge _)

      weights = weights - getLearningRate * summary.mean.asBreeze
    }

    copyValues(new LogisticRegressionModel(Vectors.fromBreeze(weights))).setParent(this)
  }

  override def copy(extra: ParamMap): Estimator[LogisticRegressionModel] = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)
}

object LogisticRegression extends DefaultParamsReadable[LogisticRegression]

class LogisticRegressionModel private[made](override val uid: String,
                                            val weights: Vector) extends Model[LogisticRegressionModel] with LogisticRegressionParameters with MLWritable {
  private[made] def this(weights: Vector) =
    this(Identifiable.randomUID("logRegModel"), weights)

  override def copy(extra: ParamMap): LogisticRegressionModel = copyValues(new LogisticRegressionModel(weights), extra)

  override def write: MLWriter = new DefaultParamsWriter(this) {
    override protected def saveImpl(path: String): Unit = {
      super.saveImpl(path)
      sqlContext.createDataFrame(Seq(Tuple1(weights))).write.parquet(path + "/weights")
    }
  }

  override def transform(dataset: Dataset[_]): DataFrame = {
    val bias = weights.asBreeze(-1)
    val weightsBreeze = weights.asBreeze(0 until weights.size - 1)

    val transformUdf = {
      dataset.sqlContext.udf.register(uid + "_transform",
        (x: Vector) => sigmoid(bias + x.asBreeze.dot(weightsBreeze))
      )
    }

    dataset.withColumn($(outputCol), transformUdf(dataset($(inputCol))))
  }

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)

}

object LogisticRegressionModel extends MLReadable[LogisticRegressionModel] {
  override def read: MLReader[LogisticRegressionModel] = new MLReader[LogisticRegressionModel] {
    override def load(path: String): LogisticRegressionModel = {
      val data = DefaultParamsReader.loadMetadata(path, sc)
      val vectors = sqlContext.read.parquet(path + "/weights")
      implicit val encoder: Encoder[Vector] = ExpressionEncoder()
      val weights = vectors.select(vectors("_1").as[Vector]).first()

      val model = new LogisticRegressionModel(weights)
      data.getAndSetParams(model)
      model
    }
  }
}


