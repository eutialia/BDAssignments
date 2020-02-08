import sys
from pyspark.sql import SparkSession, types
from pyspark.ml import PipelineModel, Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler, SQLTransformer
from pyspark.ml.regression import GBTRegressor

spark = SparkSession.builder.appName('tmax_prediction').getOrCreate()
spark.sparkContext.setLogLevel('WARN')

tmax_schema = types.StructType([
    types.StructField('station', types.StringType()),
    types.StructField('date', types.DateType()),
    types.StructField('latitude', types.FloatType()),
    types.StructField('longitude', types.FloatType()),
    types.StructField('elevation', types.FloatType()),
    types.StructField('tmax', types.FloatType()),
])

def main(train_data, val_data):
    test_tmax = spark.read.csv(train_data, schema=tmax_schema)
    train, validation = test_tmax.randomSplit([0.75, 0.25])
    sql_transformer = SQLTransformer(statement='SELECT station, dayofyear(date) AS date, latitude, longitude, elevation, tmax FROM __THIS__')
    assembler = VectorAssembler(inputCols=['latitude', 'longitude', 'elevation', 'date'], outputCol='features')
 
    classifier = GBTRegressor(featuresCol='features', labelCol='tmax')
    pipelineModel = Pipeline(stages=[sql_transformer, assembler, classifier])
    model = pipelineModel.fit(train)
    
    input_df = spark.read.load(val_data, format="csv", inferSchema="true", header="true")
    prediction = model.transform(input_df)
    prediction.toPandas().to_csv('tmax.csv')

if __name__ == '__main__':
    val_data = sys.argv[2]
    train_data = sys.argv[1]
    main(train_data, val_data)
