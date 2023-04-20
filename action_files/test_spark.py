from pyspark.sql import SparkSession

from statsforecast.utils import generate_series
from .utils import pipeline

def test_spark_flow():
    n_series = 2
    horizon = 7
    series = generate_series(n_series).reset_index()
    series['unique_id'] = series['unique_id'].astype(str)
    spark = SparkSession.builder.getOrCreate()
    series = spark.createDataFrame(series).repartition(2, 'unique_id')
    pipeline(series, n_series, horizon)
