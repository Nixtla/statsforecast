import pytest
from pyspark.sql import SparkSession

from statsforecast.utils import generate_series
from .utils import pipeline, pipeline_with_level

@pytest.fixture()
def n_series():
    return 2

@pytest.fixture()
def sample_data(n_series):
    n_series = 2
    series = generate_series(n_series).reset_index()
    series['unique_id'] = series['unique_id'].astype(str)
    spark = SparkSession.builder.getOrCreate()
    series = spark.createDataFrame(series).repartition(2, 'unique_id')
    return series

def test_spark_flow(sample_data, n_series):
    horizon = 7
    pipeline(sample_data, n_series, horizon)

def test_spark_flow_with_level(sample_data, n_series):
    horizon = 7
    pipeline_with_level(sample_data, n_series, horizon)
