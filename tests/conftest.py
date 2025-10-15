import pytest
from pyspark.sql import SparkSession
import pandas as pd
import pandas.testing as pdt

@pytest.fixture(scope="module")
def spark() -> SparkSession:
    """create a spark session we can reuse for every test"""

    return (
        SparkSession.builder.master("local").appName("Test")
        .getOrCreate()
    )


def assert_pandas_df_equal(df1: pd.DataFrame, df2: pd.DataFrame):
    # Optional: sort and reset index to avoid false mismatches due to row order
    df1_sorted = df1.sort_values(by=df1.columns.tolist()).reset_index(drop=True)
    df2_sorted = df2.sort_values(by=df2.columns.tolist()).reset_index(drop=True)

    pdt.assert_frame_equal(df1_sorted, df2_sorted, check_dtype=False)


def assert_pyspark_df_equal(df1, df2):
    df1_pd = df1.toPandas().sort_values(by=df1.columns).reset_index(drop=True)
    df2_pd = df2.toPandas().sort_values(by=df2.columns).reset_index(drop=True)

    pd.testing.assert_frame_equal(df1_pd, df2_pd, check_dtype=False)