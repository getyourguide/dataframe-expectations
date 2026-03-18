import pytest
from unittest.mock import patch

from dataframe_expectations.core.pyspark_utils import (
    get_pyspark_functions,
    is_pyspark_data_frame,
)


# ---------------------------------------------------------------------------
# get_pyspark_functions
# ---------------------------------------------------------------------------


@pytest.mark.pyspark
def test_get_pyspark_functions_returns_real_functions_when_pyspark_present():
    """When pyspark is installed, the real functions module is returned."""
    # lru_cache may already hold the result from a previous test run, so clear it.
    get_pyspark_functions.cache_clear()

    F = get_pyspark_functions()
    # The real pyspark.sql.functions module has `col` as an attribute.
    assert hasattr(F, "col")


def test_get_pyspark_functions_proxy_raises_on_attribute_access_when_pyspark_missing():
    """When pyspark is not installed, accessing any attribute raises ImportError."""
    get_pyspark_functions.cache_clear()

    with patch.dict(
        "sys.modules", {"pyspark": None, "pyspark.sql": None, "pyspark.sql.functions": None}
    ):
        get_pyspark_functions.cache_clear()
        F = get_pyspark_functions()
        with pytest.raises(ImportError, match="PySpark is required"):
            _ = F.col

    # Restore cache state for subsequent tests.
    get_pyspark_functions.cache_clear()


# ---------------------------------------------------------------------------
# is_pyspark_data_frame
# ---------------------------------------------------------------------------


def test_is_pyspark_data_frame_returns_false_for_non_spark_objects():
    import pandas as pd

    assert is_pyspark_data_frame(pd.DataFrame()) is False
    assert is_pyspark_data_frame("string") is False
    assert is_pyspark_data_frame(42) is False
    assert is_pyspark_data_frame(None) is False


@pytest.mark.pyspark
def test_is_pyspark_data_frame_returns_true_for_pyspark_df(spark):
    spark_df = spark.createDataFrame([(1,)], ["col1"])
    assert is_pyspark_data_frame(spark_df) is True


def test_is_pyspark_data_frame_returns_true_for_module_name_fallback():
    """Objects whose __module__ starts with 'pyspark.sql' are treated as Spark DataFrames."""

    class FakeRuntimeSparkDataFrame:
        pass

    FakeRuntimeSparkDataFrame.__module__ = "pyspark.sql.dataframe"

    obj = FakeRuntimeSparkDataFrame()
    assert is_pyspark_data_frame(obj) is True
