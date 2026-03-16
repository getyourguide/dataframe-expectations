import warnings

import pytest
from unittest.mock import patch

from dataframe_expectations.core import pyspark_utils
from dataframe_expectations.core.pyspark_utils import (
    get_pyspark_functions,
    is_pyspark_data_frame,
)


# ---------------------------------------------------------------------------
# get_pyspark_functions
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# FutureWarning
# ---------------------------------------------------------------------------


def test_is_pyspark_data_frame_emits_future_warning_for_pyspark_df(spark):
    """A FutureWarning is emitted the first time a PySpark DataFrame is detected."""
    # Reset the module-level flag so the warning fires fresh.
    pyspark_utils._pyspark_deprecation_warned = False

    spark_df = spark.createDataFrame([(1,)], ["col1"])

    with pytest.warns(FutureWarning, match="optional dependency"):
        is_pyspark_data_frame(spark_df)


def test_is_pyspark_data_frame_warning_fires_only_once(spark):
    """The FutureWarning fires at most once per process (guarded by the module flag)."""
    pyspark_utils._pyspark_deprecation_warned = False

    spark_df = spark.createDataFrame([(1,)], ["col1"])

    with pytest.warns(FutureWarning):
        is_pyspark_data_frame(spark_df)

    # Second call should produce no warning.
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning would raise
        is_pyspark_data_frame(spark_df)  # must not raise


def test_is_pyspark_data_frame_no_warning_for_pandas():
    """No FutureWarning is emitted when the DataFrame is a Pandas DataFrame."""
    import pandas as pd

    pyspark_utils._pyspark_deprecation_warned = False

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning would raise
        is_pyspark_data_frame(pd.DataFrame())  # must not raise
