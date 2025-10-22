import pytest
from unittest.mock import MagicMock

import pandas as pd

from dataframe_expectations import DataFrameType
from dataframe_expectations.expectations.column_expectation import (
    DataFrameColumnExpectation,
)


@pytest.fixture
def expectation():
    return DataFrameColumnExpectation(
        expectation_name="MyColumnExpectation",
        column_name="test_column",
        fn_violations_pandas=lambda df: df,
        fn_violations_pyspark=lambda df: df,
        description="Test column expectation",
        error_message="Test column expectation failed.",
    )


def test_get_expectation_name(expectation):
    """
    Test that the expectation name is the class name.
    """
    assert expectation.get_expectation_name() == "MyColumnExpectation", (
        f"Expected 'MyColumnExpectation' but got: {expectation.get_expectation_name()}"
    )


def test_validate_for_pandas_df(expectation):
    """
    Test whether row_validation() and get_filter_fn() methods are called with the right parameters for Pandas.
    """

    # Mock methods
    expectation.row_validation = MagicMock(return_value="mock_result")

    data_frame = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})

    # test validate_pandas called the right methods
    _ = expectation.validate(data_frame=data_frame)

    expectation.row_validation.assert_called_once_with(
        data_frame_type=DataFrameType.PANDAS,
        data_frame=data_frame,
        fn_violations=expectation.fn_violations_pandas,
    )


def test_validate_for_pyspark_df(expectation, spark):
    """
    Test whether row_validation() and get_filter_fn() methods are called with the right parameters for PySpark.
    """

    # Mock methods
    expectation.row_validation = MagicMock(return_value="mock_result")
    data_frame = spark.createDataFrame([(1, "a"), (2, "b"), (3, "c")], ["col1", "col2"])

    # test validate_pyspark called the right methods
    _ = expectation.validate(data_frame=data_frame)

    expectation.row_validation.assert_called_once_with(
        data_frame_type=DataFrameType.PYSPARK,
        data_frame=data_frame,
        fn_violations=expectation.fn_violations_pyspark,
    )
