import pytest
import numpy as np
import pandas as pd

from dataframe_expectations import DataFrameType
from dataframe_expectations.expectations.expectation_registry import (
    DataframeExpectationRegistry,
)
from dataframe_expectations.expectations_suite import (
    DataframeExpectationsSuite,
    DataframeExpectationsSuiteFailure,
)
from dataframe_expectations.result_message import (
    DataframeExpectationFailureMessage,
    DataframeExpectationSuccessMessage,
)


def test_expectation_pandas_success_no_nulls():
    """Test pandas success case with no null values."""
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationMaxNullCount",
        column_name="col1",
        max_count=5,
    )
    data_frame = pd.DataFrame(
        {
            "col1": [1, 2, 3, 4, 5],
            "col2": ["a", "b", "c", "d", "e"],
            "col3": [1.1, 2.2, 3.3, 4.4, 5.5],
        }
    )
    result = expectation.validate(data_frame=data_frame)
    assert str(result) == str(
        DataframeExpectationSuccessMessage(expectation_name="ExpectationMaxNullCount")
    ), f"Expected success message but got: {result}"


def test_expectation_pandas_success_within_threshold():
    """Test pandas success case with null count within threshold."""
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationMaxNullCount",
        column_name="col1",
        max_count=3,
    )
    # 2 null values in col1, which is less than max_count of 3
    data_frame = pd.DataFrame(
        {
            "col1": [1, None, 3, None, 5],
            "col2": ["a", "b", "c", "d", "e"],
        }
    )
    result = expectation.validate(data_frame=data_frame)
    assert str(result) == str(
        DataframeExpectationSuccessMessage(expectation_name="ExpectationMaxNullCount")
    ), f"Expected success message but got: {result}"


def test_expectation_pandas_success_exactly_at_threshold():
    """Test pandas success case with null count exactly at threshold."""
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationMaxNullCount",
        column_name="col1",
        max_count=2,
    )
    # Exactly 2 null values in col1
    data_frame = pd.DataFrame({"col1": [1, 2, None, 4, None], "col2": [None, "b", "c", "d", "e"]})
    result = expectation.validate(data_frame=data_frame)
    assert str(result) == str(
        DataframeExpectationSuccessMessage(expectation_name="ExpectationMaxNullCount")
    ), f"Expected success message but got: {result}"


def test_expectation_pandas_success_with_nan():
    """Test pandas success case with NaN values within threshold."""
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationMaxNullCount",
        column_name="col2",
        max_count=2,
    )
    # 1 NaN value in col2, which is less than max_count of 2
    data_frame = pd.DataFrame({"col1": [1, 2, 3], "col2": [4.0, np.nan, 6.0]})
    result = expectation.validate(data_frame=data_frame)
    assert str(result) == str(
        DataframeExpectationSuccessMessage(expectation_name="ExpectationMaxNullCount")
    ), f"Expected success message but got: {result}"


def test_expectation_pandas_failure_exceeds_threshold():
    """Test pandas failure case when null count exceeds threshold."""
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationMaxNullCount",
        column_name="col1",
        max_count=1,
    )
    # 3 null values in col1, which exceeds max_count of 1
    data_frame = pd.DataFrame(
        {"col1": [1, None, None, None, 5], "col2": [None, "b", "c", "d", "e"]}
    )
    result = expectation.validate(data_frame=data_frame)

    expected_failure_message = DataframeExpectationFailureMessage(
        expectation_str=str(expectation),
        data_frame_type=DataFrameType.PANDAS,
        message="Column 'col1' has 3 null values, expected at most 1.",
    )
    assert str(result) == str(
        expected_failure_message
    ), f"Expected failure message but got: {result}"


def test_expectation_pandas_failure_all_nulls_in_column():
    """Test pandas failure case with all null values in the specified column."""
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationMaxNullCount",
        column_name="col1",
        max_count=1,
    )
    data_frame = pd.DataFrame({"col1": [None, None, None], "col2": [1, 2, 3]})
    result = expectation.validate(data_frame=data_frame)

    expected_failure_message = DataframeExpectationFailureMessage(
        expectation_str=str(expectation),
        data_frame_type=DataFrameType.PANDAS,
        message="Column 'col1' has 3 null values, expected at most 1.",
    )
    assert str(result) == str(
        expected_failure_message
    ), f"Expected failure message but got: {result}"


def test_expectation_pandas_boundary_zero_threshold():
    """Test pandas boundary case with 0 threshold."""
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationMaxNullCount",
        column_name="col1",
        max_count=0,
    )
    data_frame = pd.DataFrame({"col1": [1, None, 3]})
    result = expectation.validate(data_frame=data_frame)

    expected_failure_message = DataframeExpectationFailureMessage(
        expectation_str=str(expectation),
        data_frame_type=DataFrameType.PANDAS,
        message="Column 'col1' has 1 null values, expected at most 0.",
    )
    assert str(result) == str(
        expected_failure_message
    ), f"Expected failure message but got: {result}"


def test_expectation_pandas_boundary_zero_threshold_success():
    """Test pandas boundary case with 0 threshold and no nulls."""
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationMaxNullCount",
        column_name="col1",
        max_count=0,
    )
    data_frame = pd.DataFrame({"col1": [1, 2, 3], "col2": [None, None, None]})
    result = expectation.validate(data_frame=data_frame)
    assert str(result) == str(
        DataframeExpectationSuccessMessage(expectation_name="ExpectationMaxNullCount")
    ), f"Expected success message but got: {result}"


def test_expectation_pandas_empty_dataframe():
    """Test pandas edge case with empty DataFrame."""
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationMaxNullCount",
        column_name="col1",
        max_count=5,
    )
    data_frame = pd.DataFrame(columns=["col1"])
    result = expectation.validate(data_frame=data_frame)
    # Empty DataFrame should have 0 nulls and pass
    assert str(result) == str(
        DataframeExpectationSuccessMessage(expectation_name="ExpectationMaxNullCount")
    ), f"Expected success message but got: {result}"


def test_expectation_pandas_single_value_null():
    """Test pandas edge case with single null value."""
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationMaxNullCount",
        column_name="col1",
        max_count=0,
    )
    data_frame = pd.DataFrame({"col1": [None]})
    result = expectation.validate(data_frame=data_frame)

    expected_failure_message = DataframeExpectationFailureMessage(
        expectation_str=str(expectation),
        data_frame_type=DataFrameType.PANDAS,
        message="Column 'col1' has 1 null values, expected at most 0.",
    )
    assert str(result) == str(
        expected_failure_message
    ), f"Expected failure message but got: {result}"


def test_expectation_pandas_single_value_not_null():
    """Test pandas edge case with single non-null value."""
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationMaxNullCount",
        column_name="col1",
        max_count=0,
    )
    data_frame = pd.DataFrame({"col1": [1]})
    result = expectation.validate(data_frame=data_frame)
    assert str(result) == str(
        DataframeExpectationSuccessMessage(expectation_name="ExpectationMaxNullCount")
    ), f"Expected success message but got: {result}"


def test_expectation_pandas_different_column_nulls_not_affecting():
    """Test that nulls in other columns don't affect the result."""
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationMaxNullCount",
        column_name="col1",
        max_count=1,
    )
    # col1 has 0 nulls, col2 has 3 nulls - should pass since we're only checking col1
    data_frame = pd.DataFrame({"col1": [1, 2, 3], "col2": [None, None, None]})
    result = expectation.validate(data_frame=data_frame)
    assert str(result) == str(
        DataframeExpectationSuccessMessage(expectation_name="ExpectationMaxNullCount")
    ), f"Expected success message but got: {result}"


def test_expectation_pyspark_success_no_nulls(spark):
    """Test PySpark success case with no null values."""
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationMaxNullCount",
        column_name="col1",
        max_count=5,
    )
    data_frame = spark.createDataFrame(
        [(1, "a", 1.1), (2, "b", 2.2), (3, "c", 3.3), (4, "d", 4.4), (5, "e", 5.5)],
        ["col1", "col2", "col3"],
    )
    result = expectation.validate(data_frame=data_frame)
    assert str(result) == str(
        DataframeExpectationSuccessMessage(expectation_name="ExpectationMaxNullCount")
    ), f"Expected success message but got: {result}"


def test_expectation_pyspark_success_within_threshold(spark):
    """Test PySpark success case with null count within threshold."""
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationMaxNullCount",
        column_name="col1",
        max_count=3,
    )
    # 2 null values in col1, which is less than max_count of 3
    data_frame = spark.createDataFrame([(1,), (None,), (3,), (None,), (5,)], ["col1"])
    result = expectation.validate(data_frame=data_frame)
    assert str(result) == str(
        DataframeExpectationSuccessMessage(expectation_name="ExpectationMaxNullCount")
    ), f"Expected success message but got: {result}"


def test_expectation_pyspark_success_exactly_at_threshold(spark):
    """Test PySpark success case with null count exactly at threshold."""
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationMaxNullCount",
        column_name="col1",
        max_count=2,
    )
    # Exactly 2 null values in col1
    data_frame = spark.createDataFrame(
        [(1, "a"), (2, None), (None, "c"), (4, "d"), (None, "e")], ["col1", "col2"]
    )
    result = expectation.validate(data_frame=data_frame)
    assert str(result) == str(
        DataframeExpectationSuccessMessage(expectation_name="ExpectationMaxNullCount")
    ), f"Expected success message but got: {result}"


def test_expectation_pyspark_failure_exceeds_threshold(spark):
    """Test PySpark failure case when null count exceeds threshold."""
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationMaxNullCount",
        column_name="col1",
        max_count=1,
    )
    # 2 null values in col1, which exceeds max_count of 1
    data_frame = spark.createDataFrame([(1, None), (None, "b"), (None, "c")], ["col1", "col2"])
    result = expectation.validate(data_frame=data_frame)

    expected_failure_message = DataframeExpectationFailureMessage(
        expectation_str=str(expectation),
        data_frame_type=DataFrameType.PYSPARK,
        message="Column 'col1' has 2 null values, expected at most 1.",
    )
    assert str(result) == str(
        expected_failure_message
    ), f"Expected failure message but got: {result}"


def test_expectation_pyspark_failure_all_nulls_in_column(spark):
    """Test PySpark failure case with all null values in the specified column."""
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationMaxNullCount",
        column_name="col1",
        max_count=2,
    )
    data_frame = spark.createDataFrame([(None,), (None,), (None,)], "col1: int")
    result = expectation.validate(data_frame=data_frame)

    expected_failure_message = DataframeExpectationFailureMessage(
        expectation_str=str(expectation),
        data_frame_type=DataFrameType.PYSPARK,
        message="Column 'col1' has 3 null values, expected at most 2.",
    )
    assert str(result) == str(
        expected_failure_message
    ), f"Expected failure message but got: {result}"


def test_expectation_pyspark_boundary_zero_threshold(spark):
    """Test PySpark boundary case with 0 threshold."""
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationMaxNullCount",
        column_name="col1",
        max_count=0,
    )
    data_frame = spark.createDataFrame([(1,), (None,), (3,)], ["col1"])
    result = expectation.validate(data_frame=data_frame)

    expected_failure_message = DataframeExpectationFailureMessage(
        expectation_str=str(expectation),
        data_frame_type=DataFrameType.PYSPARK,
        message="Column 'col1' has 1 null values, expected at most 0.",
    )
    assert str(result) == str(
        expected_failure_message
    ), f"Expected failure message but got: {result}"


def test_expectation_pyspark_boundary_zero_threshold_success(spark):
    """Test PySpark boundary case with 0 threshold and no nulls."""
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationMaxNullCount",
        column_name="col1",
        max_count=0,
    )
    data_frame = spark.createDataFrame(
        [
            {"col1": 1, "col2": None},
            {"col1": 2, "col2": None},
            {"col1": 3, "col2": None},
        ],
        schema="col1 int, col2 string",
    )
    result = expectation.validate(data_frame=data_frame)
    assert str(result) == str(
        DataframeExpectationSuccessMessage(expectation_name="ExpectationMaxNullCount")
    ), f"Expected success message but got: {result}"


def test_expectation_pyspark_empty_dataframe(spark):
    """Test PySpark edge case with empty DataFrame."""
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationMaxNullCount",
        column_name="col1",
        max_count=5,
    )
    data_frame = spark.createDataFrame([], "col1 INT")
    result = expectation.validate(data_frame=data_frame)
    # Empty DataFrame should have 0 nulls and pass
    assert str(result) == str(
        DataframeExpectationSuccessMessage(expectation_name="ExpectationMaxNullCount")
    ), f"Expected success message but got: {result}"


def test_expectation_pyspark_single_value_null(spark):
    """Test PySpark edge case with single null value."""
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationMaxNullCount",
        column_name="col1",
        max_count=0,
    )
    data_frame = spark.createDataFrame([{"col1": None}], schema="col1 int")
    result = expectation.validate(data_frame=data_frame)

    expected_failure_message = DataframeExpectationFailureMessage(
        expectation_str=str(expectation),
        data_frame_type=DataFrameType.PYSPARK,
        message="Column 'col1' has 1 null values, expected at most 0.",
    )
    assert str(result) == str(
        expected_failure_message
    ), f"Expected failure message but got: {result}"


def test_expectation_pyspark_single_value_not_null(spark):
    """Test PySpark edge case with single non-null value."""
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationMaxNullCount",
        column_name="col1",
        max_count=0,
    )
    data_frame = spark.createDataFrame([(1,)], ["col1"])
    result = expectation.validate(data_frame=data_frame)
    assert str(result) == str(
        DataframeExpectationSuccessMessage(expectation_name="ExpectationMaxNullCount")
    ), f"Expected success message but got: {result}"


def test_expectation_pyspark_different_column_nulls_not_affecting(spark):
    """Test that nulls in other columns don't affect the result."""
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationMaxNullCount",
        column_name="col1",
        max_count=1,
    )
    # col1 has 0 nulls, col2 has nulls - should pass since we're only checking col1
    data_frame = spark.createDataFrame(
        [
            {"col1": 1, "col2": None},
            {"col1": 2, "col2": None},
            {"col1": 3, "col2": None},
        ],
        schema="col1 int, col2 string",
    )
    result = expectation.validate(data_frame=data_frame)
    assert str(result) == str(
        DataframeExpectationSuccessMessage(expectation_name="ExpectationMaxNullCount")
    ), f"Expected success message but got: {result}"


def test_suite_pandas_success():
    """Test the expectation suite for pandas DataFrame with no violations."""
    expectations_suite = DataframeExpectationsSuite().expect_max_null_count(
        column_name="col1", max_count=2
    )
    data_frame = pd.DataFrame(
        {"col1": [1, None, 3], "col2": ["a", "b", "c"]}
    )  # 1 null value, which is less than max_count of 2
    result = expectations_suite.run(data_frame=data_frame)
    assert result is None, "Expected no exceptions to be raised"


def test_suite_pandas_violations():
    """Test the expectation suite for pandas DataFrame with violations."""
    expectations_suite = DataframeExpectationsSuite().expect_max_null_count(
        column_name="col1", max_count=1
    )
    data_frame = pd.DataFrame(
        {"col1": [1, None, None], "col2": ["a", "b", "c"]}
    )  # 2 null values, which exceeds max_count of 1
    with pytest.raises(DataframeExpectationsSuiteFailure):
        expectations_suite.run(data_frame=data_frame)


def test_suite_pyspark_success(spark):
    """Test the expectation suite for PySpark DataFrame with no violations."""
    expectations_suite = DataframeExpectationsSuite().expect_max_null_count(
        column_name="col1", max_count=2
    )
    data_frame = spark.createDataFrame(
        [(1, "a"), (None, "b"), (3, "c")], ["col1", "col2"]
    )  # 1 null value, which is less than max_count of 2
    result = expectations_suite.run(data_frame=data_frame)
    assert result is None, "Expected no exceptions to be raised"


def test_suite_pyspark_violations(spark):
    """Test the expectation suite for PySpark DataFrame with violations."""
    expectations_suite = DataframeExpectationsSuite().expect_max_null_count(
        column_name="col1", max_count=1
    )
    data_frame = spark.createDataFrame(
        [(1, "a"), (None, "b"), (None, "c")], ["col1", "col2"]
    )  # 2 null values, which exceeds max_count of 1
    with pytest.raises(DataframeExpectationsSuiteFailure):
        expectations_suite.run(data_frame=data_frame)


def test_suite_pyspark_column_missing_error(spark):
    """Test that an error is raised when the specified column is missing in PySpark DataFrame."""
    expectations_suite = DataframeExpectationsSuite().expect_max_null_count(
        column_name="col1", max_count=5
    )
    data_frame = spark.createDataFrame([(1, "a"), (2, "b"), (3, "c")], ["col2", "col3"])
    with pytest.raises(DataframeExpectationsSuiteFailure):
        expectations_suite.run(data_frame=data_frame)


def test_expectation_parameter_validation():
    """Test that appropriate errors are raised for invalid parameters."""
    # Test negative max_count
    with pytest.raises(ValueError) as context:
        DataframeExpectationRegistry.get_expectation(
            expectation_name="ExpectationMaxNullCount",
            column_name="col1",
            max_count=-1,
        )
    assert "max_count must be non-negative" in str(
        context.value
    ), f"Expected 'max_count must be non-negative' in error message: {str(context.value)}"


def test_expectation_mixed_data_types():
    """Test the expectation with mixed data types including nulls."""
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationMaxNullCount",
        column_name="col1",
        max_count=2,
    )
    # Mixed data types with nulls
    data_frame = pd.DataFrame({"col1": [1, "text", None, 3.14, None]})
    result = expectation.validate(data_frame=data_frame)
    assert str(result) == str(
        DataframeExpectationSuccessMessage(expectation_name="ExpectationMaxNullCount")
    ), f"Expected success message but got: {result}"


def test_expectation_large_dataset():
    """Test the expectation with a larger dataset."""
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationMaxNullCount",
        column_name="col1",
        max_count=100,
    )
    # Create a DataFrame with 1000 rows and 50 nulls
    data = [None if i % 20 == 0 else i for i in range(1000)]  # Every 20th value is None
    data_frame = pd.DataFrame({"col1": data})
    result = expectation.validate(data_frame=data_frame)
    assert str(result) == str(
        DataframeExpectationSuccessMessage(expectation_name="ExpectationMaxNullCount")
    ), f"Expected success message but got: {result}"


def test_expectation_large_threshold():
    """Test the expectation with a very large threshold."""
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationMaxNullCount",
        column_name="col1",
        max_count=1000000,
    )
    # Small DataFrame with few nulls should pass with large threshold
    data_frame = pd.DataFrame({"col1": [1, None, 3, None, 5]})
    result = expectation.validate(data_frame=data_frame)
    assert str(result) == str(
        DataframeExpectationSuccessMessage(expectation_name="ExpectationMaxNullCount")
    ), f"Expected success message but got: {result}"


def test_expectation_column_not_exists_error():
    """Test that an error is raised when the specified column does not exist."""
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationMaxNullCount",
        column_name="col1",
        max_count=5,
    )
    data_frame = pd.DataFrame({"col2": [1, 2, 3, 4, 5]})
    result = expectation.validate(data_frame=data_frame)
    # The error message might vary slightly depending on pandas version
    assert isinstance(
        result, DataframeExpectationFailureMessage
    ), f"Expected DataframeExpectationFailureMessage but got: {type(result)}"
    result_str = str(result)
    assert "col1" in result_str, f"Expected 'col1' in result message: {result_str}"
