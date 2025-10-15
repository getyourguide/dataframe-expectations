import numpy as np
import pandas as pd
import pytest

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
        expectation_name="ExpectationMaxNullPercentage",
        column_name="col1",
        max_percentage=10.0,
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
        DataframeExpectationSuccessMessage(
            expectation_name="ExpectationMaxNullPercentage"
        )
    ), f"Expected success message but got: {result}"

def test_expectation_pandas_success_within_threshold():
    """Test pandas success case with null percentage within threshold."""
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationMaxNullPercentage",
        column_name="col1",
        max_percentage=25.0,
    )
    # 4 values in col1, 1 null = 25% null
    data_frame = pd.DataFrame(
        {
            "col1": [1, None, 3, 4],
            "col2": ["a", "b", "c", "d"],  # Other columns don't affect the test
        }
    )
    result = expectation.validate(data_frame=data_frame)
    assert str(result) == str(
        DataframeExpectationSuccessMessage(
            expectation_name="ExpectationMaxNullPercentage"
        )
    ), f"Expected success message but got: {result}"

def test_expectation_pandas_success_exactly_at_threshold():
    """Test pandas success case with null percentage exactly at threshold."""
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationMaxNullPercentage",
        column_name="col1",
        max_percentage=20.0,
    )
    # 5 values in col1, 1 null = 20% null
    data_frame = pd.DataFrame(
        {"col1": [1, 2, None, 4, 5], "col2": [None, "b", "c", "d", "e"]}
    )
    result = expectation.validate(data_frame=data_frame)
    assert str(result) == str(
        DataframeExpectationSuccessMessage(
            expectation_name="ExpectationMaxNullPercentage"
        )
    ), f"Expected success message but got: {result}"

def test_expectation_pandas_success_with_nan():
    """Test pandas success case with NaN values within threshold."""
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationMaxNullPercentage",
        column_name="col2",
        max_percentage=50.0,
    )
    # 3 values in col2, 1 NaN = 33.33% null (less than 50%)
    data_frame = pd.DataFrame({"col1": [1, 2, 3], "col2": [4.0, np.nan, 6.0]})
    result = expectation.validate(data_frame=data_frame)
    assert str(result) == str(
        DataframeExpectationSuccessMessage(
            expectation_name="ExpectationMaxNullPercentage"
        )
    ), f"Expected success message but got: {result}"

def test_expectation_pandas_failure_exceeds_threshold():
    """Test pandas failure case when null percentage exceeds threshold."""
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationMaxNullPercentage",
        column_name="col1",
        max_percentage=20.0,
    )
    # 4 values in col1, 2 nulls = 50% null (exceeds 20%)
    data_frame = pd.DataFrame(
        {"col1": [1, None, 3, None], "col2": [None, "b", "c", "d"]}
    )
    result = expectation.validate(data_frame=data_frame)

    expected_failure_message = DataframeExpectationFailureMessage(
        expectation_str=str(expectation),
        data_frame_type=DataFrameType.PANDAS,
        message="Column 'col1' has 50.00% null values, expected at most 20.00%.",
    )
    assert str(result) == str(expected_failure_message), f"Expected failure message but got: {result}"

def test_expectation_pandas_failure_all_nulls_in_column():
    """Test pandas failure case with 100% null values in the specified column."""
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationMaxNullPercentage",
        column_name="col1",
        max_percentage=50.0,
    )
    data_frame = pd.DataFrame({"col1": [None, None], "col2": [1, 2]})
    result = expectation.validate(data_frame=data_frame)

    expected_failure_message = DataframeExpectationFailureMessage(
        expectation_str=str(expectation),
        data_frame_type=DataFrameType.PANDAS,
        message="Column 'col1' has 100.00% null values, expected at most 50.00%.",
    )
    assert str(result) == str(expected_failure_message), f"Expected failure message but got: {result}"

def test_expectation_pandas_boundary_zero_threshold():
    """Test pandas boundary case with 0.0% threshold."""
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationMaxNullPercentage",
        column_name="col1",
        max_percentage=0.0,
    )
    data_frame = pd.DataFrame({"col1": [1, None, 3]})
    result = expectation.validate(data_frame=data_frame)

    expected_failure_message = DataframeExpectationFailureMessage(
        expectation_str=str(expectation),
        data_frame_type=DataFrameType.PANDAS,
        message="Column 'col1' has 33.33% null values, expected at most 0.00%.",
    )
    assert str(result) == str(expected_failure_message), f"Expected failure message but got: {result}"

def test_expectation_pandas_boundary_hundred_threshold():
    """Test pandas boundary case with 100.0% threshold."""
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationMaxNullPercentage",
        column_name="col1",
        max_percentage=100.0,
    )
    data_frame = pd.DataFrame(
        {"col1": [None, None, None], "col2": [None, None, None]}
    )
    result = expectation.validate(data_frame=data_frame)
    assert str(result) == str(
        DataframeExpectationSuccessMessage(
            expectation_name="ExpectationMaxNullPercentage"
        )
    ), f"Expected success message but got: {result}"

def test_expectation_pandas_empty_dataframe():
    """Test pandas edge case with empty DataFrame."""
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationMaxNullPercentage",
        column_name="col1",
        max_percentage=10.0,
    )
    data_frame = pd.DataFrame(columns=["col1"])
    result = expectation.validate(data_frame=data_frame)
    # Empty DataFrame should have 0% nulls and pass
    assert str(result) == str(
        DataframeExpectationSuccessMessage(
            expectation_name="ExpectationMaxNullPercentage"
        )
    ), f"Expected success message but got: {result}"

def test_expectation_pandas_single_value_null():
    """Test pandas edge case with single null value."""
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationMaxNullPercentage",
        column_name="col1",
        max_percentage=50.0,
    )
    data_frame = pd.DataFrame({"col1": [None]})
    result = expectation.validate(data_frame=data_frame)

    expected_failure_message = DataframeExpectationFailureMessage(
        expectation_str=str(expectation),
        data_frame_type=DataFrameType.PANDAS,
        message="Column 'col1' has 100.00% null values, expected at most 50.00%.",
    )
    assert str(result) == str(expected_failure_message), f"Expected failure message but got: {result}"

def test_expectation_pandas_single_value_not_null():
    """Test pandas edge case with single non-null value."""
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationMaxNullPercentage",
        column_name="col1",
        max_percentage=10.0,
    )
    data_frame = pd.DataFrame({"col1": [1]})
    result = expectation.validate(data_frame=data_frame)
    assert str(result) == str(
        DataframeExpectationSuccessMessage(
            expectation_name="ExpectationMaxNullPercentage"
        )
    ), f"Expected success message but got: {result}"

def test_expectation_pandas_different_column_nulls_not_affecting():
    """Test that nulls in other columns don't affect the result."""
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationMaxNullPercentage",
        column_name="col1",
        max_percentage=10.0,
    )
    # col1 has 0% nulls, col2 has 100% nulls - should pass since we're only checking col1
    data_frame = pd.DataFrame({"col1": [1, 2, 3], "col2": [None, None, None]})
    result = expectation.validate(data_frame=data_frame)
    assert str(result) == str(
        DataframeExpectationSuccessMessage(
            expectation_name="ExpectationMaxNullPercentage"
        )
    ), f"Expected success message but got: {result}"

def test_expectation_pyspark_success_no_nulls(spark):
    """Test PySpark success case with no null values."""
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationMaxNullPercentage",
        column_name="col1",
        max_percentage=10.0,
    )
    data_frame = spark.createDataFrame(
        [(1, "a", 1.1), (2, "b", 2.2), (3, "c", 3.3), (4, "d", 4.4), (5, "e", 5.5)],
        ["col1", "col2", "col3"],
    )
    result = expectation.validate(data_frame=data_frame)
    assert str(result) == str(
        DataframeExpectationSuccessMessage(
            expectation_name="ExpectationMaxNullPercentage"
        )
    ), f"Expected success message but got: {result}"

def test_expectation_pyspark_success_within_threshold(spark):
    """Test PySpark success case with null percentage within threshold."""
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationMaxNullPercentage",
        column_name="col1",
        max_percentage=30.0,
    )
    # 4 values in col1, 1 null = 25% null
    data_frame = spark.createDataFrame([(1,), (None,), (3,), (4,)], ["col1"])
    result = expectation.validate(data_frame=data_frame)
    assert str(result) == str(
        DataframeExpectationSuccessMessage(
            expectation_name="ExpectationMaxNullPercentage"
        )
    ), f"Expected success message but got: {result}"

def test_expectation_pyspark_success_exactly_at_threshold(spark):
    """Test PySpark success case with null percentage exactly at threshold."""
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationMaxNullPercentage",
        column_name="col1",
        max_percentage=40.0,
    )
    # 5 values in col1, 2 nulls = 40% null
    data_frame = spark.createDataFrame(
        [(1, "a"), (2, None), (None, "c"), (4, "d"), (None, None)], ["col1", "col2"]
    )
    result = expectation.validate(data_frame=data_frame)
    assert str(result) == str(
        DataframeExpectationSuccessMessage(
            expectation_name="ExpectationMaxNullPercentage"
        )
    ), f"Expected success message but got: {result}"

def test_expectation_pyspark_failure_exceeds_threshold(spark):
    """Test PySpark failure case when null percentage exceeds threshold."""
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationMaxNullPercentage",
        column_name="col1",
        max_percentage=25.0,
    )
    # 3 values in col1, 2 nulls = 66.67% null (exceeds 25%)
    data_frame = spark.createDataFrame(
        [(1, None), (None, "b"), (None, "c")], ["col1", "col2"]
    )
    result = expectation.validate(data_frame=data_frame)

    expected_failure_message = DataframeExpectationFailureMessage(
        expectation_str=str(expectation),
        data_frame_type=DataFrameType.PYSPARK,
        message="Column 'col1' has 66.67% null values, expected at most 25.00%.",
    )
    assert str(result) == str(expected_failure_message), f"Expected failure message but got: {result}"

def test_expectation_pyspark_failure_all_nulls_in_column(spark):
    """Test PySpark failure case with 100% null values in the specified column."""
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationMaxNullPercentage",
        column_name="col1",
        max_percentage=75.0,
    )
    data_frame = spark.createDataFrame([(None,), (None,), (None,)], "col1: int")
    result = expectation.validate(data_frame=data_frame)

    expected_failure_message = DataframeExpectationFailureMessage(
        expectation_str=str(expectation),
        data_frame_type=DataFrameType.PYSPARK,
        message="Column 'col1' has 100.00% null values, expected at most 75.00%.",
    )
    assert str(result) == str(expected_failure_message), f"Expected failure message but got: {result}"

def test_expectation_pyspark_boundary_zero_threshold(spark):
    """Test PySpark boundary case with 0.0% threshold."""
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationMaxNullPercentage",
        column_name="col1",
        max_percentage=0.0,
    )
    data_frame = spark.createDataFrame([(1,), (None,), (3,)], ["col1"])
    result = expectation.validate(data_frame=data_frame)

    expected_failure_message = DataframeExpectationFailureMessage(
        expectation_str=str(expectation),
        data_frame_type=DataFrameType.PYSPARK,
        message="Column 'col1' has 33.33% null values, expected at most 0.00%.",
    )
    assert str(result) == str(expected_failure_message), f"Expected failure message but got: {result}"

def test_expectation_pyspark_boundary_hundred_threshold(spark):
    """Test PySpark boundary case with 100.0% threshold."""
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationMaxNullPercentage",
        column_name="col1",
        max_percentage=100.0,
    )
    data_frame = spark.createDataFrame(
        [
            {"col1": None, "col2": None},
            {"col1": None, "col2": None},
        ],
        schema="col1: int, col2: string",
    )
    result = expectation.validate(data_frame=data_frame)
    assert str(result) == str(
        DataframeExpectationSuccessMessage(
            expectation_name="ExpectationMaxNullPercentage"
        )
    ), f"Expected success message but got: {result}"

def test_expectation_pyspark_empty_dataframe(spark):
    """Test PySpark edge case with empty DataFrame."""
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationMaxNullPercentage",
        column_name="col1",
        max_percentage=10.0,
    )
    # Create empty DataFrame with schema
    data_frame = spark.createDataFrame([], "col1: int")
    result = expectation.validate(data_frame=data_frame)
    # Empty DataFrame should have 0% nulls and pass
    assert str(result) == str(
        DataframeExpectationSuccessMessage(
            expectation_name="ExpectationMaxNullPercentage"
        )
    ), f"Expected success message but got: {result}"

def test_expectation_pyspark_single_value_null(spark):
    """Test PySpark edge case with single null value."""
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationMaxNullPercentage",
        column_name="col1",
        max_percentage=50.0,
    )
    data_frame = spark.createDataFrame([(None,)], "col1: int")
    result = expectation.validate(data_frame=data_frame)

    expected_failure_message = DataframeExpectationFailureMessage(
        expectation_str=str(expectation),
        data_frame_type=DataFrameType.PYSPARK,
        message="Column 'col1' has 100.00% null values, expected at most 50.00%.",
    )
    assert str(result) == str(expected_failure_message), f"Expected failure message but got: {result}"

def test_expectation_pyspark_single_value_not_null(spark):
    """Test PySpark edge case with single non-null value."""
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationMaxNullPercentage",
        column_name="col1",
        max_percentage=10.0,
    )
    data_frame = spark.createDataFrame([(1,)], ["col1"])
    result = expectation.validate(data_frame=data_frame)
    assert str(result) == str(
        DataframeExpectationSuccessMessage(
            expectation_name="ExpectationMaxNullPercentage"
        )
    ), f"Expected success message but got: {result}"

def test_expectation_pyspark_different_column_nulls_not_affecting(spark):
    """Test that nulls in other columns don't affect the result."""
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationMaxNullPercentage",
        column_name="col1",
        max_percentage=10.0,
    )
    # col1 has 0% nulls, col2 has 100% nulls - should pass since we're only checking col1
    data_frame = spark.createDataFrame(
        [
            {"col1": 1, "col2": None},
            {"col1": 2, "col2": None},
            {"col1": 3, "col2": None},
        ],
        schema="col1: int, col2: int",
    )
    result = expectation.validate(data_frame=data_frame)
    assert str(result) == str(
        DataframeExpectationSuccessMessage(
            expectation_name="ExpectationMaxNullPercentage"
        )
    ), f"Expected success message but got: {result}"

def test_suite_pandas_success():
    """Test integration with expectations suite for pandas success case."""
    expectations_suite = DataframeExpectationsSuite().expect_max_null_percentage(
        column_name="col1", max_percentage=30.0
    )
    # 4 values in col1, 1 nulls = 25% null (should pass)
    data_frame = pd.DataFrame(
        {"col1": [1, 2, None, 4], "col2": ["a", "b", "c", "d"]}
    )
    expectations_suite.run(data_frame=data_frame)

def test_suite_pandas_violations():
    """Test integration with expectations suite for pandas failure case."""
    expectations_suite = DataframeExpectationsSuite().expect_max_null_percentage(
        column_name="col1", max_percentage=10.0
    )
    # 2 values in col1, 1 null = 50% null (exceeds 10%)
    data_frame = pd.DataFrame({"col1": [1, None], "col2": ["a", "b"]})
    with pytest.raises(DataframeExpectationsSuiteFailure):
        expectations_suite.run(data_frame=data_frame)

def test_suite_pyspark_success(spark):
    """Test integration with expectations suite for PySpark success case."""
    expectations_suite = DataframeExpectationsSuite().expect_max_null_percentage(
        column_name="col1", max_percentage=50.0
    )
    # 2 values in col1, 1 null = 50% null (equals 50%)
    data_frame = spark.createDataFrame([(1, "a"), (None, "b")], ["col1", "col2"])
    result = expectations_suite.run(data_frame=data_frame)
    assert result is None, "Expected no exceptions to be raised"

def test_suite_pyspark_violations(spark):
    """Test integration with expectations suite for PySpark failure case."""
    expectations_suite = DataframeExpectationsSuite().expect_max_null_percentage(
        column_name="col1", max_percentage=20.0
    )
    # 2 values in col1, 1 null = 50% null (exceeds 20%)
    data_frame = spark.createDataFrame([(None, "a"), (2, None)], ["col1", "col2"])
    with pytest.raises(DataframeExpectationsSuiteFailure):
        expectations_suite.run(data_frame=data_frame)

def test_expectation_parameter_validation():
    """Test parameter validation for column_name and max_percentage."""
    # Test with valid parameters
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationMaxNullPercentage",
        column_name="test_col",
        max_percentage=50.0,
    )
    assert expectation is not None, "Expected expectation to be created successfully"

    # Test string representation
    expectation_str = str(expectation)
    assert "50.0" in expectation_str, f"Expected '50.0' in expectation string: {expectation_str}"
    assert "test_col" in expectation_str, f"Expected 'test_col' in expectation string: {expectation_str}"
    assert "ExpectationMaxNullPercentage" in expectation_str, f"Expected 'ExpectationMaxNullPercentage' in expectation string: {expectation_str}"

def test_expectation_mixed_data_types():
    """Test expectation with mixed data types including various null representations."""
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationMaxNullPercentage",
        column_name="float_col",
        max_percentage=50.0,
    )
    # 4 values in float_col, 1 NaN = 25% null (less than 50%)
    data_frame = pd.DataFrame(
        {
            "int_col": [1, None, 3, 4],
            "str_col": ["a", "b", None, "d"],
            "float_col": [1.1, 2.2, 3.3, np.nan],
        }
    )
    result = expectation.validate(data_frame=data_frame)
    assert str(result) == str(
        DataframeExpectationSuccessMessage(
            expectation_name="ExpectationMaxNullPercentage"
        )
    ), f"Expected success message but got: {result}"

def test_expectation_precision_boundary():
    """Test expectation with very precise percentage boundaries."""
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationMaxNullPercentage",
        column_name="col1",
        max_percentage=25.0,
    )
    # 4 values in col1, 1 null = 25.00% null (exactly at boundary)
    data_frame = pd.DataFrame({"col1": [1, None, 3, 4]})
    result = expectation.validate(data_frame=data_frame)
    assert str(result) == str(
        DataframeExpectationSuccessMessage(
            expectation_name="ExpectationMaxNullPercentage"
        )
    ), f"Expected success message but got: {result}"

def test_expectation_column_not_exists_error():
    """Test expectation with non-existent column should fail gracefully."""
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationMaxNullPercentage",
        column_name="nonexistent_col",
        max_percentage=50.0,
    )
    data_frame = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
    result = expectation.validate(data_frame=data_frame)

    # Should get a failure message with error info
    assert isinstance(result, DataframeExpectationFailureMessage), f"Expected DataframeExpectationFailureMessage but got: {type(result)}"
    result_str = str(result)
    assert "nonexistent_col" in result_str, f"Expected 'nonexistent_col' in result message: {result_str}"
