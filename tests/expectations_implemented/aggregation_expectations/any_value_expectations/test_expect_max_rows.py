import pytest
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


def test_expectation_pandas_success_exact_count():
    """Test pandas success case with exact maximum row count."""
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationMaxRows",
        max_rows=3,
    )
    data_frame = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
    result = expectation.validate(data_frame=data_frame)
    assert str(result) == str(
        DataframeExpectationSuccessMessage(expectation_name="ExpectationMaxRows")
    ), f"Expected success message but got: {result}"


def test_expectation_pandas_success_below_max():
    """Test pandas success case with row count below maximum."""
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationMaxRows",
        max_rows=10,
    )
    data_frame = pd.DataFrame({"col1": [1, 2, 3, 4, 5], "col2": ["a", "b", "c", "d", "e"]})
    result = expectation.validate(data_frame=data_frame)
    assert str(result) == str(
        DataframeExpectationSuccessMessage(expectation_name="ExpectationMaxRows")
    ), f"Expected success message but got: {result}"


def test_expectation_pandas_success_single_row():
    """Test pandas success case with single row and max count of 1."""
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationMaxRows",
        max_rows=1,
    )
    data_frame = pd.DataFrame({"col1": [42]})
    result = expectation.validate(data_frame=data_frame)
    assert str(result) == str(
        DataframeExpectationSuccessMessage(expectation_name="ExpectationMaxRows")
    ), f"Expected success message but got: {result}"


def test_expectation_pandas_success_empty_dataframe():
    """Test pandas success case with empty DataFrame."""
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationMaxRows",
        max_rows=5,
    )
    data_frame = pd.DataFrame({"col1": []})
    result = expectation.validate(data_frame=data_frame)
    assert str(result) == str(
        DataframeExpectationSuccessMessage(expectation_name="ExpectationMaxRows")
    ), f"Expected success message but got: {result}"


def test_expectation_pandas_failure_exceeds_max():
    """Test pandas failure case when row count exceeds maximum."""
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationMaxRows",
        max_rows=3,
    )
    data_frame = pd.DataFrame({"col1": [1, 2, 3, 4, 5], "col2": ["a", "b", "c", "d", "e"]})
    result = expectation.validate(data_frame=data_frame)

    expected_failure_message = DataframeExpectationFailureMessage(
        expectation_str=str(expectation),
        data_frame_type=DataFrameType.PANDAS,
        message="DataFrame has 5 rows, expected at most 3.",
    )
    assert str(result) == str(
        expected_failure_message
    ), f"Expected failure message but got: {result}"


def test_expectation_pandas_failure_zero_max_with_data():
    """Test pandas failure case with zero max count but data present."""
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationMaxRows",
        max_rows=0,
    )
    data_frame = pd.DataFrame({"col1": [1]})
    result = expectation.validate(data_frame=data_frame)

    expected_failure_message = DataframeExpectationFailureMessage(
        expectation_str=str(expectation),
        data_frame_type=DataFrameType.PANDAS,
        message="DataFrame has 1 rows, expected at most 0.",
    )
    assert str(result) == str(
        expected_failure_message
    ), f"Expected failure message but got: {result}"


def test_expectation_pandas_boundary_zero_max_empty_df():
    """Test pandas boundary case with zero max count and empty DataFrame."""
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationMaxRows",
        max_rows=0,
    )
    data_frame = pd.DataFrame({"col1": []})
    result = expectation.validate(data_frame=data_frame)
    assert str(result) == str(
        DataframeExpectationSuccessMessage(expectation_name="ExpectationMaxRows")
    ), f"Expected success message but got: {result}"


def test_expectation_pandas_large_dataset():
    """Test pandas with larger dataset exceeding maximum."""
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationMaxRows",
        max_rows=100,
    )
    # Create DataFrame with 150 rows
    data_frame = pd.DataFrame({"col1": range(150), "col2": [f"value_{i}" for i in range(150)]})
    result = expectation.validate(data_frame=data_frame)

    expected_failure_message = DataframeExpectationFailureMessage(
        expectation_str=str(expectation),
        data_frame_type=DataFrameType.PANDAS,
        message="DataFrame has 150 rows, expected at most 100.",
    )
    assert str(result) == str(
        expected_failure_message
    ), f"Expected failure message but got: {result}"


def test_expectation_pandas_with_nulls():
    """Test pandas expectation with null values (should still count rows)."""
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationMaxRows",
        max_rows=4,
    )
    data_frame = pd.DataFrame({"col1": [1, None, 3, None, 5], "col2": [None, "b", None, "d", None]})
    result = expectation.validate(data_frame=data_frame)

    expected_failure_message = DataframeExpectationFailureMessage(
        expectation_str=str(expectation),
        data_frame_type=DataFrameType.PANDAS,
        message="DataFrame has 5 rows, expected at most 4.",
    )
    assert str(result) == str(
        expected_failure_message
    ), f"Expected failure message but got: {result}"


def test_expectation_pyspark_success_exact_count(spark):
    """Test PySpark success case with exact maximum row count."""
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationMaxRows",
        max_rows=3,
    )
    data_frame = spark.createDataFrame([(1, "a"), (2, "b"), (3, "c")], ["col1", "col2"])
    result = expectation.validate(data_frame=data_frame)
    assert str(result) == str(
        DataframeExpectationSuccessMessage(expectation_name="ExpectationMaxRows")
    ), f"Expected success message but got: {result}"


def test_expectation_pyspark_success_below_max(spark):
    """Test PySpark success case with row count below maximum."""
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationMaxRows",
        max_rows=10,
    )
    data_frame = spark.createDataFrame(
        [(1, "a"), (2, "b"), (3, "c"), (4, "d"), (5, "e")], ["col1", "col2"]
    )
    result = expectation.validate(data_frame=data_frame)
    assert str(result) == str(
        DataframeExpectationSuccessMessage(expectation_name="ExpectationMaxRows")
    ), f"Expected success message but got: {result}"


def test_expectation_pyspark_success_single_row(spark):
    """Test PySpark success case with single row."""
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationMaxRows",
        max_rows=1,
    )
    data_frame = spark.createDataFrame([(42,)], ["col1"])
    result = expectation.validate(data_frame=data_frame)
    assert str(result) == str(
        DataframeExpectationSuccessMessage(expectation_name="ExpectationMaxRows")
    ), f"Expected success message but got: {result}"


def test_expectation_pyspark_success_empty_dataframe(spark):
    """Test PySpark success case with empty DataFrame."""
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationMaxRows",
        max_rows=5,
    )
    # Create empty DataFrame with schema
    data_frame = spark.createDataFrame([], "col1: int")
    result = expectation.validate(data_frame=data_frame)
    assert str(result) == str(
        DataframeExpectationSuccessMessage(expectation_name="ExpectationMaxRows")
    ), f"Expected success message but got: {result}"


def test_expectation_pyspark_failure_exceeds_max(spark):
    """Test PySpark failure case when row count exceeds maximum."""
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationMaxRows",
        max_rows=3,
    )
    data_frame = spark.createDataFrame(
        [(1, "a"), (2, "b"), (3, "c"), (4, "d"), (5, "e")], ["col1", "col2"]
    )
    result = expectation.validate(data_frame=data_frame)

    expected_failure_message = DataframeExpectationFailureMessage(
        expectation_str=str(expectation),
        data_frame_type=DataFrameType.PYSPARK,
        message="DataFrame has 5 rows, expected at most 3.",
    )
    assert str(result) == str(
        expected_failure_message
    ), f"Expected failure message but got: {result}"


def test_expectation_pyspark_failure_zero_max_with_data(spark):
    """Test PySpark failure case with zero max count but data present."""
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationMaxRows",
        max_rows=0,
    )
    data_frame = spark.createDataFrame([(1,)], ["col1"])
    result = expectation.validate(data_frame=data_frame)

    expected_failure_message = DataframeExpectationFailureMessage(
        expectation_str=str(expectation),
        data_frame_type=DataFrameType.PYSPARK,
        message="DataFrame has 1 rows, expected at most 0.",
    )
    assert str(result) == str(
        expected_failure_message
    ), f"Expected failure message but got: {result}"


def test_expectation_pyspark_boundary_zero_max_empty_df(spark):
    """Test PySpark boundary case with zero max count and empty DataFrame."""
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationMaxRows",
        max_rows=0,
    )
    data_frame = spark.createDataFrame([], "col1: int")
    result = expectation.validate(data_frame=data_frame)
    assert str(result) == str(
        DataframeExpectationSuccessMessage(expectation_name="ExpectationMaxRows")
    ), f"Expected success message but got: {result}"


def test_expectation_pyspark_large_dataset(spark):
    """Test PySpark with larger dataset exceeding maximum."""
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationMaxRows",
        max_rows=50,
    )
    # Create DataFrame with 75 rows
    data = [(i, f"value_{i}") for i in range(75)]
    data_frame = spark.createDataFrame(data, ["col1", "col2"])
    result = expectation.validate(data_frame=data_frame)

    expected_failure_message = DataframeExpectationFailureMessage(
        expectation_str=str(expectation),
        data_frame_type=DataFrameType.PYSPARK,
        message="DataFrame has 75 rows, expected at most 50.",
    )
    assert str(result) == str(
        expected_failure_message
    ), f"Expected failure message but got: {result}"


def test_expectation_pyspark_with_nulls(spark):
    """Test PySpark expectation with null values (should still count rows)."""
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationMaxRows",
        max_rows=4,
    )
    data_frame = spark.createDataFrame(
        [(1, None), (None, "b"), (3, None), (None, "d"), (5, None)],
        ["col1", "col2"],
    )
    result = expectation.validate(data_frame=data_frame)

    expected_failure_message = DataframeExpectationFailureMessage(
        expectation_str=str(expectation),
        data_frame_type=DataFrameType.PYSPARK,
        message="DataFrame has 5 rows, expected at most 4.",
    )
    assert str(result) == str(
        expected_failure_message
    ), f"Expected failure message but got: {result}"


def test_suite_pandas_success():
    """Test integration with expectations suite for pandas success case."""
    expectations_suite = DataframeExpectationsSuite().expect_max_rows(max_rows=5)
    data_frame = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
    result = expectations_suite.run(data_frame=data_frame)
    assert result is None, "Expected no exceptions to be raised"


def test_suite_pandas_violations():
    """Test integration with expectations suite for pandas failure case."""
    expectations_suite = DataframeExpectationsSuite().expect_max_rows(max_rows=2)
    data_frame = pd.DataFrame({"col1": [1, 2, 3, 4], "col2": ["a", "b", "c", "d"]})
    with pytest.raises(DataframeExpectationsSuiteFailure):
        expectations_suite.run(data_frame=data_frame)


def test_suite_pyspark_success(spark):
    """Test integration with expectations suite for PySpark success case."""
    expectations_suite = DataframeExpectationsSuite().expect_max_rows(max_rows=5)
    data_frame = spark.createDataFrame([(1, "a"), (2, "b"), (3, "c")], ["col1", "col2"])
    result = expectations_suite.run(data_frame=data_frame)
    assert result is None, "Expected no exceptions to be raised"


def test_suite_pyspark_violations(spark):
    """Test integration with expectations suite for PySpark failure case."""
    expectations_suite = DataframeExpectationsSuite().expect_max_rows(max_rows=2)
    data_frame = spark.createDataFrame([(1, "a"), (2, "b"), (3, "c"), (4, "d")], ["col1", "col2"])
    with pytest.raises(DataframeExpectationsSuiteFailure):
        expectations_suite.run(data_frame=data_frame)


def test_expectation_parameter_validation():
    """Test parameter validation for max_rows."""
    # Test with valid parameters
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationMaxRows",
        max_rows=100,
    )
    assert expectation is not None, "Expected expectation to be created successfully"

    # Test string representation
    expectation_str = str(expectation)
    assert "100" in expectation_str, f"Expected '100' in expectation string: {expectation_str}"
    assert (
        "ExpectationMaxRows" in expectation_str
    ), f"Expected 'ExpectationMaxRows' in expectation string: {expectation_str}"


def test_expectation_boundary_conditions():
    """Test various boundary conditions for max_rows."""
    # Test with max_rows = 1
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationMaxRows",
        max_rows=1,
    )

    # Single row - should pass
    data_frame = pd.DataFrame({"col1": [1]})
    result = expectation.validate(data_frame=data_frame)
    assert str(result) == str(
        DataframeExpectationSuccessMessage(expectation_name="ExpectationMaxRows")
    ), f"Expected success message but got: {result}"

    # Two rows - should fail
    data_frame = pd.DataFrame({"col1": [1, 2]})
    result = expectation.validate(data_frame=data_frame)

    expected_failure_message = DataframeExpectationFailureMessage(
        expectation_str=str(expectation),
        data_frame_type=DataFrameType.PANDAS,
        message="DataFrame has 2 rows, expected at most 1.",
    )
    assert str(result) == str(
        expected_failure_message
    ), f"Expected failure message but got: {result}"


def test_expectation_multiple_columns():
    """Test expectation with multiple columns (should still count total rows)."""
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationMaxRows",
        max_rows=3,
    )
    data_frame = pd.DataFrame(
        {
            "col1": [1, 2, 3, 4],
            "col2": ["a", "b", "c", "d"],
            "col3": [1.1, 2.2, 3.3, 4.4],
            "col4": [True, False, True, False],
        }
    )
    result = expectation.validate(data_frame=data_frame)

    expected_failure_message = DataframeExpectationFailureMessage(
        expectation_str=str(expectation),
        data_frame_type=DataFrameType.PANDAS,
        message="DataFrame has 4 rows, expected at most 3.",
    )
    assert str(result) == str(
        expected_failure_message
    ), f"Expected failure message but got: {result}"


def test_expectation_mixed_data_types():
    """Test expectation with mixed data types in columns."""
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationMaxRows",
        max_rows=10,
    )
    data_frame = pd.DataFrame(
        {
            "int_col": [1, 2, 3, 4, 5],
            "str_col": ["a", "b", "c", "d", "e"],
            "float_col": [1.1, 2.2, 3.3, 4.4, 5.5],
            "bool_col": [True, False, True, False, True],
            "null_col": [None, None, None, None, None],
        }
    )
    result = expectation.validate(data_frame=data_frame)
    assert str(result) == str(
        DataframeExpectationSuccessMessage(expectation_name="ExpectationMaxRows")
    ), f"Expected success message but got: {result}"


def test_expectation_high_max_rows():
    """Test expectation with very high max_rows value."""
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationMaxRows",
        max_rows=1000000,  # 1 million
    )
    data_frame = pd.DataFrame({"col1": [1, 2, 3]})
    result = expectation.validate(data_frame=data_frame)
    assert str(result) == str(
        DataframeExpectationSuccessMessage(expectation_name="ExpectationMaxRows")
    ), f"Expected success message but got: {result}"


def test_expectation_identical_values():
    """Test expectation with DataFrame containing identical values."""
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationMaxRows",
        max_rows=3,
    )
    data_frame = pd.DataFrame(
        {
            "col1": [42, 42, 42, 42],  # All same values
            "col2": ["same", "same", "same", "same"],
        }
    )
    result = expectation.validate(data_frame=data_frame)

    expected_failure_message = DataframeExpectationFailureMessage(
        expectation_str=str(expectation),
        data_frame_type=DataFrameType.PANDAS,
        message="DataFrame has 4 rows, expected at most 3.",
    )
    assert str(result) == str(
        expected_failure_message
    ), f"Expected failure message but got: {result}"


def test_expectation_edge_case_max_rows_equals_actual():
    """Test edge case where max_rows exactly equals actual row count."""
    for count in [1, 5, 10, 100]:
        expectation = DataframeExpectationRegistry.get_expectation(
            expectation_name="ExpectationMaxRows",
            max_rows=count,
        )
        # Create DataFrame with exactly 'count' rows
        data_frame = pd.DataFrame({"col1": list(range(count))})
        result = expectation.validate(data_frame=data_frame)
        assert str(result) == str(
            DataframeExpectationSuccessMessage(expectation_name="ExpectationMaxRows")
        ), f"Expected success message for count {count} but got: {result}"
