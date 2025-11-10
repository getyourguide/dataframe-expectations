import pytest
import pandas as pd

from dataframe_expectations.core.types import DataFrameType
from dataframe_expectations.registry import (
    DataFrameExpectationRegistry,
)
from dataframe_expectations.suite import (
    DataFrameExpectationsSuite,
    DataFrameExpectationsSuiteFailure,
)
from dataframe_expectations.result_message import (
    DataFrameExpectationFailureMessage,
    DataFrameExpectationSuccessMessage,
)


def test_expectation_pandas_success_exact_count():
    """Test pandas success case with exact minimum row count."""
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationMinRows",
        min_rows=3,
    )
    data_frame = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
    result = expectation.validate(data_frame=data_frame)
    assert str(result) == str(
        DataFrameExpectationSuccessMessage(expectation_name="ExpectationMinRows")
    ), f"Expected success message but got: {result}"


def test_expectation_pandas_success_above_min():
    """Test pandas success case with row count above minimum."""
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationMinRows",
        min_rows=3,
    )
    data_frame = pd.DataFrame({"col1": [1, 2, 3, 4, 5], "col2": ["a", "b", "c", "d", "e"]})
    result = expectation.validate(data_frame=data_frame)
    assert str(result) == str(
        DataFrameExpectationSuccessMessage(expectation_name="ExpectationMinRows")
    ), f"Expected success message but got: {result}"


def test_expectation_pandas_success_single_row():
    """Test pandas success case with single row and min count of 1."""
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationMinRows",
        min_rows=1,
    )
    data_frame = pd.DataFrame({"col1": [42]})
    result = expectation.validate(data_frame=data_frame)
    assert str(result) == str(
        DataFrameExpectationSuccessMessage(expectation_name="ExpectationMinRows")
    ), f"Expected success message but got: {result}"


def test_expectation_pandas_success_zero_min_empty_df():
    """Test pandas success case with zero minimum and empty DataFrame."""
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationMinRows",
        min_rows=0,
    )
    data_frame = pd.DataFrame({"col1": []})
    result = expectation.validate(data_frame=data_frame)
    assert str(result) == str(
        DataFrameExpectationSuccessMessage(expectation_name="ExpectationMinRows")
    ), f"Expected success message but got: {result}"


def test_expectation_pandas_success_zero_min_with_data():
    """Test pandas success case with zero minimum and data present."""
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationMinRows",
        min_rows=0,
    )
    data_frame = pd.DataFrame({"col1": [1, 2, 3]})
    result = expectation.validate(data_frame=data_frame)
    assert str(result) == str(
        DataFrameExpectationSuccessMessage(expectation_name="ExpectationMinRows")
    ), f"Expected success message but got: {result}"


def test_expectation_pandas_failure_below_min():
    """Test pandas failure case when row count is below minimum."""
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationMinRows",
        min_rows=5,
    )
    data_frame = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
    result = expectation.validate(data_frame=data_frame)

    expected_failure_message = DataFrameExpectationFailureMessage(
        expectation_str=str(expectation),
        data_frame_type=DataFrameType.PANDAS,
        message="DataFrame has 3 rows, expected at least 5.",
    )
    assert str(result) == str(expected_failure_message), (
        f"Expected failure message but got: {result}"
    )


def test_expectation_pandas_failure_empty_with_min():
    """Test pandas failure case with empty DataFrame but minimum required."""
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationMinRows",
        min_rows=2,
    )
    data_frame = pd.DataFrame({"col1": []})
    result = expectation.validate(data_frame=data_frame)

    expected_failure_message = DataFrameExpectationFailureMessage(
        expectation_str=str(expectation),
        data_frame_type=DataFrameType.PANDAS,
        message="DataFrame has 0 rows, expected at least 2.",
    )
    assert str(result) == str(expected_failure_message), (
        f"Expected failure message but got: {result}"
    )


def test_expectation_pandas_failure_single_row_needs_more():
    """Test pandas failure case with single row but higher minimum required."""
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationMinRows",
        min_rows=3,
    )
    data_frame = pd.DataFrame({"col1": [1]})
    result = expectation.validate(data_frame=data_frame)

    expected_failure_message = DataFrameExpectationFailureMessage(
        expectation_str=str(expectation),
        data_frame_type=DataFrameType.PANDAS,
        message="DataFrame has 1 rows, expected at least 3.",
    )
    assert str(result) == str(expected_failure_message), (
        f"Expected failure message but got: {result}"
    )


def test_expectation_pandas_large_dataset():
    """Test pandas with larger dataset meeting minimum."""
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationMinRows",
        min_rows=100,
    )
    # Create DataFrame with 150 rows
    data_frame = pd.DataFrame({"col1": range(150), "col2": [f"value_{i}" for i in range(150)]})
    result = expectation.validate(data_frame=data_frame)
    assert str(result) == str(
        DataFrameExpectationSuccessMessage(expectation_name="ExpectationMinRows")
    ), f"Expected success message but got: {result}"


def test_expectation_pandas_large_dataset_failure():
    """Test pandas with dataset not meeting large minimum."""
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationMinRows",
        min_rows=200,
    )
    # Create DataFrame with 150 rows
    data_frame = pd.DataFrame({"col1": range(150), "col2": [f"value_{i}" for i in range(150)]})
    result = expectation.validate(data_frame=data_frame)

    expected_failure_message = DataFrameExpectationFailureMessage(
        expectation_str=str(expectation),
        data_frame_type=DataFrameType.PANDAS,
        message="DataFrame has 150 rows, expected at least 200.",
    )
    assert str(result) == str(expected_failure_message), (
        f"Expected failure message but got: {result}"
    )


def test_expectation_pandas_with_nulls():
    """Test pandas expectation with null values (should still count rows)."""
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationMinRows",
        min_rows=3,
    )
    data_frame = pd.DataFrame({"col1": [1, None, 3, None, 5], "col2": [None, "b", None, "d", None]})
    result = expectation.validate(data_frame=data_frame)
    assert str(result) == str(
        DataFrameExpectationSuccessMessage(expectation_name="ExpectationMinRows")
    ), f"Expected success message but got: {result}"


def test_expectation_pyspark_success_exact_count(spark):
    """Test PySpark success case with exact minimum row count."""
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationMinRows",
        min_rows=3,
    )
    data_frame = spark.createDataFrame([(1, "a"), (2, "b"), (3, "c")], ["col1", "col2"])
    result = expectation.validate(data_frame=data_frame)
    assert str(result) == str(
        DataFrameExpectationSuccessMessage(expectation_name="ExpectationMinRows")
    ), f"Expected success message but got: {result}"


def test_expectation_pyspark_success_above_min(spark):
    """Test PySpark success case with row count above minimum."""
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationMinRows",
        min_rows=3,
    )
    data_frame = spark.createDataFrame(
        [(1, "a"), (2, "b"), (3, "c"), (4, "d"), (5, "e")], ["col1", "col2"]
    )
    result = expectation.validate(data_frame=data_frame)
    assert str(result) == str(
        DataFrameExpectationSuccessMessage(expectation_name="ExpectationMinRows")
    ), f"Expected success message but got: {result}"


def test_expectation_pyspark_success_single_row(spark):
    """Test PySpark success case with single row."""
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationMinRows",
        min_rows=1,
    )
    data_frame = spark.createDataFrame([(42,)], ["col1"])
    result = expectation.validate(data_frame=data_frame)
    assert str(result) == str(
        DataFrameExpectationSuccessMessage(expectation_name="ExpectationMinRows")
    ), f"Expected success message but got: {result}"


def test_expectation_pyspark_success_zero_min_empty_df(spark):
    """Test PySpark success case with zero minimum and empty DataFrame."""
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationMinRows",
        min_rows=0,
    )
    # Create empty DataFrame with schema
    data_frame = spark.createDataFrame([], "col1: int")
    result = expectation.validate(data_frame=data_frame)
    assert str(result) == str(
        DataFrameExpectationSuccessMessage(expectation_name="ExpectationMinRows")
    ), f"Expected success message but got: {result}"


def test_expectation_pyspark_success_zero_min_with_data(spark):
    """Test PySpark success case with zero minimum and data present."""
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationMinRows",
        min_rows=0,
    )
    data_frame = spark.createDataFrame([(1,), (2,), (3,)], ["col1"])
    result = expectation.validate(data_frame=data_frame)
    assert str(result) == str(
        DataFrameExpectationSuccessMessage(expectation_name="ExpectationMinRows")
    ), f"Expected success message but got: {result}"


def test_expectation_pyspark_failure_below_min(spark):
    """Test PySpark failure case when row count is below minimum."""
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationMinRows",
        min_rows=5,
    )
    data_frame = spark.createDataFrame([(1, "a"), (2, "b"), (3, "c")], ["col1", "col2"])
    result = expectation.validate(data_frame=data_frame)

    expected_failure_message = DataFrameExpectationFailureMessage(
        expectation_str=str(expectation),
        data_frame_type=DataFrameType.PYSPARK,
        message="DataFrame has 3 rows, expected at least 5.",
    )
    assert str(result) == str(expected_failure_message), (
        f"Expected failure message but got: {result}"
    )


def test_expectation_pyspark_failure_empty_with_min(spark):
    """Test PySpark failure case with empty DataFrame but minimum required."""
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationMinRows",
        min_rows=2,
    )
    data_frame = spark.createDataFrame([], "col1: int")
    result = expectation.validate(data_frame=data_frame)

    expected_failure_message = DataFrameExpectationFailureMessage(
        expectation_str=str(expectation),
        data_frame_type=DataFrameType.PYSPARK,
        message="DataFrame has 0 rows, expected at least 2.",
    )
    assert str(result) == str(expected_failure_message), (
        f"Expected failure message but got: {result}"
    )


def test_expectation_pyspark_failure_single_row_needs_more(spark):
    """Test PySpark failure case with single row but higher minimum required."""
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationMinRows",
        min_rows=3,
    )
    data_frame = spark.createDataFrame([(1,)], ["col1"])
    result = expectation.validate(data_frame=data_frame)

    expected_failure_message = DataFrameExpectationFailureMessage(
        expectation_str=str(expectation),
        data_frame_type=DataFrameType.PYSPARK,
        message="DataFrame has 1 rows, expected at least 3.",
    )
    assert str(result) == str(expected_failure_message), (
        f"Expected failure message but got: {result}"
    )


def test_expectation_pyspark_large_dataset(spark):
    """Test PySpark with larger dataset meeting minimum."""
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationMinRows",
        min_rows=50,
    )
    # Create DataFrame with 75 rows
    data = [(i, f"value_{i}") for i in range(75)]
    data_frame = spark.createDataFrame(data, ["col1", "col2"])
    result = expectation.validate(data_frame=data_frame)
    assert str(result) == str(
        DataFrameExpectationSuccessMessage(expectation_name="ExpectationMinRows")
    ), f"Expected success message but got: {result}"


def test_expectation_pyspark_large_dataset_failure(spark):
    """Test PySpark with dataset not meeting large minimum."""
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationMinRows",
        min_rows=100,
    )
    # Create DataFrame with 75 rows
    data = [(i, f"value_{i}") for i in range(75)]
    data_frame = spark.createDataFrame(data, ["col1", "col2"])
    result = expectation.validate(data_frame=data_frame)

    expected_failure_message = DataFrameExpectationFailureMessage(
        expectation_str=str(expectation),
        data_frame_type=DataFrameType.PYSPARK,
        message="DataFrame has 75 rows, expected at least 100.",
    )
    assert str(result) == str(expected_failure_message), (
        f"Expected failure message but got: {result}"
    )


def test_expectation_pyspark_with_nulls(spark):
    """Test PySpark expectation with null values (should still count rows)."""
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationMinRows",
        min_rows=3,
    )
    data_frame = spark.createDataFrame(
        [(1, None), (None, "b"), (3, None), (None, "d"), (5, None)],
        ["col1", "col2"],
    )
    result = expectation.validate(data_frame=data_frame)
    assert str(result) == str(
        DataFrameExpectationSuccessMessage(expectation_name="ExpectationMinRows")
    ), f"Expected success message but got: {result}"


def test_suite_pandas_success():
    """Test integration with expectations suite for pandas success case."""
    expectations_suite = DataFrameExpectationsSuite().expect_min_rows(min_rows=2)
    data_frame = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
    result = expectations_suite.build().run(data_frame=data_frame)
    assert result is None, "Expected no exceptions to be raised"


def test_suite_pandas_violations():
    """Test integration with expectations suite for pandas failure case."""
    expectations_suite = DataFrameExpectationsSuite().expect_min_rows(min_rows=5)
    data_frame = pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
    with pytest.raises(DataFrameExpectationsSuiteFailure):
        expectations_suite.build().run(data_frame=data_frame)


def test_suite_pyspark_success(spark):
    """Test integration with expectations suite for PySpark success case."""
    expectations_suite = DataFrameExpectationsSuite().expect_min_rows(min_rows=2)
    data_frame = spark.createDataFrame([(1, "a"), (2, "b"), (3, "c")], ["col1", "col2"])
    result = expectations_suite.build().run(data_frame=data_frame)
    assert result is None, "Expected no exceptions to be raised"


def test_suite_pyspark_violations(spark):
    """Test integration with expectations suite for PySpark failure case."""
    expectations_suite = DataFrameExpectationsSuite().expect_min_rows(min_rows=5)
    data_frame = spark.createDataFrame([(1, "a"), (2, "b")], ["col1", "col2"])
    with pytest.raises(DataFrameExpectationsSuiteFailure):
        expectations_suite.build().run(data_frame=data_frame)


def test_expectation_parameter_validation():
    """Test parameter validation for min_rows."""
    # Test with valid parameters
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationMinRows",
        min_rows=10,
    )
    assert expectation is not None, "Expected expectation to be created successfully"

    # Test string representation
    expectation_str = str(expectation)
    assert "10" in expectation_str, f"Expected '10' in expectation string: {expectation_str}"
    assert "ExpectationMinRows" in expectation_str, (
        f"Expected 'ExpectationMinRows' in expectation string: {expectation_str}"
    )


def test_expectation_boundary_conditions():
    """Test various boundary conditions for min_rows."""
    # Test with min_rows = 1
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationMinRows",
        min_rows=1,
    )

    # Single row - should pass
    data_frame = pd.DataFrame({"col1": [1]})
    result = expectation.validate(data_frame=data_frame)
    assert str(result) == str(
        DataFrameExpectationSuccessMessage(expectation_name="ExpectationMinRows")
    ), f"Expected success message but got: {result}"

    # Empty DataFrame - should fail
    data_frame = pd.DataFrame({"col1": []})
    result = expectation.validate(data_frame=data_frame)
    assert isinstance(result, DataFrameExpectationFailureMessage), (
        f"Expected DataFrameExpectationFailureMessage but got: {type(result)}"
    )


def test_expectation_multiple_columns():
    """Test expectation with multiple columns (should still count total rows)."""
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationMinRows",
        min_rows=3,
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
    assert str(result) == str(
        DataFrameExpectationSuccessMessage(expectation_name="ExpectationMinRows")
    ), f"Expected success message but got: {result}"


def test_expectation_mixed_data_types():
    """Test expectation with mixed data types in columns."""
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationMinRows",
        min_rows=3,
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
        DataFrameExpectationSuccessMessage(expectation_name="ExpectationMinRows")
    ), f"Expected success message but got: {result}"


def test_expectation_low_min_count():
    """Test expectation with very low min_rows value."""
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationMinRows",
        min_rows=1,
    )
    data_frame = pd.DataFrame({"col1": [1, 2, 3]})
    result = expectation.validate(data_frame=data_frame)
    assert str(result) == str(
        DataFrameExpectationSuccessMessage(expectation_name="ExpectationMinRows")
    ), f"Expected success message but got: {result}"


def test_expectation_high_min_count():
    """Test expectation with very high min_rows value."""
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationMinRows",
        min_rows=1000000,  # 1 million
    )
    data_frame = pd.DataFrame({"col1": [1, 2, 3]})
    result = expectation.validate(data_frame=data_frame)

    expected_failure_message = DataFrameExpectationFailureMessage(
        expectation_str=str(expectation),
        data_frame_type=DataFrameType.PANDAS,
        message="DataFrame has 3 rows, expected at least 1000000.",
    )
    assert str(result) == str(expected_failure_message), (
        f"Expected failure message but got: {result}"
    )


def test_expectation_identical_values():
    """Test expectation with DataFrame containing identical values."""
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationMinRows",
        min_rows=3,
    )
    data_frame = pd.DataFrame(
        {
            "col1": [42, 42, 42, 42],  # All same values
            "col2": ["same", "same", "same", "same"],
        }
    )
    result = expectation.validate(data_frame=data_frame)
    assert str(result) == str(
        DataFrameExpectationSuccessMessage(expectation_name="ExpectationMinRows")
    ), f"Expected success message but got: {result}"


def test_expectation_edge_case_min_count_equals_actual():
    """Test edge case where min_rows exactly equals actual row count."""
    for count in [1, 5, 10, 100]:
        expectation = DataFrameExpectationRegistry.get_expectation(
            expectation_name="ExpectationMinRows",
            min_rows=count,
        )
        # Create DataFrame with exactly 'count' rows
        data_frame = pd.DataFrame({"col1": list(range(count))})
        result = expectation.validate(data_frame=data_frame)
        assert str(result) == str(
            DataFrameExpectationSuccessMessage(expectation_name="ExpectationMinRows")
        ), f"Expected success message for count {count} but got: {result}"


def test_expectation_zero_min_count_edge_cases():
    """Test edge cases with zero minimum count."""
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationMinRows",
        min_rows=0,
    )

    # Empty DataFrame should pass
    data_frame = pd.DataFrame({"col1": []})
    result = expectation.validate(data_frame=data_frame)
    assert str(result) == str(
        DataFrameExpectationSuccessMessage(expectation_name="ExpectationMinRows")
    ), f"Expected success message but got: {result}"

    # DataFrame with data should also pass
    data_frame = pd.DataFrame({"col1": [1, 2, 3]})
    result = expectation.validate(data_frame=data_frame)
    assert str(result) == str(
        DataFrameExpectationSuccessMessage(expectation_name="ExpectationMinRows")
    ), f"Expected success message but got: {result}"


def test_expectation_progressive_min_counts():
    """Test expectation with progressively increasing minimum counts."""
    data_frame = pd.DataFrame({"col1": [1, 2, 3, 4, 5]})  # 5 rows

    # Should pass for min_rows <= 5
    for min_rows in [0, 1, 2, 3, 4, 5]:
        expectation = DataFrameExpectationRegistry.get_expectation(
            expectation_name="ExpectationMinRows",
            min_rows=min_rows,
        )
        result = expectation.validate(data_frame=data_frame)
        assert str(result) == str(
            DataFrameExpectationSuccessMessage(expectation_name="ExpectationMinRows")
        ), f"Expected success message for min_rows {min_rows} but got: {result}"

    # Should fail for min_rows > 5
    for min_rows in [6, 7, 10, 100]:
        expectation = DataFrameExpectationRegistry.get_expectation(
            expectation_name="ExpectationMinRows",
            min_rows=min_rows,
        )
        result = expectation.validate(data_frame=data_frame)

        expected_failure_message = DataFrameExpectationFailureMessage(
            expectation_str=str(expectation),
            data_frame_type=DataFrameType.PANDAS,
            message=f"DataFrame has 5 rows, expected at least {min_rows}.",
        )
        assert str(result) == str(expected_failure_message), (
            f"Expected failure message for min_rows {min_rows} but got: {result}"
        )


def test_expectation_dataframe_structure_irrelevant():
    """Test that DataFrame structure doesn't affect row counting."""
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationMinRows",
        min_rows=3,
    )

    # Single column DataFrame
    df1 = pd.DataFrame({"col1": [1, 2, 3]})
    result1 = expectation.validate(data_frame=df1)

    # Multi-column DataFrame
    df2 = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"], "col3": [1.1, 2.2, 3.3]})
    result2 = expectation.validate(data_frame=df2)

    # Both should have same result (success)
    assert str(result1) == str(result2), f"Expected same results but got: {result1} vs {result2}"
    assert str(result1) == str(
        DataFrameExpectationSuccessMessage(expectation_name="ExpectationMinRows")
    ), f"Expected success message but got: {result1}"
