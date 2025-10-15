import pytest
import pandas as pd
from pyspark.sql.types import IntegerType, StructField, StructType

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


def test_expectation_name():
    """
    Test that the expectation name is correctly returned.
    """
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationUniqueRows",
        column_names=["col1"],
    )
    assert (
        expectation.get_expectation_name() == "ExpectationUniqueRows"
    ), f"Expected 'ExpectationUniqueRows' but got: {expectation.get_expectation_name()}"


# Tests for specific columns - Pandas
def test_expectation_pandas_success_specific_columns():
    """
    Test the expectation for pandas DataFrame with no violations on specific columns.
    """
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationUniqueRows",
        column_names=["col1", "col2"],
    )
    data_frame = pd.DataFrame(
        {
            "col1": [1, 2, 3, 1],
            "col2": [10, 20, 30, 20],  # Different combination
            "col3": [100, 100, 100, 100],  # Same values but not checked
        }
    )
    result = expectation.validate(data_frame=data_frame)
    assert str(result) == str(
        DataframeExpectationSuccessMessage(expectation_name="ExpectationUniqueRows")
    ), f"Expected success message but got: {result}"


def test_expectation_pandas_violations_specific_columns():
    """
    Test the expectation for pandas DataFrame with violations on specific columns.
    """
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationUniqueRows",
        column_names=["col1", "col2"],
    )
    data_frame = pd.DataFrame(
        {
            "col1": [1, 2, 1, 3],
            "col2": [10, 20, 10, 30],  # Duplicate combination (1, 10)
            "col3": [100, 200, 300, 400],
        }
    )
    result = expectation.validate(data_frame=data_frame)

    # Expected violations shows only one row per duplicate group with count
    expected_violations = pd.DataFrame({"col1": [1], "col2": [10], "#duplicates": [2]})
    assert str(result) == str(
        DataframeExpectationFailureMessage(
            expectation_str=str(expectation),
            data_frame_type=DataFrameType.PANDAS,
            violations_data_frame=expected_violations,
            message="Found 2 duplicate row(s). duplicate rows found for columns ['col1', 'col2']",
            limit_violations=5,
        )
    ), f"Expected failure message but got: {result}"


# Tests for all columns (empty list) - Pandas
def test_expectation_pandas_success_all_columns():
    """
    Test the expectation for pandas DataFrame with no violations on all columns.
    """
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationUniqueRows",
        column_names=[],
    )
    data_frame = pd.DataFrame({"col1": [1, 2, 3], "col2": [10, 20, 30], "col3": [100, 200, 300]})
    result = expectation.validate(data_frame=data_frame)
    assert str(result) == str(
        DataframeExpectationSuccessMessage(expectation_name="ExpectationUniqueRows")
    ), f"Expected success message but got: {result}"


def test_expectation_pandas_violations_all_columns():
    """
    Test the expectation for pandas DataFrame with violations on all columns.
    """
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationUniqueRows",
        column_names=[],
    )
    data_frame = pd.DataFrame(
        {
            "col1": [1, 2, 1],
            "col2": [10, 20, 10],  # Duplicate combination (1, 10)
            "col3": [100, 200, 100],
        }
    )
    result = expectation.validate(data_frame=data_frame)

    # Expected violations shows only one row per duplicate group with count
    expected_violations = pd.DataFrame(
        {"col1": [1], "col2": [10], "col3": [100], "#duplicates": [2]}
    )
    assert str(result) == str(
        DataframeExpectationFailureMessage(
            expectation_str=str(expectation),
            data_frame_type=DataFrameType.PANDAS,
            violations_data_frame=expected_violations,
            message="Found 2 duplicate row(s). duplicate rows found",
            limit_violations=5,
        )
    ), f"Expected failure message but got: {result}"


# Tests for specific columns - PySpark
def test_expectation_pyspark_success_specific_columns(spark):
    """
    Test the expectation for PySpark DataFrame with no violations on specific columns.
    """
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationUniqueRows",
        column_names=["col1", "col2"],
    )
    data_frame = spark.createDataFrame(
        [
            (1, 10, 100),
            (2, 20, 100),
            (3, 30, 100),
            (1, 20, 100),  # Different combination
        ],
        ["col1", "col2", "col3"],
    )
    result = expectation.validate(data_frame=data_frame)
    assert str(result) == str(
        DataframeExpectationSuccessMessage(expectation_name="ExpectationUniqueRows")
    ), f"Expected success message but got: {result}"


def test_expectation_pyspark_violations_specific_columns(spark):
    """
    Test the expectation for PySpark DataFrame with violations on specific columns.
    """
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationUniqueRows",
        column_names=["col1", "col2"],
    )
    data_frame = spark.createDataFrame(
        [
            (1, 10, 100),
            (2, 20, 200),
            (1, 10, 300),  # Duplicate combination (1, 10)
            (3, 30, 400),
        ],
        ["col1", "col2", "col3"],
    )
    result = expectation.validate(data_frame=data_frame)

    # Expected violations shows only one row per duplicate group with count
    expected_violations = spark.createDataFrame([(1, 10, 2)], ["col1", "col2", "#duplicates"])
    assert str(result) == str(
        DataframeExpectationFailureMessage(
            expectation_str=str(expectation),
            data_frame_type=DataFrameType.PYSPARK,
            violations_data_frame=expected_violations,
            message="Found 2 duplicate row(s). duplicate rows found for columns ['col1', 'col2']",
            limit_violations=5,
        )
    ), f"Expected failure message but got: {result}"


# Tests for all columns (empty list) - PySpark
def test_expectation_pyspark_success_all_columns(spark):
    """
    Test the expectation for PySpark DataFrame with no violations on all columns.
    """
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationUniqueRows",
        column_names=[],
    )
    data_frame = spark.createDataFrame(
        [(1, 10, 100), (2, 20, 200), (3, 30, 300)], ["col1", "col2", "col3"]
    )
    result = expectation.validate(data_frame=data_frame)
    assert str(result) == str(
        DataframeExpectationSuccessMessage(expectation_name="ExpectationUniqueRows")
    ), f"Expected success message but got: {result}"


def test_expectation_pyspark_violations_all_columns(spark):
    """
    Test the expectation for PySpark DataFrame with violations on all columns.
    """
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationUniqueRows",
        column_names=[],
    )
    data_frame = spark.createDataFrame(
        [
            (1, 10, 100),
            (2, 20, 200),
            (1, 10, 100),
        ],  # Duplicate combination (1, 10, 100)
        ["col1", "col2", "col3"],
    )
    result = expectation.validate(data_frame=data_frame)

    # Expected violations shows only one row per duplicate group with count
    expected_violations = spark.createDataFrame(
        [(1, 10, 100, 2)], ["col1", "col2", "col3", "#duplicates"]
    )
    assert str(result) == str(
        DataframeExpectationFailureMessage(
            expectation_str=str(expectation),
            data_frame_type=DataFrameType.PYSPARK,
            violations_data_frame=expected_violations,
            message="Found 2 duplicate row(s). duplicate rows found",
            limit_violations=5,
        )
    ), f"Expected failure message but got: {result}"


# Edge case tests
def test_column_missing_error_pandas():
    """
    Test that an error is raised when specified columns are missing in pandas DataFrame.
    """
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationUniqueRows",
        column_names=["nonexistent_col"],
    )
    data_frame = pd.DataFrame({"col1": [1, 2, 3]})
    result = expectation.validate(data_frame=data_frame)
    expected_failure_message = DataframeExpectationFailureMessage(
        expectation_str=str(expectation),
        data_frame_type=DataFrameType.PANDAS,
        message="Column 'nonexistent_col' does not exist in the DataFrame.",
    )
    assert str(result) == str(
        expected_failure_message
    ), f"Expected failure message but got: {result}"


def test_column_missing_error_pyspark(spark):
    """
    Test that an error is raised when specified columns are missing in PySpark DataFrame.
    """
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationUniqueRows",
        column_names=["nonexistent_col"],
    )
    data_frame = spark.createDataFrame([(1,), (2,), (3,)], ["col1"])
    result = expectation.validate(data_frame=data_frame)
    expected_failure_message = DataframeExpectationFailureMessage(
        expectation_str=str(expectation),
        data_frame_type=DataFrameType.PYSPARK,
        message="Column 'nonexistent_col' does not exist in the DataFrame.",
    )
    assert str(result) == str(
        expected_failure_message
    ), f"Expected failure message but got: {result}"


def test_empty_dataframe_pandas():
    """
    Test the expectation on an empty pandas DataFrame.
    """
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationUniqueRows",
        column_names=["col1"],
    )
    data_frame = pd.DataFrame({"col1": []})
    result = expectation.validate(data_frame=data_frame)
    assert str(result) == str(
        DataframeExpectationSuccessMessage(expectation_name="ExpectationUniqueRows")
    ), f"Expected success message but got: {result}"


def test_empty_dataframe_pyspark(spark):
    """
    Test the expectation on an empty PySpark DataFrame.
    """
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationUniqueRows",
        column_names=["col1"],
    )

    schema = StructType([StructField("col1", IntegerType(), True)])
    data_frame = spark.createDataFrame([], schema)
    result = expectation.validate(data_frame=data_frame)
    assert str(result) == str(
        DataframeExpectationSuccessMessage(expectation_name="ExpectationUniqueRows")
    ), f"Expected success message but got: {result}"


def test_single_row_dataframe_pandas():
    """
    Test the expectation on a single-row pandas DataFrame.
    """
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationUniqueRows",
        column_names=["col1"],
    )
    data_frame = pd.DataFrame({"col1": [1]})
    result = expectation.validate(data_frame=data_frame)
    assert str(result) == str(
        DataframeExpectationSuccessMessage(expectation_name="ExpectationUniqueRows")
    ), f"Expected success message but got: {result}"


def test_single_row_dataframe_pyspark(spark):
    """
    Test the expectation on a single-row PySpark DataFrame.
    """
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationUniqueRows",
        column_names=["col1"],
    )
    data_frame = spark.createDataFrame([(1,)], ["col1"])
    result = expectation.validate(data_frame=data_frame)
    assert str(result) == str(
        DataframeExpectationSuccessMessage(expectation_name="ExpectationUniqueRows")
    ), f"Expected success message but got: {result}"


def test_with_nulls_pandas():
    """
    Test the expectation with null values in pandas DataFrame.
    """
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationUniqueRows",
        column_names=["col1", "col2"],
    )
    data_frame = pd.DataFrame(
        {
            "col1": [1, None, 1, None],
            "col2": [10, None, 20, None],  # (None, None) appears twice
        }
    )
    result = expectation.validate(data_frame=data_frame)

    # Expected violations shows only one row per duplicate group with count
    expected_violations = pd.DataFrame({"col1": [None], "col2": [None], "#duplicates": [2]})

    assert str(result) == str(
        DataframeExpectationFailureMessage(
            expectation_str=str(expectation),
            data_frame_type=DataFrameType.PANDAS,
            violations_data_frame=expected_violations,
            message="Found 2 duplicate row(s). duplicate rows found for columns ['col1', 'col2']",
            limit_violations=5,
        )
    ), f"Expected failure message but got: {result}"


def test_with_nulls_pyspark(spark):
    """
    Test the expectation with null values in PySpark DataFrame.
    """
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationUniqueRows",
        column_names=["col1", "col2"],
    )
    data_frame = spark.createDataFrame(
        [
            (1, 10),
            (None, None),
            (1, 20),
            (None, None),  # (None, None) appears twice
        ],
        ["col1", "col2"],
    )
    result = expectation.validate(data_frame=data_frame)

    schema = StructType(
        [
            StructField("col1", IntegerType(), True),
            StructField("col2", IntegerType(), True),
            StructField("#duplicates", IntegerType(), True),
        ]
    )
    # Expected violations shows only one row per duplicate group with count
    expected_violations = spark.createDataFrame([(None, None, 2)], schema)
    assert str(result) == str(
        DataframeExpectationFailureMessage(
            expectation_str=str(expectation),
            data_frame_type=DataFrameType.PYSPARK,
            violations_data_frame=expected_violations,
            message="Found 2 duplicate row(s). duplicate rows found for columns ['col1', 'col2']",
            limit_violations=5,
        )
    ), f"Expected failure message but got: {result}"


# Test with multiple duplicate groups
def test_expectation_pandas_multiple_duplicate_groups():
    """
    Test the expectation with multiple groups of duplicates in pandas DataFrame.
    """
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationUniqueRows",
        column_names=["col1"],
    )
    data_frame = pd.DataFrame(
        {
            "col1": [1, 2, 1, 3, 2, 3],  # Three groups: (1,1), (2,2), (3,3)
            "col2": [10, 20, 30, 40, 50, 60],
        }
    )
    result = expectation.validate(data_frame=data_frame)

    # Expected violations shows one row per duplicate group with count, ordered by count then by values
    expected_violations = pd.DataFrame({"col1": [1, 2, 3], "#duplicates": [2, 2, 2]})
    assert str(result) == str(
        DataframeExpectationFailureMessage(
            expectation_str=str(expectation),
            data_frame_type=DataFrameType.PANDAS,
            violations_data_frame=expected_violations,
            message="Found 6 duplicate row(s). duplicate rows found for columns ['col1']",
            limit_violations=5,
        )
    ), f"Expected failure message but got: {result}"


def test_expectation_pyspark_multiple_duplicate_groups(spark):
    """
    Test the expectation with multiple groups of duplicates in PySpark DataFrame.
    """
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationUniqueRows",
        column_names=["col1"],
    )
    data_frame = spark.createDataFrame(
        [
            (1, 10),
            (2, 20),
            (1, 30),  # Duplicate group 1
            (3, 40),
            (2, 50),  # Duplicate group 2
            (3, 60),  # Duplicate group 3
        ],
        ["col1", "col2"],
    )
    result = expectation.validate(data_frame=data_frame)

    # Expected violations shows one row per duplicate group with count, ordered by count then by values
    expected_violations = spark.createDataFrame([(1, 2), (2, 2), (3, 2)], ["col1", "#duplicates"])
    assert str(result) == str(
        DataframeExpectationFailureMessage(
            expectation_str=str(expectation),
            data_frame_type=DataFrameType.PYSPARK,
            violations_data_frame=expected_violations,
            message="Found 6 duplicate row(s). duplicate rows found for columns ['col1']",
            limit_violations=5,
        )
    ), f"Expected failure message but got: {result}"


# Suite-level tests
def test_suite_pandas_success_specific_columns():
    """
    Test the expectation suite for pandas DataFrame with no violations on specific columns.
    """
    expectations_suite = DataframeExpectationsSuite().expect_unique_rows(column_names=["col1"])
    data_frame = pd.DataFrame({"col1": [1, 2, 3], "col2": [10, 10, 10]})
    result = expectations_suite.run(data_frame=data_frame)
    assert result is None, "Expected no exceptions to be raised"


def test_suite_pandas_violations_specific_columns():
    """
    Test the expectation suite for pandas DataFrame with violations on specific columns.
    """
    expectations_suite = DataframeExpectationsSuite().expect_unique_rows(column_names=["col1"])
    data_frame = pd.DataFrame({"col1": [1, 1, 3], "col2": [10, 20, 30]})
    with pytest.raises(DataframeExpectationsSuiteFailure):
        expectations_suite.run(data_frame=data_frame)


def test_suite_pandas_success_all_columns():
    """
    Test the expectation suite for pandas DataFrame with no violations on all columns.
    """
    expectations_suite = DataframeExpectationsSuite().expect_unique_rows(column_names=[])
    data_frame = pd.DataFrame({"col1": [1, 2, 3], "col2": [10, 20, 30]})
    result = expectations_suite.run(data_frame=data_frame)
    assert result is None, "Expected no exceptions to be raised"


def test_suite_pandas_violations_all_columns():
    """
    Test the expectation suite for pandas DataFrame with violations on all columns.
    """
    expectations_suite = DataframeExpectationsSuite().expect_unique_rows(column_names=[])
    data_frame = pd.DataFrame({"col1": [1, 1, 3], "col2": [10, 10, 30]})
    with pytest.raises(DataframeExpectationsSuiteFailure):
        expectations_suite.run(data_frame=data_frame)


def test_suite_pyspark_success_specific_columns(spark):
    """
    Test the expectation suite for PySpark DataFrame with no violations on specific columns.
    """
    expectations_suite = DataframeExpectationsSuite().expect_unique_rows(column_names=["col1"])
    data_frame = spark.createDataFrame([(1, 10), (2, 10), (3, 10)], ["col1", "col2"])
    result = expectations_suite.run(data_frame=data_frame)
    assert result is None, "Expected no exceptions to be raised"


def test_suite_pyspark_violations_specific_columns(spark):
    """
    Test the expectation suite for PySpark DataFrame with violations on specific columns.
    """
    expectations_suite = DataframeExpectationsSuite().expect_unique_rows(column_names=["col1"])
    data_frame = spark.createDataFrame([(1, 10), (1, 20), (3, 30)], ["col1", "col2"])
    with pytest.raises(DataframeExpectationsSuiteFailure):
        expectations_suite.run(data_frame=data_frame)


def test_suite_pyspark_success_all_columns(spark):
    """
    Test the expectation suite for PySpark DataFrame with no violations on all columns.
    """
    expectations_suite = DataframeExpectationsSuite().expect_unique_rows(column_names=[])
    data_frame = spark.createDataFrame([(1, 10), (2, 20), (3, 30)], ["col1", "col2"])
    result = expectations_suite.run(data_frame=data_frame)
    assert result is None, "Expected no exceptions to be raised"


def test_suite_pyspark_violations_all_columns(spark):
    """
    Test the expectation suite for PySpark DataFrame with violations on all columns.
    """
    expectations_suite = DataframeExpectationsSuite().expect_unique_rows(column_names=[])
    data_frame = spark.createDataFrame([(1, 10), (1, 10), (3, 30)], ["col1", "col2"])
    with pytest.raises(DataframeExpectationsSuiteFailure):
        expectations_suite.run(data_frame=data_frame)


def test_suite_pandas_column_missing_error():
    """
    Test that an error is raised when specified columns are missing in pandas DataFrame suite.
    """
    expectations_suite = DataframeExpectationsSuite().expect_unique_rows(
        column_names=["nonexistent_col"]
    )
    data_frame = pd.DataFrame({"col1": [1, 2, 3]})
    with pytest.raises(DataframeExpectationsSuiteFailure):
        expectations_suite.run(data_frame=data_frame)


def test_suite_pyspark_column_missing_error(spark):
    """
    Test that an error is raised when specified columns are missing in PySpark DataFrame suite.
    """
    expectations_suite = DataframeExpectationsSuite().expect_unique_rows(
        column_names=["nonexistent_col"]
    )
    data_frame = spark.createDataFrame([(1,), (2,), (3,)], ["col1"])
    with pytest.raises(DataframeExpectationsSuiteFailure):
        expectations_suite.run(data_frame=data_frame)
