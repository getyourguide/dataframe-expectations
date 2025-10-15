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


def test_expectation_name():
    """
    Test that the expectation name is correctly returned.
    """
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationDistinctColumnValuesEquals",
        column_name="col1",
        expected_value=3,
    )
    assert (
        expectation.get_expectation_name() == "ExpectationDistinctColumnValuesEquals"
    ), f"Expected 'ExpectationDistinctColumnValuesEquals' but got: {expectation.get_expectation_name()}"


def test_expectation_pandas_success():
    """
    Test the expectation for pandas DataFrame with no violations.
    """
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationDistinctColumnValuesEquals",
        column_name="col1",
        expected_value=3,
    )
    # DataFrame with exactly 3 distinct values [1, 2, 3]
    data_frame = pd.DataFrame({"col1": [1, 2, 3, 2, 1]})
    result = expectation.validate(data_frame=data_frame)
    assert str(result) == str(
        DataframeExpectationSuccessMessage(expectation_name="ExpectationDistinctColumnValuesEquals")
    ), f"Expected success message but got: {result}"


def test_expectation_pandas_success_with_nulls():
    """
    Test the expectation for pandas DataFrame with NaN values included in distinct count.
    """
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationDistinctColumnValuesEquals",
        column_name="col1",
        expected_value=3,
    )
    # DataFrame with exactly 3 distinct values [1, 2, NaN]
    data_frame = pd.DataFrame({"col1": [1, 2, None, 2, 1]})
    result = expectation.validate(data_frame=data_frame)
    assert str(result) == str(
        DataframeExpectationSuccessMessage(expectation_name="ExpectationDistinctColumnValuesEquals")
    ), f"Expected success message but got: {result}"


def test_expectation_pandas_violations_too_few():
    """
    Test the expectation for pandas DataFrame with too few distinct values.
    """
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationDistinctColumnValuesEquals",
        column_name="col1",
        expected_value=5,
    )
    # DataFrame with 2 distinct values [1, 2] when expecting 5
    data_frame = pd.DataFrame({"col1": [1, 2, 1, 2, 1]})
    result = expectation.validate(data_frame=data_frame)

    expected_failure_message = DataframeExpectationFailureMessage(
        expectation_str=str(expectation),
        data_frame_type=DataFrameType.PANDAS,
        message="Column 'col1' has 2 distinct values, expected exactly 5.",
    )
    assert str(result) == str(
        expected_failure_message
    ), f"Expected failure message but got: {result}"


def test_expectation_pandas_violations_too_many():
    """
    Test the expectation for pandas DataFrame with too many distinct values.
    """
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationDistinctColumnValuesEquals",
        column_name="col1",
        expected_value=2,
    )
    # DataFrame with 5 distinct values [1, 2, 3, 4, 5] when expecting 2
    data_frame = pd.DataFrame({"col1": [1, 2, 3, 4, 5]})
    result = expectation.validate(data_frame=data_frame)

    expected_failure_message = DataframeExpectationFailureMessage(
        expectation_str=str(expectation),
        data_frame_type=DataFrameType.PANDAS,
        message="Column 'col1' has 5 distinct values, expected exactly 2.",
    )
    assert str(result) == str(
        expected_failure_message
    ), f"Expected failure message but got: {result}"


def test_expectation_pandas_zero_expected():
    """
    Test the expectation for pandas DataFrame expecting zero distinct values.
    """
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationDistinctColumnValuesEquals",
        column_name="col1",
        expected_value=0,
    )
    # Empty DataFrame should have 0 distinct values
    data_frame = pd.DataFrame({"col1": []})
    result = expectation.validate(data_frame=data_frame)
    assert isinstance(
        result, DataframeExpectationSuccessMessage
    ), f"Expected DataframeExpectationSuccessMessage but got: {type(result)}"


def test_expectation_pandas_one_expected():
    """
    Test the expectation for pandas DataFrame expecting exactly one distinct value.
    """
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationDistinctColumnValuesEquals",
        column_name="col1",
        expected_value=1,
    )
    # DataFrame with exactly 1 distinct value
    data_frame = pd.DataFrame({"col1": [5, 5, 5, 5, 5]})
    result = expectation.validate(data_frame=data_frame)
    assert isinstance(
        result, DataframeExpectationSuccessMessage
    ), f"Expected DataframeExpectationSuccessMessage but got: {type(result)}"


def test_expectation_pyspark_success(spark):
    """
    Test the expectation for PySpark DataFrame with no violations.
    """
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationDistinctColumnValuesEquals",
        column_name="col1",
        expected_value=3,
    )
    # DataFrame with exactly 3 distinct values [1, 2, 3]
    data_frame = spark.createDataFrame([(1,), (2,), (3,), (2,), (1,)], ["col1"])
    result = expectation.validate(data_frame=data_frame)
    assert str(result) == str(
        DataframeExpectationSuccessMessage(expectation_name="ExpectationDistinctColumnValuesEquals")
    ), f"Expected success message but got: {result}"


def test_expectation_pyspark_success_with_nulls(spark):
    """
    Test the expectation for PySpark DataFrame with null values included in distinct count.
    """
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationDistinctColumnValuesEquals",
        column_name="col1",
        expected_value=3,
    )
    # DataFrame with exactly 3 distinct values [1, 2, null]
    data_frame = spark.createDataFrame([(1,), (2,), (None,), (2,), (1,)], ["col1"])
    result = expectation.validate(data_frame=data_frame)
    assert str(result) == str(
        DataframeExpectationSuccessMessage(expectation_name="ExpectationDistinctColumnValuesEquals")
    ), f"Expected success message but got: {result}"


def test_expectation_pyspark_violations_too_few(spark):
    """
    Test the expectation for PySpark DataFrame with too few distinct values.
    """
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationDistinctColumnValuesEquals",
        column_name="col1",
        expected_value=5,
    )
    # DataFrame with 2 distinct values [1, 2] when expecting 5
    data_frame = spark.createDataFrame([(1,), (2,), (1,), (2,), (1,)], ["col1"])
    result = expectation.validate(data_frame=data_frame)

    expected_failure_message = DataframeExpectationFailureMessage(
        expectation_str=str(expectation),
        data_frame_type=DataFrameType.PYSPARK,
        message="Column 'col1' has 2 distinct values, expected exactly 5.",
    )
    assert str(result) == str(
        expected_failure_message
    ), f"Expected failure message but got: {result}"


def test_expectation_pyspark_violations_too_many(spark):
    """
    Test the expectation for PySpark DataFrame with too many distinct values.
    """
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationDistinctColumnValuesEquals",
        column_name="col1",
        expected_value=2,
    )
    # DataFrame with 5 distinct values [1, 2, 3, 4, 5] when expecting 2
    data_frame = spark.createDataFrame([(1,), (2,), (3,), (4,), (5,)], ["col1"])
    result = expectation.validate(data_frame=data_frame)

    expected_failure_message = DataframeExpectationFailureMessage(
        expectation_str=str(expectation),
        data_frame_type=DataFrameType.PYSPARK,
        message="Column 'col1' has 5 distinct values, expected exactly 2.",
    )
    assert str(result) == str(
        expected_failure_message
    ), f"Expected failure message but got: {result}"


def test_expectation_pyspark_zero_expected(spark):
    """
    Test the expectation for PySpark DataFrame expecting zero distinct values.
    """
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationDistinctColumnValuesEquals",
        column_name="col1",
        expected_value=0,
    )
    # Empty DataFrame should have 0 distinct values
    data_frame = spark.createDataFrame([], "col1 INT")
    result = expectation.validate(data_frame=data_frame)
    assert isinstance(
        result, DataframeExpectationSuccessMessage
    ), f"Expected DataframeExpectationSuccessMessage but got: {type(result)}"


def test_expectation_pyspark_one_expected(spark):
    """
    Test the expectation for PySpark DataFrame expecting exactly one distinct value.
    """
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationDistinctColumnValuesEquals",
        column_name="col1",
        expected_value=1,
    )
    # DataFrame with exactly 1 distinct value
    data_frame = spark.createDataFrame([(5,), (5,), (5,), (5,), (5,)], ["col1"])
    result = expectation.validate(data_frame=data_frame)
    assert isinstance(
        result, DataframeExpectationSuccessMessage
    ), f"Expected DataframeExpectationSuccessMessage but got: {type(result)}"


def test_column_missing_error():
    """
    Test that an error is raised when the specified column is missing.
    """
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationDistinctColumnValuesEquals",
        column_name="col1",
        expected_value=3,
    )
    data_frame = pd.DataFrame({"col2": [1, 2, 3, 4, 5]})
    result = expectation.validate(data_frame=data_frame)
    expected_failure_message = DataframeExpectationFailureMessage(
        expectation_str=str(expectation),
        data_frame_type=DataFrameType.PANDAS,
        message="Column 'col1' does not exist in the DataFrame.",
    )
    assert str(result) == str(
        expected_failure_message
    ), f"Expected failure message but got: {result}"


def test_invalid_parameters():
    """
    Test that appropriate errors are raised for invalid parameters.
    """
    # Test negative expected_value
    with pytest.raises(ValueError) as context:
        DataframeExpectationRegistry.get_expectation(
            expectation_name="ExpectationDistinctColumnValuesEquals",
            column_name="col1",
            expected_value=-1,
        )
    assert "expected_value must be non-negative" in str(
        context.value
    ), f"Expected 'expected_value must be non-negative' in error message: {str(context.value)}"


def test_string_column_with_mixed_values():
    """
    Test the expectation with a string column containing mixed values.
    """
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationDistinctColumnValuesEquals",
        column_name="col1",
        expected_value=4,
    )
    # String column with exactly 4 distinct values ["A", "B", "C", None]
    data_frame = pd.DataFrame({"col1": ["A", "B", "C", "B", "A", None]})
    result = expectation.validate(data_frame=data_frame)
    assert isinstance(
        result, DataframeExpectationSuccessMessage
    ), f"Expected DataframeExpectationSuccessMessage but got: {type(result)}"


def test_string_column_case_sensitive():
    """
    Test that string comparisons are case-sensitive for distinct counting.
    """
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationDistinctColumnValuesEquals",
        column_name="col1",
        expected_value=4,
    )
    # String column with 4 distinct values ["a", "A", "b", "B"] (case-sensitive)
    data_frame = pd.DataFrame({"col1": ["a", "A", "b", "B", "a", "A"]})
    result = expectation.validate(data_frame=data_frame)
    assert isinstance(
        result, DataframeExpectationSuccessMessage
    ), f"Expected DataframeExpectationSuccessMessage but got: {type(result)}"


def test_numeric_column_with_floats():
    """
    Test the expectation with a numeric column containing floats.
    """
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationDistinctColumnValuesEquals",
        column_name="col1",
        expected_value=3,
    )
    # Float column with exactly 3 distinct values [1.1, 2.2, 3.3]
    data_frame = pd.DataFrame({"col1": [1.1, 2.2, 3.3, 2.2, 1.1]})
    result = expectation.validate(data_frame=data_frame)
    assert isinstance(
        result, DataframeExpectationSuccessMessage
    ), f"Expected DataframeExpectationSuccessMessage but got: {type(result)}"


def test_numeric_precision_handling():
    """
    Test that numeric precision is handled correctly for distinct counting.
    """
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationDistinctColumnValuesEquals",
        column_name="col1",
        expected_value=3,
    )
    # Values that might have precision issues but should be treated as distinct
    data_frame = pd.DataFrame({"col1": [1.0, 1.1, 1.2, 1.0, 1.1]})
    result = expectation.validate(data_frame=data_frame)
    assert isinstance(
        result, DataframeExpectationSuccessMessage
    ), f"Expected DataframeExpectationSuccessMessage but got: {type(result)}"


def test_boolean_column():
    """
    Test the expectation with a boolean column.
    """
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationDistinctColumnValuesEquals",
        column_name="col1",
        expected_value=2,
    )
    # Boolean column with exactly 2 distinct values [True, False]
    data_frame = pd.DataFrame({"col1": [True, False, True, False, True]})
    result = expectation.validate(data_frame=data_frame)
    assert isinstance(
        result, DataframeExpectationSuccessMessage
    ), f"Expected DataframeExpectationSuccessMessage but got: {type(result)}"


def test_boolean_column_with_none():
    """
    Test the expectation with a boolean column that includes None values.
    """
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationDistinctColumnValuesEquals",
        column_name="col1",
        expected_value=3,
    )
    # Boolean column with 3 distinct values [True, False, None]
    data_frame = pd.DataFrame({"col1": [True, False, None, False, True]})
    result = expectation.validate(data_frame=data_frame)
    assert isinstance(
        result, DataframeExpectationSuccessMessage
    ), f"Expected DataframeExpectationSuccessMessage but got: {type(result)}"


def test_datetime_column():
    """
    Test the expectation with a datetime column.
    """
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationDistinctColumnValuesEquals",
        column_name="col1",
        expected_value=3,
    )
    # Datetime column with exactly 3 distinct values
    data_frame = pd.DataFrame(
        {
            "col1": pd.to_datetime(
                [
                    "2023-01-01",
                    "2023-01-02",
                    "2023-01-03",
                    "2023-01-02",
                    "2023-01-01",
                ]
            )
        }
    )
    result = expectation.validate(data_frame=data_frame)
    assert isinstance(
        result, DataframeExpectationSuccessMessage
    ), f"Expected DataframeExpectationSuccessMessage but got: {type(result)}"


def test_datetime_column_with_timezone():
    """
    Test the expectation with a datetime column including timezone information.
    """
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationDistinctColumnValuesEquals",
        column_name="col1",
        expected_value=2,
    )
    # Datetime column with timezone - same time in different timezones should be distinct
    data_frame = pd.DataFrame(
        {
            "col1": [
                pd.Timestamp("2023-01-01 12:00:00", tz="UTC"),
                pd.Timestamp("2023-01-01 12:00:00", tz="US/Eastern"),
                pd.Timestamp("2023-01-01 12:00:00", tz="UTC"),
                pd.Timestamp("2023-01-01 12:00:00", tz="US/Eastern"),
            ]
        }
    )
    result = expectation.validate(data_frame=data_frame)
    assert isinstance(
        result, DataframeExpectationSuccessMessage
    ), f"Expected DataframeExpectationSuccessMessage but got: {type(result)}"


def test_mixed_data_types_as_object():
    """
    Test the expectation with a column containing mixed data types (as object dtype).
    """
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationDistinctColumnValuesEquals",
        column_name="col1",
        expected_value=4,
    )
    # Mixed data types: string, int, float, None
    data_frame = pd.DataFrame({"col1": ["text", 42, 3.14, None, "text", 42]})
    result = expectation.validate(data_frame=data_frame)
    assert isinstance(
        result, DataframeExpectationSuccessMessage
    ), f"Expected DataframeExpectationSuccessMessage but got: {type(result)}"


def test_large_dataset_performance():
    """
    Test the expectation with a larger dataset to ensure reasonable performance.
    """
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationDistinctColumnValuesEquals",
        column_name="col1",
        expected_value=1000,
    )
    # Create a DataFrame with exactly 1000 distinct values
    data_frame = pd.DataFrame({"col1": list(range(1000)) * 5})  # 5000 rows, 1000 distinct values
    result = expectation.validate(data_frame=data_frame)
    assert isinstance(
        result, DataframeExpectationSuccessMessage
    ), f"Expected DataframeExpectationSuccessMessage but got: {type(result)}"


def test_suite_pandas_success():
    """
    Test the expectation suite for pandas DataFrame with no violations.
    """
    expectations_suite = DataframeExpectationsSuite().expect_distinct_column_values_equals(
        column_name="col1", expected_value=3
    )
    data_frame = pd.DataFrame({"col1": [1, 2, 3, 2, 1]})  # exactly 3 distinct values
    result = expectations_suite.run(data_frame=data_frame)
    assert result is None, "Expected no exceptions to be raised"


def test_suite_pandas_violations():
    """
    Test the expectation suite for pandas DataFrame with violations.
    """
    expectations_suite = DataframeExpectationsSuite().expect_distinct_column_values_equals(
        column_name="col1", expected_value=5
    )
    data_frame = pd.DataFrame({"col1": [1, 2, 1, 2, 1]})  # 2 distinct values, expected 5
    with pytest.raises(DataframeExpectationsSuiteFailure):
        expectations_suite.run(data_frame=data_frame)


def test_suite_pyspark_success(spark):
    """
    Test the expectation suite for PySpark DataFrame with no violations.
    """
    expectations_suite = DataframeExpectationsSuite().expect_distinct_column_values_equals(
        column_name="col1", expected_value=3
    )
    data_frame = spark.createDataFrame(
        [(1,), (2,), (3,), (2,), (1,)], ["col1"]
    )  # exactly 3 distinct values
    result = expectations_suite.run(data_frame=data_frame)
    assert result is None, "Expected no exceptions to be raised"


def test_suite_pyspark_violations(spark):
    """
    Test the expectation suite for PySpark DataFrame with violations.
    """
    expectations_suite = DataframeExpectationsSuite().expect_distinct_column_values_equals(
        column_name="col1", expected_value=5
    )
    data_frame = spark.createDataFrame(
        [(1,), (2,), (1,), (2,), (1,)], ["col1"]
    )  # 2 distinct values, expected 5
    with pytest.raises(DataframeExpectationsSuiteFailure):
        expectations_suite.run(data_frame=data_frame)


def test_suite_pyspark_column_missing_error(spark):
    """
    Test that an error is raised when the specified column is missing in PySpark DataFrame.
    """
    expectations_suite = DataframeExpectationsSuite().expect_distinct_column_values_equals(
        column_name="col1", expected_value=3
    )
    data_frame = spark.createDataFrame([(1,), (2,), (3,), (4,), (5,)], ["col2"])
    with pytest.raises(DataframeExpectationsSuiteFailure):
        expectations_suite.run(data_frame=data_frame)


def test_categorical_data():
    """
    Test the expectation with categorical data.
    """
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationDistinctColumnValuesEquals",
        column_name="col1",
        expected_value=3,
    )
    # Categorical data with 3 distinct categories
    data_frame = pd.DataFrame({"col1": pd.Categorical(["A", "B", "C", "A", "B", "C", "A"])})
    result = expectation.validate(data_frame=data_frame)
    assert isinstance(
        result, DataframeExpectationSuccessMessage
    ), f"Expected DataframeExpectationSuccessMessage but got: {type(result)}"


def test_duplicate_nan_handling():
    """
    Test that multiple NaN values are counted as one distinct value.
    """
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationDistinctColumnValuesEquals",
        column_name="col1",
        expected_value=3,
    )
    # Multiple NaN values should be counted as 1 distinct value
    data_frame = pd.DataFrame({"col1": [1, 2, None, None, None, 1, 2]})
    result = expectation.validate(data_frame=data_frame)
    assert isinstance(
        result, DataframeExpectationSuccessMessage
    ), f"Expected DataframeExpectationSuccessMessage but got: {type(result)}"


def test_very_large_expected_distinct_values():
    """
    Test the expectation with a very large expected count that doesn't match actual.
    """
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationDistinctColumnValuesEquals",
        column_name="col1",
        expected_value=1000000,
    )
    # Small DataFrame with only 3 distinct values
    data_frame = pd.DataFrame({"col1": [1, 2, 3, 2, 1]})
    result = expectation.validate(data_frame=data_frame)

    expected_failure_message = DataframeExpectationFailureMessage(
        expectation_str=str(expectation),
        data_frame_type=DataFrameType.PANDAS,
        message="Column 'col1' has 3 distinct values, expected exactly 1000000.",
    )
    assert str(result) == str(
        expected_failure_message
    ), f"Expected failure message but got: {result}"


def test_string_with_whitespace_handling():
    """
    Test that strings with different whitespace are treated as distinct.
    """
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationDistinctColumnValuesEquals",
        column_name="col1",
        expected_value=4,
    )
    # Strings with different whitespace should be distinct
    data_frame = pd.DataFrame({"col1": ["test", " test", "test ", " test ", "test"]})
    result = expectation.validate(data_frame=data_frame)
    assert isinstance(
        result, DataframeExpectationSuccessMessage
    ), f"Expected DataframeExpectationSuccessMessage but got: {type(result)}"


def test_numeric_string_vs_numeric():
    """
    Test that numeric strings and numeric values are treated as distinct when in object column.
    """
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationDistinctColumnValuesEquals",
        column_name="col1",
        expected_value=2,
    )
    # String "1" and integer 1 should be distinct in object column
    data_frame = pd.DataFrame({"col1": ["1", 1, "1", 1]}, dtype=object)
    result = expectation.validate(data_frame=data_frame)
    assert isinstance(
        result, DataframeExpectationSuccessMessage
    ), f"Expected DataframeExpectationSuccessMessage but got: {type(result)}"
