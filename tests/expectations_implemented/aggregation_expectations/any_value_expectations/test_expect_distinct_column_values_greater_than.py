import pytest
import pandas as pd

from dataframe_expectations import DataFrameType
from dataframe_expectations.expectations.expectation_registry import (
    DataFrameExpectationRegistry,
)
from dataframe_expectations.expectations_suite import (
    DataFrameExpectationsSuite,
    DataFrameExpectationsSuiteFailure,
)
from dataframe_expectations.result_message import (
    DataFrameExpectationFailureMessage,
    DataFrameExpectationSuccessMessage,
)


def test_expectation_name():
    """
    Test that the expectation name is correctly returned.
    """
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationDistinctColumnValuesGreaterThan",
        column_name="col1",
        threshold=2,
    )
    assert (
        expectation.get_expectation_name() == "ExpectationDistinctColumnValuesGreaterThan"
    ), f"Expected 'ExpectationDistinctColumnValuesGreaterThan' but got: {expectation.get_expectation_name()}"


def test_expectation_pandas_success():
    """
    Test the expectation for pandas DataFrame with no violations.
    """
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationDistinctColumnValuesGreaterThan",
        column_name="col1",
        threshold=2,
    )
    # DataFrame with 3 distinct values [1, 2, 3] which is > 2
    data_frame = pd.DataFrame({"col1": [1, 2, 3, 2, 1]})
    result = expectation.validate(data_frame=data_frame)
    assert str(result) == str(
        DataFrameExpectationSuccessMessage(
            expectation_name="ExpectationDistinctColumnValuesGreaterThan"
        )
    ), f"Expected success message but got: {result}"


def test_expectation_pandas_success_with_nulls():
    """
    Test the expectation for pandas DataFrame with NaN values included in distinct count.
    """
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationDistinctColumnValuesGreaterThan",
        column_name="col1",
        threshold=2,
    )
    # DataFrame with 3 distinct values [1, 2, NaN] which is > 2
    data_frame = pd.DataFrame({"col1": [1, 2, None, 2, 1]})
    result = expectation.validate(data_frame=data_frame)
    assert str(result) == str(
        DataFrameExpectationSuccessMessage(
            expectation_name="ExpectationDistinctColumnValuesGreaterThan"
        )
    ), f"Expected success message but got: {result}"


def test_expectation_pandas_success_exact_boundary():
    """
    Test the expectation for pandas DataFrame with distinct count exactly at boundary (exclusive).
    """
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationDistinctColumnValuesGreaterThan",
        column_name="col1",
        threshold=2,
    )
    # DataFrame with 3 distinct values [1, 2, 3] which is > 2
    data_frame = pd.DataFrame({"col1": [1, 2, 3, 2, 1]})
    result = expectation.validate(data_frame=data_frame)
    assert isinstance(
        result, DataFrameExpectationSuccessMessage
    ), f"Expected DataFrameExpectationSuccessMessage but got: {type(result)}"


def test_expectation_pandas_violations_equal_to_threshold():
    """
    Test the expectation for pandas DataFrame with distinct count equal to threshold (should fail).
    """
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationDistinctColumnValuesGreaterThan",
        column_name="col1",
        threshold=3,
    )
    # DataFrame with exactly 3 distinct values [1, 2, 3] which is NOT > 3
    data_frame = pd.DataFrame({"col1": [1, 2, 3, 2, 1]})
    result = expectation.validate(data_frame=data_frame)

    expected_failure_message = DataFrameExpectationFailureMessage(
        expectation_str=str(expectation),
        data_frame_type=DataFrameType.PANDAS,
        message="Column 'col1' has 3 distinct values, expected more than 3.",
    )
    assert str(result) == str(
        expected_failure_message
    ), f"Expected failure message but got: {result}"


def test_expectation_pandas_violations_below_threshold():
    """
    Test the expectation for pandas DataFrame with distinct count below threshold.
    """
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationDistinctColumnValuesGreaterThan",
        column_name="col1",
        threshold=5,
    )
    # DataFrame with 2 distinct values [1, 2] which is NOT > 5
    data_frame = pd.DataFrame({"col1": [1, 2, 1, 2, 1]})
    result = expectation.validate(data_frame=data_frame)

    expected_failure_message = DataFrameExpectationFailureMessage(
        expectation_str=str(expectation),
        data_frame_type=DataFrameType.PANDAS,
        message="Column 'col1' has 2 distinct values, expected more than 5.",
    )
    assert str(result) == str(
        expected_failure_message
    ), f"Expected failure message but got: {result}"


def test_expectation_pandas_zero_threshold():
    """
    Test the expectation for pandas DataFrame with zero threshold.
    """
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationDistinctColumnValuesGreaterThan",
        column_name="col1",
        threshold=0,
    )
    # Any non-empty DataFrame should have > 0 distinct values
    data_frame = pd.DataFrame({"col1": [1, 1, 1]})  # 1 distinct value > 0
    result = expectation.validate(data_frame=data_frame)
    assert isinstance(
        result, DataFrameExpectationSuccessMessage
    ), f"Expected DataFrameExpectationSuccessMessage but got: {type(result)}"


def test_expectation_pandas_empty_dataframe():
    """
    Test the expectation for pandas DataFrame that is empty.
    """
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationDistinctColumnValuesGreaterThan",
        column_name="col1",
        threshold=0,
    )
    # Empty DataFrame has 0 distinct values, which is NOT > 0
    data_frame = pd.DataFrame({"col1": []})
    result = expectation.validate(data_frame=data_frame)

    expected_failure_message = DataFrameExpectationFailureMessage(
        expectation_str=str(expectation),
        data_frame_type=DataFrameType.PANDAS,
        message="Column 'col1' has 0 distinct values, expected more than 0.",
    )
    assert str(result) == str(
        expected_failure_message
    ), f"Expected failure message but got: {result}"


def test_expectation_pyspark_success(spark):
    """
    Test the expectation for PySpark DataFrame with no violations.
    """
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationDistinctColumnValuesGreaterThan",
        column_name="col1",
        threshold=2,
    )
    # DataFrame with 3 distinct values [1, 2, 3] which is > 2
    data_frame = spark.createDataFrame([(1,), (2,), (3,), (2,), (1,)], ["col1"])
    result = expectation.validate(data_frame=data_frame)
    assert str(result) == str(
        DataFrameExpectationSuccessMessage(
            expectation_name="ExpectationDistinctColumnValuesGreaterThan"
        )
    ), f"Expected success message but got: {result}"


def test_expectation_pyspark_success_with_nulls(spark):
    """
    Test the expectation for PySpark DataFrame with null values included in distinct count.
    """
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationDistinctColumnValuesGreaterThan",
        column_name="col1",
        threshold=2,
    )
    # DataFrame with 3 distinct values [1, 2, null] which is > 2
    data_frame = spark.createDataFrame([(1,), (2,), (None,), (2,), (1,)], ["col1"])
    result = expectation.validate(data_frame=data_frame)
    assert str(result) == str(
        DataFrameExpectationSuccessMessage(
            expectation_name="ExpectationDistinctColumnValuesGreaterThan"
        )
    ), f"Expected success message but got: {result}"


def test_expectation_pyspark_violations_equal_to_threshold(spark):
    """
    Test the expectation for PySpark DataFrame with distinct count equal to threshold (should fail).
    """
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationDistinctColumnValuesGreaterThan",
        column_name="col1",
        threshold=3,
    )
    # DataFrame with exactly 3 distinct values [1, 2, 3] which is NOT > 3
    data_frame = spark.createDataFrame([(1,), (2,), (3,), (2,), (1,)], ["col1"])
    result = expectation.validate(data_frame=data_frame)

    expected_failure_message = DataFrameExpectationFailureMessage(
        expectation_str=str(expectation),
        data_frame_type=DataFrameType.PYSPARK,
        message="Column 'col1' has 3 distinct values, expected more than 3.",
    )
    assert str(result) == str(
        expected_failure_message
    ), f"Expected failure message but got: {result}"


def test_expectation_pyspark_violations_below_threshold(spark):
    """
    Test the expectation for PySpark DataFrame with distinct count below threshold.
    """
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationDistinctColumnValuesGreaterThan",
        column_name="col1",
        threshold=5,
    )
    # DataFrame with 2 distinct values [1, 2] which is NOT > 5
    data_frame = spark.createDataFrame([(1,), (2,), (1,), (2,), (1,)], ["col1"])
    result = expectation.validate(data_frame=data_frame)

    expected_failure_message = DataFrameExpectationFailureMessage(
        expectation_str=str(expectation),
        data_frame_type=DataFrameType.PYSPARK,
        message="Column 'col1' has 2 distinct values, expected more than 5.",
    )
    assert str(result) == str(
        expected_failure_message
    ), f"Expected failure message but got: {result}"


def test_expectation_pyspark_empty_dataframe(spark):
    """
    Test the expectation for PySpark DataFrame that is empty.
    """
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationDistinctColumnValuesGreaterThan",
        column_name="col1",
        threshold=0,
    )
    # Empty DataFrame has 0 distinct values, which is NOT > 0
    data_frame = spark.createDataFrame([], "col1 INT")
    result = expectation.validate(data_frame=data_frame)

    expected_failure_message = DataFrameExpectationFailureMessage(
        expectation_str=str(expectation),
        data_frame_type=DataFrameType.PYSPARK,
        message="Column 'col1' has 0 distinct values, expected more than 0.",
    )
    assert str(result) == str(
        expected_failure_message
    ), f"Expected failure message but got: {result}"


def test_column_missing_error():
    """
    Test that an error is raised when the specified column is missing.
    """
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationDistinctColumnValuesGreaterThan",
        column_name="col1",
        threshold=2,
    )
    data_frame = pd.DataFrame({"col2": [1, 2, 3, 4, 5]})
    result = expectation.validate(data_frame=data_frame)
    expected_failure_message = DataFrameExpectationFailureMessage(
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
    # Test negative threshold
    with pytest.raises(ValueError) as context:
        DataFrameExpectationRegistry.get_expectation(
            expectation_name="ExpectationDistinctColumnValuesGreaterThan",
            column_name="col1",
            threshold=-1,
        )
    assert "threshold must be non-negative" in str(
        context.value
    ), f"Expected 'threshold must be non-negative' in error message: {str(context.value)}"


def test_string_column_with_mixed_values():
    """
    Test the expectation with a string column containing mixed values.
    """
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationDistinctColumnValuesGreaterThan",
        column_name="col1",
        threshold=3,
    )
    # String column with 4 distinct values ["A", "B", "C", None] which is > 3
    data_frame = pd.DataFrame({"col1": ["A", "B", "C", "B", "A", None]})
    result = expectation.validate(data_frame=data_frame)
    assert isinstance(
        result, DataFrameExpectationSuccessMessage
    ), f"Expected DataFrameExpectationSuccessMessage but got: {type(result)}"


def test_string_column_case_sensitive():
    """
    Test that string comparisons are case-sensitive for distinct counting.
    """
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationDistinctColumnValuesGreaterThan",
        column_name="col1",
        threshold=3,
    )
    # String column with 4 distinct values ["a", "A", "b", "B"] which is > 3
    data_frame = pd.DataFrame({"col1": ["a", "A", "b", "B", "a", "A"]})
    result = expectation.validate(data_frame=data_frame)
    assert isinstance(
        result, DataFrameExpectationSuccessMessage
    ), f"Expected DataFrameExpectationSuccessMessage but got: {type(result)}"


def test_numeric_column_with_floats():
    """
    Test the expectation with a numeric column containing floats.
    """
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationDistinctColumnValuesGreaterThan",
        column_name="col1",
        threshold=2,
    )
    # Float column with 3 distinct values [1.1, 2.2, 3.3] which is > 2
    data_frame = pd.DataFrame({"col1": [1.1, 2.2, 3.3, 2.2, 1.1]})
    result = expectation.validate(data_frame=data_frame)
    assert isinstance(
        result, DataFrameExpectationSuccessMessage
    ), f"Expected DataFrameExpectationSuccessMessage but got: {type(result)}"


def test_boolean_column():
    """
    Test the expectation with a boolean column.
    """
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationDistinctColumnValuesGreaterThan",
        column_name="col1",
        threshold=1,
    )
    # Boolean column with 2 distinct values [True, False] which is > 1
    data_frame = pd.DataFrame({"col1": [True, False, True, False, True]})
    result = expectation.validate(data_frame=data_frame)
    assert isinstance(
        result, DataFrameExpectationSuccessMessage
    ), f"Expected DataFrameExpectationSuccessMessage but got: {type(result)}"


def test_boolean_column_failure():
    """
    Test the expectation with a boolean column that fails the threshold.
    """
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationDistinctColumnValuesGreaterThan",
        column_name="col1",
        threshold=2,
    )
    # Boolean column with only 1 distinct value [True] which is NOT > 2
    data_frame = pd.DataFrame({"col1": [True, True, True, True, True]})
    result = expectation.validate(data_frame=data_frame)

    expected_failure_message = DataFrameExpectationFailureMessage(
        expectation_str=str(expectation),
        data_frame_type=DataFrameType.PANDAS,
        message="Column 'col1' has 1 distinct values, expected more than 2.",
    )
    assert str(result) == str(
        expected_failure_message
    ), f"Expected failure message but got: {result}"


def test_datetime_column():
    """
    Test the expectation with a datetime column.
    """
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationDistinctColumnValuesGreaterThan",
        column_name="col1",
        threshold=2,
    )
    # Datetime column with 3 distinct values which is > 2
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
        result, DataFrameExpectationSuccessMessage
    ), f"Expected DataFrameExpectationSuccessMessage but got: {type(result)}"


def test_mixed_data_types_as_object():
    """
    Test the expectation with a column containing mixed data types.
    """
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationDistinctColumnValuesGreaterThan",
        column_name="col1",
        threshold=3,
    )
    # Mixed data types: 4 distinct values ["text", 42, 3.14, None] which is > 3
    data_frame = pd.DataFrame({"col1": ["text", 42, 3.14, None, "text", 42]})
    result = expectation.validate(data_frame=data_frame)
    assert isinstance(
        result, DataFrameExpectationSuccessMessage
    ), f"Expected DataFrameExpectationSuccessMessage but got: {type(result)}"


def test_large_dataset_performance():
    """
    Test the expectation with a larger dataset to ensure reasonable performance.
    """
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationDistinctColumnValuesGreaterThan",
        column_name="col1",
        threshold=999,
    )
    # Create a DataFrame with exactly 1000 distinct values (> 999)
    data_frame = pd.DataFrame({"col1": list(range(1000)) * 5})  # 5000 rows, 1000 distinct values
    result = expectation.validate(data_frame=data_frame)
    assert isinstance(
        result, DataFrameExpectationSuccessMessage
    ), f"Expected DataFrameExpectationSuccessMessage but got: {type(result)}"


def test_large_dataset_failure():
    """
    Test the expectation with a larger dataset that fails the threshold.
    """
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationDistinctColumnValuesGreaterThan",
        column_name="col1",
        threshold=1000,
    )
    # Create a DataFrame with exactly 1000 distinct values (NOT > 1000)
    data_frame = pd.DataFrame({"col1": list(range(1000)) * 5})  # 5000 rows, 1000 distinct values
    result = expectation.validate(data_frame=data_frame)

    expected_failure_message = DataFrameExpectationFailureMessage(
        expectation_str=str(expectation),
        data_frame_type=DataFrameType.PANDAS,
        message="Column 'col1' has 1000 distinct values, expected more than 1000.",
    )
    assert str(result) == str(
        expected_failure_message
    ), f"Expected failure message but got: {result}"


def test_suite_pandas_success():
    """
    Test the expectation suite for pandas DataFrame with no violations.
    """
    expectations_suite = DataFrameExpectationsSuite().expect_distinct_column_values_greater_than(
        column_name="col1", threshold=2
    )
    data_frame = pd.DataFrame({"col1": [1, 2, 3, 2, 1]})  # 3 distinct values > 2
    result = expectations_suite.run(data_frame=data_frame)
    assert result is None, "Expected no exceptions to be raised"


def test_suite_pandas_violations():
    """
    Test the expectation suite for pandas DataFrame with violations.
    """
    expectations_suite = DataFrameExpectationsSuite().expect_distinct_column_values_greater_than(
        column_name="col1", threshold=5
    )
    data_frame = pd.DataFrame({"col1": [1, 2, 1, 2, 1]})  # 2 distinct values, need > 5
    with pytest.raises(DataFrameExpectationsSuiteFailure):
        expectations_suite.run(data_frame=data_frame)


def test_suite_pyspark_success(spark):
    """
    Test the expectation suite for PySpark DataFrame with no violations.
    """
    expectations_suite = DataFrameExpectationsSuite().expect_distinct_column_values_greater_than(
        column_name="col1", threshold=2
    )
    data_frame = spark.createDataFrame(
        [(1,), (2,), (3,), (2,), (1,)], ["col1"]
    )  # 3 distinct values > 2
    result = expectations_suite.run(data_frame=data_frame)
    assert result is None, "Expected no exceptions to be raised"


def test_suite_pyspark_violations(spark):
    """
    Test the expectation suite for PySpark DataFrame with violations.
    """
    expectations_suite = DataFrameExpectationsSuite().expect_distinct_column_values_greater_than(
        column_name="col1", threshold=5
    )
    data_frame = spark.createDataFrame(
        [(1,), (2,), (1,), (2,), (1,)], ["col1"]
    )  # 2 distinct values, need > 5
    with pytest.raises(DataFrameExpectationsSuiteFailure):
        expectations_suite.run(data_frame=data_frame)


def test_suite_pyspark_column_missing_error(spark):
    """
    Test that an error is raised when the specified column is missing in PySpark DataFrame.
    """
    expectations_suite = DataFrameExpectationsSuite().expect_distinct_column_values_greater_than(
        column_name="col1", threshold=2
    )
    data_frame = spark.createDataFrame([(1,), (2,), (3,), (4,), (5,)], ["col2"])
    with pytest.raises(DataFrameExpectationsSuiteFailure):
        expectations_suite.run(data_frame=data_frame)


def test_categorical_data():
    """
    Test the expectation with categorical data.
    """
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationDistinctColumnValuesGreaterThan",
        column_name="col1",
        threshold=2,
    )
    # Categorical data with 3 distinct categories which is > 2
    data_frame = pd.DataFrame({"col1": pd.Categorical(["A", "B", "C", "A", "B", "C", "A"])})
    result = expectation.validate(data_frame=data_frame)
    assert isinstance(
        result, DataFrameExpectationSuccessMessage
    ), f"Expected DataFrameExpectationSuccessMessage but got: {type(result)}"


def test_duplicate_nan_handling():
    """
    Test that multiple NaN values are counted as one distinct value.
    """
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationDistinctColumnValuesGreaterThan",
        column_name="col1",
        threshold=2,
    )
    # Multiple NaN values should be counted as 1, total = 3 distinct values > 2
    data_frame = pd.DataFrame({"col1": [1, 2, None, None, None, 1, 2]})
    result = expectation.validate(data_frame=data_frame)
    assert isinstance(
        result, DataFrameExpectationSuccessMessage
    ), f"Expected DataFrameExpectationSuccessMessage but got: {type(result)}"


def test_single_distinct_value_success():
    """
    Test the expectation with only one distinct value that passes threshold.
    """
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationDistinctColumnValuesGreaterThan",
        column_name="col1",
        threshold=0,
    )
    # Single distinct value (1) which is > 0
    data_frame = pd.DataFrame({"col1": [5, 5, 5, 5, 5]})
    result = expectation.validate(data_frame=data_frame)
    assert isinstance(
        result, DataFrameExpectationSuccessMessage
    ), f"Expected DataFrameExpectationSuccessMessage but got: {type(result)}"


def test_string_with_whitespace_handling():
    """
    Test that strings with different whitespace are treated as distinct.
    """
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationDistinctColumnValuesGreaterThan",
        column_name="col1",
        threshold=3,
    )
    # 4 distinct strings with different whitespace > 3
    data_frame = pd.DataFrame({"col1": ["test", " test", "test ", " test ", "test"]})
    result = expectation.validate(data_frame=data_frame)
    assert isinstance(
        result, DataFrameExpectationSuccessMessage
    ), f"Expected DataFrameExpectationSuccessMessage but got: {type(result)}"


def test_numeric_string_vs_numeric():
    """
    Test that numeric strings and numeric values are treated as distinct.
    """
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationDistinctColumnValuesGreaterThan",
        column_name="col1",
        threshold=1,
    )
    # String "1" and integer 1 are distinct, so 2 distinct values > 1
    data_frame = pd.DataFrame({"col1": ["1", 1, "1", 1]}, dtype=object)
    result = expectation.validate(data_frame=data_frame)
    assert isinstance(
        result, DataFrameExpectationSuccessMessage
    ), f"Expected DataFrameExpectationSuccessMessage but got: {type(result)}"


def test_very_high_threshold():
    """
    Test the expectation with a very high threshold that cannot be met.
    """
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationDistinctColumnValuesGreaterThan",
        column_name="col1",
        threshold=1000000,
    )
    # Small DataFrame with only 3 distinct values
    data_frame = pd.DataFrame({"col1": [1, 2, 3, 2, 1]})
    result = expectation.validate(data_frame=data_frame)

    expected_failure_message = DataFrameExpectationFailureMessage(
        expectation_str=str(expectation),
        data_frame_type=DataFrameType.PANDAS,
        message="Column 'col1' has 3 distinct values, expected more than 1000000.",
    )
    assert str(result) == str(
        expected_failure_message
    ), f"Expected failure message but got: {result}"


def test_exclusive_boundary_validation():
    """
    Test that the boundary is truly exclusive (not inclusive).
    """
    # Test with threshold = 5, actual = 5 (should fail because 5 is NOT > 5)
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationDistinctColumnValuesGreaterThan",
        column_name="col1",
        threshold=5,
    )
    data_frame = pd.DataFrame({"col1": [1, 2, 3, 4, 5, 1, 2]})  # exactly 5 distinct values
    result = expectation.validate(data_frame=data_frame)
    assert isinstance(
        result, DataFrameExpectationFailureMessage
    ), f"Expected DataFrameExpectationFailureMessage but got: {type(result)}"

    # Test with threshold = 4, actual = 5 (should pass because 5 > 4)
    expectation_pass = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationDistinctColumnValuesGreaterThan",
        column_name="col1",
        threshold=4,
    )
    result_pass = expectation_pass.validate(data_frame=data_frame)
    assert isinstance(
        result_pass, DataFrameExpectationSuccessMessage
    ), f"Expected DataFrameExpectationSuccessMessage but got: {type(result_pass)}"
