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
        expectation_name="ExpectationDistinctColumnValuesBetween",
        column_name="col1",
        min_value=2,
        max_value=5,
    )
    assert (
        expectation.get_expectation_name() == "ExpectationDistinctColumnValuesBetween"
    ), f"Expected 'ExpectationDistinctColumnValuesBetween' but got: {expectation.get_expectation_name()}"


def test_expectation_pandas_success():
    """
    Test the expectation for pandas DataFrame with no violations.
    """
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationDistinctColumnValuesBetween",
        column_name="col1",
        min_value=2,
        max_value=5,
    )
    # DataFrame with 3 distinct values [1, 2, 3] which is within range [2, 5]
    data_frame = pd.DataFrame({"col1": [1, 2, 3, 2, 1]})
    result = expectation.validate(data_frame=data_frame)
    assert str(result) == str(
        DataFrameExpectationSuccessMessage(
            expectation_name="ExpectationDistinctColumnValuesBetween"
        )
    ), f"Expected success message but got: {result}"


def test_expectation_pandas_success_with_nulls():
    """
    Test the expectation for pandas DataFrame with NaN values included in distinct count.
    """
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationDistinctColumnValuesBetween",
        column_name="col1",
        min_value=3,
        max_value=4,
    )
    # DataFrame with 3 distinct values [1, 2, NaN] which is within range [3, 4]
    data_frame = pd.DataFrame({"col1": [1, 2, None, 2, 1]})
    result = expectation.validate(data_frame=data_frame)
    assert str(result) == str(
        DataFrameExpectationSuccessMessage(
            expectation_name="ExpectationDistinctColumnValuesBetween"
        )
    ), f"Expected success message but got: {result}"


def test_expectation_pandas_violations_too_few():
    """
    Test the expectation for pandas DataFrame with too few distinct values.
    """
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationDistinctColumnValuesBetween",
        column_name="col1",
        min_value=4,
        max_value=6,
    )
    # DataFrame with 2 distinct values [1, 2] which is below range [4, 6]
    data_frame = pd.DataFrame({"col1": [1, 2, 1, 2, 1]})
    result = expectation.validate(data_frame=data_frame)

    expected_failure_message = DataFrameExpectationFailureMessage(
        expectation_str=str(expectation),
        data_frame_type=DataFrameType.PANDAS,
        message="Column 'col1' has 2 distinct values, expected between 4 and 6.",
    )
    assert str(result) == str(
        expected_failure_message
    ), f"Expected failure message but got: {result}"


def test_expectation_pandas_violations_too_many():
    """
    Test the expectation for pandas DataFrame with too many distinct values.
    """
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationDistinctColumnValuesBetween",
        column_name="col1",
        min_value=2,
        max_value=3,
    )
    # DataFrame with 5 distinct values [1, 2, 3, 4, 5] which is above range [2, 3]
    data_frame = pd.DataFrame({"col1": [1, 2, 3, 4, 5]})
    result = expectation.validate(data_frame=data_frame)

    expected_failure_message = DataFrameExpectationFailureMessage(
        expectation_str=str(expectation),
        data_frame_type=DataFrameType.PANDAS,
        message="Column 'col1' has 5 distinct values, expected between 2 and 3.",
    )
    assert str(result) == str(
        expected_failure_message
    ), f"Expected failure message but got: {result}"


def test_expectation_pandas_exact_boundaries():
    """
    Test the expectation for pandas DataFrame with distinct counts exactly at boundaries.
    """
    # Test exact minimum boundary
    expectation_min = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationDistinctColumnValuesBetween",
        column_name="col1",
        min_value=3,
        max_value=5,
    )
    data_frame_min = pd.DataFrame({"col1": [1, 2, 3, 2, 1]})  # 3 distinct values
    result_min = expectation_min.validate(data_frame=data_frame_min)
    assert isinstance(
        result_min, DataFrameExpectationSuccessMessage
    ), f"Expected DataFrameExpectationSuccessMessage but got: {type(result_min)}"

    # Test exact maximum boundary
    expectation_max = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationDistinctColumnValuesBetween",
        column_name="col1",
        min_value=3,
        max_value=5,
    )
    data_frame_max = pd.DataFrame({"col1": [1, 2, 3, 4, 5, 1]})  # 5 distinct values
    result_max = expectation_max.validate(data_frame=data_frame_max)
    assert isinstance(
        result_max, DataFrameExpectationSuccessMessage
    ), f"Expected DataFrameExpectationSuccessMessage but got: {type(result_max)}"


def test_expectation_pyspark_success(spark):
    """
    Test the expectation for PySpark DataFrame with no violations.
    """
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationDistinctColumnValuesBetween",
        column_name="col1",
        min_value=2,
        max_value=5,
    )
    # DataFrame with 3 distinct values [1, 2, 3] which is within range [2, 5]
    data_frame = spark.createDataFrame([(1,), (2,), (3,), (2,), (1,)], ["col1"])
    result = expectation.validate(data_frame=data_frame)
    assert str(result) == str(
        DataFrameExpectationSuccessMessage(
            expectation_name="ExpectationDistinctColumnValuesBetween"
        )
    ), f"Expected success message but got: {result}"


def test_expectation_pyspark_success_with_nulls(spark):
    """
    Test the expectation for PySpark DataFrame with null values included in distinct count.
    """
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationDistinctColumnValuesBetween",
        column_name="col1",
        min_value=3,
        max_value=4,
    )
    # DataFrame with 3 distinct values [1, 2, null] which is within range [3, 4]
    data_frame = spark.createDataFrame([(1,), (2,), (None,), (2,), (1,)], ["col1"])
    result = expectation.validate(data_frame=data_frame)
    assert str(result) == str(
        DataFrameExpectationSuccessMessage(
            expectation_name="ExpectationDistinctColumnValuesBetween"
        )
    ), f"Expected success message but got: {result}"


def test_expectation_pyspark_violations_too_few(spark):
    """
    Test the expectation for PySpark DataFrame with too few distinct values.
    """
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationDistinctColumnValuesBetween",
        column_name="col1",
        min_value=4,
        max_value=6,
    )
    # DataFrame with 2 distinct values [1, 2] which is below range [4, 6]
    data_frame = spark.createDataFrame([(1,), (2,), (1,), (2,), (1,)], ["col1"])
    result = expectation.validate(data_frame=data_frame)

    expected_failure_message = DataFrameExpectationFailureMessage(
        expectation_str=str(expectation),
        data_frame_type=DataFrameType.PYSPARK,
        message="Column 'col1' has 2 distinct values, expected between 4 and 6.",
    )
    assert str(result) == str(
        expected_failure_message
    ), f"Expected failure message but got: {result}"


def test_expectation_pyspark_violations_too_many(spark):
    """
    Test the expectation for PySpark DataFrame with too many distinct values.
    """
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationDistinctColumnValuesBetween",
        column_name="col1",
        min_value=2,
        max_value=3,
    )
    # DataFrame with 5 distinct values [1, 2, 3, 4, 5] which is above range [2, 3]
    data_frame = spark.createDataFrame([(1,), (2,), (3,), (4,), (5,)], ["col1"])
    result = expectation.validate(data_frame=data_frame)

    expected_failure_message = DataFrameExpectationFailureMessage(
        expectation_str=str(expectation),
        data_frame_type=DataFrameType.PYSPARK,
        message="Column 'col1' has 5 distinct values, expected between 2 and 3.",
    )
    assert str(result) == str(
        expected_failure_message
    ), f"Expected failure message but got: {result}"


def test_expectation_pyspark_exact_boundaries(spark):
    """
    Test the expectation for PySpark DataFrame with distinct counts exactly at boundaries.
    """
    # Test exact minimum boundary
    expectation_min = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationDistinctColumnValuesBetween",
        column_name="col1",
        min_value=3,
        max_value=5,
    )
    data_frame_min = spark.createDataFrame(
        [(1,), (2,), (3,), (2,), (1,)], ["col1"]
    )  # 3 distinct values
    result_min = expectation_min.validate(data_frame=data_frame_min)
    assert isinstance(
        result_min, DataFrameExpectationSuccessMessage
    ), f"Expected DataFrameExpectationSuccessMessage but got: {type(result_min)}"

    # Test exact maximum boundary
    expectation_max = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationDistinctColumnValuesBetween",
        column_name="col1",
        min_value=3,
        max_value=5,
    )
    data_frame_max = spark.createDataFrame(
        [(1,), (2,), (3,), (4,), (5,), (1,)], ["col1"]
    )  # 5 distinct values
    result_max = expectation_max.validate(data_frame=data_frame_max)
    assert isinstance(
        result_max, DataFrameExpectationSuccessMessage
    ), f"Expected DataFrameExpectationSuccessMessage but got: {type(result_max)}"


def test_column_missing_error():
    """
    Test that an error is raised when the specified column is missing.
    """
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationDistinctColumnValuesBetween",
        column_name="col1",
        min_value=2,
        max_value=5,
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
    # Test negative min_value
    with pytest.raises(ValueError) as context:
        DataFrameExpectationRegistry.get_expectation(
            expectation_name="ExpectationDistinctColumnValuesBetween",
            column_name="col1",
            min_value=-1,
            max_value=5,
        )
    assert "min_value must be non-negative" in str(
        context.value
    ), f"Expected 'min_value must be non-negative' in error message: {str(context.value)}"

    # Test negative max_value
    with pytest.raises(ValueError) as context:
        DataFrameExpectationRegistry.get_expectation(
            expectation_name="ExpectationDistinctColumnValuesBetween",
            column_name="col1",
            min_value=2,
            max_value=-1,
        )
    assert "max_value must be non-negative" in str(
        context.value
    ), f"Expected 'max_value must be non-negative' in error message: {str(context.value)}"

    # Test min_value > max_value
    with pytest.raises(ValueError) as context:
        DataFrameExpectationRegistry.get_expectation(
            expectation_name="ExpectationDistinctColumnValuesBetween",
            column_name="col1",
            min_value=5,
            max_value=2,
        )
    assert "min_value (5) must be <= max_value (2)" in str(
        context.value
    ), f"Expected 'min_value (5) must be <= max_value (2)' in error message: {str(context.value)}"


def test_edge_case_zero_range():
    """
    Test the expectation when min_value equals max_value (zero range).
    """
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationDistinctColumnValuesBetween",
        column_name="col1",
        min_value=3,
        max_value=3,
    )
    # DataFrame with exactly 3 distinct values
    data_frame = pd.DataFrame({"col1": [1, 2, 3, 2, 1]})
    result = expectation.validate(data_frame=data_frame)
    assert isinstance(
        result, DataFrameExpectationSuccessMessage
    ), f"Expected DataFrameExpectationSuccessMessage but got: {type(result)}"

    # DataFrame with 2 distinct values (should fail)
    data_frame_fail = pd.DataFrame({"col1": [1, 2, 1, 2, 1]})
    result_fail = expectation.validate(data_frame=data_frame_fail)
    assert isinstance(
        result_fail, DataFrameExpectationFailureMessage
    ), f"Expected DataFrameExpectationFailureMessage but got: {type(result_fail)}"


def test_edge_case_empty_dataframe():
    """
    Test the expectation with an empty DataFrame.
    """
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationDistinctColumnValuesBetween",
        column_name="col1",
        min_value=0,
        max_value=5,
    )
    # Empty DataFrame should have 0 distinct values
    data_frame = pd.DataFrame({"col1": []})
    result = expectation.validate(data_frame=data_frame)
    assert isinstance(
        result, DataFrameExpectationSuccessMessage
    ), f"Expected DataFrameExpectationSuccessMessage but got: {type(result)}"


def test_edge_case_single_value():
    """
    Test the expectation with a DataFrame containing a single distinct value.
    """
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationDistinctColumnValuesBetween",
        column_name="col1",
        min_value=1,
        max_value=1,
    )
    # DataFrame with 1 distinct value
    data_frame = pd.DataFrame({"col1": [1, 1, 1, 1, 1]})
    result = expectation.validate(data_frame=data_frame)
    assert isinstance(
        result, DataFrameExpectationSuccessMessage
    ), f"Expected DataFrameExpectationSuccessMessage but got: {type(result)}"


def test_suite_pandas_success():
    """
    Test the expectation suite for pandas DataFrame with no violations.
    """
    expectations_suite = DataFrameExpectationsSuite().expect_distinct_column_values_between(
        column_name="col1", min_value=2, max_value=5
    )
    data_frame = pd.DataFrame({"col1": [1, 2, 3, 2, 1]})  # 3 distinct values
    result = expectations_suite.run(data_frame=data_frame)
    assert result is None, "Expected no exceptions to be raised"


def test_suite_pandas_violations():
    """
    Test the expectation suite for pandas DataFrame with violations.
    """
    expectations_suite = DataFrameExpectationsSuite().expect_distinct_column_values_between(
        column_name="col1", min_value=4, max_value=6
    )
    data_frame = pd.DataFrame({"col1": [1, 2, 1, 2, 1]})  # 2 distinct values, expected 4-6
    with pytest.raises(DataFrameExpectationsSuiteFailure):
        expectations_suite.run(data_frame=data_frame)


def test_suite_pyspark_success(spark):
    """
    Test the expectation suite for PySpark DataFrame with no violations.
    """
    expectations_suite = DataFrameExpectationsSuite().expect_distinct_column_values_between(
        column_name="col1", min_value=2, max_value=5
    )
    data_frame = spark.createDataFrame(
        [(1,), (2,), (3,), (2,), (1,)], ["col1"]
    )  # 3 distinct values
    result = expectations_suite.run(data_frame=data_frame)
    assert result is None, "Expected no exceptions to be raised"


def test_suite_pyspark_violations(spark):
    """
    Test the expectation suite for PySpark DataFrame with violations.
    """
    expectations_suite = DataFrameExpectationsSuite().expect_distinct_column_values_between(
        column_name="col1", min_value=4, max_value=6
    )
    data_frame = spark.createDataFrame(
        [(1,), (2,), (1,), (2,), (1,)], ["col1"]
    )  # 2 distinct values, expected 4-6
    with pytest.raises(DataFrameExpectationsSuiteFailure):
        expectations_suite.run(data_frame=data_frame)


def test_suite_pyspark_column_missing_error(spark):
    """
    Test that an error is raised when the specified column is missing in PySpark DataFrame.
    """
    expectations_suite = DataFrameExpectationsSuite().expect_distinct_column_values_between(
        column_name="col1", min_value=2, max_value=5
    )
    data_frame = spark.createDataFrame([(1,), (2,), (3,), (4,), (5,)], ["col2"])
    with pytest.raises(DataFrameExpectationsSuiteFailure):
        expectations_suite.run(data_frame=data_frame)


def test_string_column_with_mixed_values():
    """
    Test the expectation with a string column containing mixed values.
    """
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationDistinctColumnValuesBetween",
        column_name="col1",
        min_value=3,
        max_value=5,
    )
    # String column with 4 distinct values ["A", "B", "C", None]
    data_frame = pd.DataFrame({"col1": ["A", "B", "C", "B", "A", None]})
    result = expectation.validate(data_frame=data_frame)
    assert isinstance(
        result, DataFrameExpectationSuccessMessage
    ), f"Expected DataFrameExpectationSuccessMessage but got: {type(result)}"


def test_numeric_column_with_floats():
    """
    Test the expectation with a numeric column containing floats.
    """
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationDistinctColumnValuesBetween",
        column_name="col1",
        min_value=2,
        max_value=4,
    )
    # Float column with 3 distinct values [1.1, 2.2, 3.3]
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
        expectation_name="ExpectationDistinctColumnValuesBetween",
        column_name="col1",
        min_value=2,
        max_value=2,
    )
    # Boolean column with 2 distinct values [True, False]
    data_frame = pd.DataFrame({"col1": [True, False, True, False, True]})
    result = expectation.validate(data_frame=data_frame)
    assert isinstance(
        result, DataFrameExpectationSuccessMessage
    ), f"Expected DataFrameExpectationSuccessMessage but got: {type(result)}"


def test_datetime_column():
    """
    Test the expectation with a datetime column.
    """
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationDistinctColumnValuesBetween",
        column_name="col1",
        min_value=2,
        max_value=4,
    )
    # Datetime column with 3 distinct values
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


def test_large_dataset_performance():
    """
    Test the expectation with a larger dataset to ensure reasonable performance.
    """
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationDistinctColumnValuesBetween",
        column_name="col1",
        min_value=900,
        max_value=1100,
    )
    # Create a DataFrame with exactly 1000 distinct values
    data_frame = pd.DataFrame({"col1": list(range(1000)) * 5})  # 5000 rows, 1000 distinct values
    result = expectation.validate(data_frame=data_frame)
    assert isinstance(
        result, DataFrameExpectationSuccessMessage
    ), f"Expected DataFrameExpectationSuccessMessage but got: {type(result)}"
