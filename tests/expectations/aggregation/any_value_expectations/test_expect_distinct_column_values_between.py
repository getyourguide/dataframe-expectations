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


def create_dataframe(df_type, data, column_name, spark):
    """Helper function to create pandas or pyspark DataFrame."""
    if df_type == "pandas":
        return pd.DataFrame(data, columns=[column_name])
    else:  # pyspark
        # Use explicit schema for all PySpark DataFrames (required for empty DataFrames)
        from pyspark.sql.types import StructType, StructField, LongType

        schema = StructType([StructField(column_name, LongType(), True)])
        return spark.createDataFrame([(val,) for val in data], schema)


def get_df_type_enum(df_type):
    """Get DataFrameType enum value."""
    return DataFrameType.PANDAS if df_type == "pandas" else DataFrameType.PYSPARK


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
    assert expectation.get_expectation_name() == "ExpectationDistinctColumnValuesBetween", (
        f"Expected 'ExpectationDistinctColumnValuesBetween' but got: {expectation.get_expectation_name()}"
    )


@pytest.mark.parametrize(
    "df_type, df_data, min_value, max_value, expected_result, expected_message",
    [
        # Basic success - 3 distinct values within range [2, 5]
        ("pandas", [1, 2, 3, 2, 1], 2, 5, "success", None),
        ("pyspark", [1, 2, 3, 2, 1], 2, 5, "success", None),
        # Success with nulls - 3 distinct values [1, 2, None] within range [3, 4]
        ("pandas", [1, 2, None, 2, 1], 3, 4, "success", None),
        ("pyspark", [1, 2, None, 2, 1], 3, 4, "success", None),
        # Exact minimum boundary - 3 distinct values at min boundary [3, 5]
        ("pandas", [1, 2, 3, 2, 1], 3, 5, "success", None),
        ("pyspark", [1, 2, 3, 2, 1], 3, 5, "success", None),
        # Exact maximum boundary - 5 distinct values at max boundary [3, 5]
        ("pandas", [1, 2, 3, 4, 5, 1], 3, 5, "success", None),
        ("pyspark", [1, 2, 3, 4, 5, 1], 3, 5, "success", None),
        # Edge case: zero range (min == max) - success with exact match
        ("pandas", [1, 2, 3, 2, 1], 3, 3, "success", None),
        ("pyspark", [1, 2, 3, 2, 1], 3, 3, "success", None),
        # Edge case: zero range (min == max) - failure when not exact
        (
            "pandas",
            [1, 2, 1, 2, 1],
            3,
            3,
            "failure",
            "Column 'col1' has 2 distinct values, expected between 3 and 3.",
        ),
        (
            "pyspark",
            [1, 2, 1, 2, 1],
            3,
            3,
            "failure",
            "Column 'col1' has 2 distinct values, expected between 3 and 3.",
        ),
        # Edge case: empty DataFrame - 0 distinct values within range [0, 5]
        ("pandas", [], 0, 5, "success", None),
        ("pyspark", [], 0, 5, "success", None),
        # Edge case: single distinct value - 1 distinct value within range [1, 1]
        ("pandas", [1, 1, 1, 1, 1], 1, 1, "success", None),
        ("pyspark", [1, 1, 1, 1, 1], 1, 1, "success", None),
        # Too few distinct values - 2 distinct, expecting [4, 6]
        (
            "pandas",
            [1, 2, 1, 2, 1],
            4,
            6,
            "failure",
            "Column 'col1' has 2 distinct values, expected between 4 and 6.",
        ),
        (
            "pyspark",
            [1, 2, 1, 2, 1],
            4,
            6,
            "failure",
            "Column 'col1' has 2 distinct values, expected between 4 and 6.",
        ),
        # Too many distinct values - 5 distinct, expecting [2, 3]
        (
            "pandas",
            [1, 2, 3, 4, 5],
            2,
            3,
            "failure",
            "Column 'col1' has 5 distinct values, expected between 2 and 3.",
        ),
        (
            "pyspark",
            [1, 2, 3, 4, 5],
            2,
            3,
            "failure",
            "Column 'col1' has 5 distinct values, expected between 2 and 3.",
        ),
    ],
    ids=[
        "pandas_success",
        "pyspark_success",
        "pandas_success_with_nulls",
        "pyspark_success_with_nulls",
        "pandas_exact_min_boundary",
        "pyspark_exact_min_boundary",
        "pandas_exact_max_boundary",
        "pyspark_exact_max_boundary",
        "pandas_zero_range_success",
        "pyspark_zero_range_success",
        "pandas_zero_range_failure",
        "pyspark_zero_range_failure",
        "pandas_empty_dataframe",
        "pyspark_empty_dataframe",
        "pandas_single_value",
        "pyspark_single_value",
        "pandas_too_few",
        "pyspark_too_few",
        "pandas_too_many",
        "pyspark_too_many",
    ],
)
def test_expectation_basic_scenarios(
    df_type, df_data, min_value, max_value, expected_result, expected_message, spark
):
    """
    Test the expectation for various scenarios across pandas and PySpark DataFrames.
    Tests both direct expectation validation and suite-based validation.
    Covers: success cases, success with nulls, exact boundaries, edge cases (zero range, empty, single value),
    too few values, and too many values.
    """
    data_frame = create_dataframe(df_type, df_data, "col1", spark)

    # Test 1: Direct expectation validation
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationDistinctColumnValuesBetween",
        column_name="col1",
        min_value=min_value,
        max_value=max_value,
    )

    result = expectation.validate(data_frame=data_frame)

    if expected_result == "success":
        assert str(result) == str(
            DataFrameExpectationSuccessMessage(
                expectation_name="ExpectationDistinctColumnValuesBetween"
            )
        ), f"Expected success message but got: {result}"
    else:  # failure
        expected_failure_message = DataFrameExpectationFailureMessage(
            expectation_str=str(expectation),
            data_frame_type=get_df_type_enum(df_type),
            message=expected_message,
        )
        assert str(result) == str(expected_failure_message), (
            f"Expected failure message but got: {result}"
        )

    # Test 2: Suite-based validation
    expectations_suite = DataFrameExpectationsSuite().expect_distinct_column_values_between(
        column_name="col1", min_value=min_value, max_value=max_value
    )

    if expected_result == "success":
        result = expectations_suite.build().run(data_frame=data_frame)
        assert result is None, "Expected no exceptions to be raised from suite"
    else:  # failure
        with pytest.raises(DataFrameExpectationsSuiteFailure):
            expectations_suite.build().run(data_frame=data_frame)


@pytest.mark.parametrize(
    "df_type",
    ["pandas", "pyspark"],
    ids=["pandas", "pyspark"],
)
def test_column_missing_error(df_type, spark):
    """
    Test that an error is raised when the specified column is missing in both pandas and PySpark.
    Tests both direct expectation validation and suite-based validation.
    """
    # Test 1: Direct expectation validation
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationDistinctColumnValuesBetween",
        column_name="col1",
        min_value=2,
        max_value=5,
    )

    if df_type == "pandas":
        data_frame = pd.DataFrame({"col2": [1, 2, 3, 4, 5]})
        df_type_enum = DataFrameType.PANDAS
    else:  # pyspark
        data_frame = spark.createDataFrame([(1,), (2,), (3,), (4,), (5,)], ["col2"])
        df_type_enum = DataFrameType.PYSPARK

    result = expectation.validate(data_frame=data_frame)
    expected_failure_message = DataFrameExpectationFailureMessage(
        expectation_str=str(expectation),
        data_frame_type=df_type_enum,
        message="Column 'col1' does not exist in the DataFrame.",
    )
    assert str(result) == str(expected_failure_message), (
        f"Expected failure message but got: {result}"
    )

    # Test 2: Suite-based validation
    expectations_suite = DataFrameExpectationsSuite().expect_distinct_column_values_between(
        column_name="col1", min_value=2, max_value=5
    )
    with pytest.raises(DataFrameExpectationsSuiteFailure):
        expectations_suite.build().run(data_frame=data_frame)


@pytest.mark.parametrize(
    "min_value, max_value, expected_error_message",
    [
        (-1, 5, "min_value must be non-negative"),
        (2, -1, "max_value must be non-negative"),
        (5, 2, "min_value (5) must be <= max_value (2)"),
    ],
    ids=["negative_min_value", "negative_max_value", "min_greater_than_max"],
)
def test_invalid_parameters(min_value, max_value, expected_error_message):
    """
    Test that appropriate errors are raised for invalid parameters.
    """
    with pytest.raises(ValueError) as context:
        DataFrameExpectationRegistry.get_expectation(
            expectation_name="ExpectationDistinctColumnValuesBetween",
            column_name="col1",
            min_value=min_value,
            max_value=max_value,
        )
    assert expected_error_message in str(context.value), (
        f"Expected '{expected_error_message}' in error message: {str(context.value)}"
    )


@pytest.mark.parametrize(
    "df_data, min_value, max_value, expected_distinct, description",
    [
        # String column with mixed values including None
        (["A", "B", "C", "B", "A", None], 3, 5, 4, "string_with_nulls"),
        # Float column
        ([1.1, 2.2, 3.3, 2.2, 1.1], 2, 4, 3, "float"),
        # Boolean column
        ([True, False, True, False, True], 2, 2, 2, "boolean"),
        # Datetime column
        (
            pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-02", "2023-01-01"]),
            2,
            4,
            3,
            "datetime",
        ),
    ],
    ids=["string_with_nulls", "float", "boolean", "datetime"],
)
def test_data_type_validation(df_data, min_value, max_value, expected_distinct, description):
    """
    Test the expectation with various data types.
    Verifies that distinct value counting works correctly across different column types.
    """
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationDistinctColumnValuesBetween",
        column_name="col1",
        min_value=min_value,
        max_value=max_value,
    )

    data_frame = pd.DataFrame({"col1": df_data})
    result = expectation.validate(data_frame=data_frame)

    assert isinstance(result, DataFrameExpectationSuccessMessage), (
        f"Expected DataFrameExpectationSuccessMessage for {description} data type but got: {type(result)}"
    )


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
    assert isinstance(result, DataFrameExpectationSuccessMessage), (
        f"Expected DataFrameExpectationSuccessMessage but got: {type(result)}"
    )
