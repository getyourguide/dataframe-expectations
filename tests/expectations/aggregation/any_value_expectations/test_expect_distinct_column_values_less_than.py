import pytest
import pandas as pd
from datetime import datetime

from dataframe_expectations.core.types import DataFrameType
from dataframe_expectations.registry import (
    DataFrameExpectationRegistry,
)
from dataframe_expectations.suite import (
    DataFrameExpectationsSuite,
    DataFrameExpectationsSuiteFailure,
    SuiteExecutionResult,
)
from dataframe_expectations.result_message import (
    DataFrameExpectationFailureMessage,
    DataFrameExpectationSuccessMessage,
)


def create_dataframe(df_type, data, column_name, spark, data_type="long"):
    """Helper function to create pandas or pyspark DataFrame.

    Args:
        df_type: "pandas" or "pyspark"
        data: List of values for the column
        column_name: Name of the column
        spark: Spark session (required for pyspark)
        data_type: Data type for the column - "long", "string", "double", "boolean", "timestamp"
    """
    if df_type == "pandas":
        return pd.DataFrame(data, columns=[column_name])
    else:  # pyspark
        # Use explicit schema for all PySpark DataFrames
        from pyspark.sql.types import (
            StructType,
            StructField,
            LongType,
            StringType,
            DoubleType,
            BooleanType,
            TimestampType,
        )

        type_mapping = {
            "long": LongType(),
            "string": StringType(),
            "double": DoubleType(),
            "boolean": BooleanType(),
            "timestamp": TimestampType(),
        }

        schema = StructType([StructField(column_name, type_mapping[data_type], True)])
        return spark.createDataFrame([(val,) for val in data], schema)


def test_expectation_name():
    """
    Test that the expectation name is correctly returned.
    """
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationDistinctColumnValuesLessThan",
        column_name="col1",
        threshold=5,
    )
    assert expectation.get_expectation_name() == "ExpectationDistinctColumnValuesLessThan", (
        f"Expected 'ExpectationDistinctColumnValuesLessThan' but got: {expectation.get_expectation_name()}"
    )


@pytest.mark.parametrize(
    "df_type, df_data, threshold, expected_result, expected_message, data_type",
    [
        # Basic success - 3 distinct values < 5
        ("pandas", [1, 2, 3, 2, 1], 5, "success", None, "long"),
        ("pyspark", [1, 2, 3, 2, 1], 5, "success", None, "long"),
        # Success with nulls - 3 distinct values [1, 2, None] < 5
        ("pandas", [1, 2, None, 2, 1], 5, "success", None, "long"),
        ("pyspark", [1, 2, None, 2, 1], 5, "success", None, "long"),
        # Empty DataFrame - 0 distinct values < 1
        ("pandas", [], 1, "success", None, "long"),
        ("pyspark", [], 1, "success", None, "long"),
        # Single value - 1 distinct value < 3
        ("pandas", [5, 5, 5, 5, 5], 3, "success", None, "long"),
        ("pyspark", [5, 5, 5, 5, 5], 3, "success", None, "long"),
        # Equal to threshold (should fail) - 3 distinct values, NOT < 3
        (
            "pandas",
            [1, 2, 3, 2, 1],
            3,
            "failure",
            "Column 'col1' has 3 distinct values, expected fewer than 3.",
            "long",
        ),
        (
            "pyspark",
            [1, 2, 3, 2, 1],
            3,
            "failure",
            "Column 'col1' has 3 distinct values, expected fewer than 3.",
            "long",
        ),
        # Above threshold (should fail) - 5 distinct values, NOT < 2
        (
            "pandas",
            [1, 2, 3, 4, 5],
            2,
            "failure",
            "Column 'col1' has 5 distinct values, expected fewer than 2.",
            "long",
        ),
        (
            "pyspark",
            [1, 2, 3, 4, 5],
            2,
            "failure",
            "Column 'col1' has 5 distinct values, expected fewer than 2.",
            "long",
        ),
        # Zero threshold edge case - empty DataFrame, 0 distinct values NOT < 0
        (
            "pandas",
            [],
            0,
            "failure",
            "Column 'col1' has 0 distinct values, expected fewer than 0.",
            "long",
        ),
        (
            "pyspark",
            [],
            0,
            "failure",
            "Column 'col1' has 0 distinct values, expected fewer than 0.",
            "long",
        ),
        # Zero threshold edge case - non-empty DataFrame, 1 distinct value NOT < 0
        (
            "pandas",
            [1, 1, 1],
            0,
            "failure",
            "Column 'col1' has 1 distinct values, expected fewer than 0.",
            "long",
        ),
        (
            "pyspark",
            [1, 1, 1],
            0,
            "failure",
            "Column 'col1' has 1 distinct values, expected fewer than 0.",
            "long",
        ),
        # Exclusive boundary test - 3 distinct values, NOT < 3
        (
            "pandas",
            [1, 2, 3, 1, 2],
            3,
            "failure",
            "Column 'col1' has 3 distinct values, expected fewer than 3.",
            "long",
        ),
        (
            "pyspark",
            [1, 2, 3, 1, 2],
            3,
            "failure",
            "Column 'col1' has 3 distinct values, expected fewer than 3.",
            "long",
        ),
        # Exclusive boundary test - 3 distinct values < 4
        ("pandas", [1, 2, 3, 1, 2], 4, "success", None, "long"),
        ("pyspark", [1, 2, 3, 1, 2], 4, "success", None, "long"),
        # String column with mixed values including None - 4 distinct values < 5
        ("pandas", ["A", "B", "C", "B", "A", None], 5, "success", None, "string"),
        ("pyspark", ["A", "B", "C", "B", "A", None], 5, "success", None, "string"),
        # String case-sensitive - 4 distinct values ["a", "A", "b", "B"] < 5
        ("pandas", ["a", "A", "b", "B", "a", "A"], 5, "success", None, "string"),
        ("pyspark", ["a", "A", "b", "B", "a", "A"], 5, "success", None, "string"),
        # Float column - 3 distinct values < 5
        ("pandas", [1.1, 2.2, 3.3, 2.2, 1.1], 5, "success", None, "double"),
        ("pyspark", [1.1, 2.2, 3.3, 2.2, 1.1], 5, "success", None, "double"),
        # Boolean column - 2 distinct values [True, False] < 3
        ("pandas", [True, False, True, False, True], 3, "success", None, "boolean"),
        ("pyspark", [True, False, True, False, True], 3, "success", None, "boolean"),
        # Boolean column failure - 2 distinct values [True, False], NOT < 2
        (
            "pandas",
            [True, False, True, False, True],
            2,
            "failure",
            "Column 'col1' has 2 distinct values, expected fewer than 2.",
            "boolean",
        ),
        (
            "pyspark",
            [True, False, True, False, True],
            2,
            "failure",
            "Column 'col1' has 2 distinct values, expected fewer than 2.",
            "boolean",
        ),
        # Boolean single value - 1 distinct value [True] < 2
        ("pandas", [True, True, True, True, True], 2, "success", None, "boolean"),
        ("pyspark", [True, True, True, True, True], 2, "success", None, "boolean"),
        # Datetime column - 3 distinct values < 5 (pandas only - uses pd.to_datetime)
        (
            "pandas",
            pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-02", "2023-01-01"]),
            5,
            "success",
            None,
            "timestamp",
        ),
        # Datetime column - 3 distinct values < 5 (pyspark - uses datetime objects)
        (
            "pyspark",
            [
                datetime(2023, 1, 1),
                datetime(2023, 1, 2),
                datetime(2023, 1, 3),
                datetime(2023, 1, 2),
                datetime(2023, 1, 1),
            ],
            5,
            "success",
            None,
            "timestamp",
        ),
        # Mixed data types - 4 distinct values ["text", 42, 3.14, None] < 5 (pandas only)
        ("pandas", ["text", 42, 3.14, None, "text", 42], 5, "success", None, "string"),
        # Categorical data - 3 distinct categories < 5 (pandas only)
        (
            "pandas",
            pd.Categorical(["A", "B", "C", "A", "B", "C", "A"]),
            5,
            "success",
            None,
            "string",
        ),
        # Multiple NaN values counted as one - 3 distinct values [1, 2, None] < 5
        ("pandas", [1, 2, None, None, None, 1, 2], 5, "success", None, "long"),
        ("pyspark", [1, 2, None, None, None, 1, 2], 5, "success", None, "long"),
        # Strings with different whitespace - 4 distinct values < 5
        ("pandas", ["test", " test", "test ", " test ", "test"], 5, "success", None, "string"),
        ("pyspark", ["test", " test", "test ", " test ", "test"], 5, "success", None, "string"),
        # Numeric strings vs numeric values - 2 distinct values < 3 (pandas only - object dtype)
        ("pandas", ["1", 1, "1", 1], 3, "success", None, "object"),
    ],
    ids=[
        "pandas_success",
        "pyspark_success",
        "pandas_success_with_nulls",
        "pyspark_success_with_nulls",
        "pandas_empty",
        "pyspark_empty",
        "pandas_single_value",
        "pyspark_single_value",
        "pandas_equal_to_threshold",
        "pyspark_equal_to_threshold",
        "pandas_above_threshold",
        "pyspark_above_threshold",
        "pandas_zero_threshold_empty",
        "pyspark_zero_threshold_empty",
        "pandas_zero_threshold_non_empty",
        "pyspark_zero_threshold_non_empty",
        "pandas_exclusive_boundary_fail",
        "pyspark_exclusive_boundary_fail",
        "pandas_exclusive_boundary_pass",
        "pyspark_exclusive_boundary_pass",
        "pandas_string_with_nulls",
        "pyspark_string_with_nulls",
        "pandas_string_case_sensitive",
        "pyspark_string_case_sensitive",
        "pandas_float",
        "pyspark_float",
        "pandas_boolean",
        "pyspark_boolean",
        "pandas_boolean_failure",
        "pyspark_boolean_failure",
        "pandas_boolean_single_value",
        "pyspark_boolean_single_value",
        "pandas_datetime",
        "pyspark_datetime",
        "pandas_mixed_data_types",
        "pandas_categorical",
        "pandas_duplicate_nan_handling",
        "pyspark_duplicate_nan_handling",
        "pandas_string_whitespace",
        "pyspark_string_whitespace",
        "pandas_numeric_string_vs_numeric",
    ],
)
def test_expectation_basic_scenarios(
    df_type, df_data, threshold, expected_result, expected_message, data_type, spark
):
    """
    Test the expectation for various scenarios across pandas and PySpark DataFrames.
    Tests both direct expectation validation and suite-based validation.
    Covers: success cases, failures (equal/above threshold), edge cases (empty, zero threshold),
    exclusive boundary, and various data types (strings, floats, booleans, datetimes, categorical, mixed types).
    """
    # Special handling for numeric_string_vs_numeric test - needs object dtype
    if data_type == "object":
        data_frame = pd.DataFrame({"col1": df_data}, dtype=object)
    else:
        data_frame = create_dataframe(df_type, df_data, "col1", spark, data_type)

    # Test 1: Direct expectation validation
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationDistinctColumnValuesLessThan",
        column_name="col1",
        threshold=threshold,
    )

    result = expectation.validate(data_frame=data_frame)

    if expected_result == "success":
        assert str(result) == str(
            DataFrameExpectationSuccessMessage(
                expectation_name="ExpectationDistinctColumnValuesLessThan"
            )
        ), f"Expected success message but got: {result}"
    else:  # failure
        expected_failure_message = DataFrameExpectationFailureMessage(
            expectation_str=str(expectation),
            data_frame_type=str(df_type),
            message=expected_message,
        )
        assert str(result) == str(expected_failure_message), (
            f"Expected failure message but got: {result}"
        )

    # Test 2: Suite-based validation
    expectations_suite = DataFrameExpectationsSuite().expect_distinct_column_values_less_than(
        column_name="col1", threshold=threshold
    )

    if expected_result == "success":
        result = expectations_suite.build().run(data_frame=data_frame)
        assert result is not None, "Expected SuiteExecutionResult"
        assert isinstance(result, SuiteExecutionResult), "Result should be SuiteExecutionResult"
        assert result.success, "Expected all expectations to pass"
        assert result.total_passed == 1, "Expected 1 passed expectation"
        assert result.total_failed == 0, "Expected 0 failed expectations"
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
        expectation_name="ExpectationDistinctColumnValuesLessThan",
        column_name="col1",
        threshold=5,
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
    expectations_suite = DataFrameExpectationsSuite().expect_distinct_column_values_less_than(
        column_name="col1", threshold=5
    )
    with pytest.raises(DataFrameExpectationsSuiteFailure):
        expectations_suite.build().run(data_frame=data_frame)


def test_invalid_parameters():
    """
    Test that appropriate errors are raised for invalid parameters.
    """
    with pytest.raises(ValueError) as context:
        DataFrameExpectationRegistry.get_expectation(
            expectation_name="ExpectationDistinctColumnValuesLessThan",
            column_name="col1",
            threshold=-1,
        )
    assert "threshold must be non-negative" in str(context.value), (
        f"Expected 'threshold must be non-negative' in error message: {str(context.value)}"
    )


def test_large_dataset_performance():
    """
    Test the expectation with a larger dataset to ensure reasonable performance.
    """
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationDistinctColumnValuesLessThan",
        column_name="col1",
        threshold=1001,
    )
    # Create a DataFrame with exactly 1000 distinct values (< 1001)
    data_frame = pd.DataFrame({"col1": list(range(1000)) * 5})  # 5000 rows, 1000 distinct values
    result = expectation.validate(data_frame=data_frame)
    assert isinstance(result, DataFrameExpectationSuccessMessage), (
        f"Expected DataFrameExpectationSuccessMessage but got: {type(result)}"
    )
