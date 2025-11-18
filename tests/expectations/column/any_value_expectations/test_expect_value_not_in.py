import pytest
import pandas as pd
from datetime import datetime, timezone

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
        return pd.DataFrame({column_name: data})
    else:  # pyspark
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
    """Test that the expectation name is correctly returned."""
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationValueNotIn",
        column_name="col1",
        values=[1, 2, 3],
    )
    assert expectation.get_expectation_name() == "ExpectationValueNotIn", (
        f"Expected 'ExpectationValueNotIn' but got: {expectation.get_expectation_name()}"
    )


@pytest.mark.parametrize(
    "df_type, data, values, should_succeed, expected_violations, expected_message, data_type",
    [
        # Basic integer scenarios - success (values NOT in [1,2,3])
        ("pandas", [4, 5, 6, 7], [1, 2, 3], True, None, None, "long"),
        ("pyspark", [4, 5, 6, 7], [1, 2, 3], True, None, None, "long"),
        ("pandas", [10, 20, 30], [1, 2, 3], True, None, None, "long"),
        ("pyspark", [10, 20, 30], [1, 2, 3], True, None, None, "long"),
        # Integer scenarios - violations (values in [1,2,3])
        (
            "pandas",
            [1, 2, 4, 5],
            [1, 2, 3],
            False,
            [1, 2],
            "Found 2 row(s) where 'col1' is in [1, 2, 3].",
            "long",
        ),
        (
            "pyspark",
            [1, 2, 4, 5],
            [1, 2, 3],
            False,
            [1, 2],
            "Found 2 row(s) where 'col1' is in [1, 2, 3].",
            "long",
        ),
        (
            "pandas",
            [1, 2, 3],
            [1, 2, 3],
            False,
            [1, 2, 3],
            "Found 3 row(s) where 'col1' is in [1, 2, 3].",
            "long",
        ),
        (
            "pyspark",
            [1, 2, 3],
            [1, 2, 3],
            False,
            [1, 2, 3],
            "Found 3 row(s) where 'col1' is in [1, 2, 3].",
            "long",
        ),
        # String data type scenarios - success
        (
            "pandas",
            ["orange", "grape", "melon"],
            ["apple", "banana"],
            True,
            None,
            None,
            "string",
        ),
        (
            "pyspark",
            ["orange", "grape", "melon"],
            ["apple", "banana"],
            True,
            None,
            None,
            "string",
        ),
        # String scenarios - violations
        (
            "pandas",
            ["apple", "orange", "banana"],
            ["apple", "banana"],
            False,
            ["apple", "banana"],
            "Found 2 row(s) where 'col1' is in ['apple', 'banana'].",
            "string",
        ),
        (
            "pyspark",
            ["apple", "orange", "banana"],
            ["apple", "banana"],
            False,
            ["apple", "banana"],
            "Found 2 row(s) where 'col1' is in ['apple', 'banana'].",
            "string",
        ),
        # String case sensitivity - success (case matters)
        ("pandas", ["Apple", "APPLE", "aPpLe"], ["apple"], True, None, None, "string"),
        ("pyspark", ["Apple", "APPLE", "aPpLe"], ["apple"], True, None, None, "string"),
        # String case sensitivity - violations (exact match)
        (
            "pandas",
            ["apple", "Apple", "banana"],
            ["apple", "banana"],
            False,
            ["apple", "banana"],
            "Found 2 row(s) where 'col1' is in ['apple', 'banana'].",
            "string",
        ),
        (
            "pyspark",
            ["apple", "Apple", "banana"],
            ["apple", "banana"],
            False,
            ["apple", "banana"],
            "Found 2 row(s) where 'col1' is in ['apple', 'banana'].",
            "string",
        ),
        # Float/Double data type scenarios - success
        ("pandas", [5.5, 6.5, 7.5], [1.5, 2.5, 3.5], True, None, None, "double"),
        ("pyspark", [5.5, 6.5, 7.5], [1.5, 2.5, 3.5], True, None, None, "double"),
        # Float scenarios - violations
        (
            "pandas",
            [1.5, 2.5, 5.5],
            [1.5, 2.5],
            False,
            [1.5, 2.5],
            "Found 2 row(s) where 'col1' is in [1.5, 2.5].",
            "double",
        ),
        (
            "pyspark",
            [1.5, 2.5, 5.5],
            [1.5, 2.5],
            False,
            [1.5, 2.5],
            "Found 2 row(s) where 'col1' is in [1.5, 2.5].",
            "double",
        ),
        # Float precision edge cases - success
        ("pandas", [1.1, 1.2, 1.3], [1.0, 2.0], True, None, None, "double"),
        ("pyspark", [1.1, 1.2, 1.3], [1.0, 2.0], True, None, None, "double"),
        # Float precision - violations
        (
            "pandas",
            [1.0, 1.5, 2.0],
            [1.0, 2.0],
            False,
            [1.0, 2.0],
            "Found 2 row(s) where 'col1' is in [1.0, 2.0].",
            "double",
        ),
        (
            "pyspark",
            [1.0, 1.5, 2.0],
            [1.0, 2.0],
            False,
            [1.0, 2.0],
            "Found 2 row(s) where 'col1' is in [1.0, 2.0].",
            "double",
        ),
        # Boolean data type scenarios - success
        ("pandas", [False, False, False], [True], True, None, None, "boolean"),
        ("pyspark", [False, False, False], [True], True, None, None, "boolean"),
        ("pandas", [True, True, True], [False], True, None, None, "boolean"),
        ("pyspark", [True, True, True], [False], True, None, None, "boolean"),
        # Boolean scenarios - violations
        (
            "pandas",
            [True, False, True],
            [True],
            False,
            [True, True],
            "Found 2 row(s) where 'col1' is in [True].",
            "boolean",
        ),
        (
            "pyspark",
            [True, False, True],
            [True],
            False,
            [True, True],
            "Found 2 row(s) where 'col1' is in [True].",
            "boolean",
        ),
        (
            "pandas",
            [True, False, False],
            [True, False],
            False,
            [True, False, False],
            "Found 3 row(s) where 'col1' is in [True, False].",
            "boolean",
        ),
        (
            "pyspark",
            [True, False, False],
            [True, False],
            False,
            [True, False, False],
            "Found 3 row(s) where 'col1' is in [True, False].",
            "boolean",
        ),
        # Timestamp/Datetime scenarios - success
        (
            "pandas",
            [datetime(2023, 1, 5), datetime(2023, 1, 6), datetime(2023, 1, 7)],
            [datetime(2023, 1, 1), datetime(2023, 1, 2)],
            True,
            None,
            None,
            "timestamp",
        ),
        (
            "pyspark",
            [datetime(2023, 1, 5), datetime(2023, 1, 6), datetime(2023, 1, 7)],
            [datetime(2023, 1, 1), datetime(2023, 1, 2)],
            True,
            None,
            None,
            "timestamp",
        ),
        # Timestamp scenarios - violations
        (
            "pandas",
            [datetime(2023, 1, 1), datetime(2023, 1, 5), datetime(2023, 1, 2)],
            [datetime(2023, 1, 1), datetime(2023, 1, 2)],
            False,
            [datetime(2023, 1, 1), datetime(2023, 1, 2)],
            "Found 2 row(s) where 'col1' is in [datetime.datetime(2023, 1, 1, 0, 0), datetime.datetime(2023, 1, 2, 0, 0)].",
            "timestamp",
        ),
        (
            "pyspark",
            [datetime(2023, 1, 1), datetime(2023, 1, 5), datetime(2023, 1, 2)],
            [datetime(2023, 1, 1), datetime(2023, 1, 2)],
            False,
            [datetime(2023, 1, 1), datetime(2023, 1, 2)],
            "Found 2 row(s) where 'col1' is in [datetime.datetime(2023, 1, 1, 0, 0), datetime.datetime(2023, 1, 2, 0, 0)].",
            "timestamp",
        ),
        # Datetime with timezone - success
        (
            "pandas",
            [
                datetime(2023, 1, 5, 12, 0, 0, tzinfo=timezone.utc),
                datetime(2023, 1, 6, 12, 0, 0, tzinfo=timezone.utc),
            ],
            [datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)],
            True,
            None,
            None,
            "timestamp",
        ),
        (
            "pyspark",
            [
                datetime(2023, 1, 5, 12, 0, 0, tzinfo=timezone.utc),
                datetime(2023, 1, 6, 12, 0, 0, tzinfo=timezone.utc),
            ],
            [datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)],
            True,
            None,
            None,
            "timestamp",
        ),
        # Empty string scenarios - success
        ("pandas", ["text", "data", "value"], [""], True, None, None, "string"),
        ("pyspark", ["text", "data", "value"], [""], True, None, None, "string"),
        # Empty string - violations
        (
            "pandas",
            ["", "text", ""],
            [""],
            False,
            ["", ""],
            "Found 2 row(s) where 'col1' is in [''].",
            "string",
        ),
        (
            "pyspark",
            ["", "text", ""],
            [""],
            False,
            ["", ""],
            "Found 2 row(s) where 'col1' is in [''].",
            "string",
        ),
        # Whitespace in strings - success
        ("pandas", [" test", "test ", " test "], ["test"], True, None, None, "string"),
        ("pyspark", [" test", "test ", " test "], ["test"], True, None, None, "string"),
        # Zero value scenarios - success
        ("pandas", [1, 2, 3], [0], True, None, None, "long"),
        ("pyspark", [1, 2, 3], [0], True, None, None, "long"),
        ("pandas", [1.5, 2.5, 3.5], [0.0], True, None, None, "double"),
        ("pyspark", [1.5, 2.5, 3.5], [0.0], True, None, None, "double"),
        # Zero value - violations
        (
            "pandas",
            [0, 1, 0],
            [0],
            False,
            [0, 0],
            "Found 2 row(s) where 'col1' is in [0].",
            "long",
        ),
        (
            "pyspark",
            [0, 1, 0],
            [0],
            False,
            [0, 0],
            "Found 2 row(s) where 'col1' is in [0].",
            "long",
        ),
        # Negative numbers - success
        ("pandas", [1, 2, 3], [-5, -10], True, None, None, "long"),
        ("pyspark", [1, 2, 3], [-5, -10], True, None, None, "long"),
        ("pandas", [1.5, 2.5, 3.5], [-3.14], True, None, None, "double"),
        ("pyspark", [1.5, 2.5, 3.5], [-3.14], True, None, None, "double"),
        # Negative numbers - violations
        (
            "pandas",
            [-5, 1, -5],
            [-5],
            False,
            [-5, -5],
            "Found 2 row(s) where 'col1' is in [-5].",
            "long",
        ),
        (
            "pyspark",
            [-5, 1, -5],
            [-5],
            False,
            [-5, -5],
            "Found 2 row(s) where 'col1' is in [-5].",
            "long",
        ),
        # Large numbers - success
        ("pandas", [999999, 1000001], [1000000], True, None, None, "long"),
        ("pyspark", [999999, 1000001], [1000000], True, None, None, "long"),
        # Large numbers - violations
        (
            "pandas",
            [1000000, 999999, 1000000],
            [1000000],
            False,
            [1000000, 1000000],
            "Found 2 row(s) where 'col1' is in [1000000].",
            "long",
        ),
        (
            "pyspark",
            [1000000, 999999, 1000000],
            [1000000],
            False,
            [1000000, 1000000],
            "Found 2 row(s) where 'col1' is in [1000000].",
            "long",
        ),
        # Single value in list - success
        ("pandas", [10, 20, 30], [5], True, None, None, "long"),
        ("pyspark", [10, 20, 30], [5], True, None, None, "long"),
        # Multiple values with partial match
        (
            "pandas",
            [1, 5, 6],
            [1, 2, 3, 4],
            False,
            [1],
            "Found 1 row(s) where 'col1' is in [1, 2, 3, 4].",
            "long",
        ),
        (
            "pyspark",
            [1, 5, 6],
            [1, 2, 3, 4],
            False,
            [1],
            "Found 1 row(s) where 'col1' is in [1, 2, 3, 4].",
            "long",
        ),
    ],
    ids=[
        "pandas_int_basic_success",
        "pyspark_int_basic_success",
        "pandas_int_success_different_range",
        "pyspark_int_success_different_range",
        "pandas_int_violations_two",
        "pyspark_int_violations_two",
        "pandas_int_violations_all",
        "pyspark_int_violations_all",
        "pandas_string_success",
        "pyspark_string_success",
        "pandas_string_violations",
        "pyspark_string_violations",
        "pandas_string_case_sensitive_success",
        "pyspark_string_case_sensitive_success",
        "pandas_string_case_sensitive_violations",
        "pyspark_string_case_sensitive_violations",
        "pandas_double_success",
        "pyspark_double_success",
        "pandas_double_violations",
        "pyspark_double_violations",
        "pandas_double_precision_success",
        "pyspark_double_precision_success",
        "pandas_double_precision_violations",
        "pyspark_double_precision_violations",
        "pandas_boolean_false_success",
        "pyspark_boolean_false_success",
        "pandas_boolean_true_success",
        "pyspark_boolean_true_success",
        "pandas_boolean_true_violations",
        "pyspark_boolean_true_violations",
        "pandas_boolean_both_violations",
        "pyspark_boolean_both_violations",
        "pandas_timestamp_success",
        "pyspark_timestamp_success",
        "pandas_timestamp_violations",
        "pyspark_timestamp_violations",
        "pandas_timestamp_with_timezone_success",
        "pyspark_timestamp_with_timezone_success",
        "pandas_empty_string_success",
        "pyspark_empty_string_success",
        "pandas_empty_string_violations",
        "pyspark_empty_string_violations",
        "pandas_string_whitespace_success",
        "pyspark_string_whitespace_success",
        "pandas_zero_int_success",
        "pyspark_zero_int_success",
        "pandas_zero_double_success",
        "pyspark_zero_double_success",
        "pandas_zero_int_violations",
        "pyspark_zero_int_violations",
        "pandas_negative_int_success",
        "pyspark_negative_int_success",
        "pandas_negative_double_success",
        "pyspark_negative_double_success",
        "pandas_negative_int_violations",
        "pyspark_negative_int_violations",
        "pandas_large_numbers_success",
        "pyspark_large_numbers_success",
        "pandas_large_numbers_violations",
        "pyspark_large_numbers_violations",
        "pandas_single_value_list_success",
        "pyspark_single_value_list_success",
        "pandas_multiple_values_partial_match",
        "pyspark_multiple_values_partial_match",
    ],
)
def test_expectation_basic_scenarios(
    df_type,
    data,
    values,
    should_succeed,
    expected_violations,
    expected_message,
    data_type,
    spark,
):
    """Test basic expectation scenarios for both pandas and PySpark DataFrames.

    Tests various data types including:
    - Integers (long): positive, negative, zero, large numbers, single/multiple values
    - Strings: case sensitivity, empty strings, whitespace
    - Floats (double): precision, zero values
    - Booleans: True/False combinations
    - Timestamps: with and without timezone
    - Edge cases: single value lists

    Note: ExpectationValueNotIn checks that values are NOT in the target list.
    Success = all values outside target list, Violations = values that are in target list.
    """
    df = create_dataframe(df_type, data, "col1", spark, data_type)

    # Test through registry
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationValueNotIn",
        column_name="col1",
        values=values,
    )
    result = expectation.validate(data_frame=df)

    if should_succeed:
        assert isinstance(result, DataFrameExpectationSuccessMessage), (
            f"Expected success but got: {result}"
        )
    else:
        assert isinstance(result, DataFrameExpectationFailureMessage), (
            f"Expected failure but got: {result}"
        )
        assert expected_message in str(result), (
            f"Expected message '{expected_message}' in result: {result}"
        )

        # Verify violations if present
        if expected_violations is not None:
            expected_violations_df = create_dataframe(
                df_type, expected_violations, "col1", spark, data_type
            )
            expected_failure = DataFrameExpectationFailureMessage(
                expectation_str=str(expectation),
                data_frame_type=str(df_type),
                violations_data_frame=expected_violations_df,
                message=expected_message,
                limit_violations=5,
            )
            assert str(result) == str(expected_failure), (
                f"Expected failure details don't match. Got: {result}"
            )

    # Test through suite
    suite = DataFrameExpectationsSuite().expect_value_not_in(column_name="col1", values=values)

    if should_succeed:
        suite_result = suite.build().run(data_frame=df)
        assert suite_result is None, f"Suite test expected None but got: {suite_result}"
    else:
        with pytest.raises(DataFrameExpectationsSuiteFailure):
            suite.build().run(data_frame=df)


@pytest.mark.parametrize(
    "df_type",
    ["pandas", "pyspark"],
    ids=["pandas", "pyspark"],
)
def test_column_missing_error(df_type, spark):
    """Test missing column error for both pandas and PySpark DataFrames."""
    if df_type == "pandas":
        df = pd.DataFrame({"col1": [4, 5, 6]})
    else:
        df = spark.createDataFrame([(4,), (5,), (6,)], ["col1"])

    expected_message = "Column 'nonexistent_col' does not exist in the DataFrame."

    # Test through registry
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationValueNotIn",
        column_name="nonexistent_col",
        values=[1, 2, 3],
    )
    result = expectation.validate(data_frame=df)
    expected_failure = DataFrameExpectationFailureMessage(
        expectation_str=str(expectation),
        data_frame_type=str(df_type),
        message=expected_message,
    )
    assert str(result) == str(expected_failure)

    # Test through suite
    suite = DataFrameExpectationsSuite().expect_value_not_in(
        column_name="nonexistent_col", values=[1, 2, 3]
    )
    with pytest.raises(DataFrameExpectationsSuiteFailure):
        suite.build().run(data_frame=df)


def test_large_dataset_performance():
    """Test the expectation with a larger dataset to ensure reasonable performance."""

    # Create a larger dataset with 10,000 rows, all NOT in [1, 2, 3] (using values 4+)
    large_data = list(range(4, 10004))  # 10,000 values from 4 to 10003
    data_frame = pd.DataFrame({"col1": large_data})

    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationValueNotIn",
        column_name="col1",
        values=[1, 2, 3],
    )

    result = expectation.validate(data_frame=data_frame)
    assert isinstance(result, DataFrameExpectationSuccessMessage), (
        f"Large dataset test failed: expected success but got {type(result)}"
    )
