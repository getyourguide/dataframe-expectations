import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timezone

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
        expectation_name="ExpectationValueNotNull",
        column_name="col1",
    )
    assert expectation.get_expectation_name() == "ExpectationValueNotNull", (
        f"Expected 'ExpectationValueNotNull' but got: {expectation.get_expectation_name()}"
    )


@pytest.mark.parametrize(
    "df_type, data, should_succeed, expected_violations, expected_message, data_type",
    [
        # Basic integer scenarios - success (no nulls)
        ("pandas", [1, 2, 3], True, None, None, "long"),
        ("pyspark", [1, 2, 3], True, None, None, "long"),
        ("pandas", [10, 20, 30, 40], True, None, None, "long"),
        ("pyspark", [10, 20, 30, 40], True, None, None, "long"),
        # Integer scenarios - violations (with None)
        (
            "pandas",
            [1, None, np.nan],
            False,
            [None, np.nan],
            "Found 2 row(s) where 'col1' is null.",
            "long",
        ),
        (
            "pyspark",
            [1, None, None],
            False,
            [None, None],
            "Found 2 row(s) where 'col1' is null.",
            "long",
        ),
        # All nulls scenario
        (
            "pandas",
            [None, None, None],
            False,
            [None, None, None],
            "Found 3 row(s) where 'col1' is null.",
            "long",
        ),
        (
            "pyspark",
            [None, None, None],
            False,
            [None, None, None],
            "Found 3 row(s) where 'col1' is null.",
            "long",
        ),
        # Single null scenario
        (
            "pandas",
            [1, 2, None],
            False,
            [None],
            "Found 1 row(s) where 'col1' is null.",
            "long",
        ),
        (
            "pyspark",
            [1, 2, None],
            False,
            [None],
            "Found 1 row(s) where 'col1' is null.",
            "long",
        ),
        # String data type scenarios - success (no nulls)
        ("pandas", ["apple", "banana", "cherry"], True, None, None, "string"),
        ("pyspark", ["apple", "banana", "cherry"], True, None, None, "string"),
        ("pandas", ["test", "data", "values"], True, None, None, "string"),
        ("pyspark", ["test", "data", "values"], True, None, None, "string"),
        # String scenarios - violations (with None)
        (
            "pandas",
            ["apple", None, "banana"],
            False,
            [None],
            "Found 1 row(s) where 'col1' is null.",
            "string",
        ),
        (
            "pyspark",
            ["apple", None, "banana"],
            False,
            [None],
            "Found 1 row(s) where 'col1' is null.",
            "string",
        ),
        # Empty string is NOT null - should succeed
        ("pandas", ["", "test", "data"], True, None, None, "string"),
        ("pyspark", ["", "test", "data"], True, None, None, "string"),
        # Whitespace is NOT null - should succeed
        ("pandas", [" ", "  ", "   "], True, None, None, "string"),
        ("pyspark", [" ", "  ", "   "], True, None, None, "string"),
        # Mixed empty strings and nulls - only nulls are violations
        (
            "pandas",
            ["", None, "test"],
            False,
            [None],
            "Found 1 row(s) where 'col1' is null.",
            "string",
        ),
        (
            "pyspark",
            ["", None, "test"],
            False,
            [None],
            "Found 1 row(s) where 'col1' is null.",
            "string",
        ),
        # Float/Double data type scenarios - success (no nulls)
        ("pandas", [1.5, 2.5, 3.5], True, None, None, "double"),
        ("pyspark", [1.5, 2.5, 3.5], True, None, None, "double"),
        ("pandas", [0.0, 1.1, 2.2], True, None, None, "double"),
        ("pyspark", [0.0, 1.1, 2.2], True, None, None, "double"),
        # Float scenarios - violations (with None/NaN)
        (
            "pandas",
            [1.5, None, 2.5],
            False,
            [None],
            "Found 1 row(s) where 'col1' is null.",
            "double",
        ),
        (
            "pyspark",
            [1.5, None, 2.5],
            False,
            [None],
            "Found 1 row(s) where 'col1' is null.",
            "double",
        ),
        (
            "pandas",
            [1.5, np.nan, 2.5],
            False,
            [np.nan],
            "Found 1 row(s) where 'col1' is null.",
            "double",
        ),
        # Zero is NOT null - should succeed
        ("pandas", [0.0, 0.0, 0.0], True, None, None, "double"),
        ("pyspark", [0.0, 0.0, 0.0], True, None, None, "double"),
        ("pandas", [0, 0, 0], True, None, None, "long"),
        ("pyspark", [0, 0, 0], True, None, None, "long"),
        # Negative numbers are NOT null - should succeed
        ("pandas", [-1, -2, -3], True, None, None, "long"),
        ("pyspark", [-1, -2, -3], True, None, None, "long"),
        ("pandas", [-1.5, -2.5, -3.5], True, None, None, "double"),
        ("pyspark", [-1.5, -2.5, -3.5], True, None, None, "double"),
        # Boolean data type scenarios - success (no nulls)
        ("pandas", [True, False, True], True, None, None, "boolean"),
        ("pyspark", [True, False, True], True, None, None, "boolean"),
        ("pandas", [False, False, False], True, None, None, "boolean"),
        ("pyspark", [False, False, False], True, None, None, "boolean"),
        # Boolean scenarios - violations (with None)
        (
            "pandas",
            [True, None, False],
            False,
            [None],
            "Found 1 row(s) where 'col1' is null.",
            "boolean",
        ),
        (
            "pyspark",
            [True, None, False],
            False,
            [None],
            "Found 1 row(s) where 'col1' is null.",
            "boolean",
        ),
        # False is NOT null - should succeed
        ("pandas", [False], True, None, None, "boolean"),
        ("pyspark", [False], True, None, None, "boolean"),
        # Timestamp/Datetime scenarios - success (no nulls)
        (
            "pandas",
            [datetime(2023, 1, 1), datetime(2023, 1, 2), datetime(2023, 1, 3)],
            True,
            None,
            None,
            "timestamp",
        ),
        (
            "pyspark",
            [datetime(2023, 1, 1), datetime(2023, 1, 2), datetime(2023, 1, 3)],
            True,
            None,
            None,
            "timestamp",
        ),
        # Timestamp scenarios - violations (with None)
        (
            "pandas",
            [datetime(2023, 1, 1), None, datetime(2023, 1, 3)],
            False,
            [None],
            "Found 1 row(s) where 'col1' is null.",
            "timestamp",
        ),
        (
            "pyspark",
            [datetime(2023, 1, 1), None, datetime(2023, 1, 3)],
            False,
            [None],
            "Found 1 row(s) where 'col1' is null.",
            "timestamp",
        ),
        # Datetime with timezone - success (no nulls)
        (
            "pandas",
            [
                datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
                datetime(2023, 1, 2, 12, 0, 0, tzinfo=timezone.utc),
            ],
            True,
            None,
            None,
            "timestamp",
        ),
        (
            "pyspark",
            [
                datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
                datetime(2023, 1, 2, 12, 0, 0, tzinfo=timezone.utc),
            ],
            True,
            None,
            None,
            "timestamp",
        ),
        # Multiple nulls in different positions
        (
            "pandas",
            [1, None, 3, None, 5],
            False,
            [None, None],
            "Found 2 row(s) where 'col1' is null.",
            "long",
        ),
        (
            "pyspark",
            [1, None, 3, None, 5],
            False,
            [None, None],
            "Found 2 row(s) where 'col1' is null.",
            "long",
        ),
        # Large numbers - success (not null)
        ("pandas", [1000000, 2000000, 3000000], True, None, None, "long"),
        ("pyspark", [1000000, 2000000, 3000000], True, None, None, "long"),
        # Single value - success
        ("pandas", [42], True, None, None, "long"),
        ("pyspark", [42], True, None, None, "long"),
        # Single null
        (
            "pandas",
            [None],
            False,
            [None],
            "Found 1 row(s) where 'col1' is null.",
            "long",
        ),
        (
            "pyspark",
            [None],
            False,
            [None],
            "Found 1 row(s) where 'col1' is null.",
            "long",
        ),
    ],
    ids=[
        "pandas_int_basic_success",
        "pyspark_int_basic_success",
        "pandas_int_multi_value_success",
        "pyspark_int_multi_value_success",
        "pandas_int_with_none_nan_violations",
        "pyspark_int_with_none_violations",
        "pandas_all_nulls_violations",
        "pyspark_all_nulls_violations",
        "pandas_single_null_violation",
        "pyspark_single_null_violation",
        "pandas_string_basic_success",
        "pyspark_string_basic_success",
        "pandas_string_multi_value_success",
        "pyspark_string_multi_value_success",
        "pandas_string_with_none_violation",
        "pyspark_string_with_none_violation",
        "pandas_empty_string_not_null_success",
        "pyspark_empty_string_not_null_success",
        "pandas_whitespace_not_null_success",
        "pyspark_whitespace_not_null_success",
        "pandas_mixed_empty_string_null_violation",
        "pyspark_mixed_empty_string_null_violation",
        "pandas_double_basic_success",
        "pyspark_double_basic_success",
        "pandas_double_with_zero_success",
        "pyspark_double_with_zero_success",
        "pandas_double_with_none_violation",
        "pyspark_double_with_none_violation",
        "pandas_double_with_nan_violation",
        "pandas_double_all_zeros_success",
        "pyspark_double_all_zeros_success",
        "pandas_int_all_zeros_success",
        "pyspark_int_all_zeros_success",
        "pandas_int_negative_values_success",
        "pyspark_int_negative_values_success",
        "pandas_double_negative_values_success",
        "pyspark_double_negative_values_success",
        "pandas_boolean_mixed_success",
        "pyspark_boolean_mixed_success",
        "pandas_boolean_all_false_success",
        "pyspark_boolean_all_false_success",
        "pandas_boolean_with_none_violation",
        "pyspark_boolean_with_none_violation",
        "pandas_boolean_false_not_null_success",
        "pyspark_boolean_false_not_null_success",
        "pandas_timestamp_basic_success",
        "pyspark_timestamp_basic_success",
        "pandas_timestamp_with_none_violation",
        "pyspark_timestamp_with_none_violation",
        "pandas_timestamp_with_tz_success",
        "pyspark_timestamp_with_tz_success",
        "pandas_multiple_nulls_violations",
        "pyspark_multiple_nulls_violations",
        "pandas_large_numbers_success",
        "pyspark_large_numbers_success",
        "pandas_single_value_success",
        "pyspark_single_value_success",
        "pandas_single_null_only_violation",
        "pyspark_single_null_only_violation",
    ],
)
def test_expectation_basic_scenarios(
    df_type,
    data,
    should_succeed,
    expected_violations,
    expected_message,
    data_type,
    spark,
):
    """Test basic expectation scenarios for both pandas and PySpark DataFrames.

    Tests various data types including:
    - Integers (long): positive, negative, zero, large numbers, with/without nulls
    - Strings: with nulls, empty strings (NOT null), whitespace (NOT null)
    - Floats (double): with None/NaN, zero values (NOT null), negative values (NOT null)
    - Booleans: True/False (NOT null), with None
    - Timestamps: with and without timezone, with None
    - Edge cases: missing columns, all nulls, single null, multiple nulls

    Note: ExpectationValueNotNull checks that values are NOT null.
    Success = no null values, Violations = null/None/NaN values.
    Empty strings, zeros, and False are NOT considered null.
    """
    df = create_dataframe(df_type, data, "col1", spark, data_type)

    # Test through registry
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationValueNotNull",
        column_name="col1",
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

        # Verify violations if present (skip detailed DataFrame comparison for pandas None values
        # as representation differs between None and np.nan in display)
        if expected_violations is not None and df_type == "pyspark":
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
    suite = DataFrameExpectationsSuite().expect_value_not_null(column_name="col1")

    if should_succeed:
        suite_result = suite.build().run(data_frame=df)
        assert suite_result is not None, "Expected SuiteExecutionResult"
        assert isinstance(suite_result, SuiteExecutionResult), (
            "Result should be SuiteExecutionResult"
        )
        assert suite_result.success, "Expected all expectations to pass"
        assert suite_result.total_passed == 1, "Expected 1 passed expectation"
        assert suite_result.total_failed == 0, "Expected 0 failed expectations"
    else:
        with pytest.raises(DataFrameExpectationsSuiteFailure):
            suite.build().run(data_frame=df)


@pytest.mark.parametrize(
    "df_type",
    [("pandas"), ("pyspark")],
    ids=["pandas", "pyspark"],
)
def test_column_missing_error(df_type, spark):
    """Test that a missing column returns the appropriate error message."""
    # Create DataFrame with a different column name (col2 instead of col1)
    df = create_dataframe(df_type, [1, 2, 3], "col2", spark, "long")

    # Test through registry - should return failure message about missing column
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationValueNotNull",
        column_name="col1",
    )
    result = expectation.validate(data_frame=df)

    assert isinstance(result, DataFrameExpectationFailureMessage), (
        f"Expected DataFrameExpectationFailureMessage but got: {type(result)}"
    )
    assert "Column 'col1' does not exist in the DataFrame" in str(result), (
        f"Expected missing column message but got: {result}"
    )

    # Test through suite - should raise DataFrameExpectationsSuiteFailure
    suite = DataFrameExpectationsSuite().expect_value_not_null(column_name="col1")
    with pytest.raises(DataFrameExpectationsSuiteFailure) as exc_info:
        suite.build().run(data_frame=df)

    assert "Column 'col1' does not exist in the DataFrame" in str(exc_info.value), (
        f"Expected missing column message in suite failure but got: {exc_info.value}"
    )


def test_large_dataset_performance():
    """Test the expectation with a larger dataset to ensure reasonable performance."""
    # Create a larger dataset with 10,000 rows, all non-null values
    large_data = list(range(1, 10001))
    data_frame = pd.DataFrame({"col1": large_data})

    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationValueNotNull",
        column_name="col1",
    )

    result = expectation.validate(data_frame=data_frame)
    assert isinstance(result, DataFrameExpectationSuccessMessage), (
        f"Large dataset test failed: expected success but got {type(result)}"
    )
