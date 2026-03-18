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
from dataframe_expectations.core.suite_result import SuiteExecutionResult
from dataframe_expectations.result_message import (
    DataFrameExpectationFailureMessage,
    DataFrameExpectationSuccessMessage,
)


def create_pyspark_dataframe(data, column_name, spark, data_type="long"):
    """Helper function to create a PySpark DataFrame."""
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
        expectation_name="ExpectationValueEquals",
        column_name="col1",
        value=5,
    )
    assert expectation.get_expectation_name() == "ExpectationValueEquals", (
        f"Expected 'ExpectationValueEquals' but got: {expectation.get_expectation_name()}"
    )


@pytest.mark.parametrize(
    "df_type, data, value, should_succeed, expected_violations, expected_message, data_type",
    [
        # Basic integer success scenarios
        ("pandas", [5, 5, 5], 5, True, None, None, "long"),
        ("pandas", [10, 10], 10, True, None, None, "long"),
        # Integer failure scenarios with violations
        (
            "pandas",
            [3, 4, 5],
            5,
            False,
            [3, 4],
            "Found 2 row(s) where 'col1' is not equal to 5.",
            "long",
        ),
        # String success scenarios
        ("pandas", ["apple", "apple", "apple"], "apple", True, None, None, "string"),
        # String failure scenarios
        (
            "pandas",
            ["apple", "banana", "apple"],
            "apple",
            False,
            ["banana"],
            "Found 1 row(s) where 'col1' is not equal to apple.",
            "string",
        ),
        # String case sensitivity
        (
            "pandas",
            ["Apple", "APPLE", "apple"],
            "apple",
            False,
            ["Apple", "APPLE"],
            "Found 2 row(s) where 'col1' is not equal to apple.",
            "string",
        ),
        # Float/Double success scenarios
        ("pandas", [3.14, 3.14, 3.14], 3.14, True, None, None, "double"),
        # Float/Double failure scenarios
        (
            "pandas",
            [3.14, 2.71, 3.14],
            3.14,
            False,
            [2.71],
            "Found 1 row(s) where 'col1' is not equal to 3.14.",
            "double",
        ),
        # Float precision edge cases
        ("pandas", [1.0, 1.0, 1.0], 1.0, True, None, None, "double"),
        (
            "pandas",
            [1.0, 1.1, 1.0],
            1.0,
            False,
            [1.1],
            "Found 1 row(s) where 'col1' is not equal to 1.0.",
            "double",
        ),
        # Boolean success scenarios - True
        ("pandas", [True, True, True], True, True, None, None, "boolean"),
        # Boolean success scenarios - False
        ("pandas", [False, False, False], False, True, None, None, "boolean"),
        # Boolean failure scenarios
        (
            "pandas",
            [True, False, True],
            True,
            False,
            [False],
            "Found 1 row(s) where 'col1' is not equal to True.",
            "boolean",
        ),
        # Timestamp success scenarios
        (
            "pandas",
            [datetime(2023, 1, 1), datetime(2023, 1, 1), datetime(2023, 1, 1)],
            datetime(2023, 1, 1),
            True,
            None,
            None,
            "timestamp",
        ),
        # Timestamp failure scenarios
        (
            "pandas",
            [datetime(2023, 1, 1), datetime(2023, 1, 2), datetime(2023, 1, 1)],
            datetime(2023, 1, 1),
            False,
            [datetime(2023, 1, 2)],
            "Found 1 row(s) where 'col1' is not equal to 2023-01-01 00:00:00.",
            "timestamp",
        ),
        # Datetime with timezone
        (
            "pandas",
            [
                datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
                datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            ],
            datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            True,
            None,
            None,
            "timestamp",
        ),
        # Empty string scenarios
        ("pandas", ["", "", ""], "", True, None, None, "string"),
        (
            "pandas",
            ["", "text", ""],
            "",
            False,
            ["text"],
            "Found 1 row(s) where 'col1' is not equal to .",
            "string",
        ),
        # Whitespace in strings
        (
            "pandas",
            ["test", " test", "test"],
            "test",
            False,
            [" test"],
            "Found 1 row(s) where 'col1' is not equal to test.",
            "string",
        ),
        # Zero value scenarios - integer
        ("pandas", [0, 0, 0], 0, True, None, None, "long"),
        # Zero value scenarios - double
        ("pandas", [0.0, 0.0, 0.0], 0.0, True, None, None, "double"),
        # Negative integer scenarios
        ("pandas", [-5, -5, -5], -5, True, None, None, "long"),
        # Negative double scenarios
        ("pandas", [-3.14, -3.14, -3.14], -3.14, True, None, None, "double"),
        # Large numbers
        ("pandas", [1000000, 1000000], 1000000, True, None, None, "long"),
    ],
    ids=[
        "pandas_int_basic_success",
        "pandas_int_success_two_rows",
        "pandas_int_failure_violations",
        "pandas_string_success",
        "pandas_string_failure",
        "pandas_string_case_sensitive",
        "pandas_double_success",
        "pandas_double_failure",
        "pandas_double_precision_success",
        "pandas_double_precision_failure",
        "pandas_boolean_true_success",
        "pandas_boolean_false_success",
        "pandas_boolean_failure",
        "pandas_timestamp_success",
        "pandas_timestamp_failure",
        "pandas_timestamp_with_timezone",
        "pandas_empty_string_success",
        "pandas_empty_string_failure",
        "pandas_string_whitespace",
        "pandas_zero_int",
        "pandas_zero_double",
        "pandas_negative_int",
        "pandas_negative_double",
        "pandas_large_numbers",
    ],
)
def test_expectation_basic_scenarios_pandas(
    df_type,
    data,
    value,
    should_succeed,
    expected_violations,
    expected_message,
    data_type,
):
    """Test basic expectation scenarios for pandas DataFrames.

    Tests various data types including:
    - Integers (long): positive, negative, zero, large numbers
    - Strings: case sensitivity, empty strings, whitespace
    - Floats (double): precision, zero, negative
    - Booleans: True/False
    - Timestamps: with and without timezone
    """
    df = pd.DataFrame({"col1": data})

    # Test through registry
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationValueEquals",
        column_name="col1",
        value=value,
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
            expected_violations_df = pd.DataFrame({"col1": expected_violations})
            expected_failure = DataFrameExpectationFailureMessage(
                expectation_str=str(expectation),
                data_frame_type="pandas",
                violations_data_frame=expected_violations_df,
                message=expected_message,
                limit_violations=5,
            )
            assert str(result) == str(expected_failure), (
                f"Expected failure details don't match. Got: {result}"
            )

    # Test through suite
    suite = DataFrameExpectationsSuite().expect_value_equals(column_name="col1", value=value)

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


@pytest.mark.pyspark
@pytest.mark.parametrize(
    "df_type, data, value, should_succeed, expected_violations, expected_message, data_type",
    [
        # Basic integer success scenarios
        ("pyspark", [5, 5, 5], 5, True, None, None, "long"),
        ("pyspark", [10, 10], 10, True, None, None, "long"),
        # Integer failure scenarios with violations
        (
            "pyspark",
            [3, 4, 5],
            5,
            False,
            [3, 4],
            "Found 2 row(s) where 'col1' is not equal to 5.",
            "long",
        ),
        # String success scenarios
        ("pyspark", ["apple", "apple", "apple"], "apple", True, None, None, "string"),
        # String failure scenarios
        (
            "pyspark",
            ["apple", "banana", "apple"],
            "apple",
            False,
            ["banana"],
            "Found 1 row(s) where 'col1' is not equal to apple.",
            "string",
        ),
        # String case sensitivity
        (
            "pyspark",
            ["Apple", "APPLE", "apple"],
            "apple",
            False,
            ["Apple", "APPLE"],
            "Found 2 row(s) where 'col1' is not equal to apple.",
            "string",
        ),
        # Float/Double success scenarios
        ("pyspark", [3.14, 3.14, 3.14], 3.14, True, None, None, "double"),
        # Float/Double failure scenarios
        (
            "pyspark",
            [3.14, 2.71, 3.14],
            3.14,
            False,
            [2.71],
            "Found 1 row(s) where 'col1' is not equal to 3.14.",
            "double",
        ),
        # Float precision edge cases
        ("pyspark", [1.0, 1.0, 1.0], 1.0, True, None, None, "double"),
        (
            "pyspark",
            [1.0, 1.1, 1.0],
            1.0,
            False,
            [1.1],
            "Found 1 row(s) where 'col1' is not equal to 1.0.",
            "double",
        ),
        # Boolean success scenarios - True
        ("pyspark", [True, True, True], True, True, None, None, "boolean"),
        # Boolean success scenarios - False
        ("pyspark", [False, False, False], False, True, None, None, "boolean"),
        # Boolean failure scenarios
        (
            "pyspark",
            [True, False, True],
            True,
            False,
            [False],
            "Found 1 row(s) where 'col1' is not equal to True.",
            "boolean",
        ),
        # Timestamp success scenarios
        (
            "pyspark",
            [datetime(2023, 1, 1), datetime(2023, 1, 1), datetime(2023, 1, 1)],
            datetime(2023, 1, 1),
            True,
            None,
            None,
            "timestamp",
        ),
        # Timestamp failure scenarios
        (
            "pyspark",
            [datetime(2023, 1, 1), datetime(2023, 1, 2), datetime(2023, 1, 1)],
            datetime(2023, 1, 1),
            False,
            [datetime(2023, 1, 2)],
            "Found 1 row(s) where 'col1' is not equal to 2023-01-01 00:00:00.",
            "timestamp",
        ),
        # Datetime with timezone
        (
            "pyspark",
            [
                datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
                datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            ],
            datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            True,
            None,
            None,
            "timestamp",
        ),
        # Empty string scenarios
        ("pyspark", ["", "", ""], "", True, None, None, "string"),
        (
            "pyspark",
            ["", "text", ""],
            "",
            False,
            ["text"],
            "Found 1 row(s) where 'col1' is not equal to .",
            "string",
        ),
        # Whitespace in strings
        (
            "pyspark",
            ["test", " test", "test"],
            "test",
            False,
            [" test"],
            "Found 1 row(s) where 'col1' is not equal to test.",
            "string",
        ),
        # Zero value scenarios - integer
        ("pyspark", [0, 0, 0], 0, True, None, None, "long"),
        # Zero value scenarios - double
        ("pyspark", [0.0, 0.0, 0.0], 0.0, True, None, None, "double"),
        # Negative integer scenarios
        ("pyspark", [-5, -5, -5], -5, True, None, None, "long"),
        # Negative double scenarios
        ("pyspark", [-3.14, -3.14, -3.14], -3.14, True, None, None, "double"),
        # Large numbers
        ("pyspark", [1000000, 1000000], 1000000, True, None, None, "long"),
    ],
    ids=[
        "pyspark_int_basic_success",
        "pyspark_int_success_two_rows",
        "pyspark_int_failure_violations",
        "pyspark_string_success",
        "pyspark_string_failure",
        "pyspark_string_case_sensitive",
        "pyspark_double_success",
        "pyspark_double_failure",
        "pyspark_double_precision_success",
        "pyspark_double_precision_failure",
        "pyspark_boolean_true_success",
        "pyspark_boolean_false_success",
        "pyspark_boolean_failure",
        "pyspark_timestamp_success",
        "pyspark_timestamp_failure",
        "pyspark_timestamp_with_timezone",
        "pyspark_empty_string_success",
        "pyspark_empty_string_failure",
        "pyspark_string_whitespace",
        "pyspark_zero_int",
        "pyspark_zero_double",
        "pyspark_negative_int",
        "pyspark_negative_double",
        "pyspark_large_numbers",
    ],
)
def test_expectation_basic_scenarios_pyspark(
    df_type,
    data,
    value,
    should_succeed,
    expected_violations,
    expected_message,
    data_type,
    spark,
):
    """Test basic expectation scenarios for PySpark DataFrames.

    Tests various data types including:
    - Integers (long): positive, negative, zero, large numbers
    - Strings: case sensitivity, empty strings, whitespace
    - Floats (double): precision, zero, negative
    - Booleans: True/False
    - Timestamps: with and without timezone
    """
    df = create_pyspark_dataframe(data, "col1", spark, data_type)

    # Test through registry
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationValueEquals",
        column_name="col1",
        value=value,
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
            expected_violations_df = create_pyspark_dataframe(
                expected_violations, "col1", spark, data_type
            )
            expected_failure = DataFrameExpectationFailureMessage(
                expectation_str=str(expectation),
                data_frame_type="pyspark",
                violations_data_frame=expected_violations_df,
                message=expected_message,
                limit_violations=5,
            )
            assert str(result) == str(expected_failure), (
                f"Expected failure details don't match. Got: {result}"
            )

    # Test through suite
    suite = DataFrameExpectationsSuite().expect_value_equals(column_name="col1", value=value)

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


def test_column_missing_error_pandas():
    """Test missing column error for pandas DataFrames."""
    df = pd.DataFrame({"col1": [5, 5, 5]})

    expected_message = "Column 'nonexistent_col' does not exist in the DataFrame."

    # Test through registry
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationValueEquals",
        column_name="nonexistent_col",
        value=5,
    )
    result = expectation.validate(data_frame=df)
    expected_failure = DataFrameExpectationFailureMessage(
        expectation_str=str(expectation),
        data_frame_type="pandas",
        message=expected_message,
    )
    assert str(result) == str(expected_failure)

    # Test through suite
    suite = DataFrameExpectationsSuite().expect_value_equals(column_name="nonexistent_col", value=5)
    with pytest.raises(DataFrameExpectationsSuiteFailure):
        suite.build().run(data_frame=df)


@pytest.mark.pyspark
def test_column_missing_error_pyspark(spark):
    """Test missing column error for PySpark DataFrames."""
    df = spark.createDataFrame([(5,), (5,), (5,)], ["col1"])

    expected_message = "Column 'nonexistent_col' does not exist in the DataFrame."

    # Test through registry
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationValueEquals",
        column_name="nonexistent_col",
        value=5,
    )
    result = expectation.validate(data_frame=df)
    expected_failure = DataFrameExpectationFailureMessage(
        expectation_str=str(expectation),
        data_frame_type="pyspark",
        message=expected_message,
    )
    assert str(result) == str(expected_failure)

    # Test through suite
    suite = DataFrameExpectationsSuite().expect_value_equals(column_name="nonexistent_col", value=5)
    with pytest.raises(DataFrameExpectationsSuiteFailure):
        suite.build().run(data_frame=df)


def test_large_dataset_performance():
    """Test the expectation with a larger dataset to ensure reasonable performance."""

    # Create a larger dataset with 10,000 rows all equal to 42
    large_data = [42] * 10000
    data_frame = pd.DataFrame({"col1": large_data})

    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationValueEquals",
        column_name="col1",
        value=42,
    )

    result = expectation.validate(data_frame=data_frame)
    assert isinstance(result, DataFrameExpectationSuccessMessage), (
        f"Large dataset test failed: expected success but got {type(result)}"
    )
