import pytest
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


def test_expectation_name():
    """Test that the expectation name is correctly returned."""
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationValueNotEquals",
        column_name="col1",
        value=5,
    )
    assert expectation.get_expectation_name() == "ExpectationValueNotEquals", (
        f"Expected 'ExpectationValueNotEquals' but got: {expectation.get_expectation_name()}"
    )


@pytest.mark.parametrize(
    "data, value, should_succeed, expected_violations, expected_message, data_type",
    [
        # Basic integer scenarios - success (values NOT equal to 5)
        ([3, 4, 6], 5, True, None, None, "long"),
        ([1, 2, 3], 5, True, None, None, "long"),
        # Integer scenarios - violations (values equal to 5)
        (
            [3, 5, 5],
            5,
            False,
            [5, 5],
            "Found 2 row(s) where 'col1' is equal to 5.",
            "long",
        ),
        (
            [5, 5, 5],
            5,
            False,
            [5, 5, 5],
            "Found 3 row(s) where 'col1' is equal to 5.",
            "long",
        ),
        # String data type scenarios - success
        (["banana", "cherry", "orange"], "apple", True, None, None, "string"),
        # String scenarios - violations
        (
            ["apple", "banana", "apple"],
            "apple",
            False,
            ["apple", "apple"],
            "Found 2 row(s) where 'col1' is equal to apple.",
            "string",
        ),
        # String case sensitivity - success (case matters)
        (["Apple", "APPLE", "aPpLe"], "apple", True, None, None, "string"),
        # String case sensitivity - violations (exact match)
        (
            ["apple", "Apple", "apple"],
            "apple",
            False,
            ["apple", "apple"],
            "Found 2 row(s) where 'col1' is equal to apple.",
            "string",
        ),
        # Float/Double data type scenarios - success
        ([1.5, 2.5, 4.5], 3.14, True, None, None, "double"),
        # Float scenarios - violations
        (
            [3.14, 2.71, 3.14],
            3.14,
            False,
            [3.14, 3.14],
            "Found 2 row(s) where 'col1' is equal to 3.14.",
            "double",
        ),
        # Float precision edge cases - success
        ([1.1, 1.2, 1.3], 1.0, True, None, None, "double"),
        # Float precision - violations
        (
            [1.0, 1.1, 1.0],
            1.0,
            False,
            [1.0, 1.0],
            "Found 2 row(s) where 'col1' is equal to 1.0.",
            "double",
        ),
        # Boolean data type scenarios - success
        ([False, False, False], True, True, None, None, "boolean"),
        ([True, True, True], False, True, None, None, "boolean"),
        # Boolean scenarios - violations
        (
            [True, False, True],
            True,
            False,
            [True, True],
            "Found 2 row(s) where 'col1' is equal to True.",
            "boolean",
        ),
        (
            [False, True, False],
            False,
            False,
            [False, False],
            "Found 2 row(s) where 'col1' is equal to False.",
            "boolean",
        ),
        # Timestamp/Datetime scenarios - success
        (
            [datetime(2023, 1, 2), datetime(2023, 1, 3), datetime(2023, 1, 4)],
            datetime(2023, 1, 1),
            True,
            None,
            None,
            "timestamp",
        ),
        # Timestamp scenarios - violations
        (
            [datetime(2023, 1, 1), datetime(2023, 1, 2), datetime(2023, 1, 1)],
            datetime(2023, 1, 1),
            False,
            [datetime(2023, 1, 1), datetime(2023, 1, 1)],
            "Found 2 row(s) where 'col1' is equal to 2023-01-01 00:00:00.",
            "timestamp",
        ),
        # Datetime with timezone - success
        (
            [
                datetime(2023, 1, 2, 12, 0, 0, tzinfo=timezone.utc),
                datetime(2023, 1, 3, 12, 0, 0, tzinfo=timezone.utc),
            ],
            datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            True,
            None,
            None,
            "timestamp_utc",
        ),
        # Empty string scenarios - success
        (["text", "data", "value"], "", True, None, None, "string"),
        # Empty string - violations
        (
            ["", "text", ""],
            "",
            False,
            ["", ""],
            "Found 2 row(s) where 'col1' is equal to .",
            "string",
        ),
        # Whitespace in strings - success
        ([" test", "test ", " test "], "test", True, None, None, "string"),
        # Zero value scenarios - success
        ([1, 2, 3], 0, True, None, None, "long"),
        ([1.5, 2.5, 3.5], 0.0, True, None, None, "double"),
        # Zero value - violations
        (
            [0, 1, 0],
            0,
            False,
            [0, 0],
            "Found 2 row(s) where 'col1' is equal to 0.",
            "long",
        ),
        # Negative numbers - success
        ([1, 2, 3], -5, True, None, None, "long"),
        ([1.5, 2.5, 3.5], -3.14, True, None, None, "double"),
        # Negative numbers - violations
        (
            [-5, 1, -5],
            -5,
            False,
            [-5, -5],
            "Found 2 row(s) where 'col1' is equal to -5.",
            "long",
        ),
        # Large numbers - success
        ([999999, 1000001], 1000000, True, None, None, "long"),
        # Large numbers - violations
        (
            [1000000, 999999, 1000000],
            1000000,
            False,
            [1000000, 1000000],
            "Found 2 row(s) where 'col1' is equal to 1000000.",
            "long",
        ),
    ],
    ids=[
        "int_basic_success",
        "int_success_different_values",
        "int_violations_two",
        "int_violations_all",
        "string_success",
        "string_violations",
        "string_case_sensitive_success",
        "string_case_sensitive_violations",
        "double_success",
        "double_violations",
        "double_precision_success",
        "double_precision_violations",
        "boolean_false_success",
        "boolean_true_success",
        "boolean_true_violations",
        "boolean_false_violations",
        "timestamp_success",
        "timestamp_violations",
        "timestamp_with_timezone_success",
        "empty_string_success",
        "empty_string_violations",
        "string_whitespace_success",
        "zero_int_success",
        "zero_double_success",
        "zero_int_violations",
        "negative_int_success",
        "negative_double_success",
        "negative_int_violations",
        "large_numbers_success",
        "large_numbers_violations",
    ],
)
def test_expectation_basic_scenarios(
    dataframe_factory,
    data,
    value,
    should_succeed,
    expected_violations,
    expected_message,
    data_type,
):
    """Test basic expectation scenarios for pandas and PySpark DataFrames."""
    df_lib, make_df = dataframe_factory
    df = make_df({"col1": (data, data_type)})

    # Test through registry
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationValueNotEquals",
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
            expected_violations_df = make_df({"col1": (expected_violations, data_type)})
            expected_failure = DataFrameExpectationFailureMessage(
                expectation_str=str(expectation),
                data_frame_type=df_lib,
                violations_data_frame=expected_violations_df,
                message=expected_message,
                limit_violations=5,
            )
            assert str(result) == str(expected_failure), (
                f"Expected failure details don't match. Got: {result}"
            )

    # Test through suite
    suite = DataFrameExpectationsSuite().expect_value_not_equals(column_name="col1", value=value)

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


def test_column_missing_error(dataframe_factory):
    """Test missing column error for pandas and PySpark DataFrames."""
    df_lib, make_df = dataframe_factory
    df = make_df({"col1": ([3, 4, 5], "long")})

    expected_message = "Column 'nonexistent_col' does not exist in the DataFrame."

    # Test through registry
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationValueNotEquals",
        column_name="nonexistent_col",
        value=5,
    )
    result = expectation.validate(data_frame=df)
    expected_failure = DataFrameExpectationFailureMessage(
        expectation_str=str(expectation),
        data_frame_type=df_lib,
        message=expected_message,
    )
    assert str(result) == str(expected_failure)

    # Test through suite
    suite = DataFrameExpectationsSuite().expect_value_not_equals(
        column_name="nonexistent_col", value=5
    )
    with pytest.raises(DataFrameExpectationsSuiteFailure):
        suite.build().run(data_frame=df)


def test_large_dataset_performance(dataframe_factory):
    """Test the expectation with a larger dataset to ensure reasonable performance."""
    df_lib, make_df = dataframe_factory

    # Create a larger dataset with 10,000 rows, all NOT equal to 42 (using 43)
    large_data = [43] * 10000
    df = make_df({"col1": (large_data, "long")})

    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationValueNotEquals",
        column_name="col1",
        value=42,
    )

    result = expectation.validate(data_frame=df)
    assert isinstance(result, DataFrameExpectationSuccessMessage), (
        f"Large dataset test failed: expected success but got {type(result)}"
    )
