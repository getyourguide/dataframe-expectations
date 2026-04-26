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
        expectation_name="ExpectationValueNotIn",
        column_name="col1",
        values=[1, 2, 3],
    )
    assert expectation.get_expectation_name() == "ExpectationValueNotIn", (
        f"Expected 'ExpectationValueNotIn' but got: {expectation.get_expectation_name()}"
    )


@pytest.mark.parametrize(
    "data, values, should_succeed, expected_violations, expected_message, data_type",
    [
        # Basic integer scenarios - success (values NOT in [1,2,3])
        ([4, 5, 6, 7], [1, 2, 3], True, None, None, "long"),
        ([10, 20, 30], [1, 2, 3], True, None, None, "long"),
        # Integer scenarios - violations (values in [1,2,3])
        (
            [1, 2, 4, 5],
            [1, 2, 3],
            False,
            [1, 2],
            "Found 2 row(s) where 'col1' is in [1, 2, 3].",
            "long",
        ),
        (
            [1, 2, 3],
            [1, 2, 3],
            False,
            [1, 2, 3],
            "Found 3 row(s) where 'col1' is in [1, 2, 3].",
            "long",
        ),
        # String data type scenarios - success
        (
            ["orange", "grape", "melon"],
            ["apple", "banana"],
            True,
            None,
            None,
            "string",
        ),
        # String scenarios - violations
        (
            ["apple", "orange", "banana"],
            ["apple", "banana"],
            False,
            ["apple", "banana"],
            "Found 2 row(s) where 'col1' is in ['apple', 'banana'].",
            "string",
        ),
        # String case sensitivity - success (case matters)
        (["Apple", "APPLE", "aPpLe"], ["apple"], True, None, None, "string"),
        # String case sensitivity - violations (exact match)
        (
            ["apple", "Apple", "banana"],
            ["apple", "banana"],
            False,
            ["apple", "banana"],
            "Found 2 row(s) where 'col1' is in ['apple', 'banana'].",
            "string",
        ),
        # Float/Double data type scenarios - success
        ([5.5, 6.5, 7.5], [1.5, 2.5, 3.5], True, None, None, "double"),
        # Float scenarios - violations
        (
            [1.5, 2.5, 5.5],
            [1.5, 2.5],
            False,
            [1.5, 2.5],
            "Found 2 row(s) where 'col1' is in [1.5, 2.5].",
            "double",
        ),
        # Float precision edge cases - success
        ([1.1, 1.2, 1.3], [1.0, 2.0], True, None, None, "double"),
        # Float precision - violations
        (
            [1.0, 1.5, 2.0],
            [1.0, 2.0],
            False,
            [1.0, 2.0],
            "Found 2 row(s) where 'col1' is in [1.0, 2.0].",
            "double",
        ),
        # Boolean data type scenarios - success
        ([False, False, False], [True], True, None, None, "boolean"),
        ([True, True, True], [False], True, None, None, "boolean"),
        # Boolean scenarios - violations
        (
            [True, False, True],
            [True],
            False,
            [True, True],
            "Found 2 row(s) where 'col1' is in [True].",
            "boolean",
        ),
        (
            [True, False, False],
            [True, False],
            False,
            [True, False, False],
            "Found 3 row(s) where 'col1' is in [True, False].",
            "boolean",
        ),
        # Timestamp/Datetime scenarios - success
        (
            [datetime(2023, 1, 5), datetime(2023, 1, 6), datetime(2023, 1, 7)],
            [datetime(2023, 1, 1), datetime(2023, 1, 2)],
            True,
            None,
            None,
            "timestamp",
        ),
        # Timestamp scenarios - violations
        (
            [datetime(2023, 1, 1), datetime(2023, 1, 5), datetime(2023, 1, 2)],
            [datetime(2023, 1, 1), datetime(2023, 1, 2)],
            False,
            [datetime(2023, 1, 1), datetime(2023, 1, 2)],
            "Found 2 row(s) where 'col1' is in [datetime.datetime(2023, 1, 1, 0, 0), datetime.datetime(2023, 1, 2, 0, 0)].",
            "timestamp",
        ),
        # Datetime with timezone - success
        (
            [
                datetime(2023, 1, 5, 12, 0, 0, tzinfo=timezone.utc),
                datetime(2023, 1, 6, 12, 0, 0, tzinfo=timezone.utc),
            ],
            [datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)],
            True,
            None,
            None,
            "timestamp_utc",
        ),
        # Empty string scenarios - success
        (["text", "data", "value"], [""], True, None, None, "string"),
        # Empty string - violations
        (
            ["", "text", ""],
            [""],
            False,
            ["", ""],
            "Found 2 row(s) where 'col1' is in [''].",
            "string",
        ),
        # Whitespace in strings - success
        ([" test", "test ", " test "], ["test"], True, None, None, "string"),
        # Zero value scenarios - success
        ([1, 2, 3], [0], True, None, None, "long"),
        ([1.5, 2.5, 3.5], [0.0], True, None, None, "double"),
        # Zero value - violations
        (
            [0, 1, 0],
            [0],
            False,
            [0, 0],
            "Found 2 row(s) where 'col1' is in [0].",
            "long",
        ),
        # Negative numbers - success
        ([1, 2, 3], [-5, -10], True, None, None, "long"),
        ([1.5, 2.5, 3.5], [-3.14], True, None, None, "double"),
        # Negative numbers - violations
        (
            [-5, 1, -5],
            [-5],
            False,
            [-5, -5],
            "Found 2 row(s) where 'col1' is in [-5].",
            "long",
        ),
        # Large numbers - success
        ([999999, 1000001], [1000000], True, None, None, "long"),
        # Large numbers - violations
        (
            [1000000, 999999, 1000000],
            [1000000],
            False,
            [1000000, 1000000],
            "Found 2 row(s) where 'col1' is in [1000000].",
            "long",
        ),
        # Single value in list - success
        ([10, 20, 30], [5], True, None, None, "long"),
        # Multiple values with partial match
        (
            [1, 5, 6],
            [1, 2, 3, 4],
            False,
            [1],
            "Found 1 row(s) where 'col1' is in [1, 2, 3, 4].",
            "long",
        ),
    ],
    ids=[
        "int_basic_success",
        "int_success_different_range",
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
        "boolean_both_violations",
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
        "single_value_list_success",
        "multiple_values_partial_match",
    ],
)
def test_expectation_basic_scenarios(
    dataframe_factory,
    data,
    values,
    should_succeed,
    expected_violations,
    expected_message,
    data_type,
):
    """Test basic expectation scenarios for pandas and PySpark DataFrames.

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
    df_lib, make_df = dataframe_factory
    df = make_df({"col1": (data, data_type)})

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
            expected_violations_df = make_df({"col1": (expected_violations, data_type)})
            expected_failure = DataFrameExpectationFailureMessage(
                expectation_str=str(expectation),
                data_frame_type=df_lib.value,
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
    df = make_df({"col1": ([4, 5, 6], "long")})

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
        data_frame_type=df_lib.value,
        message=expected_message,
    )
    assert str(result) == str(expected_failure)

    # Test through suite
    suite = DataFrameExpectationsSuite().expect_value_not_in(
        column_name="nonexistent_col", values=[1, 2, 3]
    )
    with pytest.raises(DataFrameExpectationsSuiteFailure):
        suite.build().run(data_frame=df)


def test_large_dataset_performance(dataframe_factory):
    """Test the expectation with a larger dataset to ensure reasonable performance."""
    df_lib, make_df = dataframe_factory

    # Create a larger dataset with 10,000 rows, all NOT in [1, 2, 3] (using values 4+)
    large_data = list(range(4, 10004))  # 10,000 values from 4 to 10003
    df = make_df({"col1": (large_data, "long")})

    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationValueNotIn",
        column_name="col1",
        values=[1, 2, 3],
    )

    result = expectation.validate(data_frame=df)
    assert isinstance(result, DataFrameExpectationSuccessMessage), (
        f"Large dataset test failed: expected success but got {type(result)}"
    )
