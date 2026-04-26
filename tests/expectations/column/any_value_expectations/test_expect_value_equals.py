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
        expectation_name="ExpectationValueEquals",
        column_name="col1",
        value=5,
    )
    assert expectation.get_expectation_name() == "ExpectationValueEquals", (
        f"Expected 'ExpectationValueEquals' but got: {expectation.get_expectation_name()}"
    )


@pytest.mark.parametrize(
    "data, value, should_succeed, expected_violations, expected_message, data_type",
    [
        # Basic integer success scenarios
        ([5, 5, 5], 5, True, None, None, "long"),
        ([10, 10], 10, True, None, None, "long"),
        # Integer failure scenarios with violations
        (
            [3, 4, 5],
            5,
            False,
            [3, 4],
            "Found 2 row(s) where 'col1' is not equal to 5.",
            "long",
        ),
        # String success scenarios
        (["apple", "apple", "apple"], "apple", True, None, None, "string"),
        # String failure scenarios
        (
            ["apple", "banana", "apple"],
            "apple",
            False,
            ["banana"],
            "Found 1 row(s) where 'col1' is not equal to apple.",
            "string",
        ),
        # String case sensitivity
        (
            ["Apple", "APPLE", "apple"],
            "apple",
            False,
            ["Apple", "APPLE"],
            "Found 2 row(s) where 'col1' is not equal to apple.",
            "string",
        ),
        # Float/Double success scenarios
        ([3.14, 3.14, 3.14], 3.14, True, None, None, "double"),
        # Float/Double failure scenarios
        (
            [3.14, 2.71, 3.14],
            3.14,
            False,
            [2.71],
            "Found 1 row(s) where 'col1' is not equal to 3.14.",
            "double",
        ),
        # Float precision edge cases
        ([1.0, 1.0, 1.0], 1.0, True, None, None, "double"),
        (
            [1.0, 1.1, 1.0],
            1.0,
            False,
            [1.1],
            "Found 1 row(s) where 'col1' is not equal to 1.0.",
            "double",
        ),
        # Boolean success scenarios - True
        ([True, True, True], True, True, None, None, "boolean"),
        # Boolean success scenarios - False
        ([False, False, False], False, True, None, None, "boolean"),
        # Boolean failure scenarios
        (
            [True, False, True],
            True,
            False,
            [False],
            "Found 1 row(s) where 'col1' is not equal to True.",
            "boolean",
        ),
        # Timestamp success scenarios
        (
            [datetime(2023, 1, 1), datetime(2023, 1, 1), datetime(2023, 1, 1)],
            datetime(2023, 1, 1),
            True,
            None,
            None,
            "timestamp",
        ),
        # Timestamp failure scenarios
        (
            [datetime(2023, 1, 1), datetime(2023, 1, 2), datetime(2023, 1, 1)],
            datetime(2023, 1, 1),
            False,
            [datetime(2023, 1, 2)],
            "Found 1 row(s) where 'col1' is not equal to 2023-01-01 00:00:00.",
            "timestamp",
        ),
        # Datetime with timezone
        (
            [
                datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
                datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            ],
            datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            True,
            None,
            None,
            "timestamp_utc",
        ),
        # Empty string scenarios
        (["", "", ""], "", True, None, None, "string"),
        (
            ["", "text", ""],
            "",
            False,
            ["text"],
            "Found 1 row(s) where 'col1' is not equal to .",
            "string",
        ),
        # Whitespace in strings
        (
            ["test", " test", "test"],
            "test",
            False,
            [" test"],
            "Found 1 row(s) where 'col1' is not equal to test.",
            "string",
        ),
        # Zero value scenarios - integer
        ([0, 0, 0], 0, True, None, None, "long"),
        # Zero value scenarios - double
        ([0.0, 0.0, 0.0], 0.0, True, None, None, "double"),
        # Negative integer scenarios
        ([-5, -5, -5], -5, True, None, None, "long"),
        # Negative double scenarios
        ([-3.14, -3.14, -3.14], -3.14, True, None, None, "double"),
        # Large numbers
        ([1000000, 1000000], 1000000, True, None, None, "long"),
    ],
    ids=[
        "int_basic_success",
        "int_success_two_rows",
        "int_failure_violations",
        "string_success",
        "string_failure",
        "string_case_sensitive",
        "double_success",
        "double_failure",
        "double_precision_success",
        "double_precision_failure",
        "boolean_true_success",
        "boolean_false_success",
        "boolean_failure",
        "timestamp_success",
        "timestamp_failure",
        "timestamp_with_timezone",
        "empty_string_success",
        "empty_string_failure",
        "string_whitespace",
        "zero_int",
        "zero_double",
        "negative_int",
        "negative_double",
        "large_numbers",
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
    """Test basic expectation scenarios for pandas and PySpark DataFrames.

    Tests various data types including:
    - Integers (long): positive, negative, zero, large numbers
    - Strings: case sensitivity, empty strings, whitespace
    - Floats (double): precision, zero, negative
    - Booleans: True/False
    - Timestamps: with and without timezone
    """
    df_lib, make_df = dataframe_factory
    df = make_df({"col1": (data, data_type)})

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


def test_column_missing_error(dataframe_factory):
    """Test missing column error for pandas and PySpark DataFrames."""
    df_lib, make_df = dataframe_factory
    df = make_df({"col1": ([5, 5, 5], "long")})

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
        data_frame_type=df_lib,
        message=expected_message,
    )
    assert str(result) == str(expected_failure)

    # Test through suite
    suite = DataFrameExpectationsSuite().expect_value_equals(column_name="nonexistent_col", value=5)
    with pytest.raises(DataFrameExpectationsSuiteFailure):
        suite.build().run(data_frame=df)


def test_large_dataset_performance(dataframe_factory):
    """Test the expectation with a larger dataset to ensure reasonable performance."""
    df_lib, make_df = dataframe_factory

    # Create a larger dataset with 10,000 rows all equal to 42
    large_data = [42] * 10000
    df = make_df({"col1": (large_data, "long")})

    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationValueEquals",
        column_name="col1",
        value=42,
    )

    result = expectation.validate(data_frame=df)
    assert isinstance(result, DataFrameExpectationSuccessMessage), (
        f"Large dataset test failed: expected success but got {type(result)}"
    )
