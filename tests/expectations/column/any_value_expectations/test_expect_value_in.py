import pytest
from datetime import datetime

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
        expectation_name="ExpectationValueIn",
        column_name="col1",
        values=[1, 2, 3],
    )
    assert expectation.get_expectation_name() == "ExpectationValueIn", (
        f"Expected 'ExpectationValueIn' but got: {expectation.get_expectation_name()}"
    )


@pytest.mark.parametrize(
    "data, values, should_succeed, expected_violations, expected_message, data_type",
    [
        # Basic integer scenarios - success
        ([1, 2, 3, 2, 1], [1, 2, 3], True, None, None, "long"),
        # Integer scenarios - violations
        (
            [1, 4, 5, 2, 3],
            [1, 2, 3],
            False,
            [4, 5],
            "Found 2 row(s) where 'col1' is not in [1, 2, 3].",
            "long",
        ),
        # String data type scenarios
        (
            ["apple", "banana", "cherry"],
            ["apple", "banana", "cherry"],
            True,
            None,
            None,
            "string",
        ),
        (
            ["apple", "orange", "banana"],
            ["apple", "banana"],
            False,
            ["orange"],
            "Found 1 row(s) where 'col1' is not in ['apple', 'banana'].",
            "string",
        ),
        # String case sensitivity
        (
            ["Apple", "apple", "APPLE"],
            ["apple"],
            False,
            ["Apple", "APPLE"],
            "Found 2 row(s) where 'col1' is not in ['apple'].",
            "string",
        ),
        # Float/Double data type scenarios
        ([1.5, 2.5, 3.5], [1.5, 2.5, 3.5], True, None, None, "double"),
        (
            [1.5, 4.5, 2.5],
            [1.5, 2.5],
            False,
            [4.5],
            "Found 1 row(s) where 'col1' is not in [1.5, 2.5].",
            "double",
        ),
        # Boolean data type scenarios
        ([True, False, True], [True, False], True, None, None, "boolean"),
        ([True, True, True], [True], True, None, None, "boolean"),
        (
            [True, False, True],
            [True],
            False,
            [False],
            "Found 1 row(s) where 'col1' is not in [True].",
            "boolean",
        ),
        # Timestamp/Datetime scenarios
        (
            [datetime(2023, 1, 1), datetime(2023, 1, 2), datetime(2023, 1, 3)],
            [datetime(2023, 1, 1), datetime(2023, 1, 2), datetime(2023, 1, 3)],
            True,
            None,
            None,
            "timestamp",
        ),
        (
            [datetime(2023, 1, 1), datetime(2023, 1, 4), datetime(2023, 1, 2)],
            [datetime(2023, 1, 1), datetime(2023, 1, 2)],
            False,
            [datetime(2023, 1, 4)],
            "Found 1 row(s) where 'col1' is not in [datetime.datetime(2023, 1, 1, 0, 0), datetime.datetime(2023, 1, 2, 0, 0)].",
            "timestamp",
        ),
        # Empty string scenarios
        (["", "text", ""], ["", "text"], True, None, None, "string"),
        # Zero value scenarios
        ([0, 1, 2, 0], [0, 1, 2], True, None, None, "long"),
        ([0.0, 1.0, 2.0], [0.0, 1.0, 2.0], True, None, None, "double"),
        # Negative numbers
        ([-1, -2, -3], [-1, -2, -3], True, None, None, "long"),
        (
            [-1, 5, -2],
            [-1, -2],
            False,
            [5],
            "Found 1 row(s) where 'col1' is not in [-1, -2].",
            "long",
        ),
        # Large numbers
        ([1000000, 2000000], [1000000, 2000000], True, None, None, "long"),
        # Single value in list
        ([5, 5, 5], [5], True, None, None, "long"),
        # Multiple values with single violation
        (
            [1, 2, 3, 4, 5],
            [1, 2, 3, 4],
            False,
            [5],
            "Found 1 row(s) where 'col1' is not in [1, 2, 3, 4].",
            "long",
        ),
    ],
    ids=[
        "int_basic_success",
        "int_violations",
        "string_success",
        "string_violations",
        "string_case_sensitive",
        "double_success",
        "double_violations",
        "boolean_both_values",
        "boolean_single_value",
        "boolean_violation",
        "timestamp_success",
        "timestamp_violation",
        "empty_string",
        "zero_int",
        "zero_double",
        "negative_int_success",
        "negative_int_violation",
        "large_numbers",
        "single_value_list",
        "multiple_values_single_violation",
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
    - Strings: case sensitivity, empty strings
    - Floats (double): precision, zero values
    - Booleans: True/False combinations
    - Timestamps: datetime objects
    - Edge cases: single value lists
    """
    df_lib, make_df = dataframe_factory
    df = make_df({"col1": (data, data_type)})

    # Test through registry
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationValueIn",
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
    suite = DataFrameExpectationsSuite().expect_value_in(column_name="col1", values=values)

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
    df = make_df({"col1": ([1, 2, 3], "long")})

    expected_message = "Column 'nonexistent_col' does not exist in the DataFrame."

    # Test through registry
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationValueIn",
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
    suite = DataFrameExpectationsSuite().expect_value_in(
        column_name="nonexistent_col", values=[1, 2, 3]
    )
    with pytest.raises(DataFrameExpectationsSuiteFailure):
        suite.build().run(data_frame=df)


def test_large_dataset_performance(dataframe_factory):
    """Test the expectation with a larger dataset to ensure reasonable performance."""
    df_lib, make_df = dataframe_factory

    # Create a larger dataset with 10,000 rows with values in [1, 2, 3]
    large_data = [1, 2, 3] * 3334  # Creates ~10,000 values
    df = make_df({"col1": (large_data, "long")})

    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationValueIn",
        column_name="col1",
        values=[1, 2, 3],
    )

    result = expectation.validate(data_frame=df)
    assert isinstance(result, DataFrameExpectationSuccessMessage), (
        f"Large dataset test failed: expected success but got {type(result)}"
    )
