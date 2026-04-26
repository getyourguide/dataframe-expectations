import pytest

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
        expectation_name="ExpectationStringLengthEquals",
        column_name="col1",
        length=3,
    )
    assert expectation.get_expectation_name() == "ExpectationStringLengthEquals", (
        f"Expected 'ExpectationStringLengthEquals' but got: {expectation.get_expectation_name()}"
    )


@pytest.mark.parametrize(
    "data, length, expected_result, expected_violations, expected_message",
    [
        # Basic success
        (["foo", "bar", "baz"], 3, "success", None, None),
        # Success length 2
        (["ab", "cd", "ef"], 2, "success", None, None),
        # Success length 5
        (["hello", "world", "tests"], 5, "success", None, None),
        # Basic violations
        (
            ["foo", "bar", "bazz", "foobar"],
            3,
            "failure",
            ["bazz", "foobar"],
            "Found 2 row(s) where 'col1' length is not equal to 3.",
        ),
        # All violations
        (
            ["a", "ab", "abc"],
            5,
            "failure",
            ["a", "ab", "abc"],
            "Found 3 row(s) where 'col1' length is not equal to 5.",
        ),
        # Mixed violations
        (
            ["test", "testing", "t"],
            4,
            "failure",
            ["testing", "t"],
            "Found 2 row(s) where 'col1' length is not equal to 4.",
        ),
        # Single char success
        (["a", "b", "c"], 1, "success", None, None),
        # Single char violations
        (
            ["a", "ab", "abc"],
            1,
            "failure",
            ["ab", "abc"],
            "Found 2 row(s) where 'col1' length is not equal to 1.",
        ),
        # Empty string success
        (["", "", ""], 0, "success", None, None),
        # Empty string violations length 0
        (
            ["", "a", "ab"],
            0,
            "failure",
            ["a", "ab"],
            "Found 2 row(s) where 'col1' length is not equal to 0.",
        ),
        # Empty string violations length 1
        (
            ["", "a", "ab"],
            1,
            "failure",
            ["", "ab"],
            "Found 2 row(s) where 'col1' length is not equal to 1.",
        ),
        # Whitespace length 1
        (
            ["   ", "  ", " "],
            1,
            "failure",
            ["   ", "  "],
            "Found 2 row(s) where 'col1' length is not equal to 1.",
        ),
        # Whitespace length 3
        (
            ["   ", "  ", " "],
            3,
            "failure",
            ["  ", " "],
            "Found 2 row(s) where 'col1' length is not equal to 3.",
        ),
        # Whitespace in text
        (["a b", "c d", "e f"], 3, "success", None, None),
        # Whitespace mixed
        (
            [" a ", "  a  ", "a"],
            3,
            "failure",
            ["  a  ", "a"],
            "Found 2 row(s) where 'col1' length is not equal to 3.",
        ),
        # Special chars success
        (["@@@", "!!!", "###"], 3, "success", None, None),
        # Special chars in text
        (["test@", "user!", "data#"], 5, "success", None, None),
        # Special chars violations
        (
            ["@", "!!", "###"],
            2,
            "failure",
            ["@", "###"],
            "Found 2 row(s) where 'col1' length is not equal to 2.",
        ),
        # Numbers success
        (["123", "456", "789"], 3, "success", None, None),
        # Numbers versions
        (["v1.0", "v2.0", "v3.0"], 4, "success", None, None),
        # Numbers violations
        (
            ["1", "12", "123"],
            2,
            "failure",
            ["1", "123"],
            "Found 2 row(s) where 'col1' length is not equal to 2.",
        ),
        # Length 10 success
        (["a" * 10, "b" * 10, "c" * 10], 10, "success", None, None),
        # Length 20 success
        (["a" * 20, "b" * 20, "c" * 20], 20, "success", None, None),
        # Length 10 violation
        (
            ["a" * 10, "b" * 20, "c" * 10],
            10,
            "failure",
            ["b" * 20],
            "Found 1 row(s) where 'col1' length is not equal to 10.",
        ),
        # Long strings success
        (["a" * 100, "b" * 100, "c" * 100], 100, "success", None, None),
        # Long strings violations
        (
            ["a" * 100, "b" * 99, "c" * 101],
            100,
            "failure",
            ["b" * 99, "c" * 101],
            "Found 2 row(s) where 'col1' length is not equal to 100.",
        ),
        # Mixed violations
        (
            ["short", "exactly3", "way too long"],
            8,
            "failure",
            ["short", "way too long"],
            "Found 2 row(s) where 'col1' length is not equal to 8.",
        ),
    ],
    ids=[
        "basic_success",
        "success_length_2",
        "success_length_5",
        "basic_violations",
        "all_violations",
        "mixed_violations",
        "single_char_success",
        "single_char_violations",
        "empty_string_success",
        "empty_string_violations_length_0",
        "empty_string_violations_length_1",
        "whitespace_length_1",
        "whitespace_length_3",
        "whitespace_in_text",
        "whitespace_mixed",
        "special_chars_success",
        "special_chars_in_text",
        "special_chars_violations",
        "numbers_success",
        "numbers_versions",
        "numbers_violations",
        "length_10_success",
        "length_20_success",
        "length_10_violation",
        "long_strings_success",
        "long_strings_violations",
        "mixed_short_and_long",
    ],
)
def test_expectation_basic_scenarios(
    dataframe_factory, data, length, expected_result, expected_violations, expected_message
):
    """
    Test the expectation for various scenarios across pandas and PySpark DataFrames.
    Tests both direct expectation validation and suite-based validation.
    Covers: success cases, violations, different length values, single character strings,
    empty strings, whitespace handling, special characters, numbers in strings,
    long strings, and mixed violations.
    """
    df_lib, make_df = dataframe_factory

    data_frame = make_df({"col1": (data, "string")})

    # Test 1: Direct expectation validation
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationStringLengthEquals",
        column_name="col1",
        length=length,
    )

    result = expectation.validate(data_frame=data_frame)

    if expected_result == "success":
        assert str(result) == str(
            DataFrameExpectationSuccessMessage(expectation_name="ExpectationStringLengthEquals")
        ), f"Expected success message but got: {result}"
    else:  # failure
        violations_df = make_df({"col1": (expected_violations, "string")})
        expected_failure_message = DataFrameExpectationFailureMessage(
            expectation_str=str(expectation),
            data_frame_type=df_lib,
            violations_data_frame=violations_df,
            message=expected_message,
            limit_violations=5,
        )
        assert str(result) == str(expected_failure_message), (
            f"Expected failure message but got: {result}"
        )

    # Test 2: Suite-based validation
    expectations_suite = DataFrameExpectationsSuite().expect_string_length_equals(
        column_name="col1", length=length
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


def test_column_missing_error(dataframe_factory):
    """Test that an error is raised when the specified column is missing."""
    df_lib, make_df = dataframe_factory
    expected_message = "Column 'col1' does not exist in the DataFrame."

    data_frame = make_df({"col2": (["foo", "bar", "baz"], "string")})

    # Test 1: Direct expectation validation
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationStringLengthEquals",
        column_name="col1",
        length=3,
    )
    result = expectation.validate(data_frame=data_frame)
    expected_failure = DataFrameExpectationFailureMessage(
        expectation_str=str(expectation),
        data_frame_type=df_lib,
        message=expected_message,
    )
    assert str(result) == str(expected_failure), f"Expected failure message but got: {result}"

    # Test 2: Suite-based validation
    expectations_suite = DataFrameExpectationsSuite().expect_string_length_equals(
        column_name="col1", length=3
    )
    with pytest.raises(DataFrameExpectationsSuiteFailure):
        expectations_suite.build().run(data_frame=data_frame)


def test_large_dataset_performance(dataframe_factory):
    """Test the expectation with a larger dataset to ensure performance."""
    df_lib, make_df = dataframe_factory

    # Create a larger dataset with 10,000 rows all with length 10
    large_data = ["a" * 10 for _ in range(10000)]
    data_frame = make_df({"col1": (large_data, "string")})

    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationStringLengthEquals",
        column_name="col1",
        length=10,
    )

    result = expectation.validate(data_frame=data_frame)
    # Should succeed as all values have length 10
    assert isinstance(result, DataFrameExpectationSuccessMessage), (
        f"Large dataset test failed: expected success but got {type(result)}"
    )
