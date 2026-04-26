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
        expectation_name="ExpectationStringLengthGreaterThan",
        column_name="col1",
        length=3,
    )
    assert expectation.get_expectation_name() == "ExpectationStringLengthGreaterThan", (
        f"Expected 'ExpectationStringLengthGreaterThan' but got: {expectation.get_expectation_name()}"
    )


@pytest.mark.parametrize(
    "data, length, expected_result, expected_violations, expected_message",
    [
        # Basic success
        (["foobar", "bazz", "hello"], 3, "success", None, None),
        # Success length 2
        (["abc", "abcd", "abcde"], 2, "success", None, None),
        # Success length 4
        (["hello", "world", "testing"], 4, "success", None, None),
        # Basic violations
        (
            ["foo", "bar", "bazzz"],
            3,
            "failure",
            ["foo", "bar"],
            "Found 2 row(s) where 'col1' length is not greater than 3.",
        ),
        # All violations
        (
            ["a", "ab", "abc"],
            5,
            "failure",
            ["a", "ab", "abc"],
            "Found 3 row(s) where 'col1' length is not greater than 5.",
        ),
        # Mixed violations
        (
            ["test", "testing", "t"],
            4,
            "failure",
            ["test", "t"],
            "Found 2 row(s) where 'col1' length is not greater than 4.",
        ),
        # Boundary exact violation
        (
            ["abc", "abcd", "abcde"],
            3,
            "failure",
            ["abc"],
            "Found 1 row(s) where 'col1' length is not greater than 3.",
        ),
        # Boundary success
        (["abcd", "abcde", "abcdef"], 3, "success", None, None),
        # Zero length success
        (["a", "ab", "abc"], 0, "success", None, None),
        # Zero length violation
        (
            ["", "a", "ab"],
            0,
            "failure",
            [""],
            "Found 1 row(s) where 'col1' length is not greater than 0.",
        ),
        # Single char success
        (["ab", "abc", "abcd"], 1, "success", None, None),
        # Single char violations
        (
            ["a", "b", "cd"],
            1,
            "failure",
            ["a", "b"],
            "Found 2 row(s) where 'col1' length is not greater than 1.",
        ),
        # Empty strings
        (
            ["", "a", "ab"],
            1,
            "failure",
            ["", "a"],
            "Found 2 row(s) where 'col1' length is not greater than 1.",
        ),
        # Whitespace success
        (["    ", "   ", "   "], 2, "success", None, None),
        # Whitespace in text
        (["a b c", "a b ", "a  b"], 3, "success", None, None),
        # Whitespace violations
        (
            [" a ", "  a  ", "a"],
            3,
            "failure",
            [" a ", "a"],
            "Found 2 row(s) where 'col1' length is not greater than 3.",
        ),
        # Special chars success
        (["@@@@", "!!!!", "####"], 3, "success", None, None),
        # Special chars in text
        (["test@@", "user!!", "data##"], 5, "success", None, None),
        # Special chars violations
        (
            ["@", "!!", "###"],
            2,
            "failure",
            ["@", "!!"],
            "Found 2 row(s) where 'col1' length is not greater than 2.",
        ),
        # Numbers success
        (["1234", "5678", "9012"], 3, "success", None, None),
        # Numbers versions
        (["v1.0.0", "v2.0.0", "v3.0.0"], 5, "success", None, None),
        # Numbers violations
        (
            ["1", "12", "123"],
            2,
            "failure",
            ["1", "12"],
            "Found 2 row(s) where 'col1' length is not greater than 2.",
        ),
        # Length 10 success
        (["a" * 11, "b" * 12, "c" * 13], 10, "success", None, None),
        # Length 20 success
        (["a" * 21, "b" * 22, "c" * 23], 20, "success", None, None),
        # Length 10 violation
        (
            ["a" * 10, "b" * 11, "c" * 12],
            10,
            "failure",
            ["a" * 10],
            "Found 1 row(s) where 'col1' length is not greater than 10.",
        ),
        # Long strings success
        (["a" * 101, "b" * 102, "c" * 103], 100, "success", None, None),
        # Long strings violation
        (
            ["a" * 100, "b" * 101, "c" * 102],
            100,
            "failure",
            ["a" * 100],
            "Found 1 row(s) where 'col1' length is not greater than 100.",
        ),
        # Mixed violations
        (
            ["short", "exactly8", "much longer string"],
            8,
            "failure",
            ["short", "exactly8"],
            "Found 2 row(s) where 'col1' length is not greater than 8.",
        ),
    ],
    ids=[
        "basic_success",
        "success_length_2",
        "success_length_4",
        "basic_violations",
        "all_violations",
        "mixed_violations",
        "boundary_exact_violation",
        "boundary_success",
        "zero_length_success",
        "zero_length_violation",
        "single_char_success",
        "single_char_violations",
        "empty_string_violations",
        "whitespace_success",
        "whitespace_in_text",
        "whitespace_violations",
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
        "long_strings_violation",
        "mixed_short_and_exact",
    ],
)
def test_expectation_basic_scenarios(
    dataframe_factory, data, length, expected_result, expected_violations, expected_message
):
    """
    Test the expectation for various scenarios across pandas and PySpark DataFrames.
    Tests both direct expectation validation and suite-based validation.
    Covers: success cases, violations, boundary conditions (exact length),
    zero length threshold, single character strings, empty strings,
    whitespace handling, special characters, numbers in strings, long strings,
    and mixed violations (both short and exact length).
    """
    df_lib, make_df = dataframe_factory

    data_frame = make_df({"col1": (data, "string")})

    # Test 1: Direct expectation validation
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationStringLengthGreaterThan",
        column_name="col1",
        length=length,
    )

    result = expectation.validate(data_frame=data_frame)

    if expected_result == "success":
        assert str(result) == str(
            DataFrameExpectationSuccessMessage(
                expectation_name="ExpectationStringLengthGreaterThan"
            )
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
    expectations_suite = DataFrameExpectationsSuite().expect_string_length_greater_than(
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

    data_frame = make_df({"col2": (["foobar", "bazz", "hello"], "string")})

    # Test 1: Direct expectation validation
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationStringLengthGreaterThan",
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
    expectations_suite = DataFrameExpectationsSuite().expect_string_length_greater_than(
        column_name="col1", length=3
    )
    with pytest.raises(DataFrameExpectationsSuiteFailure):
        expectations_suite.build().run(data_frame=data_frame)


def test_large_dataset_performance(dataframe_factory):
    """Test the expectation with a larger dataset to ensure performance."""
    df_lib, make_df = dataframe_factory

    # Create a larger dataset with 10,000 rows all with length > 10
    large_data = ["a" * 15 for _ in range(10000)]
    data_frame = make_df({"col1": (large_data, "string")})

    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationStringLengthGreaterThan",
        column_name="col1",
        length=10,
    )

    result = expectation.validate(data_frame=data_frame)
    # Should succeed as all values have length > 10
    assert isinstance(result, DataFrameExpectationSuccessMessage), (
        f"Large dataset test failed: expected success but got {type(result)}"
    )
