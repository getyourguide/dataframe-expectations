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
        expectation_name="ExpectationStringLengthLessThan",
        column_name="col1",
        length=5,
    )
    assert expectation.get_expectation_name() == "ExpectationStringLengthLessThan", (
        f"Expected 'ExpectationStringLengthLessThan' but got: {expectation.get_expectation_name()}"
    )


@pytest.mark.parametrize(
    "data, length, expected_result, expected_violations, expected_message",
    [
        # Basic success
        (["foo", "bar", "baz"], 5, "success", None, None),
        # Success length 2
        (["ab", "cd", "ef"], 3, "success", None, None),
        # Success length 4
        (["abc", "def", "ghi"], 5, "success", None, None),
        # Basic violations
        (
            ["foobar", "bar", "bazbaz"],
            5,
            "failure",
            ["foobar", "bazbaz"],
            "Found 2 row(s) where 'col1' length is not less than 5.",
        ),
        # All violations
        (
            ["testing", "longer", "strings"],
            5,
            "failure",
            ["testing", "longer", "strings"],
            "Found 3 row(s) where 'col1' length is not less than 5.",
        ),
        # Mixed violations
        (
            ["ok", "fail", "good"],
            3,
            "failure",
            ["fail", "good"],
            "Found 2 row(s) where 'col1' length is not less than 3.",
        ),
        # Boundary exact violation
        (
            ["test", "exam", "demo"],
            4,
            "failure",
            ["test", "exam", "demo"],
            "Found 3 row(s) where 'col1' length is not less than 4.",
        ),
        # Boundary success
        (["tes", "exa", "dem"], 4, "success", None, None),
        # Zero length success
        (["", "", ""], 1, "success", None, None),
        # Zero length violation
        (
            ["a", "b", "c"],
            1,
            "failure",
            ["a", "b", "c"],
            "Found 3 row(s) where 'col1' length is not less than 1.",
        ),
        # Single char success
        (["a", "b", "c"], 2, "success", None, None),
        # Single char violations
        (
            ["a", "b", "c"],
            1,
            "failure",
            ["a", "b", "c"],
            "Found 3 row(s) where 'col1' length is not less than 1.",
        ),
        # Empty strings
        (["", "", ""], 5, "success", None, None),
        # Whitespace success
        (["  ", " ", ""], 3, "success", None, None),
        # Whitespace in text
        (["a b", "a", "ab"], 4, "success", None, None),
        # Whitespace violations
        (
            ["    ", "     ", "      "],
            3,
            "failure",
            ["    ", "     ", "      "],
            "Found 3 row(s) where 'col1' length is not less than 3.",
        ),
        # Special chars success
        (["@#$", "!^&", "*()"], 4, "success", None, None),
        # Special chars in text
        (["test@", "ex!m", "de#o"], 6, "success", None, None),
        # Special chars violations
        (
            ["@#$%^", "&*()_", "+=-[]"],
            4,
            "failure",
            ["@#$%^", "&*()_", "+=-[]"],
            "Found 3 row(s) where 'col1' length is not less than 4.",
        ),
        # Numbers success
        (["123", "456", "789"], 4, "success", None, None),
        # Numbers versions
        (["v1.0", "v2.1", "v3.5"], 5, "success", None, None),
        # Numbers violations
        (
            ["12345", "67890", "11111"],
            4,
            "failure",
            ["12345", "67890", "11111"],
            "Found 3 row(s) where 'col1' length is not less than 4.",
        ),
        # Length 10 success
        (["a" * 9, "b" * 8, "c" * 7], 11, "success", None, None),
        # Length 20 success
        (
            ["test " * 3, "demo " * 3, "exam " * 3],
            21,
            "success",
            None,
            None,
        ),
        # Length 10 violation
        (
            ["x" * 11, "y" * 12, "z" * 13],
            10,
            "failure",
            ["x" * 11, "y" * 12, "z" * 13],
            "Found 3 row(s) where 'col1' length is not less than 10.",
        ),
        # Long strings success
        (
            ["x" * 99, "y" * 98, "z" * 97],
            101,
            "success",
            None,
            None,
        ),
        # Long strings violation
        (
            ["x" * 101, "y" * 102, "z" * 103],
            100,
            "failure",
            ["x" * 101, "y" * 102, "z" * 103],
            "Found 3 row(s) where 'col1' length is not less than 100.",
        ),
        # Mixed violations
        (
            ["ab", "abcd", "abc"],
            4,
            "failure",
            ["abcd"],
            "Found 1 row(s) where 'col1' length is not less than 4.",
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
        "empty_string_success",
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
        expectation_name="ExpectationStringLengthLessThan",
        column_name="col1",
        length=length,
    )

    result = expectation.validate(data_frame=data_frame)

    if expected_result == "success":
        assert str(result) == str(
            DataFrameExpectationSuccessMessage(expectation_name="ExpectationStringLengthLessThan")
        ), f"Expected success message but got: {result}"
    else:  # failure
        expected_violations_df = make_df({"col1": (expected_violations, "string")})
        expected_failure_message = DataFrameExpectationFailureMessage(
            expectation_str=str(expectation),
            data_frame_type=df_lib,
            violations_data_frame=expected_violations_df,
            message=expected_message,
            limit_violations=5,
        )
        assert str(result) == str(expected_failure_message), (
            f"Expected failure message but got: {result}"
        )

    # Test 2: Suite-based validation
    expectations_suite = DataFrameExpectationsSuite().expect_string_length_less_than(
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
    """Test that missing column raises appropriate error."""
    df_lib, make_df = dataframe_factory
    expected_message = "Column 'col1' does not exist in the DataFrame."

    data_frame = make_df({"col2": (["foo", "bar", "baz"], "string")})

    # Test 1: Direct expectation validation
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationStringLengthLessThan",
        column_name="col1",
        length=5,
    )
    result = expectation.validate(data_frame=data_frame)
    expected_failure = DataFrameExpectationFailureMessage(
        expectation_str=str(expectation),
        data_frame_type=df_lib,
        message=expected_message,
    )
    assert str(result) == str(expected_failure), f"Expected failure message but got: {result}"

    # Test 2: Suite-based validation
    expectations_suite = DataFrameExpectationsSuite().expect_string_length_less_than(
        column_name="col1", length=5
    )
    with pytest.raises(DataFrameExpectationsSuiteFailure):
        expectations_suite.build().run(data_frame=data_frame)


def test_large_dataset_performance(dataframe_factory):
    """Test the expectation with a larger dataset to ensure performance."""
    df_lib, make_df = dataframe_factory

    # Create a larger dataset with 10,000 rows all with length < 10
    large_data = ["abc" * 2 for _ in range(10000)]
    data_frame = make_df({"col1": (large_data, "string")})

    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationStringLengthLessThan",
        column_name="col1",
        length=10,
    )

    result = expectation.validate(data_frame=data_frame)
    # Should succeed as all values have length < 10
    assert isinstance(result, DataFrameExpectationSuccessMessage), (
        f"Large dataset test failed: expected success but got {type(result)}"
    )
