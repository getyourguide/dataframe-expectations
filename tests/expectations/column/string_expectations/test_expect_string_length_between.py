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
        expectation_name="ExpectationStringLengthBetween",
        column_name="col1",
        min_length=3,
        max_length=6,
    )
    assert expectation.get_expectation_name() == "ExpectationStringLengthBetween", (
        f"Expected 'ExpectationStringLengthBetween' but got: {expectation.get_expectation_name()}"
    )


@pytest.mark.parametrize(
    "data, min_length, max_length, expected_result, expected_violations, expected_message",
    [
        # Basic success scenarios
        (["foo", "bazz", "hello", "foobar"], 3, 6, "success", None, None),
        (["ab", "abc", "abcd"], 2, 4, "success", None, None),
        (["test", "data", "valid"], 4, 5, "success", None, None),
        # Basic violation scenarios
        (
            ["fo", "bazz", "hellothere", "foobar"],
            3,
            6,
            "failure",
            ["fo", "hellothere"],
            "Found 2 row(s) where 'col1' length is not between 3 and 6.",
        ),
        (
            ["a", "ab", "abc"],
            5,
            10,
            "failure",
            ["a", "ab", "abc"],
            "Found 3 row(s) where 'col1' length is not between 5 and 10.",
        ),
        (
            ["test", "testing", "t"],
            2,
            4,
            "failure",
            ["testing", "t"],
            "Found 2 row(s) where 'col1' length is not between 2 and 4.",
        ),
        # Boundary conditions
        (["abc", "abcd", "abcde"], 3, 5, "success", None, None),
        (
            ["ab", "abc", "abcd"],
            3,
            5,
            "failure",
            ["ab"],
            "Found 1 row(s) where 'col1' length is not between 3 and 5.",
        ),
        (["abc", "abcd", "abcde"], 3, 5, "success", None, None),
        (
            ["abc", "abcd", "abcdef"],
            3,
            5,
            "failure",
            ["abcdef"],
            "Found 1 row(s) where 'col1' length is not between 3 and 5.",
        ),
        (["abc", "def", "ghi"], 3, 3, "success", None, None),
        (
            ["ab", "abc", "abcd"],
            3,
            3,
            "failure",
            ["ab", "abcd"],
            "Found 2 row(s) where 'col1' length is not between 3 and 3.",
        ),
        (["a", "b", "c"], 1, 1, "success", None, None),
        (
            ["a", "ab", "abc"],
            1,
            1,
            "failure",
            ["ab", "abc"],
            "Found 2 row(s) where 'col1' length is not between 1 and 1.",
        ),
        # Empty string and whitespace scenarios
        (
            ["", "a", "ab"],
            1,
            3,
            "failure",
            [""],
            "Found 1 row(s) where 'col1' length is not between 1 and 3.",
        ),
        (["", "a", "ab"], 0, 3, "success", None, None),
        (["   ", "  ", " "], 1, 3, "success", None, None),
        (["a b", "a  b", "a   b"], 3, 5, "success", None, None),
        (
            [" a ", "  a  ", "a"],
            4,
            6,
            "failure",
            [" a ", "a"],
            "Found 2 row(s) where 'col1' length is not between 4 and 6.",
        ),
        # Special characters and number scenarios
        (["@@@", "!!!", "###"], 3, 3, "success", None, None),
        (["test@", "user!", "admin#"], 5, 6, "success", None, None),
        (
            ["@", "!!", "###"],
            2,
            2,
            "failure",
            ["@", "###"],
            "Found 2 row(s) where 'col1' length is not between 2 and 2.",
        ),
        (["123", "456", "789"], 3, 3, "success", None, None),
        (["v1.0", "v2.0", "v10.0"], 4, 6, "success", None, None),
        (
            ["1", "12", "123456"],
            2,
            4,
            "failure",
            ["1", "123456"],
            "Found 2 row(s) where 'col1' length is not between 2 and 4.",
        ),
        # Long string and wide range scenarios
        (["a" * 100, "b" * 100, "c" * 100], 100, 100, "success", None, None),
        (
            ["a" * 50, "b" * 100, "c" * 150],
            100,
            100,
            "failure",
            ["a" * 50, "c" * 150],
            "Found 2 row(s) where 'col1' length is not between 100 and 100.",
        ),
        (["a", "ab", "a" * 50, "a" * 100], 1, 100, "success", None, None),
        (["", "a", "ab", "abc"], 0, 3, "success", None, None),
        (
            ["", "a", "abcd"],
            0,
            3,
            "failure",
            ["abcd"],
            "Found 1 row(s) where 'col1' length is not between 0 and 3.",
        ),
    ],
    ids=[
        "basic_success",
        "success_2_4",
        "success_4_5",
        "basic_violations",
        "all_violations",
        "one_violation",
        "boundary_min_success",
        "boundary_min_violation",
        "boundary_max_success",
        "boundary_max_violation",
        "min_equals_max_success",
        "min_equals_max_violations",
        "single_char_success",
        "single_char_violations",
        "empty_string_violation",
        "empty_string_success",
        "whitespace_success",
        "whitespace_with_text",
        "whitespace_violations",
        "special_chars_success",
        "special_chars_in_text",
        "special_chars_violations",
        "numbers_success",
        "numbers_versions",
        "numbers_violations",
        "long_strings_success",
        "long_strings_violations",
        "wide_range",
        "zero_min_success",
        "zero_min_violation",
    ],
)
def test_expectation_basic_scenarios(
    dataframe_factory,
    data,
    min_length,
    max_length,
    expected_result,
    expected_violations,
    expected_message,
):
    """
    Test the expectation for various scenarios across pandas and PySpark DataFrames.
    Tests both direct expectation validation and suite-based validation.
    """
    df_lib, make_df = dataframe_factory

    data_frame = make_df({"col1": (data, "string")})

    # Test 1: Direct expectation validation
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationStringLengthBetween",
        column_name="col1",
        min_length=min_length,
        max_length=max_length,
    )

    result = expectation.validate(data_frame=data_frame)

    if expected_result == "success":
        assert str(result) == str(
            DataFrameExpectationSuccessMessage(expectation_name="ExpectationStringLengthBetween")
        ), f"Expected success message but got: {result}"
    else:
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
    expectations_suite = DataFrameExpectationsSuite().expect_string_length_between(
        column_name="col1", min_length=min_length, max_length=max_length
    )

    if expected_result == "success":
        result = expectations_suite.build().run(data_frame=data_frame)
        assert result is not None, "Expected SuiteExecutionResult"
        assert isinstance(result, SuiteExecutionResult), "Result should be SuiteExecutionResult"
        assert result.success, "Expected all expectations to pass"
        assert result.total_passed == 1, "Expected 1 passed expectation"
        assert result.total_failed == 0, "Expected 0 failed expectations"
    else:
        with pytest.raises(DataFrameExpectationsSuiteFailure):
            expectations_suite.build().run(data_frame=data_frame)


def test_column_missing_error(dataframe_factory):
    """Test that an error is raised when the specified column is missing."""
    df_lib, make_df = dataframe_factory
    expected_message = "Column 'col1' does not exist in the DataFrame."

    data_frame = make_df({"col2": (["foo", "bazz", "hello"], "string")})

    # Test 1: Direct expectation validation
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationStringLengthBetween",
        column_name="col1",
        min_length=3,
        max_length=6,
    )
    result = expectation.validate(data_frame=data_frame)
    expected_failure = DataFrameExpectationFailureMessage(
        expectation_str=str(expectation),
        data_frame_type=df_lib,
        message=expected_message,
    )
    assert str(result) == str(expected_failure), f"Expected failure message but got: {result}"

    # Test 2: Suite-based validation
    expectations_suite = DataFrameExpectationsSuite().expect_string_length_between(
        column_name="col1", min_length=3, max_length=6
    )
    with pytest.raises(DataFrameExpectationsSuiteFailure):
        expectations_suite.build().run(data_frame=data_frame)


def test_large_dataset_performance(dataframe_factory):
    """Test the expectation with a larger dataset to ensure performance."""
    df_lib, make_df = dataframe_factory

    # Create a larger dataset with 10,000 rows with lengths between 5 and 10
    large_data = [f"test_{i}" for i in range(10000)]
    data_frame = make_df({"col1": (large_data, "string")})

    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationStringLengthBetween",
        column_name="col1",
        min_length=5,
        max_length=15,
    )

    result = expectation.validate(data_frame=data_frame)
    # Should succeed as all values have lengths between 5 and 15
    assert isinstance(result, DataFrameExpectationSuccessMessage), (
        f"Large dataset test failed: expected success but got {type(result)}"
    )
