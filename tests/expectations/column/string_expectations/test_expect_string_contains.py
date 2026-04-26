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
        expectation_name="ExpectationStringContains",
        column_name="col1",
        substring="foo",
    )
    assert expectation.get_expectation_name() == "ExpectationStringContains", (
        f"Expected 'ExpectationStringContains' but got: {expectation.get_expectation_name()}"
    )


@pytest.mark.parametrize(
    "data, substring, expected_result, expected_violations, expected_message",
    [
        # Basic success scenarios
        (["foobar", "foo123", "barfoo"], "foo", "success", None, None),
        (["hello", "hello world", "say hello"], "hello", "success", None, None),
        (["test123", "test456", "testing"], "test", "success", None, None),
        # Basic violation scenarios
        (
            ["foobar", "bar", "baz"],
            "foo",
            "failure",
            ["bar", "baz"],
            "Found 2 row(s) where 'col1' does not contain 'foo'.",
        ),
        (
            ["apple", "banana", "cherry"],
            "xyz",
            "failure",
            ["apple", "banana", "cherry"],
            "Found 3 row(s) where 'col1' does not contain 'xyz'.",
        ),
        (
            ["test", "testing", "best"],
            "testing",
            "failure",
            ["test", "best"],
            "Found 2 row(s) where 'col1' does not contain 'testing'.",
        ),
        # Case sensitivity scenarios
        (
            ["Foo", "FOO", "fOo"],
            "foo",
            "failure",
            ["Foo", "FOO", "fOo"],
            "Found 3 row(s) where 'col1' does not contain 'foo'.",
        ),
        (["foo", "foobar", "barfoo"], "foo", "success", None, None),
        (
            ["Hello", "HELLO", "hello"],
            "Hello",
            "failure",
            ["HELLO", "hello"],
            "Found 2 row(s) where 'col1' does not contain 'Hello'.",
        ),
        # Substring position scenarios
        (["foobar", "foobaz", "foo"], "foo", "success", None, None),
        (["barfoo", "bazfoo", "foo"], "foo", "success", None, None),
        (["barfoobar", "bazfoobaz"], "foo", "success", None, None),
        # Single character scenarios
        (["a", "ab", "abc"], "a", "success", None, None),
        (
            ["b", "c", "d"],
            "a",
            "failure",
            ["b", "c", "d"],
            "Found 3 row(s) where 'col1' does not contain 'a'.",
        ),
        (
            ["", "text", "more"],
            "text",
            "failure",
            ["", "more"],
            "Found 2 row(s) where 'col1' does not contain 'text'.",
        ),
        # Whitespace handling scenarios
        (["foo bar", "foo  bar", "foobar"], "foo", "success", None, None),
        ([" foo", "foo ", " foo "], "foo", "success", None, None),
        (
            ["foo bar", "bar", "baz"],
            "foo bar",
            "failure",
            ["bar", "baz"],
            "Found 2 row(s) where 'col1' does not contain 'foo bar'.",
        ),
        # Special character scenarios
        (["test@email.com", "user@domain.org"], "@", "success", None, None),
        (
            ["hello!", "world!", "test"],
            "!",
            "failure",
            ["test"],
            "Found 1 row(s) where 'col1' does not contain '!'.",
        ),
        (["path/to/file", "another/path"], "/", "success", None, None),
        (
            ["#tag1", "#tag2", "notag"],
            "#",
            "failure",
            ["notag"],
            "Found 1 row(s) where 'col1' does not contain '#'.",
        ),
        # Number scenarios
        (
            ["test123", "test456", "test"],
            "123",
            "failure",
            ["test456", "test"],
            "Found 2 row(s) where 'col1' does not contain '123'.",
        ),
        (
            ["v1.0.0", "v2.0.0", "v1.1.0"],
            "1.",
            "failure",
            ["v2.0.0"],
            "Found 1 row(s) where 'col1' does not contain '1.'.",
        ),
        # Multiple occurrence and long string scenarios
        (["foofoo", "foobarfoo", "foo"], "foo", "success", None, None),
        (
            ["a" * 100 + "foo" + "b" * 100, "c" * 200],
            "foo",
            "failure",
            ["c" * 200],
            "Found 1 row(s) where 'col1' does not contain 'foo'.",
        ),
        (["football", "foolish", "foo"], "foo", "success", None, None),
        (
            ["food", "foot", "bar"],
            "foo",
            "failure",
            ["bar"],
            "Found 1 row(s) where 'col1' does not contain 'foo'.",
        ),
    ],
    ids=[
        "basic_success",
        "success_hello",
        "success_test",
        "basic_violations",
        "all_violations",
        "partial_match_violations",
        "case_sensitive_all_fail",
        "case_sensitive_mixed",
        "case_hello_violations",
        "position_start",
        "position_end",
        "position_middle",
        "single_char_success",
        "single_char_violations",
        "empty_string_violations",
        "whitespace_in_data",
        "whitespace_around",
        "whitespace_in_substring",
        "special_at_sign",
        "special_exclamation",
        "special_slash",
        "special_hash",
        "numbers_123",
        "numbers_version",
        "multiple_occurrences",
        "long_strings",
        "partial_word_success",
        "partial_word_violation",
    ],
)
def test_expectation_basic_scenarios(
    dataframe_factory, data, substring, expected_result, expected_violations, expected_message
):
    """
    Test the expectation for various scenarios across pandas and PySpark DataFrames.
    Tests both direct expectation validation and suite-based validation.
    """
    df_lib, make_df = dataframe_factory

    data_frame = make_df({"col1": (data, "string")})

    # Test 1: Direct expectation validation
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationStringContains",
        column_name="col1",
        substring=substring,
    )

    result = expectation.validate(data_frame=data_frame)

    if expected_result == "success":
        assert str(result) == str(
            DataFrameExpectationSuccessMessage(expectation_name="ExpectationStringContains")
        ), f"Expected success message but got: {result}"
    else:
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
    expectations_suite = DataFrameExpectationsSuite().expect_string_contains(
        column_name="col1", substring=substring
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

    data_frame = make_df({"col2": (["foobar", "foo123", "barfoo"], "string")})

    # Test 1: Direct expectation validation
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationStringContains",
        column_name="col1",
        substring="foo",
    )
    result = expectation.validate(data_frame=data_frame)
    expected_failure = DataFrameExpectationFailureMessage(
        expectation_str=str(expectation),
        data_frame_type=df_lib,
        message=expected_message,
    )
    assert str(result) == str(expected_failure), f"Expected failure message but got: {result}"

    # Test 2: Suite-based validation
    expectations_suite = DataFrameExpectationsSuite().expect_string_contains(
        column_name="col1", substring="foo"
    )
    with pytest.raises(DataFrameExpectationsSuiteFailure):
        expectations_suite.build().run(data_frame=data_frame)


def test_large_dataset_performance(dataframe_factory):
    """Test the expectation with a larger dataset to ensure performance."""
    df_lib, make_df = dataframe_factory

    # Create a larger dataset with 10,000 rows containing "test"
    large_data = [f"test_{i}" for i in range(10000)]
    data_frame = make_df({"col1": (large_data, "string")})

    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationStringContains",
        column_name="col1",
        substring="test",
    )

    result = expectation.validate(data_frame=data_frame)
    # Should succeed as all values contain "test"
    assert isinstance(result, DataFrameExpectationSuccessMessage), (
        f"Large dataset test failed: expected success but got {type(result)}"
    )
