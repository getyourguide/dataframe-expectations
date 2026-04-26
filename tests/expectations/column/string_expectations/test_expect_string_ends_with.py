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
        expectation_name="ExpectationStringEndsWith",
        column_name="col1",
        suffix="bar",
    )
    assert expectation.get_expectation_name() == "ExpectationStringEndsWith", (
        f"Expected 'ExpectationStringEndsWith' but got: {expectation.get_expectation_name()}"
    )


@pytest.mark.parametrize(
    "data, suffix, expected_result, expected_violations, expected_message",
    [
        # Basic success scenarios
        (["foobar", "bar", "bazbar"], "bar", "success", None, None),
        (["hello", "say hello", "greet with hello"], "hello", "success", None, None),
        (["test", "mytest", "anothertest"], "test", "success", None, None),
        # Basic violation scenarios
        (
            ["foobar", "bar", "baz"],
            "bar",
            "failure",
            ["baz"],
            "Found 1 row(s) where 'col1' does not end with 'bar'.",
        ),
        (
            ["apple", "banana", "cherry"],
            "xyz",
            "failure",
            ["apple", "banana", "cherry"],
            "Found 3 row(s) where 'col1' does not end with 'xyz'.",
        ),
        (
            ["testing", "test", "best"],
            "test",
            "failure",
            ["testing", "best"],
            "Found 2 row(s) where 'col1' does not end with 'test'.",
        ),
        # Case sensitivity scenarios
        (
            ["fooBar", "fooBAR", "foobar"],
            "bar",
            "failure",
            ["fooBar", "fooBAR"],
            "Found 2 row(s) where 'col1' does not end with 'bar'.",
        ),
        (["foobar", "bazbar", "bar"], "bar", "success", None, None),
        (
            ["testHello", "testHELLO", "testhello"],
            "hello",
            "failure",
            ["testHello", "testHELLO"],
            "Found 2 row(s) where 'col1' does not end with 'hello'.",
        ),
        # Exact match scenarios
        (
            ["bar", "foo", "baz"],
            "bar",
            "failure",
            ["foo", "baz"],
            "Found 2 row(s) where 'col1' does not end with 'bar'.",
        ),
        (["test", "test", "test"], "test", "success", None, None),
        # Single character suffix scenarios
        (["a", "ba", "cba"], "a", "success", None, None),
        (
            ["b", "c", "d"],
            "a",
            "failure",
            ["b", "c", "d"],
            "Found 3 row(s) where 'col1' does not end with 'a'.",
        ),
        (
            ["", "text", "more"],
            "text",
            "failure",
            ["", "more"],
            "Found 2 row(s) where 'col1' does not end with 'text'.",
        ),
        # Whitespace handling scenarios
        (["foo bar", "baz bar", "test bar"], "bar", "success", None, None),
        (["foo ", "bar ", "baz "], " ", "success", None, None),
        (
            ["foo bar", "bar", "baz"],
            " bar",
            "failure",
            ["bar", "baz"],
            "Found 2 row(s) where 'col1' does not end with ' bar'.",
        ),
        # Special character scenarios
        (["test@", "user@", "admin@"], "@", "success", None, None),
        (
            ["hello!", "world!", "test"],
            "!",
            "failure",
            ["test"],
            "Found 1 row(s) where 'col1' does not end with '!'.",
        ),
        (["path/to/", "another/"], "/", "success", None, None),
        (
            ["tag1#", "tag2#", "notag"],
            "#",
            "failure",
            ["notag"],
            "Found 1 row(s) where 'col1' does not end with '#'.",
        ),
        # Number scenarios
        (
            ["test123", "file123", "data456"],
            "123",
            "failure",
            ["data456"],
            "Found 1 row(s) where 'col1' does not end with '123'.",
        ),
        (
            ["v1.0", "v2.0", "v1.1"],
            ".0",
            "failure",
            ["v1.1"],
            "Found 1 row(s) where 'col1' does not end with '.0'.",
        ),
        # Long string and suffix-length scenarios
        (["a" * 100 + "bar", "b" * 100 + "bar"], "bar", "success", None, None),
        (
            ["a" * 100 + "foo", "b" * 100 + "bar"],
            "bar",
            "failure",
            ["a" * 100 + "foo"],
            "Found 1 row(s) where 'col1' does not end with 'bar'.",
        ),
        (
            ["foo", "ba", "b"],
            "foobar",
            "failure",
            ["foo", "ba", "b"],
            "Found 3 row(s) where 'col1' does not end with 'foobar'.",
        ),
        (
            ["barfoo", "foobar", "bazbar"],
            "foo",
            "failure",
            ["foobar", "bazbar"],
            "Found 2 row(s) where 'col1' does not end with 'foo'.",
        ),
    ],
    ids=[
        "basic_success",
        "success_hello",
        "success_test",
        "basic_violations",
        "all_violations",
        "partial_match_violations",
        "case_sensitive_violations",
        "case_sensitive_success",
        "case_hello_violations",
        "exact_match_partial",
        "exact_match_all",
        "single_char_success",
        "single_char_violations",
        "empty_string_violations",
        "whitespace_success",
        "whitespace_trailing",
        "whitespace_in_suffix",
        "special_at_sign",
        "special_exclamation",
        "special_slash",
        "special_hash",
        "numbers_123",
        "numbers_version",
        "long_strings_success",
        "long_strings_violation",
        "suffix_longer",
        "partial_not_at_end",
    ],
)
def test_expectation_basic_scenarios(
    dataframe_factory, data, suffix, expected_result, expected_violations, expected_message
):
    """
    Test the expectation for various scenarios across pandas and PySpark DataFrames.
    Tests both direct expectation validation and suite-based validation.
    """
    df_lib, make_df = dataframe_factory

    data_frame = make_df({"col1": (data, "string")})

    # Test 1: Direct expectation validation
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationStringEndsWith",
        column_name="col1",
        suffix=suffix,
    )

    result = expectation.validate(data_frame=data_frame)

    if expected_result == "success":
        assert str(result) == str(
            DataFrameExpectationSuccessMessage(expectation_name="ExpectationStringEndsWith")
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
    expectations_suite = DataFrameExpectationsSuite().expect_string_ends_with(
        column_name="col1", suffix=suffix
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

    data_frame = make_df({"col2": (["foobar", "bar", "bazbar"], "string")})

    # Test 1: Direct expectation validation
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationStringEndsWith",
        column_name="col1",
        suffix="bar",
    )
    result = expectation.validate(data_frame=data_frame)
    expected_failure = DataFrameExpectationFailureMessage(
        expectation_str=str(expectation),
        data_frame_type=df_lib,
        message=expected_message,
    )
    assert str(result) == str(expected_failure), f"Expected failure message but got: {result}"

    # Test 2: Suite-based validation
    expectations_suite = DataFrameExpectationsSuite().expect_string_ends_with(
        column_name="col1", suffix="bar"
    )
    with pytest.raises(DataFrameExpectationsSuiteFailure):
        expectations_suite.build().run(data_frame=data_frame)


def test_large_dataset_performance(dataframe_factory):
    """Test the expectation with a larger dataset to ensure performance."""
    df_lib, make_df = dataframe_factory

    # Create a larger dataset with 10,000 rows ending with "_test"
    large_data = [f"value_{i}_test" for i in range(10000)]
    data_frame = make_df({"col1": (large_data, "string")})

    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationStringEndsWith",
        column_name="col1",
        suffix="_test",
    )

    result = expectation.validate(data_frame=data_frame)
    # Should succeed as all values end with "_test"
    assert isinstance(result, DataFrameExpectationSuccessMessage), (
        f"Large dataset test failed: expected success but got {type(result)}"
    )
