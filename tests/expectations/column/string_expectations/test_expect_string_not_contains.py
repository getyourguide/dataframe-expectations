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
        expectation_name="ExpectationStringNotContains",
        column_name="col1",
        substring="foo",
    )
    assert expectation.get_expectation_name() == "ExpectationStringNotContains", (
        f"Expected 'ExpectationStringNotContains' but got: {expectation.get_expectation_name()}"
    )


@pytest.mark.parametrize(
    "data, substring, expected_result, expected_violations",
    [
        # Basic success
        (["bar", "baz", "qux"], "foo", "success", None),
        # Success no match
        (["test", "demo", "exam"], "xyz", "success", None),
        # Success different substring
        (["hello", "world", "python"], "java", "success", None),
        # Basic violations
        (["foobar", "bar", "foo"], "foo", "failure", ["foobar", "foo"]),
        # All violations
        (
            ["testing", "test", "attest"],
            "test",
            "failure",
            ["testing", "test", "attest"],
        ),
        # Mixed violations
        (["good", "bad", "badge"], "bad", "failure", ["bad", "badge"]),
        # Substring at beginning
        (
            ["prefix_test", "prefix_demo", "other"],
            "prefix",
            "failure",
            ["prefix_test", "prefix_demo"],
        ),
        # No substring at beginning
        (["no_match", "also_no", "nope"], "prefix", "success", None),
        # Substring at end
        (
            ["test_suffix", "demo_suffix", "other"],
            "suffix",
            "failure",
            ["test_suffix", "demo_suffix"],
        ),
        # No substring at end
        (["no_match", "also_no", "nope"], "suffix", "success", None),
        # Substring in middle
        (
            ["pre_mid_post", "another_mid_test", "nomatch"],
            "mid",
            "failure",
            ["pre_mid_post", "another_mid_test"],
        ),
        # No substring in middle
        (["no_match", "also_no", "nope"], "mid", "success", None),
        # Case sensitive success
        (["FOO", "Foo", "fOo"], "foo", "success", None),
        # Case sensitive violations
        (["foo", "FOO", "test"], "FOO", "failure", ["FOO"]),
        # Empty string success
        (["", "", ""], "foo", "success", None),
        # Empty string with violation
        (["", "foo", ""], "foo", "failure", ["foo"]),
        # Whitespace only success
        (["   ", "  ", " "], "test", "success", None),
        # Whitespace in text violations
        (
            ["test with spaces", "test", "no match"],
            "test",
            "failure",
            ["test with spaces", "test"],
        ),
        # Whitespace around violations
        (
            ["   test   ", "test", "clean"],
            "test",
            "failure",
            ["   test   ", "test"],
        ),
        # Special char at violations
        (
            ["test@email", "user@domain", "plain"],
            "@",
            "failure",
            ["test@email", "user@domain"],
        ),
        # Special char at success
        (["no-match", "also-no", "nope"], "@", "success", None),
        # Special char hash violations
        (
            ["test#tag", "demo#hash", "plain"],
            "#",
            "failure",
            ["test#tag", "demo#hash"],
        ),
        # Numbers violations
        (
            ["version123", "test456", "plain"],
            "123",
            "failure",
            ["version123"],
        ),
        # Numbers no match
        (["v1.0", "v2.1", "v3.5"], "123", "success", None),
        # Numbers exact match
        (["test", "demo", "123"], "123", "failure", ["123"]),
        # Single char success
        (["a", "b", "c"], "x", "success", None),
        # Single char violation
        (["a", "b", "c"], "a", "failure", ["a"]),
        # Long string success
        (["a" * 100, "b" * 100, "c" * 100], "x", "success", None),
        # Long string with substring
        (
            ["a" * 50 + "test" + "b" * 50, "clean" * 20, "other"],
            "test",
            "failure",
            ["a" * 50 + "test" + "b" * 50],
        ),
        # Exact match
        (["test", "demo", "exam"], "test", "failure", ["test"]),
        # No exact match
        (["test", "demo", "exam"], "testing", "success", None),
    ],
    ids=[
        "basic_success",
        "success_no_match",
        "success_different_substring",
        "basic_violations",
        "all_violations",
        "mixed_violations",
        "substring_at_beginning",
        "no_substring_at_beginning",
        "substring_at_end",
        "no_substring_at_end",
        "substring_in_middle",
        "no_substring_in_middle",
        "case_sensitive_success",
        "case_sensitive_violations",
        "empty_string_success",
        "empty_string_with_violation",
        "whitespace_only_success",
        "whitespace_in_text_violations",
        "whitespace_around_violations",
        "special_char_at_violations",
        "special_char_at_success",
        "special_char_hash_violations",
        "numbers_violations",
        "numbers_no_match",
        "numbers_exact_match",
        "single_char_success",
        "single_char_violation",
        "long_string_success",
        "long_string_with_substring",
        "exact_match",
        "no_exact_match",
    ],
)
def test_expectation_basic_scenarios(
    dataframe_factory, data, substring, expected_result, expected_violations
):
    """Test various scenarios for ExpectationStringNotContains expectation."""
    df_lib, make_df = dataframe_factory

    data_frame = make_df({"col1": (data, "string")})

    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationStringNotContains",
        column_name="col1",
        substring=substring,
    )

    result = expectation.validate(data_frame=data_frame)

    if expected_result == "success":
        assert str(result) == str(
            DataFrameExpectationSuccessMessage(expectation_name="ExpectationStringNotContains")
        ), f"Expected success message but got: {result}"

        # Also test with suite
        expectations_suite = DataFrameExpectationsSuite().expect_string_not_contains(
            column_name="col1", substring=substring
        )
        suite_result = expectations_suite.build().run(data_frame=data_frame)
        assert suite_result is not None, "Expected SuiteExecutionResult"
        assert isinstance(suite_result, SuiteExecutionResult), (
            "Result should be SuiteExecutionResult"
        )
        assert suite_result.success, "Expected all expectations to pass"
        assert suite_result.total_passed == 1, "Expected 1 passed expectation"
        assert suite_result.total_failed == 0, "Expected 0 failed expectations"
    else:  # failure
        violations_df = make_df({"col1": (expected_violations, "string")})
        expected_message = (
            f"Found {len(expected_violations)} row(s) where 'col1' contains '{substring}'."
        )

        assert str(result) == str(
            DataFrameExpectationFailureMessage(
                expectation_str=str(expectation),
                data_frame_type=df_lib,
                violations_data_frame=violations_df,
                message=expected_message,
                limit_violations=5,
            )
        ), f"Expected failure message but got: {result}"

        # Also test with suite
        expectations_suite = DataFrameExpectationsSuite().expect_string_not_contains(
            column_name="col1", substring=substring
        )
        with pytest.raises(DataFrameExpectationsSuiteFailure):
            expectations_suite.build().run(data_frame=data_frame)


def test_column_missing_error(dataframe_factory):
    """Test that missing column raises appropriate error."""
    df_lib, make_df = dataframe_factory

    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationStringNotContains",
        column_name="col1",
        substring="foo",
    )

    data_frame = make_df({"col2": (["bar", "baz", "qux"], "string")})
    result = expectation.validate(data_frame=data_frame)

    expected_failure_message = DataFrameExpectationFailureMessage(
        expectation_str=str(expectation),
        data_frame_type=df_lib,
        message="Column 'col1' does not exist in the DataFrame.",
    )
    assert str(result) == str(expected_failure_message), (
        f"Expected failure message but got: {result}"
    )

    # Also test with suite
    expectations_suite = DataFrameExpectationsSuite().expect_string_not_contains(
        column_name="col1", substring="foo"
    )
    with pytest.raises(DataFrameExpectationsSuiteFailure):
        expectations_suite.build().run(data_frame=data_frame)


def test_large_dataset_performance(dataframe_factory):
    """Test the expectation with a larger dataset to ensure performance."""
    df_lib, make_df = dataframe_factory

    # Create a larger dataset with 10,000 rows none containing "test"
    large_data = ["safe_value_" + str(i) for i in range(10000)]
    data_frame = make_df({"col1": (large_data, "string")})

    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationStringNotContains",
        column_name="col1",
        substring="test",
    )

    result = expectation.validate(data_frame=data_frame)
    # Should succeed as no values contain "test"
    assert isinstance(result, DataFrameExpectationSuccessMessage), (
        f"Large dataset test failed: expected success but got {type(result)}"
    )
