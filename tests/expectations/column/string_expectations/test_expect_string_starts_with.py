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
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationStringStartsWith",
        column_name="col1",
        prefix="foo",
    )
    assert expectation.get_expectation_name() == "ExpectationStringStartsWith", (
        f"Expected 'ExpectationStringStartsWith' but got: {expectation.get_expectation_name()}"
    )


@pytest.mark.parametrize(
    "data, prefix, expected_result, expected_violations",
    [
        # Basic success
        (["foobar", "foo123", "foobaz"], "foo", "success", None),
        # Basic violations
        (
            ["bar", "baz", "qux"],
            "foo",
            "violations",
            ["bar", "baz", "qux"],
        ),
        # Exact match success
        (["foo", "foo", "foo"], "foo", "success", None),
        # Exact match mixed
        (
            ["foo", "bar", "baz"],
            "foo",
            "violations",
            ["bar", "baz"],
        ),
        # Empty strings
        (
            ["", "", ""],
            "foo",
            "violations",
            ["", "", ""],
        ),
        # Whitespace only violations
        (
            ["   ", "  ", " "],
            "foo",
            "violations",
            ["   ", "  ", " "],
        ),
        # Whitespace in text success
        (
            ["foo bar", "foo baz", "foo qux"],
            "foo",
            "success",
            None,
        ),
        # Whitespace at end violations
        (
            ["bar foo", "baz foo", "qux foo"],
            "foo",
            "violations",
            ["bar foo", "baz foo", "qux foo"],
        ),
        # Special char at violations
        (
            ["@foo", "#foo", "$foo"],
            "@",
            "violations",
            ["#foo", "$foo"],
        ),
        # Special char in prefix violations
        (
            ["foo@bar", "foo#baz", "foo$qux"],
            "foo@",
            "violations",
            ["foo#baz", "foo$qux"],
        ),
        # Numbers success
        (["foo1", "foo2", "foo3"], "foo", "success", None),
        # Numbers at start violations
        (
            ["1foo", "2foo", "3foo"],
            "foo",
            "violations",
            ["1foo", "2foo", "3foo"],
        ),
        # Long string success
        (
            ["foo" + "a" * 97, "foo" + "b" * 97, "foo" + "c" * 97],
            "foo",
            "success",
            None,
        ),
        # Long string violations
        (
            ["a" * 100, "b" * 100, "c" * 100],
            "foo",
            "violations",
            ["a" * 100, "b" * 100, "c" * 100],
        ),
        # Mixed violations
        (
            ["foobar", "bar", "foo123"],
            "foo",
            "violations",
            ["bar"],
        ),
    ],
    ids=[
        "basic_success",
        "basic_violations",
        "exact_match_success",
        "exact_match_mixed",
        "empty_string_violations",
        "whitespace_only_violations",
        "whitespace_in_text_success",
        "whitespace_at_end_violations",
        "special_char_at_violations",
        "special_char_in_prefix_violations",
        "numbers_success",
        "numbers_at_start_violations",
        "long_string_success",
        "long_string_violations",
        "mixed_violations",
    ],
)
def test_expectation_basic_scenarios(
    dataframe_factory, data, prefix, expected_result, expected_violations
):
    """Test various scenarios for ExpectationStringStartsWith expectation."""
    df_lib, make_df = dataframe_factory

    data_frame = make_df({"col1": (data, "string")})

    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationStringStartsWith",
        column_name="col1",
        prefix=prefix,
    )

    result = expectation.validate(data_frame=data_frame)

    if expected_result == "success":
        assert str(result) == str(
            DataFrameExpectationSuccessMessage(expectation_name="ExpectationStringStartsWith")
        ), f"Expected success message but got: {result}"

        # Also test with suite
        expectations_suite = DataFrameExpectationsSuite().expect_string_starts_with(
            column_name="col1", prefix=prefix
        )
        suite_result = expectations_suite.build().run(data_frame=data_frame)
        assert suite_result is not None, "Expected SuiteExecutionResult"
        assert isinstance(suite_result, SuiteExecutionResult), (
            "Result should be SuiteExecutionResult"
        )
        assert suite_result.success, "Expected all expectations to pass"
        assert suite_result.total_passed == 1, "Expected 1 passed expectation"
        assert suite_result.total_failed == 0, "Expected 0 failed expectations"
    else:  # violations
        violations_df = make_df({"col1": (expected_violations, "string")})
        expected_message = (
            f"Found {len(expected_violations)} row(s) where 'col1' does not start with '{prefix}'."
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
        expectations_suite = DataFrameExpectationsSuite().expect_string_starts_with(
            column_name="col1", prefix=prefix
        )
        with pytest.raises(DataFrameExpectationsSuiteFailure):
            expectations_suite.build().run(data_frame=data_frame)


def test_column_missing_error(dataframe_factory):
    """Test that missing column raises appropriate error."""
    df_lib, make_df = dataframe_factory

    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationStringStartsWith",
        column_name="col1",
        prefix="foo",
    )

    data_frame = make_df({"col2": (["foobar", "foo123", "foobaz"], "string")})
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
    expectations_suite = DataFrameExpectationsSuite().expect_string_starts_with(
        column_name="col1", prefix="foo"
    )
    with pytest.raises(DataFrameExpectationsSuiteFailure):
        expectations_suite.build().run(data_frame=data_frame)


def test_large_dataset_performance(dataframe_factory):
    """Test the expectation with a larger dataset to ensure performance."""
    df_lib, make_df = dataframe_factory

    # Create a larger dataset with 10,000 rows all starting with "foo"
    large_data = ["foo" + str(i) for i in range(10000)]
    data_frame = make_df({"col1": (large_data, "string")})

    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationStringStartsWith",
        column_name="col1",
        prefix="foo",
    )

    result = expectation.validate(data_frame=data_frame)
    # Should succeed as all values start with "foo"
    assert isinstance(result, DataFrameExpectationSuccessMessage), (
        f"Large dataset test failed: expected success but got {type(result)}"
    )
