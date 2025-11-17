import pytest
import pandas as pd

from dataframe_expectations.core.types import DataFrameType
from dataframe_expectations.registry import (
    DataFrameExpectationRegistry,
)
from dataframe_expectations.suite import (
    DataFrameExpectationsSuite,
    DataFrameExpectationsSuiteFailure,
)
from dataframe_expectations.result_message import (
    DataFrameExpectationFailureMessage,
    DataFrameExpectationSuccessMessage,
)


def create_dataframe(df_type, data, column_name, spark):
    """Helper function to create pandas or pyspark DataFrame."""
    if df_type == "pandas":
        return pd.DataFrame({column_name: data})
    else:  # pyspark
        return spark.createDataFrame([(val,) for val in data], [column_name])


def get_df_type_enum(df_type):
    """Get DataFrameType enum value."""
    return DataFrameType.PANDAS if df_type == "pandas" else DataFrameType.PYSPARK


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
    "df_type, data, substring, expected_result, expected_violations, expected_message",
    [
        # Basic success scenarios - pandas
        ("pandas", ["foobar", "foo123", "barfoo"], "foo", "success", None, None),
        ("pandas", ["hello", "hello world", "say hello"], "hello", "success", None, None),
        ("pandas", ["test123", "test456", "testing"], "test", "success", None, None),
        # Basic success scenarios - pyspark
        ("pyspark", ["foobar", "foo123", "barfoo"], "foo", "success", None, None),
        ("pyspark", ["hello", "hello world", "say hello"], "hello", "success", None, None),
        ("pyspark", ["test123", "test456", "testing"], "test", "success", None, None),
        # Basic violation scenarios - pandas
        (
            "pandas",
            ["foobar", "bar", "baz"],
            "foo",
            "failure",
            ["bar", "baz"],
            "Found 2 row(s) where 'col1' does not contain 'foo'.",
        ),
        (
            "pandas",
            ["apple", "banana", "cherry"],
            "xyz",
            "failure",
            ["apple", "banana", "cherry"],
            "Found 3 row(s) where 'col1' does not contain 'xyz'.",
        ),
        (
            "pandas",
            ["test", "testing", "best"],
            "testing",
            "failure",
            ["test", "best"],
            "Found 2 row(s) where 'col1' does not contain 'testing'.",
        ),
        # Basic violation scenarios - pyspark
        (
            "pyspark",
            ["foobar", "bar", "baz"],
            "foo",
            "failure",
            ["bar", "baz"],
            "Found 2 row(s) where 'col1' does not contain 'foo'.",
        ),
        (
            "pyspark",
            ["apple", "banana", "cherry"],
            "xyz",
            "failure",
            ["apple", "banana", "cherry"],
            "Found 3 row(s) where 'col1' does not contain 'xyz'.",
        ),
        # Case sensitivity - pandas (contains is case-sensitive)
        (
            "pandas",
            ["Foo", "FOO", "fOo"],
            "foo",
            "failure",
            ["Foo", "FOO", "fOo"],
            "Found 3 row(s) where 'col1' does not contain 'foo'.",
        ),
        ("pandas", ["foo", "foobar", "barfoo"], "foo", "success", None, None),
        (
            "pandas",
            ["Hello", "HELLO", "hello"],
            "Hello",
            "failure",
            ["HELLO", "hello"],
            "Found 2 row(s) where 'col1' does not contain 'Hello'.",
        ),
        # Case sensitivity - pyspark
        (
            "pyspark",
            ["Foo", "FOO", "fOo"],
            "foo",
            "failure",
            ["Foo", "FOO", "fOo"],
            "Found 3 row(s) where 'col1' does not contain 'foo'.",
        ),
        ("pyspark", ["foo", "foobar", "barfoo"], "foo", "success", None, None),
        # Position tests - substring at start - pandas
        ("pandas", ["foobar", "foobaz", "foo"], "foo", "success", None, None),
        # Position tests - substring at end - pandas
        ("pandas", ["barfoo", "bazfoo", "foo"], "foo", "success", None, None),
        # Position tests - substring in middle - pandas
        ("pandas", ["barfoobar", "bazfoobaz"], "foo", "success", None, None),
        # Position tests - pyspark
        ("pyspark", ["foobar", "foobaz", "foo"], "foo", "success", None, None),
        ("pyspark", ["barfoo", "bazfoo", "foo"], "foo", "success", None, None),
        # Single character substring - pandas
        ("pandas", ["a", "ab", "abc"], "a", "success", None, None),
        (
            "pandas",
            ["b", "c", "d"],
            "a",
            "failure",
            ["b", "c", "d"],
            "Found 3 row(s) where 'col1' does not contain 'a'.",
        ),
        # Single character substring - pyspark
        ("pyspark", ["a", "ab", "abc"], "a", "success", None, None),
        # Empty string scenarios - pandas
        (
            "pandas",
            ["", "text", "more"],
            "text",
            "failure",
            ["", "more"],
            "Found 2 row(s) where 'col1' does not contain 'text'.",
        ),
        # Empty string scenarios - pyspark
        (
            "pyspark",
            ["", "text", "more"],
            "text",
            "failure",
            ["", "more"],
            "Found 2 row(s) where 'col1' does not contain 'text'.",
        ),
        # Whitespace handling - pandas
        ("pandas", ["foo bar", "foo  bar", "foobar"], "foo", "success", None, None),
        ("pandas", [" foo", "foo ", " foo "], "foo", "success", None, None),
        (
            "pandas",
            ["foo bar", "bar", "baz"],
            "foo bar",
            "failure",
            ["bar", "baz"],
            "Found 2 row(s) where 'col1' does not contain 'foo bar'.",
        ),
        # Whitespace handling - pyspark
        ("pyspark", ["foo bar", "foo  bar", "foobar"], "foo", "success", None, None),
        ("pyspark", [" foo", "foo ", " foo "], "foo", "success", None, None),
        # Special characters - pandas
        ("pandas", ["test@email.com", "user@domain.org"], "@", "success", None, None),
        (
            "pandas",
            ["hello!", "world!", "test"],
            "!",
            "failure",
            ["test"],
            "Found 1 row(s) where 'col1' does not contain '!'.",
        ),
        ("pandas", ["path/to/file", "another/path"], "/", "success", None, None),
        (
            "pandas",
            ["#tag1", "#tag2", "notag"],
            "#",
            "failure",
            ["notag"],
            "Found 1 row(s) where 'col1' does not contain '#'.",
        ),
        # Special characters - pyspark
        ("pyspark", ["test@email.com", "user@domain.org"], "@", "success", None, None),
        ("pyspark", ["path/to/file", "another/path"], "/", "success", None, None),
        # Numbers in strings - pandas
        (
            "pandas",
            ["test123", "test456", "test"],
            "123",
            "failure",
            ["test456", "test"],
            "Found 2 row(s) where 'col1' does not contain '123'.",
        ),
        (
            "pandas",
            ["v1.0.0", "v2.0.0", "v1.1.0"],
            "1.",
            "failure",
            ["v2.0.0"],
            "Found 1 row(s) where 'col1' does not contain '1.'.",
        ),
        # Numbers in strings - pyspark
        (
            "pyspark",
            ["test123", "test456", "test"],
            "123",
            "failure",
            ["test456", "test"],
            "Found 2 row(s) where 'col1' does not contain '123'.",
        ),
        # Multiple occurrences - pandas
        ("pandas", ["foofoo", "foobarfoo", "foo"], "foo", "success", None, None),
        # Multiple occurrences - pyspark
        ("pyspark", ["foofoo", "foobarfoo", "foo"], "foo", "success", None, None),
        # Long strings - pandas
        (
            "pandas",
            ["a" * 100 + "foo" + "b" * 100, "c" * 200],
            "foo",
            "failure",
            ["c" * 200],
            "Found 1 row(s) where 'col1' does not contain 'foo'.",
        ),
        # Long strings - pyspark
        (
            "pyspark",
            ["a" * 100 + "foo" + "b" * 100, "c" * 200],
            "foo",
            "failure",
            ["c" * 200],
            "Found 1 row(s) where 'col1' does not contain 'foo'.",
        ),
        # Partial word matches - pandas
        ("pandas", ["football", "foolish", "foo"], "foo", "success", None, None),
        (
            "pandas",
            ["food", "foot", "bar"],
            "foo",
            "failure",
            ["bar"],
            "Found 1 row(s) where 'col1' does not contain 'foo'.",
        ),
        # Partial word matches - pyspark
        ("pyspark", ["football", "foolish", "foo"], "foo", "success", None, None),
    ],
    ids=[
        "pandas_basic_success",
        "pandas_success_hello",
        "pandas_success_test",
        "pyspark_basic_success",
        "pyspark_success_hello",
        "pyspark_success_test",
        "pandas_basic_violations",
        "pandas_all_violations",
        "pandas_partial_match_violations",
        "pyspark_basic_violations",
        "pyspark_all_violations",
        "pandas_case_sensitive_all_fail",
        "pandas_case_sensitive_mixed",
        "pandas_case_hello_violations",
        "pyspark_case_sensitive_all_fail",
        "pyspark_case_sensitive_mixed",
        "pandas_position_start",
        "pandas_position_end",
        "pandas_position_middle",
        "pyspark_position_start",
        "pyspark_position_end",
        "pandas_single_char_success",
        "pandas_single_char_violations",
        "pyspark_single_char_success",
        "pandas_empty_string_violations",
        "pyspark_empty_string_violations",
        "pandas_whitespace_in_data",
        "pandas_whitespace_around",
        "pandas_whitespace_in_substring",
        "pyspark_whitespace_in_data",
        "pyspark_whitespace_around",
        "pandas_special_at_sign",
        "pandas_special_exclamation",
        "pandas_special_slash",
        "pandas_special_dollar",
        "pyspark_special_at_sign",
        "pyspark_special_slash",
        "pandas_numbers_123",
        "pandas_numbers_version",
        "pyspark_numbers_123",
        "pandas_multiple_occurrences",
        "pyspark_multiple_occurrences",
        "pandas_long_strings",
        "pyspark_long_strings",
        "pandas_partial_word_success",
        "pandas_partial_word_violation",
        "pyspark_partial_word_success",
    ],
)
def test_expectation_basic_scenarios(
    df_type, data, substring, expected_result, expected_violations, expected_message, spark
):
    """
    Test the expectation for various scenarios across pandas and PySpark DataFrames.
    Tests both direct expectation validation and suite-based validation.
    Covers: success cases, violations, case sensitivity, position of substring,
    empty strings, whitespace, special characters, numbers, multiple occurrences,
    long strings, and partial word matches.
    """
    data_frame = create_dataframe(df_type, data, "col1", spark)

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
    else:  # failure
        expected_violations_df = create_dataframe(df_type, expected_violations, "col1", spark)
        expected_failure_message = DataFrameExpectationFailureMessage(
            expectation_str=str(expectation),
            data_frame_type=get_df_type_enum(df_type),
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
        assert result is None, "Expected no exceptions to be raised from suite"
    else:  # failure
        with pytest.raises(DataFrameExpectationsSuiteFailure):
            expectations_suite.build().run(data_frame=data_frame)


@pytest.mark.parametrize(
    "df_type",
    ["pandas", "pyspark"],
    ids=["pandas", "pyspark"],
)
def test_column_missing_error(df_type, spark):
    """Test that an error is raised when the specified column is missing in both pandas and PySpark."""
    expected_message = "Column 'col1' does not exist in the DataFrame."

    if df_type == "pandas":
        data_frame = pd.DataFrame({"col2": ["foobar", "foo123", "barfoo"]})
    else:  # pyspark
        data_frame = spark.createDataFrame([("foobar",), ("foo123",), ("barfoo",)], ["col2"])

    # Test 1: Direct expectation validation
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationStringContains",
        column_name="col1",
        substring="foo",
    )
    result = expectation.validate(data_frame=data_frame)
    expected_failure = DataFrameExpectationFailureMessage(
        expectation_str=str(expectation),
        data_frame_type=get_df_type_enum(df_type),
        message=expected_message,
    )
    assert str(result) == str(expected_failure), f"Expected failure message but got: {result}"

    # Test 2: Suite-based validation
    expectations_suite = DataFrameExpectationsSuite().expect_string_contains(
        column_name="col1", substring="foo"
    )
    with pytest.raises(DataFrameExpectationsSuiteFailure):
        expectations_suite.build().run(data_frame=data_frame)


def test_large_dataset_performance():
    """Test the expectation with a larger dataset to ensure performance."""
    # Create a larger dataset with 10,000 rows containing "test"
    large_data = [f"test_{i}" for i in range(10000)]
    data_frame = pd.DataFrame({"col1": large_data})

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
