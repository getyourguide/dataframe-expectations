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
        expectation_name="ExpectationStringEndsWith",
        column_name="col1",
        suffix="bar",
    )
    assert expectation.get_expectation_name() == "ExpectationStringEndsWith", (
        f"Expected 'ExpectationStringEndsWith' but got: {expectation.get_expectation_name()}"
    )


@pytest.mark.parametrize(
    "df_type, data, suffix, expected_result, expected_violations, expected_message",
    [
        # Basic success scenarios - pandas
        ("pandas", ["foobar", "bar", "bazbar"], "bar", "success", None, None),
        ("pandas", ["hello", "say hello", "greet with hello"], "hello", "success", None, None),
        ("pandas", ["test", "mytest", "anothertest"], "test", "success", None, None),
        # Basic success scenarios - pyspark
        ("pyspark", ["foobar", "bar", "bazbar"], "bar", "success", None, None),
        ("pyspark", ["hello", "say hello", "greet with hello"], "hello", "success", None, None),
        ("pyspark", ["test", "mytest", "anothertest"], "test", "success", None, None),
        # Basic violation scenarios - pandas
        (
            "pandas",
            ["foobar", "bar", "baz"],
            "bar",
            "failure",
            ["baz"],
            "Found 1 row(s) where 'col1' does not end with 'bar'.",
        ),
        (
            "pandas",
            ["apple", "banana", "cherry"],
            "xyz",
            "failure",
            ["apple", "banana", "cherry"],
            "Found 3 row(s) where 'col1' does not end with 'xyz'.",
        ),
        (
            "pandas",
            ["testing", "test", "best"],
            "test",
            "failure",
            ["testing", "best"],
            "Found 2 row(s) where 'col1' does not end with 'test'.",
        ),
        # Basic violation scenarios - pyspark
        (
            "pyspark",
            ["foobar", "bar", "baz"],
            "bar",
            "failure",
            ["baz"],
            "Found 1 row(s) where 'col1' does not end with 'bar'.",
        ),
        (
            "pyspark",
            ["apple", "banana", "cherry"],
            "xyz",
            "failure",
            ["apple", "banana", "cherry"],
            "Found 3 row(s) where 'col1' does not end with 'xyz'.",
        ),
        # Case sensitivity - pandas (endswith is case-sensitive)
        (
            "pandas",
            ["fooBar", "fooBAR", "foobar"],
            "bar",
            "failure",
            ["fooBar", "fooBAR"],
            "Found 2 row(s) where 'col1' does not end with 'bar'.",
        ),
        ("pandas", ["foobar", "bazbar", "bar"], "bar", "success", None, None),
        (
            "pandas",
            ["testHello", "testHELLO", "testhello"],
            "hello",
            "failure",
            ["testHello", "testHELLO"],
            "Found 2 row(s) where 'col1' does not end with 'hello'.",
        ),
        # Case sensitivity - pyspark
        (
            "pyspark",
            ["fooBar", "fooBAR", "foobar"],
            "bar",
            "failure",
            ["fooBar", "fooBAR"],
            "Found 2 row(s) where 'col1' does not end with 'bar'.",
        ),
        ("pyspark", ["foobar", "bazbar", "bar"], "bar", "success", None, None),
        # Exact match - suffix equals entire string - pandas
        (
            "pandas",
            ["bar", "foo", "baz"],
            "bar",
            "failure",
            ["foo", "baz"],
            "Found 2 row(s) where 'col1' does not end with 'bar'.",
        ),
        ("pandas", ["test", "test", "test"], "test", "success", None, None),
        # Exact match - pyspark
        (
            "pyspark",
            ["bar", "foo", "baz"],
            "bar",
            "failure",
            ["foo", "baz"],
            "Found 2 row(s) where 'col1' does not end with 'bar'.",
        ),
        # Single character suffix - pandas
        ("pandas", ["a", "ba", "cba"], "a", "success", None, None),
        (
            "pandas",
            ["b", "c", "d"],
            "a",
            "failure",
            ["b", "c", "d"],
            "Found 3 row(s) where 'col1' does not end with 'a'.",
        ),
        # Single character suffix - pyspark
        ("pyspark", ["a", "ba", "cba"], "a", "success", None, None),
        # Empty string scenarios - pandas
        (
            "pandas",
            ["", "text", "more"],
            "text",
            "failure",
            ["", "more"],
            "Found 2 row(s) where 'col1' does not end with 'text'.",
        ),
        # Empty string scenarios - pyspark
        (
            "pyspark",
            ["", "text", "more"],
            "text",
            "failure",
            ["", "more"],
            "Found 2 row(s) where 'col1' does not end with 'text'.",
        ),
        # Whitespace handling - pandas
        ("pandas", ["foo bar", "baz bar", "test bar"], "bar", "success", None, None),
        ("pandas", ["foo ", "bar ", "baz "], " ", "success", None, None),
        (
            "pandas",
            ["foo bar", "bar", "baz"],
            " bar",
            "failure",
            ["bar", "baz"],
            "Found 2 row(s) where 'col1' does not end with ' bar'.",
        ),
        # Whitespace handling - pyspark
        ("pyspark", ["foo bar", "baz bar", "test bar"], "bar", "success", None, None),
        ("pyspark", ["foo ", "bar ", "baz "], " ", "success", None, None),
        # Special characters - pandas
        ("pandas", ["test@", "user@", "admin@"], "@", "success", None, None),
        (
            "pandas",
            ["hello!", "world!", "test"],
            "!",
            "failure",
            ["test"],
            "Found 1 row(s) where 'col1' does not end with '!'.",
        ),
        ("pandas", ["path/to/", "another/"], "/", "success", None, None),
        (
            "pandas",
            ["tag1#", "tag2#", "notag"],
            "#",
            "failure",
            ["notag"],
            "Found 1 row(s) where 'col1' does not end with '#'.",
        ),
        # Special characters - pyspark
        ("pyspark", ["test@", "user@", "admin@"], "@", "success", None, None),
        ("pyspark", ["path/to/", "another/"], "/", "success", None, None),
        # Numbers in strings - pandas
        (
            "pandas",
            ["test123", "file123", "data456"],
            "123",
            "failure",
            ["data456"],
            "Found 1 row(s) where 'col1' does not end with '123'.",
        ),
        (
            "pandas",
            ["v1.0", "v2.0", "v1.1"],
            ".0",
            "failure",
            ["v1.1"],
            "Found 1 row(s) where 'col1' does not end with '.0'.",
        ),
        # Numbers in strings - pyspark
        (
            "pyspark",
            ["test123", "file123", "data456"],
            "123",
            "failure",
            ["data456"],
            "Found 1 row(s) where 'col1' does not end with '123'.",
        ),
        # Long strings - pandas
        ("pandas", ["a" * 100 + "bar", "b" * 100 + "bar"], "bar", "success", None, None),
        (
            "pandas",
            ["a" * 100 + "foo", "b" * 100 + "bar"],
            "bar",
            "failure",
            ["a" * 100 + "foo"],
            "Found 1 row(s) where 'col1' does not end with 'bar'.",
        ),
        # Long strings - pyspark
        ("pyspark", ["a" * 100 + "bar", "b" * 100 + "bar"], "bar", "success", None, None),
        # Suffix longer than string - pandas
        (
            "pandas",
            ["foo", "ba", "b"],
            "foobar",
            "failure",
            ["foo", "ba", "b"],
            "Found 3 row(s) where 'col1' does not end with 'foobar'.",
        ),
        # Suffix longer than string - pyspark
        (
            "pyspark",
            ["foo", "ba", "b"],
            "foobar",
            "failure",
            ["foo", "ba", "b"],
            "Found 3 row(s) where 'col1' does not end with 'foobar'.",
        ),
        # Partial matches not at end - pandas
        (
            "pandas",
            ["barfoo", "foobar", "bazbar"],
            "foo",
            "failure",
            ["foobar", "bazbar"],
            "Found 2 row(s) where 'col1' does not end with 'foo'.",
        ),
        # Partial matches not at end - pyspark
        (
            "pyspark",
            ["barfoo", "foobar", "bazbar"],
            "foo",
            "failure",
            ["foobar", "bazbar"],
            "Found 2 row(s) where 'col1' does not end with 'foo'.",
        ),
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
        "pandas_case_sensitive_violations",
        "pandas_case_sensitive_success",
        "pandas_case_hello_violations",
        "pyspark_case_sensitive_violations",
        "pyspark_case_sensitive_success",
        "pandas_exact_match_partial",
        "pandas_exact_match_all",
        "pyspark_exact_match_partial",
        "pandas_single_char_success",
        "pandas_single_char_violations",
        "pyspark_single_char_success",
        "pandas_empty_string_violations",
        "pyspark_empty_string_violations",
        "pandas_whitespace_success",
        "pandas_whitespace_trailing",
        "pandas_whitespace_in_suffix",
        "pyspark_whitespace_success",
        "pyspark_whitespace_trailing",
        "pandas_special_at_sign",
        "pandas_special_exclamation",
        "pandas_special_slash",
        "pandas_special_hash",
        "pyspark_special_at_sign",
        "pyspark_special_slash",
        "pandas_numbers_123",
        "pandas_numbers_version",
        "pyspark_numbers_123",
        "pandas_long_strings_success",
        "pandas_long_strings_violation",
        "pyspark_long_strings_success",
        "pandas_suffix_longer",
        "pyspark_suffix_longer",
        "pandas_partial_not_at_end",
        "pyspark_partial_not_at_end",
    ],
)
def test_expectation_basic_scenarios(
    df_type, data, suffix, expected_result, expected_violations, expected_message, spark
):
    """
    Test the expectation for various scenarios across pandas and PySpark DataFrames.
    Tests both direct expectation validation and suite-based validation.
    Covers: success cases, violations, case sensitivity, exact matches,
    empty strings, whitespace, special characters, numbers, long strings,
    and partial matches not at the end.
    """
    data_frame = create_dataframe(df_type, data, "col1", spark)

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
    expectations_suite = DataFrameExpectationsSuite().expect_string_ends_with(
        column_name="col1", suffix=suffix
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
        data_frame = pd.DataFrame({"col2": ["foobar", "bar", "bazbar"]})
    else:  # pyspark
        data_frame = spark.createDataFrame([("foobar",), ("bar",), ("bazbar",)], ["col2"])

    # Test 1: Direct expectation validation
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationStringEndsWith",
        column_name="col1",
        suffix="bar",
    )
    result = expectation.validate(data_frame=data_frame)
    expected_failure = DataFrameExpectationFailureMessage(
        expectation_str=str(expectation),
        data_frame_type=get_df_type_enum(df_type),
        message=expected_message,
    )
    assert str(result) == str(expected_failure), f"Expected failure message but got: {result}"

    # Test 2: Suite-based validation
    expectations_suite = DataFrameExpectationsSuite().expect_string_ends_with(
        column_name="col1", suffix="bar"
    )
    with pytest.raises(DataFrameExpectationsSuiteFailure):
        expectations_suite.build().run(data_frame=data_frame)


def test_large_dataset_performance():
    """Test the expectation with a larger dataset to ensure performance."""
    # Create a larger dataset with 10,000 rows ending with "_test"
    large_data = [f"value_{i}_test" for i in range(10000)]
    data_frame = pd.DataFrame({"col1": large_data})

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
