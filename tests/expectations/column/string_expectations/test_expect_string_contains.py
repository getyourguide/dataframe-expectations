import pytest
import pandas as pd

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
from dataframe_expectations.core.types import DataFrameType


def create_pyspark_dataframe(data, column_name, spark):
    """Helper function to create a PySpark DataFrame."""
    from pyspark.sql.types import (
        StructType,
        StructField,
        StringType,
    )

    schema = StructType([StructField(column_name, StringType(), True)])
    return spark.createDataFrame([(val,) for val in data], schema)


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
        # Case sensitivity scenarios - pandas
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
        # Substring position scenarios - pandas
        ("pandas", ["foobar", "foobaz", "foo"], "foo", "success", None, None),
        ("pandas", ["barfoo", "bazfoo", "foo"], "foo", "success", None, None),
        ("pandas", ["barfoobar", "bazfoobaz"], "foo", "success", None, None),
        # Single character scenarios - pandas
        ("pandas", ["a", "ab", "abc"], "a", "success", None, None),
        (
            "pandas",
            ["b", "c", "d"],
            "a",
            "failure",
            ["b", "c", "d"],
            "Found 3 row(s) where 'col1' does not contain 'a'.",
        ),
        (
            "pandas",
            ["", "text", "more"],
            "text",
            "failure",
            ["", "more"],
            "Found 2 row(s) where 'col1' does not contain 'text'.",
        ),
        # Whitespace handling scenarios - pandas
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
        # Special character scenarios - pandas
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
        # Number scenarios - pandas
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
        # Multiple occurrence and long string scenarios - pandas
        ("pandas", ["foofoo", "foobarfoo", "foo"], "foo", "success", None, None),
        (
            "pandas",
            ["a" * 100 + "foo" + "b" * 100, "c" * 200],
            "foo",
            "failure",
            ["c" * 200],
            "Found 1 row(s) where 'col1' does not contain 'foo'.",
        ),
        ("pandas", ["football", "foolish", "foo"], "foo", "success", None, None),
        (
            "pandas",
            ["food", "foot", "bar"],
            "foo",
            "failure",
            ["bar"],
            "Found 1 row(s) where 'col1' does not contain 'foo'.",
        ),
    ],
    ids=[
        "pandas_basic_success",
        "pandas_success_hello",
        "pandas_success_test",
        "pandas_basic_violations",
        "pandas_all_violations",
        "pandas_partial_match_violations",
        "pandas_case_sensitive_all_fail",
        "pandas_case_sensitive_mixed",
        "pandas_case_hello_violations",
        "pandas_position_start",
        "pandas_position_end",
        "pandas_position_middle",
        "pandas_single_char_success",
        "pandas_single_char_violations",
        "pandas_empty_string_violations",
        "pandas_whitespace_in_data",
        "pandas_whitespace_around",
        "pandas_whitespace_in_substring",
        "pandas_special_at_sign",
        "pandas_special_exclamation",
        "pandas_special_slash",
        "pandas_special_hash",
        "pandas_numbers_123",
        "pandas_numbers_version",
        "pandas_multiple_occurrences",
        "pandas_long_strings",
        "pandas_partial_word_success",
        "pandas_partial_word_violation",
    ],
)
def test_expectation_basic_scenarios_pandas(
    df_type, data, substring, expected_result, expected_violations, expected_message
):
    """Test the expectation for various scenarios for pandas DataFrames."""
    data_frame = pd.DataFrame({"col1": data})

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
        expected_violations_df = pd.DataFrame({"col1": expected_violations})
        expected_failure_message = DataFrameExpectationFailureMessage(
            expectation_str=str(expectation),
            data_frame_type=DataFrameType(df_type),
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


@pytest.mark.pyspark
@pytest.mark.parametrize(
    "df_type, data, substring, expected_result, expected_violations, expected_message",
    [
        # Basic success scenarios - pyspark
        ("pyspark", ["foobar", "foo123", "barfoo"], "foo", "success", None, None),
        ("pyspark", ["hello", "hello world", "say hello"], "hello", "success", None, None),
        ("pyspark", ["test123", "test456", "testing"], "test", "success", None, None),
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
        (
            "pyspark",
            ["test", "testing", "best"],
            "testing",
            "failure",
            ["test", "best"],
            "Found 2 row(s) where 'col1' does not contain 'testing'.",
        ),
        # Case sensitivity scenarios - pyspark
        (
            "pyspark",
            ["Foo", "FOO", "fOo"],
            "foo",
            "failure",
            ["Foo", "FOO", "fOo"],
            "Found 3 row(s) where 'col1' does not contain 'foo'.",
        ),
        ("pyspark", ["foo", "foobar", "barfoo"], "foo", "success", None, None),
        (
            "pyspark",
            ["Hello", "HELLO", "hello"],
            "Hello",
            "failure",
            ["HELLO", "hello"],
            "Found 2 row(s) where 'col1' does not contain 'Hello'.",
        ),
        # Substring position scenarios - pyspark
        ("pyspark", ["foobar", "foobaz", "foo"], "foo", "success", None, None),
        ("pyspark", ["barfoo", "bazfoo", "foo"], "foo", "success", None, None),
        ("pyspark", ["barfoobar", "bazfoobaz"], "foo", "success", None, None),
        # Single character scenarios - pyspark
        ("pyspark", ["a", "ab", "abc"], "a", "success", None, None),
        (
            "pyspark",
            ["b", "c", "d"],
            "a",
            "failure",
            ["b", "c", "d"],
            "Found 3 row(s) where 'col1' does not contain 'a'.",
        ),
        (
            "pyspark",
            ["", "text", "more"],
            "text",
            "failure",
            ["", "more"],
            "Found 2 row(s) where 'col1' does not contain 'text'.",
        ),
        # Whitespace handling scenarios - pyspark
        ("pyspark", ["foo bar", "foo  bar", "foobar"], "foo", "success", None, None),
        ("pyspark", [" foo", "foo ", " foo "], "foo", "success", None, None),
        (
            "pyspark",
            ["foo bar", "bar", "baz"],
            "foo bar",
            "failure",
            ["bar", "baz"],
            "Found 2 row(s) where 'col1' does not contain 'foo bar'.",
        ),
        # Special character scenarios - pyspark
        ("pyspark", ["test@email.com", "user@domain.org"], "@", "success", None, None),
        (
            "pyspark",
            ["hello!", "world!", "test"],
            "!",
            "failure",
            ["test"],
            "Found 1 row(s) where 'col1' does not contain '!'.",
        ),
        ("pyspark", ["path/to/file", "another/path"], "/", "success", None, None),
        (
            "pyspark",
            ["#tag1", "#tag2", "notag"],
            "#",
            "failure",
            ["notag"],
            "Found 1 row(s) where 'col1' does not contain '#'.",
        ),
        # Number scenarios - pyspark
        (
            "pyspark",
            ["test123", "test456", "test"],
            "123",
            "failure",
            ["test456", "test"],
            "Found 2 row(s) where 'col1' does not contain '123'.",
        ),
        (
            "pyspark",
            ["v1.0.0", "v2.0.0", "v1.1.0"],
            "1.",
            "failure",
            ["v2.0.0"],
            "Found 1 row(s) where 'col1' does not contain '1.'.",
        ),
        # Multiple occurrence and long string scenarios - pyspark
        ("pyspark", ["foofoo", "foobarfoo", "foo"], "foo", "success", None, None),
        (
            "pyspark",
            ["a" * 100 + "foo" + "b" * 100, "c" * 200],
            "foo",
            "failure",
            ["c" * 200],
            "Found 1 row(s) where 'col1' does not contain 'foo'.",
        ),
        ("pyspark", ["football", "foolish", "foo"], "foo", "success", None, None),
        (
            "pyspark",
            ["food", "foot", "bar"],
            "foo",
            "failure",
            ["bar"],
            "Found 1 row(s) where 'col1' does not contain 'foo'.",
        ),
    ],
    ids=[
        "pyspark_basic_success",
        "pyspark_success_hello",
        "pyspark_success_test",
        "pyspark_basic_violations",
        "pyspark_all_violations",
        "pyspark_partial_match_violations",
        "pyspark_case_sensitive_all_fail",
        "pyspark_case_sensitive_mixed",
        "pyspark_case_hello_violations",
        "pyspark_position_start",
        "pyspark_position_end",
        "pyspark_position_middle",
        "pyspark_single_char_success",
        "pyspark_single_char_violations",
        "pyspark_empty_string_violations",
        "pyspark_whitespace_in_data",
        "pyspark_whitespace_around",
        "pyspark_whitespace_in_substring",
        "pyspark_special_at_sign",
        "pyspark_special_exclamation",
        "pyspark_special_slash",
        "pyspark_special_hash",
        "pyspark_numbers_123",
        "pyspark_numbers_version",
        "pyspark_multiple_occurrences",
        "pyspark_long_strings",
        "pyspark_partial_word_success",
        "pyspark_partial_word_violation",
    ],
)
def test_expectation_basic_scenarios_pyspark(
    df_type, data, substring, expected_result, expected_violations, expected_message, spark
):
    """Test the expectation for various scenarios for PySpark DataFrames."""
    data_frame = create_pyspark_dataframe(data, "col1", spark)

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
        expected_violations_df = create_pyspark_dataframe(expected_violations, "col1", spark)
        expected_failure_message = DataFrameExpectationFailureMessage(
            expectation_str=str(expectation),
            data_frame_type=DataFrameType(df_type),
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


def test_column_missing_error_pandas():
    """Test that an error is raised when the specified column is missing in pandas."""
    expected_message = "Column 'col1' does not exist in the DataFrame."

    data_frame = pd.DataFrame({"col2": ["foobar", "foo123", "barfoo"]})

    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationStringContains",
        column_name="col1",
        substring="foo",
    )
    result = expectation.validate(data_frame=data_frame)
    expected_failure = DataFrameExpectationFailureMessage(
        expectation_str=str(expectation),
        data_frame_type=DataFrameType.PANDAS,
        message=expected_message,
    )
    assert str(result) == str(expected_failure), f"Expected failure message but got: {result}"

    expectations_suite = DataFrameExpectationsSuite().expect_string_contains(
        column_name="col1", substring="foo"
    )
    with pytest.raises(DataFrameExpectationsSuiteFailure):
        expectations_suite.build().run(data_frame=data_frame)


@pytest.mark.pyspark
def test_column_missing_error_pyspark(spark):
    """Test that an error is raised when the specified column is missing in PySpark."""
    expected_message = "Column 'col1' does not exist in the DataFrame."

    data_frame = spark.createDataFrame([("foobar",), ("foo123",), ("barfoo",)], ["col2"])

    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationStringContains",
        column_name="col1",
        substring="foo",
    )
    result = expectation.validate(data_frame=data_frame)
    expected_failure = DataFrameExpectationFailureMessage(
        expectation_str=str(expectation),
        data_frame_type=DataFrameType.PYSPARK,
        message=expected_message,
    )
    assert str(result) == str(expected_failure), f"Expected failure message but got: {result}"

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
