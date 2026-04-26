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
        # Case sensitivity scenarios - pandas
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
        # Exact match scenarios - pandas
        (
            "pandas",
            ["bar", "foo", "baz"],
            "bar",
            "failure",
            ["foo", "baz"],
            "Found 2 row(s) where 'col1' does not end with 'bar'.",
        ),
        ("pandas", ["test", "test", "test"], "test", "success", None, None),
        # Single character suffix scenarios - pandas
        ("pandas", ["a", "ba", "cba"], "a", "success", None, None),
        (
            "pandas",
            ["b", "c", "d"],
            "a",
            "failure",
            ["b", "c", "d"],
            "Found 3 row(s) where 'col1' does not end with 'a'.",
        ),
        (
            "pandas",
            ["", "text", "more"],
            "text",
            "failure",
            ["", "more"],
            "Found 2 row(s) where 'col1' does not end with 'text'.",
        ),
        # Whitespace handling scenarios - pandas
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
        # Special character scenarios - pandas
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
        # Number scenarios - pandas
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
        # Long string and suffix-length scenarios - pandas
        ("pandas", ["a" * 100 + "bar", "b" * 100 + "bar"], "bar", "success", None, None),
        (
            "pandas",
            ["a" * 100 + "foo", "b" * 100 + "bar"],
            "bar",
            "failure",
            ["a" * 100 + "foo"],
            "Found 1 row(s) where 'col1' does not end with 'bar'.",
        ),
        (
            "pandas",
            ["foo", "ba", "b"],
            "foobar",
            "failure",
            ["foo", "ba", "b"],
            "Found 3 row(s) where 'col1' does not end with 'foobar'.",
        ),
        (
            "pandas",
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
        "pandas_basic_violations",
        "pandas_all_violations",
        "pandas_partial_match_violations",
        "pandas_case_sensitive_violations",
        "pandas_case_sensitive_success",
        "pandas_case_hello_violations",
        "pandas_exact_match_partial",
        "pandas_exact_match_all",
        "pandas_single_char_success",
        "pandas_single_char_violations",
        "pandas_empty_string_violations",
        "pandas_whitespace_success",
        "pandas_whitespace_trailing",
        "pandas_whitespace_in_suffix",
        "pandas_special_at_sign",
        "pandas_special_exclamation",
        "pandas_special_slash",
        "pandas_special_hash",
        "pandas_numbers_123",
        "pandas_numbers_version",
        "pandas_long_strings_success",
        "pandas_long_strings_violation",
        "pandas_suffix_longer",
        "pandas_partial_not_at_end",
    ],
)
def test_expectation_basic_scenarios_pandas(
    df_type, data, suffix, expected_result, expected_violations, expected_message
):
    """Test the expectation for various scenarios for pandas DataFrames."""
    data_frame = pd.DataFrame({"col1": data})

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


@pytest.mark.pyspark
@pytest.mark.parametrize(
    "df_type, data, suffix, expected_result, expected_violations, expected_message",
    [
        # Basic success scenarios - pyspark
        ("pyspark", ["foobar", "bar", "bazbar"], "bar", "success", None, None),
        ("pyspark", ["hello", "say hello", "greet with hello"], "hello", "success", None, None),
        ("pyspark", ["test", "mytest", "anothertest"], "test", "success", None, None),
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
        (
            "pyspark",
            ["testing", "test", "best"],
            "test",
            "failure",
            ["testing", "best"],
            "Found 2 row(s) where 'col1' does not end with 'test'.",
        ),
        # Case sensitivity scenarios - pyspark
        (
            "pyspark",
            ["fooBar", "fooBAR", "foobar"],
            "bar",
            "failure",
            ["fooBar", "fooBAR"],
            "Found 2 row(s) where 'col1' does not end with 'bar'.",
        ),
        ("pyspark", ["foobar", "bazbar", "bar"], "bar", "success", None, None),
        (
            "pyspark",
            ["testHello", "testHELLO", "testhello"],
            "hello",
            "failure",
            ["testHello", "testHELLO"],
            "Found 2 row(s) where 'col1' does not end with 'hello'.",
        ),
        # Exact match scenarios - pyspark
        (
            "pyspark",
            ["bar", "foo", "baz"],
            "bar",
            "failure",
            ["foo", "baz"],
            "Found 2 row(s) where 'col1' does not end with 'bar'.",
        ),
        ("pyspark", ["test", "test", "test"], "test", "success", None, None),
        # Single character suffix scenarios - pyspark
        ("pyspark", ["a", "ba", "cba"], "a", "success", None, None),
        (
            "pyspark",
            ["b", "c", "d"],
            "a",
            "failure",
            ["b", "c", "d"],
            "Found 3 row(s) where 'col1' does not end with 'a'.",
        ),
        (
            "pyspark",
            ["", "text", "more"],
            "text",
            "failure",
            ["", "more"],
            "Found 2 row(s) where 'col1' does not end with 'text'.",
        ),
        # Whitespace handling scenarios - pyspark
        ("pyspark", ["foo bar", "baz bar", "test bar"], "bar", "success", None, None),
        ("pyspark", ["foo ", "bar ", "baz "], " ", "success", None, None),
        (
            "pyspark",
            ["foo bar", "bar", "baz"],
            " bar",
            "failure",
            ["bar", "baz"],
            "Found 2 row(s) where 'col1' does not end with ' bar'.",
        ),
        # Special character scenarios - pyspark
        ("pyspark", ["test@", "user@", "admin@"], "@", "success", None, None),
        (
            "pyspark",
            ["hello!", "world!", "test"],
            "!",
            "failure",
            ["test"],
            "Found 1 row(s) where 'col1' does not end with '!'.",
        ),
        ("pyspark", ["path/to/", "another/"], "/", "success", None, None),
        (
            "pyspark",
            ["tag1#", "tag2#", "notag"],
            "#",
            "failure",
            ["notag"],
            "Found 1 row(s) where 'col1' does not end with '#'.",
        ),
        # Number scenarios - pyspark
        (
            "pyspark",
            ["test123", "file123", "data456"],
            "123",
            "failure",
            ["data456"],
            "Found 1 row(s) where 'col1' does not end with '123'.",
        ),
        (
            "pyspark",
            ["v1.0", "v2.0", "v1.1"],
            ".0",
            "failure",
            ["v1.1"],
            "Found 1 row(s) where 'col1' does not end with '.0'.",
        ),
        # Long string and suffix-length scenarios - pyspark
        ("pyspark", ["a" * 100 + "bar", "b" * 100 + "bar"], "bar", "success", None, None),
        (
            "pyspark",
            ["a" * 100 + "foo", "b" * 100 + "bar"],
            "bar",
            "failure",
            ["a" * 100 + "foo"],
            "Found 1 row(s) where 'col1' does not end with 'bar'.",
        ),
        (
            "pyspark",
            ["foo", "ba", "b"],
            "foobar",
            "failure",
            ["foo", "ba", "b"],
            "Found 3 row(s) where 'col1' does not end with 'foobar'.",
        ),
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
        "pyspark_basic_success",
        "pyspark_success_hello",
        "pyspark_success_test",
        "pyspark_basic_violations",
        "pyspark_all_violations",
        "pyspark_partial_match_violations",
        "pyspark_case_sensitive_violations",
        "pyspark_case_sensitive_success",
        "pyspark_case_hello_violations",
        "pyspark_exact_match_partial",
        "pyspark_exact_match_all",
        "pyspark_single_char_success",
        "pyspark_single_char_violations",
        "pyspark_empty_string_violations",
        "pyspark_whitespace_success",
        "pyspark_whitespace_trailing",
        "pyspark_whitespace_in_suffix",
        "pyspark_special_at_sign",
        "pyspark_special_exclamation",
        "pyspark_special_slash",
        "pyspark_special_hash",
        "pyspark_numbers_123",
        "pyspark_numbers_version",
        "pyspark_long_strings_success",
        "pyspark_long_strings_violation",
        "pyspark_suffix_longer",
        "pyspark_partial_not_at_end",
    ],
)
def test_expectation_basic_scenarios_pyspark(
    df_type, data, suffix, expected_result, expected_violations, expected_message, spark
):
    """Test the expectation for various scenarios for PySpark DataFrames."""
    data_frame = create_pyspark_dataframe(data, "col1", spark)

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


def test_column_missing_error_pandas():
    """Test that an error is raised when the specified column is missing in pandas."""
    expected_message = "Column 'col1' does not exist in the DataFrame."

    data_frame = pd.DataFrame({"col2": ["foobar", "bar", "bazbar"]})

    # Test 1: Direct expectation validation
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationStringEndsWith",
        column_name="col1",
        suffix="bar",
    )
    result = expectation.validate(data_frame=data_frame)
    expected_failure = DataFrameExpectationFailureMessage(
        expectation_str=str(expectation),
        data_frame_type=DataFrameType.PANDAS,
        message=expected_message,
    )
    assert str(result) == str(expected_failure), f"Expected failure message but got: {result}"

    # Test 2: Suite-based validation
    expectations_suite = DataFrameExpectationsSuite().expect_string_ends_with(
        column_name="col1", suffix="bar"
    )
    with pytest.raises(DataFrameExpectationsSuiteFailure):
        expectations_suite.build().run(data_frame=data_frame)


@pytest.mark.pyspark
def test_column_missing_error_pyspark(spark):
    """Test that an error is raised when the specified column is missing in PySpark."""
    expected_message = "Column 'col1' does not exist in the DataFrame."

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
        data_frame_type=DataFrameType.PYSPARK,
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
