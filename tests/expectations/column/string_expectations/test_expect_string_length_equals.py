import pytest
import pandas as pd

from dataframe_expectations.registry import (
    DataFrameExpectationRegistry,
)
from dataframe_expectations.suite import (
    DataFrameExpectationsSuite,
    DataFrameExpectationsSuiteFailure,
    SuiteExecutionResult,
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


def test_expectation_name():
    """Test that the expectation name is correctly returned."""
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationStringLengthEquals",
        column_name="col1",
        length=3,
    )
    assert expectation.get_expectation_name() == "ExpectationStringLengthEquals", (
        f"Expected 'ExpectationStringLengthEquals' but got: {expectation.get_expectation_name()}"
    )


@pytest.mark.parametrize(
    "df_type, data, length, expected_result, expected_violations, expected_message",
    [
        # Basic success - pandas
        ("pandas", ["foo", "bar", "baz"], 3, "success", None, None),
        # Basic success - pyspark
        ("pyspark", ["foo", "bar", "baz"], 3, "success", None, None),
        # Success length 2 - pandas
        ("pandas", ["ab", "cd", "ef"], 2, "success", None, None),
        # Success length 2 - pyspark
        ("pyspark", ["ab", "cd", "ef"], 2, "success", None, None),
        # Success length 5 - pandas
        ("pandas", ["hello", "world", "tests"], 5, "success", None, None),
        # Success length 5 - pyspark
        ("pyspark", ["hello", "world", "tests"], 5, "success", None, None),
        # Basic violations - pandas
        (
            "pandas",
            ["foo", "bar", "bazz", "foobar"],
            3,
            "failure",
            ["bazz", "foobar"],
            "Found 2 row(s) where 'col1' length is not equal to 3.",
        ),
        # Basic violations - pyspark
        (
            "pyspark",
            ["foo", "bar", "bazz", "foobar"],
            3,
            "failure",
            ["bazz", "foobar"],
            "Found 2 row(s) where 'col1' length is not equal to 3.",
        ),
        # All violations - pandas
        (
            "pandas",
            ["a", "ab", "abc"],
            5,
            "failure",
            ["a", "ab", "abc"],
            "Found 3 row(s) where 'col1' length is not equal to 5.",
        ),
        # All violations - pyspark
        (
            "pyspark",
            ["a", "ab", "abc"],
            5,
            "failure",
            ["a", "ab", "abc"],
            "Found 3 row(s) where 'col1' length is not equal to 5.",
        ),
        # Mixed violations - pandas
        (
            "pandas",
            ["test", "testing", "t"],
            4,
            "failure",
            ["testing", "t"],
            "Found 2 row(s) where 'col1' length is not equal to 4.",
        ),
        # Mixed violations - pyspark
        (
            "pyspark",
            ["test", "testing", "t"],
            4,
            "failure",
            ["testing", "t"],
            "Found 2 row(s) where 'col1' length is not equal to 4.",
        ),
        # Single char success - pandas
        ("pandas", ["a", "b", "c"], 1, "success", None, None),
        # Single char success - pyspark
        ("pyspark", ["a", "b", "c"], 1, "success", None, None),
        # Single char violations - pandas
        (
            "pandas",
            ["a", "ab", "abc"],
            1,
            "failure",
            ["ab", "abc"],
            "Found 2 row(s) where 'col1' length is not equal to 1.",
        ),
        # Single char violations - pyspark
        (
            "pyspark",
            ["a", "ab", "abc"],
            1,
            "failure",
            ["ab", "abc"],
            "Found 2 row(s) where 'col1' length is not equal to 1.",
        ),
        # Empty string success - pandas
        ("pandas", ["", "", ""], 0, "success", None, None),
        # Empty string success - pyspark
        ("pyspark", ["", "", ""], 0, "success", None, None),
        # Empty string violations length 0 - pandas
        (
            "pandas",
            ["", "a", "ab"],
            0,
            "failure",
            ["a", "ab"],
            "Found 2 row(s) where 'col1' length is not equal to 0.",
        ),
        # Empty string violations length 0 - pyspark
        (
            "pyspark",
            ["", "a", "ab"],
            0,
            "failure",
            ["a", "ab"],
            "Found 2 row(s) where 'col1' length is not equal to 0.",
        ),
        # Empty string violations length 1 - pandas
        (
            "pandas",
            ["", "a", "ab"],
            1,
            "failure",
            ["", "ab"],
            "Found 2 row(s) where 'col1' length is not equal to 1.",
        ),
        # Empty string violations length 1 - pyspark
        (
            "pyspark",
            ["", "a", "ab"],
            1,
            "failure",
            ["", "ab"],
            "Found 2 row(s) where 'col1' length is not equal to 1.",
        ),
        # Whitespace length 1 - pandas
        (
            "pandas",
            ["   ", "  ", " "],
            1,
            "failure",
            ["   ", "  "],
            "Found 2 row(s) where 'col1' length is not equal to 1.",
        ),
        # Whitespace length 1 - pyspark
        (
            "pyspark",
            ["   ", "  ", " "],
            1,
            "failure",
            ["   ", "  "],
            "Found 2 row(s) where 'col1' length is not equal to 1.",
        ),
        # Whitespace length 3 - pandas
        (
            "pandas",
            ["   ", "  ", " "],
            3,
            "failure",
            ["  ", " "],
            "Found 2 row(s) where 'col1' length is not equal to 3.",
        ),
        # Whitespace length 3 - pyspark
        (
            "pyspark",
            ["   ", "  ", " "],
            3,
            "failure",
            ["  ", " "],
            "Found 2 row(s) where 'col1' length is not equal to 3.",
        ),
        # Whitespace in text - pandas
        ("pandas", ["a b", "c d", "e f"], 3, "success", None, None),
        # Whitespace in text - pyspark
        ("pyspark", ["a b", "c d", "e f"], 3, "success", None, None),
        # Whitespace mixed - pandas
        (
            "pandas",
            [" a ", "  a  ", "a"],
            3,
            "failure",
            ["  a  ", "a"],
            "Found 2 row(s) where 'col1' length is not equal to 3.",
        ),
        # Whitespace mixed - pyspark
        (
            "pyspark",
            [" a ", "  a  ", "a"],
            3,
            "failure",
            ["  a  ", "a"],
            "Found 2 row(s) where 'col1' length is not equal to 3.",
        ),
        # Special chars success - pandas
        ("pandas", ["@@@", "!!!", "###"], 3, "success", None, None),
        # Special chars success - pyspark
        ("pyspark", ["@@@", "!!!", "###"], 3, "success", None, None),
        # Special chars in text - pandas
        ("pandas", ["test@", "user!", "data#"], 5, "success", None, None),
        # Special chars in text - pyspark
        ("pyspark", ["test@", "user!", "data#"], 5, "success", None, None),
        # Special chars violations - pandas
        (
            "pandas",
            ["@", "!!", "###"],
            2,
            "failure",
            ["@", "###"],
            "Found 2 row(s) where 'col1' length is not equal to 2.",
        ),
        # Special chars violations - pyspark
        (
            "pyspark",
            ["@", "!!", "###"],
            2,
            "failure",
            ["@", "###"],
            "Found 2 row(s) where 'col1' length is not equal to 2.",
        ),
        # Numbers success - pandas
        ("pandas", ["123", "456", "789"], 3, "success", None, None),
        # Numbers success - pyspark
        ("pyspark", ["123", "456", "789"], 3, "success", None, None),
        # Numbers versions - pandas
        ("pandas", ["v1.0", "v2.0", "v3.0"], 4, "success", None, None),
        # Numbers versions - pyspark
        ("pyspark", ["v1.0", "v2.0", "v3.0"], 4, "success", None, None),
        # Numbers violations - pandas
        (
            "pandas",
            ["1", "12", "123"],
            2,
            "failure",
            ["1", "123"],
            "Found 2 row(s) where 'col1' length is not equal to 2.",
        ),
        # Numbers violations - pyspark
        (
            "pyspark",
            ["1", "12", "123"],
            2,
            "failure",
            ["1", "123"],
            "Found 2 row(s) where 'col1' length is not equal to 2.",
        ),
        # Length 10 success - pandas
        ("pandas", ["a" * 10, "b" * 10, "c" * 10], 10, "success", None, None),
        # Length 10 success - pyspark
        ("pyspark", ["a" * 10, "b" * 10, "c" * 10], 10, "success", None, None),
        # Length 20 success - pandas
        ("pandas", ["a" * 20, "b" * 20, "c" * 20], 20, "success", None, None),
        # Length 20 success - pyspark
        ("pyspark", ["a" * 20, "b" * 20, "c" * 20], 20, "success", None, None),
        # Length 10 violation - pandas
        (
            "pandas",
            ["a" * 10, "b" * 20, "c" * 10],
            10,
            "failure",
            ["b" * 20],
            "Found 1 row(s) where 'col1' length is not equal to 10.",
        ),
        # Length 10 violation - pyspark
        (
            "pyspark",
            ["a" * 10, "b" * 20, "c" * 10],
            10,
            "failure",
            ["b" * 20],
            "Found 1 row(s) where 'col1' length is not equal to 10.",
        ),
        # Long strings success - pandas
        ("pandas", ["a" * 100, "b" * 100, "c" * 100], 100, "success", None, None),
        # Long strings success - pyspark
        ("pyspark", ["a" * 100, "b" * 100, "c" * 100], 100, "success", None, None),
        # Long strings violations - pandas
        (
            "pandas",
            ["a" * 100, "b" * 99, "c" * 101],
            100,
            "failure",
            ["b" * 99, "c" * 101],
            "Found 2 row(s) where 'col1' length is not equal to 100.",
        ),
        # Long strings violations - pyspark
        (
            "pyspark",
            ["a" * 100, "b" * 99, "c" * 101],
            100,
            "failure",
            ["b" * 99, "c" * 101],
            "Found 2 row(s) where 'col1' length is not equal to 100.",
        ),
        # Mixed violations - pandas
        (
            "pandas",
            ["short", "exactly3", "way too long"],
            8,
            "failure",
            ["short", "way too long"],
            "Found 2 row(s) where 'col1' length is not equal to 8.",
        ),
        # Mixed violations - pyspark
        (
            "pyspark",
            ["short", "exactly3", "way too long"],
            8,
            "failure",
            ["short", "way too long"],
            "Found 2 row(s) where 'col1' length is not equal to 8.",
        ),
    ],
    ids=[
        "pandas_basic_success",
        "pyspark_basic_success",
        "pandas_success_length_2",
        "pyspark_success_length_2",
        "pandas_success_length_5",
        "pyspark_success_length_5",
        "pandas_basic_violations",
        "pyspark_basic_violations",
        "pandas_all_violations",
        "pyspark_all_violations",
        "pandas_mixed_violations",
        "pyspark_mixed_violations",
        "pandas_single_char_success",
        "pyspark_single_char_success",
        "pandas_single_char_violations",
        "pyspark_single_char_violations",
        "pandas_empty_string_success",
        "pyspark_empty_string_success",
        "pandas_empty_string_violations_length_0",
        "pyspark_empty_string_violations_length_0",
        "pandas_empty_string_violations_length_1",
        "pyspark_empty_string_violations_length_1",
        "pandas_whitespace_length_1",
        "pyspark_whitespace_length_1",
        "pandas_whitespace_length_3",
        "pyspark_whitespace_length_3",
        "pandas_whitespace_in_text",
        "pyspark_whitespace_in_text",
        "pandas_whitespace_mixed",
        "pyspark_whitespace_mixed",
        "pandas_special_chars_success",
        "pyspark_special_chars_success",
        "pandas_special_chars_in_text",
        "pyspark_special_chars_in_text",
        "pandas_special_chars_violations",
        "pyspark_special_chars_violations",
        "pandas_numbers_success",
        "pyspark_numbers_success",
        "pandas_numbers_versions",
        "pyspark_numbers_versions",
        "pandas_numbers_violations",
        "pyspark_numbers_violations",
        "pandas_length_10_success",
        "pyspark_length_10_success",
        "pandas_length_20_success",
        "pyspark_length_20_success",
        "pandas_length_10_violation",
        "pyspark_length_10_violation",
        "pandas_long_strings_success",
        "pyspark_long_strings_success",
        "pandas_long_strings_violations",
        "pyspark_long_strings_violations",
        "pandas_mixed_short_and_long",
        "pyspark_mixed_short_and_long",
    ],
)
def test_expectation_basic_scenarios(
    df_type, data, length, expected_result, expected_violations, expected_message, spark
):
    """
    Test the expectation for various scenarios across pandas and PySpark DataFrames.
    Tests both direct expectation validation and suite-based validation.
    Covers: success cases, violations, different length values, single character strings,
    empty strings, whitespace handling, special characters, numbers in strings,
    long strings, and mixed violations.
    """
    data_frame = create_dataframe(df_type, data, "col1", spark)

    # Test 1: Direct expectation validation
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationStringLengthEquals",
        column_name="col1",
        length=length,
    )

    result = expectation.validate(data_frame=data_frame)

    if expected_result == "success":
        assert str(result) == str(
            DataFrameExpectationSuccessMessage(expectation_name="ExpectationStringLengthEquals")
        ), f"Expected success message but got: {result}"
    else:  # failure
        expected_violations_df = create_dataframe(df_type, expected_violations, "col1", spark)
        expected_failure_message = DataFrameExpectationFailureMessage(
            expectation_str=str(expectation),
            data_frame_type=str(df_type),
            violations_data_frame=expected_violations_df,
            message=expected_message,
            limit_violations=5,
        )
        assert str(result) == str(expected_failure_message), (
            f"Expected failure message but got: {result}"
        )

    # Test 2: Suite-based validation
    expectations_suite = DataFrameExpectationsSuite().expect_string_length_equals(
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


@pytest.mark.parametrize(
    "df_type",
    ["pandas", "pyspark"],
    ids=["pandas", "pyspark"],
)
def test_column_missing_error(df_type, spark):
    """Test that an error is raised when the specified column is missing in both pandas and PySpark."""
    expected_message = "Column 'col1' does not exist in the DataFrame."

    if df_type == "pandas":
        data_frame = pd.DataFrame({"col2": ["foo", "bar", "baz"]})
    else:  # pyspark
        data_frame = spark.createDataFrame([("foo",), ("bar",), ("baz",)], ["col2"])

    # Test 1: Direct expectation validation
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationStringLengthEquals",
        column_name="col1",
        length=3,
    )
    result = expectation.validate(data_frame=data_frame)
    expected_failure = DataFrameExpectationFailureMessage(
        expectation_str=str(expectation),
        data_frame_type=str(df_type),
        message=expected_message,
    )
    assert str(result) == str(expected_failure), f"Expected failure message but got: {result}"

    # Test 2: Suite-based validation
    expectations_suite = DataFrameExpectationsSuite().expect_string_length_equals(
        column_name="col1", length=3
    )
    with pytest.raises(DataFrameExpectationsSuiteFailure):
        expectations_suite.build().run(data_frame=data_frame)


def test_large_dataset_performance():
    """Test the expectation with a larger dataset to ensure performance."""
    # Create a larger dataset with 10,000 rows all with length 10
    large_data = ["a" * 10 for _ in range(10000)]
    data_frame = pd.DataFrame({"col1": large_data})

    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationStringLengthEquals",
        column_name="col1",
        length=10,
    )

    result = expectation.validate(data_frame=data_frame)
    # Should succeed as all values have length 10
    assert isinstance(result, DataFrameExpectationSuccessMessage), (
        f"Large dataset test failed: expected success but got {type(result)}"
    )
