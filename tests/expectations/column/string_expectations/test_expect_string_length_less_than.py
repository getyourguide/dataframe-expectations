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
    return spark.createDataFrame([(item,) for item in data], [column_name])


def test_expectation_name():
    """Test that the expectation name is correctly returned."""
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationStringLengthLessThan",
        column_name="col1",
        length=5,
    )
    assert expectation.get_expectation_name() == "ExpectationStringLengthLessThan", (
        f"Expected 'ExpectationStringLengthLessThan' but got: {expectation.get_expectation_name()}"
    )


@pytest.mark.parametrize(
    "df_type, data, length, expected_result, expected_violations, expected_message",
    [
        # Basic success - pandas
        ("pandas", ["foo", "bar", "baz"], 5, "success", None, None),
        # Success length 2 - pandas
        ("pandas", ["ab", "cd", "ef"], 3, "success", None, None),
        # Success length 4 - pandas
        ("pandas", ["abc", "def", "ghi"], 5, "success", None, None),
        # Basic violations - pandas
        (
            "pandas",
            ["foobar", "bar", "bazbaz"],
            5,
            "failure",
            ["foobar", "bazbaz"],
            "Found 2 row(s) where 'col1' length is not less than 5.",
        ),
        # All violations - pandas
        (
            "pandas",
            ["testing", "longer", "strings"],
            5,
            "failure",
            ["testing", "longer", "strings"],
            "Found 3 row(s) where 'col1' length is not less than 5.",
        ),
        # Mixed violations - pandas
        (
            "pandas",
            ["ok", "fail", "good"],
            3,
            "failure",
            ["fail", "good"],
            "Found 2 row(s) where 'col1' length is not less than 3.",
        ),
        # Boundary exact violation - pandas
        (
            "pandas",
            ["test", "exam", "demo"],
            4,
            "failure",
            ["test", "exam", "demo"],
            "Found 3 row(s) where 'col1' length is not less than 4.",
        ),
        # Boundary success - pandas
        ("pandas", ["tes", "exa", "dem"], 4, "success", None, None),
        # Zero length success - pandas
        ("pandas", ["", "", ""], 1, "success", None, None),
        # Zero length violation - pandas
        (
            "pandas",
            ["a", "b", "c"],
            1,
            "failure",
            ["a", "b", "c"],
            "Found 3 row(s) where 'col1' length is not less than 1.",
        ),
        # Single char success - pandas
        ("pandas", ["a", "b", "c"], 2, "success", None, None),
        # Single char violations - pandas
        (
            "pandas",
            ["a", "b", "c"],
            1,
            "failure",
            ["a", "b", "c"],
            "Found 3 row(s) where 'col1' length is not less than 1.",
        ),
        # Empty strings - pandas
        ("pandas", ["", "", ""], 5, "success", None, None),
        # Whitespace success - pandas
        ("pandas", ["  ", " ", ""], 3, "success", None, None),
        # Whitespace in text - pandas
        ("pandas", ["a b", "a", "ab"], 4, "success", None, None),
        # Whitespace violations - pandas
        (
            "pandas",
            ["    ", "     ", "      "],
            3,
            "failure",
            ["    ", "     ", "      "],
            "Found 3 row(s) where 'col1' length is not less than 3.",
        ),
        # Special chars success - pandas
        ("pandas", ["@#$", "!^&", "*()"], 4, "success", None, None),
        # Special chars in text - pandas
        ("pandas", ["test@", "ex!m", "de#o"], 6, "success", None, None),
        # Special chars violations - pandas
        (
            "pandas",
            ["@#$%^", "&*()_", "+=-[]"],
            4,
            "failure",
            ["@#$%^", "&*()_", "+=-[]"],
            "Found 3 row(s) where 'col1' length is not less than 4.",
        ),
        # Numbers success - pandas
        ("pandas", ["123", "456", "789"], 4, "success", None, None),
        # Numbers versions - pandas
        ("pandas", ["v1.0", "v2.1", "v3.5"], 5, "success", None, None),
        # Numbers violations - pandas
        (
            "pandas",
            ["12345", "67890", "11111"],
            4,
            "failure",
            ["12345", "67890", "11111"],
            "Found 3 row(s) where 'col1' length is not less than 4.",
        ),
        # Length 10 success - pandas
        ("pandas", ["a" * 9, "b" * 8, "c" * 7], 11, "success", None, None),
        # Length 20 success - pandas
        (
            "pandas",
            ["test " * 3, "demo " * 3, "exam " * 3],
            21,
            "success",
            None,
            None,
        ),
        # Length 10 violation - pandas
        (
            "pandas",
            ["x" * 11, "y" * 12, "z" * 13],
            10,
            "failure",
            ["x" * 11, "y" * 12, "z" * 13],
            "Found 3 row(s) where 'col1' length is not less than 10.",
        ),
        # Long strings success - pandas
        (
            "pandas",
            ["x" * 99, "y" * 98, "z" * 97],
            101,
            "success",
            None,
            None,
        ),
        # Long strings violation - pandas
        (
            "pandas",
            ["x" * 101, "y" * 102, "z" * 103],
            100,
            "failure",
            ["x" * 101, "y" * 102, "z" * 103],
            "Found 3 row(s) where 'col1' length is not less than 100.",
        ),
        # Mixed violations - pandas
        (
            "pandas",
            ["ab", "abcd", "abc"],
            4,
            "failure",
            ["abcd"],
            "Found 1 row(s) where 'col1' length is not less than 4.",
        ),
    ],
    ids=[
        "pandas_basic_success",
        "pandas_success_length_2",
        "pandas_success_length_4",
        "pandas_basic_violations",
        "pandas_all_violations",
        "pandas_mixed_violations",
        "pandas_boundary_exact_violation",
        "pandas_boundary_success",
        "pandas_zero_length_success",
        "pandas_zero_length_violation",
        "pandas_single_char_success",
        "pandas_single_char_violations",
        "pandas_empty_string_success",
        "pandas_whitespace_success",
        "pandas_whitespace_in_text",
        "pandas_whitespace_violations",
        "pandas_special_chars_success",
        "pandas_special_chars_in_text",
        "pandas_special_chars_violations",
        "pandas_numbers_success",
        "pandas_numbers_versions",
        "pandas_numbers_violations",
        "pandas_length_10_success",
        "pandas_length_20_success",
        "pandas_length_10_violation",
        "pandas_long_strings_success",
        "pandas_long_strings_violation",
        "pandas_mixed_short_and_exact",
    ],
)
def test_expectation_basic_scenarios_pandas(
    df_type, data, length, expected_result, expected_violations, expected_message
):
    """
    Test the expectation for various scenarios across pandas DataFrames.
    Tests both direct expectation validation and suite-based validation.
    Covers: success cases, violations, boundary conditions (exact length),
    zero length threshold, single character strings, empty strings,
    whitespace handling, special characters, numbers in strings, long strings,
    and mixed violations (both short and exact length).
    """
    data_frame = pd.DataFrame({"col1": data})

    # Test 1: Direct expectation validation
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationStringLengthLessThan",
        column_name="col1",
        length=length,
    )

    result = expectation.validate(data_frame=data_frame)

    if expected_result == "success":
        assert str(result) == str(
            DataFrameExpectationSuccessMessage(expectation_name="ExpectationStringLengthLessThan")
        ), f"Expected success message but got: {result}"
    else:  # failure
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
    expectations_suite = DataFrameExpectationsSuite().expect_string_length_less_than(
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


@pytest.mark.pyspark
@pytest.mark.parametrize(
    "df_type, data, length, expected_result, expected_violations, expected_message",
    [
        # Basic success - pyspark
        ("pyspark", ["foo", "bar", "baz"], 5, "success", None, None),
        # Success length 2 - pyspark
        ("pyspark", ["ab", "cd", "ef"], 3, "success", None, None),
        # Success length 4 - pyspark
        ("pyspark", ["abc", "def", "ghi"], 5, "success", None, None),
        # Basic violations - pyspark
        (
            "pyspark",
            ["foobar", "bar", "bazbaz"],
            5,
            "failure",
            ["foobar", "bazbaz"],
            "Found 2 row(s) where 'col1' length is not less than 5.",
        ),
        # All violations - pyspark
        (
            "pyspark",
            ["testing", "longer", "strings"],
            5,
            "failure",
            ["testing", "longer", "strings"],
            "Found 3 row(s) where 'col1' length is not less than 5.",
        ),
        # Mixed violations - pyspark
        (
            "pyspark",
            ["ok", "fail", "good"],
            3,
            "failure",
            ["fail", "good"],
            "Found 2 row(s) where 'col1' length is not less than 3.",
        ),
        # Boundary exact violation - pyspark
        (
            "pyspark",
            ["test", "exam", "demo"],
            4,
            "failure",
            ["test", "exam", "demo"],
            "Found 3 row(s) where 'col1' length is not less than 4.",
        ),
        # Boundary success - pyspark
        ("pyspark", ["tes", "exa", "dem"], 4, "success", None, None),
        # Zero length success - pyspark
        ("pyspark", ["", "", ""], 1, "success", None, None),
        # Zero length violation - pyspark
        (
            "pyspark",
            ["a", "b", "c"],
            1,
            "failure",
            ["a", "b", "c"],
            "Found 3 row(s) where 'col1' length is not less than 1.",
        ),
        # Single char success - pyspark
        ("pyspark", ["a", "b", "c"], 2, "success", None, None),
        # Single char violations - pyspark
        (
            "pyspark",
            ["a", "b", "c"],
            1,
            "failure",
            ["a", "b", "c"],
            "Found 3 row(s) where 'col1' length is not less than 1.",
        ),
        # Empty strings - pyspark
        ("pyspark", ["", "", ""], 5, "success", None, None),
        # Whitespace success - pyspark
        ("pyspark", ["  ", " ", ""], 3, "success", None, None),
        # Whitespace in text - pyspark
        ("pyspark", ["a b", "a", "ab"], 4, "success", None, None),
        # Whitespace violations - pyspark
        (
            "pyspark",
            ["    ", "     ", "      "],
            3,
            "failure",
            ["    ", "     ", "      "],
            "Found 3 row(s) where 'col1' length is not less than 3.",
        ),
        # Special chars success - pyspark
        ("pyspark", ["@#$", "!^&", "*()"], 4, "success", None, None),
        # Special chars in text - pyspark
        ("pyspark", ["test@", "ex!m", "de#o"], 6, "success", None, None),
        # Special chars violations - pyspark
        (
            "pyspark",
            ["@#$%^", "&*()_", "+=-[]"],
            4,
            "failure",
            ["@#$%^", "&*()_", "+=-[]"],
            "Found 3 row(s) where 'col1' length is not less than 4.",
        ),
        # Numbers success - pyspark
        ("pyspark", ["123", "456", "789"], 4, "success", None, None),
        # Numbers versions - pyspark
        ("pyspark", ["v1.0", "v2.1", "v3.5"], 5, "success", None, None),
        # Numbers violations - pyspark
        (
            "pyspark",
            ["12345", "67890", "11111"],
            4,
            "failure",
            ["12345", "67890", "11111"],
            "Found 3 row(s) where 'col1' length is not less than 4.",
        ),
        # Length 10 success - pyspark
        ("pyspark", ["a" * 9, "b" * 8, "c" * 7], 11, "success", None, None),
        # Length 20 success - pyspark
        (
            "pyspark",
            ["test " * 3, "demo " * 3, "exam " * 3],
            21,
            "success",
            None,
            None,
        ),
        # Length 10 violation - pyspark
        (
            "pyspark",
            ["x" * 11, "y" * 12, "z" * 13],
            10,
            "failure",
            ["x" * 11, "y" * 12, "z" * 13],
            "Found 3 row(s) where 'col1' length is not less than 10.",
        ),
        # Long strings success - pyspark
        (
            "pyspark",
            ["a" * 99, "b" * 98, "c" * 97],
            101,
            "success",
            None,
            None,
        ),
        # Long strings violation - pyspark
        (
            "pyspark",
            ["x" * 101, "y" * 102, "z" * 103],
            100,
            "failure",
            ["x" * 101, "y" * 102, "z" * 103],
            "Found 3 row(s) where 'col1' length is not less than 100.",
        ),
        # Mixed violations - pyspark
        (
            "pyspark",
            ["ab", "abcd", "abc"],
            4,
            "failure",
            ["abcd"],
            "Found 1 row(s) where 'col1' length is not less than 4.",
        ),
    ],
    ids=[
        "pyspark_basic_success",
        "pyspark_success_length_2",
        "pyspark_success_length_4",
        "pyspark_basic_violations",
        "pyspark_all_violations",
        "pyspark_mixed_violations",
        "pyspark_boundary_exact_violation",
        "pyspark_boundary_success",
        "pyspark_zero_length_success",
        "pyspark_zero_length_violation",
        "pyspark_single_char_success",
        "pyspark_single_char_violations",
        "pyspark_empty_string_success",
        "pyspark_whitespace_success",
        "pyspark_whitespace_in_text",
        "pyspark_whitespace_violations",
        "pyspark_special_chars_success",
        "pyspark_special_chars_in_text",
        "pyspark_special_chars_violations",
        "pyspark_numbers_success",
        "pyspark_numbers_versions",
        "pyspark_numbers_violations",
        "pyspark_length_10_success",
        "pyspark_length_20_success",
        "pyspark_length_10_violation",
        "pyspark_long_strings_success",
        "pyspark_long_strings_violation",
        "pyspark_mixed_short_and_exact",
    ],
)
def test_expectation_basic_scenarios_pyspark(
    df_type, data, length, expected_result, expected_violations, expected_message, spark
):
    """
    Test the expectation for various scenarios across PySpark DataFrames.
    Tests both direct expectation validation and suite-based validation.
    Covers: success cases, violations, boundary conditions (exact length),
    zero length threshold, single character strings, empty strings,
    whitespace handling, special characters, numbers in strings, long strings,
    and mixed violations (both short and exact length).
    """
    data_frame = create_pyspark_dataframe(data, "col1", spark)

    # Test 1: Direct expectation validation
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationStringLengthLessThan",
        column_name="col1",
        length=length,
    )

    result = expectation.validate(data_frame=data_frame)

    if expected_result == "success":
        assert str(result) == str(
            DataFrameExpectationSuccessMessage(expectation_name="ExpectationStringLengthLessThan")
        ), f"Expected success message but got: {result}"
    else:  # failure
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
    expectations_suite = DataFrameExpectationsSuite().expect_string_length_less_than(
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


def test_column_missing_error_pandas():
    """Test that missing column raises appropriate error in pandas."""
    expected_message = "Column 'col1' does not exist in the DataFrame."

    data_frame = pd.DataFrame({"col2": ["foo", "bar", "baz"]})

    # Test 1: Direct expectation validation
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationStringLengthLessThan",
        column_name="col1",
        length=5,
    )
    result = expectation.validate(data_frame=data_frame)
    expected_failure = DataFrameExpectationFailureMessage(
        expectation_str=str(expectation),
        data_frame_type=DataFrameType.PANDAS,
        message=expected_message,
    )
    assert str(result) == str(expected_failure), f"Expected failure message but got: {result}"

    # Test 2: Suite-based validation
    expectations_suite = DataFrameExpectationsSuite().expect_string_length_less_than(
        column_name="col1", length=5
    )
    with pytest.raises(DataFrameExpectationsSuiteFailure):
        expectations_suite.build().run(data_frame=data_frame)


@pytest.mark.pyspark
def test_column_missing_error_pyspark(spark):
    """Test that missing column raises appropriate error in PySpark."""
    expected_message = "Column 'col1' does not exist in the DataFrame."

    data_frame = spark.createDataFrame([("foo",), ("bar",), ("baz",)], ["col2"])

    # Test 1: Direct expectation validation
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationStringLengthLessThan",
        column_name="col1",
        length=5,
    )
    result = expectation.validate(data_frame=data_frame)
    expected_failure = DataFrameExpectationFailureMessage(
        expectation_str=str(expectation),
        data_frame_type=DataFrameType.PYSPARK,
        message=expected_message,
    )
    assert str(result) == str(expected_failure), f"Expected failure message but got: {result}"

    # Test 2: Suite-based validation
    expectations_suite = DataFrameExpectationsSuite().expect_string_length_less_than(
        column_name="col1", length=5
    )
    with pytest.raises(DataFrameExpectationsSuiteFailure):
        expectations_suite.build().run(data_frame=data_frame)


def test_large_dataset_performance():
    """Test the expectation with a larger dataset to ensure performance."""
    # Create a larger dataset with 10,000 rows all with length < 10
    large_data = ["abc" * 2 for _ in range(10000)]
    data_frame = pd.DataFrame({"col1": large_data})

    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationStringLengthLessThan",
        column_name="col1",
        length=10,
    )

    result = expectation.validate(data_frame=data_frame)
    # Should succeed as all values have length < 10
    assert isinstance(result, DataFrameExpectationSuccessMessage), (
        f"Large dataset test failed: expected success but got {type(result)}"
    )
