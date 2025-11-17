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
        expectation_name="ExpectationStringLengthGreaterThan",
        column_name="col1",
        length=3,
    )
    assert expectation.get_expectation_name() == "ExpectationStringLengthGreaterThan", (
        f"Expected 'ExpectationStringLengthGreaterThan' but got: {expectation.get_expectation_name()}"
    )


@pytest.mark.parametrize(
    "df_type, data, length, expected_result, expected_violations, expected_message",
    [
        # Basic success scenarios - pandas
        ("pandas", ["foobar", "bazz", "hello"], 3, "success", None, None),
        ("pandas", ["abc", "abcd", "abcde"], 2, "success", None, None),
        ("pandas", ["hello", "world", "testing"], 4, "success", None, None),
        # Basic success scenarios - pyspark
        ("pyspark", ["foobar", "bazz", "hello"], 3, "success", None, None),
        ("pyspark", ["abc", "abcd", "abcde"], 2, "success", None, None),
        ("pyspark", ["hello", "world", "testing"], 4, "success", None, None),
        # Basic violation scenarios - pandas
        (
            "pandas",
            ["foo", "bar", "bazzz"],
            3,
            "failure",
            ["foo", "bar"],
            "Found 2 row(s) where 'col1' length is not greater than 3.",
        ),
        (
            "pandas",
            ["a", "ab", "abc"],
            5,
            "failure",
            ["a", "ab", "abc"],
            "Found 3 row(s) where 'col1' length is not greater than 5.",
        ),
        (
            "pandas",
            ["test", "testing", "t"],
            4,
            "failure",
            ["test", "t"],
            "Found 2 row(s) where 'col1' length is not greater than 4.",
        ),
        # Basic violation scenarios - pyspark
        (
            "pyspark",
            ["foo", "bar", "bazzz"],
            3,
            "failure",
            ["foo", "bar"],
            "Found 2 row(s) where 'col1' length is not greater than 3.",
        ),
        (
            "pyspark",
            ["a", "ab", "abc"],
            5,
            "failure",
            ["a", "ab", "abc"],
            "Found 3 row(s) where 'col1' length is not greater than 5.",
        ),
        # Boundary conditions - exact length - pandas
        (
            "pandas",
            ["abc", "abcd", "abcde"],
            3,
            "failure",
            ["abc"],
            "Found 1 row(s) where 'col1' length is not greater than 3.",
        ),
        ("pandas", ["abcd", "abcde", "abcdef"], 3, "success", None, None),
        # Boundary conditions - exact length - pyspark
        (
            "pyspark",
            ["abc", "abcd", "abcde"],
            3,
            "failure",
            ["abc"],
            "Found 1 row(s) where 'col1' length is not greater than 3.",
        ),
        ("pyspark", ["abcd", "abcde", "abcdef"], 3, "success", None, None),
        # Zero length - pandas
        ("pandas", ["a", "ab", "abc"], 0, "success", None, None),
        (
            "pandas",
            ["", "a", "ab"],
            0,
            "failure",
            [""],
            "Found 1 row(s) where 'col1' length is not greater than 0.",
        ),
        # Zero length - pyspark
        ("pyspark", ["a", "ab", "abc"], 0, "success", None, None),
        # Single character strings - pandas
        ("pandas", ["ab", "abc", "abcd"], 1, "success", None, None),
        (
            "pandas",
            ["a", "b", "cd"],
            1,
            "failure",
            ["a", "b"],
            "Found 2 row(s) where 'col1' length is not greater than 1.",
        ),
        # Single character strings - pyspark
        ("pyspark", ["ab", "abc", "abcd"], 1, "success", None, None),
        # Empty strings - pandas
        (
            "pandas",
            ["", "a", "ab"],
            1,
            "failure",
            ["", "a"],
            "Found 2 row(s) where 'col1' length is not greater than 1.",
        ),
        # Empty strings - pyspark
        (
            "pyspark",
            ["", "a", "ab"],
            1,
            "failure",
            ["", "a"],
            "Found 2 row(s) where 'col1' length is not greater than 1.",
        ),
        # Whitespace handling - pandas
        ("pandas", ["    ", "   ", "   "], 2, "success", None, None),
        ("pandas", ["a b c", "a b ", "a  b"], 3, "success", None, None),
        (
            "pandas",
            [" a ", "  a  ", "a"],
            3,
            "failure",
            [" a ", "a"],
            "Found 2 row(s) where 'col1' length is not greater than 3.",
        ),
        # Whitespace handling - pyspark
        ("pyspark", ["    ", "   ", "   "], 2, "success", None, None),
        ("pyspark", ["a b c", "a b ", "a  b"], 3, "success", None, None),
        # Special characters - pandas
        ("pandas", ["@@@@", "!!!!", "####"], 3, "success", None, None),
        ("pandas", ["test@@", "user!!", "data##"], 5, "success", None, None),
        (
            "pandas",
            ["@", "!!", "###"],
            2,
            "failure",
            ["@", "!!"],
            "Found 2 row(s) where 'col1' length is not greater than 2.",
        ),
        # Special characters - pyspark
        ("pyspark", ["@@@@", "!!!!", "####"], 3, "success", None, None),
        ("pyspark", ["test@@", "user!!", "data##"], 5, "success", None, None),
        # Numbers in strings - pandas
        ("pandas", ["1234", "5678", "9012"], 3, "success", None, None),
        ("pandas", ["v1.0.0", "v2.0.0", "v3.0.0"], 5, "success", None, None),
        (
            "pandas",
            ["1", "12", "123"],
            2,
            "failure",
            ["1", "12"],
            "Found 2 row(s) where 'col1' length is not greater than 2.",
        ),
        # Numbers in strings - pyspark
        ("pyspark", ["1234", "5678", "9012"], 3, "success", None, None),
        ("pyspark", ["v1.0.0", "v2.0.0", "v3.0.0"], 5, "success", None, None),
        # Different length thresholds - pandas
        ("pandas", ["a" * 11, "b" * 12, "c" * 13], 10, "success", None, None),
        ("pandas", ["a" * 21, "b" * 22, "c" * 23], 20, "success", None, None),
        (
            "pandas",
            ["a" * 10, "b" * 11, "c" * 12],
            10,
            "failure",
            ["a" * 10],
            "Found 1 row(s) where 'col1' length is not greater than 10.",
        ),
        # Different length thresholds - pyspark
        ("pyspark", ["a" * 11, "b" * 12, "c" * 13], 10, "success", None, None),
        ("pyspark", ["a" * 21, "b" * 22, "c" * 23], 20, "success", None, None),
        # Long strings - pandas
        ("pandas", ["a" * 101, "b" * 102, "c" * 103], 100, "success", None, None),
        (
            "pandas",
            ["a" * 100, "b" * 101, "c" * 102],
            100,
            "failure",
            ["a" * 100],
            "Found 1 row(s) where 'col1' length is not greater than 100.",
        ),
        # Long strings - pyspark
        ("pyspark", ["a" * 101, "b" * 102, "c" * 103], 100, "success", None, None),
        # Mixed violations - pandas
        (
            "pandas",
            ["short", "exactly8", "much longer string"],
            8,
            "failure",
            ["short", "exactly8"],
            "Found 2 row(s) where 'col1' length is not greater than 8.",
        ),
        # Mixed violations - pyspark
        (
            "pyspark",
            ["short", "exactly8", "much longer string"],
            8,
            "failure",
            ["short", "exactly8"],
            "Found 2 row(s) where 'col1' length is not greater than 8.",
        ),
    ],
    ids=[
        "pandas_basic_success",
        "pandas_success_length_2",
        "pandas_success_length_4",
        "pyspark_basic_success",
        "pyspark_success_length_2",
        "pyspark_success_length_4",
        "pandas_basic_violations",
        "pandas_all_violations",
        "pandas_mixed_violations",
        "pyspark_basic_violations",
        "pyspark_all_violations",
        "pandas_boundary_exact_violation",
        "pandas_boundary_success",
        "pyspark_boundary_exact_violation",
        "pyspark_boundary_success",
        "pandas_zero_length_success",
        "pandas_zero_length_violation",
        "pyspark_zero_length_success",
        "pandas_single_char_success",
        "pandas_single_char_violations",
        "pyspark_single_char_success",
        "pandas_empty_string_violations",
        "pyspark_empty_string_violations",
        "pandas_whitespace_success",
        "pandas_whitespace_in_text",
        "pandas_whitespace_violations",
        "pyspark_whitespace_success",
        "pyspark_whitespace_in_text",
        "pandas_special_chars_success",
        "pandas_special_chars_in_text",
        "pandas_special_chars_violations",
        "pyspark_special_chars_success",
        "pyspark_special_chars_in_text",
        "pandas_numbers_success",
        "pandas_numbers_versions",
        "pandas_numbers_violations",
        "pyspark_numbers_success",
        "pyspark_numbers_versions",
        "pandas_length_10_success",
        "pandas_length_20_success",
        "pandas_length_10_violation",
        "pyspark_length_10_success",
        "pyspark_length_20_success",
        "pandas_long_strings_success",
        "pandas_long_strings_violation",
        "pyspark_long_strings_success",
        "pandas_mixed_short_and_exact",
        "pyspark_mixed_short_and_exact",
    ],
)
def test_expectation_basic_scenarios(
    df_type, data, length, expected_result, expected_violations, expected_message, spark
):
    """
    Test the expectation for various scenarios across pandas and PySpark DataFrames.
    Tests both direct expectation validation and suite-based validation.
    Covers: success cases, violations, boundary conditions (exact length),
    zero length threshold, single character strings, empty strings,
    whitespace handling, special characters, numbers in strings, long strings,
    and mixed violations (both short and exact length).
    """
    data_frame = create_dataframe(df_type, data, "col1", spark)

    # Test 1: Direct expectation validation
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationStringLengthGreaterThan",
        column_name="col1",
        length=length,
    )

    result = expectation.validate(data_frame=data_frame)

    if expected_result == "success":
        assert str(result) == str(
            DataFrameExpectationSuccessMessage(
                expectation_name="ExpectationStringLengthGreaterThan"
            )
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
    expectations_suite = DataFrameExpectationsSuite().expect_string_length_greater_than(
        column_name="col1", length=length
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
        data_frame = pd.DataFrame({"col2": ["foobar", "bazz", "hello"]})
    else:  # pyspark
        data_frame = spark.createDataFrame([("foobar",), ("bazz",), ("hello",)], ["col2"])

    # Test 1: Direct expectation validation
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationStringLengthGreaterThan",
        column_name="col1",
        length=3,
    )
    result = expectation.validate(data_frame=data_frame)
    expected_failure = DataFrameExpectationFailureMessage(
        expectation_str=str(expectation),
        data_frame_type=get_df_type_enum(df_type),
        message=expected_message,
    )
    assert str(result) == str(expected_failure), f"Expected failure message but got: {result}"

    # Test 2: Suite-based validation
    expectations_suite = DataFrameExpectationsSuite().expect_string_length_greater_than(
        column_name="col1", length=3
    )
    with pytest.raises(DataFrameExpectationsSuiteFailure):
        expectations_suite.build().run(data_frame=data_frame)


def test_large_dataset_performance():
    """Test the expectation with a larger dataset to ensure performance."""
    # Create a larger dataset with 10,000 rows all with length > 10
    large_data = ["a" * 15 for _ in range(10000)]
    data_frame = pd.DataFrame({"col1": large_data})

    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationStringLengthGreaterThan",
        column_name="col1",
        length=10,
    )

    result = expectation.validate(data_frame=data_frame)
    # Should succeed as all values have length > 10
    assert isinstance(result, DataFrameExpectationSuccessMessage), (
        f"Large dataset test failed: expected success but got {type(result)}"
    )
