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
        expectation_name="ExpectationStringLengthBetween",
        column_name="col1",
        min_length=3,
        max_length=6,
    )
    assert expectation.get_expectation_name() == "ExpectationStringLengthBetween", (
        f"Expected 'ExpectationStringLengthBetween' but got: {expectation.get_expectation_name()}"
    )


@pytest.mark.parametrize(
    "df_type, data, min_length, max_length, expected_result, expected_violations, expected_message",
    [
        # Basic success scenarios - pandas
        ("pandas", ["foo", "bazz", "hello", "foobar"], 3, 6, "success", None, None),
        ("pandas", ["ab", "abc", "abcd"], 2, 4, "success", None, None),
        ("pandas", ["test", "data", "valid"], 4, 5, "success", None, None),
        # Basic success scenarios - pyspark
        ("pyspark", ["foo", "bazz", "hello", "foobar"], 3, 6, "success", None, None),
        ("pyspark", ["ab", "abc", "abcd"], 2, 4, "success", None, None),
        ("pyspark", ["test", "data", "valid"], 4, 5, "success", None, None),
        # Basic violation scenarios - pandas
        (
            "pandas",
            ["fo", "bazz", "hellothere", "foobar"],
            3,
            6,
            "failure",
            ["fo", "hellothere"],
            "Found 2 row(s) where 'col1' length is not between 3 and 6.",
        ),
        (
            "pandas",
            ["a", "ab", "abc"],
            5,
            10,
            "failure",
            ["a", "ab", "abc"],
            "Found 3 row(s) where 'col1' length is not between 5 and 10.",
        ),
        (
            "pandas",
            ["test", "testing", "t"],
            2,
            4,
            "failure",
            ["testing", "t"],
            "Found 2 row(s) where 'col1' length is not between 2 and 4.",
        ),
        # Basic violation scenarios - pyspark
        (
            "pyspark",
            ["fo", "bazz", "hellothere", "foobar"],
            3,
            6,
            "failure",
            ["fo", "hellothere"],
            "Found 2 row(s) where 'col1' length is not between 3 and 6.",
        ),
        (
            "pyspark",
            ["a", "ab", "abc"],
            5,
            10,
            "failure",
            ["a", "ab", "abc"],
            "Found 3 row(s) where 'col1' length is not between 5 and 10.",
        ),
        # Boundary conditions - exact min length - pandas
        ("pandas", ["abc", "abcd", "abcde"], 3, 5, "success", None, None),
        (
            "pandas",
            ["ab", "abc", "abcd"],
            3,
            5,
            "failure",
            ["ab"],
            "Found 1 row(s) where 'col1' length is not between 3 and 5.",
        ),
        # Boundary conditions - exact max length - pandas
        ("pandas", ["abc", "abcd", "abcde"], 3, 5, "success", None, None),
        (
            "pandas",
            ["abc", "abcd", "abcdef"],
            3,
            5,
            "failure",
            ["abcdef"],
            "Found 1 row(s) where 'col1' length is not between 3 and 5.",
        ),
        # Boundary conditions - pyspark
        ("pyspark", ["abc", "abcd", "abcde"], 3, 5, "success", None, None),
        (
            "pyspark",
            ["ab", "abc", "abcd"],
            3,
            5,
            "failure",
            ["ab"],
            "Found 1 row(s) where 'col1' length is not between 3 and 5.",
        ),
        # Min equals max - pandas
        ("pandas", ["abc", "def", "ghi"], 3, 3, "success", None, None),
        (
            "pandas",
            ["ab", "abc", "abcd"],
            3,
            3,
            "failure",
            ["ab", "abcd"],
            "Found 2 row(s) where 'col1' length is not between 3 and 3.",
        ),
        # Min equals max - pyspark
        ("pyspark", ["abc", "def", "ghi"], 3, 3, "success", None, None),
        # Single character strings - pandas
        ("pandas", ["a", "b", "c"], 1, 1, "success", None, None),
        (
            "pandas",
            ["a", "ab", "abc"],
            1,
            1,
            "failure",
            ["ab", "abc"],
            "Found 2 row(s) where 'col1' length is not between 1 and 1.",
        ),
        # Single character strings - pyspark
        ("pyspark", ["a", "b", "c"], 1, 1, "success", None, None),
        # Empty strings - pandas
        (
            "pandas",
            ["", "a", "ab"],
            1,
            3,
            "failure",
            [""],
            "Found 1 row(s) where 'col1' length is not between 1 and 3.",
        ),
        ("pandas", ["", "a", "ab"], 0, 3, "success", None, None),
        # Empty strings - pyspark
        (
            "pyspark",
            ["", "a", "ab"],
            1,
            3,
            "failure",
            [""],
            "Found 1 row(s) where 'col1' length is not between 1 and 3.",
        ),
        # Whitespace handling - pandas
        ("pandas", ["   ", "  ", " "], 1, 3, "success", None, None),
        ("pandas", ["a b", "a  b", "a   b"], 3, 5, "success", None, None),
        (
            "pandas",
            [" a ", "  a  ", "a"],
            4,
            6,
            "failure",
            [" a ", "a"],
            "Found 2 row(s) where 'col1' length is not between 4 and 6.",
        ),
        # Whitespace handling - pyspark
        ("pyspark", ["   ", "  ", " "], 1, 3, "success", None, None),
        ("pyspark", ["a b", "a  b", "a   b"], 3, 5, "success", None, None),
        # Special characters - pandas
        ("pandas", ["@@@", "!!!", "###"], 3, 3, "success", None, None),
        ("pandas", ["test@", "user!", "admin#"], 5, 6, "success", None, None),
        (
            "pandas",
            ["@", "!!", "###"],
            2,
            2,
            "failure",
            ["@", "###"],
            "Found 2 row(s) where 'col1' length is not between 2 and 2.",
        ),
        # Special characters - pyspark
        ("pyspark", ["@@@", "!!!", "###"], 3, 3, "success", None, None),
        ("pyspark", ["test@", "user!", "admin#"], 5, 6, "success", None, None),
        # Numbers in strings - pandas
        ("pandas", ["123", "456", "789"], 3, 3, "success", None, None),
        ("pandas", ["v1.0", "v2.0", "v10.0"], 4, 6, "success", None, None),
        (
            "pandas",
            ["1", "12", "123456"],
            2,
            4,
            "failure",
            ["1", "123456"],
            "Found 2 row(s) where 'col1' length is not between 2 and 4.",
        ),
        # Numbers in strings - pyspark
        ("pyspark", ["123", "456", "789"], 3, 3, "success", None, None),
        ("pyspark", ["v1.0", "v2.0", "v10.0"], 4, 6, "success", None, None),
        # Long strings - pandas
        ("pandas", ["a" * 100, "b" * 100, "c" * 100], 100, 100, "success", None, None),
        (
            "pandas",
            ["a" * 50, "b" * 100, "c" * 150],
            100,
            100,
            "failure",
            ["a" * 50, "c" * 150],
            "Found 2 row(s) where 'col1' length is not between 100 and 100.",
        ),
        # Long strings - pyspark
        ("pyspark", ["a" * 100, "b" * 100, "c" * 100], 100, 100, "success", None, None),
        # Wide range - pandas
        ("pandas", ["a", "ab", "a" * 50, "a" * 100], 1, 100, "success", None, None),
        # Wide range - pyspark
        ("pyspark", ["a", "ab", "a" * 50, "a" * 100], 1, 100, "success", None, None),
        # Zero min length - pandas
        ("pandas", ["", "a", "ab", "abc"], 0, 3, "success", None, None),
        (
            "pandas",
            ["", "a", "abcd"],
            0,
            3,
            "failure",
            ["abcd"],
            "Found 1 row(s) where 'col1' length is not between 0 and 3.",
        ),
        # Zero min length - pyspark
        ("pyspark", ["", "a", "ab", "abc"], 0, 3, "success", None, None),
    ],
    ids=[
        "pandas_basic_success",
        "pandas_success_2_4",
        "pandas_success_4_5",
        "pyspark_basic_success",
        "pyspark_success_2_4",
        "pyspark_success_4_5",
        "pandas_basic_violations",
        "pandas_all_violations",
        "pandas_one_violation",
        "pyspark_basic_violations",
        "pyspark_all_violations",
        "pandas_boundary_min_success",
        "pandas_boundary_min_violation",
        "pandas_boundary_max_success",
        "pandas_boundary_max_violation",
        "pyspark_boundary_min_success",
        "pyspark_boundary_min_violation",
        "pandas_min_equals_max_success",
        "pandas_min_equals_max_violations",
        "pyspark_min_equals_max_success",
        "pandas_single_char_success",
        "pandas_single_char_violations",
        "pyspark_single_char_success",
        "pandas_empty_string_violation",
        "pandas_empty_string_success",
        "pyspark_empty_string_violation",
        "pandas_whitespace_success",
        "pandas_whitespace_with_text",
        "pandas_whitespace_violations",
        "pyspark_whitespace_success",
        "pyspark_whitespace_with_text",
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
        "pandas_long_strings_success",
        "pandas_long_strings_violations",
        "pyspark_long_strings_success",
        "pandas_wide_range",
        "pyspark_wide_range",
        "pandas_zero_min_success",
        "pandas_zero_min_violation",
        "pyspark_zero_min_success",
    ],
)
def test_expectation_basic_scenarios(
    df_type,
    data,
    min_length,
    max_length,
    expected_result,
    expected_violations,
    expected_message,
    spark,
):
    """
    Test the expectation for various scenarios across pandas and PySpark DataFrames.
    Tests both direct expectation validation and suite-based validation.
    Covers: success cases, violations, boundary conditions (min/max exact),
    min equals max scenarios, single character strings, empty strings,
    whitespace handling, special characters, numbers in strings, long strings,
    wide ranges, and zero min length.
    """
    data_frame = create_dataframe(df_type, data, "col1", spark)

    # Test 1: Direct expectation validation
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationStringLengthBetween",
        column_name="col1",
        min_length=min_length,
        max_length=max_length,
    )

    result = expectation.validate(data_frame=data_frame)

    if expected_result == "success":
        assert str(result) == str(
            DataFrameExpectationSuccessMessage(expectation_name="ExpectationStringLengthBetween")
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
    expectations_suite = DataFrameExpectationsSuite().expect_string_length_between(
        column_name="col1", min_length=min_length, max_length=max_length
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
        data_frame = pd.DataFrame({"col2": ["foo", "bazz", "hello"]})
    else:  # pyspark
        data_frame = spark.createDataFrame([("foo",), ("bazz",), ("hello",)], ["col2"])

    # Test 1: Direct expectation validation
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationStringLengthBetween",
        column_name="col1",
        min_length=3,
        max_length=6,
    )
    result = expectation.validate(data_frame=data_frame)
    expected_failure = DataFrameExpectationFailureMessage(
        expectation_str=str(expectation),
        data_frame_type=get_df_type_enum(df_type),
        message=expected_message,
    )
    assert str(result) == str(expected_failure), f"Expected failure message but got: {result}"

    # Test 2: Suite-based validation
    expectations_suite = DataFrameExpectationsSuite().expect_string_length_between(
        column_name="col1", min_length=3, max_length=6
    )
    with pytest.raises(DataFrameExpectationsSuiteFailure):
        expectations_suite.build().run(data_frame=data_frame)


def test_large_dataset_performance():
    """Test the expectation with a larger dataset to ensure performance."""
    # Create a larger dataset with 10,000 rows with lengths between 5 and 10
    large_data = [f"test_{i}" for i in range(10000)]
    data_frame = pd.DataFrame({"col1": large_data})

    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationStringLengthBetween",
        column_name="col1",
        min_length=5,
        max_length=15,
    )

    result = expectation.validate(data_frame=data_frame)
    # Should succeed as all values have lengths between 5 and 15
    assert isinstance(result, DataFrameExpectationSuccessMessage), (
        f"Large dataset test failed: expected success but got {type(result)}"
    )
