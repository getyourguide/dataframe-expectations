import pytest
import pandas as pd

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
    """Helper function to create either pandas or PySpark DataFrame."""
    if df_type == "pandas":
        return pd.DataFrame({column_name: data})
    else:  # pyspark
        return spark.createDataFrame([(item,) for item in data], [column_name])


def test_expectation_name():
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationStringLengthLessThan",
        column_name="col1",
        length=5,
    )
    assert expectation.get_expectation_name() == "ExpectationStringLengthLessThan", (
        f"Expected 'ExpectationStringLengthLessThan' but got: {expectation.get_expectation_name()}"
    )


@pytest.mark.parametrize(
    "df_type,data,length,expected_result,expected_violations",
    [
        # Basic success - pandas
        ("pandas", ["foo", "bar", "baz"], 5, "success", None),
        # Basic success - pyspark
        ("pyspark", ["foo", "bar", "baz"], 5, "success", None),
        # Success length 2 - pandas
        ("pandas", ["ab", "cd", "ef"], 3, "success", None),
        # Success length 2 - pyspark
        ("pyspark", ["ab", "cd", "ef"], 3, "success", None),
        # Success length 4 - pandas
        ("pandas", ["abc", "def", "ghi"], 5, "success", None),
        # Success length 4 - pyspark
        ("pyspark", ["abc", "def", "ghi"], 5, "success", None),
        # Basic violations - pandas
        (
            "pandas",
            ["foobar", "bar", "bazbaz"],
            5,
            "violations",
            ["foobar", "bazbaz"],
        ),
        # Basic violations - pyspark
        (
            "pyspark",
            ["foobar", "bar", "bazbaz"],
            5,
            "violations",
            ["foobar", "bazbaz"],
        ),
        # All violations - pandas
        (
            "pandas",
            ["testing", "longer", "strings"],
            5,
            "violations",
            ["testing", "longer", "strings"],
        ),
        # All violations - pyspark
        (
            "pyspark",
            ["testing", "longer", "strings"],
            5,
            "violations",
            ["testing", "longer", "strings"],
        ),
        # Mixed violations - pandas
        (
            "pandas",
            ["ok", "fail", "good"],
            3,
            "violations",
            ["fail", "good"],
        ),
        # Mixed violations - pyspark
        (
            "pyspark",
            ["ok", "fail", "good"],
            3,
            "violations",
            ["fail", "good"],
        ),
        # Boundary exact violation - pandas
        (
            "pandas",
            ["test", "exam", "demo"],
            4,
            "violations",
            ["test", "exam", "demo"],
        ),
        # Boundary exact violation - pyspark
        (
            "pyspark",
            ["test", "exam", "demo"],
            4,
            "violations",
            ["test", "exam", "demo"],
        ),
        # Boundary success - pandas
        ("pandas", ["tes", "exa", "dem"], 4, "success", None),
        # Boundary success - pyspark
        ("pyspark", ["tes", "exa", "dem"], 4, "success", None),
        # Zero length success - pandas
        ("pandas", ["", "", ""], 1, "success", None),
        # Zero length success - pyspark
        ("pyspark", ["", "", ""], 1, "success", None),
        # Zero length violation - pandas
        (
            "pandas",
            ["a", "b", "c"],
            1,
            "violations",
            ["a", "b", "c"],
        ),
        # Zero length violation - pyspark
        (
            "pyspark",
            ["a", "b", "c"],
            1,
            "violations",
            ["a", "b", "c"],
        ),
        # Single char success - pandas
        ("pandas", ["a", "b", "c"], 2, "success", None),
        # Single char success - pyspark
        ("pyspark", ["a", "b", "c"], 2, "success", None),
        # Single char violations - pandas
        (
            "pandas",
            ["a", "b", "c"],
            1,
            "violations",
            ["a", "b", "c"],
        ),
        # Single char violations - pyspark
        (
            "pyspark",
            ["a", "b", "c"],
            1,
            "violations",
            ["a", "b", "c"],
        ),
        # Empty strings - pandas
        ("pandas", ["", "", ""], 5, "success", None),
        # Empty strings - pyspark
        ("pyspark", ["", "", ""], 5, "success", None),
        # Whitespace success - pandas
        ("pandas", ["  ", " ", ""], 3, "success", None),
        # Whitespace success - pyspark
        ("pyspark", ["  ", " ", ""], 3, "success", None),
        # Whitespace in text - pandas
        ("pandas", ["a b", "a", "ab"], 4, "success", None),
        # Whitespace in text - pyspark
        ("pyspark", ["a b", "a", "ab"], 4, "success", None),
        # Whitespace violations - pandas
        (
            "pandas",
            ["    ", "     ", "      "],
            3,
            "violations",
            ["    ", "     ", "      "],
        ),
        # Whitespace violations - pyspark
        (
            "pyspark",
            ["    ", "     ", "      "],
            3,
            "violations",
            ["    ", "     ", "      "],
        ),
        # Special chars success - pandas
        ("pandas", ["@#$", "!^&", "*()"], 4, "success", None),
        # Special chars success - pyspark
        ("pyspark", ["@#$", "!^&", "*()"], 4, "success", None),
        # Special chars in text - pandas
        ("pandas", ["test@", "ex!m", "de#o"], 6, "success", None),
        # Special chars in text - pyspark
        ("pyspark", ["test@", "ex!m", "de#o"], 6, "success", None),
        # Special chars violations - pandas
        (
            "pandas",
            ["@#$%^", "&*()_", "+=-[]"],
            4,
            "violations",
            ["@#$%^", "&*()_", "+=-[]"],
        ),
        # Special chars violations - pyspark
        (
            "pyspark",
            ["@#$%^", "&*()_", "+=-[]"],
            4,
            "violations",
            ["@#$%^", "&*()_", "+=-[]"],
        ),
        # Numbers success - pandas
        ("pandas", ["123", "456", "789"], 4, "success", None),
        # Numbers success - pyspark
        ("pyspark", ["123", "456", "789"], 4, "success", None),
        # Numbers versions - pandas
        ("pandas", ["v1.0", "v2.1", "v3.5"], 5, "success", None),
        # Numbers versions - pyspark
        ("pyspark", ["v1.0", "v2.1", "v3.5"], 5, "success", None),
        # Numbers violations - pandas
        (
            "pandas",
            ["12345", "67890", "11111"],
            4,
            "violations",
            ["12345", "67890", "11111"],
        ),
        # Numbers violations - pyspark
        (
            "pyspark",
            ["12345", "67890", "11111"],
            4,
            "violations",
            ["12345", "67890", "11111"],
        ),
        # Length 10 success - pandas
        ("pandas", ["a" * 9, "b" * 8, "c" * 7], 11, "success", None),
        # Length 10 success - pyspark
        ("pyspark", ["a" * 9, "b" * 8, "c" * 7], 11, "success", None),
        # Length 20 success - pandas
        (
            "pandas",
            ["test " * 3, "demo " * 3, "exam " * 3],
            21,
            "success",
            None,
        ),
        # Length 20 success - pyspark
        (
            "pyspark",
            ["test " * 3, "demo " * 3, "exam " * 3],
            21,
            "success",
            None,
        ),
        # Length 10 violation - pandas
        (
            "pandas",
            ["x" * 11, "y" * 12, "z" * 13],
            10,
            "violations",
            ["x" * 11, "y" * 12, "z" * 13],
        ),
        # Length 10 violation - pyspark
        (
            "pyspark",
            ["x" * 11, "y" * 12, "z" * 13],
            10,
            "violations",
            ["x" * 11, "y" * 12, "z" * 13],
        ),
        # Long strings success - pandas
        (
            "pandas",
            ["x" * 99, "y" * 98, "z" * 97],
            101,
            "success",
            None,
        ),
        # Long strings success - pyspark
        (
            "pyspark",
            ["a" * 99, "b" * 98, "c" * 97],
            101,
            "success",
            None,
        ),
        # Long strings violation - pandas
        (
            "pandas",
            ["x" * 101, "y" * 102, "z" * 103],
            100,
            "violations",
            ["x" * 101, "y" * 102, "z" * 103],
        ),
        # Long strings violation - pyspark
        (
            "pyspark",
            ["x" * 101, "y" * 102, "z" * 103],
            100,
            "violations",
            ["x" * 101, "y" * 102, "z" * 103],
        ),
        # Mixed violations - pandas
        (
            "pandas",
            ["ab", "abcd", "abc"],
            4,
            "violations",
            ["abcd"],
        ),
        # Mixed violations - pyspark
        (
            "pyspark",
            ["ab", "abcd", "abc"],
            4,
            "violations",
            ["abcd"],
        ),
    ],
    ids=[
        "pandas_basic_success",
        "pyspark_basic_success",
        "pandas_success_length_2",
        "pyspark_success_length_2",
        "pandas_success_length_4",
        "pyspark_success_length_4",
        "pandas_basic_violations",
        "pyspark_basic_violations",
        "pandas_all_violations",
        "pyspark_all_violations",
        "pandas_mixed_violations",
        "pyspark_mixed_violations",
        "pandas_boundary_exact_violation",
        "pyspark_boundary_exact_violation",
        "pandas_boundary_success",
        "pyspark_boundary_success",
        "pandas_zero_length_success",
        "pyspark_zero_length_success",
        "pandas_zero_length_violation",
        "pyspark_zero_length_violation",
        "pandas_single_char_success",
        "pyspark_single_char_success",
        "pandas_single_char_violations",
        "pyspark_single_char_violations",
        "pandas_empty_string_success",
        "pyspark_empty_string_success",
        "pandas_whitespace_success",
        "pyspark_whitespace_success",
        "pandas_whitespace_in_text",
        "pyspark_whitespace_in_text",
        "pandas_whitespace_violations",
        "pyspark_whitespace_violations",
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
        "pandas_long_strings_violation",
        "pyspark_long_strings_violation",
        "pandas_mixed_short_and_exact",
        "pyspark_mixed_short_and_exact",
    ],
)
def test_expectation_basic_scenarios(
    df_type, data, length, expected_result, expected_violations, spark
):
    """Test various scenarios for ExpectationStringLengthLessThan expectation."""
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationStringLengthLessThan",
        column_name="col1",
        length=length,
    )

    data_frame = create_dataframe(df_type, data, "col1", spark)
    result = expectation.validate(data_frame=data_frame)

    if expected_result == "success":
        assert str(result) == str(
            DataFrameExpectationSuccessMessage(expectation_name="ExpectationStringLengthLessThan")
        ), f"Expected success message but got: {result}"

        # Also test with suite
        expectations_suite = DataFrameExpectationsSuite().expect_string_length_less_than(
            column_name="col1", length=length
        )
        suite_result = expectations_suite.build().run(data_frame=data_frame)
        assert suite_result is None, "Expected no exceptions to be raised"
    else:  # violations
        violations_df = create_dataframe(df_type, expected_violations, "col1", spark)
        expected_message = f"Found {len(expected_violations)} row(s) where 'col1' length is not less than {length}."

        assert str(result) == str(
            DataFrameExpectationFailureMessage(
                expectation_str=str(expectation),
                data_frame_type=str(df_type),
                violations_data_frame=violations_df,
                message=expected_message,
                limit_violations=5,
            )
        ), f"Expected failure message but got: {result}"

        # Also test with suite
        expectations_suite = DataFrameExpectationsSuite().expect_string_length_less_than(
            column_name="col1", length=length
        )
        with pytest.raises(DataFrameExpectationsSuiteFailure):
            expectations_suite.build().run(data_frame=data_frame)


@pytest.mark.parametrize("df_type", ["pandas", "pyspark"])
def test_column_missing_error(df_type, spark):
    """Test that missing column raises appropriate error."""
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationStringLengthLessThan",
        column_name="col1",
        length=5,
    )

    data_frame = create_dataframe(df_type, ["foo", "bar", "baz"], "col2", spark)
    result = expectation.validate(data_frame=data_frame)

    expected_failure_message = DataFrameExpectationFailureMessage(
        expectation_str=str(expectation),
        data_frame_type=str(df_type),
        message="Column 'col1' does not exist in the DataFrame.",
    )
    assert str(result) == str(expected_failure_message), (
        f"Expected failure message but got: {result}"
    )

    # Also test with suite
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
