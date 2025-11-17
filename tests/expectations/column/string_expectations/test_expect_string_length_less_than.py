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
    """Helper function to create either pandas or PySpark DataFrame."""
    if df_type == "pandas":
        return pd.DataFrame({column_name: data})
    else:  # pyspark
        return spark.createDataFrame([(item,) for item in data], [column_name])


def get_df_type_enum(df_type):
    """Helper function to get DataFrameType enum."""
    return DataFrameType.PANDAS if df_type == "pandas" else DataFrameType.PYSPARK


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
    "df_type,data,length,expected_result,expected_violations,test_id",
    [
        # Basic success and violations - pandas
        ("pandas", ["foo", "bar", "baz"], 5, "success", None, "pandas_basic_success"),
        ("pandas", ["ab", "cd", "ef"], 3, "success", None, "pandas_success_length_2"),
        ("pandas", ["abc", "def", "ghi"], 5, "success", None, "pandas_success_length_4"),
        # Basic success and violations - pyspark
        ("pyspark", ["foo", "bar", "baz"], 5, "success", None, "pyspark_basic_success"),
        ("pyspark", ["ab", "cd", "ef"], 3, "success", None, "pyspark_success_length_2"),
        ("pyspark", ["abc", "def", "ghi"], 5, "success", None, "pyspark_success_length_4"),
        # Basic violations - pandas
        (
            "pandas",
            ["foobar", "bar", "bazbaz"],
            5,
            "violations",
            ["foobar", "bazbaz"],
            "pandas_basic_violations",
        ),
        (
            "pandas",
            ["testing", "longer", "strings"],
            5,
            "violations",
            ["testing", "longer", "strings"],
            "pandas_all_violations",
        ),
        (
            "pandas",
            ["ok", "fail", "good"],
            3,
            "violations",
            ["fail", "good"],
            "pandas_mixed_violations",
        ),
        # Basic violations - pyspark
        (
            "pyspark",
            ["foobar", "bar", "bazbaz"],
            5,
            "violations",
            ["foobar", "bazbaz"],
            "pyspark_basic_violations",
        ),
        (
            "pyspark",
            ["testing", "longer", "strings"],
            5,
            "violations",
            ["testing", "longer", "strings"],
            "pyspark_all_violations",
        ),
        # Boundary conditions - pandas (exact length equals threshold)
        (
            "pandas",
            ["test", "exam", "demo"],
            4,
            "violations",
            ["test", "exam", "demo"],
            "pandas_boundary_exact_violation",
        ),
        ("pandas", ["tes", "exa", "dem"], 4, "success", None, "pandas_boundary_success"),
        # Boundary conditions - pyspark
        (
            "pyspark",
            ["test", "exam", "demo"],
            4,
            "violations",
            ["test", "exam", "demo"],
            "pyspark_boundary_exact_violation",
        ),
        ("pyspark", ["tes", "exa", "dem"], 4, "success", None, "pyspark_boundary_success"),
        # Zero length threshold - pandas
        ("pandas", ["", "", ""], 1, "success", None, "pandas_zero_length_success"),
        (
            "pandas",
            ["a", "b", "c"],
            1,
            "violations",
            ["a", "b", "c"],
            "pandas_zero_length_violation",
        ),
        # Zero length threshold - pyspark
        ("pyspark", ["", "", ""], 1, "success", None, "pyspark_zero_length_success"),
        # Single character strings - pandas
        ("pandas", ["a", "b", "c"], 2, "success", None, "pandas_single_char_success"),
        (
            "pandas",
            ["a", "b", "c"],
            1,
            "violations",
            ["a", "b", "c"],
            "pandas_single_char_violations",
        ),
        # Single character strings - pyspark
        ("pyspark", ["a", "b", "c"], 2, "success", None, "pyspark_single_char_success"),
        # Empty strings - pandas
        ("pandas", ["", "", ""], 5, "success", None, "pandas_empty_string_success"),
        # Empty strings - pyspark
        ("pyspark", ["", "", ""], 5, "success", None, "pyspark_empty_string_success"),
        # Whitespace handling - pandas
        ("pandas", ["  ", " ", ""], 3, "success", None, "pandas_whitespace_success"),
        ("pandas", ["a b", "a", "ab"], 4, "success", None, "pandas_whitespace_in_text"),
        (
            "pandas",
            ["    ", "     ", "      "],
            3,
            "violations",
            ["    ", "     ", "      "],
            "pandas_whitespace_violations",
        ),
        # Whitespace handling - pyspark
        ("pyspark", ["  ", " ", ""], 3, "success", None, "pyspark_whitespace_success"),
        ("pyspark", ["a b", "a", "ab"], 4, "success", None, "pyspark_whitespace_in_text"),
        # Special characters - pandas
        ("pandas", ["@#$", "!^&", "*()"], 4, "success", None, "pandas_special_chars_success"),
        ("pandas", ["test@", "ex!m", "de#o"], 6, "success", None, "pandas_special_chars_in_text"),
        (
            "pandas",
            ["@#$%^", "&*()_", "+=-[]"],
            4,
            "violations",
            ["@#$%^", "&*()_", "+=-[]"],
            "pandas_special_chars_violations",
        ),
        # Special characters - pyspark
        ("pyspark", ["@#$", "!^&", "*()"], 4, "success", None, "pyspark_special_chars_success"),
        ("pyspark", ["test@", "ex!m", "de#o"], 6, "success", None, "pyspark_special_chars_in_text"),
        # Numbers in strings - pandas
        ("pandas", ["123", "456", "789"], 4, "success", None, "pandas_numbers_success"),
        ("pandas", ["v1.0", "v2.1", "v3.5"], 5, "success", None, "pandas_numbers_versions"),
        (
            "pandas",
            ["12345", "67890", "11111"],
            4,
            "violations",
            ["12345", "67890", "11111"],
            "pandas_numbers_violations",
        ),
        # Numbers in strings - pyspark
        ("pyspark", ["123", "456", "789"], 4, "success", None, "pyspark_numbers_success"),
        ("pyspark", ["v1.0", "v2.1", "v3.5"], 5, "success", None, "pyspark_numbers_versions"),
        # Long strings - pandas
        ("pandas", ["a" * 9, "b" * 8, "c" * 7], 11, "success", None, "pandas_length_10_success"),
        (
            "pandas",
            ["test " * 3, "demo " * 3, "exam " * 3],
            21,
            "success",
            None,
            "pandas_length_20_success",
        ),
        (
            "pandas",
            ["x" * 11, "y" * 12, "z" * 13],
            10,
            "violations",
            ["x" * 11, "y" * 12, "z" * 13],
            "pandas_length_10_violation",
        ),
        # Long strings - pyspark
        ("pyspark", ["a" * 9, "b" * 8, "c" * 7], 11, "success", None, "pyspark_length_10_success"),
        (
            "pyspark",
            ["test " * 3, "demo " * 3, "exam " * 3],
            21,
            "success",
            None,
            "pyspark_length_20_success",
        ),
        # Very long strings - pandas
        (
            "pyspark",
            ["a" * 99, "b" * 98, "c" * 97],
            101,
            "success",
            None,
            "pyspark_long_strings_success",
        ),
        (
            "pandas",
            ["x" * 99, "y" * 98, "z" * 97],
            101,
            "success",
            None,
            "pandas_long_strings_success",
        ),
        (
            "pandas",
            ["x" * 101, "y" * 102, "z" * 103],
            100,
            "violations",
            ["x" * 101, "y" * 102, "z" * 103],
            "pandas_long_strings_violation",
        ),
        # Mixed violations - pandas
        (
            "pandas",
            ["ab", "abcd", "abc"],
            4,
            "violations",
            ["abcd"],
            "pandas_mixed_short_and_exact",
        ),
        # Mixed violations - pyspark
        (
            "pyspark",
            ["ab", "abcd", "abc"],
            4,
            "violations",
            ["abcd"],
            "pyspark_mixed_short_and_exact",
        ),
    ],
)
def test_expectation_basic_scenarios(
    df_type, data, length, expected_result, expected_violations, test_id, spark
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
        ), f"Test {test_id}: Expected success message but got: {result}"

        # Also test with suite
        expectations_suite = DataFrameExpectationsSuite().expect_string_length_less_than(
            column_name="col1", length=length
        )
        suite_result = expectations_suite.build().run(data_frame=data_frame)
        assert suite_result is None, f"Test {test_id}: Expected no exceptions to be raised"
    else:  # violations
        violations_df = create_dataframe(df_type, expected_violations, "col1", spark)
        expected_message = f"Found {len(expected_violations)} row(s) where 'col1' length is not less than {length}."

        assert str(result) == str(
            DataFrameExpectationFailureMessage(
                expectation_str=str(expectation),
                data_frame_type=get_df_type_enum(df_type),
                violations_data_frame=violations_df,
                message=expected_message,
                limit_violations=5,
            )
        ), f"Test {test_id}: Expected failure message but got: {result}"

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
        data_frame_type=get_df_type_enum(df_type),
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
