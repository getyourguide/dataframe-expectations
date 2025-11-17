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
        expectation_name="ExpectationStringStartsWith",
        column_name="col1",
        prefix="foo",
    )
    assert expectation.get_expectation_name() == "ExpectationStringStartsWith", (
        f"Expected 'ExpectationStringStartsWith' but got: {expectation.get_expectation_name()}"
    )


@pytest.mark.parametrize(
    "df_type,data,prefix,expected_result,expected_violations,test_id",
    [
        # Basic success and violations - pandas
        ("pandas", ["foobar", "foo123", "foobaz"], "foo", "success", None, "pandas_basic_success"),
        (
            "pandas",
            ["bar", "baz", "qux"],
            "foo",
            "violations",
            ["bar", "baz", "qux"],
            "pandas_basic_violations",
        ),
        # Basic success and violations - pyspark
        (
            "pyspark",
            ["foobar", "foo123", "foobaz"],
            "foo",
            "success",
            None,
            "pyspark_basic_success",
        ),
        (
            "pyspark",
            ["bar", "baz", "qux"],
            "foo",
            "violations",
            ["bar", "baz", "qux"],
            "pyspark_basic_violations",
        ),
        # Boundary conditions - pandas (exact match)
        ("pandas", ["foo", "foo", "foo"], "foo", "success", None, "pandas_exact_match_success"),
        (
            "pandas",
            ["foo", "bar", "baz"],
            "foo",
            "violations",
            ["bar", "baz"],
            "pandas_exact_match_mixed",
        ),
        # Boundary conditions - pyspark
        ("pyspark", ["foo", "foo", "foo"], "foo", "success", None, "pyspark_exact_match_success"),
        (
            "pyspark",
            ["foo", "bar", "baz"],
            "foo",
            "violations",
            ["bar", "baz"],
            "pyspark_exact_match_mixed",
        ),
        # Empty strings - pandas
        (
            "pandas",
            ["", "", ""],
            "foo",
            "violations",
            ["", "", ""],
            "pandas_empty_string_violations",
        ),
        # Empty strings - pyspark
        (
            "pyspark",
            ["", "", ""],
            "foo",
            "violations",
            ["", "", ""],
            "pyspark_empty_string_violations",
        ),
        # Whitespace handling - pandas
        (
            "pandas",
            ["   ", "  ", " "],
            "foo",
            "violations",
            ["   ", "  ", " "],
            "pandas_whitespace_only_violations",
        ),
        (
            "pandas",
            ["foo bar", "foo baz", "foo qux"],
            "foo",
            "success",
            None,
            "pandas_whitespace_in_text_success",
        ),
        (
            "pandas",
            ["bar foo", "baz foo", "qux foo"],
            "foo",
            "violations",
            ["bar foo", "baz foo", "qux foo"],
            "pandas_whitespace_at_end_violations",
        ),
        # Whitespace handling - pyspark
        (
            "pyspark",
            ["   ", "  ", " "],
            "foo",
            "violations",
            ["   ", "  ", " "],
            "pyspark_whitespace_only_violations",
        ),
        (
            "pyspark",
            ["foo bar", "foo baz", "foo qux"],
            "foo",
            "success",
            None,
            "pyspark_whitespace_in_text_success",
        ),
        # Special characters - pandas
        (
            "pandas",
            ["@foo", "#foo", "$foo"],
            "@",
            "violations",
            ["#foo", "$foo"],
            "pandas_special_char_at_violations",
        ),
        (
            "pandas",
            ["foo@bar", "foo#baz", "foo$qux"],
            "foo@",
            "violations",
            ["foo#baz", "foo$qux"],
            "pandas_special_char_in_prefix_success",
        ),
        # Special characters - pyspark
        (
            "pyspark",
            ["@foo", "#foo", "$foo"],
            "@",
            "violations",
            ["#foo", "$foo"],
            "pyspark_special_char_at_violations",
        ),
        (
            "pyspark",
            ["foo@bar", "foo#baz", "foo$qux"],
            "foo@",
            "violations",
            ["foo#baz", "foo$qux"],
            "pyspark_special_char_in_prefix_success",
        ),
        # Numbers in strings - pandas
        ("pandas", ["foo1", "foo2", "foo3"], "foo", "success", None, "pandas_numbers_success"),
        (
            "pandas",
            ["1foo", "2foo", "3foo"],
            "foo",
            "violations",
            ["1foo", "2foo", "3foo"],
            "pandas_numbers_at_start_violations",
        ),
        # Numbers in strings - pyspark
        ("pyspark", ["foo1", "foo2", "foo3"], "foo", "success", None, "pyspark_numbers_success"),
        (
            "pyspark",
            ["1foo", "2foo", "3foo"],
            "foo",
            "violations",
            ["1foo", "2foo", "3foo"],
            "pyspark_numbers_at_start_violations",
        ),
        # Long strings - pandas
        (
            "pandas",
            ["foo" + "a" * 97, "foo" + "b" * 97, "foo" + "c" * 97],
            "foo",
            "success",
            None,
            "pandas_long_string_success",
        ),
        (
            "pandas",
            ["a" * 100, "b" * 100, "c" * 100],
            "foo",
            "violations",
            ["a" * 100, "b" * 100, "c" * 100],
            "pandas_long_string_violations",
        ),
        # Long strings - pyspark
        (
            "pyspark",
            ["foo" + "a" * 97, "foo" + "b" * 97, "foo" + "c" * 97],
            "foo",
            "success",
            None,
            "pyspark_long_string_success",
        ),
        (
            "pyspark",
            ["a" * 100, "b" * 100, "c" * 100],
            "foo",
            "violations",
            ["a" * 100, "b" * 100, "c" * 100],
            "pyspark_long_string_violations",
        ),
        # Mixed violations - pandas
        (
            "pandas",
            ["foobar", "bar", "foo123"],
            "foo",
            "violations",
            ["bar"],
            "pandas_mixed_violations",
        ),
        # Mixed violations - pyspark
        (
            "pyspark",
            ["foobar", "bar", "foo123"],
            "foo",
            "violations",
            ["bar"],
            "pyspark_mixed_violations",
        ),
    ],
)
def test_expectation_basic_scenarios(
    df_type, data, prefix, expected_result, expected_violations, test_id, spark
):
    """Test various scenarios for ExpectationStringStartsWith expectation."""
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationStringStartsWith",
        column_name="col1",
        prefix=prefix,
    )

    data_frame = create_dataframe(df_type, data, "col1", spark)
    result = expectation.validate(data_frame=data_frame)

    if expected_result == "success":
        assert str(result) == str(
            DataFrameExpectationSuccessMessage(expectation_name="ExpectationStringStartsWith")
        ), f"Test {test_id}: Expected success message but got: {result}"

        # Also test with suite
        expectations_suite = DataFrameExpectationsSuite().expect_string_starts_with(
            column_name="col1", prefix=prefix
        )
        suite_result = expectations_suite.build().run(data_frame=data_frame)
        assert suite_result is None, f"Test {test_id}: Expected no exceptions to be raised"
    else:  # violations
        violations_df = create_dataframe(df_type, expected_violations, "col1", spark)
        expected_message = (
            f"Found {len(expected_violations)} row(s) where 'col1' does not start with '{prefix}'."
        )

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
        expectations_suite = DataFrameExpectationsSuite().expect_string_starts_with(
            column_name="col1", prefix=prefix
        )
        with pytest.raises(DataFrameExpectationsSuiteFailure):
            expectations_suite.build().run(data_frame=data_frame)


@pytest.mark.parametrize("df_type", ["pandas", "pyspark"])
def test_column_missing_error(df_type, spark):
    """Test that missing column raises appropriate error."""
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationStringStartsWith",
        column_name="col1",
        prefix="foo",
    )

    data_frame = create_dataframe(df_type, ["foobar", "foo123", "foobaz"], "col2", spark)
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
    expectations_suite = DataFrameExpectationsSuite().expect_string_starts_with(
        column_name="col1", prefix="foo"
    )
    with pytest.raises(DataFrameExpectationsSuiteFailure):
        expectations_suite.build().run(data_frame=data_frame)


def test_large_dataset_performance():
    """Test the expectation with a larger dataset to ensure performance."""
    # Create a larger dataset with 10,000 rows all starting with "foo"
    large_data = ["foo" + str(i) for i in range(10000)]
    data_frame = pd.DataFrame({"col1": large_data})

    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationStringStartsWith",
        column_name="col1",
        prefix="foo",
    )

    result = expectation.validate(data_frame=data_frame)
    # Should succeed as all values start with "foo"
    assert isinstance(result, DataFrameExpectationSuccessMessage), (
        f"Large dataset test failed: expected success but got {type(result)}"
    )
