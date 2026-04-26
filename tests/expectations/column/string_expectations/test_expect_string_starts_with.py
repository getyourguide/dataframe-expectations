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
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationStringStartsWith",
        column_name="col1",
        prefix="foo",
    )
    assert expectation.get_expectation_name() == "ExpectationStringStartsWith", (
        f"Expected 'ExpectationStringStartsWith' but got: {expectation.get_expectation_name()}"
    )


@pytest.mark.parametrize(
    "df_type,data,prefix,expected_result,expected_violations",
    [
        # Basic success - pandas
        ("pandas", ["foobar", "foo123", "foobaz"], "foo", "success", None),
        # Basic violations - pandas
        (
            "pandas",
            ["bar", "baz", "qux"],
            "foo",
            "violations",
            ["bar", "baz", "qux"],
        ),
        # Exact match success - pandas
        ("pandas", ["foo", "foo", "foo"], "foo", "success", None),
        # Exact match mixed - pandas
        (
            "pandas",
            ["foo", "bar", "baz"],
            "foo",
            "violations",
            ["bar", "baz"],
        ),
        # Empty strings - pandas
        (
            "pandas",
            ["", "", ""],
            "foo",
            "violations",
            ["", "", ""],
        ),
        # Whitespace only violations - pandas
        (
            "pandas",
            ["   ", "  ", " "],
            "foo",
            "violations",
            ["   ", "  ", " "],
        ),
        # Whitespace in text success - pandas
        (
            "pandas",
            ["foo bar", "foo baz", "foo qux"],
            "foo",
            "success",
            None,
        ),
        # Whitespace at end violations - pandas
        (
            "pandas",
            ["bar foo", "baz foo", "qux foo"],
            "foo",
            "violations",
            ["bar foo", "baz foo", "qux foo"],
        ),
        # Special char at violations - pandas
        (
            "pandas",
            ["@foo", "#foo", "$foo"],
            "@",
            "violations",
            ["#foo", "$foo"],
        ),
        # Special char in prefix violations - pandas
        (
            "pandas",
            ["foo@bar", "foo#baz", "foo$qux"],
            "foo@",
            "violations",
            ["foo#baz", "foo$qux"],
        ),
        # Numbers success - pandas
        ("pandas", ["foo1", "foo2", "foo3"], "foo", "success", None),
        # Numbers at start violations - pandas
        (
            "pandas",
            ["1foo", "2foo", "3foo"],
            "foo",
            "violations",
            ["1foo", "2foo", "3foo"],
        ),
        # Long string success - pandas
        (
            "pandas",
            ["foo" + "a" * 97, "foo" + "b" * 97, "foo" + "c" * 97],
            "foo",
            "success",
            None,
        ),
        # Long string violations - pandas
        (
            "pandas",
            ["a" * 100, "b" * 100, "c" * 100],
            "foo",
            "violations",
            ["a" * 100, "b" * 100, "c" * 100],
        ),
        # Mixed violations - pandas
        (
            "pandas",
            ["foobar", "bar", "foo123"],
            "foo",
            "violations",
            ["bar"],
        ),
    ],
    ids=[
        "pandas_basic_success",
        "pandas_basic_violations",
        "pandas_exact_match_success",
        "pandas_exact_match_mixed",
        "pandas_empty_string_violations",
        "pandas_whitespace_only_violations",
        "pandas_whitespace_in_text_success",
        "pandas_whitespace_at_end_violations",
        "pandas_special_char_at_violations",
        "pandas_special_char_in_prefix_violations",
        "pandas_numbers_success",
        "pandas_numbers_at_start_violations",
        "pandas_long_string_success",
        "pandas_long_string_violations",
        "pandas_mixed_violations",
    ],
)
def test_expectation_basic_scenarios_pandas(
    df_type, data, prefix, expected_result, expected_violations
):
    """Test various scenarios for ExpectationStringStartsWith expectation."""
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationStringStartsWith",
        column_name="col1",
        prefix=prefix,
    )

    data_frame = pd.DataFrame({"col1": data})
    result = expectation.validate(data_frame=data_frame)

    if expected_result == "success":
        assert str(result) == str(
            DataFrameExpectationSuccessMessage(expectation_name="ExpectationStringStartsWith")
        ), f"Expected success message but got: {result}"

        # Also test with suite
        expectations_suite = DataFrameExpectationsSuite().expect_string_starts_with(
            column_name="col1", prefix=prefix
        )
        suite_result = expectations_suite.build().run(data_frame=data_frame)
        assert suite_result is not None, "Expected SuiteExecutionResult"
        assert isinstance(suite_result, SuiteExecutionResult), (
            "Result should be SuiteExecutionResult"
        )
        assert suite_result.success, "Expected all expectations to pass"
        assert suite_result.total_passed == 1, "Expected 1 passed expectation"
        assert suite_result.total_failed == 0, "Expected 0 failed expectations"
    else:  # violations
        violations_df = pd.DataFrame({"col1": expected_violations})
        expected_message = (
            f"Found {len(expected_violations)} row(s) where 'col1' does not start with '{prefix}'."
        )

        assert str(result) == str(
            DataFrameExpectationFailureMessage(
                expectation_str=str(expectation),
                data_frame_type=DataFrameType(df_type),
                violations_data_frame=violations_df,
                message=expected_message,
                limit_violations=5,
            )
        ), f"Expected failure message but got: {result}"

        # Also test with suite
        expectations_suite = DataFrameExpectationsSuite().expect_string_starts_with(
            column_name="col1", prefix=prefix
        )
        with pytest.raises(DataFrameExpectationsSuiteFailure):
            expectations_suite.build().run(data_frame=data_frame)


@pytest.mark.pyspark
@pytest.mark.parametrize(
    "df_type,data,prefix,expected_result,expected_violations",
    [
        # Basic success - pyspark
        ("pyspark", ["foobar", "foo123", "foobaz"], "foo", "success", None),
        # Basic violations - pyspark
        (
            "pyspark",
            ["bar", "baz", "qux"],
            "foo",
            "violations",
            ["bar", "baz", "qux"],
        ),
        # Exact match success - pyspark
        ("pyspark", ["foo", "foo", "foo"], "foo", "success", None),
        # Exact match mixed - pyspark
        (
            "pyspark",
            ["foo", "bar", "baz"],
            "foo",
            "violations",
            ["bar", "baz"],
        ),
        # Empty strings - pyspark
        (
            "pyspark",
            ["", "", ""],
            "foo",
            "violations",
            ["", "", ""],
        ),
        # Whitespace only violations - pyspark
        (
            "pyspark",
            ["   ", "  ", " "],
            "foo",
            "violations",
            ["   ", "  ", " "],
        ),
        # Whitespace in text success - pyspark
        (
            "pyspark",
            ["foo bar", "foo baz", "foo qux"],
            "foo",
            "success",
            None,
        ),
        # Whitespace at end violations - pyspark
        (
            "pyspark",
            ["bar foo", "baz foo", "qux foo"],
            "foo",
            "violations",
            ["bar foo", "baz foo", "qux foo"],
        ),
        # Special char at violations - pyspark
        (
            "pyspark",
            ["@foo", "#foo", "$foo"],
            "@",
            "violations",
            ["#foo", "$foo"],
        ),
        # Special char in prefix violations - pyspark
        (
            "pyspark",
            ["foo@bar", "foo#baz", "foo$qux"],
            "foo@",
            "violations",
            ["foo#baz", "foo$qux"],
        ),
        # Numbers success - pyspark
        ("pyspark", ["foo1", "foo2", "foo3"], "foo", "success", None),
        # Numbers at start violations - pyspark
        (
            "pyspark",
            ["1foo", "2foo", "3foo"],
            "foo",
            "violations",
            ["1foo", "2foo", "3foo"],
        ),
        # Long string success - pyspark
        (
            "pyspark",
            ["foo" + "a" * 97, "foo" + "b" * 97, "foo" + "c" * 97],
            "foo",
            "success",
            None,
        ),
        # Long string violations - pyspark
        (
            "pyspark",
            ["a" * 100, "b" * 100, "c" * 100],
            "foo",
            "violations",
            ["a" * 100, "b" * 100, "c" * 100],
        ),
        # Mixed violations - pyspark
        (
            "pyspark",
            ["foobar", "bar", "foo123"],
            "foo",
            "violations",
            ["bar"],
        ),
    ],
    ids=[
        "pyspark_basic_success",
        "pyspark_basic_violations",
        "pyspark_exact_match_success",
        "pyspark_exact_match_mixed",
        "pyspark_empty_string_violations",
        "pyspark_whitespace_only_violations",
        "pyspark_whitespace_in_text_success",
        "pyspark_whitespace_at_end_violations",
        "pyspark_special_char_at_violations",
        "pyspark_special_char_in_prefix_violations",
        "pyspark_numbers_success",
        "pyspark_numbers_at_start_violations",
        "pyspark_long_string_success",
        "pyspark_long_string_violations",
        "pyspark_mixed_violations",
    ],
)
def test_expectation_basic_scenarios_pyspark(
    df_type, data, prefix, expected_result, expected_violations, spark
):
    """Test various scenarios for ExpectationStringStartsWith expectation."""
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationStringStartsWith",
        column_name="col1",
        prefix=prefix,
    )

    data_frame = create_pyspark_dataframe(data, "col1", spark)
    result = expectation.validate(data_frame=data_frame)

    if expected_result == "success":
        assert str(result) == str(
            DataFrameExpectationSuccessMessage(expectation_name="ExpectationStringStartsWith")
        ), f"Expected success message but got: {result}"

        # Also test with suite
        expectations_suite = DataFrameExpectationsSuite().expect_string_starts_with(
            column_name="col1", prefix=prefix
        )
        suite_result = expectations_suite.build().run(data_frame=data_frame)
        assert suite_result is not None, "Expected SuiteExecutionResult"
        assert isinstance(suite_result, SuiteExecutionResult), (
            "Result should be SuiteExecutionResult"
        )
        assert suite_result.success, "Expected all expectations to pass"
        assert suite_result.total_passed == 1, "Expected 1 passed expectation"
        assert suite_result.total_failed == 0, "Expected 0 failed expectations"
    else:  # violations
        violations_df = create_pyspark_dataframe(expected_violations, "col1", spark)
        expected_message = (
            f"Found {len(expected_violations)} row(s) where 'col1' does not start with '{prefix}'."
        )

        assert str(result) == str(
            DataFrameExpectationFailureMessage(
                expectation_str=str(expectation),
                data_frame_type=DataFrameType(df_type),
                violations_data_frame=violations_df,
                message=expected_message,
                limit_violations=5,
            )
        ), f"Expected failure message but got: {result}"

        # Also test with suite
        expectations_suite = DataFrameExpectationsSuite().expect_string_starts_with(
            column_name="col1", prefix=prefix
        )
        with pytest.raises(DataFrameExpectationsSuiteFailure):
            expectations_suite.build().run(data_frame=data_frame)


def test_column_missing_error_pandas():
    """Test that missing column raises appropriate error for pandas."""
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationStringStartsWith",
        column_name="col1",
        prefix="foo",
    )

    data_frame = pd.DataFrame({"col2": ["foobar", "foo123", "foobaz"]})
    result = expectation.validate(data_frame=data_frame)

    expected_failure_message = DataFrameExpectationFailureMessage(
        expectation_str=str(expectation),
        data_frame_type=DataFrameType.PANDAS,
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


@pytest.mark.pyspark
def test_column_missing_error_pyspark(spark):
    """Test that missing column raises appropriate error for pyspark."""
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationStringStartsWith",
        column_name="col1",
        prefix="foo",
    )

    data_frame = create_pyspark_dataframe(["foobar", "foo123", "foobaz"], "col2", spark)
    result = expectation.validate(data_frame=data_frame)

    expected_failure_message = DataFrameExpectationFailureMessage(
        expectation_str=str(expectation),
        data_frame_type=DataFrameType.PYSPARK,
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
