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
        expectation_name="ExpectationStringNotContains",
        column_name="col1",
        substring="foo",
    )
    assert expectation.get_expectation_name() == "ExpectationStringNotContains", (
        f"Expected 'ExpectationStringNotContains' but got: {expectation.get_expectation_name()}"
    )


@pytest.mark.parametrize(
    "df_type,data,substring,expected_result,expected_violations,test_id",
    [
        # Basic success and violations - pandas
        ("pandas", ["bar", "baz", "qux"], "foo", "success", None, "pandas_basic_success"),
        ("pandas", ["test", "demo", "exam"], "xyz", "success", None, "pandas_success_no_match"),
        (
            "pandas",
            ["hello", "world", "python"],
            "java",
            "success",
            None,
            "pandas_success_different_substring",
        ),
        # Basic success and violations - pyspark
        ("pyspark", ["bar", "baz", "qux"], "foo", "success", None, "pyspark_basic_success"),
        ("pyspark", ["test", "demo", "exam"], "xyz", "success", None, "pyspark_success_no_match"),
        (
            "pyspark",
            ["hello", "world", "python"],
            "java",
            "success",
            None,
            "pyspark_success_different_substring",
        ),
        # Basic violations - pandas
        (
            "pandas",
            ["foobar", "bar", "foo"],
            "foo",
            "violations",
            ["foobar", "foo"],
            "pandas_basic_violations",
        ),
        (
            "pandas",
            ["testing", "test", "attest"],
            "test",
            "violations",
            ["testing", "test", "attest"],
            "pandas_all_violations",
        ),
        (
            "pandas",
            ["good", "bad", "badge"],
            "bad",
            "violations",
            ["bad", "badge"],
            "pandas_mixed_violations",
        ),
        # Basic violations - pyspark
        (
            "pyspark",
            ["foobar", "bar", "foo"],
            "foo",
            "violations",
            ["foobar", "foo"],
            "pyspark_basic_violations",
        ),
        (
            "pyspark",
            ["testing", "test", "attest"],
            "test",
            "violations",
            ["testing", "test", "attest"],
            "pyspark_all_violations",
        ),
        # Substring at beginning - pandas
        (
            "pandas",
            ["prefix_test", "prefix_demo", "other"],
            "prefix",
            "violations",
            ["prefix_test", "prefix_demo"],
            "pandas_substring_at_beginning",
        ),
        (
            "pandas",
            ["no_match", "also_no", "nope"],
            "prefix",
            "success",
            None,
            "pandas_no_substring_at_beginning",
        ),
        # Substring at beginning - pyspark
        (
            "pyspark",
            ["prefix_test", "prefix_demo", "other"],
            "prefix",
            "violations",
            ["prefix_test", "prefix_demo"],
            "pyspark_substring_at_beginning",
        ),
        (
            "pyspark",
            ["no_match", "also_no", "nope"],
            "prefix",
            "success",
            None,
            "pyspark_no_substring_at_beginning",
        ),
        # Substring at end - pandas
        (
            "pandas",
            ["test_suffix", "demo_suffix", "other"],
            "suffix",
            "violations",
            ["test_suffix", "demo_suffix"],
            "pandas_substring_at_end",
        ),
        (
            "pandas",
            ["no_match", "also_no", "nope"],
            "suffix",
            "success",
            None,
            "pandas_no_substring_at_end",
        ),
        # Substring at end - pyspark
        (
            "pyspark",
            ["test_suffix", "demo_suffix", "other"],
            "suffix",
            "violations",
            ["test_suffix", "demo_suffix"],
            "pyspark_substring_at_end",
        ),
        (
            "pyspark",
            ["no_match", "also_no", "nope"],
            "suffix",
            "success",
            None,
            "pyspark_no_substring_at_end",
        ),
        # Substring in middle - pandas
        (
            "pandas",
            ["pre_mid_post", "another_mid_test", "nomatch"],
            "mid",
            "violations",
            ["pre_mid_post", "another_mid_test"],
            "pandas_substring_in_middle",
        ),
        (
            "pandas",
            ["no_match", "also_no", "nope"],
            "mid",
            "success",
            None,
            "pandas_no_substring_in_middle",
        ),
        # Substring in middle - pyspark
        (
            "pyspark",
            ["pre_mid_post", "another_mid_test", "nomatch"],
            "mid",
            "violations",
            ["pre_mid_post", "another_mid_test"],
            "pyspark_substring_in_middle",
        ),
        # Case sensitivity - pandas
        ("pandas", ["FOO", "Foo", "fOo"], "foo", "success", None, "pandas_case_sensitive_success"),
        (
            "pandas",
            ["foo", "FOO", "test"],
            "FOO",
            "violations",
            ["FOO"],
            "pandas_case_sensitive_violations",
        ),
        # Case sensitivity - pyspark
        (
            "pyspark",
            ["FOO", "Foo", "fOo"],
            "foo",
            "success",
            None,
            "pyspark_case_sensitive_success",
        ),
        (
            "pyspark",
            ["foo", "FOO", "test"],
            "FOO",
            "violations",
            ["FOO"],
            "pyspark_case_sensitive_violations",
        ),
        # Empty strings - pandas
        ("pandas", ["", "", ""], "foo", "success", None, "pandas_empty_string_success"),
        (
            "pandas",
            ["", "foo", ""],
            "foo",
            "violations",
            ["foo"],
            "pandas_empty_string_with_violation",
        ),
        # Empty strings - pyspark
        ("pyspark", ["", "", ""], "foo", "success", None, "pyspark_empty_string_success"),
        # Whitespace handling - pandas
        ("pandas", ["   ", "  ", " "], "test", "success", None, "pandas_whitespace_only_success"),
        (
            "pandas",
            ["test with spaces", "test", "no match"],
            "test",
            "violations",
            ["test with spaces", "test"],
            "pandas_whitespace_in_text_violations",
        ),
        (
            "pandas",
            ["   test   ", "test", "clean"],
            "test",
            "violations",
            ["   test   ", "test"],
            "pandas_whitespace_around_violations",
        ),
        # Whitespace handling - pyspark
        ("pyspark", ["   ", "  ", " "], "test", "success", None, "pyspark_whitespace_only_success"),
        (
            "pyspark",
            ["test with spaces", "test", "no match"],
            "test",
            "violations",
            ["test with spaces", "test"],
            "pyspark_whitespace_in_text_violations",
        ),
        # Special characters - pandas
        (
            "pandas",
            ["test@email", "user@domain", "plain"],
            "@",
            "violations",
            ["test@email", "user@domain"],
            "pandas_special_char_at_violations",
        ),
        (
            "pandas",
            ["no-match", "also-no", "nope"],
            "@",
            "success",
            None,
            "pandas_special_char_at_success",
        ),
        (
            "pandas",
            ["test#tag", "demo#hash", "plain"],
            "#",
            "violations",
            ["test#tag", "demo#hash"],
            "pandas_special_char_hash_violations",
        ),
        # Special characters - pyspark
        (
            "pyspark",
            ["test@email", "user@domain", "plain"],
            "@",
            "violations",
            ["test@email", "user@domain"],
            "pyspark_special_char_at_violations",
        ),
        (
            "pyspark",
            ["no-match", "also-no", "nope"],
            "@",
            "success",
            None,
            "pyspark_special_char_at_success",
        ),
        # Numbers in strings - pandas
        (
            "pandas",
            ["version123", "test456", "plain"],
            "123",
            "violations",
            ["version123"],
            "pandas_numbers_violations",
        ),
        ("pandas", ["v1.0", "v2.1", "v3.5"], "123", "success", None, "pandas_numbers_no_match"),
        (
            "pandas",
            ["test", "demo", "123"],
            "123",
            "violations",
            ["123"],
            "pandas_numbers_exact_match",
        ),
        # Numbers in strings - pyspark
        (
            "pyspark",
            ["version123", "test456", "plain"],
            "123",
            "violations",
            ["version123"],
            "pyspark_numbers_violations",
        ),
        ("pyspark", ["v1.0", "v2.1", "v3.5"], "123", "success", None, "pyspark_numbers_no_match"),
        # Single character substring - pandas
        ("pandas", ["a", "b", "c"], "x", "success", None, "pandas_single_char_success"),
        ("pandas", ["a", "b", "c"], "a", "violations", ["a"], "pandas_single_char_violation"),
        # Single character substring - pyspark
        ("pyspark", ["a", "b", "c"], "x", "success", None, "pyspark_single_char_success"),
        ("pyspark", ["a", "b", "c"], "a", "violations", ["a"], "pyspark_single_char_violation"),
        # Long strings and substrings - pandas
        (
            "pandas",
            ["a" * 100, "b" * 100, "c" * 100],
            "x",
            "success",
            None,
            "pandas_long_string_success",
        ),
        (
            "pandas",
            ["a" * 50 + "test" + "b" * 50, "clean" * 20, "other"],
            "test",
            "violations",
            ["a" * 50 + "test" + "b" * 50],
            "pandas_long_string_with_substring",
        ),
        # Long strings and substrings - pyspark
        (
            "pyspark",
            ["a" * 100, "b" * 100, "c" * 100],
            "x",
            "success",
            None,
            "pyspark_long_string_success",
        ),
        (
            "pyspark",
            ["a" * 50 + "test" + "b" * 50, "clean" * 20, "other"],
            "test",
            "violations",
            ["a" * 50 + "test" + "b" * 50],
            "pyspark_long_string_with_substring",
        ),
        # Exact match (string equals substring) - pandas
        ("pandas", ["test", "demo", "exam"], "test", "violations", ["test"], "pandas_exact_match"),
        ("pandas", ["test", "demo", "exam"], "testing", "success", None, "pandas_no_exact_match"),
        # Exact match - pyspark
        (
            "pyspark",
            ["test", "demo", "exam"],
            "test",
            "violations",
            ["test"],
            "pyspark_exact_match",
        ),
    ],
)
def test_expectation_basic_scenarios(
    df_type, data, substring, expected_result, expected_violations, test_id, spark
):
    """Test various scenarios for ExpectationStringNotContains expectation."""
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationStringNotContains",
        column_name="col1",
        substring=substring,
    )

    data_frame = create_dataframe(df_type, data, "col1", spark)
    result = expectation.validate(data_frame=data_frame)

    if expected_result == "success":
        assert str(result) == str(
            DataFrameExpectationSuccessMessage(expectation_name="ExpectationStringNotContains")
        ), f"Test {test_id}: Expected success message but got: {result}"

        # Also test with suite
        expectations_suite = DataFrameExpectationsSuite().expect_string_not_contains(
            column_name="col1", substring=substring
        )
        suite_result = expectations_suite.build().run(data_frame=data_frame)
        assert suite_result is None, f"Test {test_id}: Expected no exceptions to be raised"
    else:  # violations
        violations_df = create_dataframe(df_type, expected_violations, "col1", spark)
        expected_message = (
            f"Found {len(expected_violations)} row(s) where 'col1' contains '{substring}'."
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
        expectations_suite = DataFrameExpectationsSuite().expect_string_not_contains(
            column_name="col1", substring=substring
        )
        with pytest.raises(DataFrameExpectationsSuiteFailure):
            expectations_suite.build().run(data_frame=data_frame)


@pytest.mark.parametrize("df_type", ["pandas", "pyspark"])
def test_column_missing_error(df_type, spark):
    """Test that missing column raises appropriate error."""
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationStringNotContains",
        column_name="col1",
        substring="foo",
    )

    data_frame = create_dataframe(df_type, ["bar", "baz", "qux"], "col2", spark)
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
    expectations_suite = DataFrameExpectationsSuite().expect_string_not_contains(
        column_name="col1", substring="foo"
    )
    with pytest.raises(DataFrameExpectationsSuiteFailure):
        expectations_suite.build().run(data_frame=data_frame)


def test_large_dataset_performance():
    """Test the expectation with a larger dataset to ensure performance."""
    # Create a larger dataset with 10,000 rows none containing "test"
    large_data = ["safe_value_" + str(i) for i in range(10000)]
    data_frame = pd.DataFrame({"col1": large_data})

    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationStringNotContains",
        column_name="col1",
        substring="test",
    )

    result = expectation.validate(data_frame=data_frame)
    # Should succeed as no values contain "test"
    assert isinstance(result, DataFrameExpectationSuccessMessage), (
        f"Large dataset test failed: expected success but got {type(result)}"
    )
