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
    """Helper function to create either pandas or PySpark DataFrame."""
    if df_type == "pandas":
        return pd.DataFrame({column_name: data})
    else:  # pyspark
        return spark.createDataFrame([(item,) for item in data], [column_name])


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
    "df_type, data, substring, expected_result, expected_violations",
    [
        # Basic success - pandas
        ("pandas", ["bar", "baz", "qux"], "foo", "success", None),
        # Basic success - pyspark
        ("pyspark", ["bar", "baz", "qux"], "foo", "success", None),
        # Success no match - pandas
        ("pandas", ["test", "demo", "exam"], "xyz", "success", None),
        # Success no match - pyspark
        ("pyspark", ["test", "demo", "exam"], "xyz", "success", None),
        # Success different substring - pandas
        (
            "pandas",
            ["hello", "world", "python"],
            "java",
            "success",
            None,
        ),
        # Success different substring - pyspark
        (
            "pyspark",
            ["hello", "world", "python"],
            "java",
            "success",
            None,
        ),
        # Basic violations - pandas
        (
            "pandas",
            ["foobar", "bar", "foo"],
            "foo",
            "violations",
            ["foobar", "foo"],
        ),
        # Basic violations - pyspark
        (
            "pyspark",
            ["foobar", "bar", "foo"],
            "foo",
            "violations",
            ["foobar", "foo"],
        ),
        # All violations - pandas
        (
            "pandas",
            ["testing", "test", "attest"],
            "test",
            "violations",
            ["testing", "test", "attest"],
        ),
        # All violations - pyspark
        (
            "pyspark",
            ["testing", "test", "attest"],
            "test",
            "violations",
            ["testing", "test", "attest"],
        ),
        # Mixed violations - pandas
        (
            "pandas",
            ["good", "bad", "badge"],
            "bad",
            "violations",
            ["bad", "badge"],
        ),
        # Mixed violations - pyspark
        (
            "pyspark",
            ["good", "bad", "badge"],
            "bad",
            "violations",
            ["bad", "badge"],
        ),
        # Substring at beginning - pandas
        (
            "pandas",
            ["prefix_test", "prefix_demo", "other"],
            "prefix",
            "violations",
            ["prefix_test", "prefix_demo"],
        ),
        # Substring at beginning - pyspark
        (
            "pyspark",
            ["prefix_test", "prefix_demo", "other"],
            "prefix",
            "violations",
            ["prefix_test", "prefix_demo"],
        ),
        # No substring at beginning - pandas
        (
            "pandas",
            ["no_match", "also_no", "nope"],
            "prefix",
            "success",
            None,
        ),
        # No substring at beginning - pyspark
        (
            "pyspark",
            ["no_match", "also_no", "nope"],
            "prefix",
            "success",
            None,
        ),
        # Substring at end - pandas
        (
            "pandas",
            ["test_suffix", "demo_suffix", "other"],
            "suffix",
            "violations",
            ["test_suffix", "demo_suffix"],
        ),
        # Substring at end - pyspark
        (
            "pyspark",
            ["test_suffix", "demo_suffix", "other"],
            "suffix",
            "violations",
            ["test_suffix", "demo_suffix"],
        ),
        # No substring at end - pandas
        (
            "pandas",
            ["no_match", "also_no", "nope"],
            "suffix",
            "success",
            None,
        ),
        # No substring at end - pyspark
        (
            "pyspark",
            ["no_match", "also_no", "nope"],
            "suffix",
            "success",
            None,
        ),
        # Substring in middle - pandas
        (
            "pandas",
            ["pre_mid_post", "another_mid_test", "nomatch"],
            "mid",
            "violations",
            ["pre_mid_post", "another_mid_test"],
        ),
        # Substring in middle - pyspark
        (
            "pyspark",
            ["pre_mid_post", "another_mid_test", "nomatch"],
            "mid",
            "violations",
            ["pre_mid_post", "another_mid_test"],
        ),
        # No substring in middle - pandas
        (
            "pandas",
            ["no_match", "also_no", "nope"],
            "mid",
            "success",
            None,
        ),
        # No substring in middle - pyspark
        (
            "pyspark",
            ["no_match", "also_no", "nope"],
            "mid",
            "success",
            None,
        ),
        # Case sensitive success - pandas
        ("pandas", ["FOO", "Foo", "fOo"], "foo", "success", None),
        # Case sensitive success - pyspark
        (
            "pyspark",
            ["FOO", "Foo", "fOo"],
            "foo",
            "success",
            None,
        ),
        # Case sensitive violations - pandas
        (
            "pandas",
            ["foo", "FOO", "test"],
            "FOO",
            "violations",
            ["FOO"],
        ),
        # Case sensitive violations - pyspark
        (
            "pyspark",
            ["foo", "FOO", "test"],
            "FOO",
            "violations",
            ["FOO"],
        ),
        # Empty string success - pandas
        ("pandas", ["", "", ""], "foo", "success", None),
        # Empty string success - pyspark
        ("pyspark", ["", "", ""], "foo", "success", None),
        # Empty string with violation - pandas
        (
            "pandas",
            ["", "foo", ""],
            "foo",
            "violations",
            ["foo"],
        ),
        # Empty string with violation - pyspark
        (
            "pyspark",
            ["", "foo", ""],
            "foo",
            "violations",
            ["foo"],
        ),
        # Whitespace only success - pandas
        ("pandas", ["   ", "  ", " "], "test", "success", None),
        # Whitespace only success - pyspark
        ("pyspark", ["   ", "  ", " "], "test", "success", None),
        # Whitespace in text violations - pandas
        (
            "pandas",
            ["test with spaces", "test", "no match"],
            "test",
            "violations",
            ["test with spaces", "test"],
        ),
        # Whitespace in text violations - pyspark
        (
            "pyspark",
            ["test with spaces", "test", "no match"],
            "test",
            "violations",
            ["test with spaces", "test"],
        ),
        # Whitespace around violations - pandas
        (
            "pandas",
            ["   test   ", "test", "clean"],
            "test",
            "violations",
            ["   test   ", "test"],
        ),
        # Whitespace around violations - pyspark
        (
            "pyspark",
            ["   test   ", "test", "clean"],
            "test",
            "violations",
            ["   test   ", "test"],
        ),
        # Special char at violations - pandas
        (
            "pandas",
            ["test@email", "user@domain", "plain"],
            "@",
            "violations",
            ["test@email", "user@domain"],
        ),
        # Special char at violations - pyspark
        (
            "pyspark",
            ["test@email", "user@domain", "plain"],
            "@",
            "violations",
            ["test@email", "user@domain"],
        ),
        # Special char at success - pandas
        (
            "pandas",
            ["no-match", "also-no", "nope"],
            "@",
            "success",
            None,
        ),
        # Special char at success - pyspark
        (
            "pyspark",
            ["no-match", "also-no", "nope"],
            "@",
            "success",
            None,
        ),
        # Special char hash violations - pandas
        (
            "pandas",
            ["test#tag", "demo#hash", "plain"],
            "#",
            "violations",
            ["test#tag", "demo#hash"],
        ),
        # Special char hash violations - pyspark
        (
            "pyspark",
            ["test#tag", "demo#hash", "plain"],
            "#",
            "violations",
            ["test#tag", "demo#hash"],
        ),
        # Numbers violations - pandas
        (
            "pandas",
            ["version123", "test456", "plain"],
            "123",
            "violations",
            ["version123"],
        ),
        # Numbers violations - pyspark
        (
            "pyspark",
            ["version123", "test456", "plain"],
            "123",
            "violations",
            ["version123"],
        ),
        # Numbers no match - pandas
        ("pandas", ["v1.0", "v2.1", "v3.5"], "123", "success", None),
        # Numbers no match - pyspark
        ("pyspark", ["v1.0", "v2.1", "v3.5"], "123", "success", None),
        # Numbers exact match - pandas
        (
            "pandas",
            ["test", "demo", "123"],
            "123",
            "violations",
            ["123"],
        ),
        # Numbers exact match - pyspark
        (
            "pyspark",
            ["test", "demo", "123"],
            "123",
            "violations",
            ["123"],
        ),
        # Single char success - pandas
        ("pandas", ["a", "b", "c"], "x", "success", None),
        # Single char success - pyspark
        ("pyspark", ["a", "b", "c"], "x", "success", None),
        # Single char violation - pandas
        ("pandas", ["a", "b", "c"], "a", "violations", ["a"]),
        # Single char violation - pyspark
        ("pyspark", ["a", "b", "c"], "a", "violations", ["a"]),
        # Long string success - pandas
        (
            "pandas",
            ["a" * 100, "b" * 100, "c" * 100],
            "x",
            "success",
            None,
        ),
        # Long string success - pyspark
        (
            "pyspark",
            ["a" * 100, "b" * 100, "c" * 100],
            "x",
            "success",
            None,
        ),
        # Long string with substring - pandas
        (
            "pandas",
            ["a" * 50 + "test" + "b" * 50, "clean" * 20, "other"],
            "test",
            "violations",
            ["a" * 50 + "test" + "b" * 50],
        ),
        # Long string with substring - pyspark
        (
            "pyspark",
            ["a" * 50 + "test" + "b" * 50, "clean" * 20, "other"],
            "test",
            "violations",
            ["a" * 50 + "test" + "b" * 50],
        ),
        # Exact match - pandas
        ("pandas", ["test", "demo", "exam"], "test", "violations", ["test"]),
        # Exact match - pyspark
        (
            "pyspark",
            ["test", "demo", "exam"],
            "test",
            "violations",
            ["test"],
        ),
        # No exact match - pandas
        ("pandas", ["test", "demo", "exam"], "testing", "success", None),
        # No exact match - pyspark
        ("pyspark", ["test", "demo", "exam"], "testing", "success", None),
    ],
    ids=[
        "pandas_basic_success",
        "pyspark_basic_success",
        "pandas_success_no_match",
        "pyspark_success_no_match",
        "pandas_success_different_substring",
        "pyspark_success_different_substring",
        "pandas_basic_violations",
        "pyspark_basic_violations",
        "pandas_all_violations",
        "pyspark_all_violations",
        "pandas_mixed_violations",
        "pyspark_mixed_violations",
        "pandas_substring_at_beginning",
        "pyspark_substring_at_beginning",
        "pandas_no_substring_at_beginning",
        "pyspark_no_substring_at_beginning",
        "pandas_substring_at_end",
        "pyspark_substring_at_end",
        "pandas_no_substring_at_end",
        "pyspark_no_substring_at_end",
        "pandas_substring_in_middle",
        "pyspark_substring_in_middle",
        "pandas_no_substring_in_middle",
        "pyspark_no_substring_in_middle",
        "pandas_case_sensitive_success",
        "pyspark_case_sensitive_success",
        "pandas_case_sensitive_violations",
        "pyspark_case_sensitive_violations",
        "pandas_empty_string_success",
        "pyspark_empty_string_success",
        "pandas_empty_string_with_violation",
        "pyspark_empty_string_with_violation",
        "pandas_whitespace_only_success",
        "pyspark_whitespace_only_success",
        "pandas_whitespace_in_text_violations",
        "pyspark_whitespace_in_text_violations",
        "pandas_whitespace_around_violations",
        "pyspark_whitespace_around_violations",
        "pandas_special_char_at_violations",
        "pyspark_special_char_at_violations",
        "pandas_special_char_at_success",
        "pyspark_special_char_at_success",
        "pandas_special_char_hash_violations",
        "pyspark_special_char_hash_violations",
        "pandas_numbers_violations",
        "pyspark_numbers_violations",
        "pandas_numbers_no_match",
        "pyspark_numbers_no_match",
        "pandas_numbers_exact_match",
        "pyspark_numbers_exact_match",
        "pandas_single_char_success",
        "pyspark_single_char_success",
        "pandas_single_char_violation",
        "pyspark_single_char_violation",
        "pandas_long_string_success",
        "pyspark_long_string_success",
        "pandas_long_string_with_substring",
        "pyspark_long_string_with_substring",
        "pandas_exact_match",
        "pyspark_exact_match",
        "pandas_no_exact_match",
        "pyspark_no_exact_match",
    ],
)
def test_expectation_basic_scenarios(
    df_type, data, substring, expected_result, expected_violations, spark
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
        ), f"Expected success message but got: {result}"

        # Also test with suite
        expectations_suite = DataFrameExpectationsSuite().expect_string_not_contains(
            column_name="col1", substring=substring
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
        violations_df = create_dataframe(df_type, expected_violations, "col1", spark)
        expected_message = (
            f"Found {len(expected_violations)} row(s) where 'col1' contains '{substring}'."
        )

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
        data_frame_type=str(df_type),
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
