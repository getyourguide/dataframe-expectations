import pytest

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


@pytest.mark.parametrize(
    "df_data, column_name, max_count, expected_result, expected_message",
    [
        # No nulls - should pass
        (
            {"col1": ([1, 2, 3, 4, 5], "long"), "col2": (["a", "b", "c", "d", "e"], "string")},
            "col1",
            5,
            "success",
            None,
        ),
        # Within threshold - 2 nulls < 3
        ({"col1": ([1, None, 3, None, 5], "long")}, "col1", 3, "success", None),
        # Exactly at threshold - 2 nulls <= 2
        ({"col1": ([1, 2, None, 4, None], "long")}, "col1", 2, "success", None),
        # With NaN/null in double column - 1 null <= 2
        (
            {"col1": ([1, 2, 3], "long"), "col2": ([4.0, None, 6.0], "double")},
            "col2",
            2,
            "success",
            None,
        ),
        # Exceeds threshold - 3 nulls > 1
        (
            {"col1": ([1, None, None, None, 5], "long")},
            "col1",
            1,
            "failure",
            "Column 'col1' has 3 null values, expected at most 1.",
        ),
        # All nulls in column - 3 nulls > 1
        (
            {"col1": ([None, None, None], "long"), "col2": ([1, 2, 3], "long")},
            "col1",
            1,
            "failure",
            "Column 'col1' has 3 null values, expected at most 1.",
        ),
        # Zero threshold failure - 1 null > 0
        (
            {"col1": ([1, None, 3], "long")},
            "col1",
            0,
            "failure",
            "Column 'col1' has 1 null values, expected at most 0.",
        ),
        # Zero threshold success - 0 nulls <= 0
        (
            {"col1": ([1, 2, 3], "long"), "col2": ([None, None, None], "long")},
            "col1",
            0,
            "success",
            None,
        ),
        # Single null value - 1 null > 0
        (
            {"col1": ([None], "long")},
            "col1",
            0,
            "failure",
            "Column 'col1' has 1 null values, expected at most 0.",
        ),
        # Single non-null value - 0 nulls <= 0
        ({"col1": ([1], "long")}, "col1", 0, "success", None),
        # Other columns with nulls don't affect - 0 nulls in col1 <= 1
        (
            {"col1": ([1, 2, 3], "long"), "col2": ([None, None, None], "long")},
            "col1",
            1,
            "success",
            None,
        ),
        # Large threshold - 2 nulls <= 1000000
        ({"col1": ([1, None, 3, None, 5], "long")}, "col1", 1000000, "success", None),
    ],
    ids=[
        "no_nulls",
        "within_threshold",
        "exactly_at_threshold",
        "with_null_double",
        "exceeds_threshold",
        "all_nulls",
        "zero_threshold_failure",
        "zero_threshold_success",
        "single_null",
        "single_not_null",
        "other_columns_ignored",
        "large_threshold",
    ],
)
def test_expectation_basic_scenarios(
    dataframe_factory, df_data, column_name, max_count, expected_result, expected_message
):
    """
    Test the expectation for various scenarios across pandas and PySpark DataFrames.
    Tests both direct expectation validation and suite-based validation.
    Covers: success cases, failures (exceeds threshold), edge cases (zero threshold, single values),
    boundary conditions, column isolation, and various data types.
    """
    df_lib, make_df = dataframe_factory
    data_frame = make_df(df_data)

    # Test 1: Direct expectation validation
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationMaxNullCount",
        column_name=column_name,
        max_count=max_count,
    )

    result = expectation.validate(data_frame=data_frame)

    if expected_result == "success":
        assert str(result) == str(
            DataFrameExpectationSuccessMessage(expectation_name="ExpectationMaxNullCount")
        ), f"Expected success message but got: {result}"
    else:  # failure
        expected_failure_message = DataFrameExpectationFailureMessage(
            expectation_str=str(expectation),
            data_frame_type=df_lib.value,
            message=expected_message,
        )
        assert str(result) == str(expected_failure_message), (
            f"Expected failure message but got: {result}"
        )

    # Test 2: Suite-based validation
    expectations_suite = DataFrameExpectationsSuite().expect_max_null_count(
        column_name=column_name, max_count=max_count
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


def test_column_missing_error(dataframe_factory):
    """
    Test that an error is raised when the specified column is missing.
    Tests both direct expectation validation and suite-based validation.
    """
    df_lib, make_df = dataframe_factory
    data_frame = make_df({"col2": ([1, 2, 3, 4, 5], "long")})

    # Test 1: Direct expectation validation
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationMaxNullCount",
        column_name="col1",
        max_count=5,
    )

    result = expectation.validate(data_frame=data_frame)
    # The error message might vary, but should be a failure
    assert isinstance(result, DataFrameExpectationFailureMessage), (
        f"Expected DataFrameExpectationFailureMessage but got: {type(result)}"
    )
    result_str = str(result)
    assert "col1" in result_str, f"Expected 'col1' in result message: {result_str}"

    # Test 2: Suite-based validation
    expectations_suite = DataFrameExpectationsSuite().expect_max_null_count(
        column_name="col1", max_count=5
    )
    with pytest.raises(DataFrameExpectationsSuiteFailure):
        expectations_suite.build().run(data_frame=data_frame)


def test_invalid_parameters():
    """Test that appropriate errors are raised for invalid parameters."""
    # Test negative max_count
    with pytest.raises(ValueError) as context:
        DataFrameExpectationRegistry.get_expectation(
            expectation_name="ExpectationMaxNullCount",
            column_name="col1",
            max_count=-1,
        )
    assert "max_count must be non-negative" in str(context.value), (
        f"Expected 'max_count must be non-negative' in error message: {str(context.value)}"
    )


def test_large_dataset_performance(dataframe_factory):
    """Test the expectation with a larger dataset to ensure reasonable performance."""
    df_lib, make_df = dataframe_factory

    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationMaxNullCount",
        column_name="col1",
        max_count=100,
    )
    # Create a DataFrame with 1000 rows and 50 nulls (every 20th value is None)
    data = [None if i % 20 == 0 else i for i in range(1000)]
    data_frame = make_df({"col1": (data, "long")})
    result = expectation.validate(data_frame=data_frame)
    assert str(result) == str(
        DataFrameExpectationSuccessMessage(expectation_name="ExpectationMaxNullCount")
    ), f"Expected success message but got: {result}"
