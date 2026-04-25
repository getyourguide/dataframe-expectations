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
    "df_data, column_name, max_percentage, expected_result, expected_message",
    [
        # No nulls - should pass
        (
            {"col1": ([1, 2, 3, 4, 5], "long"), "col2": (["a", "b", "c", "d", "e"], "string")},
            "col1",
            10.0,
            "success",
            None,
        ),
        # Within threshold - 25% < 30%
        ({"col1": ([1, None, 3, 4], "long")}, "col1", 30.0, "success", None),
        # Exactly at threshold - 20% == 20%
        (
            {"col1": ([1, 2, None, 4, 5], "long"), "col2": ([None, "b", "c", "d", "e"], "string")},
            "col1",
            20.0,
            "success",
            None,
        ),
        # With NaN/None - 33.33% < 50%
        (
            {"col1": ([1, 2, 3], "long"), "col2": ([4.0, None, 6.0], "double")},
            "col2",
            50.0,
            "success",
            None,
        ),
        # Exceeds threshold - 50% > 20%
        (
            {"col1": ([1, None, 3, None], "long"), "col2": ([None, "b", "c", "d"], "string")},
            "col1",
            20.0,
            "failure",
            "Column 'col1' has 50.00% null values, expected at most 20.00%.",
        ),
        # All nulls in column - 100% > 50%
        (
            {"col1": ([None, None], "long"), "col2": ([1, 2], "long")},
            "col1",
            50.0,
            "failure",
            "Column 'col1' has 100.00% null values, expected at most 50.00%.",
        ),
        # Zero threshold failure - 33.33% > 0%
        (
            {"col1": ([1, None, 3], "long")},
            "col1",
            0.0,
            "failure",
            "Column 'col1' has 33.33% null values, expected at most 0.00%.",
        ),
        # Hundred threshold success - 100% <= 100%
        (
            {"col1": ([None, None, None], "long"), "col2": ([None, None, None], "long")},
            "col1",
            100.0,
            "success",
            None,
        ),
        # Empty DataFrame - 0% <= 10%
        ({"col1": ([], "long")}, "col1", 10.0, "success", None),
        # Single null value - 100% > 50%
        (
            {"col1": ([None], "long")},
            "col1",
            50.0,
            "failure",
            "Column 'col1' has 100.00% null values, expected at most 50.00%.",
        ),
        # Single non-null value - 0% <= 10%
        ({"col1": ([1], "long")}, "col1", 10.0, "success", None),
        # Other columns with nulls don't affect - 0% in col1 <= 10%
        (
            {"col1": ([1, 2, 3], "long"), "col2": ([None, None, None], "long")},
            "col1",
            10.0,
            "success",
            None,
        ),
        # Mixed data types with nulls - 25% < 50%
        (
            {
                "int_col": ([1, None, 3, 4], "long"),
                "str_col": (["a", "b", None, "d"], "string"),
                "float_col": ([1.1, 2.2, 3.3, None], "double"),
            },
            "float_col",
            50.0,
            "success",
            None,
        ),
        # Precision boundary - 25% == 25%
        ({"col1": ([1, None, 3, 4], "long")}, "col1", 25.0, "success", None),
    ],
    ids=[
        "no_nulls",
        "within_threshold",
        "exactly_at_threshold",
        "with_nan",
        "exceeds_threshold",
        "all_nulls",
        "zero_threshold_failure",
        "hundred_threshold_success",
        "empty",
        "single_null",
        "single_not_null",
        "other_columns_ignored",
        "mixed_data_types",
        "precision_boundary",
    ],
)
def test_expectation_basic_scenarios(
    dataframe_factory, df_data, column_name, max_percentage, expected_result, expected_message
):
    """
    Test the expectation for various scenarios across pandas and PySpark DataFrames.
    Tests both direct expectation validation and suite-based validation.
    Covers: success cases, failures (exceeds threshold), edge cases (empty, zero/hundred threshold, single values),
    boundary conditions, column isolation, mixed data types, and precision boundaries.
    """
    df_lib, make_df = dataframe_factory
    data_frame = make_df(df_data)

    # Test 1: Direct expectation validation
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationMaxNullPercentage",
        column_name=column_name,
        max_percentage=max_percentage,
    )

    result = expectation.validate(data_frame=data_frame)

    if expected_result == "success":
        assert str(result) == str(
            DataFrameExpectationSuccessMessage(expectation_name="ExpectationMaxNullPercentage")
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
    expectations_suite = DataFrameExpectationsSuite().expect_max_null_percentage(
        column_name=column_name, max_percentage=max_percentage
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

    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationMaxNullPercentage",
        column_name="nonexistent_col",
        max_percentage=50.0,
    )

    data_frame = make_df({"col1": ([1, 2, 3], "long"), "col2": ([4, 5, 6], "long")})

    result = expectation.validate(data_frame=data_frame)
    assert isinstance(result, DataFrameExpectationFailureMessage), (
        f"Expected DataFrameExpectationFailureMessage but got: {type(result)}"
    )
    result_str = str(result)
    assert "nonexistent_col" in result_str, (
        f"Expected 'nonexistent_col' in result message: {result_str}"
    )

    expectations_suite = DataFrameExpectationsSuite().expect_max_null_percentage(
        column_name="nonexistent_col", max_percentage=50.0
    )
    with pytest.raises(DataFrameExpectationsSuiteFailure):
        expectations_suite.build().run(data_frame=data_frame)


def test_invalid_parameters():
    """Test that appropriate errors are raised for invalid parameters."""
    # Test negative max_percentage
    with pytest.raises(ValueError) as context:
        DataFrameExpectationRegistry.get_expectation(
            expectation_name="ExpectationMaxNullPercentage",
            column_name="col1",
            max_percentage=-1.0,
        )
    assert "max_percentage must be between" in str(context.value), (
        f"Expected 'max_percentage must be between' in error message: {str(context.value)}"
    )

    # Test max_percentage > 100
    with pytest.raises(ValueError) as context:
        DataFrameExpectationRegistry.get_expectation(
            expectation_name="ExpectationMaxNullPercentage",
            column_name="col1",
            max_percentage=101.0,
        )
    assert "max_percentage must be between" in str(context.value), (
        f"Expected 'max_percentage must be between' in error message: {str(context.value)}"
    )


def test_large_dataset_performance(dataframe_factory):
    """Test the expectation with a larger dataset to ensure reasonable performance."""
    df_lib, make_df = dataframe_factory

    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationMaxNullPercentage",
        column_name="col1",
        max_percentage=10.0,
    )
    # Create a DataFrame with 1000 rows and 50 nulls (every 20th value is None) = 5% null
    data = [None if i % 20 == 0 else i for i in range(1000)]
    data_frame = make_df({"col1": (data, "long")})
    result = expectation.validate(data_frame=data_frame)
    assert str(result) == str(
        DataFrameExpectationSuccessMessage(expectation_name="ExpectationMaxNullPercentage")
    ), f"Expected success message but got: {result}"
