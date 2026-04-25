import numpy as np
import pytest

from dataframe_expectations.registry import (
    DataFrameExpectationRegistry,
)
from dataframe_expectations.result_message import (
    DataFrameExpectationFailureMessage,
    DataFrameExpectationSuccessMessage,
)
from dataframe_expectations.suite import (
    DataFrameExpectationsSuite,
    DataFrameExpectationsSuiteFailure,
)


def test_expectation_name():
    """Test that the expectation name is correctly returned."""
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationColumnMinBetween",
        column_name="test_col",
        min_value=10,
        max_value=20,
    )
    # Note: minimum expectation delegates to quantile expectation
    assert expectation.get_expectation_name() == "ExpectationColumnQuantileBetween", (
        f"Expected 'ExpectationColumnQuantileBetween' but got: {expectation.get_expectation_name()}"
    )


def test_expectation_description():
    """Test that the expectation description is correctly returned."""
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationColumnMinBetween",
        column_name="test_col",
        min_value=10,
        max_value=20,
    )
    description = expectation.get_description()
    assert "minimum" in description, f"Expected 'minimum' in description: {description}"
    assert "test_col" in description, f"Expected 'test_col' in description: {description}"
    assert "10" in description, f"Expected '10' in description: {description}"
    assert "20" in description, f"Expected '20' in description: {description}"
    # Verify quantile properties
    assert expectation.quantile == 0.0, (
        f"Expected quantile to be 0.0 but got: {expectation.quantile}"
    )
    assert expectation.quantile_desc == "minimum", (
        f"Expected quantile_desc to be 'minimum' but got: {expectation.quantile_desc}"
    )


@pytest.mark.parametrize(
    "data, min_value, max_value, should_succeed, expected_message",
    [
        # Basic success scenarios
        ([20, 25, 30, 35], 15, 25, True, None),  # min = 20
        # Single row scenarios
        ([25], 20, 30, True, None),  # min = 25
        # Negative value scenarios
        ([-20, -15, -10, -5], -25, -15, True, None),  # min = -20
        # Float value scenarios
        ([1.1, 2.5, 3.7, 3.8], 1.0, 1.5, True, None),  # min = 1.1
        # Identical value scenarios
        ([25, 25, 25, 25], 24, 26, True, None),  # min = 25
        # Mixed type scenarios
        ([20, 25.5, 30, 37], 15, 25, True, None),  # min = 20
        # Zero scenarios
        ([-5, 0, 0, 2], -10, -1, True, None),  # min = -5
        # Null scenarios
        ([20, None, 35, None, 25], 15, 25, True, None),  # min = 20
        # Boundary scenarios - exact minimum
        ([20, 25, 30, 35], 20, 25, True, None),  # min = 20
        # Boundary scenarios - exact maximum
        ([20, 25, 30, 35], 15, 20, True, None),  # min = 20
        # Minimum calculation - mixed order
        ([100, 50, 75, 25], 24, 26, True, None),  # min = 25
        # Minimum calculation - zero
        ([0, 1, 2, 3], -0.1, 0.1, True, None),  # min = 0
        # Minimum calculation - negative
        ([-10, -5, -1, -20], -20.1, -19.9, True, None),  # min = -20
        # Minimum calculation - small differences
        ([1.001, 1.002, 1.003], 1.0, 1.002, True, None),  # min = 1.001
        # Minimum calculation - large numbers
        ([1e6, 1e5, 1e4], 1e4 - 100, 1e4 + 100, True, None),  # min = 1e4
        # Minimum calculation - very small numbers
        ([1e-6, 1e-5, 1e-4], 1e-7, 1e-5, True, None),  # min = 1e-6
        # Outlier impact - extreme low outlier
        ([1, 2, 3, -1000], -1100, -900, True, None),
        # Outlier impact - significant outlier
        ([100, 200, 300, 50], 40, 60, True, None),
        # Outlier impact - small outlier
        ([1.5, 2.0, 2.5, 0.1], 0.05, 0.15, True, None),
        # Identical values - integer repetition
        ([42, 42, 42, 42], 41.9, 42.1, True, None),
        # Identical values - float repetition
        ([3.14, 3.14, 3.14], 3.13, 3.15, True, None),
        # Identical values - negative repetition
        ([-7, -7, -7, -7, -7], -7.1, -6.9, True, None),
        # Identical values - zero repetition
        ([0, 0, 0], -0.1, 0.1, True, None),
        # Failure scenarios - minimum too low
        (
            [20, 25, 30, 35],
            25,
            35,
            False,
            "Column 'col1' minimum value 20.0 is not between 25 and 35.",
        ),
        # Failure scenarios - minimum too high
        (
            [20, 25, 30, 35],
            10,
            15,
            False,
            "Column 'col1' minimum value 20.0 is not between 10 and 15.",
        ),
        # Failure scenarios - all nulls
        ([None, None, None], 15, 25, False, "Column 'col1' contains only null values."),
        # Failure scenarios - empty
        ([], 15, 25, False, "Column 'col1' contains only null values."),
    ],
    ids=[
        "basic_success",
        "single_row",
        "negative_values",
        "float_values",
        "identical_values",
        "mixed_types",
        "with_zeros",
        "with_nulls",
        "boundary_exact_min",
        "boundary_exact_max",
        "calc_mixed_order",
        "calc_zero",
        "calc_negative",
        "calc_small_differences",
        "calc_large_numbers",
        "calc_very_small_numbers",
        "outlier_extreme_low",
        "outlier_significant",
        "outlier_small",
        "identical_integer",
        "identical_float",
        "identical_negative",
        "identical_zero",
        "min_too_low",
        "min_too_high",
        "all_nulls",
        "empty",
    ],
)
def test_expectation_basic_scenarios(
    data, min_value, max_value, should_succeed, expected_message, dataframe_factory
):
    """Test basic expectation scenarios."""
    df_lib, make_df = dataframe_factory
    df = make_df({"col1": (data, "double")})

    # Test through registry
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationColumnMinBetween",
        column_name="col1",
        min_value=min_value,
        max_value=max_value,
    )
    result = expectation.validate(data_frame=df)

    if should_succeed:
        assert isinstance(result, DataFrameExpectationSuccessMessage), (
            f"Expected success but got: {result}"
        )
    else:
        assert isinstance(result, DataFrameExpectationFailureMessage), (
            f"Expected failure but got: {result}"
        )
        assert expected_message in str(result), (
            f"Expected message '{expected_message}' in result: {result}"
        )

    # Test through suite
    suite = DataFrameExpectationsSuite().expect_column_min_between(
        column_name="col1", min_value=min_value, max_value=max_value
    )

    if should_succeed:
        suite_result = suite.build().run(data_frame=df)
        assert suite_result is not None, "Expected SuiteExecutionResult"
        assert suite_result.success, "Expected all expectations to pass"
    else:
        with pytest.raises(DataFrameExpectationsSuiteFailure):
            suite.build().run(data_frame=df)


def test_column_missing_error(dataframe_factory):
    """Test missing column error."""
    df_lib, make_df = dataframe_factory
    df = make_df({"col1": ([20, 25, 30, 35], "double")})
    expected_message = "Column 'nonexistent_col' does not exist in the DataFrame."

    # Test through registry
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationColumnMinBetween",
        column_name="nonexistent_col",
        min_value=15,
        max_value=25,
    )
    result = expectation.validate(data_frame=df)
    expected_failure = DataFrameExpectationFailureMessage(
        expectation_str=str(expectation),
        data_frame_type=df_lib.value,
        message=expected_message,
    )
    assert str(result) == str(expected_failure)

    # Test through suite
    suite = DataFrameExpectationsSuite().expect_column_min_between(
        column_name="nonexistent_col", min_value=15, max_value=25
    )
    with pytest.raises(DataFrameExpectationsSuiteFailure):
        suite.build().run(data_frame=df)


def test_large_dataset_performance(dataframe_factory):
    """Test the expectation with a larger dataset to ensure performance."""
    df_lib, make_df = dataframe_factory

    # Create a larger dataset with minimum around 10
    large_data = np.random.uniform(10, 60, 1000).tolist()
    data_frame = make_df({"col1": (large_data, "double")})

    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationColumnMinBetween",
        column_name="col1",
        min_value=9,
        max_value=12,
    )

    result = expectation.validate(data_frame=data_frame)
    # Should succeed as the minimum of uniform(10, 60) should be around 10
    assert isinstance(result, DataFrameExpectationSuccessMessage)
