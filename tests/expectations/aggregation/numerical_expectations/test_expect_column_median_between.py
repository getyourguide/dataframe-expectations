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
        expectation_name="ExpectationColumnMedianBetween",
        column_name="test_col",
        min_value=10,
        max_value=20,
    )
    # Note: median expectation delegates to quantile expectation
    assert expectation.get_expectation_name() == "ExpectationColumnQuantileBetween", (
        f"Expected 'ExpectationColumnQuantileBetween' but got: {expectation.get_expectation_name()}"
    )


def test_expectation_description():
    """Test that the expectation description is correctly returned."""
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationColumnMedianBetween",
        column_name="test_col",
        min_value=10,
        max_value=20,
    )
    description = expectation.get_description()
    assert "median" in description, f"Expected 'median' in description: {description}"
    assert "test_col" in description, f"Expected 'test_col' in description: {description}"
    assert "10" in description, f"Expected '10' in description: {description}"
    assert "20" in description, f"Expected '20' in description: {description}"
    # Verify quantile properties
    assert expectation.quantile == 0.5, (
        f"Expected quantile to be 0.5 but got: {expectation.quantile}"
    )
    assert expectation.quantile_desc == "median", (
        f"Expected quantile_desc to be 'median' but got: {expectation.quantile_desc}"
    )


@pytest.mark.parametrize(
    "data, min_value, max_value, should_succeed, expected_message",
    [
        # Basic success scenarios
        ([20, 25, 30, 35], 25, 30, True, None),  # median = 27.5
        # Single row scenarios
        ([25], 20, 30, True, None),  # median = 25
        # Negative value scenarios
        ([-20, -15, -10, -5], -15, -10, True, None),  # median = -12.5
        # Float value scenarios
        ([1.1, 2.5, 3.7, 3.8], 2.5, 3.5, True, None),  # median = 3.1
        # Identical value scenarios
        ([25, 25, 25, 25], 24, 26, True, None),  # median = 25
        # Mixed type scenarios
        ([20, 25.5, 30, 37], 27, 29, True, None),  # median = 27.75
        # Zero scenarios
        ([-5, 0, 0, 5], -1, 1, True, None),  # median = 0
        # Odd count scenarios
        ([10, 20, 30], 19, 21, True, None),  # median = 20
        # Even count scenarios
        ([10, 20, 30, 40], 24, 26, True, None),  # median = 25
        # Boundary scenarios - exact minimum
        ([20, 25, 30, 35], 27.5, 30, True, None),  # median = 27.5
        # Boundary scenarios - exact maximum
        ([20, 25, 30, 35], 25, 27.5, True, None),  # median = 27.5
        # Median calculation - odd count (middle element)
        ([1, 2, 3], 1.9, 2.1, True, None),
        # Median calculation - even count (average of middle two)
        ([1, 2, 3, 4], 2.4, 2.6, True, None),
        # Median calculation - single element
        ([5], 4.9, 5.1, True, None),
        # Median calculation - all identical values
        ([10, 10, 10], 9.9, 10.1, True, None),
        # Median calculation - two elements (average)
        ([1, 100], 50.4, 50.6, True, None),
        # Median calculation - odd count with outlier
        ([1, 2, 100], 1.9, 2.1, True, None),
        # Median calculation - even count with outliers
        ([1, 2, 99, 100], 50.4, 50.6, True, None),
        # Outlier resistance - high outlier
        ([1, 2, 3, 1000], 1.5, 2.5, True, None),
        # Outlier resistance - low outlier
        ([-1000, 10, 20, 30], 14, 16, True, None),
        # Outlier resistance - extreme high outlier
        ([1, 2, 3, 4, 5, 1000000], 2.5, 3.5, True, None),
        # Outlier resistance - extreme low outlier
        ([-1000000, 1, 2, 3, 4, 5], 2.4, 2.6, True, None),
        # Failure scenarios - median too low
        (
            [20, 25, 30, 35],
            30,
            35,
            False,
            "Column 'col1' median value 27.5 is not between 30 and 35.",
        ),
        # Failure scenarios - median too high
        (
            [20, 25, 30, 35],
            20,
            25,
            False,
            "Column 'col1' median value 27.5 is not between 20 and 25.",
        ),
        # Failure scenarios - median out of range
        (
            [10, 20, 30],
            25,
            30,
            False,
            "Column 'col1' median value 20.0 is not between 25 and 30.",
        ),
        # Failure scenarios - empty
        ([], 25, 30, False, "Column 'col1' contains only null values."),
    ],
    ids=[
        "basic_success",
        "single_row",
        "negative_values",
        "float_values",
        "identical_values",
        "mixed_types",
        "with_zeros",
        "odd_count",
        "even_count",
        "boundary_exact_min",
        "boundary_exact_max",
        "calc_odd_count",
        "calc_even_count",
        "calc_single_element",
        "calc_all_identical",
        "calc_two_elements",
        "calc_odd_with_outlier",
        "calc_even_with_outliers",
        "outlier_high",
        "outlier_low",
        "outlier_extreme_high",
        "outlier_extreme_low",
        "median_too_low",
        "median_too_high",
        "median_out_of_range",
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
        expectation_name="ExpectationColumnMedianBetween",
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
    suite = DataFrameExpectationsSuite().expect_column_median_between(
        column_name="col1", min_value=min_value, max_value=max_value
    )

    if should_succeed:
        suite_result = suite.build().run(data_frame=df)
        assert suite_result is not None, "Expected SuiteExecutionResult"
        assert suite_result.success, "Expected all expectations to pass"
    else:
        with pytest.raises(DataFrameExpectationsSuiteFailure):
            suite.build().run(data_frame=df)


def test_with_nulls(dataframe_factory):
    """Test median calculation with null values."""
    df_lib, make_df = dataframe_factory
    df = make_df({"col1": ([20, None, 30, None, 40], "double")})

    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationColumnMedianBetween",
        column_name="col1",
        min_value=25,
        max_value=35,
    )
    result = expectation.validate(data_frame=df)

    # Both pandas and PySpark ignore nulls, median of [20, 30, 40] = 30
    assert isinstance(result, DataFrameExpectationSuccessMessage)


def test_all_nulls(dataframe_factory):
    """Test median calculation with all null values."""
    df_lib, make_df = dataframe_factory
    df = make_df({"col1": ([None, None, None], "double")})

    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationColumnMedianBetween",
        column_name="col1",
        min_value=25,
        max_value=30,
    )
    result = expectation.validate(data_frame=df)

    assert isinstance(result, DataFrameExpectationFailureMessage)
    assert "contains only null values" in str(result)

    # Suite should raise failure
    suite = DataFrameExpectationsSuite().expect_column_median_between(
        column_name="col1", min_value=25, max_value=30
    )
    with pytest.raises(DataFrameExpectationsSuiteFailure):
        suite.build().run(data_frame=df)


def test_column_missing_error(dataframe_factory):
    """Test missing column error."""
    df_lib, make_df = dataframe_factory
    df = make_df({"col1": ([20, 25, 30, 35], "double")})
    expected_message = "Column 'nonexistent_col' does not exist in the DataFrame."

    # Test through registry
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationColumnMedianBetween",
        column_name="nonexistent_col",
        min_value=25,
        max_value=30,
    )
    result = expectation.validate(data_frame=df)
    expected_failure = DataFrameExpectationFailureMessage(
        expectation_str=str(expectation),
        data_frame_type=df_lib.value,
        message=expected_message,
    )
    assert str(result) == str(expected_failure)

    # Test through suite
    suite = DataFrameExpectationsSuite().expect_column_median_between(
        column_name="nonexistent_col", min_value=25, max_value=30
    )
    with pytest.raises(DataFrameExpectationsSuiteFailure):
        suite.build().run(data_frame=df)


def test_precision_handling(dataframe_factory):
    """Test median calculation precision with various numeric types."""
    df_lib, make_df = dataframe_factory

    # Test scenarios with different levels of precision
    precision_tests = [
        # (data, description)
        ([1.1111, 2.2222, 3.3333], "high precision decimals"),
        ([0.1, 0.2, 0.3, 0.4, 0.5], "decimal sequence"),
        ([1e-6, 2e-6, 3e-6, 4e-6, 5e-6], "scientific notation"),
        ([1.0, 1.5, 2.0, 2.5, 3.0], "half increments"),
    ]

    for data, description in precision_tests:
        data_frame = make_df({"col1": (data, "double")})
        calculated_median = np.median(data)

        # Use a small range around the calculated median
        min_val = calculated_median - 0.001
        max_val = calculated_median + 0.001

        expectation = DataFrameExpectationRegistry.get_expectation(
            expectation_name="ExpectationColumnMedianBetween",
            column_name="col1",
            min_value=min_val,
            max_value=max_val,
        )
        result = expectation.validate(data_frame=data_frame)
        assert isinstance(result, DataFrameExpectationSuccessMessage), (
            f"Precision test failed for {description}: expected success but got {type(result)}"
        )


def test_large_dataset_performance(dataframe_factory):
    """Test the expectation with a larger dataset to ensure performance."""
    df_lib, make_df = dataframe_factory

    # Create a larger dataset with median around 50
    large_data = np.random.normal(50, 10, 1001).tolist()  # Use odd count for deterministic median
    data_frame = make_df({"col1": (large_data, "double")})

    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationColumnMedianBetween",
        column_name="col1",
        min_value=45,
        max_value=55,
    )

    result = expectation.validate(data_frame=data_frame)
    # Should succeed as the median of normal(50, 10) should be around 50
    assert isinstance(result, DataFrameExpectationSuccessMessage)
