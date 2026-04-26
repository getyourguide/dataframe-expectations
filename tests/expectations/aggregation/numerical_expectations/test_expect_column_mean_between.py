import numpy as np
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


def test_expectation_name():
    """Test that the expectation name is correctly returned."""
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationColumnMeanBetween",
        column_name="col1",
        min_value=10,
        max_value=20,
    )
    assert expectation.get_expectation_name() == "ExpectationColumnMeanBetween", (
        f"Expected 'ExpectationColumnMeanBetween' but got: {expectation.get_expectation_name()}"
    )


def test_expectation_description():
    """Test that the expectation description is correctly returned."""
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationColumnMeanBetween",
        column_name="test_col",
        min_value=10,
        max_value=20,
    )
    description = expectation.get_description()
    assert "mean" in description, f"Expected 'mean' in description: {description}"
    assert "test_col" in description, f"Expected 'test_col' in description: {description}"
    assert "10" in description, f"Expected '10' in description: {description}"
    assert "20" in description, f"Expected '20' in description: {description}"


@pytest.mark.parametrize(
    "data, arrow_type, min_value, max_value, expected_result, expected_message",
    [
        # Basic success scenarios
        ([20, 25, 30, 35], "long", 25, 30, "success", None),  # mean = 27.5
        # Single row scenarios
        ([25], "long", 20, 30, "success", None),  # mean = 25
        # Negative value scenarios
        ([-20, -15, -10, -5], "long", -15, -10, "success", None),  # mean = -12.5
        # Float value scenarios
        ([1.1, 2.5, 3.7, 3.8], "double", 2.5, 3.0, "success", None),  # mean = 2.775
        # Identical value scenarios
        ([25, 25, 25, 25], "long", 24, 26, "success", None),  # mean = 25
        # Mixed type scenarios (as double)
        ([20.0, 25.5, 30.0, 37.0], "double", 27, 29, "success", None),  # mean = 28.125
        # Zero scenarios
        ([-5, 0, 0, 5], "long", -2, 2, "success", None),  # mean = 0
        # Null scenarios (use long to preserve null handling)
        ([20, None, 30, None, 40], "long", 25, 35, "success", None),  # mean = 30
        # Boundary scenarios - exact min boundary (mean = 27.5)
        ([20, 25, 30, 35], "long", 27.5, 30, "success", None),
        # Boundary scenarios - exact max boundary (mean = 27.5)
        ([20, 25, 30, 35], "long", 25, 27.5, "success", None),
        # Failure scenarios - mean too low
        (
            [20, 25, 30, 35],
            "long",
            30,
            35,
            "failure",
            "Column 'col1' mean value 27.5 is not between 30 and 35.",
        ),
        # Failure scenarios - mean too high
        (
            [20, 25, 30, 35],
            "long",
            20,
            25,
            "failure",
            "Column 'col1' mean value 27.5 is not between 20 and 25.",
        ),
        # Failure scenarios - all nulls
        (
            [None, None, None],
            "long",
            25,
            30,
            "failure",
            "Column 'col1' contains only null values.",
        ),
        # Failure scenarios - empty
        ([], "long", 25, 30, "failure", "Column 'col1' contains only null values."),
        # Outlier scenarios - high
        ([1, 2, 3, 100], "long", 20, 30, "success", None),  # mean = 26.5
        # Outlier scenarios - low
        ([-100, 10, 20, 30], "long", -15, -5, "success", None),  # mean = -10
        # Outlier scenarios - extreme
        ([1, 2, 3, 4, 5, 1000], "long", 150, 200, "success", None),  # mean ≈ 169.17
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
        "mean_too_low",
        "mean_too_high",
        "all_nulls",
        "empty",
        "outlier_high",
        "outlier_low",
        "outlier_extreme",
    ],
)
def test_expectation_basic_scenarios(
    dataframe_factory, data, arrow_type, min_value, max_value, expected_result, expected_message
):
    """
    Test the expectation for various scenarios across pandas and PySpark DataFrames.
    Tests both direct expectation validation and suite-based validation.
    Covers: success cases, boundary conditions, failures (mean out of range, nulls, empty),
    and various data types (integers, floats, negatives, nulls, mixed types, outliers).
    """
    df_lib, make_df = dataframe_factory
    data_frame = make_df({"col1": (data, arrow_type)})

    # Test 1: Direct expectation validation
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationColumnMeanBetween",
        column_name="col1",
        min_value=min_value,
        max_value=max_value,
    )

    result = expectation.validate(data_frame=data_frame)

    if expected_result == "success":
        assert str(result) == str(
            DataFrameExpectationSuccessMessage(expectation_name="ExpectationColumnMeanBetween")
        ), f"Expected success message but got: {result}"
    else:  # failure
        expected_failure_message = DataFrameExpectationFailureMessage(
            expectation_str=str(expectation),
            data_frame_type=df_lib,
            message=expected_message,
        )
        assert str(result) == str(expected_failure_message), (
            f"Expected failure message but got: {result}"
        )

    # Test 2: Suite-based validation
    expectations_suite = DataFrameExpectationsSuite().expect_column_mean_between(
        column_name="col1", min_value=min_value, max_value=max_value
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
    """Test that an error is raised when the specified column is missing."""
    df_lib, make_df = dataframe_factory
    expected_message = "Column 'nonexistent_col' does not exist in the DataFrame."

    data_frame = make_df({"col1": ([20, 25, 30, 35], "long")})

    # Test 1: Direct expectation validation
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationColumnMeanBetween",
        column_name="nonexistent_col",
        min_value=25,
        max_value=30,
    )
    result = expectation.validate(data_frame=data_frame)
    expected_failure = DataFrameExpectationFailureMessage(
        expectation_str=str(expectation),
        data_frame_type=df_lib,
        message=expected_message,
    )
    assert str(result) == str(expected_failure), f"Expected failure message but got: {result}"

    # Test 2: Suite-based validation
    expectations_suite = DataFrameExpectationsSuite().expect_column_mean_between(
        column_name="nonexistent_col", min_value=25, max_value=30
    )
    with pytest.raises(DataFrameExpectationsSuiteFailure):
        expectations_suite.build().run(data_frame=data_frame)


def test_precision_handling(dataframe_factory):
    """Test mean calculation precision with various numeric types."""
    df_lib, make_df = dataframe_factory
    # Test scenarios with different levels of precision
    precision_tests = [
        # (data, description)
        ([1.1111, 2.2222, 3.3333], "high precision decimals"),
        ([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], "integer sequence"),
        ([0.1, 0.2, 0.3, 0.4, 0.5], "decimal sequence"),
        ([1e-6, 2e-6, 3e-6], "scientific notation"),
    ]

    for data, description in precision_tests:
        data_frame = make_df({"col1": (data, "double")})
        calculated_mean = sum(data) / len(data)

        # Use a range around the calculated mean
        min_val = calculated_mean - 0.1
        max_val = calculated_mean + 0.1

        expectation = DataFrameExpectationRegistry.get_expectation(
            expectation_name="ExpectationColumnMeanBetween",
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
    # Create a larger dataset with mean around 50
    large_data = np.random.normal(50, 10, 1000).tolist()
    data_frame = make_df({"col1": (large_data, "double")})

    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationColumnMeanBetween",
        column_name="col1",
        min_value=45,
        max_value=55,
    )

    result = expectation.validate(data_frame=data_frame)
    # Should succeed as the mean of normal(50, 10) should be around 50
    assert isinstance(result, DataFrameExpectationSuccessMessage), (
        f"Large dataset test failed: expected success but got {type(result)}"
    )
