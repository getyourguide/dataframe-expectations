import pytest
import numpy as np
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


def test_expectation_name():
    """Test that the expectation name is correctly returned."""
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationValueLessThanEquals",
        column_name="col1",
        value=2,
    )
    assert expectation.get_expectation_name() == "ExpectationValueLessThanEquals", (
        f"Expected 'ExpectationValueLessThanEquals' but got: {expectation.get_expectation_name()}"
    )


@pytest.mark.parametrize(
    "data, threshold, expected_result, expected_violations, expected_message",
    [
        # Basic success scenarios
        ([1, 2, 3], 5, "success", None, None),
        ([1, 2, 3], 4, "success", None, None),
        ([5, 10, 15], 20, "success", None, None),
        # Success at threshold (key difference from less than)
        ([3, 4, 5], 5, "success", None, None),
        ([5, 5, 5], 5, "success", None, None),
        # Basic violation scenarios
        (
            [4, 5, 6],
            3,
            "failure",
            [4, 5, 6],
            "Found 3 row(s) where 'col1' is not less than or equal to 3.",
        ),
        (
            [3, 4, 5, 6],
            4,
            "failure",
            [5, 6],
            "Found 2 row(s) where 'col1' is not less than or equal to 4.",
        ),
        # Boundary conditions - at threshold (success for <=)
        ([2, 3, 4], 4, "success", None, None),
        ([1, 2, 3], 3, "success", None, None),
        # Boundary conditions - just above threshold (violation)
        (
            [3, 4, 5],
            4,
            "failure",
            [5],
            "Found 1 row(s) where 'col1' is not less than or equal to 4.",
        ),
        (
            [2, 2.1, 3],
            2,
            "failure",
            [2.1, 3],
            "Found 2 row(s) where 'col1' is not less than or equal to 2.",
        ),
        # Negative values - success
        ([-3, -2, -1], 0, "success", None, None),
        ([-5, -4, -3], -2, "success", None, None),
        ([-10, -5, 0], 5, "success", None, None),
        # Negative values - success at threshold
        ([-2, -1, 0], 0, "success", None, None),
        # Negative values - violations
        (
            [-1, 0, 1],
            -2,
            "failure",
            [-1, 0, 1],
            "Found 3 row(s) where 'col1' is not less than or equal to -2.",
        ),
        (
            [-3, -2, -1],
            -2,
            "failure",
            [-1],
            "Found 1 row(s) where 'col1' is not less than or equal to -2.",
        ),
        # Float values - success
        ([1.5, 2.0, 2.5], 3.0, "success", None, None),
        ([1.0, 2.0, 3.0], 3.0, "success", None, None),
        ([9.7, 9.8, 10.0], 10.0, "success", None, None),
        # Float values - violations
        (
            [2.0, 2.5, 3.0],
            2.4,
            "failure",
            [2.5, 3.0],
            "Found 2 row(s) where 'col1' is not less than or equal to 2.4.",
        ),
        (
            [1.5, 2.5, 3.5],
            2.9,
            "failure",
            [3.5],
            "Found 1 row(s) where 'col1' is not less than or equal to 2.9.",
        ),
        # Zero as threshold - success
        ([-2, -1, 0], 0, "success", None, None),
        ([-1.0, -0.5, 0.0], 0, "success", None, None),
        # Zero as threshold - violations
        (
            [0, 1, 2],
            -1,
            "failure",
            [0, 1, 2],
            "Found 3 row(s) where 'col1' is not less than or equal to -1.",
        ),
        (
            [-1, 0, 1],
            0,
            "failure",
            [1],
            "Found 1 row(s) where 'col1' is not less than or equal to 0.",
        ),
        # Zero in data - success (including zero at threshold)
        ([0, 1, 2], 2, "success", None, None),
        # Single value - success
        ([3], 5, "success", None, None),
        ([5], 5, "success", None, None),
        ([0], 10, "success", None, None),
        # Single value - violation
        (
            [5],
            4,
            "failure",
            [5],
            "Found 1 row(s) where 'col1' is not less than or equal to 4.",
        ),
        (
            [10],
            5,
            "failure",
            [10],
            "Found 1 row(s) where 'col1' is not less than or equal to 5.",
        ),
        # All values equal to threshold (success for <=)
        ([5, 5, 5, 5], 5, "success", None, None),
        # Mixed integers and floats
        ([1, 1.5, 2, 2.5], 3, "success", None, None),
        (
            [2, 2.5, 3, 3.5],
            2.9,
            "failure",
            [3, 3.5],
            "Found 2 row(s) where 'col1' is not less than or equal to 2.9.",
        ),
        # Large values
        ([1000, 2000, 3000], 3000, "success", None, None),
        (
            [1000, 1500, 2000],
            999,
            "failure",
            [1000, 1500, 2000],
            "Found 3 row(s) where 'col1' is not less than or equal to 999.",
        ),
        # All values above threshold
        (
            [6, 7, 8],
            5,
            "failure",
            [6, 7, 8],
            "Found 3 row(s) where 'col1' is not less than or equal to 5.",
        ),
        # With nulls - success (nulls are ignored)
        ([1, None, 2, None, 3], 3, "success", None, None),
        ([5, None, 10, None], 15, "success", None, None),
        # With nulls - violations
        (
            [2.0, None, 3.0, 4.0],
            3,
            "failure",
            [4.0],
            "Found 1 row(s) where 'col1' is not less than or equal to 3.",
        ),
    ],
    ids=[
        "basic_success",
        "success_different_data",
        "success_large_values",
        "success_at_threshold",
        "success_all_equal_threshold",
        "basic_violations",
        "partial_violations",
        "boundary_at_threshold_success",
        "boundary_at_threshold_success_2",
        "boundary_above_threshold",
        "boundary_above_threshold_float",
        "negative_success",
        "negative_range_success",
        "negative_large_success",
        "negative_at_threshold_success",
        "negative_violations",
        "negative_partial_violations",
        "float_success",
        "float_at_threshold_success",
        "float_precise_success",
        "float_violations",
        "float_partial_violations",
        "zero_threshold_success",
        "zero_threshold_float_success",
        "zero_threshold_violations",
        "zero_threshold_partial_violations",
        "zero_in_data_success",
        "single_value_success",
        "single_value_at_threshold_success",
        "single_value_large_success",
        "single_value_violation",
        "single_value_above_violation",
        "all_equal_threshold_success",
        "mixed_types_success",
        "mixed_types_violations",
        "large_values_success",
        "large_values_violations",
        "all_above_threshold",
        "with_nulls_success",
        "with_nulls_large_success",
        "with_nulls_violations",
    ],
)
def test_expectation_basic_scenarios(
    dataframe_factory, data, threshold, expected_result, expected_violations, expected_message
):
    """
    Test the expectation for various scenarios across DataFrame libraries.
    Tests both direct expectation validation and suite-based validation.
    Covers: success cases, boundary conditions (including equals), violations, negative values,
    floats, zero values, single values, mixed types, large values, and nulls.
    """
    df_lib, make_df = dataframe_factory

    # Determine data type based on whether we have float values (excluding None)
    has_float = any(isinstance(val, float) for val in data if val is not None)
    arrow_type = "double" if has_float else "long"

    data_frame = make_df({"col1": (data, arrow_type)})

    # Test 1: Direct expectation validation
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationValueLessThanEquals",
        column_name="col1",
        value=threshold,
    )

    result = expectation.validate(data_frame=data_frame)

    if expected_result == "success":
        assert str(result) == str(
            DataFrameExpectationSuccessMessage(expectation_name="ExpectationValueLessThanEquals")
        ), f"Expected success message but got: {result}"
    else:  # failure
        expected_violations_df = make_df({"col1": (expected_violations, arrow_type)})
        expected_failure_message = DataFrameExpectationFailureMessage(
            expectation_str=str(expectation),
            data_frame_type=df_lib,
            violations_data_frame=expected_violations_df,
            message=expected_message,
            limit_violations=5,
        )
        assert str(result) == str(expected_failure_message), (
            f"Expected failure message but got: {result}"
        )

    # Test 2: Suite-based validation
    expectations_suite = DataFrameExpectationsSuite().expect_value_less_than_equals(
        column_name="col1", value=threshold
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
    expected_message = "Column 'col1' does not exist in the DataFrame."

    data_frame = make_df({"col2": ([3, 4, 5], "long")})

    # Test 1: Direct expectation validation
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationValueLessThanEquals",
        column_name="col1",
        value=2,
    )
    result = expectation.validate(data_frame=data_frame)
    expected_failure = DataFrameExpectationFailureMessage(
        expectation_str=str(expectation),
        data_frame_type=df_lib,
        message=expected_message,
    )
    assert str(result) == str(expected_failure), f"Expected failure message but got: {result}"

    # Test 2: Suite-based validation
    expectations_suite = DataFrameExpectationsSuite().expect_value_less_than_equals(
        column_name="col1", value=2
    )
    with pytest.raises(DataFrameExpectationsSuiteFailure):
        expectations_suite.build().run(data_frame=data_frame)


def test_large_dataset_performance(dataframe_factory):
    """Test the expectation with a larger dataset to ensure performance."""
    df_lib, make_df = dataframe_factory

    # Create a larger dataset with values between 0 and 90
    large_data = np.random.uniform(0, 90, 10000).tolist()
    data_frame = make_df({"col1": (large_data, "double")})

    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationValueLessThanEquals",
        column_name="col1",
        value=100,
    )

    result = expectation.validate(data_frame=data_frame)
    # Should succeed as all values from uniform(0, 90) are <= 100
    assert isinstance(result, DataFrameExpectationSuccessMessage), (
        f"Large dataset test failed: expected success but got {type(result)}"
    )


def test_boundary_difference_from_less_than():
    """Test that <= behaves differently from < at boundary conditions."""
    data_frame = pd.DataFrame({"col1": [5, 5, 5]})

    # With LessThanEquals, all 5s should pass when threshold is 5
    expectation_lte = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationValueLessThanEquals",
        column_name="col1",
        value=5,
    )
    result_lte = expectation_lte.validate(data_frame=data_frame)
    assert isinstance(result_lte, DataFrameExpectationSuccessMessage), (
        f"LessThanEquals should pass when values equal threshold, got: {result_lte}"
    )

    # With LessThan, all 5s should fail when threshold is 5
    expectation_lt = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationValueLessThan",
        column_name="col1",
        value=5,
    )
    result_lt = expectation_lt.validate(data_frame=data_frame)
    assert isinstance(result_lt, DataFrameExpectationFailureMessage), (
        f"LessThan should fail when values equal threshold, got: {result_lt}"
    )
