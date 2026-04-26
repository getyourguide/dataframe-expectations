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
        expectation_name="ExpectationValueGreaterThan",
        column_name="col1",
        value=2,
    )
    assert expectation.get_expectation_name() == "ExpectationValueGreaterThan", (
        f"Expected 'ExpectationValueGreaterThan' but got: {expectation.get_expectation_name()}"
    )


@pytest.mark.parametrize(
    "data, threshold, expected_result, expected_violations, expected_message",
    [
        # Basic success scenarios
        ([3, 4, 5], 2, "success", None, None),
        ([4, 5, 6], 3, "success", None, None),
        ([10, 20, 30], 5, "success", None, None),
        # Basic violation scenarios
        (
            [3, 4, 5],
            3,
            "failure",
            [3],
            "Found 1 row(s) where 'col1' is not greater than 3.",
        ),
        (
            [1, 2, 3],
            3,
            "failure",
            [1, 2, 3],
            "Found 3 row(s) where 'col1' is not greater than 3.",
        ),
        (
            [2, 3, 4, 5],
            4,
            "failure",
            [2, 3, 4],
            "Found 3 row(s) where 'col1' is not greater than 4.",
        ),
        # Boundary conditions - just above threshold
        ([3, 4, 5], 2, "success", None, None),
        ([2.1, 3, 4], 2, "success", None, None),
        # Boundary conditions - at threshold (violation)
        (
            [2, 3, 4],
            2,
            "failure",
            [2],
            "Found 1 row(s) where 'col1' is not greater than 2.",
        ),
        (
            [5, 5, 5],
            5,
            "failure",
            [5, 5, 5],
            "Found 3 row(s) where 'col1' is not greater than 5.",
        ),
        # Boundary conditions - just below threshold
        (
            [1, 2, 3],
            2,
            "failure",
            [1, 2],
            "Found 2 row(s) where 'col1' is not greater than 2.",
        ),
        # Negative values - success
        ([0, 1, 2], -1, "success", None, None),
        ([-2, -1, 0], -3, "success", None, None),
        ([-5, -3, 0], -10, "success", None, None),
        # Negative values - violations
        (
            [-3, -2, -1],
            -2,
            "failure",
            [-3, -2],
            "Found 2 row(s) where 'col1' is not greater than -2.",
        ),
        (
            [-5, -4, -3],
            -3,
            "failure",
            [-5, -4, -3],
            "Found 3 row(s) where 'col1' is not greater than -3.",
        ),
        # Float values - success
        ([2.5, 3.7, 4.2], 2.0, "success", None, None),
        ([3.1, 4.5, 5.9], 3.0, "success", None, None),
        ([10.1, 10.2, 10.3], 10.0, "success", None, None),
        # Float values - violations
        (
            [2.0, 2.5, 3.0],
            2.5,
            "failure",
            [2.0, 2.5],
            "Found 2 row(s) where 'col1' is not greater than 2.5.",
        ),
        (
            [1.5, 2.5, 3.5],
            3.0,
            "failure",
            [1.5, 2.5],
            "Found 2 row(s) where 'col1' is not greater than 3.0.",
        ),
        # Zero as threshold - success
        ([1, 2, 3], 0, "success", None, None),
        ([0.1, 0.5, 1.0], 0, "success", None, None),
        # Zero as threshold - violations
        (
            [-2, -1, 0],
            0,
            "failure",
            [-2, -1, 0],
            "Found 3 row(s) where 'col1' is not greater than 0.",
        ),
        (
            [-1, 0, 1],
            0,
            "failure",
            [-1, 0],
            "Found 2 row(s) where 'col1' is not greater than 0.",
        ),
        # Zero in data - success
        ([1, 2, 3], -1, "success", None, None),
        # Zero in data - violations
        (
            [0, 1, 2],
            0,
            "failure",
            [0],
            "Found 1 row(s) where 'col1' is not greater than 0.",
        ),
        # Single value - success
        ([5], 4, "success", None, None),
        ([10], 0, "success", None, None),
        # Single value - violation
        (
            [3],
            3,
            "failure",
            [3],
            "Found 1 row(s) where 'col1' is not greater than 3.",
        ),
        (
            [2],
            5,
            "failure",
            [2],
            "Found 1 row(s) where 'col1' is not greater than 5.",
        ),
        # All values equal to threshold
        (
            [5, 5, 5, 5],
            5,
            "failure",
            [5, 5, 5, 5],
            "Found 4 row(s) where 'col1' is not greater than 5.",
        ),
        # Mixed integers and floats
        ([3, 3.5, 4, 4.5], 2, "success", None, None),
        (
            [2, 2.5, 3, 3.5],
            3,
            "failure",
            [2, 2.5, 3],
            "Found 3 row(s) where 'col1' is not greater than 3.",
        ),
        # Large values
        ([1000, 2000, 3000], 999, "success", None, None),
        (
            [1000, 1500, 2000],
            2000,
            "failure",
            [1000, 1500, 2000],
            "Found 3 row(s) where 'col1' is not greater than 2000.",
        ),
        # All values below threshold
        (
            [1, 2, 3],
            5,
            "failure",
            [1, 2, 3],
            "Found 3 row(s) where 'col1' is not greater than 5.",
        ),
        # With nulls - success (nulls are ignored)
        ([3, None, 4, None, 5], 2, "success", None, None),
        ([10, None, 20, None], 5, "success", None, None),
        # With nulls - violations
        (
            [2.0, None, 3.0, 4.0],
            3,
            "failure",
            [2.0, 3.0],
            "Found 2 row(s) where 'col1' is not greater than 3.",
        ),
    ],
    ids=[
        "basic_success",
        "success_different_data",
        "success_large_values",
        "basic_violations",
        "all_violations",
        "partial_violations",
        "boundary_just_above",
        "boundary_just_above_float",
        "boundary_at_threshold",
        "boundary_all_at_threshold",
        "boundary_below_threshold",
        "negative_success",
        "negative_range_success",
        "negative_large_success",
        "negative_violations",
        "negative_all_violations",
        "float_success",
        "float_different_success",
        "float_precise_success",
        "float_violations",
        "float_mixed_violations",
        "zero_threshold_success",
        "zero_threshold_float_success",
        "zero_threshold_violations",
        "zero_threshold_mixed_violations",
        "zero_in_data_success",
        "zero_in_data_violation",
        "single_value_success",
        "single_value_large_success",
        "single_value_violation",
        "single_value_below_violation",
        "all_equal_threshold",
        "mixed_types_success",
        "mixed_types_violations",
        "large_values_success",
        "large_values_violations",
        "all_below_threshold",
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
    Covers: success cases, boundary conditions, violations, negative values, floats,
    zero values, single values, mixed types, large values, and nulls.
    """
    df_lib, make_df = dataframe_factory

    # Determine data type based on whether we have float values (excluding None)
    has_float = any(isinstance(val, float) for val in data if val is not None)
    arrow_type = "double" if has_float else "long"

    data_frame = make_df({"col1": (data, arrow_type)})

    # Test 1: Direct expectation validation
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationValueGreaterThan",
        column_name="col1",
        value=threshold,
    )

    result = expectation.validate(data_frame=data_frame)

    if expected_result == "success":
        assert str(result) == str(
            DataFrameExpectationSuccessMessage(expectation_name="ExpectationValueGreaterThan")
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
    expectations_suite = DataFrameExpectationsSuite().expect_value_greater_than(
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
        expectation_name="ExpectationValueGreaterThan",
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
    expectations_suite = DataFrameExpectationsSuite().expect_value_greater_than(
        column_name="col1", value=2
    )
    with pytest.raises(DataFrameExpectationsSuiteFailure):
        expectations_suite.build().run(data_frame=data_frame)


def test_large_dataset_performance():
    """Test the expectation with a larger dataset to ensure performance."""
    # Create a larger dataset with values between 10 and 100
    large_data = np.random.uniform(10, 100, 10000).tolist()
    data_frame = pd.DataFrame({"col1": large_data})

    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationValueGreaterThan",
        column_name="col1",
        value=5,
    )

    result = expectation.validate(data_frame=data_frame)
    # Should succeed as all values from uniform(10, 100) are > 5
    assert isinstance(result, DataFrameExpectationSuccessMessage), (
        f"Large dataset test failed: expected success but got {type(result)}"
    )
