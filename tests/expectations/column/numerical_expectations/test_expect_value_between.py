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
        expectation_name="ExpectationValueBetween",
        column_name="col1",
        min_value=2,
        max_value=5,
    )
    assert expectation.get_expectation_name() == "ExpectationValueBetween", (
        f"Expected 'ExpectationValueBetween' but got: {expectation.get_expectation_name()}"
    )


@pytest.mark.parametrize(
    "data, min_value, max_value, expected_result, expected_violations, expected_message",
    [
        # Basic success
        ([2, 3, 4, 5], 2, 5, "success", None, None),
        # Subset success
        ([3, 4], 2, 5, "success", None, None),
        # Identical values
        ([5, 5, 5, 5], 2, 5, "success", None, None),
        # Basic violations
        (
            [1, 2, 3, 6],
            2,
            5,
            "failure",
            [1, 6],
            "Found 2 row(s) where 'col1' is not between 2 and 5.",
        ),
        # All violations
        (
            [0, 1, 6, 7],
            2,
            5,
            "failure",
            [0, 1, 6, 7],
            "Found 4 row(s) where 'col1' is not between 2 and 5.",
        ),
        # Boundary exact min
        ([2, 2, 2], 2, 5, "success", None, None),
        # Boundary exact max
        ([5, 5, 5], 2, 5, "success", None, None),
        # Boundary min max
        ([2, 3, 4, 5], 2, 5, "success", None, None),
        # Boundary below min
        (
            [1, 2, 3],
            2,
            5,
            "failure",
            [1],
            "Found 1 row(s) where 'col1' is not between 2 and 5.",
        ),
        # Boundary above max
        (
            [3, 4, 6],
            2,
            5,
            "failure",
            [6],
            "Found 1 row(s) where 'col1' is not between 2 and 5.",
        ),
        # Negative success
        ([-5, -3, -2, 0], -5, 0, "success", None, None),
        # Negative range success
        ([-10, -8, -6], -10, -5, "success", None, None),
        # Negative violations
        (
            [-6, -3, 1],
            -5,
            0,
            "failure",
            [-6, 1],
            "Found 2 row(s) where 'col1' is not between -5 and 0.",
        ),
        # Float success
        ([2.5, 3.7, 4.2], 2.0, 5.0, "success", None, None),
        # Float mixed success
        ([2.0, 2.5, 3.0, 4.5, 5.0], 2.0, 5.0, "success", None, None),
        # Float violations
        (
            [1.5, 2.5, 5.5],
            2.0,
            5.0,
            "failure",
            [1.5, 5.5],
            "Found 2 row(s) where 'col1' is not between 2.0 and 5.0.",
        ),
        # Zero in range
        ([-2, -1, 0, 1, 2], -2, 2, "success", None, None),
        # All zeros
        ([0, 0, 0], 0, 1, "success", None, None),
        # Single value success
        ([3], 2, 5, "success", None, None),
        # Single value violation
        (
            [1],
            2,
            5,
            "failure",
            [1],
            "Found 1 row(s) where 'col1' is not between 2 and 5.",
        ),
        # Mixed integers and floats - success
        ([2, 2.5, 3, 4.5, 5], 2, 5, "success", None, None),
        # Mixed integers and floats - violations
        (
            [1.5, 2, 3, 5.5],
            2,
            5,
            "failure",
            [1.5, 5.5],
            "Found 2 row(s) where 'col1' is not between 2 and 5.",
        ),
        # Large range success
        ([100, 500, 900], 0, 1000, "success", None, None),
        # Large range violations
        (
            [-100, 500, 1100],
            0,
            1000,
            "failure",
            [-100, 1100],
            "Found 2 row(s) where 'col1' is not between 0 and 1000.",
        ),
        # All below range
        (
            [0, 1, 1],
            2,
            5,
            "failure",
            [0, 1, 1],
            "Found 3 row(s) where 'col1' is not between 2 and 5.",
        ),
        # All above range
        (
            [6, 7, 8],
            2,
            5,
            "failure",
            [6, 7, 8],
            "Found 3 row(s) where 'col1' is not between 2 and 5.",
        ),
        # With nulls success
        ([2, None, 3, None, 4], 2, 5, "success", None, None),
        # With nulls violations
        (
            [1, None, 3, 6],
            2,
            5,
            "failure",
            [1, 6],
            "Found 2 row(s) where 'col1' is not between 2 and 5.",
        ),
    ],
    ids=[
        "basic_success",
        "subset_success",
        "identical_values",
        "basic_violations",
        "all_violations",
        "boundary_exact_min",
        "boundary_exact_max",
        "boundary_min_max",
        "boundary_below_min",
        "boundary_above_max",
        "negative_success",
        "negative_range_success",
        "negative_violations",
        "float_success",
        "float_mixed_success",
        "float_violations",
        "zero_in_range",
        "all_zeros",
        "single_value_success",
        "single_value_violation",
        "mixed_types_success",
        "mixed_types_violations",
        "large_range_success",
        "large_range_violations",
        "all_below_range",
        "all_above_range",
        "with_nulls_success",
        "with_nulls_violations",
    ],
)
def test_expectation_basic_scenarios(
    dataframe_factory,
    data,
    min_value,
    max_value,
    expected_result,
    expected_violations,
    expected_message,
):
    """
    Test the expectation for various scenarios across pandas and PySpark DataFrames.
    Tests both direct expectation validation and suite-based validation.
    Covers: success cases, boundary conditions, violations, negative values, floats,
    zero values, single values, mixed types, large ranges, and nulls.
    """
    df_lib, make_df = dataframe_factory

    # Determine arrow type based on data
    has_float = any(isinstance(val, float) for val in data if val is not None)
    arrow_type = "double" if has_float else "long"

    data_frame = make_df({"col1": (data, arrow_type)})

    # Test 1: Direct expectation validation
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationValueBetween",
        column_name="col1",
        min_value=min_value,
        max_value=max_value,
    )

    result = expectation.validate(data_frame=data_frame)

    if expected_result == "success":
        assert str(result) == str(
            DataFrameExpectationSuccessMessage(expectation_name="ExpectationValueBetween")
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
    expectations_suite = DataFrameExpectationsSuite().expect_value_between(
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
    expected_message = "Column 'col1' does not exist in the DataFrame."

    data_frame = make_df({"col2": ([2, 3, 4], "long")})

    # Test 1: Direct expectation validation
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationValueBetween",
        column_name="col1",
        min_value=2,
        max_value=5,
    )
    result = expectation.validate(data_frame=data_frame)
    expected_failure = DataFrameExpectationFailureMessage(
        expectation_str=str(expectation),
        data_frame_type=df_lib,
        message=expected_message,
    )
    assert str(result) == str(expected_failure), f"Expected failure message but got: {result}"

    # Test 2: Suite-based validation
    expectations_suite = DataFrameExpectationsSuite().expect_value_between(
        column_name="col1", min_value=2, max_value=5
    )
    with pytest.raises(DataFrameExpectationsSuiteFailure):
        expectations_suite.build().run(data_frame=data_frame)


def test_large_dataset_performance():
    """Test the expectation with a larger dataset to ensure performance."""
    # Create a larger dataset with values between 10 and 100
    large_data = np.random.uniform(10, 100, 10000).tolist()
    data_frame = pd.DataFrame({"col1": large_data})

    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationValueBetween",
        column_name="col1",
        min_value=5,
        max_value=105,
    )

    result = expectation.validate(data_frame=data_frame)
    # Should succeed as all values from uniform(10, 100) are between 5 and 105
    assert isinstance(result, DataFrameExpectationSuccessMessage), (
        f"Large dataset test failed: expected success but got {type(result)}"
    )
