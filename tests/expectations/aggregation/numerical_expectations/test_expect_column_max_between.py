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
        expectation_name="ExpectationColumnMaxBetween",
        column_name="col1",
        min_value=10,
        max_value=20,
    )
    assert expectation.get_expectation_name() == "ExpectationColumnQuantileBetween", (
        f"Expected 'ExpectationColumnQuantileBetween' but got: {expectation.get_expectation_name()}"
    )


def test_expectation_description():
    """Test that the expectation description is correctly returned."""
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationColumnMaxBetween",
        column_name="test_col",
        min_value=10,
        max_value=20,
    )
    description = expectation.get_description()
    assert "maximum" in description, f"Expected 'maximum' in description: {description}"
    assert "test_col" in description, f"Expected 'test_col' in description: {description}"
    assert "10" in description, f"Expected '10' in description: {description}"
    assert "20" in description, f"Expected '20' in description: {description}"


@pytest.mark.parametrize(
    "data, arrow_type, min_value, max_value, expected_result, expected_message",
    [
        # Basic success
        ([20, 25, 30, 35], "long", 30, 40, "success", None),
        # Single row
        ([35], "long", 30, 40, "success", None),
        # Negative values
        ([-20, -15, -10, -3], "long", -5, 0, "success", None),
        # Float values
        ([1.1, 2.5, 3.7, 3.8], "double", 3.5, 4.0, "success", None),
        # Identical values
        ([25, 25, 25, 25], "long", 24, 26, "success", None),
        # Mixed types (as double)
        ([20.0, 25.5, 30.0, 37.0], "double", 35, 40, "success", None),
        # Zero values
        ([-5, 0, 0, -2], "long", -1, 1, "success", None),
        # With nulls (as long to preserve null semantics)
        ([20, None, 35, None, 25], "long", 30, 40, "success", None),
        # Boundary exact min
        ([20, 25, 30, 35], "long", 35, 40, "success", None),
        # Boundary exact max
        ([20, 25, 30, 35], "long", 30, 35, "success", None),
        # Max below range
        (
            [20, 25, 30, 35],
            "long",
            40,
            50,
            "failure",
            "Column 'col1' maximum value 35 is not between 40 and 50.",
        ),
        # All nulls
        (
            [None, None, None],
            "long",
            30,
            40,
            "failure",
            "Column 'col1' contains only null values.",
        ),
        # Empty
        ([], "long", 30, 40, "failure", "Column 'col1' contains only null values."),
    ],
    ids=[
        "basic_success",
        "single_row",
        "negative_values",
        "float_values",
        "identical_values",
        "mixed_types",
        "zero_values",
        "with_nulls",
        "boundary_exact_min",
        "boundary_exact_max",
        "max_below_range",
        "all_nulls",
        "empty",
    ],
)
def test_expectation_basic_scenarios(
    dataframe_factory, data, arrow_type, min_value, max_value, expected_result, expected_message
):
    """
    Test the expectation for various scenarios across pandas and PySpark DataFrames.
    Tests both direct expectation validation and suite-based validation.
    Covers: success cases, boundary conditions, failures (out of range, nulls, empty),
    and various data types (integers, floats, negatives, nulls, mixed types).
    """
    df_lib, make_df = dataframe_factory
    data_frame = make_df({"col1": (data, arrow_type)})

    # Test 1: Direct expectation validation
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationColumnMaxBetween",
        column_name="col1",
        min_value=min_value,
        max_value=max_value,
    )

    result = expectation.validate(data_frame=data_frame)

    if expected_result == "success":
        assert str(result) == str(
            DataFrameExpectationSuccessMessage(expectation_name="ExpectationColumnQuantileBetween")
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
    expectations_suite = DataFrameExpectationsSuite().expect_column_max_between(
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
        expectation_name="ExpectationColumnMaxBetween",
        column_name="nonexistent_col",
        min_value=30,
        max_value=40,
    )
    result = expectation.validate(data_frame=data_frame)
    expected_failure = DataFrameExpectationFailureMessage(
        expectation_str=str(expectation),
        data_frame_type=df_lib.value,
        message=expected_message,
    )
    assert str(result) == str(expected_failure), f"Expected failure message but got: {result}"

    # Test 2: Suite-based validation
    expectations_suite = DataFrameExpectationsSuite().expect_column_max_between(
        column_name="nonexistent_col", min_value=30, max_value=40
    )
    with pytest.raises(DataFrameExpectationsSuiteFailure):
        expectations_suite.build().run(data_frame=data_frame)


def test_large_dataset_performance(dataframe_factory):
    """Test the expectation with a larger dataset to ensure performance."""
    df_lib, make_df = dataframe_factory
    # Create a larger dataset with max around 60
    large_data = np.random.uniform(10, 60, 1000).tolist()
    data_frame = make_df({"col1": (large_data, "double")})

    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationColumnMaxBetween",
        column_name="col1",
        min_value=55,
        max_value=65,
    )

    result = expectation.validate(data_frame=data_frame)
    # Should succeed as the max of uniform(10, 60) should be around 60
    assert isinstance(result, DataFrameExpectationSuccessMessage), (
        f"Large dataset test failed: expected success but got {type(result)}"
    )
