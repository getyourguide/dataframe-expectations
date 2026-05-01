import numpy as np
import pytest

from dataframe_expectations.core.types import DataFrameType
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
        expectation_name="ExpectationColumnQuantileBetween",
        column_name="test_col",
        quantile=0.5,
        min_value=20,
        max_value=30,
    )
    assert expectation.get_expectation_name() == "ExpectationColumnQuantileBetween", (
        f"Expected 'ExpectationColumnQuantileBetween' but got: {expectation.get_expectation_name()}"
    )


def test_expectation_description():
    """Test that description messages are correct for different quantiles."""
    test_cases = [
        (0.0, "minimum"),
        (0.25, "25th percentile"),
        (0.5, "median"),
        (0.75, "75th percentile"),
        (1.0, "maximum"),
        (0.9, "0.9 quantile"),
    ]

    for quantile, expected_desc in test_cases:
        exp = DataFrameExpectationRegistry.get_expectation(
            expectation_name="ExpectationColumnQuantileBetween",
            column_name="test_col",
            quantile=quantile,
            min_value=10,
            max_value=20,
        )
        assert exp.quantile_desc == expected_desc, (
            f"Expected quantile_desc '{expected_desc}' for quantile {quantile} but got: {exp.quantile_desc}"
        )
        assert expected_desc in exp.get_description(), (
            f"Expected '{expected_desc}' in description: {exp.get_description()}"
        )


@pytest.mark.parametrize(
    "data, quantile, min_value, max_value, should_succeed, expected_message",
    [
        # Quantile 0.0 (minimum) scenarios
        ([20, 25, 30, 35], 0.0, 15, 25, True, None),  # min = 20
        # Quantile 1.0 (maximum) scenarios
        ([20, 25, 30, 35], 1.0, 30, 40, True, None),  # max = 35
        # Quantile 0.5 (median) scenarios
        ([20, 25, 30, 35], 0.5, 25, 30, True, None),  # median = 27.5
        # Quantile 0.25 (25th percentile) scenarios
        ([20, 25, 30, 35], 0.25, 20, 25, True, None),  # 25th percentile = 22.5
        # Single row scenarios
        ([25], 0.5, 20, 30, True, None),  # median = 25
        # Null scenarios
        ([20, None, 25, None, 30], 0.5, 20, 30, True, None),  # median = 25
        # Failure scenarios - minimum (quantile 0.0)
        (
            [20, 25, 30, 35],
            0.0,
            25,
            35,
            False,
            "Column 'col1' minimum value 20.0 is not between 25 and 35.",
        ),
        # Failure scenarios - maximum (quantile 1.0)
        (
            [20, 25, 30, 35],
            1.0,
            40,
            50,
            False,
            "Column 'col1' maximum value 35.0 is not between 40 and 50.",
        ),
        # Failure scenarios - median (quantile 0.5)
        (
            [20, 25, 30, 35],
            0.5,
            30,
            35,
            False,
            "Column 'col1' median value 27.5 is not between 30 and 35.",
        ),
        # Failure scenarios - all nulls
        (
            [None, None, None],
            0.5,
            20,
            30,
            False,
            "Column 'col1' contains only null values.",
        ),
        # Failure scenarios - empty
        ([], 0.5, 20, 30, False, "Column 'col1' contains only null values."),
    ],
    ids=[
        "quantile_0_min",
        "quantile_1_max",
        "quantile_0_5_median",
        "quantile_0_25",
        "single_row",
        "with_nulls",
        "fail_min_too_low",
        "fail_max_too_low",
        "fail_median_too_low",
        "all_nulls",
        "empty",
    ],
)
def test_expectation_basic_scenarios(
    data, quantile, min_value, max_value, should_succeed, expected_message, dataframe_factory
):
    """Test basic expectation scenarios."""
    df_lib, make_df = dataframe_factory
    df = make_df({"col1": (data, "double")})

    # Test through registry
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationColumnQuantileBetween",
        column_name="col1",
        quantile=quantile,
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
    suite = DataFrameExpectationsSuite().expect_column_quantile_between(
        column_name="col1", quantile=quantile, min_value=min_value, max_value=max_value
    )

    if should_succeed:
        suite_result = suite.build().run(data_frame=df)
        assert suite_result is not None, "Expected SuiteExecutionResult"
        assert suite_result.success, "Expected all expectations to pass"
    else:
        with pytest.raises(DataFrameExpectationsSuiteFailure):
            suite.build().run(data_frame=df)


def test_75th_percentile_failure(dataframe_factory):
    """Test 75th percentile failure with library-specific expected values.

    Pandas and PySpark compute percentiles differently (linear interpolation vs approx),
    so expected values differ.
    """
    df_lib, make_df = dataframe_factory
    df = make_df({"col1": ([20, 25, 30, 35], "double")})

    match df_lib:
        case DataFrameType.PANDAS:
            # Pandas np.quantile uses linear interpolation: 31.25
            min_value, max_value = 25, 30
            expected_val = np.quantile([20, 25, 30, 35], 0.75)
            expected_message = (
                f"Column 'col1' 75th percentile value {expected_val} is not between 25 and 30."
            )
        case DataFrameType.PYSPARK | DataFrameType.POLARS:
            # PySpark percentile_approx returns 30
            min_value, max_value = 32, 40
            expected_message = "Column 'col1' 75th percentile value 30.0 is not between 32 and 40."

    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationColumnQuantileBetween",
        column_name="col1",
        quantile=0.75,
        min_value=min_value,
        max_value=max_value,
    )
    result = expectation.validate(data_frame=df)

    assert isinstance(result, DataFrameExpectationFailureMessage)
    assert expected_message in str(result), f"Expected '{expected_message}' in result: {result}"


def test_other_quantile_scenarios(dataframe_factory):
    """Test other quantile scenarios that differ between libraries."""
    df_lib, make_df = dataframe_factory

    df = make_df({"col1": ([10, 20, 30, 40, 50], "double")})

    match df_lib:
        case DataFrameType.PANDAS:
            # Pandas uses linear interpolation: quantile(0.33) ≈ 23.2
            min_value, max_value = 20, 30
        case DataFrameType.PYSPARK | DataFrameType.POLARS:
            # PySpark percentile_approx and Polars both return 20.0
            min_value, max_value = 15, 25

    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationColumnQuantileBetween",
        column_name="col1",
        quantile=0.33,
        min_value=min_value,
        max_value=max_value,
    )
    result = expectation.validate(data_frame=df)
    assert isinstance(result, DataFrameExpectationSuccessMessage), (
        f"Expected success but got: {result}"
    )


def test_column_missing_error(dataframe_factory):
    """Test missing column error."""
    df_lib, make_df = dataframe_factory
    df = make_df({"col1": ([20, 25, 30, 35], "double")})
    expected_message = "Column 'nonexistent_col' does not exist in the DataFrame."

    # Test through registry
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationColumnQuantileBetween",
        column_name="nonexistent_col",
        quantile=0.5,
        min_value=25,
        max_value=30,
    )
    result = expectation.validate(data_frame=df)
    expected_failure = DataFrameExpectationFailureMessage(
        expectation_str=str(expectation),
        data_frame_type=df_lib,
        message=expected_message,
    )
    assert str(result) == str(expected_failure)

    # Test through suite
    suite = DataFrameExpectationsSuite().expect_column_quantile_between(
        column_name="nonexistent_col", quantile=0.5, min_value=25, max_value=30
    )
    with pytest.raises(DataFrameExpectationsSuiteFailure):
        suite.build().run(data_frame=df)


def test_invalid_quantile_range():
    """Test that invalid quantile values raise ValueError."""
    invalid_quantiles = [
        (1.5, "greater than 1.0"),
        (-0.1, "less than 0.0"),
    ]

    for invalid_quantile, description in invalid_quantiles:
        with pytest.raises(ValueError) as context:
            DataFrameExpectationRegistry.get_expectation(
                expectation_name="ExpectationColumnQuantileBetween",
                column_name="col1",
                quantile=invalid_quantile,
                min_value=20,
                max_value=30,
            )
        assert "Quantile must be between 0.0 and 1.0" in str(context.value), (
            f"Expected quantile validation error for {description} but got: {str(context.value)}"
        )


def test_large_dataset_performance(dataframe_factory):
    """Test the expectation with a larger dataset to ensure performance."""
    df_lib, make_df = dataframe_factory

    # Create a larger dataset
    large_data = np.random.normal(50, 10, 1000).tolist()
    data_frame = make_df({"col1": (large_data, "double")})

    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationColumnQuantileBetween",
        column_name="col1",
        quantile=0.5,  # median
        min_value=45,
        max_value=55,
    )

    result = expectation.validate(data_frame=data_frame)
    # Should succeed as the median of normal(50, 10) should be around 50
    assert isinstance(result, DataFrameExpectationSuccessMessage), (
        f"Large dataset test failed: expected success but got {type(result)}"
    )
