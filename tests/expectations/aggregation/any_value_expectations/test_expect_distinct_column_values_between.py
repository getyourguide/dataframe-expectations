import pytest
import pandas as pd

from dataframe_expectations.registry import DataFrameExpectationRegistry
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
    """
    Test that the expectation name is correctly returned.
    """
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationDistinctColumnValuesBetween",
        column_name="col1",
        min_value=2,
        max_value=5,
    )
    assert expectation.get_expectation_name() == "ExpectationDistinctColumnValuesBetween", (
        f"Expected 'ExpectationDistinctColumnValuesBetween' but got: {expectation.get_expectation_name()}"
    )


@pytest.mark.parametrize(
    "df_data, min_value, max_value, expected_result, expected_message, arrow_type",
    [
        ([1, 2, 3, 2, 1], 2, 5, "success", None, "long"),
        ([1, 2, None, 2, 1], 3, 4, "success", None, "long"),
        ([1, 2, 3, 2, 1], 3, 5, "success", None, "long"),
        ([1, 2, 3, 4, 5, 1], 3, 5, "success", None, "long"),
        ([1, 2, 3, 2, 1], 3, 3, "success", None, "long"),
        (
            [1, 2, 1, 2, 1],
            3,
            3,
            "failure",
            "Column 'col1' has 2 distinct values, expected between 3 and 3.",
            "long",
        ),
        ([], 0, 5, "success", None, "long"),
        ([1, 1, 1, 1, 1], 1, 1, "success", None, "long"),
        (
            [1, 2, 1, 2, 1],
            4,
            6,
            "failure",
            "Column 'col1' has 2 distinct values, expected between 4 and 6.",
            "long",
        ),
        (
            [1, 2, 3, 4, 5],
            2,
            3,
            "failure",
            "Column 'col1' has 5 distinct values, expected between 2 and 3.",
            "long",
        ),
        (["A", "B", "C", "B", "A", None], 3, 5, "success", None, "string"),
        ([1.1, 2.2, 3.3, 2.2, 1.1], 2, 4, "success", None, "double"),
        ([True, False, True, False, True], 2, 2, "success", None, "boolean"),
        (
            pd.to_datetime(
                ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-02", "2023-01-01"]
            ).tolist(),
            2,
            4,
            "success",
            None,
            "timestamp",
        ),
        ([-10, -20, -30, -20, -10], 2, 4, "success", None, "long"),
        ([-1, 0, 1, 0, -1], 2, 4, "success", None, "long"),
        ([1000000, 2000000, 3000000, 2000000, 1000000], 2, 4, "success", None, "long"),
        ([None, None, None, None], 0, 1, "success", None, "long"),
    ],
    ids=[
        "success",
        "success_with_nulls",
        "exact_min_boundary",
        "exact_max_boundary",
        "zero_range_success",
        "zero_range_failure",
        "empty_dataframe",
        "single_value",
        "too_few",
        "too_many",
        "string_with_nulls",
        "float",
        "boolean",
        "datetime",
        "negative_integers",
        "mixed_positive_negative",
        "large_integers",
        "all_nulls",
    ],
)
def test_expectation_basic_scenarios(
    dataframe_factory, df_data, min_value, max_value, expected_result, expected_message, arrow_type
):
    df_lib, make_df = dataframe_factory
    data_frame = make_df({"col1": (df_data, arrow_type)})

    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationDistinctColumnValuesBetween",
        column_name="col1",
        min_value=min_value,
        max_value=max_value,
    )

    result = expectation.validate(data_frame=data_frame)

    if expected_result == "success":
        assert str(result) == str(
            DataFrameExpectationSuccessMessage(
                expectation_name="ExpectationDistinctColumnValuesBetween"
            )
        ), f"Expected success message but got: {result}"
    else:
        expected_failure_message = DataFrameExpectationFailureMessage(
            expectation_str=str(expectation),
            data_frame_type=df_lib,
            message=expected_message,
        )
        assert str(result) == str(expected_failure_message), (
            f"Expected failure message but got: {result}"
        )

    expectations_suite = DataFrameExpectationsSuite().expect_distinct_column_values_between(
        column_name="col1", min_value=min_value, max_value=max_value
    )

    if expected_result == "success":
        result = expectations_suite.build().run(data_frame=data_frame)
        assert result is not None
        assert isinstance(result, SuiteExecutionResult)
        assert result.success
        assert result.total_passed == 1
        assert result.total_failed == 0
    else:
        with pytest.raises(DataFrameExpectationsSuiteFailure):
            expectations_suite.build().run(data_frame=data_frame)


def test_column_missing_error(dataframe_factory):
    df_lib, make_df = dataframe_factory
    data_frame = make_df({"col2": ([1, 2, 3, 4, 5], "long")})

    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationDistinctColumnValuesBetween",
        column_name="col1",
        min_value=2,
        max_value=5,
    )

    result = expectation.validate(data_frame=data_frame)
    expected_failure_message = DataFrameExpectationFailureMessage(
        expectation_str=str(expectation),
        data_frame_type=df_lib,
        message="Column 'col1' does not exist in the DataFrame.",
    )
    assert str(result) == str(expected_failure_message), (
        f"Expected failure message but got: {result}"
    )

    expectations_suite = DataFrameExpectationsSuite().expect_distinct_column_values_between(
        column_name="col1", min_value=2, max_value=5
    )
    with pytest.raises(DataFrameExpectationsSuiteFailure):
        expectations_suite.build().run(data_frame=data_frame)


@pytest.mark.parametrize(
    "min_value, max_value, expected_error_message",
    [
        (-1, 5, "min_value must be non-negative"),
        (2, -1, "max_value must be non-negative"),
        (5, 2, "min_value (5) must be <= max_value (2)"),
    ],
    ids=["negative_min_value", "negative_max_value", "min_greater_than_max"],
)
def test_invalid_parameters(min_value, max_value, expected_error_message):
    """
    Test that appropriate errors are raised for invalid parameters.
    """
    with pytest.raises(ValueError) as context:
        DataFrameExpectationRegistry.get_expectation(
            expectation_name="ExpectationDistinctColumnValuesBetween",
            column_name="col1",
            min_value=min_value,
            max_value=max_value,
        )
    assert expected_error_message in str(context.value), (
        f"Expected '{expected_error_message}' in error message: {str(context.value)}"
    )


def test_large_dataset_performance(dataframe_factory):
    """
    Test the expectation with a larger dataset to ensure reasonable performance.
    """
    df_lib, make_df = dataframe_factory

    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationDistinctColumnValuesBetween",
        column_name="col1",
        min_value=900,
        max_value=1100,
    )
    # Create a DataFrame with exactly 1000 distinct values
    data_frame = make_df(
        {"col1": (list(range(1000)) * 5, "long")}
    )  # 5000 rows, 1000 distinct values
    result = expectation.validate(data_frame=data_frame)
    assert isinstance(result, DataFrameExpectationSuccessMessage), (
        f"Expected DataFrameExpectationSuccessMessage but got: {type(result)}"
    )
