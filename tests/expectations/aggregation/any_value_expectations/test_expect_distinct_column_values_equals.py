import pytest
import pandas as pd
from datetime import datetime, timezone

from dataframe_expectations.core.types import DataFrameType
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
    """
    Test that the expectation name is correctly returned.
    """
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationDistinctColumnValuesEquals",
        column_name="col1",
        expected_value=3,
    )
    assert expectation.get_expectation_name() == "ExpectationDistinctColumnValuesEquals", (
        f"Expected 'ExpectationDistinctColumnValuesEquals' but got: {expectation.get_expectation_name()}"
    )


def _assert_distinct_column_values_equals(
    data_frame, df_lib, expected_value, expected_result, expected_message
):
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationDistinctColumnValuesEquals",
        column_name="col1",
        expected_value=expected_value,
    )

    result = expectation.validate(data_frame=data_frame)

    if expected_result == "success":
        assert str(result) == str(
            DataFrameExpectationSuccessMessage(
                expectation_name="ExpectationDistinctColumnValuesEquals"
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

    expectations_suite = DataFrameExpectationsSuite().expect_distinct_column_values_equals(
        column_name="col1", expected_value=expected_value
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


@pytest.mark.parametrize(
    "df_data, expected_value, expected_result, expected_message, arrow_type",
    [
        ([1, 2, 3, 2, 1], 3, "success", None, "long"),
        ([1, 2, None, 2, 1], 3, "success", None, "long"),
        (
            [1, 2, 1, 2, 1],
            5,
            "failure",
            "Column 'col1' has 2 distinct values, expected exactly 5.",
            "long",
        ),
        (
            [1, 2, 3, 4, 5],
            2,
            "failure",
            "Column 'col1' has 5 distinct values, expected exactly 2.",
            "long",
        ),
        ([], 0, "success", None, "long"),
        ([5, 5, 5, 5, 5], 1, "success", None, "long"),
        (["A", "B", "C", "B", "A", None], 4, "success", None, "string"),
        (["a", "A", "b", "B", "a", "A"], 4, "success", None, "string"),
        ([1.1, 2.2, 3.3, 2.2, 1.1], 3, "success", None, "double"),
        ([1.0, 1.1, 1.2, 1.0, 1.1], 3, "success", None, "double"),
        ([True, False, True, False, True], 2, "success", None, "boolean"),
        ([True, False, None, False, True], 3, "success", None, "boolean"),
        (
            [
                datetime(2023, 1, 1),
                datetime(2023, 1, 2),
                datetime(2023, 1, 3),
                datetime(2023, 1, 2),
                datetime(2023, 1, 1),
            ],
            3,
            "success",
            None,
            "timestamp",
        ),
        (
            [
                datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
                datetime(2023, 1, 2, 12, 0, 0, tzinfo=timezone.utc),
                datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
                datetime(2023, 1, 2, 12, 0, 0, tzinfo=timezone.utc),
            ],
            2,
            "success",
            None,
            "timestamp_utc",
        ),
        ([1, 2, None, None, None, 1, 2], 3, "success", None, "long"),
        (["test", " test", "test ", " test ", "test"], 4, "success", None, "string"),
    ],
    ids=[
        "success",
        "success_with_nulls",
        "too_few",
        "too_many",
        "empty",
        "single_value",
        "string_with_nulls",
        "string_case_sensitive",
        "float",
        "numeric_precision",
        "boolean",
        "boolean_with_none",
        "datetime",
        "datetime_with_timezone",
        "duplicate_nan_handling",
        "string_whitespace",
    ],
)
def test_expectation_basic_scenarios(
    dataframe_factory, df_data, expected_value, expected_result, expected_message, arrow_type
):
    df_lib, make_df = dataframe_factory
    data_frame = make_df({"col1": (df_data, arrow_type)})
    _assert_distinct_column_values_equals(
        data_frame, df_lib, expected_value, expected_result, expected_message
    )


@pytest.mark.pandas
@pytest.mark.parametrize(
    "df_data, expected_value, expected_result, expected_message, data_type",
    [
        (["text", 42, 3.14, None, "text", 42], 4, "success", None, None),
        (pd.Categorical(["A", "B", "C", "A", "B", "C", "A"]), 3, "success", None, None),
        (["1", 1, "1", 1], 2, "success", None, "object"),
    ],
    ids=["mixed_data_types", "categorical", "numeric_string_vs_numeric"],
)
def test_expectation_basic_scenarios_pandas_only(
    df_data, expected_value, expected_result, expected_message, data_type
):
    if data_type == "object":
        data_frame = pd.DataFrame({"col1": df_data}, dtype=object)
    else:
        data_frame = pd.DataFrame(df_data, columns=["col1"])
    _assert_distinct_column_values_equals(
        data_frame, DataFrameType.PANDAS, expected_value, expected_result, expected_message
    )


def test_column_missing_error(dataframe_factory):
    df_lib, make_df = dataframe_factory
    data_frame = make_df({"col2": ([1, 2, 3, 4, 5], "long")})

    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationDistinctColumnValuesEquals",
        column_name="col1",
        expected_value=3,
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

    expectations_suite = DataFrameExpectationsSuite().expect_distinct_column_values_equals(
        column_name="col1", expected_value=3
    )
    with pytest.raises(DataFrameExpectationsSuiteFailure):
        expectations_suite.build().run(data_frame=data_frame)


@pytest.mark.parametrize(
    "expected_value, expected_error_message",
    [
        (-1, "expected_value must be non-negative"),
    ],
    ids=["negative_expected_value"],
)
def test_invalid_parameters(expected_value, expected_error_message):
    """
    Test that appropriate errors are raised for invalid parameters.
    """
    with pytest.raises(ValueError) as context:
        DataFrameExpectationRegistry.get_expectation(
            expectation_name="ExpectationDistinctColumnValuesEquals",
            column_name="col1",
            expected_value=expected_value,
        )
    assert expected_error_message in str(context.value), (
        f"Expected '{expected_error_message}' in error message: {str(context.value)}"
    )


def test_large_dataset_performance():
    """
    Test the expectation with a larger dataset to ensure reasonable performance.
    """
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationDistinctColumnValuesEquals",
        column_name="col1",
        expected_value=1000,
    )
    # Create a DataFrame with exactly 1000 distinct values
    data_frame = pd.DataFrame({"col1": list(range(1000)) * 5})  # 5000 rows, 1000 distinct values
    result = expectation.validate(data_frame=data_frame)
    assert isinstance(result, DataFrameExpectationSuccessMessage), (
        f"Expected DataFrameExpectationSuccessMessage but got: {type(result)}"
    )
