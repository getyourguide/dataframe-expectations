import pytest
import pandas as pd

from dataframe_expectations.core.types import DataFrameType
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
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationDistinctColumnValuesGreaterThan",
        column_name="col1",
        threshold=2,
    )
    assert expectation.get_expectation_name() == "ExpectationDistinctColumnValuesGreaterThan"


@pytest.mark.parametrize(
    "df_data, threshold, expected_result, expected_message, arrow_type",
    [
        ([1, 2, 3, 2, 1], 2, "success", None, "long"),
        ([1, 2, None, 2, 1], 2, "success", None, "long"),
        (
            [1, 2, 3, 2, 1],
            3,
            "failure",
            "Column 'col1' has 3 distinct values, expected more than 3.",
            "long",
        ),
        (
            [1, 2, 1, 2, 1],
            5,
            "failure",
            "Column 'col1' has 2 distinct values, expected more than 5.",
            "long",
        ),
        ([1, 1, 1], 0, "success", None, "long"),
        ([], 0, "failure", "Column 'col1' has 0 distinct values, expected more than 0.", "long"),
        (
            [1, 2, 3, 4, 5, 1, 2],
            5,
            "failure",
            "Column 'col1' has 5 distinct values, expected more than 5.",
            "long",
        ),
        ([1, 2, 3, 4, 5, 1, 2], 4, "success", None, "long"),
        (["A", "B", "C", "B", "A", None], 3, "success", None, "string"),
        (["a", "A", "b", "B", "a", "A"], 3, "success", None, "string"),
        ([1.1, 2.2, 3.3, 2.2, 1.1], 2, "success", None, "double"),
        ([True, False, True, False, True], 1, "success", None, "boolean"),
        (
            [True, True, True, True, True],
            2,
            "failure",
            "Column 'col1' has 1 distinct values, expected more than 2.",
            "boolean",
        ),
        ([1, 2, None, None, None, 1, 2], 2, "success", None, "long"),
        (["test", " test", "test ", " test ", "test"], 3, "success", None, "string"),
    ],
    ids=[
        "success",
        "success_with_nulls",
        "equal_to_threshold",
        "below_threshold",
        "zero_threshold",
        "empty",
        "exclusive_boundary_fail",
        "exclusive_boundary_pass",
        "string_with_nulls",
        "string_case_sensitive",
        "float",
        "boolean",
        "boolean_failure",
        "duplicate_nan_handling",
        "string_whitespace",
    ],
)
def test_expectation_basic_scenarios(
    dataframe_factory, df_data, threshold, expected_result, expected_message, arrow_type
):
    df_lib, make_df = dataframe_factory
    data_frame = make_df({"col1": (df_data, arrow_type)})

    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationDistinctColumnValuesGreaterThan",
        column_name="col1",
        threshold=threshold,
    )
    result = expectation.validate(data_frame=data_frame)

    if expected_result == "success":
        assert str(result) == str(
            DataFrameExpectationSuccessMessage(
                expectation_name="ExpectationDistinctColumnValuesGreaterThan"
            )
        )
    else:
        expected_failure_message = DataFrameExpectationFailureMessage(
            expectation_str=str(expectation),
            data_frame_type=df_lib,
            message=expected_message,
        )
        assert str(result) == str(expected_failure_message)

    expectations_suite = DataFrameExpectationsSuite().expect_distinct_column_values_greater_than(
        column_name="col1", threshold=threshold
    )
    if expected_result == "success":
        result = expectations_suite.build().run(data_frame=data_frame)
        assert isinstance(result, SuiteExecutionResult)
        assert result.success
        assert result.total_passed == 1
        assert result.total_failed == 0
    else:
        with pytest.raises(DataFrameExpectationsSuiteFailure):
            expectations_suite.build().run(data_frame=data_frame)


@pytest.mark.parametrize(
    "df_data, threshold, expected_result, expected_message",
    [
        (
            ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-02", "2023-01-01"],
            2,
            "success",
            None,
        ),
    ],
    ids=["datetime"],
)
def test_expectation_datetime_scenario(
    dataframe_factory, df_data, threshold, expected_result, expected_message
):
    df_lib, make_df = dataframe_factory
    match df_lib:
        case DataFrameType.PANDAS:
            import pandas as pd

            data_frame = make_df({"col1": (pd.to_datetime(df_data), "timestamp")})
        case DataFrameType.PYSPARK:
            from datetime import datetime

            parsed = [datetime.fromisoformat(d) for d in df_data]
            data_frame = make_df({"col1": (parsed, "timestamp")})

    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationDistinctColumnValuesGreaterThan",
        column_name="col1",
        threshold=threshold,
    )
    result = expectation.validate(data_frame=data_frame)
    assert str(result) == str(
        DataFrameExpectationSuccessMessage(
            expectation_name="ExpectationDistinctColumnValuesGreaterThan"
        )
    )


@pytest.mark.pandas
@pytest.mark.parametrize(
    "df_data, threshold",
    [
        (pd.Categorical(["A", "B", "C", "A", "B", "C", "A"]), 2),
        (["text", 42, 3.14, None, "text", 42], 3),
    ],
    ids=["categorical", "mixed_data_types"],
)
def test_expectation_pandas_only_scenarios(df_data, threshold):
    """Tests for pandas-specific data types that cannot be represented in PySpark."""
    data_frame = pd.DataFrame({"col1": df_data})

    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationDistinctColumnValuesGreaterThan",
        column_name="col1",
        threshold=threshold,
    )
    result = expectation.validate(data_frame=data_frame)
    assert str(result) == str(
        DataFrameExpectationSuccessMessage(
            expectation_name="ExpectationDistinctColumnValuesGreaterThan"
        )
    )


@pytest.mark.pandas
def test_expectation_pandas_object_dtype():
    """Numeric strings vs numeric values - pandas object dtype only."""
    data_frame = pd.DataFrame({"col1": ["1", 1, "1", 1]}, dtype=object)
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationDistinctColumnValuesGreaterThan",
        column_name="col1",
        threshold=1,
    )
    result = expectation.validate(data_frame=data_frame)
    assert str(result) == str(
        DataFrameExpectationSuccessMessage(
            expectation_name="ExpectationDistinctColumnValuesGreaterThan"
        )
    )


def test_column_missing_error(dataframe_factory):
    df_lib, make_df = dataframe_factory
    data_frame = make_df({"col2": ([1, 2, 3, 4, 5], "long")})

    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationDistinctColumnValuesGreaterThan",
        column_name="col1",
        threshold=2,
    )
    result = expectation.validate(data_frame=data_frame)
    expected_failure_message = DataFrameExpectationFailureMessage(
        expectation_str=str(expectation),
        data_frame_type=df_lib,
        message="Column 'col1' does not exist in the DataFrame.",
    )
    assert str(result) == str(expected_failure_message)

    expectations_suite = DataFrameExpectationsSuite().expect_distinct_column_values_greater_than(
        column_name="col1", threshold=2
    )
    with pytest.raises(DataFrameExpectationsSuiteFailure):
        expectations_suite.build().run(data_frame=data_frame)


def test_invalid_parameters():
    with pytest.raises(ValueError) as context:
        DataFrameExpectationRegistry.get_expectation(
            expectation_name="ExpectationDistinctColumnValuesGreaterThan",
            column_name="col1",
            threshold=-1,
        )
    assert "threshold must be non-negative" in str(context.value)


def test_large_dataset_performance():
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationDistinctColumnValuesGreaterThan",
        column_name="col1",
        threshold=999,
    )
    data_frame = pd.DataFrame({"col1": list(range(1000)) * 5})
    result = expectation.validate(data_frame=data_frame)
    assert isinstance(result, DataFrameExpectationSuccessMessage)
