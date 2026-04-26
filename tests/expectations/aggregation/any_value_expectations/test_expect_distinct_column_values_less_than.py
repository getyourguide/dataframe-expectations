import pytest
from datetime import datetime

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
        expectation_name="ExpectationDistinctColumnValuesLessThan",
        column_name="col1",
        threshold=5,
    )
    assert expectation.get_expectation_name() == "ExpectationDistinctColumnValuesLessThan", (
        f"Expected 'ExpectationDistinctColumnValuesLessThan' but got: {expectation.get_expectation_name()}"
    )


@pytest.mark.parametrize(
    "df_data, threshold, expected_result, expected_message, arrow_type",
    [
        # Basic success - 3 distinct values < 5
        ([1, 2, 3, 2, 1], 5, "success", None, "long"),
        # Success with nulls - 3 distinct values [1, 2, None] < 5
        ([1, 2, None, 2, 1], 5, "success", None, "long"),
        # Empty DataFrame - 0 distinct values < 1
        ([], 1, "success", None, "long"),
        # Single value - 1 distinct value < 3
        ([5, 5, 5, 5, 5], 3, "success", None, "long"),
        # Equal to threshold (should fail) - 3 distinct values, NOT < 3
        (
            [1, 2, 3, 2, 1],
            3,
            "failure",
            "Column 'col1' has 3 distinct values, expected fewer than 3.",
            "long",
        ),
        # Above threshold (should fail) - 5 distinct values, NOT < 2
        (
            [1, 2, 3, 4, 5],
            2,
            "failure",
            "Column 'col1' has 5 distinct values, expected fewer than 2.",
            "long",
        ),
        # Zero threshold edge case - empty DataFrame, 0 distinct values NOT < 0
        ([], 0, "failure", "Column 'col1' has 0 distinct values, expected fewer than 0.", "long"),
        # Zero threshold edge case - non-empty DataFrame, 1 distinct value NOT < 0
        (
            [1, 1, 1],
            0,
            "failure",
            "Column 'col1' has 1 distinct values, expected fewer than 0.",
            "long",
        ),
        # Exclusive boundary test - 3 distinct values, NOT < 3
        (
            [1, 2, 3, 1, 2],
            3,
            "failure",
            "Column 'col1' has 3 distinct values, expected fewer than 3.",
            "long",
        ),
        # Exclusive boundary test - 3 distinct values < 4
        ([1, 2, 3, 1, 2], 4, "success", None, "long"),
        # String column with mixed values including None - 4 distinct values < 5
        (["A", "B", "C", "B", "A", None], 5, "success", None, "string"),
        # String case-sensitive - 4 distinct values ["a", "A", "b", "B"] < 5
        (["a", "A", "b", "B", "a", "A"], 5, "success", None, "string"),
        # Float column - 3 distinct values < 5
        ([1.1, 2.2, 3.3, 2.2, 1.1], 5, "success", None, "double"),
        # Boolean column - 2 distinct values [True, False] < 3
        ([True, False, True, False, True], 3, "success", None, "boolean"),
        # Boolean column failure - 2 distinct values [True, False], NOT < 2
        (
            [True, False, True, False, True],
            2,
            "failure",
            "Column 'col1' has 2 distinct values, expected fewer than 2.",
            "boolean",
        ),
        # Boolean single value - 1 distinct value [True] < 2
        ([True, True, True, True, True], 2, "success", None, "boolean"),
        # Datetime column - 3 distinct values < 5
        (
            [
                datetime(2023, 1, 1),
                datetime(2023, 1, 2),
                datetime(2023, 1, 3),
                datetime(2023, 1, 2),
                datetime(2023, 1, 1),
            ],
            5,
            "success",
            None,
            "timestamp",
        ),
        # Multiple NaN values counted as one - 3 distinct values [1, 2, None] < 5
        ([1, 2, None, None, None, 1, 2], 5, "success", None, "long"),
        # Strings with different whitespace - 4 distinct values < 5
        (["test", " test", "test ", " test ", "test"], 5, "success", None, "string"),
    ],
    ids=[
        "success",
        "success_with_nulls",
        "empty",
        "single_value",
        "equal_to_threshold",
        "above_threshold",
        "zero_threshold_empty",
        "zero_threshold_non_empty",
        "exclusive_boundary_fail",
        "exclusive_boundary_pass",
        "string_with_nulls",
        "string_case_sensitive",
        "float",
        "boolean",
        "boolean_failure",
        "boolean_single_value",
        "datetime",
        "duplicate_nan_handling",
        "string_whitespace",
    ],
)
def test_expectation_basic_scenarios(
    dataframe_factory, df_data, threshold, expected_result, expected_message, arrow_type
):
    """
    Test the expectation for various scenarios across pandas and PySpark DataFrames.
    Tests both direct expectation validation and suite-based validation.
    """
    df_lib, make_df = dataframe_factory
    data_frame = make_df({"col1": (df_data, arrow_type)})

    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationDistinctColumnValuesLessThan",
        column_name="col1",
        threshold=threshold,
    )

    result = expectation.validate(data_frame=data_frame)

    if expected_result == "success":
        assert str(result) == str(
            DataFrameExpectationSuccessMessage(
                expectation_name="ExpectationDistinctColumnValuesLessThan"
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

    expectations_suite = DataFrameExpectationsSuite().expect_distinct_column_values_less_than(
        column_name="col1", threshold=threshold
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
    "df_data, threshold, expected_result, expected_message, arrow_type",
    [
        # Mixed data types - 4 distinct values ["text", 42, 3.14, None] < 5 (pandas only)
        (["text", 42, 3.14, None, "text", 42], 5, "success", None, "string"),
        # Strings with numeric strings vs numeric values - 2 distinct values < 3 (pandas only - object dtype)
        (["1", 1, "1", 1], 3, "success", None, "string"),
    ],
    ids=[
        "mixed_data_types",
        "numeric_string_vs_numeric",
    ],
)
@pytest.mark.pandas
def test_expectation_pandas_only_scenarios(
    df_data, threshold, expected_result, expected_message, arrow_type
):
    """
    Test pandas-only scenarios that cannot be unified (mixed types, categorical, object dtype).
    """
    import pandas as pd

    data_frame = pd.DataFrame({"col1": df_data})

    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationDistinctColumnValuesLessThan",
        column_name="col1",
        threshold=threshold,
    )

    result = expectation.validate(data_frame=data_frame)

    assert str(result) == str(
        DataFrameExpectationSuccessMessage(
            expectation_name="ExpectationDistinctColumnValuesLessThan"
        )
    ), f"Expected success message but got: {result}"


@pytest.mark.pandas
def test_expectation_pandas_categorical():
    """Test categorical data - 3 distinct categories < 5 (pandas only)."""
    import pandas as pd

    data_frame = pd.DataFrame({"col1": pd.Categorical(["A", "B", "C", "A", "B", "C", "A"])})

    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationDistinctColumnValuesLessThan",
        column_name="col1",
        threshold=5,
    )

    result = expectation.validate(data_frame=data_frame)
    assert str(result) == str(
        DataFrameExpectationSuccessMessage(
            expectation_name="ExpectationDistinctColumnValuesLessThan"
        )
    ), f"Expected success message but got: {result}"


def test_column_missing_error(dataframe_factory):
    """
    Test that an error is raised when the specified column is missing.
    Tests both direct expectation validation and suite-based validation.
    """
    df_lib, make_df = dataframe_factory
    data_frame = make_df({"col2": ([1, 2, 3, 4, 5], "long")})

    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationDistinctColumnValuesLessThan",
        column_name="col1",
        threshold=5,
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

    expectations_suite = DataFrameExpectationsSuite().expect_distinct_column_values_less_than(
        column_name="col1", threshold=5
    )
    with pytest.raises(DataFrameExpectationsSuiteFailure):
        expectations_suite.build().run(data_frame=data_frame)


def test_invalid_parameters():
    """Test that appropriate errors are raised for invalid parameters."""
    with pytest.raises(ValueError) as context:
        DataFrameExpectationRegistry.get_expectation(
            expectation_name="ExpectationDistinctColumnValuesLessThan",
            column_name="col1",
            threshold=-1,
        )
    assert "threshold must be non-negative" in str(context.value)


def test_large_dataset_performance(dataframe_factory):
    """Test the expectation with a larger dataset to ensure reasonable performance."""
    df_lib, make_df = dataframe_factory

    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationDistinctColumnValuesLessThan",
        column_name="col1",
        threshold=1001,
    )
    data_frame = make_df({"col1": (list(range(1000)) * 5, "long")})
    result = expectation.validate(data_frame=data_frame)
    assert isinstance(result, DataFrameExpectationSuccessMessage), (
        f"Expected DataFrameExpectationSuccessMessage but got: {type(result)}"
    )
