import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timezone

from dataframe_expectations.registry import (
    DataFrameExpectationRegistry,
)
from dataframe_expectations.suite import (
    DataFrameExpectationsSuite,
    DataFrameExpectationsSuiteFailure,
)
from dataframe_expectations.core.suite_result import SuiteExecutionResult
from dataframe_expectations.core.types import DataFrameType
from dataframe_expectations.result_message import (
    DataFrameExpectationFailureMessage,
    DataFrameExpectationSuccessMessage,
)


def test_expectation_name():
    """Test that the expectation name is correctly returned."""
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationValueNotNull",
        column_name="col1",
    )
    assert expectation.get_expectation_name() == "ExpectationValueNotNull", (
        f"Expected 'ExpectationValueNotNull' but got: {expectation.get_expectation_name()}"
    )


def _assert_expectation(
    df,
    df_lib_value: DataFrameType,
    should_succeed: bool,
    expected_message: str | None = None,
    expected_violations_df=None,
):
    """Common assertion logic for ExpectationValueNotNull tests.

    Tests the expectation through both the registry and suite APIs.

    Args:
        df: DataFrame to test
        df_lib_value: The DataFrameType enum value
        should_succeed: Whether the expectation should pass
        expected_message: Expected failure message (for failures)
        expected_violations_df: Expected violations DataFrame (for failures)
    """
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationValueNotNull",
        column_name="col1",
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
        if expected_message:
            assert expected_message in str(result), (
                f"Expected message '{expected_message}' in result: {result}"
            )

        if expected_violations_df is not None:
            expected_failure = DataFrameExpectationFailureMessage(
                expectation_str=str(expectation),
                data_frame_type=df_lib_value,
                violations_data_frame=expected_violations_df,
                message=expected_message,
                limit_violations=5,
            )
            assert str(result) == str(expected_failure), (
                f"Expected failure details don't match. Got: {result}"
            )

    # Test through suite
    suite = DataFrameExpectationsSuite().expect_value_not_null(column_name="col1")
    if should_succeed:
        suite_result = suite.build().run(data_frame=df)
        assert suite_result is not None, "Expected SuiteExecutionResult"
        assert isinstance(suite_result, SuiteExecutionResult)
        assert suite_result.success, "Expected all expectations to pass"
        assert suite_result.total_passed == 1
        assert suite_result.total_failed == 0
    else:
        with pytest.raises(DataFrameExpectationsSuiteFailure):
            suite.build().run(data_frame=df)


@pytest.mark.parametrize(
    "data, should_succeed, expected_violations, expected_message, data_type",
    [
        # Basic integer scenarios - success (no nulls)
        ([1, 2, 3], True, None, None, "long"),
        ([10, 20, 30, 40], True, None, None, "long"),
        # Integer scenarios - violations (with None)
        (
            [1, None, None],
            False,
            [None, None],
            "Found 2 row(s) where 'col1' is null.",
            "long",
        ),
        # All nulls scenario
        (
            [None, None, None],
            False,
            [None, None, None],
            "Found 3 row(s) where 'col1' is null.",
            "long",
        ),
        # Single null scenario
        (
            [1, 2, None],
            False,
            [None],
            "Found 1 row(s) where 'col1' is null.",
            "long",
        ),
        # String data type scenarios - success (no nulls)
        (["apple", "banana", "cherry"], True, None, None, "string"),
        (["test", "data", "values"], True, None, None, "string"),
        # String scenarios - violations (with None)
        (
            ["apple", None, "banana"],
            False,
            [None],
            "Found 1 row(s) where 'col1' is null.",
            "string",
        ),
        # Empty string is NOT null - should succeed
        (["", "test", "data"], True, None, None, "string"),
        # Whitespace is NOT null - should succeed
        ([" ", "  ", "   "], True, None, None, "string"),
        # Mixed empty strings and nulls - only nulls are violations
        (
            ["", None, "test"],
            False,
            [None],
            "Found 1 row(s) where 'col1' is null.",
            "string",
        ),
        # Float/Double data type scenarios - success (no nulls)
        ([1.5, 2.5, 3.5], True, None, None, "double"),
        ([0.0, 1.1, 2.2], True, None, None, "double"),
        # Float scenarios - violations (with None)
        (
            [1.5, None, 2.5],
            False,
            [None],
            "Found 1 row(s) where 'col1' is null.",
            "double",
        ),
        # Zero is NOT null - should succeed
        ([0.0, 0.0, 0.0], True, None, None, "double"),
        ([0, 0, 0], True, None, None, "long"),
        # Negative numbers are NOT null - should succeed
        ([-1, -2, -3], True, None, None, "long"),
        ([-1.5, -2.5, -3.5], True, None, None, "double"),
        # Boolean data type scenarios - success (no nulls)
        ([True, False, True], True, None, None, "boolean"),
        ([False, False, False], True, None, None, "boolean"),
        # Boolean scenarios - violations (with None)
        (
            [True, None, False],
            False,
            [None],
            "Found 1 row(s) where 'col1' is null.",
            "boolean",
        ),
        # False is NOT null - should succeed
        ([False], True, None, None, "boolean"),
        # Timestamp/Datetime scenarios - success (no nulls)
        (
            [datetime(2023, 1, 1), datetime(2023, 1, 2), datetime(2023, 1, 3)],
            True,
            None,
            None,
            "timestamp",
        ),
        # Timestamp scenarios - violations (with None)
        (
            [datetime(2023, 1, 1), None, datetime(2023, 1, 3)],
            False,
            [None],
            "Found 1 row(s) where 'col1' is null.",
            "timestamp",
        ),
        # Datetime with timezone - success (no nulls)
        (
            [
                datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
                datetime(2023, 1, 2, 12, 0, 0, tzinfo=timezone.utc),
            ],
            True,
            None,
            None,
            "timestamp_utc",
        ),
        # Multiple nulls in different positions
        (
            [1, None, 3, None, 5],
            False,
            [None, None],
            "Found 2 row(s) where 'col1' is null.",
            "long",
        ),
        # Large numbers - success (not null)
        ([1000000, 2000000, 3000000], True, None, None, "long"),
        # Single value - success
        ([42], True, None, None, "long"),
        # Single null
        (
            [None],
            False,
            [None],
            "Found 1 row(s) where 'col1' is null.",
            "long",
        ),
    ],
    ids=[
        "int_basic_success",
        "int_multi_value_success",
        "int_with_none_violations",
        "all_nulls_violations",
        "single_null_violation",
        "string_basic_success",
        "string_multi_value_success",
        "string_with_none_violation",
        "empty_string_not_null_success",
        "whitespace_not_null_success",
        "mixed_empty_string_null_violation",
        "double_basic_success",
        "double_with_zero_success",
        "double_with_none_violation",
        "double_all_zeros_success",
        "int_all_zeros_success",
        "int_negative_values_success",
        "double_negative_values_success",
        "boolean_mixed_success",
        "boolean_all_false_success",
        "boolean_with_none_violation",
        "boolean_false_not_null_success",
        "timestamp_basic_success",
        "timestamp_with_none_violation",
        "timestamp_with_tz_success",
        "multiple_nulls_violations",
        "large_numbers_success",
        "single_value_success",
        "single_null_only_violation",
    ],
)
def test_expectation_basic_scenarios(
    dataframe_factory,
    data,
    should_succeed,
    expected_violations,
    expected_message,
    data_type,
):
    """Test basic expectation scenarios for pandas and PySpark DataFrames.

    Tests various data types including:
    - Integers (long): positive, negative, zero, large numbers, with/without nulls
    - Strings: with nulls, empty strings (NOT null), whitespace (NOT null)
    - Floats (double): with None, zero values (NOT null), negative values (NOT null)
    - Booleans: True/False (NOT null), with None
    - Timestamps: with and without timezone, with None
    - Edge cases: all nulls, single null, multiple nulls

    Note: ExpectationValueNotNull checks that values are NOT null.
    Success = no null values, Violations = null/None values.
    Empty strings, zeros, and False are NOT considered null.
    """
    df_lib, make_df = dataframe_factory
    df = make_df({"col1": (data, data_type)})

    expected_violations_df = (
        make_df({"col1": (expected_violations, data_type)})
        if expected_violations is not None
        else None
    )
    _assert_expectation(
        df,
        df_lib_value=df_lib,
        should_succeed=should_succeed,
        expected_message=expected_message,
        expected_violations_df=expected_violations_df,
    )


# Pandas-only tests for np.nan behavior (not supported in PySpark)
@pytest.mark.parametrize(
    "data, expected_violations, expected_message, data_type",
    [
        # Integer with np.nan (pandas-specific)
        (
            [1, None, np.nan],
            [None, np.nan],
            "Found 2 row(s) where 'col1' is null.",
            "long",
        ),
        # Double with np.nan (pandas-specific)
        (
            [1.5, np.nan, 2.5],
            [np.nan],
            "Found 1 row(s) where 'col1' is null.",
            "double",
        ),
    ],
    ids=[
        "pandas_int_with_none_nan_violations",
        "pandas_double_with_nan_violation",
    ],
)
def test_expectation_nan_scenarios_pandas(
    data,
    expected_violations,
    expected_message,
    data_type,
):
    """Test pandas-specific np.nan scenarios (not applicable to PySpark).

    Note: np.nan is a pandas-specific representation of "not a number" which
    pandas treats as null-like. PySpark doesn't support np.nan in the same way.
    """
    df = pd.DataFrame({"col1": data})
    expected_violations_df = pd.DataFrame({"col1": expected_violations})

    _assert_expectation(
        df,
        df_lib_value=DataFrameType.PANDAS,
        should_succeed=False,
        expected_message=expected_message,
        expected_violations_df=expected_violations_df,
    )


def test_column_missing_error(dataframe_factory):
    """Test that a missing column returns the appropriate error message."""
    df_lib, make_df = dataframe_factory
    df = make_df({"col2": ([1, 2, 3], "long")})

    # Test through registry - should return failure message about missing column
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationValueNotNull",
        column_name="col1",
    )
    result = expectation.validate(data_frame=df)

    assert isinstance(result, DataFrameExpectationFailureMessage), (
        f"Expected DataFrameExpectationFailureMessage but got: {type(result)}"
    )
    assert "Column 'col1' does not exist in the DataFrame" in str(result), (
        f"Expected missing column message but got: {result}"
    )

    # Test through suite - should raise DataFrameExpectationsSuiteFailure
    suite = DataFrameExpectationsSuite().expect_value_not_null(column_name="col1")
    with pytest.raises(DataFrameExpectationsSuiteFailure) as exc_info:
        suite.build().run(data_frame=df)

    assert "Column 'col1' does not exist in the DataFrame" in str(exc_info.value), (
        f"Expected missing column message in suite failure but got: {exc_info.value}"
    )


def test_large_dataset_performance(dataframe_factory):
    """Test the expectation with a larger dataset to ensure reasonable performance."""
    df_lib, make_df = dataframe_factory

    # Create a larger dataset with 10,000 rows, all non-null values
    large_data = list(range(1, 10001))
    df = make_df({"col1": (large_data, "long")})

    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationValueNotNull",
        column_name="col1",
    )

    result = expectation.validate(data_frame=df)
    assert isinstance(result, DataFrameExpectationSuccessMessage), (
        f"Large dataset test failed: expected success but got {type(result)}"
    )
