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
        expectation_name="ExpectationValueNull",
        column_name="col1",
    )
    assert expectation.get_expectation_name() == "ExpectationValueNull", (
        f"Expected 'ExpectationValueNull' but got: {expectation.get_expectation_name()}"
    )


def _assert_expectation(
    df,
    df_lib_value: DataFrameType,
    should_succeed: bool,
    expected_message: str | None = None,
    expected_violations_df=None,
):
    """Common assertion logic for ExpectationValueNull tests."""
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationValueNull",
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
    suite = DataFrameExpectationsSuite().expect_value_null(column_name="col1")
    if should_succeed:
        suite_result = suite.build().run(data_frame=df)
        assert suite_result is not None, "Expected SuiteExecutionResult"
        assert isinstance(suite_result, SuiteExecutionResult), (
            "Result should be SuiteExecutionResult"
        )
        assert suite_result.success, "Expected all expectations to pass"
        assert suite_result.total_passed == 1, "Expected 1 passed expectation"
        assert suite_result.total_failed == 0, "Expected 0 failed expectations"
    else:
        with pytest.raises(DataFrameExpectationsSuiteFailure):
            suite.build().run(data_frame=df)


@pytest.mark.parametrize(
    "data, should_succeed, expected_violations, expected_message, data_type",
    [
        # Basic integer scenarios - success (all nulls)
        ([None, None, None], True, None, None, "long"),
        # Integer scenarios - violations (with non-null values)
        (
            [None, 1, 2],
            False,
            [1, 2],
            "Found 2 row(s) where 'col1' is not null.",
            "long",
        ),
        # All non-null values - violations
        (
            [1, 2, 3],
            False,
            [1, 2, 3],
            "Found 3 row(s) where 'col1' is not null.",
            "long",
        ),
        # Single non-null value - violation
        (
            [None, None, 5],
            False,
            [5],
            "Found 1 row(s) where 'col1' is not null.",
            "long",
        ),
        # String data type scenarios - success (all nulls)
        ([None, None, None], True, None, None, "string"),
        # String scenarios - violations (with non-null values)
        (
            [None, "apple", "banana"],
            False,
            ["apple", "banana"],
            "Found 2 row(s) where 'col1' is not null.",
            "string",
        ),
        # Empty string is NOT null - violation
        (
            [None, "", None],
            False,
            [""],
            "Found 1 row(s) where 'col1' is not null.",
            "string",
        ),
        # Whitespace is NOT null - violation
        (
            [None, " ", None],
            False,
            [" "],
            "Found 1 row(s) where 'col1' is not null.",
            "string",
        ),
        # All empty strings - violations (empty strings are NOT null)
        (
            ["", "", ""],
            False,
            ["", "", ""],
            "Found 3 row(s) where 'col1' is not null.",
            "string",
        ),
        # Float/Double data type scenarios - success (all nulls)
        ([None, None, None], True, None, None, "double"),
        # Float scenarios - violations (with non-null values)
        (
            [None, 1.5, 2.5],
            False,
            [1.5, 2.5],
            "Found 2 row(s) where 'col1' is not null.",
            "double",
        ),
        # Zero is NOT null - violation
        (
            [None, 0.0, None],
            False,
            [0.0],
            "Found 1 row(s) where 'col1' is not null.",
            "double",
        ),
        (
            [None, 0, None],
            False,
            [0],
            "Found 1 row(s) where 'col1' is not null.",
            "long",
        ),
        # Negative numbers are NOT null - violations
        (
            [None, -5, None],
            False,
            [-5],
            "Found 1 row(s) where 'col1' is not null.",
            "long",
        ),
        (
            [None, -3.14, None],
            False,
            [-3.14],
            "Found 1 row(s) where 'col1' is not null.",
            "double",
        ),
        # Boolean data type scenarios - success (all nulls)
        ([None, None, None], True, None, None, "boolean"),
        # Boolean scenarios - violations (with True/False)
        (
            [None, True, False],
            False,
            [True, False],
            "Found 2 row(s) where 'col1' is not null.",
            "boolean",
        ),
        # False is NOT null - violation
        (
            [None, False, None],
            False,
            [False],
            "Found 1 row(s) where 'col1' is not null.",
            "boolean",
        ),
        # True is NOT null - violation
        (
            [None, True, None],
            False,
            [True],
            "Found 1 row(s) where 'col1' is not null.",
            "boolean",
        ),
        # Timestamp/Datetime scenarios - success (all nulls)
        ([None, None, None], True, None, None, "timestamp"),
        # Timestamp scenarios - violations (with datetime values)
        (
            [None, datetime(2023, 1, 1), datetime(2023, 1, 2)],
            False,
            [datetime(2023, 1, 1), datetime(2023, 1, 2)],
            "Found 2 row(s) where 'col1' is not null.",
            "timestamp",
        ),
        # Single datetime - violation
        (
            [None, datetime(2023, 1, 1), None],
            False,
            [datetime(2023, 1, 1)],
            "Found 1 row(s) where 'col1' is not null.",
            "timestamp",
        ),
        # Datetime with timezone - violation
        (
            [None, datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc), None],
            False,
            [datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)],
            "Found 1 row(s) where 'col1' is not null.",
            "timestamp_utc",
        ),
        # Multiple non-nulls in different positions
        (
            [1, None, 3, None, 5],
            False,
            [1, 3, 5],
            "Found 3 row(s) where 'col1' is not null.",
            "long",
        ),
        # Large numbers - violations (NOT null)
        (
            [None, 1000000, None],
            False,
            [1000000],
            "Found 1 row(s) where 'col1' is not null.",
            "long",
        ),
        # Single null - success
        ([None], True, None, None, "long"),
        # Single non-null - violation
        (
            [42],
            False,
            [42],
            "Found 1 row(s) where 'col1' is not null.",
            "long",
        ),
    ],
    ids=[
        "int_all_nulls_success",
        "int_with_non_null_violations",
        "int_all_non_null_violations",
        "int_single_non_null_violation",
        "string_all_nulls_success",
        "string_with_non_null_violations",
        "string_empty_string_violation",
        "string_whitespace_violation",
        "string_all_empty_strings_violations",
        "double_all_nulls_success",
        "double_with_non_null_violations",
        "double_zero_violation",
        "int_zero_violation",
        "int_negative_violation",
        "double_negative_violation",
        "boolean_all_nulls_success",
        "boolean_true_false_violations",
        "boolean_false_violation",
        "boolean_true_violation",
        "timestamp_all_nulls_success",
        "timestamp_with_values_violations",
        "timestamp_single_value_violation",
        "timestamp_with_tz_violation",
        "multiple_non_nulls_violations",
        "large_number_violation",
        "single_null_success",
        "single_non_null_violation",
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
    - Integers (long): all nulls (success), with non-null values (violations)
    - Strings: all nulls (success), empty strings (NOT null - violation), whitespace (NOT null - violation)
    - Floats (double): all nulls (success), zero values (NOT null - violation), negative values (NOT null - violation)
    - Booleans: all nulls (success), True/False (NOT null - violations)
    - Timestamps: all nulls (success), with datetime values (violations)
    - Edge cases: single null, single non-null, multiple non-nulls

    Note: ExpectationValueNull checks that ALL values ARE null.
    Success = all null values, Violations = any non-null values.
    Empty strings, zeros, and False are NOT null and will be violations.
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
    "data, expected_message, data_type",
    [
        # np.nan is treated as null in pandas
        ([None, np.nan, None], None, "long"),
        ([None, np.nan, None], None, "double"),
    ],
    ids=["int_with_nan_success", "double_with_nan_success"],
)
def test_expectation_nan_scenarios_pandas(data, expected_message, data_type):
    """Test pandas-specific np.nan scenarios (not applicable to PySpark).

    np.nan is treated as null in pandas, so columns with only None/np.nan should succeed.
    """
    df = pd.DataFrame({"col1": data})
    _assert_expectation(
        df,
        df_lib_value=DataFrameType.PANDAS,
        should_succeed=True,
        expected_message=expected_message,
    )


def test_column_missing_error(dataframe_factory):
    """Test that a missing column returns the appropriate error message."""
    df_lib, make_df = dataframe_factory
    df = make_df({"col2": ([None, None, None], "long")})

    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationValueNull",
        column_name="col1",
    )
    result = expectation.validate(data_frame=df)

    assert isinstance(result, DataFrameExpectationFailureMessage), (
        f"Expected DataFrameExpectationFailureMessage but got: {type(result)}"
    )
    assert "Column 'col1' does not exist in the DataFrame" in str(result), (
        f"Expected missing column message but got: {result}"
    )

    suite = DataFrameExpectationsSuite().expect_value_null(column_name="col1")
    with pytest.raises(DataFrameExpectationsSuiteFailure) as exc_info:
        suite.build().run(data_frame=df)

    assert "Column 'col1' does not exist in the DataFrame" in str(exc_info.value), (
        f"Expected missing column message in suite failure but got: {exc_info.value}"
    )


def test_large_dataset_performance(dataframe_factory):
    """Test the expectation with a larger dataset to ensure reasonable performance."""
    df_lib, make_df = dataframe_factory
    large_data = [None] * 10000
    df = make_df({"col1": (large_data, "long")})

    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationValueNull",
        column_name="col1",
    )

    result = expectation.validate(data_frame=df)
    assert isinstance(result, DataFrameExpectationSuccessMessage), (
        f"Large dataset test failed: expected success but got {type(result)}"
    )
