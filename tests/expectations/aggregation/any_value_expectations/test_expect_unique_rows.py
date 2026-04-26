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
        expectation_name="ExpectationUniqueRows",
        column_names=["col1"],
    )
    assert expectation.get_expectation_name() == "ExpectationUniqueRows", (
        f"Expected 'ExpectationUniqueRows' but got: {expectation.get_expectation_name()}"
    )


@pytest.mark.parametrize(
    "df_data, column_names, expected_result, expected_violations, expected_message",
    [
        # Success - specific columns (unique combinations)
        (
            {
                "col1": ([1, 2, 3, 1], "long"),
                "col2": ([10, 20, 30, 20], "long"),
                "col3": ([100, 100, 100, 100], "long"),
            },
            ["col1", "col2"],
            "success",
            None,
            None,
        ),
        # Success - all columns (empty list)
        (
            {
                "col1": ([1, 2, 3], "long"),
                "col2": ([10, 20, 30], "long"),
                "col3": ([100, 200, 300], "long"),
            },
            [],
            "success",
            None,
            None,
        ),
        # Success - empty DataFrame
        (
            {"col1": ([], "long")},
            ["col1"],
            "success",
            None,
            None,
        ),
        # Success - single row
        (
            {"col1": ([1], "long")},
            ["col1"],
            "success",
            None,
            None,
        ),
        # Failure - specific columns with duplicates
        (
            {
                "col1": ([1, 2, 1, 3], "long"),
                "col2": ([10, 20, 10, 30], "long"),
                "col3": ([100, 200, 300, 400], "long"),
            },
            ["col1", "col2"],
            "failure",
            {"col1": ([1], "long"), "col2": ([10], "long"), "#duplicates": ([2], "long")},
            "Found 2 duplicate row(s). duplicate rows found for columns ['col1', 'col2']",
        ),
        # Failure - all columns with duplicates
        (
            {
                "col1": ([1, 2, 1], "long"),
                "col2": ([10, 20, 10], "long"),
                "col3": ([100, 200, 100], "long"),
            },
            [],
            "failure",
            {
                "col1": ([1], "long"),
                "col2": ([10], "long"),
                "col3": ([100], "long"),
                "#duplicates": ([2], "long"),
            },
            "Found 2 duplicate row(s). duplicate rows found",
        ),
        # Failure - multiple duplicate groups
        (
            {"col1": ([1, 2, 1, 3, 2, 3], "long"), "col2": ([10, 20, 30, 40, 50, 60], "long")},
            ["col1"],
            "failure",
            {"col1": ([1, 2, 3], "long"), "#duplicates": ([2, 2, 2], "long")},
            "Found 6 duplicate row(s). duplicate rows found for columns ['col1']",
        ),
        # Failure - with nulls (nulls counted as duplicates)
        # Use message-only assertion since handling of null violations differs between libraries
        (
            {
                "col1": ([1, None, 1, None], "long"),
                "col2": ([10, None, 20, None], "long"),
            },
            ["col1", "col2"],
            "failure",
            None,  # Message-only check
            "Found 2 duplicate row(s). duplicate rows found for columns ['col1', 'col2']",
        ),
    ],
    ids=[
        "success_specific_columns",
        "success_all_columns",
        "empty",
        "single_row",
        "violations_specific_columns",
        "violations_all_columns",
        "multiple_duplicate_groups",
        "with_nulls",
    ],
)
def test_expectation_basic_scenarios(
    dataframe_factory, df_data, column_names, expected_result, expected_violations, expected_message
):
    """
    Test the expectation for various scenarios across pandas and PySpark DataFrames.
    Tests both direct expectation validation and suite-based validation.
    Covers: success cases (specific columns, all columns, empty, single row),
    failure cases (duplicates on specific/all columns, multiple groups, with nulls).
    """
    df_lib, make_df = dataframe_factory
    data_frame = make_df(df_data)

    # Test 1: Direct expectation validation
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationUniqueRows",
        column_names=column_names,
    )

    result = expectation.validate(data_frame=data_frame)

    if expected_result == "success":
        assert str(result) == str(
            DataFrameExpectationSuccessMessage(expectation_name="ExpectationUniqueRows")
        ), f"Expected success message but got: {result}"
    else:  # failure
        if expected_violations is None:
            # Message-only check — used when violations contain all-null group keys
            # because handling differs between libraries.
            assert expected_message in str(result), f"Expected failure message but got: {result}"
        else:
            # Create violations DataFrame using the factory
            violations_df = make_df(expected_violations)

            expected_failure_message = DataFrameExpectationFailureMessage(
                expectation_str=str(expectation),
                data_frame_type=df_lib,
                violations_data_frame=violations_df,
                message=expected_message,
                limit_violations=5,
            )
            assert str(result) == str(expected_failure_message), (
                f"Expected failure message but got: {result}"
            )

    # Test 2: Suite-based validation
    expectations_suite = DataFrameExpectationsSuite().expect_unique_rows(column_names=column_names)

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
    """
    Test that an error is raised when specified columns are missing.
    Tests both direct expectation validation and suite-based validation.
    """
    df_lib, make_df = dataframe_factory
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationUniqueRows",
        column_names=["nonexistent_col"],
    )

    data_frame = make_df({"col1": ([1, 2, 3], "long")})

    # Test 1: Direct expectation validation
    result = expectation.validate(data_frame=data_frame)
    expected_failure_message = DataFrameExpectationFailureMessage(
        expectation_str=str(expectation),
        data_frame_type=df_lib,
        message="Column 'nonexistent_col' does not exist in the DataFrame.",
    )
    assert str(result) == str(expected_failure_message), (
        f"Expected failure message but got: {result}"
    )

    # Test 2: Suite-based validation
    expectations_suite = DataFrameExpectationsSuite().expect_unique_rows(
        column_names=["nonexistent_col"]
    )
    with pytest.raises(DataFrameExpectationsSuiteFailure):
        expectations_suite.build().run(data_frame=data_frame)


def test_large_dataset_performance(dataframe_factory):
    """
    Test the expectation with a larger dataset to ensure reasonable performance.
    """
    df_lib, make_df = dataframe_factory

    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationUniqueRows",
        column_names=["col1"],
    )
    # Create a DataFrame with unique values
    data_frame = make_df({"col1": (list(range(10000)), "long")})
    result = expectation.validate(data_frame=data_frame)
    assert isinstance(result, DataFrameExpectationSuccessMessage), (
        f"Expected DataFrameExpectationSuccessMessage but got: {type(result)}"
    )
