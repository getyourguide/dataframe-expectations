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


@pytest.mark.parametrize(
    "df_data, max_rows, expected_result, expected_message",
    [
        # Exact count - 3 rows == 3 max
        ({"col1": ([1, 2, 3], "long"), "col2": (["a", "b", "c"], "string")}, 3, "success", None),
        # Below max - 5 rows < 10 max
        (
            {"col1": ([1, 2, 3, 4, 5], "long"), "col2": (["a", "b", "c", "d", "e"], "string")},
            10,
            "success",
            None,
        ),
        # Single row - 1 row == 1 max
        ({"col1": ([42], "long")}, 1, "success", None),
        # Empty DataFrame - 0 rows <= 5 max
        ({"col1": ([], "long")}, 5, "success", None),
        # Exceeds max - 5 rows > 3 max
        (
            {"col1": ([1, 2, 3, 4, 5], "long"), "col2": (["a", "b", "c", "d", "e"], "string")},
            3,
            "failure",
            "DataFrame has 5 rows, expected at most 3.",
        ),
        # Zero max with data - 1 row > 0 max
        ({"col1": ([1], "long")}, 0, "failure", "DataFrame has 1 rows, expected at most 0."),
        # Zero max empty - 0 rows == 0 max
        ({"col1": ([], "long")}, 0, "success", None),
        # Large dataset - 150 rows > 100 max
        (
            {
                "col1": (list(range(150)), "long"),
                "col2": ([f"value_{i}" for i in range(150)], "string"),
            },
            100,
            "failure",
            "DataFrame has 150 rows, expected at most 100.",
        ),
        # With nulls - 5 rows > 4 max (nulls don't affect row count)
        (
            {
                "col1": ([1, None, 3, None, 5], "long"),
                "col2": ([None, "b", None, "d", None], "string"),
            },
            4,
            "failure",
            "DataFrame has 5 rows, expected at most 4.",
        ),
        # Multiple columns - 4 rows > 3 max
        (
            {
                "col1": ([1, 2, 3, 4], "long"),
                "col2": (["a", "b", "c", "d"], "string"),
                "col3": ([1.1, 2.2, 3.3, 4.4], "double"),
                "col4": ([True, False, True, False], "boolean"),
            },
            3,
            "failure",
            "DataFrame has 4 rows, expected at most 3.",
        ),
        # Mixed data types - 5 rows <= 10 max
        (
            {
                "int_col": ([1, 2, 3, 4, 5], "long"),
                "str_col": (["a", "b", "c", "d", "e"], "string"),
                "float_col": ([1.1, 2.2, 3.3, 4.4, 5.5], "double"),
                "bool_col": ([True, False, True, False, True], "boolean"),
                "null_col": ([None, None, None, None, None], "long"),
            },
            10,
            "success",
            None,
        ),
        # High max rows - 3 rows << 1000000 max
        ({"col1": ([1, 2, 3], "long")}, 1000000, "success", None),
        # Boundary condition 1 - 1 row == 1 max
        ({"col1": ([1], "long")}, 1, "success", None),
        # Boundary condition 2 - 2 rows > 1 max
        ({"col1": ([1, 2], "long")}, 1, "failure", "DataFrame has 2 rows, expected at most 1."),
        # Identical values - 4 rows > 3 max
        (
            {
                "col1": ([42, 42, 42, 42], "long"),
                "col2": (["same", "same", "same", "same"], "string"),
            },
            3,
            "failure",
            "DataFrame has 4 rows, expected at most 3.",
        ),
    ],
    ids=[
        "exact_count",
        "below_max",
        "single_row",
        "empty",
        "exceeds_max",
        "zero_max_with_data",
        "zero_max_empty",
        "large_dataset",
        "with_nulls",
        "multiple_columns",
        "mixed_data_types",
        "high_max_rows",
        "boundary_condition_1",
        "boundary_condition_2",
        "identical_values",
    ],
)
def test_expectation_basic_scenarios(
    dataframe_factory, df_data, max_rows, expected_result, expected_message
):
    """
    Test the expectation for various scenarios across pandas and PySpark DataFrames.
    Tests both direct expectation validation and suite-based validation.
    Covers: success cases, failures (exceeds max), edge cases (empty, zero max, single row),
    boundary conditions, large datasets, nulls, multiple columns, mixed data types, and identical values.
    """
    df_lib, make_df = dataframe_factory
    data_frame = make_df(df_data)

    # Test 1: Direct expectation validation
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationMaxRows",
        max_rows=max_rows,
    )

    result = expectation.validate(data_frame=data_frame)

    if expected_result == "success":
        assert str(result) == str(
            DataFrameExpectationSuccessMessage(expectation_name="ExpectationMaxRows")
        ), f"Expected success message but got: {result}"
    else:  # failure
        expected_failure_message = DataFrameExpectationFailureMessage(
            expectation_str=str(expectation),
            data_frame_type=df_lib,
            message=expected_message,
        )
        assert str(result) == str(expected_failure_message), (
            f"Expected failure message but got: {result}"
        )

    # Test 2: Suite-based validation
    expectations_suite = DataFrameExpectationsSuite().expect_max_rows(max_rows=max_rows)

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


def test_invalid_parameters():
    """Test that appropriate errors are raised for invalid parameters."""
    # Test negative max_rows
    with pytest.raises(ValueError) as context:
        DataFrameExpectationRegistry.get_expectation(
            expectation_name="ExpectationMaxRows",
            max_rows=-1,
        )
    assert "max_rows must be non-negative" in str(context.value), (
        f"Expected 'max_rows must be non-negative' in error message: {str(context.value)}"
    )


def test_large_dataset_performance(dataframe_factory):
    """Test the expectation with a larger dataset to ensure reasonable performance."""
    df_lib, make_df = dataframe_factory
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationMaxRows",
        max_rows=1500,
    )
    # Create a DataFrame with 1000 rows
    data_frame = make_df({"col1": (list(range(1000)), "long")})
    result = expectation.validate(data_frame=data_frame)
    assert str(result) == str(
        DataFrameExpectationSuccessMessage(expectation_name="ExpectationMaxRows")
    ), f"Expected success message but got: {result}"
