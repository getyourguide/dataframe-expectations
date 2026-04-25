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
    "df_data, min_rows, expected_result, expected_message",
    [
        # Exact count - 3 rows == 3 min
        ({"col1": ([1, 2, 3], "long"), "col2": (["a", "b", "c"], "string")}, 3, "success", None),
        # Above min - 5 rows > 3 min
        (
            {"col1": ([1, 2, 3, 4, 5], "long"), "col2": (["a", "b", "c", "d", "e"], "string")},
            3,
            "success",
            None,
        ),
        # Single row - 1 row == 1 min
        ({"col1": ([42], "long")}, 1, "success", None),
        # Zero min empty - 0 rows == 0 min
        ({"col1": ([], "long")}, 0, "success", None),
        # Zero min with data - 3 rows >= 0 min
        ({"col1": ([1, 2, 3], "long")}, 0, "success", None),
        # With nulls - 5 rows >= 3 min (nulls don't affect row count)
        (
            {
                "col1": ([1, None, 3, None, 5], "long"),
                "col2": ([None, "b", None, "d", None], "string"),
            },
            3,
            "success",
            None,
        ),
        # Below min - 3 rows < 5 min
        (
            {"col1": ([1, 2, 3], "long"), "col2": (["a", "b", "c"], "string")},
            5,
            "failure",
            "DataFrame has 3 rows, expected at least 5.",
        ),
        # Empty needs min - 0 rows < 2 min
        ({"col1": ([], "long")}, 2, "failure", "DataFrame has 0 rows, expected at least 2."),
        # Single row needs more - 1 row < 3 min
        ({"col1": ([1], "long")}, 3, "failure", "DataFrame has 1 rows, expected at least 3."),
        # Large dataset success - 150 rows >= 100 min
        (
            {
                "col1": (list(range(150)), "long"),
                "col2": ([f"value_{i}" for i in range(150)], "string"),
            },
            100,
            "success",
            None,
        ),
        # Large dataset failure - 150 rows < 200 min
        (
            {
                "col1": (list(range(150)), "long"),
                "col2": ([f"value_{i}" for i in range(150)], "string"),
            },
            200,
            "failure",
            "DataFrame has 150 rows, expected at least 200.",
        ),
        # Multiple columns - 4 rows >= 3 min
        (
            {
                "col1": ([1, 2, 3, 4], "long"),
                "col2": (["a", "b", "c", "d"], "string"),
                "col3": ([1.1, 2.2, 3.3, 4.4], "double"),
                "col4": ([True, False, True, False], "boolean"),
            },
            3,
            "success",
            None,
        ),
        # Mixed data types - 5 rows >= 3 min
        (
            {
                "int_col": ([1, 2, 3, 4, 5], "long"),
                "str_col": (["a", "b", "c", "d", "e"], "string"),
                "float_col": ([1.1, 2.2, 3.3, 4.4, 5.5], "double"),
                "bool_col": ([True, False, True, False, True], "boolean"),
                "null_col": ([None, None, None, None, None], "long"),
            },
            3,
            "success",
            None,
        ),
        # Low min count - 3 rows >= 1 min
        ({"col1": ([1, 2, 3], "long")}, 1, "success", None),
        # High min count - 3 rows < 1000000 min
        (
            {"col1": ([1, 2, 3], "long")},
            1000000,
            "failure",
            "DataFrame has 3 rows, expected at least 1000000.",
        ),
        # Identical values - 4 rows >= 3 min
        (
            {
                "col1": ([42, 42, 42, 42], "long"),
                "col2": (["same", "same", "same", "same"], "string"),
            },
            3,
            "success",
            None,
        ),
        # Boundary condition - 1 row == 1 min (edge case equals actual)
        ({"col1": ([1], "long")}, 1, "success", None),
        # Progressive counts - 5 rows meets various minimums
        ({"col1": ([1, 2, 3, 4, 5], "long")}, 5, "success", None),
        ({"col1": ([1, 2, 3, 4, 5], "long")}, 4, "success", None),
        (
            {"col1": ([1, 2, 3, 4, 5], "long")},
            6,
            "failure",
            "DataFrame has 5 rows, expected at least 6.",
        ),
    ],
    ids=[
        "exact_count",
        "above_min",
        "single_row",
        "zero_min_empty",
        "zero_min_with_data",
        "with_nulls",
        "below_min",
        "empty_needs_min",
        "single_row_needs_more",
        "large_dataset",
        "large_dataset_failure",
        "multiple_columns",
        "mixed_data_types",
        "low_min_count",
        "high_min_count",
        "identical_values",
        "boundary_condition",
        "progressive_count_at_min",
        "progressive_count_below_min",
        "progressive_count_above_min",
    ],
)
def test_expectation_basic_scenarios(
    dataframe_factory, df_data, min_rows, expected_result, expected_message
):
    """
    Test the expectation for various scenarios across pandas and PySpark DataFrames.
    Tests both direct expectation validation and suite-based validation.
    Covers: success cases (exact, above min, zero min), failures (below min, empty), edge cases,
    boundary conditions, large datasets, nulls, multiple columns, mixed data types, identical values,
    progressive counts, and dataframe structure variations.
    """
    df_lib, make_df = dataframe_factory
    data_frame = make_df(df_data)

    # Test 1: Direct expectation validation
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationMinRows",
        min_rows=min_rows,
    )

    result = expectation.validate(data_frame=data_frame)

    if expected_result == "success":
        assert str(result) == str(
            DataFrameExpectationSuccessMessage(expectation_name="ExpectationMinRows")
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
    expectations_suite = DataFrameExpectationsSuite().expect_min_rows(min_rows=min_rows)

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
    # Test negative min_rows
    with pytest.raises(ValueError) as context:
        DataFrameExpectationRegistry.get_expectation(
            expectation_name="ExpectationMinRows",
            min_rows=-1,
        )
    assert "min_rows must be non-negative" in str(context.value), (
        f"Expected 'min_rows must be non-negative' in error message: {str(context.value)}"
    )


def test_large_dataset_performance(dataframe_factory):
    """Test the expectation with a larger dataset to ensure reasonable performance."""
    df_lib, make_df = dataframe_factory
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationMinRows",
        min_rows=500,
    )
    # Create a DataFrame with 1000 rows
    data_frame = make_df({"col1": (list(range(1000)), "long")})
    result = expectation.validate(data_frame=data_frame)
    assert str(result) == str(
        DataFrameExpectationSuccessMessage(expectation_name="ExpectationMinRows")
    ), f"Expected success message but got: {result}"
