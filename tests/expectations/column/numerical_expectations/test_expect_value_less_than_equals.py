import pytest
import pandas as pd

from dataframe_expectations.registry import (
    DataFrameExpectationRegistry,
)
from dataframe_expectations.suite import (
    DataFrameExpectationsSuite,
    DataFrameExpectationsSuiteFailure,
)
from dataframe_expectations.result_message import (
    DataFrameExpectationFailureMessage,
    DataFrameExpectationSuccessMessage,
)
from dataframe_expectations.core.suite_result import SuiteExecutionResult


def create_dataframe(df_type, data, column_name, spark, data_type="long"):
    """Helper function to create pandas or pyspark DataFrame.

    Args:
        df_type: "pandas" or "pyspark"
        data: List of values for the column
        column_name: Name of the column
        spark: Spark session (required for pyspark)
        data_type: Data type for the column - "long", "double"
    """
    if df_type == "pandas":
        return pd.DataFrame({column_name: data})
    else:  # pyspark
        from pyspark.sql.types import (
            StructType,
            StructField,
            LongType,
            DoubleType,
        )

        type_mapping = {
            "long": LongType(),
            "double": DoubleType(),
        }

        # Convert integers to floats when using double type to avoid type mismatch
        if data_type == "double":
            data = [float(val) if val is not None else None for val in data]

        schema = StructType([StructField(column_name, type_mapping[data_type], True)])
        return spark.createDataFrame([(val,) for val in data], schema)


def test_expectation_name():
    """Test that the expectation name is correctly returned."""
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationValueLessThanEquals",
        column_name="col1",
        value=2,
    )
    assert expectation.get_expectation_name() == "ExpectationValueLessThanEquals", (
        f"Expected 'ExpectationValueLessThanEquals' but got: {expectation.get_expectation_name()}"
    )


@pytest.mark.parametrize(
    "df_type, data, threshold, expected_result, expected_violations, expected_message",
    [
        # Basic success scenarios
        ("pandas", [1, 2, 3], 5, "success", None, None),
        ("pyspark", [1, 2, 3], 5, "success", None, None),
        ("pandas", [1, 2, 3], 4, "success", None, None),
        ("pyspark", [1, 2, 3], 4, "success", None, None),
        ("pandas", [5, 10, 15], 20, "success", None, None),
        ("pyspark", [5, 10, 15], 20, "success", None, None),
        # Success at threshold (key difference from less than)
        ("pandas", [3, 4, 5], 5, "success", None, None),
        ("pyspark", [3, 4, 5], 5, "success", None, None),
        ("pandas", [5, 5, 5], 5, "success", None, None),
        ("pyspark", [5, 5, 5], 5, "success", None, None),
        # Basic violation scenarios
        (
            "pandas",
            [4, 5, 6],
            3,
            "failure",
            [4, 5, 6],
            "Found 3 row(s) where 'col1' is not less than or equal to 3.",
        ),
        (
            "pyspark",
            [4, 5, 6],
            3,
            "failure",
            [4, 5, 6],
            "Found 3 row(s) where 'col1' is not less than or equal to 3.",
        ),
        (
            "pandas",
            [3, 4, 5, 6],
            4,
            "failure",
            [5, 6],
            "Found 2 row(s) where 'col1' is not less than or equal to 4.",
        ),
        (
            "pyspark",
            [3, 4, 5, 6],
            4,
            "failure",
            [5, 6],
            "Found 2 row(s) where 'col1' is not less than or equal to 4.",
        ),
        # Boundary conditions - at threshold (success for <=)
        ("pandas", [2, 3, 4], 4, "success", None, None),
        ("pyspark", [2, 3, 4], 4, "success", None, None),
        ("pandas", [1, 2, 3], 3, "success", None, None),
        ("pyspark", [1, 2, 3], 3, "success", None, None),
        # Boundary conditions - just above threshold (violation)
        (
            "pandas",
            [3, 4, 5],
            4,
            "failure",
            [5],
            "Found 1 row(s) where 'col1' is not less than or equal to 4.",
        ),
        (
            "pyspark",
            [3, 4, 5],
            4,
            "failure",
            [5],
            "Found 1 row(s) where 'col1' is not less than or equal to 4.",
        ),
        (
            "pandas",
            [2, 2.1, 3],
            2,
            "failure",
            [2.1, 3],
            "Found 2 row(s) where 'col1' is not less than or equal to 2.",
        ),
        (
            "pyspark",
            [2, 2.1, 3],
            2,
            "failure",
            [2.1, 3],
            "Found 2 row(s) where 'col1' is not less than or equal to 2.",
        ),
        # Negative values - success
        ("pandas", [-3, -2, -1], 0, "success", None, None),
        ("pyspark", [-3, -2, -1], 0, "success", None, None),
        ("pandas", [-5, -4, -3], -2, "success", None, None),
        ("pyspark", [-5, -4, -3], -2, "success", None, None),
        ("pandas", [-10, -5, 0], 5, "success", None, None),
        ("pyspark", [-10, -5, 0], 5, "success", None, None),
        # Negative values - success at threshold
        ("pandas", [-2, -1, 0], 0, "success", None, None),
        ("pyspark", [-2, -1, 0], 0, "success", None, None),
        # Negative values - violations
        (
            "pandas",
            [-1, 0, 1],
            -2,
            "failure",
            [-1, 0, 1],
            "Found 3 row(s) where 'col1' is not less than or equal to -2.",
        ),
        (
            "pyspark",
            [-1, 0, 1],
            -2,
            "failure",
            [-1, 0, 1],
            "Found 3 row(s) where 'col1' is not less than or equal to -2.",
        ),
        (
            "pandas",
            [-3, -2, -1],
            -2,
            "failure",
            [-1],
            "Found 1 row(s) where 'col1' is not less than or equal to -2.",
        ),
        (
            "pyspark",
            [-3, -2, -1],
            -2,
            "failure",
            [-1],
            "Found 1 row(s) where 'col1' is not less than or equal to -2.",
        ),
        # Float values - success
        ("pandas", [1.5, 2.0, 2.5], 3.0, "success", None, None),
        ("pyspark", [1.5, 2.0, 2.5], 3.0, "success", None, None),
        ("pandas", [1.0, 2.0, 3.0], 3.0, "success", None, None),
        ("pyspark", [1.0, 2.0, 3.0], 3.0, "success", None, None),
        ("pandas", [9.7, 9.8, 10.0], 10.0, "success", None, None),
        ("pyspark", [9.7, 9.8, 10.0], 10.0, "success", None, None),
        # Float values - violations
        (
            "pandas",
            [2.0, 2.5, 3.0],
            2.4,
            "failure",
            [2.5, 3.0],
            "Found 2 row(s) where 'col1' is not less than or equal to 2.4.",
        ),
        (
            "pyspark",
            [2.0, 2.5, 3.0],
            2.4,
            "failure",
            [2.5, 3.0],
            "Found 2 row(s) where 'col1' is not less than or equal to 2.4.",
        ),
        (
            "pandas",
            [1.5, 2.5, 3.5],
            2.9,
            "failure",
            [3.5],
            "Found 1 row(s) where 'col1' is not less than or equal to 2.9.",
        ),
        (
            "pyspark",
            [1.5, 2.5, 3.5],
            2.9,
            "failure",
            [3.5],
            "Found 1 row(s) where 'col1' is not less than or equal to 2.9.",
        ),
        # Zero as threshold - success
        ("pandas", [-2, -1, 0], 0, "success", None, None),
        ("pyspark", [-2, -1, 0], 0, "success", None, None),
        ("pandas", [-1.0, -0.5, 0.0], 0, "success", None, None),
        ("pyspark", [-1.0, -0.5, 0.0], 0, "success", None, None),
        # Zero as threshold - violations
        (
            "pandas",
            [0, 1, 2],
            -1,
            "failure",
            [0, 1, 2],
            "Found 3 row(s) where 'col1' is not less than or equal to -1.",
        ),
        (
            "pyspark",
            [0, 1, 2],
            -1,
            "failure",
            [0, 1, 2],
            "Found 3 row(s) where 'col1' is not less than or equal to -1.",
        ),
        (
            "pandas",
            [-1, 0, 1],
            0,
            "failure",
            [1],
            "Found 1 row(s) where 'col1' is not less than or equal to 0.",
        ),
        (
            "pyspark",
            [-1, 0, 1],
            0,
            "failure",
            [1],
            "Found 1 row(s) where 'col1' is not less than or equal to 0.",
        ),
        # Zero in data - success (including zero at threshold)
        ("pandas", [0, 1, 2], 2, "success", None, None),
        ("pyspark", [0, 1, 2], 2, "success", None, None),
        # Single value - success
        ("pandas", [3], 5, "success", None, None),
        ("pyspark", [3], 5, "success", None, None),
        ("pandas", [5], 5, "success", None, None),
        ("pyspark", [5], 5, "success", None, None),
        ("pandas", [0], 10, "success", None, None),
        ("pyspark", [0], 10, "success", None, None),
        # Single value - violation
        (
            "pandas",
            [5],
            4,
            "failure",
            [5],
            "Found 1 row(s) where 'col1' is not less than or equal to 4.",
        ),
        (
            "pyspark",
            [5],
            4,
            "failure",
            [5],
            "Found 1 row(s) where 'col1' is not less than or equal to 4.",
        ),
        (
            "pandas",
            [10],
            5,
            "failure",
            [10],
            "Found 1 row(s) where 'col1' is not less than or equal to 5.",
        ),
        (
            "pyspark",
            [10],
            5,
            "failure",
            [10],
            "Found 1 row(s) where 'col1' is not less than or equal to 5.",
        ),
        # All values equal to threshold (success for <=)
        ("pandas", [5, 5, 5, 5], 5, "success", None, None),
        ("pyspark", [5, 5, 5, 5], 5, "success", None, None),
        # Mixed integers and floats
        ("pandas", [1, 1.5, 2, 2.5], 3, "success", None, None),
        ("pyspark", [1, 1.5, 2, 2.5], 3, "success", None, None),
        (
            "pandas",
            [2, 2.5, 3, 3.5],
            2.9,
            "failure",
            [3, 3.5],
            "Found 2 row(s) where 'col1' is not less than or equal to 2.9.",
        ),
        (
            "pyspark",
            [2, 2.5, 3, 3.5],
            2.9,
            "failure",
            [3, 3.5],
            "Found 2 row(s) where 'col1' is not less than or equal to 2.9.",
        ),
        # Large values
        ("pandas", [1000, 2000, 3000], 3000, "success", None, None),
        ("pyspark", [1000, 2000, 3000], 3000, "success", None, None),
        (
            "pandas",
            [1000, 1500, 2000],
            999,
            "failure",
            [1000, 1500, 2000],
            "Found 3 row(s) where 'col1' is not less than or equal to 999.",
        ),
        (
            "pyspark",
            [1000, 1500, 2000],
            999,
            "failure",
            [1000, 1500, 2000],
            "Found 3 row(s) where 'col1' is not less than or equal to 999.",
        ),
        # All values above threshold
        (
            "pandas",
            [6, 7, 8],
            5,
            "failure",
            [6, 7, 8],
            "Found 3 row(s) where 'col1' is not less than or equal to 5.",
        ),
        (
            "pyspark",
            [6, 7, 8],
            5,
            "failure",
            [6, 7, 8],
            "Found 3 row(s) where 'col1' is not less than or equal to 5.",
        ),
        # With nulls - success (nulls are ignored)
        ("pandas", [1, None, 2, None, 3], 3, "success", None, None),
        ("pyspark", [1, None, 2, None, 3], 3, "success", None, None),
        ("pandas", [5, None, 10, None], 15, "success", None, None),
        ("pyspark", [5, None, 10, None], 15, "success", None, None),
        # With nulls - violations
        (
            "pandas",
            [2.0, None, 3.0, 4.0],
            3,
            "failure",
            [4.0],
            "Found 1 row(s) where 'col1' is not less than or equal to 3.",
        ),
        (
            "pyspark",
            [2, None, 3, 4],
            3,
            "failure",
            [4],
            "Found 1 row(s) where 'col1' is not less than or equal to 3.",
        ),
    ],
    ids=[
        "pandas_basic_success",
        "pyspark_basic_success",
        "pandas_success_different_data",
        "pyspark_success_different_data",
        "pandas_success_large_values",
        "pyspark_success_large_values",
        "pandas_success_at_threshold",
        "pyspark_success_at_threshold",
        "pandas_success_all_equal_threshold",
        "pyspark_success_all_equal_threshold",
        "pandas_basic_violations",
        "pyspark_basic_violations",
        "pandas_partial_violations",
        "pyspark_partial_violations",
        "pandas_boundary_at_threshold_success",
        "pyspark_boundary_at_threshold_success",
        "pandas_boundary_at_threshold_success_2",
        "pyspark_boundary_at_threshold_success_2",
        "pandas_boundary_above_threshold",
        "pyspark_boundary_above_threshold",
        "pandas_boundary_above_threshold_float",
        "pyspark_boundary_above_threshold_float",
        "pandas_negative_success",
        "pyspark_negative_success",
        "pandas_negative_range_success",
        "pyspark_negative_range_success",
        "pandas_negative_large_success",
        "pyspark_negative_large_success",
        "pandas_negative_at_threshold_success",
        "pyspark_negative_at_threshold_success",
        "pandas_negative_violations",
        "pyspark_negative_violations",
        "pandas_negative_partial_violations",
        "pyspark_negative_partial_violations",
        "pandas_float_success",
        "pyspark_float_success",
        "pandas_float_at_threshold_success",
        "pyspark_float_at_threshold_success",
        "pandas_float_precise_success",
        "pyspark_float_precise_success",
        "pandas_float_violations",
        "pyspark_float_violations",
        "pandas_float_partial_violations",
        "pyspark_float_partial_violations",
        "pandas_zero_threshold_success",
        "pyspark_zero_threshold_success",
        "pandas_zero_threshold_float_success",
        "pyspark_zero_threshold_float_success",
        "pandas_zero_threshold_violations",
        "pyspark_zero_threshold_violations",
        "pandas_zero_threshold_partial_violations",
        "pyspark_zero_threshold_partial_violations",
        "pandas_zero_in_data_success",
        "pyspark_zero_in_data_success",
        "pandas_single_value_success",
        "pyspark_single_value_success",
        "pandas_single_value_at_threshold_success",
        "pyspark_single_value_at_threshold_success",
        "pandas_single_value_large_success",
        "pyspark_single_value_large_success",
        "pandas_single_value_violation",
        "pyspark_single_value_violation",
        "pandas_single_value_above_violation",
        "pyspark_single_value_above_violation",
        "pandas_all_equal_threshold_success",
        "pyspark_all_equal_threshold_success",
        "pandas_mixed_types_success",
        "pyspark_mixed_types_success",
        "pandas_mixed_types_violations",
        "pyspark_mixed_types_violations",
        "pandas_large_values_success",
        "pyspark_large_values_success",
        "pandas_large_values_violations",
        "pyspark_large_values_violations",
        "pandas_all_above_threshold",
        "pyspark_all_above_threshold",
        "pandas_with_nulls_success",
        "pyspark_with_nulls_success",
        "pandas_with_nulls_large_success",
        "pyspark_with_nulls_large_success",
        "pandas_with_nulls_violations",
        "pyspark_with_nulls_violations",
    ],
)
def test_expectation_basic_scenarios(
    df_type, data, threshold, expected_result, expected_violations, expected_message, spark
):
    """
    Test the expectation for various scenarios across pandas and PySpark DataFrames.
    Tests both direct expectation validation and suite-based validation.
    Covers: success cases, boundary conditions (including equals), violations, negative values,
    floats, zero values, single values, mixed types, large values, and nulls.
    """
    # Determine data type based on whether we have float values (excluding None)
    has_float = any(isinstance(val, float) for val in data if val is not None)
    data_type = "double" if has_float else "long"
    data_frame = create_dataframe(df_type, data, "col1", spark, data_type)

    # Test 1: Direct expectation validation
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationValueLessThanEquals",
        column_name="col1",
        value=threshold,
    )

    result = expectation.validate(data_frame=data_frame)

    if expected_result == "success":
        assert str(result) == str(
            DataFrameExpectationSuccessMessage(expectation_name="ExpectationValueLessThanEquals")
        ), f"Expected success message but got: {result}"
    else:  # failure
        expected_violations_df = create_dataframe(
            df_type, expected_violations, "col1", spark, data_type
        )
        expected_failure_message = DataFrameExpectationFailureMessage(
            expectation_str=str(expectation),
            data_frame_type=str(df_type),
            violations_data_frame=expected_violations_df,
            message=expected_message,
            limit_violations=5,
        )
        assert str(result) == str(expected_failure_message), (
            f"Expected failure message but got: {result}"
        )

    # Test 2: Suite-based validation
    expectations_suite = DataFrameExpectationsSuite().expect_value_less_than_equals(
        column_name="col1", value=threshold
    )

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


@pytest.mark.parametrize(
    "df_type",
    ["pandas", "pyspark"],
    ids=["pandas", "pyspark"],
)
def test_column_missing_error(df_type, spark):
    """Test that an error is raised when the specified column is missing in both pandas and PySpark."""
    expected_message = "Column 'col1' does not exist in the DataFrame."

    if df_type == "pandas":
        data_frame = pd.DataFrame({"col2": [3, 4, 5]})
    else:  # pyspark
        data_frame = spark.createDataFrame([(3,), (4,), (5,)], ["col2"])

    # Test 1: Direct expectation validation
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationValueLessThanEquals",
        column_name="col1",
        value=2,
    )
    result = expectation.validate(data_frame=data_frame)
    expected_failure = DataFrameExpectationFailureMessage(
        expectation_str=str(expectation),
        data_frame_type=str(df_type),
        message=expected_message,
    )
    assert str(result) == str(expected_failure), f"Expected failure message but got: {result}"

    # Test 2: Suite-based validation
    expectations_suite = DataFrameExpectationsSuite().expect_value_less_than_equals(
        column_name="col1", value=2
    )
    with pytest.raises(DataFrameExpectationsSuiteFailure):
        expectations_suite.build().run(data_frame=data_frame)


def test_large_dataset_performance():
    """Test the expectation with a larger dataset to ensure performance."""
    import numpy as np

    # Create a larger dataset with values between 0 and 90
    large_data = np.random.uniform(0, 90, 10000).tolist()
    data_frame = pd.DataFrame({"col1": large_data})

    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationValueLessThanEquals",
        column_name="col1",
        value=100,
    )

    result = expectation.validate(data_frame=data_frame)
    # Should succeed as all values from uniform(0, 90) are <= 100
    assert isinstance(result, DataFrameExpectationSuccessMessage), (
        f"Large dataset test failed: expected success but got {type(result)}"
    )


def test_boundary_difference_from_less_than():
    """Test that <= behaves differently from < at boundary conditions."""
    data_frame = pd.DataFrame({"col1": [5, 5, 5]})

    # With LessThanEquals, all 5s should pass when threshold is 5
    expectation_lte = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationValueLessThanEquals",
        column_name="col1",
        value=5,
    )
    result_lte = expectation_lte.validate(data_frame=data_frame)
    assert isinstance(result_lte, DataFrameExpectationSuccessMessage), (
        f"LessThanEquals should pass when values equal threshold, got: {result_lte}"
    )

    # With LessThan, all 5s should fail when threshold is 5
    expectation_lt = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationValueLessThan",
        column_name="col1",
        value=5,
    )
    result_lt = expectation_lt.validate(data_frame=data_frame)
    assert isinstance(result_lt, DataFrameExpectationFailureMessage), (
        f"LessThan should fail when values equal threshold, got: {result_lt}"
    )
