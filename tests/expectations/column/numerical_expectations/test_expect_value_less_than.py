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
        expectation_name="ExpectationValueLessThan",
        column_name="col1",
        value=2,
    )
    assert expectation.get_expectation_name() == "ExpectationValueLessThan", (
        f"Expected 'ExpectationValueLessThan' but got: {expectation.get_expectation_name()}"
    )


@pytest.mark.parametrize(
    "df_type, data, threshold, expected_result, expected_violations, expected_message",
    [
        # Basic success scenarios
        ("pandas", [3, 4, 5], 6, "success", None, None),
        ("pyspark", [3, 4, 5], 6, "success", None, None),
        ("pandas", [1, 2, 3], 4, "success", None, None),
        ("pyspark", [1, 2, 3], 4, "success", None, None),
        ("pandas", [0, 1, 2], 5, "success", None, None),
        ("pyspark", [0, 1, 2], 5, "success", None, None),
        # Basic violation scenarios
        (
            "pandas",
            [3, 4, 5],
            5,
            "failure",
            [5],
            "Found 1 row(s) where 'col1' is not less than 5.",
        ),
        (
            "pyspark",
            [3, 4, 5],
            5,
            "failure",
            [5],
            "Found 1 row(s) where 'col1' is not less than 5.",
        ),
        (
            "pandas",
            [3, 4, 5],
            3,
            "failure",
            [3, 4, 5],
            "Found 3 row(s) where 'col1' is not less than 3.",
        ),
        (
            "pyspark",
            [3, 4, 5],
            3,
            "failure",
            [3, 4, 5],
            "Found 3 row(s) where 'col1' is not less than 3.",
        ),
        (
            "pandas",
            [2, 3, 4, 5],
            4,
            "failure",
            [4, 5],
            "Found 2 row(s) where 'col1' is not less than 4.",
        ),
        (
            "pyspark",
            [2, 3, 4, 5],
            4,
            "failure",
            [4, 5],
            "Found 2 row(s) where 'col1' is not less than 4.",
        ),
        # Boundary conditions - just below threshold - pandas
        ("pandas", [1, 2, 3], 4, "success", None, None),
        ("pyspark", [1, 2, 3], 4, "success", None, None),
        ("pandas", [1.5, 1.8, 1.9], 2.0, "success", None, None),
        ("pyspark", [1.5, 1.8, 1.9], 2.0, "success", None, None),
        # Boundary conditions - at threshold (violation) - pandas
        (
            "pandas",
            [2, 3, 4],
            2,
            "failure",
            [2, 3, 4],
            "Found 3 row(s) where 'col1' is not less than 2.",
        ),
        (
            "pyspark",
            [2, 3, 4],
            2,
            "failure",
            [2, 3, 4],
            "Found 3 row(s) where 'col1' is not less than 2.",
        ),
        (
            "pandas",
            [5, 5, 5],
            5,
            "failure",
            [5, 5, 5],
            "Found 3 row(s) where 'col1' is not less than 5.",
        ),
        (
            "pyspark",
            [5, 5, 5],
            5,
            "failure",
            [5, 5, 5],
            "Found 3 row(s) where 'col1' is not less than 5.",
        ),
        # Boundary conditions - just above threshold - pandas
        (
            "pandas",
            [3, 4, 5],
            3,
            "failure",
            [3, 4, 5],
            "Found 3 row(s) where 'col1' is not less than 3.",
        ),
        (
            "pyspark",
            [3, 4, 5],
            3,
            "failure",
            [3, 4, 5],
            "Found 3 row(s) where 'col1' is not less than 3.",
        ),
        # Negative values - success
        ("pandas", [-5, -3, -2], 0, "success", None, None),
        ("pyspark", [-5, -3, -2], 0, "success", None, None),
        ("pandas", [-10, -8, -6], -5, "success", None, None),
        ("pyspark", [-10, -8, -6], -5, "success", None, None),
        ("pandas", [-3, -2, -1], 0, "success", None, None),
        ("pyspark", [-3, -2, -1], 0, "success", None, None),
        # Negative values - violations
        (
            "pandas",
            [-1, -2, -3],
            -2,
            "failure",
            [-1, -2],
            "Found 2 row(s) where 'col1' is not less than -2.",
        ),
        (
            "pyspark",
            [-1, -2, -3],
            -2,
            "failure",
            [-1, -2],
            "Found 2 row(s) where 'col1' is not less than -2.",
        ),
        (
            "pandas",
            [-3, -4, -5],
            -5,
            "failure",
            [-3, -4, -5],
            "Found 3 row(s) where 'col1' is not less than -5.",
        ),
        (
            "pyspark",
            [-3, -4, -5],
            -5,
            "failure",
            [-3, -4, -5],
            "Found 3 row(s) where 'col1' is not less than -5.",
        ),
        # Float values - success
        ("pandas", [1.5, 2.3, 3.8], 4.0, "success", None, None),
        ("pyspark", [1.5, 2.3, 3.8], 4.0, "success", None, None),
        ("pandas", [0.5, 1.5, 2.5], 3.0, "success", None, None),
        ("pyspark", [0.5, 1.5, 2.5], 3.0, "success", None, None),
        ("pandas", [9.8, 9.9, 9.95], 10.0, "success", None, None),
        ("pyspark", [9.8, 9.9, 9.95], 10.0, "success", None, None),
        # Float values - violations
        (
            "pandas",
            [2.5, 3.0, 3.5],
            2.5,
            "failure",
            [2.5, 3.0, 3.5],
            "Found 3 row(s) where 'col1' is not less than 2.5.",
        ),
        (
            "pyspark",
            [2.5, 3.0, 3.5],
            2.5,
            "failure",
            [2.5, 3.0, 3.5],
            "Found 3 row(s) where 'col1' is not less than 2.5.",
        ),
        (
            "pandas",
            [1.5, 2.5, 3.5],
            2.0,
            "failure",
            [2.5, 3.5],
            "Found 2 row(s) where 'col1' is not less than 2.0.",
        ),
        (
            "pyspark",
            [1.5, 2.5, 3.5],
            2.0,
            "failure",
            [2.5, 3.5],
            "Found 2 row(s) where 'col1' is not less than 2.0.",
        ),
        # Zero as threshold - success
        ("pandas", [-3, -2, -1], 0, "success", None, None),
        ("pyspark", [-3, -2, -1], 0, "success", None, None),
        ("pandas", [-1.0, -0.5, -0.1], 0, "success", None, None),
        ("pyspark", [-1.0, -0.5, -0.1], 0, "success", None, None),
        # Zero as threshold - violations
        (
            "pandas",
            [0, 1, 2],
            0,
            "failure",
            [0, 1, 2],
            "Found 3 row(s) where 'col1' is not less than 0.",
        ),
        (
            "pyspark",
            [0, 1, 2],
            0,
            "failure",
            [0, 1, 2],
            "Found 3 row(s) where 'col1' is not less than 0.",
        ),
        (
            "pandas",
            [-1, 0, 1],
            0,
            "failure",
            [0, 1],
            "Found 2 row(s) where 'col1' is not less than 0.",
        ),
        (
            "pyspark",
            [-1, 0, 1],
            0,
            "failure",
            [0, 1],
            "Found 2 row(s) where 'col1' is not less than 0.",
        ),
        # Zero in data - success - pandas
        ("pandas", [-2, -1, 0], 1, "success", None, None),
        ("pyspark", [-2, -1, 0], 1, "success", None, None),
        # Zero in data - violations - pandas
        (
            "pandas",
            [0, 1, 2],
            0,
            "failure",
            [0, 1, 2],
            "Found 3 row(s) where 'col1' is not less than 0.",
        ),
        (
            "pyspark",
            [0, 1, 2],
            0,
            "failure",
            [0, 1, 2],
            "Found 3 row(s) where 'col1' is not less than 0.",
        ),
        # Single value - success
        ("pandas", [3], 4, "success", None, None),
        ("pyspark", [3], 4, "success", None, None),
        ("pandas", [0], 10, "success", None, None),
        ("pyspark", [0], 10, "success", None, None),
        # Single value - violation
        (
            "pandas",
            [5],
            5,
            "failure",
            [5],
            "Found 1 row(s) where 'col1' is not less than 5.",
        ),
        (
            "pyspark",
            [5],
            5,
            "failure",
            [5],
            "Found 1 row(s) where 'col1' is not less than 5.",
        ),
        (
            "pandas",
            [10],
            5,
            "failure",
            [10],
            "Found 1 row(s) where 'col1' is not less than 5.",
        ),
        (
            "pyspark",
            [10],
            5,
            "failure",
            [10],
            "Found 1 row(s) where 'col1' is not less than 5.",
        ),
        # All values equal to threshold
        (
            "pandas",
            [5, 5, 5, 5],
            5,
            "failure",
            [5, 5, 5, 5],
            "Found 4 row(s) where 'col1' is not less than 5.",
        ),
        (
            "pyspark",
            [5, 5, 5, 5],
            5,
            "failure",
            [5, 5, 5, 5],
            "Found 4 row(s) where 'col1' is not less than 5.",
        ),
        # Mixed integers and floats
        ("pandas", [1, 1.5, 2, 2.5], 3, "success", None, None),
        ("pyspark", [1, 1.5, 2, 2.5], 3, "success", None, None),
        (
            "pandas",
            [2, 2.5, 3, 3.5],
            2.5,
            "failure",
            [2.5, 3, 3.5],
            "Found 3 row(s) where 'col1' is not less than 2.5.",
        ),
        (
            "pyspark",
            [2, 2.5, 3, 3.5],
            2.5,
            "failure",
            [2.5, 3, 3.5],
            "Found 3 row(s) where 'col1' is not less than 2.5.",
        ),
        # Large values
        ("pandas", [100, 500, 900], 1000, "success", None, None),
        ("pyspark", [100, 500, 900], 1000, "success", None, None),
        (
            "pandas",
            [1000, 1500, 2000],
            1000,
            "failure",
            [1000, 1500, 2000],
            "Found 3 row(s) where 'col1' is not less than 1000.",
        ),
        (
            "pyspark",
            [1000, 1500, 2000],
            1000,
            "failure",
            [1000, 1500, 2000],
            "Found 3 row(s) where 'col1' is not less than 1000.",
        ),
        # All values above threshold
        (
            "pandas",
            [6, 7, 8],
            5,
            "failure",
            [6, 7, 8],
            "Found 3 row(s) where 'col1' is not less than 5.",
        ),
        (
            "pyspark",
            [6, 7, 8],
            5,
            "failure",
            [6, 7, 8],
            "Found 3 row(s) where 'col1' is not less than 5.",
        ),
        # With nulls - success (nulls are ignored)
        ("pandas", [1, None, 2, None, 3], 5, "success", None, None),
        ("pyspark", [1, None, 2, None, 3], 5, "success", None, None),
        ("pandas", [0, None, 5, None], 10, "success", None, None),
        ("pyspark", [0, None, 5, None], 10, "success", None, None),
        # With nulls - violations
        (
            "pandas",
            [2.0, None, 5.0, 6.0],
            5,
            "failure",
            [5.0, 6.0],
            "Found 2 row(s) where 'col1' is not less than 5.",
        ),
        (
            "pyspark",
            [2, None, 5, 6],
            5,
            "failure",
            [5, 6],
            "Found 2 row(s) where 'col1' is not less than 5.",
        ),
    ],
    ids=[
        "pandas_basic_success",
        "pyspark_basic_success",
        "pandas_success_different_data",
        "pyspark_success_different_data",
        "pandas_success_small_values",
        "pyspark_success_small_values",
        "pandas_basic_violations",
        "pyspark_basic_violations",
        "pandas_all_violations",
        "pyspark_all_violations",
        "pandas_partial_violations",
        "pyspark_partial_violations",
        "pandas_boundary_just_below",
        "pyspark_boundary_just_below",
        "pandas_boundary_just_below_float",
        "pyspark_boundary_just_below_float",
        "pandas_boundary_at_threshold",
        "pyspark_boundary_at_threshold",
        "pandas_boundary_all_at_threshold",
        "pyspark_boundary_all_at_threshold",
        "pandas_boundary_above_threshold",
        "pyspark_boundary_above_threshold",
        "pandas_negative_success",
        "pyspark_negative_success",
        "pandas_negative_range_success",
        "pyspark_negative_range_success",
        "pandas_negative_to_zero_success",
        "pyspark_negative_to_zero_success",
        "pandas_negative_violations",
        "pyspark_negative_violations",
        "pandas_negative_all_violations",
        "pyspark_negative_all_violations",
        "pandas_float_success",
        "pyspark_float_success",
        "pandas_float_different_success",
        "pyspark_float_different_success",
        "pandas_float_precise_success",
        "pyspark_float_precise_success",
        "pandas_float_violations",
        "pyspark_float_violations",
        "pandas_float_mixed_violations",
        "pyspark_float_mixed_violations",
        "pandas_zero_threshold_success",
        "pyspark_zero_threshold_success",
        "pandas_zero_threshold_float_success",
        "pyspark_zero_threshold_float_success",
        "pandas_zero_threshold_violations",
        "pyspark_zero_threshold_violations",
        "pandas_zero_threshold_mixed_violations",
        "pyspark_zero_threshold_mixed_violations",
        "pandas_zero_in_data_success",
        "pyspark_zero_in_data_success",
        "pandas_zero_in_data_violation",
        "pyspark_zero_in_data_violation",
        "pandas_single_value_success",
        "pyspark_single_value_success",
        "pandas_single_value_large_success",
        "pyspark_single_value_large_success",
        "pandas_single_value_violation",
        "pyspark_single_value_violation",
        "pandas_single_value_above_violation",
        "pyspark_single_value_above_violation",
        "pandas_all_equal_threshold",
        "pyspark_all_equal_threshold",
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
    Covers: success cases, boundary conditions, violations, negative values, floats,
    zero values, single values, mixed types, large values, and nulls.
    """
    # Determine data type based on whether we have float values (excluding None)
    has_float = any(isinstance(val, float) for val in data if val is not None)
    data_type = "double" if has_float else "long"
    data_frame = create_dataframe(df_type, data, "col1", spark, data_type)

    # Test 1: Direct expectation validation
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationValueLessThan",
        column_name="col1",
        value=threshold,
    )

    result = expectation.validate(data_frame=data_frame)

    if expected_result == "success":
        assert str(result) == str(
            DataFrameExpectationSuccessMessage(expectation_name="ExpectationValueLessThan")
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
    expectations_suite = DataFrameExpectationsSuite().expect_value_less_than(
        column_name="col1", value=threshold
    )

    if expected_result == "success":
        result = expectations_suite.build().run(data_frame=data_frame)
        assert result is None, "Expected no exceptions to be raised from suite"
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
        expectation_name="ExpectationValueLessThan",
        column_name="col1",
        value=5,
    )
    result = expectation.validate(data_frame=data_frame)
    expected_failure = DataFrameExpectationFailureMessage(
        expectation_str=str(expectation),
        data_frame_type=str(df_type),
        message=expected_message,
    )
    assert str(result) == str(expected_failure), f"Expected failure message but got: {result}"

    # Test 2: Suite-based validation
    expectations_suite = DataFrameExpectationsSuite().expect_value_less_than(
        column_name="col1", value=5
    )
    with pytest.raises(DataFrameExpectationsSuiteFailure):
        expectations_suite.build().run(data_frame=data_frame)


def test_large_dataset_performance():
    """Test the expectation with a larger dataset to ensure performance."""
    import numpy as np

    # Create a larger dataset with values between 10 and 100
    large_data = np.random.uniform(10, 100, 10000).tolist()
    data_frame = pd.DataFrame({"col1": large_data})

    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationValueLessThan",
        column_name="col1",
        value=105,
    )

    result = expectation.validate(data_frame=data_frame)
    # Should succeed as all values from uniform(10, 100) are < 105
    assert isinstance(result, DataFrameExpectationSuccessMessage), (
        f"Large dataset test failed: expected success but got {type(result)}"
    )
