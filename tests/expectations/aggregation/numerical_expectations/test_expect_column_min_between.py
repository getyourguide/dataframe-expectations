import pytest
import pandas as pd
from pyspark.sql.types import DoubleType, StructField, StructType

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


def create_dataframe(df_type, data, column_name, spark):
    """Create a pandas or PySpark DataFrame."""
    if df_type == "pandas":
        return pd.DataFrame({column_name: data})
    elif df_type == "pyspark":
        # Handle empty or all-null data with explicit schema
        if not data or all(v is None for v in data):
            schema = StructType([StructField(column_name, DoubleType(), True)])
            return spark.createDataFrame([[v] for v in data], schema=schema)

        # Use explicit DoubleType schema if the data contains any float values
        # This ensures consistent type handling for mixed int/float data
        has_float = any(isinstance(v, float) for v in data if v is not None)
        if has_float:
            float_data = [[float(v) if v is not None else None] for v in data]
            schema = StructType([StructField(column_name, DoubleType(), True)])
            return spark.createDataFrame(float_data, schema=schema)
        else:
            # For pure integer data, let PySpark infer the schema
            return spark.createDataFrame([[v] for v in data], schema=[column_name])


def test_expectation_name():
    """Test that the expectation name is correctly returned."""
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationColumnMinBetween",
        column_name="test_col",
        min_value=10,
        max_value=20,
    )
    # Note: minimum expectation delegates to quantile expectation
    assert expectation.get_expectation_name() == "ExpectationColumnQuantileBetween", (
        f"Expected 'ExpectationColumnQuantileBetween' but got: {expectation.get_expectation_name()}"
    )


def test_expectation_description():
    """Test that the expectation description is correctly returned."""
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationColumnMinBetween",
        column_name="test_col",
        min_value=10,
        max_value=20,
    )
    description = expectation.get_description()
    assert "minimum" in description, f"Expected 'minimum' in description: {description}"
    assert "test_col" in description, f"Expected 'test_col' in description: {description}"
    assert "10" in description, f"Expected '10' in description: {description}"
    assert "20" in description, f"Expected '20' in description: {description}"
    # Verify quantile properties
    assert expectation.quantile == 0.0, (
        f"Expected quantile to be 0.0 but got: {expectation.quantile}"
    )
    assert expectation.quantile_desc == "minimum", (
        f"Expected quantile_desc to be 'minimum' but got: {expectation.quantile_desc}"
    )


@pytest.mark.parametrize(
    "df_type, data, min_value, max_value, should_succeed, expected_message",
    [
        # Basic success scenarios
        ("pandas", [20, 25, 30, 35], 15, 25, True, None),  # min = 20
        ("pyspark", [20, 25, 30, 35], 15, 25, True, None),  # min = 20
        # Single row scenarios
        ("pandas", [25], 20, 30, True, None),  # min = 25
        ("pyspark", [25], 20, 30, True, None),  # min = 25
        # Negative value scenarios
        ("pandas", [-20, -15, -10, -5], -25, -15, True, None),  # min = -20
        ("pyspark", [-20, -15, -10, -5], -25, -15, True, None),  # min = -20
        # Float value scenarios
        ("pandas", [1.1, 2.5, 3.7, 3.8], 1.0, 1.5, True, None),  # min = 1.1
        ("pyspark", [1.1, 2.5, 3.7, 3.8], 1.0, 1.5, True, None),  # min = 1.1
        # Identical value scenarios
        ("pandas", [25, 25, 25, 25], 24, 26, True, None),  # min = 25
        ("pyspark", [25, 25, 25, 25], 24, 26, True, None),  # min = 25
        # Mixed type scenarios
        ("pandas", [20, 25.5, 30, 37], 15, 25, True, None),  # min = 20
        ("pyspark", [20, 25.5, 30, 37], 15, 25, True, None),  # min = 20
        # Zero scenarios
        ("pandas", [-5, 0, 0, 2], -10, -1, True, None),  # min = -5
        ("pyspark", [-5, 0, 0, 2], -10, -1, True, None),  # min = -5
        # Null scenarios
        ("pandas", [20, None, 35, None, 25], 15, 25, True, None),  # min = 20
        ("pyspark", [20, None, 35, None, 25], 15, 25, True, None),  # min = 20
        # Boundary scenarios - exact minimum
        ("pandas", [20, 25, 30, 35], 20, 25, True, None),  # min = 20
        ("pyspark", [20, 25, 30, 35], 20, 25, True, None),  # min = 20
        # Boundary scenarios - exact maximum
        ("pandas", [20, 25, 30, 35], 15, 20, True, None),  # min = 20
        ("pyspark", [20, 25, 30, 35], 15, 20, True, None),  # min = 20
        # Minimum calculation - mixed order
        ("pandas", [100, 50, 75, 25], 24, 26, True, None),  # min = 25
        ("pyspark", [100, 50, 75, 25], 24, 26, True, None),  # min = 25
        # Minimum calculation - zero
        ("pandas", [0, 1, 2, 3], -0.1, 0.1, True, None),  # min = 0
        ("pyspark", [0, 1, 2, 3], -0.1, 0.1, True, None),  # min = 0
        # Minimum calculation - negative
        ("pandas", [-10, -5, -1, -20], -20.1, -19.9, True, None),  # min = -20
        ("pyspark", [-10, -5, -1, -20], -20.1, -19.9, True, None),  # min = -20
        # Minimum calculation - small differences
        ("pandas", [1.001, 1.002, 1.003], 1.0, 1.002, True, None),  # min = 1.001
        ("pyspark", [1.001, 1.002, 1.003], 1.0, 1.002, True, None),  # min = 1.001
        # Minimum calculation - large numbers
        ("pandas", [1e6, 1e5, 1e4], 1e4 - 100, 1e4 + 100, True, None),  # min = 1e4
        ("pyspark", [1e6, 1e5, 1e4], 1e4 - 100, 1e4 + 100, True, None),  # min = 1e4
        # Minimum calculation - very small numbers
        ("pandas", [1e-6, 1e-5, 1e-4], 1e-7, 1e-5, True, None),  # min = 1e-6
        ("pyspark", [1e-6, 1e-5, 1e-4], 1e-7, 1e-5, True, None),  # min = 1e-6
        # Outlier impact - extreme low outlier
        ("pandas", [1, 2, 3, -1000], -1100, -900, True, None),
        ("pyspark", [1, 2, 3, -1000], -1100, -900, True, None),
        # Outlier impact - significant outlier
        ("pandas", [100, 200, 300, 50], 40, 60, True, None),
        ("pyspark", [100, 200, 300, 50], 40, 60, True, None),
        # Outlier impact - small outlier
        ("pandas", [1.5, 2.0, 2.5, 0.1], 0.05, 0.15, True, None),
        ("pyspark", [1.5, 2.0, 2.5, 0.1], 0.05, 0.15, True, None),
        # Identical values - integer repetition
        ("pandas", [42, 42, 42, 42], 41.9, 42.1, True, None),
        ("pyspark", [42, 42, 42, 42], 41.9, 42.1, True, None),
        # Identical values - float repetition
        ("pandas", [3.14, 3.14, 3.14], 3.13, 3.15, True, None),
        ("pyspark", [3.14, 3.14, 3.14], 3.13, 3.15, True, None),
        # Identical values - negative repetition
        ("pandas", [-7, -7, -7, -7, -7], -7.1, -6.9, True, None),
        ("pyspark", [-7, -7, -7, -7, -7], -7.1, -6.9, True, None),
        # Identical values - zero repetition
        ("pandas", [0, 0, 0], -0.1, 0.1, True, None),
        ("pyspark", [0, 0, 0], -0.1, 0.1, True, None),
        # Failure scenarios - minimum too low
        (
            "pandas",
            [20, 25, 30, 35],
            25,
            35,
            False,
            "Column 'col1' minimum value 20 is not between 25 and 35.",
        ),
        (
            "pyspark",
            [20, 25, 30, 35],
            25,
            35,
            False,
            "Column 'col1' minimum value 20 is not between 25 and 35.",
        ),
        # Failure scenarios - minimum too high
        (
            "pandas",
            [20, 25, 30, 35],
            10,
            15,
            False,
            "Column 'col1' minimum value 20 is not between 10 and 15.",
        ),
        (
            "pyspark",
            [20, 25, 30, 35],
            10,
            15,
            False,
            "Column 'col1' minimum value 20 is not between 10 and 15.",
        ),
        # Failure scenarios - all nulls
        ("pandas", [None, None, None], 15, 25, False, "Column 'col1' contains only null values."),
        ("pyspark", [None, None, None], 15, 25, False, "Column 'col1' contains only null values."),
        # Failure scenarios - empty
        ("pandas", [], 15, 25, False, "Column 'col1' contains only null values."),
        ("pyspark", [], 15, 25, False, "Column 'col1' contains only null values."),
    ],
    ids=[
        "pandas_basic_success",
        "pyspark_basic_success",
        "pandas_single_row",
        "pyspark_single_row",
        "pandas_negative_values",
        "pyspark_negative_values",
        "pandas_float_values",
        "pyspark_float_values",
        "pandas_identical_values",
        "pyspark_identical_values",
        "pandas_mixed_types",
        "pyspark_mixed_types",
        "pandas_with_zeros",
        "pyspark_with_zeros",
        "pandas_with_nulls",
        "pyspark_with_nulls",
        "pandas_boundary_exact_min",
        "pyspark_boundary_exact_min",
        "pandas_boundary_exact_max",
        "pyspark_boundary_exact_max",
        "pandas_calc_mixed_order",
        "pyspark_calc_mixed_order",
        "pandas_calc_zero",
        "pyspark_calc_zero",
        "pandas_calc_negative",
        "pyspark_calc_negative",
        "pandas_calc_small_differences",
        "pyspark_calc_small_differences",
        "pandas_calc_large_numbers",
        "pyspark_calc_large_numbers",
        "pandas_calc_very_small_numbers",
        "pyspark_calc_very_small_numbers",
        "pandas_outlier_extreme_low",
        "pyspark_outlier_extreme_low",
        "pandas_outlier_significant",
        "pyspark_outlier_significant",
        "pandas_outlier_small",
        "pyspark_outlier_small",
        "pandas_identical_integer",
        "pyspark_identical_integer",
        "pandas_identical_float",
        "pyspark_identical_float",
        "pandas_identical_negative",
        "pyspark_identical_negative",
        "pandas_identical_zero",
        "pyspark_identical_zero",
        "pandas_min_too_low",
        "pyspark_min_too_low",
        "pandas_min_too_high",
        "pyspark_min_too_high",
        "pandas_all_nulls",
        "pyspark_all_nulls",
        "pandas_empty",
        "pyspark_empty",
    ],
)
def test_expectation_basic_scenarios(
    df_type, data, min_value, max_value, should_succeed, expected_message, spark
):
    """Test basic expectation scenarios for both pandas and PySpark DataFrames."""
    df = create_dataframe(df_type, data, "col1", spark)

    # Test through registry
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationColumnMinBetween",
        column_name="col1",
        min_value=min_value,
        max_value=max_value,
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
        assert expected_message in str(result), (
            f"Expected message '{expected_message}' in result: {result}"
        )

    # Test through suite
    suite = DataFrameExpectationsSuite().expect_column_min_between(
        column_name="col1", min_value=min_value, max_value=max_value
    )

    if should_succeed:
        suite_result = suite.build().run(data_frame=df)
        assert suite_result is None, f"Suite test expected None but got: {suite_result}"
    else:
        with pytest.raises(DataFrameExpectationsSuiteFailure):
            suite.build().run(data_frame=df)


@pytest.mark.parametrize(
    "df_type",
    ["pandas", "pyspark"],
)
def test_column_missing_error(df_type, spark):
    """Test missing column error for both pandas and PySpark DataFrames."""
    if df_type == "pandas":
        df = pd.DataFrame({"col1": [20, 25, 30, 35]})
    else:
        df = spark.createDataFrame([(20,), (25,), (30,), (35,)], ["col1"])

    expected_message = "Column 'nonexistent_col' does not exist in the DataFrame."

    # Test through registry
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationColumnMinBetween",
        column_name="nonexistent_col",
        min_value=15,
        max_value=25,
    )
    result = expectation.validate(data_frame=df)
    expected_failure = DataFrameExpectationFailureMessage(
        expectation_str=str(expectation),
        data_frame_type=str(df_type),
        message=expected_message,
    )
    assert str(result) == str(expected_failure)

    # Test through suite
    suite = DataFrameExpectationsSuite().expect_column_min_between(
        column_name="nonexistent_col", min_value=15, max_value=25
    )
    with pytest.raises(DataFrameExpectationsSuiteFailure):
        suite.build().run(data_frame=df)


def test_large_dataset_performance():
    """Test the expectation with a larger dataset to ensure performance."""
    import numpy as np

    # Create a larger dataset with minimum around 10
    large_data = np.random.uniform(10, 60, 1000).tolist()
    data_frame = pd.DataFrame({"col1": large_data})

    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationColumnMinBetween",
        column_name="col1",
        min_value=9,
        max_value=12,
    )

    result = expectation.validate(data_frame=data_frame)
    # Should succeed as the minimum of uniform(10, 60) should be around 10
    assert isinstance(result, DataFrameExpectationSuccessMessage)
