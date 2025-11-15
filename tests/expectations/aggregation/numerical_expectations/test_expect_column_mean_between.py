import pytest
import pandas as pd

from dataframe_expectations.core.types import DataFrameType
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
    """Helper function to create pandas or pyspark DataFrame."""
    if df_type == "pandas":
        return pd.DataFrame({column_name: data})
    else:  # pyspark
        from pyspark.sql.types import DoubleType, StructField, StructType

        if not data:  # Empty DataFrame
            schema = StructType([StructField(column_name, DoubleType(), True)])
            return spark.createDataFrame([], schema)
        # Handle all nulls case with explicit schema
        if all(val is None for val in data):
            schema = StructType([StructField(column_name, DoubleType(), True)])
            return spark.createDataFrame([{column_name: None} for _ in data], schema)
        return spark.createDataFrame([(val,) for val in data], [column_name])


def get_df_type_enum(df_type):
    """Get DataFrameType enum value."""
    return DataFrameType.PANDAS if df_type == "pandas" else DataFrameType.PYSPARK


def test_expectation_name():
    """Test that the expectation name is correctly returned."""
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationColumnMeanBetween",
        column_name="col1",
        min_value=10,
        max_value=20,
    )
    assert expectation.get_expectation_name() == "ExpectationColumnMeanBetween", (
        f"Expected 'ExpectationColumnMeanBetween' but got: {expectation.get_expectation_name()}"
    )


def test_expectation_description():
    """Test that the expectation description is correctly returned."""
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationColumnMeanBetween",
        column_name="test_col",
        min_value=10,
        max_value=20,
    )
    description = expectation.get_description()
    assert "mean" in description, f"Expected 'mean' in description: {description}"
    assert "test_col" in description, f"Expected 'test_col' in description: {description}"
    assert "10" in description, f"Expected '10' in description: {description}"
    assert "20" in description, f"Expected '20' in description: {description}"


@pytest.mark.parametrize(
    "df_type, data, min_value, max_value, expected_result, expected_message",
    [
        # Success scenarios - pandas
        ("pandas", [20, 25, 30, 35], 25, 30, "success", None),  # mean = 27.5
        ("pandas", [25], 20, 30, "success", None),  # mean = 25
        ("pandas", [-20, -15, -10, -5], -15, -10, "success", None),  # mean = -12.5
        ("pandas", [1.1, 2.5, 3.7, 3.8], 2.5, 3.0, "success", None),  # mean = 2.775
        ("pandas", [25, 25, 25, 25], 24, 26, "success", None),  # mean = 25
        ("pandas", [20, 25.5, 30, 37], 27, 29, "success", None),  # mean = 28.125
        ("pandas", [-5, 0, 0, 5], -2, 2, "success", None),  # mean = 0
        ("pandas", [20, None, 30, None, 40], 25, 35, "success", None),  # mean = 30
        # Success scenarios - pyspark
        ("pyspark", [20, 25, 30, 35], 25, 30, "success", None),  # mean = 27.5
        ("pyspark", [25], 20, 30, "success", None),  # mean = 25
        ("pyspark", [-20, -15, -10, -5], -15, -10, "success", None),  # mean = -12.5
        ("pyspark", [20, None, 30, None, 40], 25, 35, "success", None),  # mean = 30
        # Boundary scenarios - pandas (mean = 27.5)
        ("pandas", [20, 25, 30, 35], 27.5, 30, "success", None),  # exact min boundary
        ("pandas", [20, 25, 30, 35], 25, 27.5, "success", None),  # exact max boundary
        # Boundary scenarios - pyspark (mean = 27.5)
        ("pyspark", [20, 25, 30, 35], 27.5, 30, "success", None),  # exact min boundary
        ("pyspark", [20, 25, 30, 35], 25, 27.5, "success", None),  # exact max boundary
        # Failure scenarios - pandas
        (
            "pandas",
            [20, 25, 30, 35],
            30,
            35,
            "failure",
            "Column 'col1' mean value 27.5 is not between 30 and 35.",
        ),  # mean too low
        (
            "pandas",
            [20, 25, 30, 35],
            20,
            25,
            "failure",
            "Column 'col1' mean value 27.5 is not between 20 and 25.",
        ),  # mean too high
        (
            "pandas",
            [None, None, None],
            25,
            30,
            "failure",
            "Column 'col1' contains only null values.",
        ),
        ("pandas", [], 25, 30, "failure", "Column 'col1' contains only null values."),
        # Failure scenarios - pyspark
        (
            "pyspark",
            [20, 25, 30, 35],
            30,
            35,
            "failure",
            "Column 'col1' mean value 27.5 is not between 30 and 35.",
        ),  # mean too low
        (
            "pyspark",
            [20, 25, 30, 35],
            20,
            25,
            "failure",
            "Column 'col1' mean value 27.5 is not between 20 and 25.",
        ),  # mean too high
        (
            "pyspark",
            [None, None, None],
            25,
            30,
            "failure",
            "Column 'col1' contains only null values.",
        ),
        ("pyspark", [], 25, 30, "failure", "Column 'col1' contains only null values."),
        # Outlier scenarios - pandas
        ("pandas", [1, 2, 3, 100], 20, 30, "success", None),  # mean = 26.5, single high outlier
        ("pandas", [-100, 10, 20, 30], -15, -5, "success", None),  # mean = -10, single low outlier
        (
            "pandas",
            [1, 2, 3, 4, 5, 1000],
            150,
            200,
            "success",
            None,
        ),  # mean ≈ 169.17, extreme outlier
        # Outlier scenarios - pyspark
        ("pyspark", [1, 2, 3, 100], 20, 30, "success", None),  # mean = 26.5
        ("pyspark", [-100, 10, 20, 30], -15, -5, "success", None),  # mean = -10
        ("pyspark", [1, 2, 3, 4, 5, 1000], 150, 200, "success", None),  # mean ≈ 169.17
    ],
    ids=[
        "pandas_basic_success",
        "pandas_single_row",
        "pandas_negative_values",
        "pandas_float_values",
        "pandas_identical_values",
        "pandas_mixed_types",
        "pandas_with_zeros",
        "pandas_with_nulls",
        "pyspark_basic_success",
        "pyspark_single_row",
        "pyspark_negative_values",
        "pyspark_with_nulls",
        "pandas_boundary_exact_min",
        "pandas_boundary_exact_max",
        "pyspark_boundary_exact_min",
        "pyspark_boundary_exact_max",
        "pandas_mean_too_low",
        "pandas_mean_too_high",
        "pandas_all_nulls",
        "pandas_empty",
        "pyspark_mean_too_low",
        "pyspark_mean_too_high",
        "pyspark_all_nulls",
        "pyspark_empty",
        "pandas_outlier_high",
        "pandas_outlier_low",
        "pandas_outlier_extreme",
        "pyspark_outlier_high",
        "pyspark_outlier_low",
        "pyspark_outlier_extreme",
    ],
)
def test_expectation_basic_scenarios(
    df_type, data, min_value, max_value, expected_result, expected_message, spark
):
    """
    Test the expectation for various scenarios across pandas and PySpark DataFrames.
    Tests both direct expectation validation and suite-based validation.
    Covers: success cases, boundary conditions, failures (mean out of range, nulls, empty),
    and various data types (integers, floats, negatives, nulls, mixed types, outliers).
    """
    data_frame = create_dataframe(df_type, data, "col1", spark)

    # Test 1: Direct expectation validation
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationColumnMeanBetween",
        column_name="col1",
        min_value=min_value,
        max_value=max_value,
    )

    result = expectation.validate(data_frame=data_frame)

    if expected_result == "success":
        assert str(result) == str(
            DataFrameExpectationSuccessMessage(expectation_name="ExpectationColumnMeanBetween")
        ), f"Expected success message but got: {result}"
    else:  # failure
        expected_failure_message = DataFrameExpectationFailureMessage(
            expectation_str=str(expectation),
            data_frame_type=get_df_type_enum(df_type),
            message=expected_message,
        )
        assert str(result) == str(expected_failure_message), (
            f"Expected failure message but got: {result}"
        )

    # Test 2: Suite-based validation
    expectations_suite = DataFrameExpectationsSuite().expect_column_mean_between(
        column_name="col1", min_value=min_value, max_value=max_value
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
    expected_message = "Column 'nonexistent_col' does not exist in the DataFrame."

    if df_type == "pandas":
        data_frame = pd.DataFrame({"col1": [20, 25, 30, 35]})
    else:  # pyspark
        data_frame = spark.createDataFrame([(20,), (25,), (30,), (35,)], ["col1"])

    # Test 1: Direct expectation validation
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationColumnMeanBetween",
        column_name="nonexistent_col",
        min_value=25,
        max_value=30,
    )
    result = expectation.validate(data_frame=data_frame)
    expected_failure = DataFrameExpectationFailureMessage(
        expectation_str=str(expectation),
        data_frame_type=get_df_type_enum(df_type),
        message=expected_message,
    )
    assert str(result) == str(expected_failure), f"Expected failure message but got: {result}"

    # Test 2: Suite-based validation
    expectations_suite = DataFrameExpectationsSuite().expect_column_mean_between(
        column_name="nonexistent_col", min_value=25, max_value=30
    )
    with pytest.raises(DataFrameExpectationsSuiteFailure):
        expectations_suite.build().run(data_frame=data_frame)


def test_precision_handling():
    """Test mean calculation precision with various numeric types."""
    # Test scenarios with different levels of precision
    precision_tests = [
        # (data, description)
        ([1.1111, 2.2222, 3.3333], "high precision decimals"),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9], "integer sequence"),
        ([0.1, 0.2, 0.3, 0.4, 0.5], "decimal sequence"),
        ([1e-6, 2e-6, 3e-6], "scientific notation"),
    ]

    for data, description in precision_tests:
        data_frame = pd.DataFrame({"col1": data})
        calculated_mean = sum(data) / len(data)

        # Use a range around the calculated mean
        min_val = calculated_mean - 0.1
        max_val = calculated_mean + 0.1

        expectation = DataFrameExpectationRegistry.get_expectation(
            expectation_name="ExpectationColumnMeanBetween",
            column_name="col1",
            min_value=min_val,
            max_value=max_val,
        )
        result = expectation.validate(data_frame=data_frame)
        assert isinstance(result, DataFrameExpectationSuccessMessage), (
            f"Precision test failed for {description}: expected success but got {type(result)}"
        )


def test_large_dataset_performance():
    """Test the expectation with a larger dataset to ensure performance."""
    import numpy as np

    # Create a larger dataset with mean around 50
    large_data = np.random.normal(50, 10, 1000).tolist()
    data_frame = pd.DataFrame({"col1": large_data})

    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationColumnMeanBetween",
        column_name="col1",
        min_value=45,
        max_value=55,
    )

    result = expectation.validate(data_frame=data_frame)
    # Should succeed as the mean of normal(50, 10) should be around 50
    assert isinstance(result, DataFrameExpectationSuccessMessage), (
        f"Large dataset test failed: expected success but got {type(result)}"
    )
