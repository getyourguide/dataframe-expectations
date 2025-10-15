import pytest
import numpy as np
import pandas as pd

from dataframe_expectations import DataFrameType
from dataframe_expectations.expectations.expectation_registry import (
    DataframeExpectationRegistry,
)
from dataframe_expectations.expectations_suite import (
    DataframeExpectationsSuite,
    DataframeExpectationsSuiteFailure,
)
from dataframe_expectations.result_message import (
    DataframeExpectationFailureMessage,
    DataframeExpectationSuccessMessage,
)



def test_expectation_name_and_description():
    """Test that the expectation name and description are correctly returned."""
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationColumnQuantileBetween",
        column_name="test_col",
        quantile=0.5,
        min_value=20,
        max_value=30,
    )

    # Test expectation name
    assert expectation.get_expectation_name() == "ExpectationColumnQuantileBetween", f"Expected 'ExpectationColumnQuantileBetween' but got: {expectation.get_expectation_name()}"

    # Test description messages for different quantiles
    test_cases = [
        (0.0, "minimum"),
        (0.25, "25th percentile"),
        (0.5, "median"),
        (0.75, "75th percentile"),
        (1.0, "maximum"),
        (0.9, "0.9 quantile"),
    ]

    for quantile, expected_desc in test_cases:
        exp = DataframeExpectationRegistry.get_expectation(
            expectation_name="ExpectationColumnQuantileBetween",
            column_name="test_col",
            quantile=quantile,
            min_value=10,
            max_value=20,
        )
        assert exp.quantile_desc == expected_desc, f"Expected quantile_desc '{expected_desc}' for quantile {quantile} but got: {exp.quantile_desc}"
        assert expected_desc in exp.get_description(), f"Expected '{expected_desc}' in description: {exp.get_description()}"

def test_pandas_success_registry_and_suite():
    """Test successful validation for pandas DataFrames through both registry and suite."""
    # Test data scenarios for different quantiles
    test_scenarios = [
        # (data, quantile, min_value, max_value, description)
        ([20, 25, 30, 35], 0.0, 15, 25, "minimum success"),  # min = 20
        ([20, 25, 30, 35], 1.0, 30, 40, "maximum success"),  # max = 35
        ([20, 25, 30, 35], 0.5, 25, 30, "median success"),  # median = 27.5
        ([20, 25, 30, 35], 0.25, 20, 25, "25th percentile success"),  # 25th = 22.5
        ([10, 20, 30, 40, 50], 0.33, 20, 30, "33rd percentile success"),  # ~23.2
        ([25], 0.5, 20, 30, "single row median"),  # median = 25
        ([20, None, 25, None, 30], 0.5, 20, 30, "with nulls median"),  # median = 25
    ]

    for data, quantile, min_val, max_val, description in test_scenarios:
        data_frame = pd.DataFrame({"col1": data})

        # Test through registry
        expectation = DataframeExpectationRegistry.get_expectation(
            expectation_name="ExpectationColumnQuantileBetween",
            column_name="col1",
            quantile=quantile,
            min_value=min_val,
            max_value=max_val,
        )
        result = expectation.validate(data_frame=data_frame)
        assert str(result) == str(
            DataframeExpectationSuccessMessage(
                expectation_name="ExpectationColumnQuantileBetween"
            )
        ), f"Registry test failed for {description}: expected success but got {result}"

        # Test through suite
        suite = DataframeExpectationsSuite().expect_column_quantile_between(
            column_name="col1",
            quantile=quantile,
            min_value=min_val,
            max_value=max_val,
        )
        suite_result = suite.run(data_frame=data_frame)
        assert suite_result is None, f"Suite test failed for {description}: expected None but got {suite_result}"

def test_pandas_failure_registry_and_suite():
    """Test failure validation for pandas DataFrames through both registry and suite."""
    # Test data scenarios for different quantiles
    test_scenarios = [
        # (data, quantile, min_value, max_value, expected_message)
        (
            [20, 25, 30, 35],
            0.0,
            25,
            35,
            "Column 'col1' minimum value 20 is not between 25 and 35.",
        ),
        (
            [20, 25, 30, 35],
            1.0,
            40,
            50,
            "Column 'col1' maximum value 35 is not between 40 and 50.",
        ),
        (
            [20, 25, 30, 35],
            0.5,
            30,
            35,
            "Column 'col1' median value 27.5 is not between 30 and 35.",
        ),
        (
            [20, 25, 30, 35],
            0.75,
            25,
            30,
            f"Column 'col1' 75th percentile value {np.quantile([20, 25, 30, 35], 0.75)} is not between 25 and 30.",
        ),
        (
            [None, None, None],
            0.5,
            20,
            30,
            "Column 'col1' contains only null values.",
        ),
        ([], 0.5, 20, 30, "Column 'col1' contains only null values."),
    ]

    for data, quantile, min_val, max_val, expected_message in test_scenarios:
        data_frame = pd.DataFrame({"col1": data})

        # Test through registry
        expectation = DataframeExpectationRegistry.get_expectation(
            expectation_name="ExpectationColumnQuantileBetween",
            column_name="col1",
            quantile=quantile,
            min_value=min_val,
            max_value=max_val,
        )
        result = expectation.validate(data_frame=data_frame)
        expected_failure = DataframeExpectationFailureMessage(
            expectation_str=str(expectation),
            data_frame_type=DataFrameType.PANDAS,
            message=expected_message,
        )
        assert str(result) == str(expected_failure), f"Registry test failed for quantile {quantile}: expected failure message but got {result}"

        # Test through suite
        suite = DataframeExpectationsSuite().expect_column_quantile_between(
            column_name="col1",
            quantile=quantile,
            min_value=min_val,
            max_value=max_val,
        )
        with pytest.raises(DataframeExpectationsSuiteFailure):
            suite.run(data_frame=data_frame)

def test_pyspark_success_registry_and_suite(spark):
    """Test successful validation for PySpark DataFrames through both registry and suite."""
    # Test data scenarios for different quantiles
    test_scenarios = [
        # (data, quantile, min_value, max_value, description)
        ([20, 25, 30, 35], 0.0, 15, 25, "minimum success"),  # min = 20
        ([20, 25, 30, 35], 1.0, 30, 40, "maximum success"),  # max = 35
        ([20, 25, 30, 35], 0.5, 25, 30, "median success"),  # median ≈ 27.5
        ([20, 25, 30, 35], 0.9, 30, 40, "90th percentile success"),  # ≈ 34
        ([25], 0.5, 20, 30, "single row median"),  # median = 25
        ([20, None, 25, None, 30], 0.5, 20, 30, "with nulls median"),  # median ≈ 25
    ]

    for data, quantile, min_val, max_val, description in test_scenarios:
        data_frame = spark.createDataFrame([(val,) for val in data], ["col1"])

        # Test through registry
        expectation = DataframeExpectationRegistry.get_expectation(
            expectation_name="ExpectationColumnQuantileBetween",
            column_name="col1",
            quantile=quantile,
            min_value=min_val,
            max_value=max_val,
        )
        result = expectation.validate(data_frame=data_frame)
        assert str(result) == str(
            DataframeExpectationSuccessMessage(
                expectation_name="ExpectationColumnQuantileBetween"
            )
        ), f"Registry test failed for {description}: expected success but got {result}"

        # Test through suite
        suite = DataframeExpectationsSuite().expect_column_quantile_between(
            column_name="col1",
            quantile=quantile,
            min_value=min_val,
            max_value=max_val,
        )
        suite_result = suite.run(data_frame=data_frame)
        assert suite_result is None, f"Suite test failed for {description}: expected None but got {suite_result}"

def test_pyspark_failure_registry_and_suite(spark):
    """Test failure validation for PySpark DataFrames through both registry and suite."""
    # Test data scenarios for different quantiles
    test_scenarios = [
        # (data, quantile, min_value, max_value, expected_message)
        (
            [20, 25, 30, 35],
            0.0,
            25,
            35,
            "Column 'col1' minimum value 20 is not between 25 and 35.",
        ),
        (
            [20, 25, 30, 35],
            1.0,
            40,
            50,
            "Column 'col1' maximum value 35 is not between 40 and 50.",
        ),
        (
            [20, 25, 30, 35],
            0.5,
            30,
            35,
            f"Column 'col1' median value {np.median([20, 25, 30, 35])} is not between 30 and 35.",
        ),
    ]

    for data, quantile, min_val, max_val, expected_message in test_scenarios:
        data_frame = spark.createDataFrame([(val,) for val in data], ["col1"])

        # Test through registry
        expectation = DataframeExpectationRegistry.get_expectation(
            expectation_name="ExpectationColumnQuantileBetween",
            column_name="col1",
            quantile=quantile,
            min_value=min_val,
            max_value=max_val,
        )
        result = expectation.validate(data_frame=data_frame)
        expected_failure = DataframeExpectationFailureMessage(
            expectation_str=str(expectation),
            data_frame_type=DataFrameType.PYSPARK,
            message=expected_message,
        )
        assert str(result) == str(expected_failure), f"Expected failure message but got: {result}"

        # Test through suite
        suite = DataframeExpectationsSuite().expect_column_quantile_between(
            column_name="col1",
            quantile=quantile,
            min_value=min_val,
            max_value=max_val,
        )
        with pytest.raises(DataframeExpectationsSuiteFailure):
            suite.run(data_frame=data_frame)

def test_pyspark_null_scenarios_registry_and_suite(spark):
    """Test null scenarios for PySpark DataFrames through both registry and suite."""
    from pyspark.sql.types import IntegerType, StructField, StructType

    # Test scenarios
    test_scenarios = [
        # (data_frame_creation, expected_message, description)
        (
            lambda: spark.createDataFrame(
                [{"col1": None}, {"col1": None}, {"col1": None}],
                schema="struct<col1: double>",
            ),
            "Column 'col1' contains only null values.",
            "all nulls",
        ),
        (
            lambda: spark.createDataFrame(
                [], StructType([StructField("col1", IntegerType(), True)])
            ),
            "Column 'col1' contains only null values.",
            "empty dataframe",
        ),
    ]

    for df_creator, expected_message, description in test_scenarios:
        data_frame = df_creator()

        # Test through registry
        expectation = DataframeExpectationRegistry.get_expectation(
            expectation_name="ExpectationColumnQuantileBetween",
            column_name="col1",
            quantile=0.5,
            min_value=20,
            max_value=30,
        )
        result = expectation.validate(data_frame=data_frame)
        expected_failure = DataframeExpectationFailureMessage(
            expectation_str=str(expectation),
            data_frame_type=DataFrameType.PYSPARK,
            message=expected_message,
        )
        assert str(result) == str(expected_failure), f"Registry test failed for {description}: expected failure message but got {result}"

        # Test through suite
        suite = DataframeExpectationsSuite().expect_column_quantile_between(
            column_name="col1", quantile=0.5, min_value=20, max_value=30
        )
        with pytest.raises(DataframeExpectationsSuiteFailure):
            suite.run(data_frame=data_frame)

def test_invalid_quantile_range():
    """Test that invalid quantile values raise ValueError."""
    invalid_quantiles = [
        (1.5, "greater than 1.0"),
        (-0.1, "less than 0.0"),
    ]

    for invalid_quantile, description in invalid_quantiles:
        with pytest.raises(ValueError) as context:
            DataframeExpectationRegistry.get_expectation(
                expectation_name="ExpectationColumnQuantileBetween",
                column_name="col1",
                quantile=invalid_quantile,
                min_value=20,
                max_value=30,
            )
        assert "Quantile must be between 0.0 and 1.0" in str(context.value), f"Expected quantile validation error for {description} but got: {str(context.value)}"

def test_boundary_quantile_values():
    """Test quantile values at the boundaries (0.0 and 1.0)."""
    boundary_cases = [
        (0.0, "minimum"),
        (1.0, "maximum"),
    ]

    for quantile, expected_desc in boundary_cases:
        expectation = DataframeExpectationRegistry.get_expectation(
            expectation_name="ExpectationColumnQuantileBetween",
            column_name="col1",
            quantile=quantile,
            min_value=15,
            max_value=25,
        )
        assert expectation.quantile == quantile, f"Expected quantile {quantile} but got: {expectation.quantile}"
        assert expectation.quantile_desc == expected_desc, f"Expected quantile_desc '{expected_desc}' but got: {expectation.quantile_desc}"

def test_large_dataset_performance():
    """Test the expectation with a larger dataset to ensure performance."""
    # Create a larger dataset
    large_data = np.random.normal(50, 10, 1000).tolist()
    data_frame = pd.DataFrame({"col1": large_data})

    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationColumnQuantileBetween",
        column_name="col1",
        quantile=0.5,  # median
        min_value=45,
        max_value=55,
    )

    result = expectation.validate(data_frame=data_frame)
    # Should succeed as the median of normal(50, 10) should be around 50
    assert isinstance(result, DataframeExpectationSuccessMessage), f"Large dataset test failed: expected success but got {type(result)}"
