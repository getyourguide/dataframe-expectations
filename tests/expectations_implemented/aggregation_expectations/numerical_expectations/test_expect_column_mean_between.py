import pytest
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
        expectation_name="ExpectationColumnMeanBetween",
        column_name="test_col",
        min_value=10,
        max_value=20,
    )

    # Test expectation name
    assert expectation.get_expectation_name() == "ExpectationColumnMeanBetween", f"Expected 'ExpectationColumnMeanBetween' but got: {expectation.get_expectation_name()}"

    # Test description
    description = expectation.get_description()
    assert "mean" in description, f"Expected 'mean' in description: {description}"
    assert "test_col" in description, f"Expected 'test_col' in description: {description}"
    assert "10" in description, f"Expected '10' in description: {description}"
    assert "20" in description, f"Expected '20' in description: {description}"

def test_pandas_success_registry_and_suite():
    """Test successful validation for pandas DataFrames through both registry and suite."""
    # Test data scenarios
    test_scenarios = [
        # (data, min_value, max_value, description)
        ([20, 25, 30, 35], 25, 30, "basic success case"),  # mean = 27.5
        ([25], 20, 30, "single row"),  # mean = 25
        ([-20, -15, -10, -5], -15, -10, "negative values"),  # mean = -12.5
        ([1.1, 2.5, 3.7, 3.8], 2.5, 3.0, "float values"),  # mean = 2.775
        ([25, 25, 25, 25], 24, 26, "identical values"),  # mean = 25
        ([20, 25.5, 30, 37], 27, 29, "mixed data types"),  # mean = 28.125
        ([-5, 0, 0, 5], -2, 2, "with zeros"),  # mean = 0
        (
            [20, None, 30, None, 40],
            25,
            35,
            "with nulls",
        ),  # mean = 30 (nulls ignored)
    ]

    for data, min_val, max_val, description in test_scenarios:
        data_frame = pd.DataFrame({"col1": data})

        # Test through registry
        expectation = DataframeExpectationRegistry.get_expectation(
            expectation_name="ExpectationColumnMeanBetween",
            column_name="col1",
            min_value=min_val,
            max_value=max_val,
        )
        result = expectation.validate(data_frame=data_frame)
        assert str(result) == str(
            DataframeExpectationSuccessMessage(
                expectation_name="ExpectationColumnMeanBetween"
            )
        ), f"Registry test failed for {description}: expected success but got {result}"

        # Test through suite
        suite = DataframeExpectationsSuite().expect_column_mean_between(
            column_name="col1", min_value=min_val, max_value=max_val
        )
        suite_result = suite.run(data_frame=data_frame)
        assert suite_result is None, f"Suite test failed for {description}: expected None but got {suite_result}"

def test_pandas_failure_registry_and_suite():
    """Test failure validation for pandas DataFrames through both registry and suite."""
    # Test data scenarios
    test_scenarios = [
        # (data, min_value, max_value, expected_mean, description)
        ([20, 25, 30, 35], 30, 35, 27.5, "mean too low"),
        ([20, 25, 30, 35], 20, 25, 27.5, "mean too high"),
        ([None, None, None], 25, 30, None, "all nulls"),
        ([], 25, 30, None, "empty dataframe"),
    ]

    for data, min_val, max_val, expected_mean, description in test_scenarios:
        data_frame = pd.DataFrame({"col1": data})

        # Determine expected message
        if expected_mean is None:
            expected_message = "Column 'col1' contains only null values."
        else:
            expected_message = f"Column 'col1' mean value {expected_mean} is not between {min_val} and {max_val}."

        # Test through registry
        expectation = DataframeExpectationRegistry.get_expectation(
            expectation_name="ExpectationColumnMeanBetween",
            column_name="col1",
            min_value=min_val,
            max_value=max_val,
        )
        result = expectation.validate(data_frame=data_frame)
        expected_failure = DataframeExpectationFailureMessage(
            expectation_str=str(expectation),
            data_frame_type=DataFrameType.PANDAS,
            message=expected_message,
        )
        assert str(result) == str(expected_failure), f"Registry test failed for {description}: expected failure message but got {result}"

        # Test through suite
        suite = DataframeExpectationsSuite().expect_column_mean_between(
            column_name="col1", min_value=min_val, max_value=max_val
        )
        with pytest.raises(DataframeExpectationsSuiteFailure):
            suite.run(data_frame=data_frame)

def test_pandas_missing_column_registry_and_suite():
    """Test missing column error for pandas DataFrames through both registry and suite."""
    data_frame = pd.DataFrame({"col1": [20, 25, 30, 35]})
    expected_message = "Column 'nonexistent_col' does not exist in the DataFrame."

    # Test through registry
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationColumnMeanBetween",
        column_name="nonexistent_col",
        min_value=25,
        max_value=30,
    )
    result = expectation.validate(data_frame=data_frame)
    expected_failure = DataframeExpectationFailureMessage(
        expectation_str=str(expectation),
        data_frame_type=DataFrameType.PANDAS,
        message=expected_message,
    )
    assert str(result) == str(expected_failure), f"Expected failure message but got: {result}"

    # Test through suite
    suite = DataframeExpectationsSuite().expect_column_mean_between(
        column_name="nonexistent_col", min_value=25, max_value=30
    )
    with pytest.raises(DataframeExpectationsSuiteFailure):
        suite.run(data_frame=data_frame)

def test_pyspark_success_registry_and_suite(spark):
    """Test successful validation for PySpark DataFrames through both registry and suite."""
    # Test data scenarios
    test_scenarios = [
        # (data, min_value, max_value, description)
        ([20, 25, 30, 35], 25, 30, "basic success case"),  # mean = 27.5
        ([25], 20, 30, "single row"),  # mean = 25
        ([-20, -15, -10, -5], -15, -10, "negative values"),  # mean = -12.5
        ([20, None, 30, None, 40], 25, 35, "with nulls"),  # mean = 30
    ]

    for data, min_val, max_val, description in test_scenarios:
        data_frame = spark.createDataFrame([(val,) for val in data], ["col1"])

        # Test through registry
        expectation = DataframeExpectationRegistry.get_expectation(
            expectation_name="ExpectationColumnMeanBetween",
            column_name="col1",
            min_value=min_val,
            max_value=max_val,
        )
        result = expectation.validate(data_frame=data_frame)
        assert str(result) == str(
            DataframeExpectationSuccessMessage(
                expectation_name="ExpectationColumnMeanBetween"
            )
        ), f"Registry test failed for {description}: expected success but got {result}"

        # Test through suite
        suite = DataframeExpectationsSuite().expect_column_mean_between(
            column_name="col1", min_value=min_val, max_value=max_val
        )
        suite_result = suite.run(data_frame=data_frame)
        assert suite_result is None, f"Suite test failed for {description}: expected None but got {suite_result}"

def test_pyspark_failure_registry_and_suite(spark):
    """Test failure validation for PySpark DataFrames through both registry and suite."""
    # Test data scenarios
    test_scenarios = [
        # (data, min_value, max_value, expected_mean, description)
        ([20, 25, 30, 35], 30, 35, 27.5, "mean too low"),
        ([20, 25, 30, 35], 20, 25, 27.5, "mean too high"),
    ]

    for data, min_val, max_val, expected_mean, description in test_scenarios:
        data_frame = spark.createDataFrame([(val,) for val in data], ["col1"])
        expected_message = f"Column 'col1' mean value {expected_mean} is not between {min_val} and {max_val}."

        # Test through registry
        expectation = DataframeExpectationRegistry.get_expectation(
            expectation_name="ExpectationColumnMeanBetween",
            column_name="col1",
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
        suite = DataframeExpectationsSuite().expect_column_mean_between(
            column_name="col1", min_value=min_val, max_value=max_val
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
            expectation_name="ExpectationColumnMeanBetween",
            column_name="col1",
            min_value=25,
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
        suite = DataframeExpectationsSuite().expect_column_mean_between(
            column_name="col1", min_value=25, max_value=30
        )
        with pytest.raises(DataframeExpectationsSuiteFailure):
            suite.run(data_frame=data_frame)

def test_pyspark_missing_column_registry_and_suite(spark):
    """Test missing column error for PySpark DataFrames through both registry and suite."""
    data_frame = spark.createDataFrame([(20,), (25,), (30,), (35,)], ["col1"])
    expected_message = "Column 'nonexistent_col' does not exist in the DataFrame."

    # Test through registry
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationColumnMeanBetween",
        column_name="nonexistent_col",
        min_value=25,
        max_value=30,
    )
    result = expectation.validate(data_frame=data_frame)
    expected_failure = DataframeExpectationFailureMessage(
        expectation_str=str(expectation),
        data_frame_type=DataFrameType.PYSPARK,
        message=expected_message,
    )
    assert str(result) == str(expected_failure), f"Expected failure message but got: {result}"

    # Test through suite
    suite = DataframeExpectationsSuite().expect_column_mean_between(
        column_name="nonexistent_col", min_value=25, max_value=30
    )
    with pytest.raises(DataframeExpectationsSuiteFailure):
        suite.run(data_frame=data_frame)

def test_boundary_values_both_dataframes(spark):
    """Test boundary values for both pandas and PySpark DataFrames."""
    test_data = [20, 25, 30, 35]  # mean = 27.5

    # Test boundary scenarios
    boundary_tests = [
        (27.5, 30, "exact minimum boundary"),  # mean exactly at min
        (25, 27.5, "exact maximum boundary"),  # mean exactly at max
    ]

    for min_val, max_val, boundary_desc in boundary_tests:
        for df_type, data_frame in [
            ("pandas", pd.DataFrame({"col1": test_data})),
            (
                "pyspark",
                spark.createDataFrame([(val,) for val in test_data], ["col1"]),
            ),
        ]:
            expectation = DataframeExpectationRegistry.get_expectation(
                expectation_name="ExpectationColumnMeanBetween",
                column_name="col1",
                min_value=min_val,
                max_value=max_val,
            )
            result = expectation.validate(data_frame=data_frame)
            assert isinstance(result, DataframeExpectationSuccessMessage), f"Boundary test failed for {df_type} with {boundary_desc}: expected success but got {type(result)}"

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

        expectation = DataframeExpectationRegistry.get_expectation(
            expectation_name="ExpectationColumnMeanBetween",
            column_name="col1",
            min_value=min_val,
            max_value=max_val,
        )
        result = expectation.validate(data_frame=data_frame)
        assert isinstance(result, DataframeExpectationSuccessMessage), f"Precision test failed for {description}: expected success but got {type(result)}"

def test_suite_chaining():
    """Test that the suite method returns self for method chaining."""
    suite = DataframeExpectationsSuite()
    result = suite.expect_column_mean_between(
        column_name="col1", min_value=25, max_value=30
    )
    assert result is suite, f"Expected suite chaining to return same instance but got: {result}"

def test_large_dataset_performance():
    """Test the expectation with a larger dataset to ensure performance."""
    import numpy as np

    # Create a larger dataset with mean around 50
    large_data = np.random.normal(50, 10, 1000).tolist()
    data_frame = pd.DataFrame({"col1": large_data})

    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationColumnMeanBetween",
        column_name="col1",
        min_value=45,
        max_value=55,
    )

    result = expectation.validate(data_frame=data_frame)
    # Should succeed as the mean of normal(50, 10) should be around 50
    assert isinstance(result, DataframeExpectationSuccessMessage), f"Large dataset test failed: expected success but got {type(result)}"

def test_outlier_handling(spark):
    """Test mean calculation with outliers."""
    # Test data with outliers
    outlier_scenarios = [
        # (data, min_val, max_val, description)
        ([1, 2, 3, 100], 20, 30, "single high outlier"),  # mean = 26.5
        ([-100, 10, 20, 30], -15, -5, "single low outlier"),  # mean = -10
        ([1, 2, 3, 4, 5, 1000], 150, 200, "extreme outlier"),  # mean ≈ 169.17
    ]

    for data, min_val, max_val, description in outlier_scenarios:
        # Test with pandas
        data_frame = pd.DataFrame({"col1": data})
        expectation = DataframeExpectationRegistry.get_expectation(
            expectation_name="ExpectationColumnMeanBetween",
            column_name="col1",
            min_value=min_val,
            max_value=max_val,
        )
        result = expectation.validate(data_frame=data_frame)
        assert isinstance(result, DataframeExpectationSuccessMessage), f"Pandas outlier test failed for {description}: expected success but got {type(result)}"

        # Test with PySpark
        pyspark_df = spark.createDataFrame([(val,) for val in data], ["col1"])
        result_pyspark = expectation.validate(data_frame=pyspark_df)
        assert isinstance(result_pyspark, DataframeExpectationSuccessMessage), f"PySpark outlier test failed for {description}: expected success but got {type(result_pyspark)}"
