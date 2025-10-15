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
        expectation_name="ExpectationColumnMaxBetween",
        column_name="test_col",
        min_value=10,
        max_value=20,
    )

    # Test expectation name
    assert (
        expectation.get_expectation_name() == "ExpectationColumnQuantileBetween"
    ), f"Expected 'ExpectationColumnQuantileBetween' but got: {expectation.get_expectation_name()}"

    # Test description
    description = expectation.get_description()
    assert "maximum" in description, f"Expected 'maximum' in description: {description}"
    assert "test_col" in description, f"Expected 'test_col' in description: {description}"
    assert "10" in description, f"Expected '10' in description: {description}"
    assert "20" in description, f"Expected '20' in description: {description}"


def test_pandas_success_registry_and_suite():
    """Test successful validation for pandas DataFrames through both registry and suite."""
    # Test data scenarios
    test_scenarios = [
        # (data, min_value, max_value, description)
        ([20, 25, 30, 35], 30, 40, "basic success case"),
        ([35], 30, 40, "single row"),
        ([-20, -15, -10, -3], -5, 0, "negative values"),
        ([1.1, 2.5, 3.7, 3.8], 3.5, 4.0, "float values"),
        ([25, 25, 25, 25], 24, 26, "identical values"),
        ([20, 25.5, 30, 37], 35, 40, "mixed data types"),
        ([-5, 0, 0, -2], -1, 1, "zero values"),
        ([20, None, 35, None, 25], 30, 40, "with nulls"),
    ]

    for data, min_val, max_val, description in test_scenarios:
        data_frame = pd.DataFrame({"col1": data})

        # Test through registry
        expectation = DataframeExpectationRegistry.get_expectation(
            expectation_name="ExpectationColumnMaxBetween",
            column_name="col1",
            min_value=min_val,
            max_value=max_val,
        )
        result = expectation.validate(data_frame=data_frame)
        assert str(result) == str(
            DataframeExpectationSuccessMessage(expectation_name="ExpectationColumnQuantileBetween")
        ), f"Registry test failed for {description}: expected success but got {result}"

        # Test through suite
        suite = DataframeExpectationsSuite().expect_column_max_between(
            column_name="col1", min_value=min_val, max_value=max_val
        )
        suite_result = suite.run(data_frame=data_frame)
        assert (
            suite_result is None
        ), f"Suite test failed for {description}: expected None but got {suite_result}"


def test_pandas_failure_registry_and_suite():
    """Test failure validation for pandas DataFrames through both registry and suite."""
    # Test data scenarios
    test_scenarios = [
        # (data, min_value, max_value, expected_message)
        (
            [20, 25, 30, 35],
            40,
            50,
            "Column 'col1' maximum value 35 is not between 40 and 50.",
        ),
        ([None, None, None], 30, 40, "Column 'col1' contains only null values."),
        ([], 30, 40, "Column 'col1' contains only null values."),
    ]

    for data, min_val, max_val, expected_message in test_scenarios:
        data_frame = pd.DataFrame({"col1": data})

        # Test through registry
        expectation = DataframeExpectationRegistry.get_expectation(
            expectation_name="ExpectationColumnMaxBetween",
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
        assert str(result) == str(
            expected_failure
        ), f"Registry test failed for data {data}: expected failure message but got {result}"

        # Test through suite
        suite = DataframeExpectationsSuite().expect_column_max_between(
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
        expectation_name="ExpectationColumnMaxBetween",
        column_name="nonexistent_col",
        min_value=30,
        max_value=40,
    )
    result = expectation.validate(data_frame=data_frame)
    expected_failure = DataframeExpectationFailureMessage(
        expectation_str=str(expectation),
        data_frame_type=DataFrameType.PANDAS,
        message=expected_message,
    )
    assert str(result) == str(expected_failure), f"Expected failure message but got: {result}"

    # Test through suite
    suite = DataframeExpectationsSuite().expect_column_max_between(
        column_name="nonexistent_col", min_value=30, max_value=40
    )
    with pytest.raises(DataframeExpectationsSuiteFailure):
        suite.run(data_frame=data_frame)


def test_pyspark_success_registry_and_suite(spark):
    """Test successful validation for PySpark DataFrames through both registry and suite."""
    # Test data scenarios
    test_scenarios = [
        # (data, min_value, max_value, description)
        ([20, 25, 30, 35], 30, 40, "basic success case"),
        ([35], 30, 40, "single row"),
        ([-20, -15, -10, -3], -5, 0, "negative values"),
        ([20, None, 35, None, 25], 30, 40, "with nulls"),
    ]

    for data, min_val, max_val, description in test_scenarios:
        data_frame = spark.createDataFrame([(val,) for val in data], ["col1"])

        # Test through registry
        expectation = DataframeExpectationRegistry.get_expectation(
            expectation_name="ExpectationColumnMaxBetween",
            column_name="col1",
            min_value=min_val,
            max_value=max_val,
        )
        result = expectation.validate(data_frame=data_frame)
        assert str(result) == str(
            DataframeExpectationSuccessMessage(expectation_name="ExpectationColumnQuantileBetween")
        ), f"Registry test failed for {description}: expected success but got {result}"

        # Test through suite
        suite = DataframeExpectationsSuite().expect_column_max_between(
            column_name="col1", min_value=min_val, max_value=max_val
        )
        suite_result = suite.run(data_frame=data_frame)
        assert (
            suite_result is None
        ), f"Suite test failed for {description}: expected None but got {suite_result}"


def test_pyspark_failure_registry_and_suite(spark):
    """Test failure validation for PySpark DataFrames through both registry and suite."""
    # Test data scenarios
    test_scenarios = [
        # (data, min_value, max_value, expected_message)
        (
            [20, 25, 30, 35],
            40,
            50,
            "Column 'col1' maximum value 35 is not between 40 and 50.",
        ),
    ]

    for data, min_val, max_val, expected_message in test_scenarios:
        data_frame = spark.createDataFrame([(val,) for val in data], ["col1"])

        # Test through registry
        expectation = DataframeExpectationRegistry.get_expectation(
            expectation_name="ExpectationColumnMaxBetween",
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
        suite = DataframeExpectationsSuite().expect_column_max_between(
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
            expectation_name="ExpectationColumnMaxBetween",
            column_name="col1",
            min_value=30,
            max_value=40,
        )
        result = expectation.validate(data_frame=data_frame)
        expected_failure = DataframeExpectationFailureMessage(
            expectation_str=str(expectation),
            data_frame_type=DataFrameType.PYSPARK,
            message=expected_message,
        )
        assert str(result) == str(
            expected_failure
        ), f"Registry test failed for {description}: expected failure message but got {result}"

        # Test through suite
        suite = DataframeExpectationsSuite().expect_column_max_between(
            column_name="col1", min_value=30, max_value=40
        )
        with pytest.raises(DataframeExpectationsSuiteFailure):
            suite.run(data_frame=data_frame)


def test_pyspark_missing_column_registry_and_suite(spark):
    """Test missing column error for PySpark DataFrames through both registry and suite."""
    data_frame = spark.createDataFrame([(20,), (25,), (30,), (35,)], ["col1"])
    expected_message = "Column 'nonexistent_col' does not exist in the DataFrame."

    # Test through registry
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationColumnMaxBetween",
        column_name="nonexistent_col",
        min_value=30,
        max_value=40,
    )
    result = expectation.validate(data_frame=data_frame)
    expected_failure = DataframeExpectationFailureMessage(
        expectation_str=str(expectation),
        data_frame_type=DataFrameType.PYSPARK,
        message=expected_message,
    )
    assert str(result) == str(expected_failure), f"Expected failure message but got: {result}"

    # Test through suite
    suite = DataframeExpectationsSuite().expect_column_max_between(
        column_name="nonexistent_col", min_value=30, max_value=40
    )
    with pytest.raises(DataframeExpectationsSuiteFailure):
        suite.run(data_frame=data_frame)


def test_boundary_values_both_dataframes(spark):
    """Test boundary values for both pandas and PySpark DataFrames."""
    test_data = [20, 25, 30, 35]  # max = 35

    # Test exact minimum boundary
    for df_type, data_frame in [
        ("pandas", pd.DataFrame({"col1": test_data})),
        ("pyspark", spark.createDataFrame([(val,) for val in test_data], ["col1"])),
    ]:
        expectation = DataframeExpectationRegistry.get_expectation(
            expectation_name="ExpectationColumnMaxBetween",
            column_name="col1",
            min_value=35,  # exact minimum boundary
            max_value=40,
        )
        result = expectation.validate(data_frame=data_frame)
        assert isinstance(
            result, DataframeExpectationSuccessMessage
        ), f"Minimum boundary test failed for {df_type}: expected success but got {type(result)}"

        expectation = DataframeExpectationRegistry.get_expectation(
            expectation_name="ExpectationColumnMaxBetween",
            column_name="col1",
            min_value=30,
            max_value=35,  # exact maximum boundary
        )
        result = expectation.validate(data_frame=data_frame)
        assert isinstance(
            result, DataframeExpectationSuccessMessage
        ), f"Maximum boundary test failed for {df_type}: expected success but got {type(result)}"


def test_suite_chaining():
    """Test that the suite method returns self for method chaining."""
    suite = DataframeExpectationsSuite()
    result = suite.expect_column_max_between(column_name="col1", min_value=30, max_value=40)
    assert result is suite, f"Expected suite chaining to return same instance but got: {result}"


def test_large_dataset_performance():
    """Test the expectation with a larger dataset to ensure performance."""
    import numpy as np

    # Create a larger dataset with max around 60
    large_data = np.random.uniform(10, 60, 1000).tolist()
    data_frame = pd.DataFrame({"col1": large_data})

    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationColumnMaxBetween",
        column_name="col1",
        min_value=55,
        max_value=65,
    )

    result = expectation.validate(data_frame=data_frame)
    # Should succeed as the max of uniform(10, 60) should be around 60
    assert isinstance(
        result, DataframeExpectationSuccessMessage
    ), f"Large dataset test failed: expected success but got {type(result)}"
