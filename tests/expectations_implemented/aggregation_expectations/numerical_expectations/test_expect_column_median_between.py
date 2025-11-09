import pytest
import pandas as pd

from dataframe_expectations import DataFrameType
from dataframe_expectations.expectations.expectation_registry import (
    DataFrameExpectationRegistry,
)
from dataframe_expectations.expectations_suite import (
    DataFrameExpectationsSuite,
    DataFrameExpectationsSuiteFailure,
)
from dataframe_expectations.result_message import (
    DataFrameExpectationFailureMessage,
    DataFrameExpectationSuccessMessage,
)


def test_expectation_name_and_description():
    """Test that the expectation name and description are correctly returned."""
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationColumnMedianBetween",
        column_name="test_col",
        min_value=10,
        max_value=20,
    )

    # Test expectation name (should delegate to quantile expectation)
    assert expectation.get_expectation_name() == "ExpectationColumnQuantileBetween", (
        f"Expected 'ExpectationColumnQuantileBetween' but got: {expectation.get_expectation_name()}"
    )

    # Test description
    description = expectation.get_description()
    assert "median" in description, f"Expected 'median' in description: {description}"
    assert "test_col" in description, f"Expected 'test_col' in description: {description}"
    assert "10" in description, f"Expected '10' in description: {description}"
    assert "20" in description, f"Expected '20' in description: {description}"

    # Test that quantile is correctly set to 0.5
    assert expectation.quantile == 0.5, (
        f"Expected quantile to be 0.5 but got: {expectation.quantile}"
    )
    assert expectation.quantile_desc == "median", (
        f"Expected quantile_desc to be 'median' but got: {expectation.quantile_desc}"
    )


def test_pandas_success_registry_and_suite():
    """Test successful validation for pandas DataFrames through both registry and suite."""
    # Test data scenarios
    test_scenarios = [
        # (data, min_value, max_value, description)
        ([20, 25, 30, 35], 25, 30, "basic success case"),  # median = 27.5
        ([25], 20, 30, "single row"),  # median = 25
        ([-20, -15, -10, -5], -15, -10, "negative values"),  # median = -12.5
        ([1.1, 2.5, 3.7, 3.8], 2.5, 3.5, "float values"),  # median = 3.1
        ([25, 25, 25, 25], 24, 26, "identical values"),  # median = 25
        ([20, 25.5, 30, 37], 27, 29, "mixed data types"),  # median = 27.75
        ([-5, 0, 0, 5], -1, 1, "with zeros"),  # median = 0
        (
            [20, None, 30, None, 40],
            25,
            35,
            "with nulls",
        ),  # median = 30 (nulls ignored)
        ([10, 20, 30], 19, 21, "odd number of values"),  # median = 20
        ([10, 20, 30, 40], 24, 26, "even number of values"),  # median = 25
    ]

    for data, min_val, max_val, description in test_scenarios:
        data_frame = pd.DataFrame({"col1": data})

        # Test through registry
        expectation = DataFrameExpectationRegistry.get_expectation(
            expectation_name="ExpectationColumnMedianBetween",
            column_name="col1",
            min_value=min_val,
            max_value=max_val,
        )
        result = expectation.validate(data_frame=data_frame)
        assert str(result) == str(
            DataFrameExpectationSuccessMessage(expectation_name="ExpectationColumnQuantileBetween")
        ), f"Registry test failed for {description}: expected success but got {result}"

        # Test through suite
        suite = DataFrameExpectationsSuite().expect_column_median_between(
            column_name="col1", min_value=min_val, max_value=max_val
        )
        suite_result = suite.build().run(data_frame=data_frame)
        assert suite_result is None, (
            f"Suite test failed for {description}: expected None but got {suite_result}"
        )


def test_pandas_failure_registry_and_suite():
    """Test failure validation for pandas DataFrames through both registry and suite."""
    # Test data scenarios
    test_scenarios = [
        # (data, min_value, max_value, expected_median, description)
        ([20, 25, 30, 35], 30, 35, 27.5, "median too low"),
        ([20, 25, 30, 35], 20, 25, 27.5, "median too high"),
        ([10, 20, 30], 25, 30, 20.0, "odd count median out of range"),
        ([None, None, None], 25, 30.0, None, "all nulls"),
        ([], 25, 30.0, None, "empty dataframe"),
    ]

    for data, min_val, max_val, expected_median, description in test_scenarios:
        data_frame = pd.DataFrame({"col1": data})

        # Determine expected message
        if expected_median is None:
            expected_message = "Column 'col1' contains only null values."
        else:
            expected_message = f"Column 'col1' median value {expected_median} is not between {min_val} and {max_val}."

        # Test through registry
        expectation = DataFrameExpectationRegistry.get_expectation(
            expectation_name="ExpectationColumnMedianBetween",
            column_name="col1",
            min_value=min_val,
            max_value=max_val,
        )
        result = expectation.validate(data_frame=data_frame)
        expected_failure = DataFrameExpectationFailureMessage(
            expectation_str=str(expectation),
            data_frame_type=DataFrameType.PANDAS,
            message=expected_message,
        )
        assert str(result) == str(expected_failure), (
            f"Registry test failed for {description}: expected failure message but got {result}"
        )

        # Test through suite
        suite = DataFrameExpectationsSuite().expect_column_median_between(
            column_name="col1", min_value=min_val, max_value=max_val
        )
        with pytest.raises(DataFrameExpectationsSuiteFailure):
            suite.build().run(data_frame=data_frame)


def test_pandas_missing_column_registry_and_suite():
    """Test missing column error for pandas DataFrames through both registry and suite."""
    data_frame = pd.DataFrame({"col1": [20, 25, 30, 35]})
    expected_message = "Column 'nonexistent_col' does not exist in the DataFrame."

    # Test through registry
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationColumnMedianBetween",
        column_name="nonexistent_col",
        min_value=25,
        max_value=30,
    )
    result = expectation.validate(data_frame=data_frame)
    expected_failure = DataFrameExpectationFailureMessage(
        expectation_str=str(expectation),
        data_frame_type=DataFrameType.PANDAS,
        message=expected_message,
    )
    assert str(result) == str(expected_failure)

    # Test through suite
    suite = DataFrameExpectationsSuite().expect_column_median_between(
        column_name="nonexistent_col", min_value=25, max_value=30
    )
    with pytest.raises(DataFrameExpectationsSuiteFailure):
        suite.build().run(data_frame=data_frame)


def test_pyspark_success_registry_and_suite(spark):
    """Test successful validation for PySpark DataFrames through both registry and suite."""
    # Test data scenarios
    test_scenarios = [
        # (data, min_value, max_value, description)
        ([20, 25, 30, 35], 25, 30, "basic success case"),  # median ≈ 27.5
        ([25], 20, 30, "single row"),  # median = 25
        ([-20, -15, -10, -5], -15, -10, "negative values"),  # median ≈ -12.5
        ([20, None, 30, None, 40], 25, 35, "with nulls"),  # median ≈ 30
        ([10, 20, 30], 19, 21, "odd number of values"),  # median ≈ 20
        ([10, 20, 30, 40], 24, 26, "even number of values"),  # median ≈ 25
    ]

    for data, min_val, max_val, description in test_scenarios:
        data_frame = spark.createDataFrame([(val,) for val in data], ["col1"])

        # Test through registry
        expectation = DataFrameExpectationRegistry.get_expectation(
            expectation_name="ExpectationColumnMedianBetween",
            column_name="col1",
            min_value=min_val,
            max_value=max_val,
        )
        result = expectation.validate(data_frame=data_frame)
        assert str(result) == str(
            DataFrameExpectationSuccessMessage(expectation_name="ExpectationColumnQuantileBetween")
        ), f"Registry test failed for {description}: expected success but got {result}"

        # Test through suite
        suite = DataFrameExpectationsSuite().expect_column_median_between(
            column_name="col1", min_value=min_val, max_value=max_val
        )
        suite_result = suite.build().run(data_frame=data_frame)
        assert suite_result is None, (
            f"Suite test failed for {description}: expected None but got {suite_result}"
        )


def test_pyspark_failure_registry_and_suite(spark):
    """Test failure validation for PySpark DataFrames through both registry and suite."""
    import numpy as np

    # Test data scenarios
    test_scenarios = [
        # (data, min_value, max_value, description)
        ([20, 25, 30, 35], 30, 35, "median too low"),
        ([20, 25, 30, 35], 20, 25, "median too high"),
        ([10, 20, 30], 25, 30, "odd count median out of range"),
    ]

    for data, min_val, max_val, description in test_scenarios:
        data_frame = spark.createDataFrame([(val,) for val in data], ["col1"])

        # Calculate expected median for error message
        expected_median = np.median(data)
        expected_message = (
            f"Column 'col1' median value {expected_median} is not between {min_val} and {max_val}."
        )

        # Test through registry
        expectation = DataFrameExpectationRegistry.get_expectation(
            expectation_name="ExpectationColumnMedianBetween",
            column_name="col1",
            min_value=min_val,
            max_value=max_val,
        )
        result = expectation.validate(data_frame=data_frame)
        expected_failure = DataFrameExpectationFailureMessage(
            expectation_str=str(expectation),
            data_frame_type=DataFrameType.PYSPARK,
            message=expected_message,
        )
        assert str(result) == str(expected_failure), f"Expected failure message but got: {result}"

        # Test through suite
        suite = DataFrameExpectationsSuite().expect_column_median_between(
            column_name="col1", min_value=min_val, max_value=max_val
        )
        with pytest.raises(DataFrameExpectationsSuiteFailure):
            suite.build().run(data_frame=data_frame)


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
        expectation = DataFrameExpectationRegistry.get_expectation(
            expectation_name="ExpectationColumnMedianBetween",
            column_name="col1",
            min_value=25,
            max_value=30,
        )
        result = expectation.validate(data_frame=data_frame)
        expected_failure = DataFrameExpectationFailureMessage(
            expectation_str=str(expectation),
            data_frame_type=DataFrameType.PYSPARK,
            message=expected_message,
        )
        assert str(result) == str(expected_failure), (
            f"Registry test failed for {description}: expected failure message but got {result}"
        )

        # Test through suite
        suite = DataFrameExpectationsSuite().expect_column_median_between(
            column_name="col1", min_value=25, max_value=30
        )
        with pytest.raises(DataFrameExpectationsSuiteFailure):
            suite.build().run(data_frame=data_frame)


def test_pyspark_missing_column_registry_and_suite(spark):
    """Test missing column error for PySpark DataFrames through both registry and suite."""
    data_frame = spark.createDataFrame([(20,), (25,), (30,), (35,)], ["col1"])
    expected_message = "Column 'nonexistent_col' does not exist in the DataFrame."

    # Test through registry
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationColumnMedianBetween",
        column_name="nonexistent_col",
        min_value=25,
        max_value=30,
    )
    result = expectation.validate(data_frame=data_frame)
    expected_failure = DataFrameExpectationFailureMessage(
        expectation_str=str(expectation),
        data_frame_type=DataFrameType.PYSPARK,
        message=expected_message,
    )
    assert str(result) == str(expected_failure), f"Expected failure message but got: {result}"

    # Test through suite
    suite = DataFrameExpectationsSuite().expect_column_median_between(
        column_name="nonexistent_col", min_value=25, max_value=30
    )
    with pytest.raises(DataFrameExpectationsSuiteFailure):
        suite.build().run(data_frame=data_frame)


def test_boundary_values_both_dataframes(spark):
    """Test boundary values for both pandas and PySpark DataFrames."""
    test_data = [20, 25, 30, 35]  # median = 27.5

    # Test boundary scenarios
    boundary_tests = [
        (27.5, 30, "exact minimum boundary"),  # median exactly at min
        (25, 27.5, "exact maximum boundary"),  # median exactly at max
    ]

    for min_val, max_val, boundary_desc in boundary_tests:
        for df_type, data_frame in [
            ("pandas", pd.DataFrame({"col1": test_data})),
            (
                "pyspark",
                spark.createDataFrame([(val,) for val in test_data], ["col1"]),
            ),
        ]:
            expectation = DataFrameExpectationRegistry.get_expectation(
                expectation_name="ExpectationColumnMedianBetween",
                column_name="col1",
                min_value=min_val,
                max_value=max_val,
            )
            result = expectation.validate(data_frame=data_frame)
            assert isinstance(result, DataFrameExpectationSuccessMessage), (
                f"Boundary test failed for {df_type} with {boundary_desc}: expected success but got {type(result)}"
            )


def test_median_calculation_specifics(spark):
    """Test median calculation specifics for odd vs even number of elements."""
    median_scenarios = [
        # (data, expected_median, description)
        ([1, 2, 3], 2, "odd count - middle element"),
        ([1, 2, 3, 4], 2.5, "even count - average of middle two"),
        ([5], 5, "single element"),
        ([10, 10, 10], 10, "all identical values"),
        ([1, 100], 50.5, "two elements - average"),
        ([1, 2, 100], 2, "odd count with outlier"),
        ([1, 2, 99, 100], 50.5, "even count with outliers"),
    ]

    for data, expected_median, description in median_scenarios:
        # Set bounds around expected median
        min_val = expected_median - 0.1
        max_val = expected_median + 0.1

        # Test pandas
        data_frame = pd.DataFrame({"col1": data})
        expectation = DataFrameExpectationRegistry.get_expectation(
            expectation_name="ExpectationColumnMedianBetween",
            column_name="col1",
            min_value=min_val,
            max_value=max_val,
        )
        result = expectation.validate(data_frame=data_frame)
        assert isinstance(result, DataFrameExpectationSuccessMessage), (
            f"Pandas median test failed for {description}: expected success but got {type(result)}"
        )

        # Test PySpark (for non-single element cases)
        if len(data) > 1:
            pyspark_df = spark.createDataFrame([(val,) for val in data], ["col1"])
            result_pyspark = expectation.validate(data_frame=pyspark_df)
            assert isinstance(result_pyspark, DataFrameExpectationSuccessMessage), (
                f"PySpark median test failed for {description}: expected success but got {type(result_pyspark)}"
            )


def test_precision_handling():
    """Test median calculation precision with various numeric types."""
    # Test scenarios with different levels of precision
    precision_tests = [
        # (data, description)
        ([1.1111, 2.2222, 3.3333], "high precision decimals"),
        ([0.1, 0.2, 0.3, 0.4, 0.5], "decimal sequence"),
        ([1e-6, 2e-6, 3e-6, 4e-6, 5e-6], "scientific notation"),
        ([1.0, 1.5, 2.0, 2.5, 3.0], "half increments"),
    ]

    for data, description in precision_tests:
        data_frame = pd.DataFrame({"col1": data})
        import numpy as np

        calculated_median = np.median(data)

        # Use a small range around the calculated median
        min_val = calculated_median - 0.001
        max_val = calculated_median + 0.001

        expectation = DataFrameExpectationRegistry.get_expectation(
            expectation_name="ExpectationColumnMedianBetween",
            column_name="col1",
            min_value=min_val,
            max_value=max_val,
        )
        result = expectation.validate(data_frame=data_frame)
        assert isinstance(result, DataFrameExpectationSuccessMessage), (
            f"Precision test failed for {description}: expected success but got {type(result)}"
        )


def test_suite_chaining():
    """Test that the suite method returns self for method chaining."""
    suite = DataFrameExpectationsSuite()
    result = suite.expect_column_median_between(column_name="col1", min_value=25, max_value=30)
    assert result is suite, f"Expected suite chaining to return same instance but got: {result}"


def test_large_dataset_performance():
    """Test the expectation with a larger dataset to ensure performance."""
    import numpy as np

    # Create a larger dataset with median around 50
    large_data = np.random.normal(50, 10, 1001).tolist()  # Use odd count for deterministic median
    data_frame = pd.DataFrame({"col1": large_data})

    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationColumnMedianBetween",
        column_name="col1",
        min_value=45,
        max_value=55,
    )

    result = expectation.validate(data_frame=data_frame)
    # Should succeed as the median of normal(50, 10) should be around 50
    assert isinstance(result, DataFrameExpectationSuccessMessage)


def test_outlier_resistance(spark):
    """Test that median is resistant to outliers (unlike mean)."""
    # Test data where median is stable despite extreme outliers
    outlier_scenarios = [
        # (data, min_val, max_val, description)
        (
            [1, 2, 3, 1000],
            1.5,
            2.5,
            "high outlier doesn't affect median",
        ),  # median = 2.5
        (
            [-1000, 10, 20, 30],
            14,
            16,
            "low outlier doesn't affect median",
        ),  # median = 15
        (
            [1, 2, 3, 4, 5, 1000000],
            2.5,
            3.5,
            "extreme outlier ignored",
        ),  # median = 3.5
        (
            [-1000000, 1, 2, 3, 4, 5],
            2.5,
            3.5,
            "extreme negative outlier ignored",
        ),  # median = 2.5
    ]

    for data, min_val, max_val, description in outlier_scenarios:
        # Test with pandas
        data_frame = pd.DataFrame({"col1": data})
        expectation = DataFrameExpectationRegistry.get_expectation(
            expectation_name="ExpectationColumnMedianBetween",
            column_name="col1",
            min_value=min_val,
            max_value=max_val,
        )
        result = expectation.validate(data_frame=data_frame)
        assert isinstance(result, DataFrameExpectationSuccessMessage), (
            f"Pandas outlier test failed for {description}: expected success but got {type(result)}"
        )

        # Test with PySpark
        pyspark_df = spark.createDataFrame([(val,) for val in data], ["col1"])
        result_pyspark = expectation.validate(data_frame=pyspark_df)
        assert isinstance(result_pyspark, DataFrameExpectationSuccessMessage), (
            f"PySpark outlier test failed for {description}: expected success but got {type(result_pyspark)}"
        )
