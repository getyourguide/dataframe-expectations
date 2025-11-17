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
        # Pandas success scenarios
        ("pandas", [20, 25, 30, 35], 15, 25, True, None),  # min = 20, basic success
        ("pandas", [25], 20, 30, True, None),  # min = 25, single row
        ("pandas", [-20, -15, -10, -5], -25, -15, True, None),  # min = -20, negative values
        ("pandas", [1.1, 2.5, 3.7, 3.8], 1.0, 1.5, True, None),  # min = 1.1, float values
        ("pandas", [25, 25, 25, 25], 24, 26, True, None),  # min = 25, identical values
        ("pandas", [20, 25.5, 30, 37], 15, 25, True, None),  # min = 20, mixed data types
        ("pandas", [-5, 0, 0, 2], -10, -1, True, None),  # min = -5, with zeros
        ("pandas", [20, None, 35, None, 25], 15, 25, True, None),  # min = 20, with nulls
        # PySpark success scenarios
        ("pyspark", [20, 25, 30, 35], 15, 25, True, None),  # min = 20, basic success
        ("pyspark", [25], 20, 30, True, None),  # min = 25, single row
        ("pyspark", [-20, -15, -10, -5], -25, -15, True, None),  # min = -20, negative values
        ("pyspark", [20, None, 35, None, 25], 15, 25, True, None),  # min = 20, with nulls
        # Boundary scenarios
        ("pandas", [20, 25, 30, 35], 20, 25, True, None),  # min = 20, exact minimum
        ("pandas", [20, 25, 30, 35], 15, 20, True, None),  # min = 20, exact maximum
        ("pyspark", [20, 25, 30, 35], 20, 25, True, None),  # min = 20, exact minimum
        ("pyspark", [20, 25, 30, 35], 15, 20, True, None),  # min = 20, exact maximum
        # Minimum calculation specifics
        ("pandas", [100, 50, 75, 25], 24, 26, True, None),  # min = 25, mixed order
        ("pandas", [0, 1, 2, 3], -0.1, 0.1, True, None),  # min = 0, minimum is zero
        ("pandas", [-10, -5, -1, -20], -20.1, -19.9, True, None),  # min = -20, with negatives
        ("pandas", [1.001, 1.002, 1.003], 1.0, 1.002, True, None),  # min = 1.001, small differences
        ("pandas", [1e6, 1e5, 1e4], 1e4 - 100, 1e4 + 100, True, None),  # min = 1e4, large numbers
        ("pandas", [1e-6, 1e-5, 1e-4], 1e-7, 1e-5, True, None),  # min = 1e-6, very small numbers
        ("pyspark", [100, 50, 75, 25], 24, 26, True, None),  # min = 25, mixed order
        ("pyspark", [0, 1, 2, 3], -0.1, 0.1, True, None),  # min = 0, minimum is zero
        ("pyspark", [-10, -5, -1, -20], -20.1, -19.9, True, None),  # min = -20, with negatives
        (
            "pyspark",
            [1.001, 1.002, 1.003],
            1.0,
            1.002,
            True,
            None,
        ),  # min = 1.001, small differences
        ("pyspark", [1e6, 1e5, 1e4], 1e4 - 100, 1e4 + 100, True, None),  # min = 1e4, large numbers
        ("pyspark", [1e-6, 1e-5, 1e-4], 1e-7, 1e-5, True, None),  # min = 1e-6, very small numbers
        # Outlier impact scenarios (minimum is sensitive to outliers)
        (
            "pandas",
            [1, 2, 3, -1000],
            -1100,
            -900,
            True,
            None,
        ),  # extreme low outlier becomes minimum
        (
            "pandas",
            [100, 200, 300, 50],
            40,
            60,
            True,
            None,
        ),  # outlier changes minimum significantly
        ("pandas", [1.5, 2.0, 2.5, 0.1], 0.05, 0.15, True, None),  # small outlier affects minimum
        (
            "pyspark",
            [1, 2, 3, -1000],
            -1100,
            -900,
            True,
            None,
        ),  # extreme low outlier becomes minimum
        (
            "pyspark",
            [100, 200, 300, 50],
            40,
            60,
            True,
            None,
        ),  # outlier changes minimum significantly
        ("pyspark", [1.5, 2.0, 2.5, 0.1], 0.05, 0.15, True, None),  # small outlier affects minimum
        # Identical value scenarios
        ("pandas", [42, 42, 42, 42], 41.9, 42.1, True, None),  # integer repetition
        ("pandas", [3.14, 3.14, 3.14], 3.13, 3.15, True, None),  # float repetition
        ("pandas", [-7, -7, -7, -7, -7], -7.1, -6.9, True, None),  # negative repetition
        ("pandas", [0, 0, 0], -0.1, 0.1, True, None),  # zero repetition
        ("pyspark", [42, 42, 42, 42], 41.9, 42.1, True, None),  # integer repetition
        ("pyspark", [3.14, 3.14, 3.14], 3.13, 3.15, True, None),  # float repetition
        ("pyspark", [-7, -7, -7, -7, -7], -7.1, -6.9, True, None),  # negative repetition
        ("pyspark", [0, 0, 0], -0.1, 0.1, True, None),  # zero repetition
        # Pandas failure scenarios
        (
            "pandas",
            [20, 25, 30, 35],
            25,
            35,
            False,
            "Column 'col1' minimum value 20 is not between 25 and 35.",
        ),
        (
            "pandas",
            [20, 25, 30, 35],
            10,
            15,
            False,
            "Column 'col1' minimum value 20 is not between 10 and 15.",
        ),
        ("pandas", [None, None, None], 15, 25, False, "Column 'col1' contains only null values."),
        ("pandas", [], 15, 25, False, "Column 'col1' contains only null values."),
        # PySpark failure scenarios
        (
            "pyspark",
            [20, 25, 30, 35],
            25,
            35,
            False,
            "Column 'col1' minimum value 20 is not between 25 and 35.",
        ),
        (
            "pyspark",
            [20, 25, 30, 35],
            10,
            15,
            False,
            "Column 'col1' minimum value 20 is not between 10 and 15.",
        ),
        ("pyspark", [None, None, None], 15, 25, False, "Column 'col1' contains only null values."),
        ("pyspark", [], 15, 25, False, "Column 'col1' contains only null values."),
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
        data_frame_type=get_df_type_enum(df_type),
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
