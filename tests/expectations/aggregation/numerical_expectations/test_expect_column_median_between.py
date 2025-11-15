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
        expectation_name="ExpectationColumnMedianBetween",
        column_name="test_col",
        min_value=10,
        max_value=20,
    )
    # Note: median expectation delegates to quantile expectation
    assert expectation.get_expectation_name() == "ExpectationColumnQuantileBetween", (
        f"Expected 'ExpectationColumnQuantileBetween' but got: {expectation.get_expectation_name()}"
    )


def test_expectation_description():
    """Test that the expectation description is correctly returned."""
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationColumnMedianBetween",
        column_name="test_col",
        min_value=10,
        max_value=20,
    )
    description = expectation.get_description()
    assert "median" in description, f"Expected 'median' in description: {description}"
    assert "test_col" in description, f"Expected 'test_col' in description: {description}"
    assert "10" in description, f"Expected '10' in description: {description}"
    assert "20" in description, f"Expected '20' in description: {description}"
    # Verify quantile properties
    assert expectation.quantile == 0.5, (
        f"Expected quantile to be 0.5 but got: {expectation.quantile}"
    )
    assert expectation.quantile_desc == "median", (
        f"Expected quantile_desc to be 'median' but got: {expectation.quantile_desc}"
    )


@pytest.mark.parametrize(
    "df_type,data,min_value,max_value,should_succeed,expected_message",
    [
        # Pandas success scenarios
        ("pandas", [20, 25, 30, 35], 25, 30, True, None),  # median = 27.5, basic success
        ("pandas", [25], 20, 30, True, None),  # median = 25, single row
        ("pandas", [-20, -15, -10, -5], -15, -10, True, None),  # median = -12.5, negative values
        ("pandas", [1.1, 2.5, 3.7, 3.8], 2.5, 3.5, True, None),  # median = 3.1, float values
        ("pandas", [25, 25, 25, 25], 24, 26, True, None),  # median = 25, identical values
        ("pandas", [20, 25.5, 30, 37], 27, 29, True, None),  # median = 27.75, mixed data types
        ("pandas", [-5, 0, 0, 5], -1, 1, True, None),  # median = 0, with zeros
        ("pandas", [20, None, 30, None, 40], 25, 35, True, None),  # median = 30, with nulls
        ("pandas", [10, 20, 30], 19, 21, True, None),  # median = 20, odd count
        ("pandas", [10, 20, 30, 40], 24, 26, True, None),  # median = 25, even count
        # PySpark success scenarios
        ("pyspark", [20, 25, 30, 35], 25, 30, True, None),  # median ≈ 27.5, basic success
        ("pyspark", [25], 20, 30, True, None),  # median = 25, single row
        ("pyspark", [-20, -15, -10, -5], -15, -10, True, None),  # median ≈ -12.5, negative values
        ("pyspark", [20, None, 30, None, 40], 25, 35, True, None),  # median ≈ 30, with nulls
        ("pyspark", [10, 20, 30], 19, 21, True, None),  # median ≈ 20, odd count
        ("pyspark", [10, 20, 30, 40], 24, 26, True, None),  # median ≈ 25, even count
        # Boundary scenarios
        ("pandas", [20, 25, 30, 35], 27.5, 30, True, None),  # median = 27.5, exact minimum
        ("pandas", [20, 25, 30, 35], 25, 27.5, True, None),  # median = 27.5, exact maximum
        ("pyspark", [20, 25, 30, 35], 27.5, 30, True, None),  # median = 27.5, exact minimum
        ("pyspark", [20, 25, 30, 35], 25, 27.5, True, None),  # median = 27.5, exact maximum
        # Median calculation specifics
        ("pandas", [1, 2, 3], 1.9, 2.1, True, None),  # odd count - middle element
        ("pandas", [1, 2, 3, 4], 2.4, 2.6, True, None),  # even count - average of middle two
        ("pandas", [5], 4.9, 5.1, True, None),  # single element
        ("pandas", [10, 10, 10], 9.9, 10.1, True, None),  # all identical values
        ("pandas", [1, 100], 50.4, 50.6, True, None),  # two elements - average
        ("pandas", [1, 2, 100], 1.9, 2.1, True, None),  # odd count with outlier
        ("pandas", [1, 2, 99, 100], 50.4, 50.6, True, None),  # even count with outliers
        ("pyspark", [1, 2, 3], 1.9, 2.1, True, None),  # odd count
        ("pyspark", [1, 2, 3, 4], 2.4, 2.6, True, None),  # even count
        ("pyspark", [10, 10, 10], 9.9, 10.1, True, None),  # all identical values
        ("pyspark", [1, 100], 50.4, 50.6, True, None),  # two elements - average
        ("pyspark", [1, 2, 100], 1.9, 2.1, True, None),  # odd count with outlier
        ("pyspark", [1, 2, 99, 100], 50.4, 50.6, True, None),  # even count with outliers
        # Outlier resistance scenarios
        ("pandas", [1, 2, 3, 1000], 1.5, 2.5, True, None),  # high outlier doesn't affect median
        ("pandas", [-1000, 10, 20, 30], 14, 16, True, None),  # low outlier doesn't affect median
        ("pandas", [1, 2, 3, 4, 5, 1000000], 2.5, 3.5, True, None),  # extreme outlier ignored
        (
            "pandas",
            [-1000000, 1, 2, 3, 4, 5],
            2.4,
            2.6,
            True,
            None,
        ),  # extreme negative outlier ignored
        ("pyspark", [1, 2, 3, 1000], 1.5, 2.5, True, None),  # high outlier
        ("pyspark", [-1000, 10, 20, 30], 14, 16, True, None),  # low outlier
        ("pyspark", [1, 2, 3, 4, 5, 1000000], 2.5, 3.5, True, None),  # extreme outlier
        ("pyspark", [-1000000, 1, 2, 3, 4, 5], 2.4, 2.6, True, None),  # extreme negative outlier
        # Pandas failure scenarios
        (
            "pandas",
            [20, 25, 30, 35],
            30,
            35,
            False,
            "Column 'col1' median value 27.5 is not between 30 and 35.",
        ),
        (
            "pandas",
            [20, 25, 30, 35],
            20,
            25,
            False,
            "Column 'col1' median value 27.5 is not between 20 and 25.",
        ),
        (
            "pandas",
            [10, 20, 30],
            25,
            30,
            False,
            "Column 'col1' median value 20.0 is not between 25 and 30.",
        ),
        ("pandas", [None, None, None], 25, 30, False, "Column 'col1' contains only null values."),
        ("pandas", [], 25, 30, False, "Column 'col1' contains only null values."),
        # PySpark failure scenarios
        (
            "pyspark",
            [20, 25, 30, 35],
            30,
            35,
            False,
            "Column 'col1' median value 27.5 is not between 30 and 35.",
        ),
        (
            "pyspark",
            [20, 25, 30, 35],
            20,
            25,
            False,
            "Column 'col1' median value 27.5 is not between 20 and 25.",
        ),
        (
            "pyspark",
            [10, 20, 30],
            25,
            30,
            False,
            "Column 'col1' median value 20.0 is not between 25 and 30.",
        ),
        ("pyspark", [None, None, None], 25, 30, False, "Column 'col1' contains only null values."),
        ("pyspark", [], 25, 30, False, "Column 'col1' contains only null values."),
    ],
)
def test_expectation_basic_scenarios(
    df_type, data, min_value, max_value, should_succeed, expected_message, spark
):
    """Test basic expectation scenarios for both pandas and PySpark DataFrames."""
    df = create_dataframe(df_type, data, "col1", spark)

    # Test through registry
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationColumnMedianBetween",
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
    suite = DataFrameExpectationsSuite().expect_column_median_between(
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
        expectation_name="ExpectationColumnMedianBetween",
        column_name="nonexistent_col",
        min_value=25,
        max_value=30,
    )
    result = expectation.validate(data_frame=df)
    expected_failure = DataFrameExpectationFailureMessage(
        expectation_str=str(expectation),
        data_frame_type=get_df_type_enum(df_type),
        message=expected_message,
    )
    assert str(result) == str(expected_failure)

    # Test through suite
    suite = DataFrameExpectationsSuite().expect_column_median_between(
        column_name="nonexistent_col", min_value=25, max_value=30
    )
    with pytest.raises(DataFrameExpectationsSuiteFailure):
        suite.build().run(data_frame=df)


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
