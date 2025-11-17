import pytest
import numpy as np
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
        expectation_name="ExpectationColumnQuantileBetween",
        column_name="test_col",
        quantile=0.5,
        min_value=20,
        max_value=30,
    )
    assert expectation.get_expectation_name() == "ExpectationColumnQuantileBetween", (
        f"Expected 'ExpectationColumnQuantileBetween' but got: {expectation.get_expectation_name()}"
    )


def test_expectation_description():
    """Test that description messages are correct for different quantiles."""
    test_cases = [
        (0.0, "minimum"),
        (0.25, "25th percentile"),
        (0.5, "median"),
        (0.75, "75th percentile"),
        (1.0, "maximum"),
        (0.9, "0.9 quantile"),
    ]

    for quantile, expected_desc in test_cases:
        exp = DataFrameExpectationRegistry.get_expectation(
            expectation_name="ExpectationColumnQuantileBetween",
            column_name="test_col",
            quantile=quantile,
            min_value=10,
            max_value=20,
        )
        assert exp.quantile_desc == expected_desc, (
            f"Expected quantile_desc '{expected_desc}' for quantile {quantile} but got: {exp.quantile_desc}"
        )
        assert expected_desc in exp.get_description(), (
            f"Expected '{expected_desc}' in description: {exp.get_description()}"
        )


@pytest.mark.parametrize(
    "df_type, data, quantile, min_value, max_value, should_succeed, expected_message",
    [
        # Pandas success scenarios - various quantiles
        ("pandas", [20, 25, 30, 35], 0.0, 15, 25, True, None),  # min = 20
        ("pandas", [20, 25, 30, 35], 1.0, 30, 40, True, None),  # max = 35
        ("pandas", [20, 25, 30, 35], 0.5, 25, 30, True, None),  # median = 27.5
        ("pandas", [20, 25, 30, 35], 0.25, 20, 25, True, None),  # 25th percentile = 22.5
        ("pandas", [10, 20, 30, 40, 50], 0.33, 20, 30, True, None),  # 33rd percentile ≈ 23.2
        ("pandas", [25], 0.5, 20, 30, True, None),  # median = 25, single row
        ("pandas", [20, None, 25, None, 30], 0.5, 20, 30, True, None),  # median = 25, with nulls
        # PySpark success scenarios - various quantiles
        ("pyspark", [20, 25, 30, 35], 0.0, 15, 25, True, None),  # min = 20
        ("pyspark", [20, 25, 30, 35], 1.0, 30, 40, True, None),  # max = 35
        ("pyspark", [20, 25, 30, 35], 0.5, 25, 30, True, None),  # median ≈ 27.5
        ("pyspark", [20, 25, 30, 35], 0.9, 30, 40, True, None),  # 90th percentile ≈ 34
        ("pyspark", [25], 0.5, 20, 30, True, None),  # median = 25, single row
        ("pyspark", [20, None, 25, None, 30], 0.5, 20, 30, True, None),  # median ≈ 25, with nulls
        # Boundary quantile values
        ("pandas", [20, 25, 30, 35], 0.0, 15, 25, True, None),  # quantile = 0.0 (minimum)
        ("pandas", [20, 25, 30, 35], 1.0, 30, 40, True, None),  # quantile = 1.0 (maximum)
        # Pandas failure scenarios
        (
            "pandas",
            [20, 25, 30, 35],
            0.0,
            25,
            35,
            False,
            "Column 'col1' minimum value 20 is not between 25 and 35.",
        ),
        (
            "pandas",
            [20, 25, 30, 35],
            1.0,
            40,
            50,
            False,
            "Column 'col1' maximum value 35 is not between 40 and 50.",
        ),
        (
            "pandas",
            [20, 25, 30, 35],
            0.5,
            30,
            35,
            False,
            "Column 'col1' median value 27.5 is not between 30 and 35.",
        ),
        (
            "pandas",
            [20, 25, 30, 35],
            0.75,
            25,
            30,
            False,
            f"Column 'col1' 75th percentile value {np.quantile([20, 25, 30, 35], 0.75)} is not between 25 and 30.",
        ),
        (
            "pandas",
            [None, None, None],
            0.5,
            20,
            30,
            False,
            "Column 'col1' contains only null values.",
        ),
        ("pandas", [], 0.5, 20, 30, False, "Column 'col1' contains only null values."),
        # PySpark failure scenarios
        (
            "pyspark",
            [20, 25, 30, 35],
            0.0,
            25,
            35,
            False,
            "Column 'col1' minimum value 20 is not between 25 and 35.",
        ),
        (
            "pyspark",
            [20, 25, 30, 35],
            1.0,
            40,
            50,
            False,
            "Column 'col1' maximum value 35 is not between 40 and 50.",
        ),
        (
            "pyspark",
            [20, 25, 30, 35],
            0.5,
            30,
            35,
            False,
            f"Column 'col1' median value {np.median([20, 25, 30, 35])} is not between 30 and 35.",
        ),
        (
            "pyspark",
            [None, None, None],
            0.5,
            20,
            30,
            False,
            "Column 'col1' contains only null values.",
        ),
        ("pyspark", [], 0.5, 20, 30, False, "Column 'col1' contains only null values."),
    ],
)
def test_expectation_basic_scenarios(
    df_type, data, quantile, min_value, max_value, should_succeed, expected_message, spark
):
    """Test basic expectation scenarios for both pandas and PySpark DataFrames."""
    df = create_dataframe(df_type, data, "col1", spark)

    # Test through registry
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationColumnQuantileBetween",
        column_name="col1",
        quantile=quantile,
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
    suite = DataFrameExpectationsSuite().expect_column_quantile_between(
        column_name="col1", quantile=quantile, min_value=min_value, max_value=max_value
    )

    if should_succeed:
        suite_result = suite.build().run(data_frame=df)
        assert suite_result is None, f"Suite test expected None but got: {suite_result}"
    else:
        with pytest.raises(DataFrameExpectationsSuiteFailure):
            suite.build().run(data_frame=df)


def test_invalid_quantile_range():
    """Test that invalid quantile values raise ValueError."""
    invalid_quantiles = [
        (1.5, "greater than 1.0"),
        (-0.1, "less than 0.0"),
    ]

    for invalid_quantile, description in invalid_quantiles:
        with pytest.raises(ValueError) as context:
            DataFrameExpectationRegistry.get_expectation(
                expectation_name="ExpectationColumnQuantileBetween",
                column_name="col1",
                quantile=invalid_quantile,
                min_value=20,
                max_value=30,
            )
        assert "Quantile must be between 0.0 and 1.0" in str(context.value), (
            f"Expected quantile validation error for {description} but got: {str(context.value)}"
        )


def test_large_dataset_performance():
    """Test the expectation with a larger dataset to ensure performance."""
    # Create a larger dataset
    large_data = np.random.normal(50, 10, 1000).tolist()
    data_frame = pd.DataFrame({"col1": large_data})

    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationColumnQuantileBetween",
        column_name="col1",
        quantile=0.5,  # median
        min_value=45,
        max_value=55,
    )

    result = expectation.validate(data_frame=data_frame)
    # Should succeed as the median of normal(50, 10) should be around 50
    assert isinstance(result, DataFrameExpectationSuccessMessage), (
        f"Large dataset test failed: expected success but got {type(result)}"
    )
