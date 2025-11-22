import pytest
import numpy as np
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
        # Quantile 0.0 (minimum) scenarios
        ("pandas", [20, 25, 30, 35], 0.0, 15, 25, True, None),  # min = 20
        ("pyspark", [20, 25, 30, 35], 0.0, 15, 25, True, None),  # min = 20
        # Quantile 1.0 (maximum) scenarios
        ("pandas", [20, 25, 30, 35], 1.0, 30, 40, True, None),  # max = 35
        ("pyspark", [20, 25, 30, 35], 1.0, 30, 40, True, None),  # max = 35
        # Quantile 0.5 (median) scenarios
        ("pandas", [20, 25, 30, 35], 0.5, 25, 30, True, None),  # median = 27.5
        ("pyspark", [20, 25, 30, 35], 0.5, 25, 30, True, None),  # median ≈ 27.5
        # Quantile 0.25 (25th percentile) scenarios
        ("pandas", [20, 25, 30, 35], 0.25, 20, 25, True, None),  # 25th percentile = 22.5
        ("pyspark", [20, 25, 30, 35], 0.25, 20, 25, True, None),  # 25th percentile ≈ 22.5
        # Other quantile scenarios
        ("pandas", [10, 20, 30, 40, 50], 0.33, 20, 30, True, None),  # 33rd percentile ≈ 23.2
        ("pyspark", [20, 25, 30, 35], 0.9, 30, 40, True, None),  # 90th percentile ≈ 34
        # Single row scenarios
        ("pandas", [25], 0.5, 20, 30, True, None),  # median = 25
        ("pyspark", [25], 0.5, 20, 30, True, None),  # median = 25
        # Null scenarios
        ("pandas", [20, None, 25, None, 30], 0.5, 20, 30, True, None),  # median = 25
        ("pyspark", [20, None, 25, None, 30], 0.5, 20, 30, True, None),  # median ≈ 25
        # Failure scenarios - minimum (quantile 0.0)
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
            "pyspark",
            [20, 25, 30, 35],
            0.0,
            25,
            35,
            False,
            "Column 'col1' minimum value 20 is not between 25 and 35.",
        ),
        # Failure scenarios - maximum (quantile 1.0)
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
            "pyspark",
            [20, 25, 30, 35],
            1.0,
            40,
            50,
            False,
            "Column 'col1' maximum value 35 is not between 40 and 50.",
        ),
        # Failure scenarios - median (quantile 0.5)
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
            "pyspark",
            [20, 25, 30, 35],
            0.5,
            30,
            35,
            False,
            f"Column 'col1' median value {np.median([20, 25, 30, 35])} is not between 30 and 35.",
        ),
        # Failure scenarios - 75th percentile
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
            "pyspark",
            [20, 25, 30, 35],
            0.75,
            32,
            40,
            False,
            "Column 'col1' 75th percentile value 30 is not between 32 and 40.",
        ),
        # Failure scenarios - all nulls
        (
            "pandas",
            [None, None, None],
            0.5,
            20,
            30,
            False,
            "Column 'col1' contains only null values.",
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
        # Failure scenarios - empty
        ("pandas", [], 0.5, 20, 30, False, "Column 'col1' contains only null values."),
        ("pyspark", [], 0.5, 20, 30, False, "Column 'col1' contains only null values."),
    ],
    ids=[
        "pandas_quantile_0_min",
        "pyspark_quantile_0_min",
        "pandas_quantile_1_max",
        "pyspark_quantile_1_max",
        "pandas_quantile_0_5_median",
        "pyspark_quantile_0_5_median",
        "pandas_quantile_0_25",
        "pyspark_quantile_0_25",
        "pandas_quantile_0_33",
        "pyspark_quantile_0_9",
        "pandas_single_row",
        "pyspark_single_row",
        "pandas_with_nulls",
        "pyspark_with_nulls",
        "pandas_fail_min_too_low",
        "pyspark_fail_min_too_low",
        "pandas_fail_max_too_low",
        "pyspark_fail_max_too_low",
        "pandas_fail_median_too_low",
        "pyspark_fail_median_too_low",
        "pandas_fail_75th_percentile",
        "pyspark_fail_75th_percentile",
        "pandas_all_nulls",
        "pyspark_all_nulls",
        "pandas_empty",
        "pyspark_empty",
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
        assert suite_result is not None, "Expected SuiteExecutionResult"
        assert suite_result.success, "Expected all expectations to pass"
    else:
        with pytest.raises(DataFrameExpectationsSuiteFailure):
            suite.build().run(data_frame=df)


@pytest.mark.parametrize(
    "df_type",
    ["pandas", "pyspark"],
    ids=["pandas", "pyspark"],
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
        expectation_name="ExpectationColumnQuantileBetween",
        column_name="nonexistent_col",
        quantile=0.5,
        min_value=25,
        max_value=30,
    )
    result = expectation.validate(data_frame=df)
    expected_failure = DataFrameExpectationFailureMessage(
        expectation_str=str(expectation),
        data_frame_type=str(df_type),
        message=expected_message,
    )
    assert str(result) == str(expected_failure)

    # Test through suite
    suite = DataFrameExpectationsSuite().expect_column_quantile_between(
        column_name="nonexistent_col", quantile=0.5, min_value=25, max_value=30
    )
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
