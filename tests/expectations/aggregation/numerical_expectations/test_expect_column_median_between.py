import pytest
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
    "df_type, data, min_value, max_value, should_succeed, expected_message",
    [
        # Basic success scenarios
        ("pandas", [20, 25, 30, 35], 25, 30, True, None),  # median = 27.5
        ("pyspark", [20, 25, 30, 35], 25, 30, True, None),  # median ≈ 27.5
        # Single row scenarios
        ("pandas", [25], 20, 30, True, None),  # median = 25
        ("pyspark", [25], 20, 30, True, None),  # median = 25
        # Negative value scenarios
        ("pandas", [-20, -15, -10, -5], -15, -10, True, None),  # median = -12.5
        ("pyspark", [-20, -15, -10, -5], -15, -10, True, None),  # median ≈ -12.5
        # Float value scenarios
        ("pandas", [1.1, 2.5, 3.7, 3.8], 2.5, 3.5, True, None),  # median = 3.1
        ("pyspark", [1.1, 2.5, 3.7, 3.8], 2.5, 3.5, True, None),  # median ≈ 3.1
        # Identical value scenarios
        ("pandas", [25, 25, 25, 25], 24, 26, True, None),  # median = 25
        ("pyspark", [25, 25, 25, 25], 24, 26, True, None),  # median = 25
        # Mixed type scenarios
        ("pandas", [20, 25.5, 30, 37], 27, 29, True, None),  # median = 27.75
        ("pyspark", [20, 25.5, 30, 37], 27, 29, True, None),  # median ≈ 27.75
        # Zero scenarios
        ("pandas", [-5, 0, 0, 5], -1, 1, True, None),  # median = 0
        ("pyspark", [-5, 0, 0, 5], -1, 1, True, None),  # median = 0
        # Null scenarios
        ("pandas", [20, None, 30, None, 40], 25, 35, True, None),  # median = 30
        ("pyspark", [20, None, 30, None, 40], 25, 35, True, None),  # median ≈ 30
        # Odd count scenarios
        ("pandas", [10, 20, 30], 19, 21, True, None),  # median = 20
        ("pyspark", [10, 20, 30], 19, 21, True, None),  # median ≈ 20
        # Even count scenarios
        ("pandas", [10, 20, 30, 40], 24, 26, True, None),  # median = 25
        ("pyspark", [10, 20, 30, 40], 24, 26, True, None),  # median ≈ 25
        # Boundary scenarios - exact minimum
        ("pandas", [20, 25, 30, 35], 27.5, 30, True, None),  # median = 27.5
        ("pyspark", [20, 25, 30, 35], 27.5, 30, True, None),  # median = 27.5
        # Boundary scenarios - exact maximum
        ("pandas", [20, 25, 30, 35], 25, 27.5, True, None),  # median = 27.5
        ("pyspark", [20, 25, 30, 35], 25, 27.5, True, None),  # median = 27.5
        # Median calculation - odd count (middle element)
        ("pandas", [1, 2, 3], 1.9, 2.1, True, None),
        ("pyspark", [1, 2, 3], 1.9, 2.1, True, None),
        # Median calculation - even count (average of middle two)
        ("pandas", [1, 2, 3, 4], 2.4, 2.6, True, None),
        ("pyspark", [1, 2, 3, 4], 2.4, 2.6, True, None),
        # Median calculation - single element
        ("pandas", [5], 4.9, 5.1, True, None),
        ("pyspark", [5], 4.9, 5.1, True, None),
        # Median calculation - all identical values
        ("pandas", [10, 10, 10], 9.9, 10.1, True, None),
        ("pyspark", [10, 10, 10], 9.9, 10.1, True, None),
        # Median calculation - two elements (average)
        ("pandas", [1, 100], 50.4, 50.6, True, None),
        ("pyspark", [1, 100], 50.4, 50.6, True, None),
        # Median calculation - odd count with outlier
        ("pandas", [1, 2, 100], 1.9, 2.1, True, None),
        ("pyspark", [1, 2, 100], 1.9, 2.1, True, None),
        # Median calculation - even count with outliers
        ("pandas", [1, 2, 99, 100], 50.4, 50.6, True, None),
        ("pyspark", [1, 2, 99, 100], 50.4, 50.6, True, None),
        # Outlier resistance - high outlier
        ("pandas", [1, 2, 3, 1000], 1.5, 2.5, True, None),
        ("pyspark", [1, 2, 3, 1000], 1.5, 2.5, True, None),
        # Outlier resistance - low outlier
        ("pandas", [-1000, 10, 20, 30], 14, 16, True, None),
        ("pyspark", [-1000, 10, 20, 30], 14, 16, True, None),
        # Outlier resistance - extreme high outlier
        ("pandas", [1, 2, 3, 4, 5, 1000000], 2.5, 3.5, True, None),
        ("pyspark", [1, 2, 3, 4, 5, 1000000], 2.5, 3.5, True, None),
        # Outlier resistance - extreme low outlier
        ("pandas", [-1000000, 1, 2, 3, 4, 5], 2.4, 2.6, True, None),
        ("pyspark", [-1000000, 1, 2, 3, 4, 5], 2.4, 2.6, True, None),
        # Failure scenarios - median too low
        (
            "pandas",
            [20, 25, 30, 35],
            30,
            35,
            False,
            "Column 'col1' median value 27.5 is not between 30 and 35.",
        ),
        (
            "pyspark",
            [20, 25, 30, 35],
            30,
            35,
            False,
            "Column 'col1' median value 27.5 is not between 30 and 35.",
        ),
        # Failure scenarios - median too high
        (
            "pandas",
            [20, 25, 30, 35],
            20,
            25,
            False,
            "Column 'col1' median value 27.5 is not between 20 and 25.",
        ),
        (
            "pyspark",
            [20, 25, 30, 35],
            20,
            25,
            False,
            "Column 'col1' median value 27.5 is not between 20 and 25.",
        ),
        # Failure scenarios - median out of range
        (
            "pandas",
            [10, 20, 30],
            25,
            30,
            False,
            "Column 'col1' median value 20.0 is not between 25 and 30.",
        ),
        (
            "pyspark",
            [10, 20, 30],
            25,
            30,
            False,
            "Column 'col1' median value 20.0 is not between 25 and 30.",
        ),
        # Failure scenarios - all nulls
        ("pandas", [None, None, None], 25, 30, False, "Column 'col1' contains only null values."),
        ("pyspark", [None, None, None], 25, 30, False, "Column 'col1' contains only null values."),
        # Failure scenarios - empty
        ("pandas", [], 25, 30, False, "Column 'col1' contains only null values."),
        ("pyspark", [], 25, 30, False, "Column 'col1' contains only null values."),
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
        "pandas_odd_count",
        "pyspark_odd_count",
        "pandas_even_count",
        "pyspark_even_count",
        "pandas_boundary_exact_min",
        "pyspark_boundary_exact_min",
        "pandas_boundary_exact_max",
        "pyspark_boundary_exact_max",
        "pandas_calc_odd_count",
        "pyspark_calc_odd_count",
        "pandas_calc_even_count",
        "pyspark_calc_even_count",
        "pandas_calc_single_element",
        "pyspark_calc_single_element",
        "pandas_calc_all_identical",
        "pyspark_calc_all_identical",
        "pandas_calc_two_elements",
        "pyspark_calc_two_elements",
        "pandas_calc_odd_with_outlier",
        "pyspark_calc_odd_with_outlier",
        "pandas_calc_even_with_outliers",
        "pyspark_calc_even_with_outliers",
        "pandas_outlier_high",
        "pyspark_outlier_high",
        "pandas_outlier_low",
        "pyspark_outlier_low",
        "pandas_outlier_extreme_high",
        "pyspark_outlier_extreme_high",
        "pandas_outlier_extreme_low",
        "pyspark_outlier_extreme_low",
        "pandas_median_too_low",
        "pyspark_median_too_low",
        "pandas_median_too_high",
        "pyspark_median_too_high",
        "pandas_median_out_of_range",
        "pyspark_median_out_of_range",
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
        data_frame_type=str(df_type),
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
