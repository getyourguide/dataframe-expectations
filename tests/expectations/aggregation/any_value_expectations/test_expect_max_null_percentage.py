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


def create_dataframe(df_type, df_data, spark):
    """Helper function to create pandas or pyspark DataFrame with explicit schema for PySpark."""
    if df_type == "pandas":
        return pd.DataFrame(df_data)

    # PySpark: requires explicit schema to handle all-null columns
    from pyspark.sql.types import StructType, StructField, IntegerType, StringType, DoubleType

    if not df_data:
        return spark.createDataFrame([], StructType([StructField("col1", IntegerType(), True)]))

    type_map = {str: StringType(), float: DoubleType()}

    def infer_type(values):
        """Infer PySpark type from first non-None value, default to IntegerType."""
        sample = next((v for v in values if v is not None), None)
        return type_map.get(type(sample), IntegerType())

    columns = list(df_data.keys())
    schema = StructType([StructField(col, infer_type(df_data[col]), True) for col in columns])
    # Transform from column-oriented {col1: [v1, v2], col2: [v3, v4]} to row-oriented [(v1, v3), (v2, v4)]
    rows = list(zip(*[df_data[col] for col in columns]))

    return spark.createDataFrame(rows, schema)


def get_df_type_enum(df_type):
    """Get DataFrameType enum value."""
    return DataFrameType.PANDAS if df_type == "pandas" else DataFrameType.PYSPARK


@pytest.mark.parametrize(
    "df_type, df_data, column_name, max_percentage, expected_result, expected_message",
    [
        # No nulls - should pass
        (
            "pandas",
            {"col1": [1, 2, 3, 4, 5], "col2": ["a", "b", "c", "d", "e"]},
            "col1",
            10.0,
            "success",
            None,
        ),
        (
            "pyspark",
            {"col1": [1, 2, 3, 4, 5], "col2": ["a", "b", "c", "d", "e"]},
            "col1",
            10.0,
            "success",
            None,
        ),
        # Within threshold - 25% < 30%
        ("pandas", {"col1": [1, None, 3, 4]}, "col1", 30.0, "success", None),
        ("pyspark", {"col1": [1, None, 3, 4]}, "col1", 30.0, "success", None),
        # Exactly at threshold - 20% == 20%
        (
            "pandas",
            {"col1": [1, 2, None, 4, 5], "col2": [None, "b", "c", "d", "e"]},
            "col1",
            20.0,
            "success",
            None,
        ),
        ("pyspark", {"col1": [1, 2, None, 4, 5]}, "col1", 20.0, "success", None),
        # With NaN - 33.33% < 50%
        ("pandas", {"col1": [1, 2, 3], "col2": [4.0, np.nan, 6.0]}, "col2", 50.0, "success", None),
        # Exceeds threshold - 50% > 20%
        (
            "pandas",
            {"col1": [1, None, 3, None], "col2": [None, "b", "c", "d"]},
            "col1",
            20.0,
            "failure",
            "Column 'col1' has 50.00% null values, expected at most 20.00%.",
        ),
        (
            "pyspark",
            {"col1": [1, None, None], "col2": ["a", "b", "c"]},
            "col1",
            25.0,
            "failure",
            "Column 'col1' has 66.67% null values, expected at most 25.00%.",
        ),
        # All nulls in column - 100% > 50%
        (
            "pandas",
            {"col1": [None, None], "col2": [1, 2]},
            "col1",
            50.0,
            "failure",
            "Column 'col1' has 100.00% null values, expected at most 50.00%.",
        ),
        (
            "pyspark",
            {"col1": [None, None, None]},
            "col1",
            75.0,
            "failure",
            "Column 'col1' has 100.00% null values, expected at most 75.00%.",
        ),
        # Zero threshold failure - 33.33% > 0%
        (
            "pandas",
            {"col1": [1, None, 3]},
            "col1",
            0.0,
            "failure",
            "Column 'col1' has 33.33% null values, expected at most 0.00%.",
        ),
        (
            "pyspark",
            {"col1": [1, None, 3]},
            "col1",
            0.0,
            "failure",
            "Column 'col1' has 33.33% null values, expected at most 0.00%.",
        ),
        # Hundred threshold success - 100% <= 100%
        (
            "pandas",
            {"col1": [None, None, None], "col2": [None, None, None]},
            "col1",
            100.0,
            "success",
            None,
        ),
        ("pyspark", {"col1": [None, None], "col2": [None, None]}, "col1", 100.0, "success", None),
        # Empty DataFrame - 0% <= 10%
        ("pandas", {"col1": []}, "col1", 10.0, "success", None),
        ("pyspark", {}, "col1", 10.0, "success", None),
        # Single null value - 100% > 50%
        (
            "pandas",
            {"col1": [None]},
            "col1",
            50.0,
            "failure",
            "Column 'col1' has 100.00% null values, expected at most 50.00%.",
        ),
        (
            "pyspark",
            {"col1": [None]},
            "col1",
            50.0,
            "failure",
            "Column 'col1' has 100.00% null values, expected at most 50.00%.",
        ),
        # Single non-null value - 0% <= 10%
        ("pandas", {"col1": [1]}, "col1", 10.0, "success", None),
        ("pyspark", {"col1": [1]}, "col1", 10.0, "success", None),
        # Other columns with nulls don't affect - 0% in col1 <= 10%
        ("pandas", {"col1": [1, 2, 3], "col2": [None, None, None]}, "col1", 10.0, "success", None),
        ("pyspark", {"col1": [1, 2, 3], "col2": [None, None, None]}, "col1", 10.0, "success", None),
        # Mixed data types with nulls - 25% < 50%
        (
            "pandas",
            {
                "int_col": [1, None, 3, 4],
                "str_col": ["a", "b", None, "d"],
                "float_col": [1.1, 2.2, 3.3, np.nan],
            },
            "float_col",
            50.0,
            "success",
            None,
        ),
        # Precision boundary - 25% == 25%
        ("pandas", {"col1": [1, None, 3, 4]}, "col1", 25.0, "success", None),
    ],
    ids=[
        "pandas_no_nulls",
        "pyspark_no_nulls",
        "pandas_within_threshold",
        "pyspark_within_threshold",
        "pandas_exactly_at_threshold",
        "pyspark_exactly_at_threshold",
        "pandas_with_nan",
        "pandas_exceeds_threshold",
        "pyspark_exceeds_threshold",
        "pandas_all_nulls",
        "pyspark_all_nulls",
        "pandas_zero_threshold_failure",
        "pyspark_zero_threshold_failure",
        "pandas_hundred_threshold_success",
        "pyspark_hundred_threshold_success",
        "pandas_empty",
        "pyspark_empty",
        "pandas_single_null",
        "pyspark_single_null",
        "pandas_single_not_null",
        "pyspark_single_not_null",
        "pandas_other_columns_ignored",
        "pyspark_other_columns_ignored",
        "pandas_mixed_data_types",
        "pandas_precision_boundary",
    ],
)
def test_expectation_basic_scenarios(
    df_type, df_data, column_name, max_percentage, expected_result, expected_message, spark
):
    """
    Test the expectation for various scenarios across pandas and PySpark DataFrames.
    Tests both direct expectation validation and suite-based validation.
    Covers: success cases, failures (exceeds threshold), edge cases (empty, zero/hundred threshold, single values),
    boundary conditions, column isolation, mixed data types, and precision boundaries.
    """
    data_frame = create_dataframe(df_type, df_data, spark)

    # Test 1: Direct expectation validation
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationMaxNullPercentage",
        column_name=column_name,
        max_percentage=max_percentage,
    )

    result = expectation.validate(data_frame=data_frame)

    if expected_result == "success":
        assert str(result) == str(
            DataFrameExpectationSuccessMessage(expectation_name="ExpectationMaxNullPercentage")
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
    expectations_suite = DataFrameExpectationsSuite().expect_max_null_percentage(
        column_name=column_name, max_percentage=max_percentage
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
    """
    Test that an error is raised when the specified column is missing in both pandas and PySpark.
    Tests both direct expectation validation and suite-based validation.
    """
    # Test 1: Direct expectation validation
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationMaxNullPercentage",
        column_name="nonexistent_col",
        max_percentage=50.0,
    )

    if df_type == "pandas":
        data_frame = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
    else:  # pyspark
        data_frame = spark.createDataFrame([(1, 4), (2, 5), (3, 6)], ["col1", "col2"])

    result = expectation.validate(data_frame=data_frame)
    # The error message might vary, but should be a failure
    assert isinstance(result, DataFrameExpectationFailureMessage), (
        f"Expected DataFrameExpectationFailureMessage but got: {type(result)}"
    )
    result_str = str(result)
    assert "nonexistent_col" in result_str, (
        f"Expected 'nonexistent_col' in result message: {result_str}"
    )

    # Test 2: Suite-based validation
    expectations_suite = DataFrameExpectationsSuite().expect_max_null_percentage(
        column_name="nonexistent_col", max_percentage=50.0
    )
    with pytest.raises(DataFrameExpectationsSuiteFailure):
        expectations_suite.build().run(data_frame=data_frame)


def test_invalid_parameters():
    """Test that appropriate errors are raised for invalid parameters."""
    # Test negative max_percentage
    with pytest.raises(ValueError) as context:
        DataFrameExpectationRegistry.get_expectation(
            expectation_name="ExpectationMaxNullPercentage",
            column_name="col1",
            max_percentage=-1.0,
        )
    assert "max_percentage must be between" in str(context.value), (
        f"Expected 'max_percentage must be between' in error message: {str(context.value)}"
    )

    # Test max_percentage > 100
    with pytest.raises(ValueError) as context:
        DataFrameExpectationRegistry.get_expectation(
            expectation_name="ExpectationMaxNullPercentage",
            column_name="col1",
            max_percentage=101.0,
        )
    assert "max_percentage must be between" in str(context.value), (
        f"Expected 'max_percentage must be between' in error message: {str(context.value)}"
    )


def test_large_dataset_performance():
    """Test the expectation with a larger dataset to ensure reasonable performance."""
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationMaxNullPercentage",
        column_name="col1",
        max_percentage=10.0,
    )
    # Create a DataFrame with 1000 rows and 50 nulls (every 20th value is None) = 5% null
    data = [None if i % 20 == 0 else i for i in range(1000)]
    data_frame = pd.DataFrame({"col1": data})
    result = expectation.validate(data_frame=data_frame)
    assert str(result) == str(
        DataFrameExpectationSuccessMessage(expectation_name="ExpectationMaxNullPercentage")
    ), f"Expected success message but got: {result}"
