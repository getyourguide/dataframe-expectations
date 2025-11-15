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
        expectation_name="ExpectationColumnMaxBetween",
        column_name="col1",
        min_value=10,
        max_value=20,
    )
    assert expectation.get_expectation_name() == "ExpectationColumnQuantileBetween", (
        f"Expected 'ExpectationColumnQuantileBetween' but got: {expectation.get_expectation_name()}"
    )


def test_expectation_description():
    """Test that the expectation description is correctly returned."""
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationColumnMaxBetween",
        column_name="test_col",
        min_value=10,
        max_value=20,
    )
    description = expectation.get_description()
    assert "maximum" in description, f"Expected 'maximum' in description: {description}"
    assert "test_col" in description, f"Expected 'test_col' in description: {description}"
    assert "10" in description, f"Expected '10' in description: {description}"
    assert "20" in description, f"Expected '20' in description: {description}"


@pytest.mark.parametrize(
    "df_type, data, min_value, max_value, expected_result, expected_message",
    [
        # Success scenarios - pandas
        ("pandas", [20, 25, 30, 35], 30, 40, "success", None),
        ("pandas", [35], 30, 40, "success", None),
        ("pandas", [-20, -15, -10, -3], -5, 0, "success", None),
        ("pandas", [1.1, 2.5, 3.7, 3.8], 3.5, 4.0, "success", None),
        ("pandas", [25, 25, 25, 25], 24, 26, "success", None),
        ("pandas", [20, 25.5, 30, 37], 35, 40, "success", None),
        ("pandas", [-5, 0, 0, -2], -1, 1, "success", None),
        ("pandas", [20, None, 35, None, 25], 30, 40, "success", None),
        # Success scenarios - pyspark
        ("pyspark", [20, 25, 30, 35], 30, 40, "success", None),
        ("pyspark", [35], 30, 40, "success", None),
        ("pyspark", [-20, -15, -10, -3], -5, 0, "success", None),
        ("pyspark", [20, None, 35, None, 25], 30, 40, "success", None),
        # Boundary scenarios - pandas (max = 35)
        ("pandas", [20, 25, 30, 35], 35, 40, "success", None),  # exact min boundary
        ("pandas", [20, 25, 30, 35], 30, 35, "success", None),  # exact max boundary
        # Boundary scenarios - pyspark (max = 35)
        ("pyspark", [20, 25, 30, 35], 35, 40, "success", None),  # exact min boundary
        ("pyspark", [20, 25, 30, 35], 30, 35, "success", None),  # exact max boundary
        # Failure scenarios - pandas
        (
            "pandas",
            [20, 25, 30, 35],
            40,
            50,
            "failure",
            "Column 'col1' maximum value 35 is not between 40 and 50.",
        ),
        (
            "pandas",
            [None, None, None],
            30,
            40,
            "failure",
            "Column 'col1' contains only null values.",
        ),
        ("pandas", [], 30, 40, "failure", "Column 'col1' contains only null values."),
        # Failure scenarios - pyspark
        (
            "pyspark",
            [20, 25, 30, 35],
            40,
            50,
            "failure",
            "Column 'col1' maximum value 35 is not between 40 and 50.",
        ),
        (
            "pyspark",
            [None, None, None],
            30,
            40,
            "failure",
            "Column 'col1' contains only null values.",
        ),
        ("pyspark", [], 30, 40, "failure", "Column 'col1' contains only null values."),
    ],
    ids=[
        "pandas_basic_success",
        "pandas_single_row",
        "pandas_negative_values",
        "pandas_float_values",
        "pandas_identical_values",
        "pandas_mixed_types",
        "pandas_zero_values",
        "pandas_with_nulls",
        "pyspark_basic_success",
        "pyspark_single_row",
        "pyspark_negative_values",
        "pyspark_with_nulls",
        "pandas_boundary_exact_min",
        "pandas_boundary_exact_max",
        "pyspark_boundary_exact_min",
        "pyspark_boundary_exact_max",
        "pandas_max_below_range",
        "pandas_all_nulls",
        "pandas_empty",
        "pyspark_max_below_range",
        "pyspark_all_nulls",
        "pyspark_empty",
    ],
)
def test_expectation_basic_scenarios(
    df_type, data, min_value, max_value, expected_result, expected_message, spark
):
    """
    Test the expectation for various scenarios across pandas and PySpark DataFrames.
    Tests both direct expectation validation and suite-based validation.
    Covers: success cases, boundary conditions, failures (out of range, nulls, empty),
    and various data types (integers, floats, negatives, nulls, mixed types).
    """
    data_frame = create_dataframe(df_type, data, "col1", spark)

    # Test 1: Direct expectation validation
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationColumnMaxBetween",
        column_name="col1",
        min_value=min_value,
        max_value=max_value,
    )

    result = expectation.validate(data_frame=data_frame)

    if expected_result == "success":
        assert str(result) == str(
            DataFrameExpectationSuccessMessage(expectation_name="ExpectationColumnQuantileBetween")
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
    expectations_suite = DataFrameExpectationsSuite().expect_column_max_between(
        column_name="col1", min_value=min_value, max_value=max_value
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
    """Test that an error is raised when the specified column is missing in both pandas and PySpark."""
    expected_message = "Column 'nonexistent_col' does not exist in the DataFrame."

    if df_type == "pandas":
        data_frame = pd.DataFrame({"col1": [20, 25, 30, 35]})
    else:  # pyspark
        data_frame = spark.createDataFrame([(20,), (25,), (30,), (35,)], ["col1"])

    # Test 1: Direct expectation validation
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationColumnMaxBetween",
        column_name="nonexistent_col",
        min_value=30,
        max_value=40,
    )
    result = expectation.validate(data_frame=data_frame)
    expected_failure = DataFrameExpectationFailureMessage(
        expectation_str=str(expectation),
        data_frame_type=get_df_type_enum(df_type),
        message=expected_message,
    )
    assert str(result) == str(expected_failure), f"Expected failure message but got: {result}"

    # Test 2: Suite-based validation
    expectations_suite = DataFrameExpectationsSuite().expect_column_max_between(
        column_name="nonexistent_col", min_value=30, max_value=40
    )
    with pytest.raises(DataFrameExpectationsSuiteFailure):
        expectations_suite.build().run(data_frame=data_frame)


def test_large_dataset_performance():
    """Test the expectation with a larger dataset to ensure performance."""
    import numpy as np

    # Create a larger dataset with max around 60
    large_data = np.random.uniform(10, 60, 1000).tolist()
    data_frame = pd.DataFrame({"col1": large_data})

    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationColumnMaxBetween",
        column_name="col1",
        min_value=55,
        max_value=65,
    )

    result = expectation.validate(data_frame=data_frame)
    # Should succeed as the max of uniform(10, 60) should be around 60
    assert isinstance(result, DataFrameExpectationSuccessMessage), (
        f"Large dataset test failed: expected success but got {type(result)}"
    )
