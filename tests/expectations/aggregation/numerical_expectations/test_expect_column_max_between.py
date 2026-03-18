import pytest
import pandas as pd

from dataframe_expectations.registry import (
    DataFrameExpectationRegistry,
)
from dataframe_expectations.suite import (
    DataFrameExpectationsSuite,
    DataFrameExpectationsSuiteFailure,
)
from dataframe_expectations.core.suite_result import SuiteExecutionResult
from dataframe_expectations.result_message import (
    DataFrameExpectationFailureMessage,
    DataFrameExpectationSuccessMessage,
)


def create_pyspark_dataframe(data, column_name, spark):
    """Helper function to create a PySpark DataFrame."""
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
        # Basic success
        ("pandas", [20, 25, 30, 35], 30, 40, "success", None),
        # Single row
        ("pandas", [35], 30, 40, "success", None),
        # Negative values
        ("pandas", [-20, -15, -10, -3], -5, 0, "success", None),
        # Float values
        ("pandas", [1.1, 2.5, 3.7, 3.8], 3.5, 4.0, "success", None),
        # Identical values
        ("pandas", [25, 25, 25, 25], 24, 26, "success", None),
        # Mixed types
        ("pandas", [20, 25.5, 30, 37], 35, 40, "success", None),
        # Zero values
        ("pandas", [-5, 0, 0, -2], -1, 1, "success", None),
        # With nulls
        ("pandas", [20, None, 35, None, 25], 30, 40, "success", None),
        # Boundary exact min
        ("pandas", [20, 25, 30, 35], 35, 40, "success", None),
        # Boundary exact max
        ("pandas", [20, 25, 30, 35], 30, 35, "success", None),
        # Max below range
        (
            "pandas",
            [20, 25, 30, 35],
            40,
            50,
            "failure",
            "Column 'col1' maximum value 35 is not between 40 and 50.",
        ),
        # All nulls
        (
            "pandas",
            [None, None, None],
            30,
            40,
            "failure",
            "Column 'col1' contains only null values.",
        ),
        # Empty
        ("pandas", [], 30, 40, "failure", "Column 'col1' contains only null values."),
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
        "pandas_boundary_exact_min",
        "pandas_boundary_exact_max",
        "pandas_max_below_range",
        "pandas_all_nulls",
        "pandas_empty",
    ],
)
def test_expectation_basic_scenarios_pandas(
    df_type, data, min_value, max_value, expected_result, expected_message
):
    """
    Test the expectation for various scenarios across pandas DataFrames.
    Tests both direct expectation validation and suite-based validation.
    Covers: success cases, boundary conditions, failures (out of range, nulls, empty),
    and various data types (integers, floats, negatives, nulls, mixed types).
    """
    data_frame = pd.DataFrame({"col1": data})

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
            data_frame_type=str(df_type),
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
        assert result is not None, "Expected SuiteExecutionResult"
        assert isinstance(result, SuiteExecutionResult), "Result should be SuiteExecutionResult"
        assert result.success, "Expected all expectations to pass"
        assert result.total_passed == 1, "Expected 1 passed expectation"
        assert result.total_failed == 0, "Expected 0 failed expectations"
    else:  # failure
        with pytest.raises(DataFrameExpectationsSuiteFailure):
            expectations_suite.build().run(data_frame=data_frame)


@pytest.mark.pyspark
@pytest.mark.parametrize(
    "df_type, data, min_value, max_value, expected_result, expected_message",
    [
        # Basic success
        ("pyspark", [20, 25, 30, 35], 30, 40, "success", None),
        # Single row
        ("pyspark", [35], 30, 40, "success", None),
        # Negative values
        ("pyspark", [-20, -15, -10, -3], -5, 0, "success", None),
        # Float values
        ("pyspark", [1.1, 2.5, 3.7, 3.8], 3.5, 4.0, "success", None),
        # Identical values
        ("pyspark", [25, 25, 25, 25], 24, 26, "success", None),
        # Mixed types
        ("pyspark", [20, 25.5, 30, 37], 35, 40, "success", None),
        # Zero values
        ("pyspark", [-5, 0, 0, -2], -1, 1, "success", None),
        # With nulls
        ("pyspark", [20, None, 35, None, 25], 30, 40, "success", None),
        # Boundary exact min
        ("pyspark", [20, 25, 30, 35], 35, 40, "success", None),
        # Boundary exact max
        ("pyspark", [20, 25, 30, 35], 30, 35, "success", None),
        # Max below range
        (
            "pyspark",
            [20, 25, 30, 35],
            40,
            50,
            "failure",
            "Column 'col1' maximum value 35 is not between 40 and 50.",
        ),
        # All nulls
        (
            "pyspark",
            [None, None, None],
            30,
            40,
            "failure",
            "Column 'col1' contains only null values.",
        ),
        # Empty
        ("pyspark", [], 30, 40, "failure", "Column 'col1' contains only null values."),
    ],
    ids=[
        "pyspark_basic_success",
        "pyspark_single_row",
        "pyspark_negative_values",
        "pyspark_float_values",
        "pyspark_identical_values",
        "pyspark_mixed_types",
        "pyspark_zero_values",
        "pyspark_with_nulls",
        "pyspark_boundary_exact_min",
        "pyspark_boundary_exact_max",
        "pyspark_max_below_range",
        "pyspark_all_nulls",
        "pyspark_empty",
    ],
)
def test_expectation_basic_scenarios_pyspark(
    df_type, data, min_value, max_value, expected_result, expected_message, spark
):
    """
    Test the expectation for various scenarios across PySpark DataFrames.
    Tests both direct expectation validation and suite-based validation.
    Covers: success cases, boundary conditions, failures (out of range, nulls, empty),
    and various data types (integers, floats, negatives, nulls, mixed types).
    """
    data_frame = create_pyspark_dataframe(data, "col1", spark)

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
            data_frame_type=str(df_type),
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
        assert result is not None, "Expected SuiteExecutionResult"
        assert isinstance(result, SuiteExecutionResult), "Result should be SuiteExecutionResult"
        assert result.success, "Expected all expectations to pass"
        assert result.total_passed == 1, "Expected 1 passed expectation"
        assert result.total_failed == 0, "Expected 0 failed expectations"
    else:  # failure
        with pytest.raises(DataFrameExpectationsSuiteFailure):
            expectations_suite.build().run(data_frame=data_frame)


def test_column_missing_error_pandas():
    """Test that an error is raised when the specified column is missing in pandas."""
    expected_message = "Column 'nonexistent_col' does not exist in the DataFrame."

    data_frame = pd.DataFrame({"col1": [20, 25, 30, 35]})

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
        data_frame_type="pandas",
        message=expected_message,
    )
    assert str(result) == str(expected_failure), f"Expected failure message but got: {result}"

    # Test 2: Suite-based validation
    expectations_suite = DataFrameExpectationsSuite().expect_column_max_between(
        column_name="nonexistent_col", min_value=30, max_value=40
    )
    with pytest.raises(DataFrameExpectationsSuiteFailure):
        expectations_suite.build().run(data_frame=data_frame)


@pytest.mark.pyspark
def test_column_missing_error_pyspark(spark):
    """Test that an error is raised when the specified column is missing in PySpark."""
    expected_message = "Column 'nonexistent_col' does not exist in the DataFrame."

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
        data_frame_type="pyspark",
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
