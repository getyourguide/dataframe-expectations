import pytest
import numpy as np
import pandas as pd

from dataframe_expectations.registry import (
    DataFrameExpectationRegistry,
)
from dataframe_expectations.suite import (
    DataFrameExpectationsSuite,
    DataFrameExpectationsSuiteFailure,
    SuiteExecutionResult,
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


@pytest.mark.parametrize(
    "df_type, df_data, column_name, max_count, expected_result, expected_message",
    [
        # No nulls - should pass
        (
            "pandas",
            {"col1": [1, 2, 3, 4, 5], "col2": ["a", "b", "c", "d", "e"]},
            "col1",
            5,
            "success",
            None,
        ),
        (
            "pyspark",
            {"col1": [1, 2, 3, 4, 5], "col2": ["a", "b", "c", "d", "e"]},
            "col1",
            5,
            "success",
            None,
        ),
        # Within threshold - 2 nulls < 3
        ("pandas", {"col1": [1, None, 3, None, 5]}, "col1", 3, "success", None),
        ("pyspark", {"col1": [1, None, 3, None, 5]}, "col1", 3, "success", None),
        # Exactly at threshold - 2 nulls <= 2
        ("pandas", {"col1": [1, 2, None, 4, None]}, "col1", 2, "success", None),
        ("pyspark", {"col1": [1, 2, None, 4, None]}, "col1", 2, "success", None),
        # With NaN - 1 null <= 2
        ("pandas", {"col1": [1, 2, 3], "col2": [4.0, np.nan, 6.0]}, "col2", 2, "success", None),
        ("pyspark", {"col1": [1, 2, 3], "col2": [4.0, None, 6.0]}, "col2", 2, "success", None),
        # Exceeds threshold - 3 nulls > 1
        (
            "pandas",
            {"col1": [1, None, None, None, 5]},
            "col1",
            1,
            "failure",
            "Column 'col1' has 3 null values, expected at most 1.",
        ),
        (
            "pyspark",
            {"col1": [1, None, None]},
            "col1",
            1,
            "failure",
            "Column 'col1' has 2 null values, expected at most 1.",
        ),
        # All nulls in column - 3 nulls > 1
        (
            "pandas",
            {"col1": [None, None, None], "col2": [1, 2, 3]},
            "col1",
            1,
            "failure",
            "Column 'col1' has 3 null values, expected at most 1.",
        ),
        (
            "pyspark",
            {"col1": [None, None, None]},
            "col1",
            2,
            "failure",
            "Column 'col1' has 3 null values, expected at most 2.",
        ),
        # Zero threshold failure - 1 null > 0
        (
            "pandas",
            {"col1": [1, None, 3]},
            "col1",
            0,
            "failure",
            "Column 'col1' has 1 null values, expected at most 0.",
        ),
        (
            "pyspark",
            {"col1": [1, None, 3]},
            "col1",
            0,
            "failure",
            "Column 'col1' has 1 null values, expected at most 0.",
        ),
        # Zero threshold success - 0 nulls <= 0
        ("pandas", {"col1": [1, 2, 3], "col2": [None, None, None]}, "col1", 0, "success", None),
        ("pyspark", {"col1": [1, 2, 3], "col2": [None, None, None]}, "col1", 0, "success", None),
        # Empty DataFrame - 0 nulls <= 5
        ("pandas", {"col1": []}, "col1", 5, "success", None),
        ("pyspark", {}, "col1", 5, "success", None),
        # Single null value - 1 null > 0
        (
            "pandas",
            {"col1": [None]},
            "col1",
            0,
            "failure",
            "Column 'col1' has 1 null values, expected at most 0.",
        ),
        (
            "pyspark",
            {"col1": [None]},
            "col1",
            0,
            "failure",
            "Column 'col1' has 1 null values, expected at most 0.",
        ),
        # Single non-null value - 0 nulls <= 0
        ("pandas", {"col1": [1]}, "col1", 0, "success", None),
        ("pyspark", {"col1": [1]}, "col1", 0, "success", None),
        # Other columns with nulls don't affect - 0 nulls in col1 <= 1
        ("pandas", {"col1": [1, 2, 3], "col2": [None, None, None]}, "col1", 1, "success", None),
        ("pyspark", {"col1": [1, 2, 3], "col2": [None, None, None]}, "col1", 1, "success", None),
        # Mixed data types with nulls - 2 nulls <= 2
        ("pandas", {"col1": [1, "text", None, 3.14, None]}, "col1", 2, "success", None),
        ("pyspark", {"col1": [1, 2, None, 4, None]}, "col1", 2, "success", None),
        # Large threshold - 2 nulls <= 1000000
        ("pandas", {"col1": [1, None, 3, None, 5]}, "col1", 1000000, "success", None),
        ("pyspark", {"col1": [1, None, 3, None, 5]}, "col1", 1000000, "success", None),
    ],
    ids=[
        "pandas_no_nulls",
        "pyspark_no_nulls",
        "pandas_within_threshold",
        "pyspark_within_threshold",
        "pandas_exactly_at_threshold",
        "pyspark_exactly_at_threshold",
        "pandas_with_nan",
        "pyspark_with_nan",
        "pandas_exceeds_threshold",
        "pyspark_exceeds_threshold",
        "pandas_all_nulls",
        "pyspark_all_nulls",
        "pandas_zero_threshold_failure",
        "pyspark_zero_threshold_failure",
        "pandas_zero_threshold_success",
        "pyspark_zero_threshold_success",
        "pandas_empty",
        "pyspark_empty",
        "pandas_single_null",
        "pyspark_single_null",
        "pandas_single_not_null",
        "pyspark_single_not_null",
        "pandas_other_columns_ignored",
        "pyspark_other_columns_ignored",
        "pandas_mixed_data_types",
        "pyspark_mixed_data_types",
        "pandas_large_threshold",
        "pyspark_large_threshold",
    ],
)
def test_expectation_basic_scenarios(
    df_type, df_data, column_name, max_count, expected_result, expected_message, spark
):
    """
    Test the expectation for various scenarios across pandas and PySpark DataFrames.
    Tests both direct expectation validation and suite-based validation.
    Covers: success cases, failures (exceeds threshold), edge cases (empty, zero threshold, single values),
    boundary conditions, column isolation, and various data types.
    """
    data_frame = create_dataframe(df_type, df_data, spark)

    # Test 1: Direct expectation validation
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationMaxNullCount",
        column_name=column_name,
        max_count=max_count,
    )

    result = expectation.validate(data_frame=data_frame)

    if expected_result == "success":
        assert str(result) == str(
            DataFrameExpectationSuccessMessage(expectation_name="ExpectationMaxNullCount")
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
    expectations_suite = DataFrameExpectationsSuite().expect_max_null_count(
        column_name=column_name, max_count=max_count
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
        expectation_name="ExpectationMaxNullCount",
        column_name="col1",
        max_count=5,
    )

    if df_type == "pandas":
        data_frame = pd.DataFrame({"col2": [1, 2, 3, 4, 5]})
    else:  # pyspark
        data_frame = spark.createDataFrame([(1,), (2,), (3,)], ["col2"])

    result = expectation.validate(data_frame=data_frame)
    # The error message might vary, but should be a failure
    assert isinstance(result, DataFrameExpectationFailureMessage), (
        f"Expected DataFrameExpectationFailureMessage but got: {type(result)}"
    )
    result_str = str(result)
    assert "col1" in result_str, f"Expected 'col1' in result message: {result_str}"

    # Test 2: Suite-based validation
    expectations_suite = DataFrameExpectationsSuite().expect_max_null_count(
        column_name="col1", max_count=5
    )
    with pytest.raises(DataFrameExpectationsSuiteFailure):
        expectations_suite.build().run(data_frame=data_frame)


def test_invalid_parameters():
    """Test that appropriate errors are raised for invalid parameters."""
    # Test negative max_count
    with pytest.raises(ValueError) as context:
        DataFrameExpectationRegistry.get_expectation(
            expectation_name="ExpectationMaxNullCount",
            column_name="col1",
            max_count=-1,
        )
    assert "max_count must be non-negative" in str(context.value), (
        f"Expected 'max_count must be non-negative' in error message: {str(context.value)}"
    )


def test_large_dataset_performance():
    """Test the expectation with a larger dataset to ensure reasonable performance."""
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationMaxNullCount",
        column_name="col1",
        max_count=100,
    )
    # Create a DataFrame with 1000 rows and 50 nulls (every 20th value is None)
    data = [None if i % 20 == 0 else i for i in range(1000)]
    data_frame = pd.DataFrame({"col1": data})
    result = expectation.validate(data_frame=data_frame)
    assert str(result) == str(
        DataFrameExpectationSuccessMessage(expectation_name="ExpectationMaxNullCount")
    ), f"Expected success message but got: {result}"
