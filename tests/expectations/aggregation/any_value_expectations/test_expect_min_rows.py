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


def create_dataframe(df_type, df_data, spark):
    """Helper function to create pandas or pyspark DataFrame."""
    if df_type == "pandas":
        return pd.DataFrame(df_data)

    # PySpark
    from pyspark.sql.types import (
        StructType,
        StructField,
        IntegerType,
        StringType,
        DoubleType,
        BooleanType,
    )

    if not df_data:
        return spark.createDataFrame([], StructType([StructField("col1", IntegerType(), True)]))

    type_map = {str: StringType(), float: DoubleType(), bool: BooleanType()}

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
    "df_type, df_data, min_rows, expected_result, expected_message",
    [
        # Exact count - 3 rows == 3 min
        ("pandas", {"col1": [1, 2, 3], "col2": ["a", "b", "c"]}, 3, "success", None),
        ("pyspark", {"col1": [1, 2, 3], "col2": ["a", "b", "c"]}, 3, "success", None),
        # Above min - 5 rows > 3 min
        (
            "pandas",
            {"col1": [1, 2, 3, 4, 5], "col2": ["a", "b", "c", "d", "e"]},
            3,
            "success",
            None,
        ),
        (
            "pyspark",
            {"col1": [1, 2, 3, 4, 5], "col2": ["a", "b", "c", "d", "e"]},
            3,
            "success",
            None,
        ),
        # Single row - 1 row == 1 min
        ("pandas", {"col1": [42]}, 1, "success", None),
        ("pyspark", {"col1": [42]}, 1, "success", None),
        # Zero min empty - 0 rows == 0 min
        ("pandas", {"col1": []}, 0, "success", None),
        ("pyspark", {}, 0, "success", None),
        # Zero min with data - 3 rows >= 0 min
        ("pandas", {"col1": [1, 2, 3]}, 0, "success", None),
        ("pyspark", {"col1": [1, 2, 3]}, 0, "success", None),
        # With nulls - 5 rows >= 3 min (nulls don't affect row count)
        (
            "pandas",
            {"col1": [1, None, 3, None, 5], "col2": [None, "b", None, "d", None]},
            3,
            "success",
            None,
        ),
        (
            "pyspark",
            {"col1": [1, None, 3, None, 5], "col2": [None, "b", None, "d", None]},
            3,
            "success",
            None,
        ),
        # Below min - 3 rows < 5 min
        (
            "pandas",
            {"col1": [1, 2, 3], "col2": ["a", "b", "c"]},
            5,
            "failure",
            "DataFrame has 3 rows, expected at least 5.",
        ),
        (
            "pyspark",
            {"col1": [1, 2, 3], "col2": ["a", "b", "c"]},
            5,
            "failure",
            "DataFrame has 3 rows, expected at least 5.",
        ),
        # Empty needs min - 0 rows < 2 min
        ("pandas", {"col1": []}, 2, "failure", "DataFrame has 0 rows, expected at least 2."),
        ("pyspark", {}, 2, "failure", "DataFrame has 0 rows, expected at least 2."),
        # Single row needs more - 1 row < 3 min
        ("pandas", {"col1": [1]}, 3, "failure", "DataFrame has 1 rows, expected at least 3."),
        ("pyspark", {"col1": [1]}, 3, "failure", "DataFrame has 1 rows, expected at least 3."),
        # Large dataset success - 150 rows >= 100 min
        (
            "pandas",
            {"col1": list(range(150)), "col2": [f"value_{i}" for i in range(150)]},
            100,
            "success",
            None,
        ),
        (
            "pyspark",
            {"col1": list(range(75)), "col2": [f"value_{i}" for i in range(75)]},
            50,
            "success",
            None,
        ),
        # Large dataset failure - 150 rows < 200 min
        (
            "pandas",
            {"col1": list(range(150)), "col2": [f"value_{i}" for i in range(150)]},
            200,
            "failure",
            "DataFrame has 150 rows, expected at least 200.",
        ),
        (
            "pyspark",
            {"col1": list(range(75)), "col2": [f"value_{i}" for i in range(75)]},
            100,
            "failure",
            "DataFrame has 75 rows, expected at least 100.",
        ),
        # Multiple columns - 4 rows >= 3 min
        (
            "pandas",
            {
                "col1": [1, 2, 3, 4],
                "col2": ["a", "b", "c", "d"],
                "col3": [1.1, 2.2, 3.3, 4.4],
                "col4": [True, False, True, False],
            },
            3,
            "success",
            None,
        ),
        # Mixed data types - 5 rows >= 3 min
        (
            "pandas",
            {
                "int_col": [1, 2, 3, 4, 5],
                "str_col": ["a", "b", "c", "d", "e"],
                "float_col": [1.1, 2.2, 3.3, 4.4, 5.5],
                "bool_col": [True, False, True, False, True],
                "null_col": [None, None, None, None, None],
            },
            3,
            "success",
            None,
        ),
        # Low min count - 3 rows >= 1 min
        ("pandas", {"col1": [1, 2, 3]}, 1, "success", None),
        # High min count - 3 rows < 1000000 min
        (
            "pandas",
            {"col1": [1, 2, 3]},
            1000000,
            "failure",
            "DataFrame has 3 rows, expected at least 1000000.",
        ),
        # Identical values - 4 rows >= 3 min
        (
            "pandas",
            {"col1": [42, 42, 42, 42], "col2": ["same", "same", "same", "same"]},
            3,
            "success",
            None,
        ),
        # Boundary condition - 1 row == 1 min (edge case equals actual)
        ("pandas", {"col1": [1]}, 1, "success", None),
        # Progressive counts - 5 rows meets various minimums
        ("pandas", {"col1": [1, 2, 3, 4, 5]}, 5, "success", None),
        ("pandas", {"col1": [1, 2, 3, 4, 5]}, 4, "success", None),
        (
            "pandas",
            {"col1": [1, 2, 3, 4, 5]},
            6,
            "failure",
            "DataFrame has 5 rows, expected at least 6.",
        ),
    ],
    ids=[
        "pandas_exact_count",
        "pyspark_exact_count",
        "pandas_above_min",
        "pyspark_above_min",
        "pandas_single_row",
        "pyspark_single_row",
        "pandas_zero_min_empty",
        "pyspark_zero_min_empty",
        "pandas_zero_min_with_data",
        "pyspark_zero_min_with_data",
        "pandas_with_nulls",
        "pyspark_with_nulls",
        "pandas_below_min",
        "pyspark_below_min",
        "pandas_empty_needs_min",
        "pyspark_empty_needs_min",
        "pandas_single_row_needs_more",
        "pyspark_single_row_needs_more",
        "pandas_large_dataset",
        "pyspark_large_dataset",
        "pandas_large_dataset_failure",
        "pyspark_large_dataset_failure",
        "pandas_multiple_columns",
        "pandas_mixed_data_types",
        "pandas_low_min_count",
        "pandas_high_min_count",
        "pandas_identical_values",
        "pandas_boundary_condition",
        "pandas_progressive_count_at_min",
        "pandas_progressive_count_below_min",
        "pandas_progressive_count_above_min",
    ],
)
def test_expectation_basic_scenarios(
    df_type, df_data, min_rows, expected_result, expected_message, spark
):
    """
    Test the expectation for various scenarios across pandas and PySpark DataFrames.
    Tests both direct expectation validation and suite-based validation.
    Covers: success cases (exact, above min, zero min), failures (below min, empty), edge cases,
    boundary conditions, large datasets, nulls, multiple columns, mixed data types, identical values,
    progressive counts, and dataframe structure variations.
    """
    data_frame = create_dataframe(df_type, df_data, spark)

    # Test 1: Direct expectation validation
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationMinRows",
        min_rows=min_rows,
    )

    result = expectation.validate(data_frame=data_frame)

    if expected_result == "success":
        assert str(result) == str(
            DataFrameExpectationSuccessMessage(expectation_name="ExpectationMinRows")
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
    expectations_suite = DataFrameExpectationsSuite().expect_min_rows(min_rows=min_rows)

    if expected_result == "success":
        result = expectations_suite.build().run(data_frame=data_frame)
        assert result is None, "Expected no exceptions to be raised from suite"
    else:  # failure
        with pytest.raises(DataFrameExpectationsSuiteFailure):
            expectations_suite.build().run(data_frame=data_frame)


def test_invalid_parameters():
    """Test that appropriate errors are raised for invalid parameters."""
    # Test negative min_rows
    with pytest.raises(ValueError) as context:
        DataFrameExpectationRegistry.get_expectation(
            expectation_name="ExpectationMinRows",
            min_rows=-1,
        )
    assert "min_rows must be non-negative" in str(context.value), (
        f"Expected 'min_rows must be non-negative' in error message: {str(context.value)}"
    )


def test_large_dataset_performance():
    """Test the expectation with a larger dataset to ensure reasonable performance."""
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationMinRows",
        min_rows=500,
    )
    # Create a DataFrame with 1000 rows
    data_frame = pd.DataFrame({"col1": list(range(1000))})
    result = expectation.validate(data_frame=data_frame)
    assert str(result) == str(
        DataFrameExpectationSuccessMessage(expectation_name="ExpectationMinRows")
    ), f"Expected success message but got: {result}"
