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


@pytest.mark.parametrize(
    "df_type, df_data, max_rows, expected_result, expected_message",
    [
        # Exact count - 3 rows == 3 max
        ("pandas", {"col1": [1, 2, 3], "col2": ["a", "b", "c"]}, 3, "success", None),
        ("pyspark", {"col1": [1, 2, 3], "col2": ["a", "b", "c"]}, 3, "success", None),
        # Below max - 5 rows < 10 max
        (
            "pandas",
            {"col1": [1, 2, 3, 4, 5], "col2": ["a", "b", "c", "d", "e"]},
            10,
            "success",
            None,
        ),
        (
            "pyspark",
            {"col1": [1, 2, 3, 4, 5], "col2": ["a", "b", "c", "d", "e"]},
            10,
            "success",
            None,
        ),
        # Single row - 1 row == 1 max
        ("pandas", {"col1": [42]}, 1, "success", None),
        ("pyspark", {"col1": [42]}, 1, "success", None),
        # Empty DataFrame - 0 rows <= 5 max
        ("pandas", {"col1": []}, 5, "success", None),
        ("pyspark", {}, 5, "success", None),
        # Exceeds max - 5 rows > 3 max
        (
            "pandas",
            {"col1": [1, 2, 3, 4, 5], "col2": ["a", "b", "c", "d", "e"]},
            3,
            "failure",
            "DataFrame has 5 rows, expected at most 3.",
        ),
        (
            "pyspark",
            {"col1": [1, 2, 3, 4, 5], "col2": ["a", "b", "c", "d", "e"]},
            3,
            "failure",
            "DataFrame has 5 rows, expected at most 3.",
        ),
        # Zero max with data - 1 row > 0 max
        ("pandas", {"col1": [1]}, 0, "failure", "DataFrame has 1 rows, expected at most 0."),
        ("pyspark", {"col1": [1]}, 0, "failure", "DataFrame has 1 rows, expected at most 0."),
        # Zero max empty - 0 rows == 0 max
        ("pandas", {"col1": []}, 0, "success", None),
        ("pyspark", {}, 0, "success", None),
        # Large dataset - 150 rows > 100 max
        (
            "pandas",
            {"col1": list(range(150)), "col2": [f"value_{i}" for i in range(150)]},
            100,
            "failure",
            "DataFrame has 150 rows, expected at most 100.",
        ),
        (
            "pyspark",
            {"col1": list(range(75)), "col2": [f"value_{i}" for i in range(75)]},
            50,
            "failure",
            "DataFrame has 75 rows, expected at most 50.",
        ),
        # With nulls - 5 rows > 4 max (nulls don't affect row count)
        (
            "pandas",
            {"col1": [1, None, 3, None, 5], "col2": [None, "b", None, "d", None]},
            4,
            "failure",
            "DataFrame has 5 rows, expected at most 4.",
        ),
        (
            "pyspark",
            {"col1": [1, None, 3, None, 5], "col2": [None, "b", None, "d", None]},
            4,
            "failure",
            "DataFrame has 5 rows, expected at most 4.",
        ),
        # Multiple columns - 4 rows > 3 max
        (
            "pandas",
            {
                "col1": [1, 2, 3, 4],
                "col2": ["a", "b", "c", "d"],
                "col3": [1.1, 2.2, 3.3, 4.4],
                "col4": [True, False, True, False],
            },
            3,
            "failure",
            "DataFrame has 4 rows, expected at most 3.",
        ),
        (
            "pyspark",
            {
                "col1": [1, 2, 3, 4],
                "col2": ["a", "b", "c", "d"],
                "col3": [1.1, 2.2, 3.3, 4.4],
                "col4": [True, False, True, False],
            },
            3,
            "failure",
            "DataFrame has 4 rows, expected at most 3.",
        ),
        # Mixed data types - 5 rows <= 10 max
        (
            "pandas",
            {
                "int_col": [1, 2, 3, 4, 5],
                "str_col": ["a", "b", "c", "d", "e"],
                "float_col": [1.1, 2.2, 3.3, 4.4, 5.5],
                "bool_col": [True, False, True, False, True],
                "null_col": [None, None, None, None, None],
            },
            10,
            "success",
            None,
        ),
        (
            "pyspark",
            {
                "int_col": [1, 2, 3, 4, 5],
                "str_col": ["a", "b", "c", "d", "e"],
                "float_col": [1.1, 2.2, 3.3, 4.4, 5.5],
                "bool_col": [True, False, True, False, True],
                "null_col": [None, None, None, None, None],
            },
            10,
            "success",
            None,
        ),
        # High max rows - 3 rows << 1000000 max
        ("pandas", {"col1": [1, 2, 3]}, 1000000, "success", None),
        ("pyspark", {"col1": [1, 2, 3]}, 1000000, "success", None),
        # Boundary condition 1 - 1 row == 1 max
        ("pandas", {"col1": [1]}, 1, "success", None),
        ("pyspark", {"col1": [1]}, 1, "success", None),
        # Boundary condition 2 - 2 rows > 1 max
        ("pandas", {"col1": [1, 2]}, 1, "failure", "DataFrame has 2 rows, expected at most 1."),
        ("pyspark", {"col1": [1, 2]}, 1, "failure", "DataFrame has 2 rows, expected at most 1."),
        # Identical values - 4 rows > 3 max
        (
            "pandas",
            {"col1": [42, 42, 42, 42], "col2": ["same", "same", "same", "same"]},
            3,
            "failure",
            "DataFrame has 4 rows, expected at most 3.",
        ),
        (
            "pyspark",
            {"col1": [42, 42, 42, 42], "col2": ["same", "same", "same", "same"]},
            3,
            "failure",
            "DataFrame has 4 rows, expected at most 3.",
        ),
    ],
    ids=[
        "pandas_exact_count",
        "pyspark_exact_count",
        "pandas_below_max",
        "pyspark_below_max",
        "pandas_single_row",
        "pyspark_single_row",
        "pandas_empty",
        "pyspark_empty",
        "pandas_exceeds_max",
        "pyspark_exceeds_max",
        "pandas_zero_max_with_data",
        "pyspark_zero_max_with_data",
        "pandas_zero_max_empty",
        "pyspark_zero_max_empty",
        "pandas_large_dataset",
        "pyspark_large_dataset",
        "pandas_with_nulls",
        "pyspark_with_nulls",
        "pandas_multiple_columns",
        "pyspark_multiple_columns",
        "pandas_mixed_data_types",
        "pyspark_mixed_data_types",
        "pandas_high_max_rows",
        "pyspark_high_max_rows",
        "pandas_boundary_condition_1",
        "pyspark_boundary_condition_1",
        "pandas_boundary_condition_2",
        "pyspark_boundary_condition_2",
        "pandas_identical_values",
        "pyspark_identical_values",
    ],
)
def test_expectation_basic_scenarios(
    df_type, df_data, max_rows, expected_result, expected_message, spark
):
    """
    Test the expectation for various scenarios across pandas and PySpark DataFrames.
    Tests both direct expectation validation and suite-based validation.
    Covers: success cases, failures (exceeds max), edge cases (empty, zero max, single row),
    boundary conditions, large datasets, nulls, multiple columns, mixed data types, and identical values.
    """
    data_frame = create_dataframe(df_type, df_data, spark)

    # Test 1: Direct expectation validation
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationMaxRows",
        max_rows=max_rows,
    )

    result = expectation.validate(data_frame=data_frame)

    if expected_result == "success":
        assert str(result) == str(
            DataFrameExpectationSuccessMessage(expectation_name="ExpectationMaxRows")
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
    expectations_suite = DataFrameExpectationsSuite().expect_max_rows(max_rows=max_rows)

    if expected_result == "success":
        result = expectations_suite.build().run(data_frame=data_frame)
        assert result is None, "Expected no exceptions to be raised from suite"
    else:  # failure
        with pytest.raises(DataFrameExpectationsSuiteFailure):
            expectations_suite.build().run(data_frame=data_frame)


def test_invalid_parameters():
    """Test that appropriate errors are raised for invalid parameters."""
    # Test negative max_rows
    with pytest.raises(ValueError) as context:
        DataFrameExpectationRegistry.get_expectation(
            expectation_name="ExpectationMaxRows",
            max_rows=-1,
        )
    assert "max_rows must be non-negative" in str(context.value), (
        f"Expected 'max_rows must be non-negative' in error message: {str(context.value)}"
    )


def test_large_dataset_performance():
    """Test the expectation with a larger dataset to ensure reasonable performance."""
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationMaxRows",
        max_rows=1500,
    )
    # Create a DataFrame with 1000 rows
    data_frame = pd.DataFrame({"col1": list(range(1000))})
    result = expectation.validate(data_frame=data_frame)
    assert str(result) == str(
        DataFrameExpectationSuccessMessage(expectation_name="ExpectationMaxRows")
    ), f"Expected success message but got: {result}"
