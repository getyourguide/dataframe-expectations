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


def create_dataframe(df_type, data, column_name, spark, data_type="long"):
    """Helper function to create pandas or pyspark DataFrame.

    Args:
        df_type: "pandas" or "pyspark"
        data: List of values for the column
        column_name: Name of the column
        spark: Spark session (required for pyspark)
        data_type: Data type for the column - "long", "double"
    """
    if df_type == "pandas":
        return pd.DataFrame({column_name: data})
    else:  # pyspark
        from pyspark.sql.types import (
            StructType,
            StructField,
            LongType,
            DoubleType,
        )

        type_mapping = {
            "long": LongType(),
            "double": DoubleType(),
        }

        schema = StructType([StructField(column_name, type_mapping[data_type], True)])
        return spark.createDataFrame([(val,) for val in data], schema)


def test_expectation_name():
    """Test that the expectation name is correctly returned."""
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationValueBetween",
        column_name="col1",
        min_value=2,
        max_value=5,
    )
    assert expectation.get_expectation_name() == "ExpectationValueBetween", (
        f"Expected 'ExpectationValueBetween' but got: {expectation.get_expectation_name()}"
    )


@pytest.mark.parametrize(
    "df_type, data, min_value, max_value, expected_result, expected_violations, expected_message",
    [
        # Basic success - pandas
        ("pandas", [2, 3, 4, 5], 2, 5, "success", None, None),
        # Basic success - pyspark
        ("pyspark", [2, 3, 4, 5], 2, 5, "success", None, None),
        # Subset success - pandas
        ("pandas", [3, 4], 2, 5, "success", None, None),
        # Subset success - pyspark
        ("pyspark", [3, 4], 2, 5, "success", None, None),
        # Identical values - pandas
        ("pandas", [5, 5, 5, 5], 2, 5, "success", None, None),
        # Identical values - pyspark
        ("pyspark", [5, 5, 5, 5], 2, 5, "success", None, None),
        # Basic violations - pandas
        (
            "pandas",
            [1, 2, 3, 6],
            2,
            5,
            "failure",
            [1, 6],
            "Found 2 row(s) where 'col1' is not between 2 and 5.",
        ),
        # Basic violations - pyspark
        (
            "pyspark",
            [1, 2, 3, 6],
            2,
            5,
            "failure",
            [1, 6],
            "Found 2 row(s) where 'col1' is not between 2 and 5.",
        ),
        # All violations - pandas
        (
            "pandas",
            [0, 1, 6, 7],
            2,
            5,
            "failure",
            [0, 1, 6, 7],
            "Found 4 row(s) where 'col1' is not between 2 and 5.",
        ),
        # All violations - pyspark
        (
            "pyspark",
            [0, 1, 6, 7],
            2,
            5,
            "failure",
            [0, 1, 6, 7],
            "Found 4 row(s) where 'col1' is not between 2 and 5.",
        ),
        # Boundary exact min - pandas
        ("pandas", [2, 2, 2], 2, 5, "success", None, None),
        # Boundary exact min - pyspark
        ("pyspark", [2, 2, 2], 2, 5, "success", None, None),
        # Boundary exact max - pandas
        ("pandas", [5, 5, 5], 2, 5, "success", None, None),
        # Boundary exact max - pyspark
        ("pyspark", [5, 5, 5], 2, 5, "success", None, None),
        # Boundary min max - pandas
        ("pandas", [2, 3, 4, 5], 2, 5, "success", None, None),
        # Boundary min max - pyspark
        ("pyspark", [2, 3, 4, 5], 2, 5, "success", None, None),
        # Boundary below min - pandas
        (
            "pandas",
            [1, 2, 3],
            2,
            5,
            "failure",
            [1],
            "Found 1 row(s) where 'col1' is not between 2 and 5.",
        ),
        # Boundary below min - pyspark
        (
            "pyspark",
            [1, 2, 3],
            2,
            5,
            "failure",
            [1],
            "Found 1 row(s) where 'col1' is not between 2 and 5.",
        ),
        # Boundary above max - pandas
        (
            "pandas",
            [3, 4, 6],
            2,
            5,
            "failure",
            [6],
            "Found 1 row(s) where 'col1' is not between 2 and 5.",
        ),
        # Boundary above max - pyspark
        (
            "pyspark",
            [3, 4, 6],
            2,
            5,
            "failure",
            [6],
            "Found 1 row(s) where 'col1' is not between 2 and 5.",
        ),
        # Negative success - pandas
        ("pandas", [-5, -3, -2, 0], -5, 0, "success", None, None),
        # Negative success - pyspark
        ("pyspark", [-5, -3, -2, 0], -5, 0, "success", None, None),
        # Negative range success - pandas
        ("pandas", [-10, -8, -6], -10, -5, "success", None, None),
        # Negative range success - pyspark
        ("pyspark", [-10, -8, -6], -10, -5, "success", None, None),
        # Negative violations - pandas
        (
            "pandas",
            [-6, -3, 1],
            -5,
            0,
            "failure",
            [-6, 1],
            "Found 2 row(s) where 'col1' is not between -5 and 0.",
        ),
        # Negative violations - pyspark
        (
            "pyspark",
            [-6, -3, 1],
            -5,
            0,
            "failure",
            [-6, 1],
            "Found 2 row(s) where 'col1' is not between -5 and 0.",
        ),
        # Float success - pandas
        ("pandas", [2.5, 3.7, 4.2], 2.0, 5.0, "success", None, None),
        # Float success - pyspark
        ("pyspark", [2.5, 3.7, 4.2], 2.0, 5.0, "success", None, None),
        # Float mixed success - pandas
        ("pandas", [2.0, 2.5, 3.0, 4.5, 5.0], 2.0, 5.0, "success", None, None),
        # Float mixed success - pyspark
        ("pyspark", [2.0, 2.5, 3.0, 4.5, 5.0], 2.0, 5.0, "success", None, None),
        # Float violations - pandas
        (
            "pandas",
            [1.5, 2.5, 5.5],
            2.0,
            5.0,
            "failure",
            [1.5, 5.5],
            "Found 2 row(s) where 'col1' is not between 2.0 and 5.0.",
        ),
        # Float violations - pyspark
        (
            "pyspark",
            [1.5, 2.5, 5.5],
            2.0,
            5.0,
            "failure",
            [1.5, 5.5],
            "Found 2 row(s) where 'col1' is not between 2.0 and 5.0.",
        ),
        # Zero in range - pandas
        ("pandas", [-2, -1, 0, 1, 2], -2, 2, "success", None, None),
        # Zero in range - pyspark
        ("pyspark", [-2, -1, 0, 1, 2], -2, 2, "success", None, None),
        # All zeros - pandas
        ("pandas", [0, 0, 0], 0, 1, "success", None, None),
        # All zeros - pyspark
        ("pyspark", [0, 0, 0], 0, 1, "success", None, None),
        # Single value success - pandas
        ("pandas", [3], 2, 5, "success", None, None),
        # Single value success - pyspark
        ("pyspark", [3], 2, 5, "success", None, None),
        # Single value violation - pandas
        (
            "pandas",
            [1],
            2,
            5,
            "failure",
            [1],
            "Found 1 row(s) where 'col1' is not between 2 and 5.",
        ),
        # Single value violation - pyspark
        (
            "pyspark",
            [6],
            2,
            5,
            "failure",
            [6],
            "Found 1 row(s) where 'col1' is not between 2 and 5.",
        ),
        # Mixed integers and floats - pandas
        ("pandas", [2, 2.5, 3, 4.5, 5], 2, 5, "success", None, None),
        (
            "pandas",
            [1.5, 2, 3, 5.5],
            2,
            5,
            "failure",
            [1.5, 5.5],
            "Found 2 row(s) where 'col1' is not between 2 and 5.",
        ),
        # Large range success - pandas
        ("pandas", [100, 500, 900], 0, 1000, "success", None, None),
        # Large range success - pyspark
        ("pyspark", [100, 500, 900], 0, 1000, "success", None, None),
        # Large range violations - pandas
        (
            "pandas",
            [-100, 500, 1100],
            0,
            1000,
            "failure",
            [-100, 1100],
            "Found 2 row(s) where 'col1' is not between 0 and 1000.",
        ),
        # Large range violations - pyspark
        (
            "pyspark",
            [-100, 500, 1100],
            0,
            1000,
            "failure",
            [-100, 1100],
            "Found 2 row(s) where 'col1' is not between 0 and 1000.",
        ),
        # All below range - pandas
        (
            "pandas",
            [0, 1, 1],
            2,
            5,
            "failure",
            [0, 1, 1],
            "Found 3 row(s) where 'col1' is not between 2 and 5.",
        ),
        # All below range - pyspark
        (
            "pyspark",
            [0, 1, 1],
            2,
            5,
            "failure",
            [0, 1, 1],
            "Found 3 row(s) where 'col1' is not between 2 and 5.",
        ),
        # All above range - pandas
        (
            "pandas",
            [6, 7, 8],
            2,
            5,
            "failure",
            [6, 7, 8],
            "Found 3 row(s) where 'col1' is not between 2 and 5.",
        ),
        # All above range - pyspark
        (
            "pyspark",
            [6, 7, 8],
            2,
            5,
            "failure",
            [6, 7, 8],
            "Found 3 row(s) where 'col1' is not between 2 and 5.",
        ),
        # With nulls success - pandas
        ("pandas", [2, None, 3, None, 4], 2, 5, "success", None, None),
        # With nulls success - pyspark
        ("pyspark", [2, None, 3, None, 4], 2, 5, "success", None, None),
        # With nulls violations - pandas
        (
            "pandas",
            [1.0, None, 3.0, 6.0],
            2,
            5,
            "failure",
            [1.0, 6.0],
            "Found 2 row(s) where 'col1' is not between 2 and 5.",
        ),
        # With nulls violations - pyspark
        (
            "pyspark",
            [1, None, 3, 6],
            2,
            5,
            "failure",
            [1, 6],
            "Found 2 row(s) where 'col1' is not between 2 and 5.",
        ),
    ],
    ids=[
        "pandas_basic_success",
        "pyspark_basic_success",
        "pandas_subset_success",
        "pyspark_subset_success",
        "pandas_identical_values",
        "pyspark_identical_values",
        "pandas_basic_violations",
        "pyspark_basic_violations",
        "pandas_all_violations",
        "pyspark_all_violations",
        "pandas_boundary_exact_min",
        "pyspark_boundary_exact_min",
        "pandas_boundary_exact_max",
        "pyspark_boundary_exact_max",
        "pandas_boundary_min_max",
        "pyspark_boundary_min_max",
        "pandas_boundary_below_min",
        "pyspark_boundary_below_min",
        "pandas_boundary_above_max",
        "pyspark_boundary_above_max",
        "pandas_negative_success",
        "pyspark_negative_success",
        "pandas_negative_range_success",
        "pyspark_negative_range_success",
        "pandas_negative_violations",
        "pyspark_negative_violations",
        "pandas_float_success",
        "pyspark_float_success",
        "pandas_float_mixed_success",
        "pyspark_float_mixed_success",
        "pandas_float_violations",
        "pyspark_float_violations",
        "pandas_zero_in_range",
        "pyspark_zero_in_range",
        "pandas_all_zeros",
        "pyspark_all_zeros",
        "pandas_single_value_success",
        "pyspark_single_value_success",
        "pandas_single_value_violation",
        "pyspark_single_value_violation",
        "pandas_mixed_types_success",
        "pandas_mixed_types_violations",
        "pandas_large_range_success",
        "pyspark_large_range_success",
        "pandas_large_range_violations",
        "pyspark_large_range_violations",
        "pandas_all_below_range",
        "pyspark_all_below_range",
        "pandas_all_above_range",
        "pyspark_all_above_range",
        "pandas_with_nulls_success",
        "pyspark_with_nulls_success",
        "pandas_with_nulls_violations",
        "pyspark_with_nulls_violations",
    ],
)
def test_expectation_basic_scenarios(
    df_type,
    data,
    min_value,
    max_value,
    expected_result,
    expected_violations,
    expected_message,
    spark,
):
    """
    Test the expectation for various scenarios across pandas and PySpark DataFrames.
    Tests both direct expectation validation and suite-based validation.
    Covers: success cases, boundary conditions, violations, negative values, floats,
    zero values, single values, mixed types, large ranges, and nulls.
    """
    # Determine data type based on whether we have float values (excluding None)
    has_float = any(isinstance(val, float) for val in data if val is not None)
    data_type = "double" if has_float else "long"
    data_frame = create_dataframe(df_type, data, "col1", spark, data_type)

    # Test 1: Direct expectation validation
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationValueBetween",
        column_name="col1",
        min_value=min_value,
        max_value=max_value,
    )

    result = expectation.validate(data_frame=data_frame)

    if expected_result == "success":
        assert str(result) == str(
            DataFrameExpectationSuccessMessage(expectation_name="ExpectationValueBetween")
        ), f"Expected success message but got: {result}"
    else:  # failure
        expected_violations_df = create_dataframe(
            df_type, expected_violations, "col1", spark, data_type
        )
        expected_failure_message = DataFrameExpectationFailureMessage(
            expectation_str=str(expectation),
            data_frame_type=str(df_type),
            violations_data_frame=expected_violations_df,
            message=expected_message,
            limit_violations=5,
        )
        assert str(result) == str(expected_failure_message), (
            f"Expected failure message but got: {result}"
        )

    # Test 2: Suite-based validation
    expectations_suite = DataFrameExpectationsSuite().expect_value_between(
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
    expected_message = "Column 'col1' does not exist in the DataFrame."

    if df_type == "pandas":
        data_frame = pd.DataFrame({"col2": [2, 3, 4]})
    else:  # pyspark
        data_frame = spark.createDataFrame([(2,), (3,), (4,)], ["col2"])

    # Test 1: Direct expectation validation
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationValueBetween",
        column_name="col1",
        min_value=2,
        max_value=5,
    )
    result = expectation.validate(data_frame=data_frame)
    expected_failure = DataFrameExpectationFailureMessage(
        expectation_str=str(expectation),
        data_frame_type=str(df_type),
        message=expected_message,
    )
    assert str(result) == str(expected_failure), f"Expected failure message but got: {result}"

    # Test 2: Suite-based validation
    expectations_suite = DataFrameExpectationsSuite().expect_value_between(
        column_name="col1", min_value=2, max_value=5
    )
    with pytest.raises(DataFrameExpectationsSuiteFailure):
        expectations_suite.build().run(data_frame=data_frame)


def test_large_dataset_performance():
    """Test the expectation with a larger dataset to ensure performance."""
    import numpy as np

    # Create a larger dataset with values between 10 and 100
    large_data = np.random.uniform(10, 100, 10000).tolist()
    data_frame = pd.DataFrame({"col1": large_data})

    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationValueBetween",
        column_name="col1",
        min_value=5,
        max_value=105,
    )

    result = expectation.validate(data_frame=data_frame)
    # Should succeed as all values from uniform(10, 100) are between 5 and 105
    assert isinstance(result, DataFrameExpectationSuccessMessage), (
        f"Large dataset test failed: expected success but got {type(result)}"
    )
