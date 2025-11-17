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
        return spark.createDataFrame([(val,) for val in data], [column_name])


def get_df_type_enum(df_type):
    """Get DataFrameType enum value."""
    return DataFrameType.PANDAS if df_type == "pandas" else DataFrameType.PYSPARK


def test_expectation_name():
    """Test that the expectation name is correctly returned."""
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationValueLessThan",
        column_name="col1",
        value=2,
    )
    assert expectation.get_expectation_name() == "ExpectationValueLessThan", (
        f"Expected 'ExpectationValueLessThan' but got: {expectation.get_expectation_name()}"
    )


@pytest.mark.parametrize(
    "df_type, data, threshold, expected_result, expected_violations, expected_message",
    [
        # Basic success scenarios - pandas
        ("pandas", [3, 4, 5], 6, "success", None, None),
        ("pandas", [1, 2, 3], 4, "success", None, None),
        ("pandas", [0, 1, 2], 5, "success", None, None),
        # Basic success scenarios - pyspark
        ("pyspark", [3, 4, 5], 6, "success", None, None),
        ("pyspark", [1, 2, 3], 4, "success", None, None),
        ("pyspark", [0, 1, 2], 5, "success", None, None),
        # Basic violation scenarios - pandas
        (
            "pandas",
            [3, 4, 5],
            5,
            "failure",
            [5],
            "Found 1 row(s) where 'col1' is not less than 5.",
        ),
        (
            "pandas",
            [3, 4, 5],
            3,
            "failure",
            [3, 4, 5],
            "Found 3 row(s) where 'col1' is not less than 3.",
        ),
        (
            "pandas",
            [2, 3, 4, 5],
            4,
            "failure",
            [4, 5],
            "Found 2 row(s) where 'col1' is not less than 4.",
        ),
        # Basic violation scenarios - pyspark
        (
            "pyspark",
            [3, 4, 5],
            5,
            "failure",
            [5],
            "Found 1 row(s) where 'col1' is not less than 5.",
        ),
        (
            "pyspark",
            [3, 4, 5],
            3,
            "failure",
            [3, 4, 5],
            "Found 3 row(s) where 'col1' is not less than 3.",
        ),
        (
            "pyspark",
            [2, 3, 4, 5],
            4,
            "failure",
            [4, 5],
            "Found 2 row(s) where 'col1' is not less than 4.",
        ),
        # Boundary conditions - just below threshold - pandas
        ("pandas", [1, 2, 3], 4, "success", None, None),
        ("pandas", [1.5, 1.8, 1.9], 2.0, "success", None, None),
        # Boundary conditions - at threshold (violation) - pandas
        (
            "pandas",
            [2, 3, 4],
            2,
            "failure",
            [2, 3, 4],
            "Found 3 row(s) where 'col1' is not less than 2.",
        ),
        (
            "pandas",
            [5, 5, 5],
            5,
            "failure",
            [5, 5, 5],
            "Found 3 row(s) where 'col1' is not less than 5.",
        ),
        # Boundary conditions - just above threshold - pandas
        (
            "pandas",
            [3, 4, 5],
            3,
            "failure",
            [3, 4, 5],
            "Found 3 row(s) where 'col1' is not less than 3.",
        ),
        # Boundary conditions - pyspark
        ("pyspark", [1, 2, 3], 4, "success", None, None),
        (
            "pyspark",
            [2, 3, 4],
            2,
            "failure",
            [2, 3, 4],
            "Found 3 row(s) where 'col1' is not less than 2.",
        ),
        (
            "pyspark",
            [5, 5, 5],
            5,
            "failure",
            [5, 5, 5],
            "Found 3 row(s) where 'col1' is not less than 5.",
        ),
        # Negative values - success - pandas
        ("pandas", [-5, -3, -2], 0, "success", None, None),
        ("pandas", [-10, -8, -6], -5, "success", None, None),
        ("pandas", [-3, -2, -1], 0, "success", None, None),
        # Negative values - success - pyspark
        ("pyspark", [-5, -3, -2], 0, "success", None, None),
        ("pyspark", [-10, -8, -6], -5, "success", None, None),
        # Negative values - violations - pandas
        (
            "pandas",
            [-1, -2, -3],
            -2,
            "failure",
            [-1, -2],
            "Found 2 row(s) where 'col1' is not less than -2.",
        ),
        (
            "pandas",
            [-3, -4, -5],
            -5,
            "failure",
            [-3, -4, -5],
            "Found 3 row(s) where 'col1' is not less than -5.",
        ),
        # Negative values - violations - pyspark
        (
            "pyspark",
            [-1, -2, -3],
            -2,
            "failure",
            [-1, -2],
            "Found 2 row(s) where 'col1' is not less than -2.",
        ),
        # Float values - success - pandas
        ("pandas", [1.5, 2.3, 3.8], 4.0, "success", None, None),
        ("pandas", [0.5, 1.5, 2.5], 3.0, "success", None, None),
        ("pandas", [9.8, 9.9, 9.95], 10.0, "success", None, None),
        # Float values - success - pyspark
        ("pyspark", [1.5, 2.3, 3.8], 4.0, "success", None, None),
        ("pyspark", [0.5, 1.5, 2.5], 3.0, "success", None, None),
        # Float values - violations - pandas
        (
            "pandas",
            [2.5, 3.0, 3.5],
            2.5,
            "failure",
            [2.5, 3.0, 3.5],
            "Found 3 row(s) where 'col1' is not less than 2.5.",
        ),
        (
            "pandas",
            [1.5, 2.5, 3.5],
            2.0,
            "failure",
            [2.5, 3.5],
            "Found 2 row(s) where 'col1' is not less than 2.0.",
        ),
        # Float values - violations - pyspark
        (
            "pyspark",
            [2.5, 3.0, 3.5],
            2.5,
            "failure",
            [2.5, 3.0, 3.5],
            "Found 3 row(s) where 'col1' is not less than 2.5.",
        ),
        # Zero as threshold - success - pandas
        ("pandas", [-3, -2, -1], 0, "success", None, None),
        ("pandas", [-1.0, -0.5, -0.1], 0, "success", None, None),
        # Zero as threshold - success - pyspark
        ("pyspark", [-3, -2, -1], 0, "success", None, None),
        # Zero as threshold - violations - pandas
        (
            "pandas",
            [0, 1, 2],
            0,
            "failure",
            [0, 1, 2],
            "Found 3 row(s) where 'col1' is not less than 0.",
        ),
        (
            "pandas",
            [-1, 0, 1],
            0,
            "failure",
            [0, 1],
            "Found 2 row(s) where 'col1' is not less than 0.",
        ),
        # Zero as threshold - violations - pyspark
        (
            "pyspark",
            [0, 1, 2],
            0,
            "failure",
            [0, 1, 2],
            "Found 3 row(s) where 'col1' is not less than 0.",
        ),
        # Zero in data - success - pandas
        ("pandas", [-2, -1, 0], 1, "success", None, None),
        # Zero in data - violations - pandas
        (
            "pandas",
            [0, 1, 2],
            0,
            "failure",
            [0, 1, 2],
            "Found 3 row(s) where 'col1' is not less than 0.",
        ),
        # Single value - success - pandas
        ("pandas", [3], 4, "success", None, None),
        ("pandas", [0], 10, "success", None, None),
        # Single value - success - pyspark
        ("pyspark", [3], 4, "success", None, None),
        # Single value - violation - pandas
        (
            "pandas",
            [5],
            5,
            "failure",
            [5],
            "Found 1 row(s) where 'col1' is not less than 5.",
        ),
        (
            "pandas",
            [10],
            5,
            "failure",
            [10],
            "Found 1 row(s) where 'col1' is not less than 5.",
        ),
        # Single value - violation - pyspark
        (
            "pyspark",
            [5],
            5,
            "failure",
            [5],
            "Found 1 row(s) where 'col1' is not less than 5.",
        ),
        # All values equal to threshold - pandas
        (
            "pandas",
            [5, 5, 5, 5],
            5,
            "failure",
            [5, 5, 5, 5],
            "Found 4 row(s) where 'col1' is not less than 5.",
        ),
        # All values equal to threshold - pyspark
        (
            "pyspark",
            [5, 5, 5, 5],
            5,
            "failure",
            [5, 5, 5, 5],
            "Found 4 row(s) where 'col1' is not less than 5.",
        ),
        # Mixed integers and floats - pandas
        ("pandas", [1, 1.5, 2, 2.5], 3, "success", None, None),
        (
            "pandas",
            [2, 2.5, 3, 3.5],
            2.5,
            "failure",
            [2.5, 3, 3.5],
            "Found 3 row(s) where 'col1' is not less than 2.5.",
        ),
        # Large values - pandas
        ("pandas", [100, 500, 900], 1000, "success", None, None),
        (
            "pandas",
            [1000, 1500, 2000],
            1000,
            "failure",
            [1000, 1500, 2000],
            "Found 3 row(s) where 'col1' is not less than 1000.",
        ),
        # Large values - pyspark
        ("pyspark", [100, 500, 900], 1000, "success", None, None),
        # All values above threshold - pandas
        (
            "pandas",
            [6, 7, 8],
            5,
            "failure",
            [6, 7, 8],
            "Found 3 row(s) where 'col1' is not less than 5.",
        ),
        # All values above threshold - pyspark
        (
            "pyspark",
            [6, 7, 8],
            5,
            "failure",
            [6, 7, 8],
            "Found 3 row(s) where 'col1' is not less than 5.",
        ),
        # With nulls - success (nulls are ignored) - pandas
        ("pandas", [1, None, 2, None, 3], 5, "success", None, None),
        ("pandas", [0, None, 5, None], 10, "success", None, None),
        # With nulls - success - pyspark
        ("pyspark", [1, None, 2, None, 3], 5, "success", None, None),
        # With nulls - violations - pandas (use floats to avoid type conversion issues)
        (
            "pandas",
            [2.0, None, 5.0, 6.0],
            5,
            "failure",
            [5.0, 6.0],
            "Found 2 row(s) where 'col1' is not less than 5.",
        ),
        # With nulls - violations - pyspark
        (
            "pyspark",
            [2, None, 5, 6],
            5,
            "failure",
            [5, 6],
            "Found 2 row(s) where 'col1' is not less than 5.",
        ),
    ],
    ids=[
        "pandas_basic_success",
        "pandas_success_different_data",
        "pandas_success_small_values",
        "pyspark_basic_success",
        "pyspark_success_different_data",
        "pyspark_success_small_values",
        "pandas_basic_violations",
        "pandas_all_violations",
        "pandas_partial_violations",
        "pyspark_basic_violations",
        "pyspark_all_violations",
        "pyspark_partial_violations",
        "pandas_boundary_just_below",
        "pandas_boundary_just_below_float",
        "pandas_boundary_at_threshold",
        "pandas_boundary_all_at_threshold",
        "pandas_boundary_above_threshold",
        "pyspark_boundary_just_below",
        "pyspark_boundary_at_threshold",
        "pyspark_boundary_all_at_threshold",
        "pandas_negative_success",
        "pandas_negative_range_success",
        "pandas_negative_to_zero_success",
        "pyspark_negative_success",
        "pyspark_negative_range_success",
        "pandas_negative_violations",
        "pandas_negative_all_violations",
        "pyspark_negative_violations",
        "pandas_float_success",
        "pandas_float_different_success",
        "pandas_float_precise_success",
        "pyspark_float_success",
        "pyspark_float_different_success",
        "pandas_float_violations",
        "pandas_float_mixed_violations",
        "pyspark_float_violations",
        "pandas_zero_threshold_success",
        "pandas_zero_threshold_float_success",
        "pyspark_zero_threshold_success",
        "pandas_zero_threshold_violations",
        "pandas_zero_threshold_mixed_violations",
        "pyspark_zero_threshold_violations",
        "pandas_zero_in_data_success",
        "pandas_zero_in_data_violation",
        "pandas_single_value_success",
        "pandas_single_value_large_success",
        "pyspark_single_value_success",
        "pandas_single_value_violation",
        "pandas_single_value_above_violation",
        "pyspark_single_value_violation",
        "pandas_all_equal_threshold",
        "pyspark_all_equal_threshold",
        "pandas_mixed_types_success",
        "pandas_mixed_types_violations",
        "pandas_large_values_success",
        "pandas_large_values_violations",
        "pyspark_large_values_success",
        "pandas_all_above_threshold",
        "pyspark_all_above_threshold",
        "pandas_with_nulls_success",
        "pandas_with_nulls_large_success",
        "pyspark_with_nulls_success",
        "pandas_with_nulls_violations",
        "pyspark_with_nulls_violations",
    ],
)
def test_expectation_basic_scenarios(
    df_type, data, threshold, expected_result, expected_violations, expected_message, spark
):
    """
    Test the expectation for various scenarios across pandas and PySpark DataFrames.
    Tests both direct expectation validation and suite-based validation.
    Covers: success cases, boundary conditions, violations, negative values, floats,
    zero values, single values, mixed types, large values, and nulls.
    """
    data_frame = create_dataframe(df_type, data, "col1", spark)

    # Test 1: Direct expectation validation
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationValueLessThan",
        column_name="col1",
        value=threshold,
    )

    result = expectation.validate(data_frame=data_frame)

    if expected_result == "success":
        assert str(result) == str(
            DataFrameExpectationSuccessMessage(expectation_name="ExpectationValueLessThan")
        ), f"Expected success message but got: {result}"
    else:  # failure
        expected_violations_df = create_dataframe(df_type, expected_violations, "col1", spark)
        expected_failure_message = DataFrameExpectationFailureMessage(
            expectation_str=str(expectation),
            data_frame_type=get_df_type_enum(df_type),
            violations_data_frame=expected_violations_df,
            message=expected_message,
            limit_violations=5,
        )
        assert str(result) == str(expected_failure_message), (
            f"Expected failure message but got: {result}"
        )

    # Test 2: Suite-based validation
    expectations_suite = DataFrameExpectationsSuite().expect_value_less_than(
        column_name="col1", value=threshold
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
        data_frame = pd.DataFrame({"col2": [3, 4, 5]})
    else:  # pyspark
        data_frame = spark.createDataFrame([(3,), (4,), (5,)], ["col2"])

    # Test 1: Direct expectation validation
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationValueLessThan",
        column_name="col1",
        value=5,
    )
    result = expectation.validate(data_frame=data_frame)
    expected_failure = DataFrameExpectationFailureMessage(
        expectation_str=str(expectation),
        data_frame_type=get_df_type_enum(df_type),
        message=expected_message,
    )
    assert str(result) == str(expected_failure), f"Expected failure message but got: {result}"

    # Test 2: Suite-based validation
    expectations_suite = DataFrameExpectationsSuite().expect_value_less_than(
        column_name="col1", value=5
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
        expectation_name="ExpectationValueLessThan",
        column_name="col1",
        value=105,
    )

    result = expectation.validate(data_frame=data_frame)
    # Should succeed as all values from uniform(10, 100) are < 105
    assert isinstance(result, DataFrameExpectationSuccessMessage), (
        f"Large dataset test failed: expected success but got {type(result)}"
    )
