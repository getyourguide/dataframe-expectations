import pytest
import pandas as pd
from datetime import datetime

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
        data_type: Data type for the column - "long", "string", "double", "boolean", "timestamp"
    """
    if df_type == "pandas":
        return pd.DataFrame({column_name: data})
    else:  # pyspark
        from pyspark.sql.types import (
            StructType,
            StructField,
            LongType,
            StringType,
            DoubleType,
            BooleanType,
            TimestampType,
        )

        type_mapping = {
            "long": LongType(),
            "string": StringType(),
            "double": DoubleType(),
            "boolean": BooleanType(),
            "timestamp": TimestampType(),
        }

        schema = StructType([StructField(column_name, type_mapping[data_type], True)])
        return spark.createDataFrame([(val,) for val in data], schema)


def test_expectation_name():
    """Test that the expectation name is correctly returned."""
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationValueIn",
        column_name="col1",
        values=[1, 2, 3],
    )
    assert expectation.get_expectation_name() == "ExpectationValueIn", (
        f"Expected 'ExpectationValueIn' but got: {expectation.get_expectation_name()}"
    )


@pytest.mark.parametrize(
    "df_type, data, values, should_succeed, expected_violations, expected_message, data_type",
    [
        # Basic integer scenarios - success
        ("pandas", [1, 2, 3, 2, 1], [1, 2, 3], True, None, None, "long"),
        ("pyspark", [1, 2, 3, 2, 1], [1, 2, 3], True, None, None, "long"),
        # Integer scenarios - violations
        (
            "pandas",
            [1, 4, 5, 2, 3],
            [1, 2, 3],
            False,
            [4, 5],
            "Found 2 row(s) where 'col1' is not in [1, 2, 3].",
            "long",
        ),
        (
            "pyspark",
            [1, 4, 5, 2, 3],
            [1, 2, 3],
            False,
            [4, 5],
            "Found 2 row(s) where 'col1' is not in [1, 2, 3].",
            "long",
        ),
        # String data type scenarios
        (
            "pandas",
            ["apple", "banana", "cherry"],
            ["apple", "banana", "cherry"],
            True,
            None,
            None,
            "string",
        ),
        (
            "pyspark",
            ["apple", "banana", "cherry"],
            ["apple", "banana", "cherry"],
            True,
            None,
            None,
            "string",
        ),
        (
            "pandas",
            ["apple", "orange", "banana"],
            ["apple", "banana"],
            False,
            ["orange"],
            "Found 1 row(s) where 'col1' is not in ['apple', 'banana'].",
            "string",
        ),
        (
            "pyspark",
            ["apple", "orange", "banana"],
            ["apple", "banana"],
            False,
            ["orange"],
            "Found 1 row(s) where 'col1' is not in ['apple', 'banana'].",
            "string",
        ),
        # String case sensitivity
        (
            "pandas",
            ["Apple", "apple", "APPLE"],
            ["apple"],
            False,
            ["Apple", "APPLE"],
            "Found 2 row(s) where 'col1' is not in ['apple'].",
            "string",
        ),
        (
            "pyspark",
            ["Apple", "apple", "APPLE"],
            ["apple"],
            False,
            ["Apple", "APPLE"],
            "Found 2 row(s) where 'col1' is not in ['apple'].",
            "string",
        ),
        # Float/Double data type scenarios
        ("pandas", [1.5, 2.5, 3.5], [1.5, 2.5, 3.5], True, None, None, "double"),
        ("pyspark", [1.5, 2.5, 3.5], [1.5, 2.5, 3.5], True, None, None, "double"),
        (
            "pandas",
            [1.5, 4.5, 2.5],
            [1.5, 2.5],
            False,
            [4.5],
            "Found 1 row(s) where 'col1' is not in [1.5, 2.5].",
            "double",
        ),
        (
            "pyspark",
            [1.5, 4.5, 2.5],
            [1.5, 2.5],
            False,
            [4.5],
            "Found 1 row(s) where 'col1' is not in [1.5, 2.5].",
            "double",
        ),
        # Boolean data type scenarios
        ("pandas", [True, False, True], [True, False], True, None, None, "boolean"),
        ("pyspark", [True, False, True], [True, False], True, None, None, "boolean"),
        ("pandas", [True, True, True], [True], True, None, None, "boolean"),
        ("pyspark", [True, True, True], [True], True, None, None, "boolean"),
        (
            "pandas",
            [True, False, True],
            [True],
            False,
            [False],
            "Found 1 row(s) where 'col1' is not in [True].",
            "boolean",
        ),
        (
            "pyspark",
            [True, False, True],
            [True],
            False,
            [False],
            "Found 1 row(s) where 'col1' is not in [True].",
            "boolean",
        ),
        # Timestamp/Datetime scenarios
        (
            "pandas",
            [datetime(2023, 1, 1), datetime(2023, 1, 2), datetime(2023, 1, 3)],
            [datetime(2023, 1, 1), datetime(2023, 1, 2), datetime(2023, 1, 3)],
            True,
            None,
            None,
            "timestamp",
        ),
        (
            "pyspark",
            [datetime(2023, 1, 1), datetime(2023, 1, 2), datetime(2023, 1, 3)],
            [datetime(2023, 1, 1), datetime(2023, 1, 2), datetime(2023, 1, 3)],
            True,
            None,
            None,
            "timestamp",
        ),
        (
            "pandas",
            [datetime(2023, 1, 1), datetime(2023, 1, 4), datetime(2023, 1, 2)],
            [datetime(2023, 1, 1), datetime(2023, 1, 2)],
            False,
            [datetime(2023, 1, 4)],
            "Found 1 row(s) where 'col1' is not in [datetime.datetime(2023, 1, 1, 0, 0), datetime.datetime(2023, 1, 2, 0, 0)].",
            "timestamp",
        ),
        (
            "pyspark",
            [datetime(2023, 1, 1), datetime(2023, 1, 4), datetime(2023, 1, 2)],
            [datetime(2023, 1, 1), datetime(2023, 1, 2)],
            False,
            [datetime(2023, 1, 4)],
            "Found 1 row(s) where 'col1' is not in [datetime.datetime(2023, 1, 1, 0, 0), datetime.datetime(2023, 1, 2, 0, 0)].",
            "timestamp",
        ),
        # Empty string scenarios
        ("pandas", ["", "text", ""], ["", "text"], True, None, None, "string"),
        ("pyspark", ["", "text", ""], ["", "text"], True, None, None, "string"),
        # Zero value scenarios
        ("pandas", [0, 1, 2, 0], [0, 1, 2], True, None, None, "long"),
        ("pyspark", [0, 1, 2, 0], [0, 1, 2], True, None, None, "long"),
        ("pandas", [0.0, 1.0, 2.0], [0.0, 1.0, 2.0], True, None, None, "double"),
        ("pyspark", [0.0, 1.0, 2.0], [0.0, 1.0, 2.0], True, None, None, "double"),
        # Negative numbers
        ("pandas", [-1, -2, -3], [-1, -2, -3], True, None, None, "long"),
        ("pyspark", [-1, -2, -3], [-1, -2, -3], True, None, None, "long"),
        (
            "pandas",
            [-1, 5, -2],
            [-1, -2],
            False,
            [5],
            "Found 1 row(s) where 'col1' is not in [-1, -2].",
            "long",
        ),
        (
            "pyspark",
            [-1, 5, -2],
            [-1, -2],
            False,
            [5],
            "Found 1 row(s) where 'col1' is not in [-1, -2].",
            "long",
        ),
        # Large numbers
        ("pandas", [1000000, 2000000], [1000000, 2000000], True, None, None, "long"),
        ("pyspark", [1000000, 2000000], [1000000, 2000000], True, None, None, "long"),
        # Single value in list
        ("pandas", [5, 5, 5], [5], True, None, None, "long"),
        ("pyspark", [5, 5, 5], [5], True, None, None, "long"),
        # Multiple values with single violation
        (
            "pandas",
            [1, 2, 3, 4, 5],
            [1, 2, 3, 4],
            False,
            [5],
            "Found 1 row(s) where 'col1' is not in [1, 2, 3, 4].",
            "long",
        ),
        (
            "pyspark",
            [1, 2, 3, 4, 5],
            [1, 2, 3, 4],
            False,
            [5],
            "Found 1 row(s) where 'col1' is not in [1, 2, 3, 4].",
            "long",
        ),
    ],
    ids=[
        "pandas_int_basic_success",
        "pyspark_int_basic_success",
        "pandas_int_violations",
        "pyspark_int_violations",
        "pandas_string_success",
        "pyspark_string_success",
        "pandas_string_violations",
        "pyspark_string_violations",
        "pandas_string_case_sensitive",
        "pyspark_string_case_sensitive",
        "pandas_double_success",
        "pyspark_double_success",
        "pandas_double_violations",
        "pyspark_double_violations",
        "pandas_boolean_both_values",
        "pyspark_boolean_both_values",
        "pandas_boolean_single_value",
        "pyspark_boolean_single_value",
        "pandas_boolean_violation",
        "pyspark_boolean_violation",
        "pandas_timestamp_success",
        "pyspark_timestamp_success",
        "pandas_timestamp_violation",
        "pyspark_timestamp_violation",
        "pandas_empty_string",
        "pyspark_empty_string",
        "pandas_zero_int",
        "pyspark_zero_int",
        "pandas_zero_double",
        "pyspark_zero_double",
        "pandas_negative_int_success",
        "pyspark_negative_int_success",
        "pandas_negative_int_violation",
        "pyspark_negative_int_violation",
        "pandas_large_numbers",
        "pyspark_large_numbers",
        "pandas_single_value_list",
        "pyspark_single_value_list",
        "pandas_multiple_values_single_violation",
        "pyspark_multiple_values_single_violation",
    ],
)
def test_expectation_basic_scenarios(
    df_type,
    data,
    values,
    should_succeed,
    expected_violations,
    expected_message,
    data_type,
    spark,
):
    """Test basic expectation scenarios for both pandas and PySpark DataFrames.

    Tests various data types including:
    - Integers (long): positive, negative, zero, large numbers, single/multiple values
    - Strings: case sensitivity, empty strings
    - Floats (double): precision, zero values
    - Booleans: True/False combinations
    - Timestamps: datetime objects
    - Edge cases: single value lists
    """
    df = create_dataframe(df_type, data, "col1", spark, data_type)

    # Test through registry
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationValueIn",
        column_name="col1",
        values=values,
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

        # Verify violations if present
        if expected_violations is not None:
            expected_violations_df = create_dataframe(
                df_type, expected_violations, "col1", spark, data_type
            )
            expected_failure = DataFrameExpectationFailureMessage(
                expectation_str=str(expectation),
                data_frame_type=str(df_type),
                violations_data_frame=expected_violations_df,
                message=expected_message,
                limit_violations=5,
            )
            assert str(result) == str(expected_failure), (
                f"Expected failure details don't match. Got: {result}"
            )

    # Test through suite
    suite = DataFrameExpectationsSuite().expect_value_in(column_name="col1", values=values)

    if should_succeed:
        suite_result = suite.build().run(data_frame=df)
        assert suite_result is None, f"Suite test expected None but got: {suite_result}"
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
        df = pd.DataFrame({"col1": [1, 2, 3]})
    else:
        df = spark.createDataFrame([(1,), (2,), (3,)], ["col1"])

    expected_message = "Column 'nonexistent_col' does not exist in the DataFrame."

    # Test through registry
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationValueIn",
        column_name="nonexistent_col",
        values=[1, 2, 3],
    )
    result = expectation.validate(data_frame=df)
    expected_failure = DataFrameExpectationFailureMessage(
        expectation_str=str(expectation),
        data_frame_type=str(df_type),
        message=expected_message,
    )
    assert str(result) == str(expected_failure)

    # Test through suite
    suite = DataFrameExpectationsSuite().expect_value_in(
        column_name="nonexistent_col", values=[1, 2, 3]
    )
    with pytest.raises(DataFrameExpectationsSuiteFailure):
        suite.build().run(data_frame=df)


def test_large_dataset_performance():
    """Test the expectation with a larger dataset to ensure reasonable performance."""

    # Create a larger dataset with 10,000 rows with values in [1, 2, 3]
    large_data = [1, 2, 3] * 3334  # Creates ~10,000 values
    data_frame = pd.DataFrame({"col1": large_data})

    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationValueIn",
        column_name="col1",
        values=[1, 2, 3],
    )

    result = expectation.validate(data_frame=data_frame)
    assert isinstance(result, DataFrameExpectationSuccessMessage), (
        f"Large dataset test failed: expected success but got {type(result)}"
    )
