import pytest
import pandas as pd
from datetime import datetime, timezone

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


def get_df_type_enum(df_type):
    """Get DataFrameType enum value."""
    return DataFrameType.PANDAS if df_type == "pandas" else DataFrameType.PYSPARK


def test_expectation_name():
    """Test that the expectation name is correctly returned."""
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationValueNotEquals",
        column_name="col1",
        value=5,
    )
    assert expectation.get_expectation_name() == "ExpectationValueNotEquals", (
        f"Expected 'ExpectationValueNotEquals' but got: {expectation.get_expectation_name()}"
    )


@pytest.mark.parametrize(
    "df_type, data,value, should_succeed, expected_violations, expected_message, missing_column, data_type",
    [
        # Basic integer scenarios - success (values NOT equal to 5)
        ("pandas", [3, 4, 6], 5, True, None, None, False, "long"),
        ("pyspark", [3, 4, 6], 5, True, None, None, False, "long"),
        ("pandas", [1, 2, 3], 5, True, None, None, False, "long"),
        ("pyspark", [1, 2, 3], 5, True, None, None, False, "long"),
        # Integer scenarios - violations (values equal to 5)
        (
            "pandas",
            [3, 5, 5],
            5,
            False,
            [5, 5],
            "Found 2 row(s) where 'col1' is equal to 5.",
            False,
            "long",
        ),
        (
            "pyspark",
            [3, 5, 5],
            5,
            False,
            [5, 5],
            "Found 2 row(s) where 'col1' is equal to 5.",
            False,
            "long",
        ),
        (
            "pandas",
            [5, 5, 5],
            5,
            False,
            [5, 5, 5],
            "Found 3 row(s) where 'col1' is equal to 5.",
            False,
            "long",
        ),
        (
            "pyspark",
            [5, 5, 5],
            5,
            False,
            [5, 5, 5],
            "Found 3 row(s) where 'col1' is equal to 5.",
            False,
            "long",
        ),
        # String data type scenarios - success
        ("pandas", ["banana", "cherry", "orange"], "apple", True, None, None, False, "string"),
        ("pyspark", ["banana", "cherry", "orange"], "apple", True, None, None, False, "string"),
        # String scenarios - violations
        (
            "pandas",
            ["apple", "banana", "apple"],
            "apple",
            False,
            ["apple", "apple"],
            "Found 2 row(s) where 'col1' is equal to apple.",
            False,
            "string",
        ),
        (
            "pyspark",
            ["apple", "banana", "apple"],
            "apple",
            False,
            ["apple", "apple"],
            "Found 2 row(s) where 'col1' is equal to apple.",
            False,
            "string",
        ),
        # String case sensitivity - success (case matters)
        ("pandas", ["Apple", "APPLE", "aPpLe"], "apple", True, None, None, False, "string"),
        ("pyspark", ["Apple", "APPLE", "aPpLe"], "apple", True, None, None, False, "string"),
        # String case sensitivity - violations (exact match)
        (
            "pandas",
            ["apple", "Apple", "apple"],
            "apple",
            False,
            ["apple", "apple"],
            "Found 2 row(s) where 'col1' is equal to apple.",
            False,
            "string",
        ),
        (
            "pyspark",
            ["apple", "Apple", "apple"],
            "apple",
            False,
            ["apple", "apple"],
            "Found 2 row(s) where 'col1' is equal to apple.",
            False,
            "string",
        ),
        # Float/Double data type scenarios - success
        ("pandas", [1.5, 2.5, 4.5], 3.14, True, None, None, False, "double"),
        ("pyspark", [1.5, 2.5, 4.5], 3.14, True, None, None, False, "double"),
        # Float scenarios - violations
        (
            "pandas",
            [3.14, 2.71, 3.14],
            3.14,
            False,
            [3.14, 3.14],
            "Found 2 row(s) where 'col1' is equal to 3.14.",
            False,
            "double",
        ),
        (
            "pyspark",
            [3.14, 2.71, 3.14],
            3.14,
            False,
            [3.14, 3.14],
            "Found 2 row(s) where 'col1' is equal to 3.14.",
            False,
            "double",
        ),
        # Float precision edge cases - success
        ("pandas", [1.1, 1.2, 1.3], 1.0, True, None, None, False, "double"),
        ("pyspark", [1.1, 1.2, 1.3], 1.0, True, None, None, False, "double"),
        # Float precision - violations
        (
            "pandas",
            [1.0, 1.1, 1.0],
            1.0,
            False,
            [1.0, 1.0],
            "Found 2 row(s) where 'col1' is equal to 1.0.",
            False,
            "double",
        ),
        # Boolean data type scenarios - success
        ("pandas", [False, False, False], True, True, None, None, False, "boolean"),
        ("pyspark", [False, False, False], True, True, None, None, False, "boolean"),
        ("pandas", [True, True, True], False, True, None, None, False, "boolean"),
        ("pyspark", [True, True, True], False, True, None, None, False, "boolean"),
        # Boolean scenarios - violations
        (
            "pandas",
            [True, False, True],
            True,
            False,
            [True, True],
            "Found 2 row(s) where 'col1' is equal to True.",
            False,
            "boolean",
        ),
        (
            "pyspark",
            [True, False, True],
            True,
            False,
            [True, True],
            "Found 2 row(s) where 'col1' is equal to True.",
            False,
            "boolean",
        ),
        (
            "pandas",
            [False, True, False],
            False,
            False,
            [False, False],
            "Found 2 row(s) where 'col1' is equal to False.",
            False,
            "boolean",
        ),
        (
            "pyspark",
            [False, True, False],
            False,
            False,
            [False, False],
            "Found 2 row(s) where 'col1' is equal to False.",
            False,
            "boolean",
        ),
        # Timestamp/Datetime scenarios - success
        (
            "pandas",
            [datetime(2023, 1, 2), datetime(2023, 1, 3), datetime(2023, 1, 4)],
            datetime(2023, 1, 1),
            True,
            None,
            None,
            False,
            "timestamp",
        ),
        (
            "pyspark",
            [datetime(2023, 1, 2), datetime(2023, 1, 3), datetime(2023, 1, 4)],
            datetime(2023, 1, 1),
            True,
            None,
            None,
            False,
            "timestamp",
        ),
        # Timestamp scenarios - violations
        (
            "pandas",
            [datetime(2023, 1, 1), datetime(2023, 1, 2), datetime(2023, 1, 1)],
            datetime(2023, 1, 1),
            False,
            [datetime(2023, 1, 1), datetime(2023, 1, 1)],
            "Found 2 row(s) where 'col1' is equal to 2023-01-01 00:00:00.",
            False,
            "timestamp",
        ),
        (
            "pyspark",
            [datetime(2023, 1, 1), datetime(2023, 1, 2), datetime(2023, 1, 1)],
            datetime(2023, 1, 1),
            False,
            [datetime(2023, 1, 1), datetime(2023, 1, 1)],
            "Found 2 row(s) where 'col1' is equal to 2023-01-01 00:00:00.",
            False,
            "timestamp",
        ),
        # Datetime with timezone - success
        (
            "pandas",
            [
                datetime(2023, 1, 2, 12, 0, 0, tzinfo=timezone.utc),
                datetime(2023, 1, 3, 12, 0, 0, tzinfo=timezone.utc),
            ],
            datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            True,
            None,
            None,
            False,
            "timestamp",
        ),
        (
            "pyspark",
            [
                datetime(2023, 1, 2, 12, 0, 0, tzinfo=timezone.utc),
                datetime(2023, 1, 3, 12, 0, 0, tzinfo=timezone.utc),
            ],
            datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            True,
            None,
            None,
            False,
            "timestamp",
        ),
        # Empty string scenarios - success
        ("pandas", ["text", "data", "value"], "", True, None, None, False, "string"),
        ("pyspark", ["text", "data", "value"], "", True, None, None, False, "string"),
        # Empty string - violations
        (
            "pandas",
            ["", "text", ""],
            "",
            False,
            ["", ""],
            "Found 2 row(s) where 'col1' is equal to .",
            False,
            "string",
        ),
        (
            "pyspark",
            ["", "text", ""],
            "",
            False,
            ["", ""],
            "Found 2 row(s) where 'col1' is equal to .",
            False,
            "string",
        ),
        # Whitespace in strings - success
        ("pandas", [" test", "test ", " test "], "test", True, None, None, False, "string"),
        ("pyspark", [" test", "test ", " test "], "test", True, None, None, False, "string"),
        # Zero value scenarios - success
        ("pandas", [1, 2, 3], 0, True, None, None, False, "long"),
        ("pyspark", [1, 2, 3], 0, True, None, None, False, "long"),
        ("pandas", [1.5, 2.5, 3.5], 0.0, True, None, None, False, "double"),
        ("pyspark", [1.5, 2.5, 3.5], 0.0, True, None, None, False, "double"),
        # Zero value - violations
        (
            "pandas",
            [0, 1, 0],
            0,
            False,
            [0, 0],
            "Found 2 row(s) where 'col1' is equal to 0.",
            False,
            "long",
        ),
        (
            "pyspark",
            [0, 1, 0],
            0,
            False,
            [0, 0],
            "Found 2 row(s) where 'col1' is equal to 0.",
            False,
            "long",
        ),
        # Negative numbers - success
        ("pandas", [1, 2, 3], -5, True, None, None, False, "long"),
        ("pyspark", [1, 2, 3], -5, True, None, None, False, "long"),
        ("pandas", [1.5, 2.5, 3.5], -3.14, True, None, None, False, "double"),
        ("pyspark", [1.5, 2.5, 3.5], -3.14, True, None, None, False, "double"),
        # Negative numbers - violations
        (
            "pandas",
            [-5, 1, -5],
            -5,
            False,
            [-5, -5],
            "Found 2 row(s) where 'col1' is equal to -5.",
            False,
            "long",
        ),
        (
            "pyspark",
            [-5, 1, -5],
            -5,
            False,
            [-5, -5],
            "Found 2 row(s) where 'col1' is equal to -5.",
            False,
            "long",
        ),
        # Large numbers - success
        ("pandas", [999999, 1000001], 1000000, True, None, None, False, "long"),
        ("pyspark", [999999, 1000001], 1000000, True, None, None, False, "long"),
        # Large numbers - violations
        (
            "pandas",
            [1000000, 999999, 1000000],
            1000000,
            False,
            [1000000, 1000000],
            "Found 2 row(s) where 'col1' is equal to 1000000.",
            False,
            "long",
        ),
        (
            "pyspark",
            [1000000, 999999, 1000000],
            1000000,
            False,
            [1000000, 1000000],
            "Found 2 row(s) where 'col1' is equal to 1000000.",
            False,
            "long",
        ),
        # Missing column scenarios
        (
            "pandas",
            [3, 4, 5],
            5,
            False,
            None,
            "Column 'col1' does not exist in the DataFrame.",
            True,
            "long",
        ),
        (
            "pyspark",
            [3, 4, 5],
            5,
            False,
            None,
            "Column 'col1' does not exist in the DataFrame.",
            True,
            "long",
        ),
    ],
)
def test_expectation_basic_scenarios(
    df_type,
    data,
    value,
    should_succeed,
    expected_violations,
    expected_message,
    missing_column,
    data_type,
    spark,
):
    """Test basic expectation scenarios for both pandas and PySpark DataFrames.

    Tests various data types including:
    - Integers (long): positive, negative, zero, large numbers
    - Strings: case sensitivity, empty strings, whitespace
    - Floats (double): precision, zero, negative
    - Booleans: True/False
    - Timestamps: with and without timezone
    - Edge cases: missing columns

    Note: ExpectationValueNotEquals checks that values are NOT equal to the target.
    Success = all values differ from target, Violations = values that equal target.
    """
    # Create DataFrame with different column name if testing missing column scenario
    column_name = "col2" if missing_column else "col1"
    df = create_dataframe(df_type, data, column_name, spark, data_type)

    # Test through registry
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationValueNotEquals",
        column_name="col1",
        value=value,
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
                data_frame_type=get_df_type_enum(df_type),
                violations_data_frame=expected_violations_df,
                message=expected_message,
                limit_violations=5,
            )
            assert str(result) == str(expected_failure), (
                f"Expected failure details don't match. Got: {result}"
            )

    # Test through suite
    suite = DataFrameExpectationsSuite().expect_value_not_equals(column_name="col1", value=value)

    if should_succeed:
        suite_result = suite.build().run(data_frame=df)
        assert suite_result is None, f"Suite test expected None but got: {suite_result}"
    else:
        with pytest.raises(DataFrameExpectationsSuiteFailure):
            suite.build().run(data_frame=df)


def test_large_dataset_performance():
    """Test the expectation with a larger dataset to ensure reasonable performance."""

    # Create a larger dataset with 10,000 rows, all NOT equal to 42 (using 43)
    large_data = [43] * 10000
    data_frame = pd.DataFrame({"col1": large_data})

    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationValueNotEquals",
        column_name="col1",
        value=42,
    )

    result = expectation.validate(data_frame=data_frame)
    assert isinstance(result, DataFrameExpectationSuccessMessage), (
        f"Large dataset test failed: expected success but got {type(result)}"
    )
