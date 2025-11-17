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
        expectation_name="ExpectationValueEquals",
        column_name="col1",
        value=5,
    )
    assert expectation.get_expectation_name() == "ExpectationValueEquals", (
        f"Expected 'ExpectationValueEquals' but got: {expectation.get_expectation_name()}"
    )


@pytest.mark.parametrize(
    "df_type, data, value, should_succeed, expected_violations, expected_message, missing_column, data_type",
    [
        # Basic integer scenarios
        ("pandas", [5, 5, 5], 5, True, None, None, False, "long"),
        ("pandas", [10, 10], 10, True, None, None, False, "long"),
        ("pyspark", [5, 5, 5], 5, True, None, None, False, "long"),
        ("pyspark", [10, 10], 10, True, None, None, False, "long"),
        # Integer failure scenarios with violations
        (
            "pandas",
            [3, 4, 5],
            5,
            False,
            [3, 4],
            "Found 2 row(s) where 'col1' is not equal to 5.",
            False,
            "long",
        ),
        (
            "pyspark",
            [3, 4, 5],
            5,
            False,
            [3, 4],
            "Found 2 row(s) where 'col1' is not equal to 5.",
            False,
            "long",
        ),
        # String data type scenarios
        ("pandas", ["apple", "apple", "apple"], "apple", True, None, None, False, "string"),
        ("pyspark", ["apple", "apple", "apple"], "apple", True, None, None, False, "string"),
        (
            "pandas",
            ["apple", "banana", "apple"],
            "apple",
            False,
            ["banana"],
            "Found 1 row(s) where 'col1' is not equal to apple.",
            False,
            "string",
        ),
        (
            "pyspark",
            ["apple", "banana", "apple"],
            "apple",
            False,
            ["banana"],
            "Found 1 row(s) where 'col1' is not equal to apple.",
            False,
            "string",
        ),
        # String case sensitivity
        (
            "pandas",
            ["Apple", "APPLE", "apple"],
            "apple",
            False,
            ["Apple", "APPLE"],
            "Found 2 row(s) where 'col1' is not equal to apple.",
            False,
            "string",
        ),
        (
            "pyspark",
            ["Apple", "APPLE", "apple"],
            "apple",
            False,
            ["Apple", "APPLE"],
            "Found 2 row(s) where 'col1' is not equal to apple.",
            False,
            "string",
        ),
        # Float/Double data type scenarios
        ("pandas", [3.14, 3.14, 3.14], 3.14, True, None, None, False, "double"),
        ("pyspark", [3.14, 3.14, 3.14], 3.14, True, None, None, False, "double"),
        (
            "pandas",
            [3.14, 2.71, 3.14],
            3.14,
            False,
            [2.71],
            "Found 1 row(s) where 'col1' is not equal to 3.14.",
            False,
            "double",
        ),
        (
            "pyspark",
            [3.14, 2.71, 3.14],
            3.14,
            False,
            [2.71],
            "Found 1 row(s) where 'col1' is not equal to 3.14.",
            False,
            "double",
        ),
        # Float precision edge cases
        ("pandas", [1.0, 1.0, 1.0], 1.0, True, None, None, False, "double"),
        ("pyspark", [1.0, 1.0, 1.0], 1.0, True, None, None, False, "double"),
        (
            "pandas",
            [1.0, 1.1, 1.0],
            1.0,
            False,
            [1.1],
            "Found 1 row(s) where 'col1' is not equal to 1.0.",
            False,
            "double",
        ),
        # Boolean data type scenarios
        ("pandas", [True, True, True], True, True, None, None, False, "boolean"),
        ("pyspark", [True, True, True], True, True, None, None, False, "boolean"),
        ("pandas", [False, False, False], False, True, None, None, False, "boolean"),
        ("pyspark", [False, False, False], False, True, None, None, False, "boolean"),
        (
            "pandas",
            [True, False, True],
            True,
            False,
            [False],
            "Found 1 row(s) where 'col1' is not equal to True.",
            False,
            "boolean",
        ),
        (
            "pyspark",
            [True, False, True],
            True,
            False,
            [False],
            "Found 1 row(s) where 'col1' is not equal to True.",
            False,
            "boolean",
        ),
        # Timestamp/Datetime scenarios
        (
            "pandas",
            [datetime(2023, 1, 1), datetime(2023, 1, 1), datetime(2023, 1, 1)],
            datetime(2023, 1, 1),
            True,
            None,
            None,
            False,
            "timestamp",
        ),
        (
            "pyspark",
            [datetime(2023, 1, 1), datetime(2023, 1, 1), datetime(2023, 1, 1)],
            datetime(2023, 1, 1),
            True,
            None,
            None,
            False,
            "timestamp",
        ),
        (
            "pandas",
            [datetime(2023, 1, 1), datetime(2023, 1, 2), datetime(2023, 1, 1)],
            datetime(2023, 1, 1),
            False,
            [datetime(2023, 1, 2)],
            "Found 1 row(s) where 'col1' is not equal to 2023-01-01 00:00:00.",
            False,
            "timestamp",
        ),
        (
            "pyspark",
            [datetime(2023, 1, 1), datetime(2023, 1, 2), datetime(2023, 1, 1)],
            datetime(2023, 1, 1),
            False,
            [datetime(2023, 1, 2)],
            "Found 1 row(s) where 'col1' is not equal to 2023-01-01 00:00:00.",
            False,
            "timestamp",
        ),
        # Datetime with timezone
        (
            "pandas",
            [
                datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
                datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
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
                datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
                datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            ],
            datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            True,
            None,
            None,
            False,
            "timestamp",
        ),
        # Empty string vs non-empty
        ("pandas", ["", "", ""], "", True, None, None, False, "string"),
        ("pyspark", ["", "", ""], "", True, None, None, False, "string"),
        (
            "pandas",
            ["", "text", ""],
            "",
            False,
            ["text"],
            "Found 1 row(s) where 'col1' is not equal to .",
            False,
            "string",
        ),
        # Whitespace in strings
        (
            "pandas",
            ["test", " test", "test"],
            "test",
            False,
            [" test"],
            "Found 1 row(s) where 'col1' is not equal to test.",
            False,
            "string",
        ),
        (
            "pyspark",
            ["test", " test", "test"],
            "test",
            False,
            [" test"],
            "Found 1 row(s) where 'col1' is not equal to test.",
            False,
            "string",
        ),
        # Zero value scenarios
        ("pandas", [0, 0, 0], 0, True, None, None, False, "long"),
        ("pyspark", [0, 0, 0], 0, True, None, None, False, "long"),
        ("pandas", [0.0, 0.0, 0.0], 0.0, True, None, None, False, "double"),
        ("pyspark", [0.0, 0.0, 0.0], 0.0, True, None, None, False, "double"),
        # Negative numbers
        ("pandas", [-5, -5, -5], -5, True, None, None, False, "long"),
        ("pyspark", [-5, -5, -5], -5, True, None, None, False, "long"),
        ("pandas", [-3.14, -3.14, -3.14], -3.14, True, None, None, False, "double"),
        ("pyspark", [-3.14, -3.14, -3.14], -3.14, True, None, None, False, "double"),
        # Large numbers
        ("pandas", [1000000, 1000000], 1000000, True, None, None, False, "long"),
        ("pyspark", [1000000, 1000000], 1000000, True, None, None, False, "long"),
        # Missing column scenarios
        (
            "pandas",
            [5, 5, 5],
            5,
            False,
            None,
            "Column 'col1' does not exist in the DataFrame.",
            True,
            "long",
        ),
        (
            "pyspark",
            [5, 5, 5],
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
    """
    # Create DataFrame with different column name if testing missing column scenario
    column_name = "col2" if missing_column else "col1"
    df = create_dataframe(df_type, data, column_name, spark, data_type)

    # Test through registry
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationValueEquals",
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
    suite = DataFrameExpectationsSuite().expect_value_equals(column_name="col1", value=value)

    if should_succeed:
        suite_result = suite.build().run(data_frame=df)
        assert suite_result is None, f"Suite test expected None but got: {suite_result}"
    else:
        with pytest.raises(DataFrameExpectationsSuiteFailure):
            suite.build().run(data_frame=df)


def test_large_dataset_performance():
    """Test the expectation with a larger dataset to ensure reasonable performance."""

    # Create a larger dataset with 10,000 rows all equal to 42
    large_data = [42] * 10000
    data_frame = pd.DataFrame({"col1": large_data})

    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationValueEquals",
        column_name="col1",
        value=42,
    )

    result = expectation.validate(data_frame=data_frame)
    assert isinstance(result, DataFrameExpectationSuccessMessage), (
        f"Large dataset test failed: expected success but got {type(result)}"
    )
