import pytest
import numpy as np
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
        expectation_name="ExpectationValueNull",
        column_name="col1",
    )
    assert expectation.get_expectation_name() == "ExpectationValueNull", (
        f"Expected 'ExpectationValueNull' but got: {expectation.get_expectation_name()}"
    )


@pytest.mark.parametrize(
    "df_type, data, should_succeed, expected_violations, expected_message, missing_column, data_type",
    [
        # Basic integer scenarios - success (all nulls)
        ("pandas", [None, None, None], True, None, None, False, "long"),
        ("pyspark", [None, None, None], True, None, None, False, "long"),
        ("pandas", [None, np.nan, None], True, None, None, False, "long"),
        # Integer scenarios - violations (with non-null values)
        (
            "pandas",
            [None, 1, 2],
            False,
            [1, 2],
            "Found 2 row(s) where 'col1' is not null.",
            False,
            "long",
        ),
        (
            "pyspark",
            [None, 1, 2],
            False,
            [1, 2],
            "Found 2 row(s) where 'col1' is not null.",
            False,
            "long",
        ),
        # All non-null values - violations
        (
            "pandas",
            [1, 2, 3],
            False,
            [1, 2, 3],
            "Found 3 row(s) where 'col1' is not null.",
            False,
            "long",
        ),
        (
            "pyspark",
            [1, 2, 3],
            False,
            [1, 2, 3],
            "Found 3 row(s) where 'col1' is not null.",
            False,
            "long",
        ),
        # Single non-null value - violation
        (
            "pandas",
            [None, None, 5],
            False,
            [5],
            "Found 1 row(s) where 'col1' is not null.",
            False,
            "long",
        ),
        (
            "pyspark",
            [None, None, 5],
            False,
            [5],
            "Found 1 row(s) where 'col1' is not null.",
            False,
            "long",
        ),
        # String data type scenarios - success (all nulls)
        ("pandas", [None, None, None], True, None, None, False, "string"),
        ("pyspark", [None, None, None], True, None, None, False, "string"),
        # String scenarios - violations (with non-null values)
        (
            "pandas",
            [None, "apple", "banana"],
            False,
            ["apple", "banana"],
            "Found 2 row(s) where 'col1' is not null.",
            False,
            "string",
        ),
        (
            "pyspark",
            [None, "apple", "banana"],
            False,
            ["apple", "banana"],
            "Found 2 row(s) where 'col1' is not null.",
            False,
            "string",
        ),
        # Empty string is NOT null - violation
        (
            "pandas",
            [None, "", None],
            False,
            [""],
            "Found 1 row(s) where 'col1' is not null.",
            False,
            "string",
        ),
        (
            "pyspark",
            [None, "", None],
            False,
            [""],
            "Found 1 row(s) where 'col1' is not null.",
            False,
            "string",
        ),
        # Whitespace is NOT null - violation
        (
            "pandas",
            [None, " ", None],
            False,
            [" "],
            "Found 1 row(s) where 'col1' is not null.",
            False,
            "string",
        ),
        (
            "pyspark",
            [None, " ", None],
            False,
            [" "],
            "Found 1 row(s) where 'col1' is not null.",
            False,
            "string",
        ),
        # All empty strings - violations (empty strings are NOT null)
        (
            "pandas",
            ["", "", ""],
            False,
            ["", "", ""],
            "Found 3 row(s) where 'col1' is not null.",
            False,
            "string",
        ),
        (
            "pyspark",
            ["", "", ""],
            False,
            ["", "", ""],
            "Found 3 row(s) where 'col1' is not null.",
            False,
            "string",
        ),
        # Float/Double data type scenarios - success (all nulls)
        ("pandas", [None, None, None], True, None, None, False, "double"),
        ("pyspark", [None, None, None], True, None, None, False, "double"),
        ("pandas", [None, np.nan, None], True, None, None, False, "double"),
        # Float scenarios - violations (with non-null values)
        (
            "pandas",
            [None, 1.5, 2.5],
            False,
            [1.5, 2.5],
            "Found 2 row(s) where 'col1' is not null.",
            False,
            "double",
        ),
        (
            "pyspark",
            [None, 1.5, 2.5],
            False,
            [1.5, 2.5],
            "Found 2 row(s) where 'col1' is not null.",
            False,
            "double",
        ),
        # Zero is NOT null - violation
        (
            "pandas",
            [None, 0.0, None],
            False,
            [0.0],
            "Found 1 row(s) where 'col1' is not null.",
            False,
            "double",
        ),
        (
            "pyspark",
            [None, 0.0, None],
            False,
            [0.0],
            "Found 1 row(s) where 'col1' is not null.",
            False,
            "double",
        ),
        (
            "pandas",
            [None, 0, None],
            False,
            [0],
            "Found 1 row(s) where 'col1' is not null.",
            False,
            "long",
        ),
        (
            "pyspark",
            [None, 0, None],
            False,
            [0],
            "Found 1 row(s) where 'col1' is not null.",
            False,
            "long",
        ),
        # Negative numbers are NOT null - violations
        (
            "pandas",
            [None, -5, None],
            False,
            [-5],
            "Found 1 row(s) where 'col1' is not null.",
            False,
            "long",
        ),
        (
            "pyspark",
            [None, -5, None],
            False,
            [-5],
            "Found 1 row(s) where 'col1' is not null.",
            False,
            "long",
        ),
        (
            "pandas",
            [None, -3.14, None],
            False,
            [-3.14],
            "Found 1 row(s) where 'col1' is not null.",
            False,
            "double",
        ),
        (
            "pyspark",
            [None, -3.14, None],
            False,
            [-3.14],
            "Found 1 row(s) where 'col1' is not null.",
            False,
            "double",
        ),
        # Boolean data type scenarios - success (all nulls)
        ("pandas", [None, None, None], True, None, None, False, "boolean"),
        ("pyspark", [None, None, None], True, None, None, False, "boolean"),
        # Boolean scenarios - violations (with True/False)
        (
            "pandas",
            [None, True, False],
            False,
            [True, False],
            "Found 2 row(s) where 'col1' is not null.",
            False,
            "boolean",
        ),
        (
            "pyspark",
            [None, True, False],
            False,
            [True, False],
            "Found 2 row(s) where 'col1' is not null.",
            False,
            "boolean",
        ),
        # False is NOT null - violation
        (
            "pandas",
            [None, False, None],
            False,
            [False],
            "Found 1 row(s) where 'col1' is not null.",
            False,
            "boolean",
        ),
        (
            "pyspark",
            [None, False, None],
            False,
            [False],
            "Found 1 row(s) where 'col1' is not null.",
            False,
            "boolean",
        ),
        # True is NOT null - violation
        (
            "pandas",
            [None, True, None],
            False,
            [True],
            "Found 1 row(s) where 'col1' is not null.",
            False,
            "boolean",
        ),
        (
            "pyspark",
            [None, True, None],
            False,
            [True],
            "Found 1 row(s) where 'col1' is not null.",
            False,
            "boolean",
        ),
        # Timestamp/Datetime scenarios - success (all nulls)
        ("pandas", [None, None, None], True, None, None, False, "timestamp"),
        ("pyspark", [None, None, None], True, None, None, False, "timestamp"),
        # Timestamp scenarios - violations (with datetime values)
        (
            "pandas",
            [None, datetime(2023, 1, 1), datetime(2023, 1, 2)],
            False,
            [datetime(2023, 1, 1), datetime(2023, 1, 2)],
            "Found 2 row(s) where 'col1' is not null.",
            False,
            "timestamp",
        ),
        (
            "pyspark",
            [None, datetime(2023, 1, 1), datetime(2023, 1, 2)],
            False,
            [datetime(2023, 1, 1), datetime(2023, 1, 2)],
            "Found 2 row(s) where 'col1' is not null.",
            False,
            "timestamp",
        ),
        # Single datetime - violation
        (
            "pandas",
            [None, datetime(2023, 1, 1), None],
            False,
            [datetime(2023, 1, 1)],
            "Found 1 row(s) where 'col1' is not null.",
            False,
            "timestamp",
        ),
        (
            "pyspark",
            [None, datetime(2023, 1, 1), None],
            False,
            [datetime(2023, 1, 1)],
            "Found 1 row(s) where 'col1' is not null.",
            False,
            "timestamp",
        ),
        # Datetime with timezone - violation
        (
            "pandas",
            [None, datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc), None],
            False,
            [datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)],
            "Found 1 row(s) where 'col1' is not null.",
            False,
            "timestamp",
        ),
        (
            "pyspark",
            [None, datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc), None],
            False,
            [datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)],
            "Found 1 row(s) where 'col1' is not null.",
            False,
            "timestamp",
        ),
        # Multiple non-nulls in different positions
        (
            "pandas",
            [1, None, 3, None, 5],
            False,
            [1, 3, 5],
            "Found 3 row(s) where 'col1' is not null.",
            False,
            "long",
        ),
        (
            "pyspark",
            [1, None, 3, None, 5],
            False,
            [1, 3, 5],
            "Found 3 row(s) where 'col1' is not null.",
            False,
            "long",
        ),
        # Large numbers - violations (NOT null)
        (
            "pandas",
            [None, 1000000, None],
            False,
            [1000000],
            "Found 1 row(s) where 'col1' is not null.",
            False,
            "long",
        ),
        (
            "pyspark",
            [None, 1000000, None],
            False,
            [1000000],
            "Found 1 row(s) where 'col1' is not null.",
            False,
            "long",
        ),
        # Single null - success
        ("pandas", [None], True, None, None, False, "long"),
        ("pyspark", [None], True, None, None, False, "long"),
        # Single non-null - violation
        (
            "pandas",
            [42],
            False,
            [42],
            "Found 1 row(s) where 'col1' is not null.",
            False,
            "long",
        ),
        (
            "pyspark",
            [42],
            False,
            [42],
            "Found 1 row(s) where 'col1' is not null.",
            False,
            "long",
        ),
        # Missing column scenarios
        (
            "pandas",
            [None, None, None],
            False,
            None,
            "Column 'col1' does not exist in the DataFrame.",
            True,
            "long",
        ),
        (
            "pyspark",
            [None, None, None],
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
    should_succeed,
    expected_violations,
    expected_message,
    missing_column,
    data_type,
    spark,
):
    """Test basic expectation scenarios for both pandas and PySpark DataFrames.

    Tests various data types including:
    - Integers (long): all nulls (success), with non-null values (violations)
    - Strings: all nulls (success), empty strings (NOT null - violation), whitespace (NOT null - violation)
    - Floats (double): all nulls (success), zero values (NOT null - violation), negative values (NOT null - violation)
    - Booleans: all nulls (success), True/False (NOT null - violations)
    - Timestamps: all nulls (success), with datetime values (violations)
    - Edge cases: missing columns, single null, single non-null, multiple non-nulls

    Note: ExpectationValueNull checks that ALL values ARE null.
    Success = all null values, Violations = any non-null values.
    Empty strings, zeros, and False are NOT null and will be violations.
    """
    # Create DataFrame with different column name if testing missing column scenario
    column_name = "col2" if missing_column else "col1"
    df = create_dataframe(df_type, data, column_name, spark, data_type)

    # Test through registry
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationValueNull",
        column_name="col1",
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

        # Verify violations if present (skip detailed DataFrame comparison for pandas None values
        # as representation differs between None and np.nan in display)
        if expected_violations is not None and df_type == "pyspark":
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
    suite = DataFrameExpectationsSuite().expect_value_null(column_name="col1")

    if should_succeed:
        suite_result = suite.build().run(data_frame=df)
        assert suite_result is None, f"Suite test expected None but got: {suite_result}"
    else:
        with pytest.raises(DataFrameExpectationsSuiteFailure):
            suite.build().run(data_frame=df)


def test_large_dataset_performance():
    """Test the expectation with a larger dataset to ensure reasonable performance."""
    # Create a larger dataset with 10,000 rows, all null values
    large_data = [None] * 10000
    data_frame = pd.DataFrame({"col1": large_data})

    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationValueNull",
        column_name="col1",
    )

    result = expectation.validate(data_frame=data_frame)
    assert isinstance(result, DataFrameExpectationSuccessMessage), (
        f"Large dataset test failed: expected success but got {type(result)}"
    )
