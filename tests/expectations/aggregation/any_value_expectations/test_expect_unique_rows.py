import pytest
import pandas as pd
from pyspark.sql.types import IntegerType, StructField, StructType

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
    from pyspark.sql.types import StructType, StructField, IntegerType

    if not df_data:
        schema = StructType([StructField("col1", IntegerType(), True)])
        return spark.createDataFrame([], schema)

    # For PySpark, df_data is (rows, columns) tuple
    rows, columns = df_data
    return spark.createDataFrame(rows, columns)


def get_df_type_enum(df_type):
    """Get DataFrameType enum value."""
    return DataFrameType.PANDAS if df_type == "pandas" else DataFrameType.PYSPARK


def test_expectation_name():
    """Test that the expectation name is correctly returned."""
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationUniqueRows",
        column_names=["col1"],
    )
    assert expectation.get_expectation_name() == "ExpectationUniqueRows", (
        f"Expected 'ExpectationUniqueRows' but got: {expectation.get_expectation_name()}"
    )


@pytest.mark.parametrize(
    "df_type, df_data, column_names, expected_result, expected_violations, expected_message",
    [
        # Pandas success - specific columns (unique combinations)
        (
            "pandas",
            {"col1": [1, 2, 3, 1], "col2": [10, 20, 30, 20], "col3": [100, 100, 100, 100]},
            ["col1", "col2"],
            "success",
            None,
            None,
        ),
        # Pandas success - all columns (empty list)
        (
            "pandas",
            {"col1": [1, 2, 3], "col2": [10, 20, 30], "col3": [100, 200, 300]},
            [],
            "success",
            None,
            None,
        ),
        # Pandas success - empty DataFrame
        (
            "pandas",
            {"col1": []},
            ["col1"],
            "success",
            None,
            None,
        ),
        # Pandas success - single row
        (
            "pandas",
            {"col1": [1]},
            ["col1"],
            "success",
            None,
            None,
        ),
        # Pandas failure - specific columns with duplicates
        (
            "pandas",
            {"col1": [1, 2, 1, 3], "col2": [10, 20, 10, 30], "col3": [100, 200, 300, 400]},
            ["col1", "col2"],
            "failure",
            pd.DataFrame({"col1": [1], "col2": [10], "#duplicates": [2]}),
            "Found 2 duplicate row(s). duplicate rows found for columns ['col1', 'col2']",
        ),
        # Pandas failure - all columns with duplicates
        (
            "pandas",
            {"col1": [1, 2, 1], "col2": [10, 20, 10], "col3": [100, 200, 100]},
            [],
            "failure",
            pd.DataFrame({"col1": [1], "col2": [10], "col3": [100], "#duplicates": [2]}),
            "Found 2 duplicate row(s). duplicate rows found",
        ),
        # Pandas failure - multiple duplicate groups
        (
            "pandas",
            {"col1": [1, 2, 1, 3, 2, 3], "col2": [10, 20, 30, 40, 50, 60]},
            ["col1"],
            "failure",
            pd.DataFrame({"col1": [1, 2, 3], "#duplicates": [2, 2, 2]}),
            "Found 6 duplicate row(s). duplicate rows found for columns ['col1']",
        ),
        # Pandas failure - with nulls (nulls counted as duplicates)
        (
            "pandas",
            {"col1": [1, None, 1, None], "col2": [10, None, 20, None]},
            ["col1", "col2"],
            "failure",
            pd.DataFrame({"col1": [None], "col2": [None], "#duplicates": [2]}),
            "Found 2 duplicate row(s). duplicate rows found for columns ['col1', 'col2']",
        ),
        # PySpark success - specific columns
        (
            "pyspark",
            ([(1, 10, 100), (2, 20, 100), (3, 30, 100), (1, 20, 100)], ["col1", "col2", "col3"]),
            ["col1", "col2"],
            "success",
            None,
            None,
        ),
        # PySpark success - all columns
        (
            "pyspark",
            ([(1, 10, 100), (2, 20, 200), (3, 30, 300)], ["col1", "col2", "col3"]),
            [],
            "success",
            None,
            None,
        ),
        # PySpark success - empty DataFrame
        (
            "pyspark",
            ([], "col1: int"),
            ["col1"],
            "success",
            None,
            None,
        ),
        # PySpark success - single row
        (
            "pyspark",
            ([(1,)], ["col1"]),
            ["col1"],
            "success",
            None,
            None,
        ),
        # PySpark failure - specific columns with duplicates
        (
            "pyspark",
            ([(1, 10, 100), (2, 20, 200), (1, 10, 300), (3, 30, 400)], ["col1", "col2", "col3"]),
            ["col1", "col2"],
            "failure",
            ([(1, 10, 2)], ["col1", "col2", "#duplicates"]),
            "Found 2 duplicate row(s). duplicate rows found for columns ['col1', 'col2']",
        ),
        # PySpark failure - all columns with duplicates
        (
            "pyspark",
            ([(1, 10, 100), (2, 20, 200), (1, 10, 100)], ["col1", "col2", "col3"]),
            [],
            "failure",
            ([(1, 10, 100, 2)], ["col1", "col2", "col3", "#duplicates"]),
            "Found 2 duplicate row(s). duplicate rows found",
        ),
        # PySpark failure - multiple duplicate groups
        (
            "pyspark",
            ([(1, 10), (2, 20), (1, 30), (3, 40), (2, 50), (3, 60)], ["col1", "col2"]),
            ["col1"],
            "failure",
            ([(1, 2), (2, 2), (3, 2)], ["col1", "#duplicates"]),
            "Found 6 duplicate row(s). duplicate rows found for columns ['col1']",
        ),
        # PySpark failure - with nulls
        (
            "pyspark",
            ([(1, 10), (None, None), (1, 20), (None, None)], ["col1", "col2"]),
            ["col1", "col2"],
            "failure",
            (
                StructType(
                    [
                        StructField("col1", IntegerType(), True),
                        StructField("col2", IntegerType(), True),
                        StructField("#duplicates", IntegerType(), True),
                    ]
                ),
                [(None, None, 2)],
            ),
            "Found 2 duplicate row(s). duplicate rows found for columns ['col1', 'col2']",
        ),
    ],
    ids=[
        "pandas_success_specific_columns",
        "pandas_success_all_columns",
        "pandas_empty",
        "pandas_single_row",
        "pandas_violations_specific_columns",
        "pandas_violations_all_columns",
        "pandas_multiple_duplicate_groups",
        "pandas_with_nulls",
        "pyspark_success_specific_columns",
        "pyspark_success_all_columns",
        "pyspark_empty",
        "pyspark_single_row",
        "pyspark_violations_specific_columns",
        "pyspark_violations_all_columns",
        "pyspark_multiple_duplicate_groups",
        "pyspark_with_nulls",
    ],
)
def test_expectation_basic_scenarios(
    df_type, df_data, column_names, expected_result, expected_violations, expected_message, spark
):
    """
    Test the expectation for various scenarios across pandas and PySpark DataFrames.
    Tests both direct expectation validation and suite-based validation.
    Covers: success cases (specific columns, all columns, empty, single row),
    failure cases (duplicates on specific/all columns, multiple groups, with nulls).
    """
    data_frame = create_dataframe(df_type, df_data, spark)

    # Test 1: Direct expectation validation
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationUniqueRows",
        column_names=column_names,
    )

    result = expectation.validate(data_frame=data_frame)

    if expected_result == "success":
        assert str(result) == str(
            DataFrameExpectationSuccessMessage(expectation_name="ExpectationUniqueRows")
        ), f"Expected success message but got: {result}"
    else:  # failure
        # Create violations DataFrame
        if df_type == "pandas":
            violations_df = expected_violations
        else:  # pyspark
            if isinstance(expected_violations[0], StructType):
                # Schema and data provided separately
                schema, data = expected_violations
                violations_df = spark.createDataFrame(data, schema)
            else:
                # Rows and columns provided
                rows, columns = expected_violations
                violations_df = spark.createDataFrame(rows, columns)

        expected_failure_message = DataFrameExpectationFailureMessage(
            expectation_str=str(expectation),
            data_frame_type=get_df_type_enum(df_type),
            violations_data_frame=violations_df,
            message=expected_message,
            limit_violations=5,
        )
        assert str(result) == str(expected_failure_message), (
            f"Expected failure message but got: {result}"
        )

    # Test 2: Suite-based validation
    expectations_suite = DataFrameExpectationsSuite().expect_unique_rows(column_names=column_names)

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
    """
    Test that an error is raised when specified columns are missing.
    Tests both direct expectation validation and suite-based validation.
    """
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationUniqueRows",
        column_names=["nonexistent_col"],
    )

    if df_type == "pandas":
        data_frame = pd.DataFrame({"col1": [1, 2, 3]})
    else:  # pyspark
        data_frame = spark.createDataFrame([(1,), (2,), (3,)], ["col1"])

    # Test 1: Direct expectation validation
    result = expectation.validate(data_frame=data_frame)
    expected_failure_message = DataFrameExpectationFailureMessage(
        expectation_str=str(expectation),
        data_frame_type=get_df_type_enum(df_type),
        message="Column 'nonexistent_col' does not exist in the DataFrame.",
    )
    assert str(result) == str(expected_failure_message), (
        f"Expected failure message but got: {result}"
    )

    # Test 2: Suite-based validation
    expectations_suite = DataFrameExpectationsSuite().expect_unique_rows(
        column_names=["nonexistent_col"]
    )
    with pytest.raises(DataFrameExpectationsSuiteFailure):
        expectations_suite.build().run(data_frame=data_frame)
