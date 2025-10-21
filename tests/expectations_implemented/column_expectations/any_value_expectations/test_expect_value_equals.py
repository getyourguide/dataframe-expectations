import pytest
import pandas as pd

from dataframe_expectations import DataFrameType
from dataframe_expectations.expectations.expectation_registry import (
    DataFrameExpectationRegistry,
)
from dataframe_expectations.expectations_suite import (
    DataFrameExpectationsSuite,
    DataFrameExpectationsSuiteFailure,
)
from dataframe_expectations.result_message import (
    DataFrameExpectationFailureMessage,
    DataFrameExpectationSuccessMessage,
)


def test_expectation_name():
    """
    Test that the expectation name is correctly returned.
    """
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationValueEquals",
        column_name="col1",
        value=5,
    )
    assert (
        expectation.get_expectation_name() == "ExpectationValueEquals"
    ), f"Expected 'ExpectationValueEquals' but got: {expectation.get_expectation_name()}"


def test_expectation_pandas_success():
    """
    Test the expectation for pandas DataFrame with no violations.
    """
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationValueEquals",
        column_name="col1",
        value=5,
    )
    data_frame = pd.DataFrame({"col1": [5, 5, 5]})
    result = expectation.validate(data_frame=data_frame)
    assert str(result) == str(
        DataFrameExpectationSuccessMessage(expectation_name="ExpectationValueEquals")
    ), f"Expected success message but got: {result}"


def test_expectation_pandas_violations():
    """
    Test the expectation for pandas DataFrame with violations.
    This method should be implemented in the subclass.
    """
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationValueEquals",
        column_name="col1",
        value=5,
    )
    data_frame = pd.DataFrame({"col1": [3, 4, 5]})
    result = expectation.validate(data_frame=data_frame)

    expected_violations = pd.DataFrame({"col1": [3, 4]})
    assert str(result) == str(
        DataFrameExpectationFailureMessage(
            expectation_str=str(expectation),
            data_frame_type=DataFrameType.PANDAS,
            violations_data_frame=expected_violations,
            message="Found 2 row(s) where 'col1' is not equal to 5.",
            limit_violations=5,
        )
    ), f"Expected failure message but got: {result}"


def test_expectation_pyspark_success(spark):
    """
    Test the expectation for PySpark DataFrame with no violations.
    """
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationValueEquals",
        column_name="col1",
        value=5,
    )
    data_frame = spark.createDataFrame([(5,), (5,), (5,)], ["col1"])
    result = expectation.validate(data_frame=data_frame)
    assert str(result) == str(
        DataFrameExpectationSuccessMessage(expectation_name="ExpectationValueEquals")
    ), f"Expected success message but got: {result}"


def test_expectation_pyspark_violations(spark):
    """
    Test the expectation for PySpark DataFrame with violations.
    """
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationValueEquals",
        column_name="col1",
        value=5,
    )
    data_frame = spark.createDataFrame([(3,), (4,), (5,)], ["col1"])
    result = expectation.validate(data_frame=data_frame)

    expected_violations = spark.createDataFrame([(3,), (4,)], ["col1"])
    assert str(result) == str(
        DataFrameExpectationFailureMessage(
            expectation_str=str(expectation),
            data_frame_type=DataFrameType.PYSPARK,
            violations_data_frame=expected_violations,
            message="Found 2 row(s) where 'col1' is not equal to 5.",
            limit_violations=5,
        )
    ), f"Expected failure message but got: {result}"


def test_column_missing_error():
    """
    Test that an error is raised when the specified column is missing.
    """
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationValueEquals",
        column_name="col1",
        value=5,
    )
    data_frame = pd.DataFrame({"col2": [5, 5, 5]})

    result = expectation.validate(data_frame=data_frame)
    expected_failure_message = DataFrameExpectationFailureMessage(
        expectation_str=str(expectation),
        data_frame_type=DataFrameType.PANDAS,
        message="Column 'col1' does not exist in the DataFrame.",
    )

    assert str(result) == str(
        expected_failure_message
    ), f"Expected failure message but got: {result}"


def test_suite_pandas_success():
    """
    Test the expectation suite for pandas DataFrame with no violations.
    """
    expectations_suite = DataFrameExpectationsSuite().expect_value_equals(
        column_name="col1", value=5
    )
    data_frame = pd.DataFrame({"col1": [5, 5, 5]})
    result = expectations_suite.run(data_frame=data_frame)
    assert result is None, "Expected no exceptions to be raised"


def test_suite_pandas_violations():
    """
    Test the expectation suite for pandas DataFrame with violations.
    """
    expectations_suite = DataFrameExpectationsSuite().expect_value_equals(
        column_name="col1", value=5
    )
    data_frame = pd.DataFrame({"col1": [3, 4, 5]})
    with pytest.raises(DataFrameExpectationsSuiteFailure):
        expectations_suite.run(data_frame=data_frame)


def test_suite_pyspark_success(spark):
    """
    Test the expectation suite for PySpark DataFrame with no violations.
    """
    expectations_suite = DataFrameExpectationsSuite().expect_value_equals(
        column_name="col1", value=5
    )
    data_frame = spark.createDataFrame([(5,), (5,), (5,)], ["col1"])
    result = expectations_suite.run(data_frame=data_frame)
    assert result is None, "Expected no exceptions to be raised"


def test_suite_pyspark_violations(spark):
    """
    Test the expectation suite for PySpark DataFrame with violations.
    """
    expectations_suite = DataFrameExpectationsSuite().expect_value_equals(
        column_name="col1", value=5
    )
    data_frame = spark.createDataFrame([(3,), (4,), (5,)], ["col1"])
    with pytest.raises(DataFrameExpectationsSuiteFailure):
        expectations_suite.run(data_frame=data_frame)


def test_suite_pyspark_column_missing_error(spark):
    """
    Test that an error is raised when the specified column is missing in PySpark DataFrame.
    """
    expectations_suite = DataFrameExpectationsSuite().expect_value_equals(
        column_name="col1", value=5
    )
    data_frame = spark.createDataFrame([(5,), (5,), (5,)], ["col2"])
    with pytest.raises(DataFrameExpectationsSuiteFailure):
        expectations_suite.run(data_frame=data_frame)
