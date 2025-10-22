import pytest
import numpy as np
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
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationValueNotNull",
        column_name="col1",
    )
    assert expectation.get_expectation_name() == "ExpectationValueNotNull", (
        f"Expected 'ExpectationValueNotNull' but got: {expectation.get_expectation_name()}"
    )


def test_expectation_pandas_success():
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationValueNotNull",
        column_name="col1",
    )
    data_frame = pd.DataFrame({"col1": [1, 2, 3]})
    result = expectation.validate(data_frame=data_frame)
    assert str(result) == str(
        DataFrameExpectationSuccessMessage(expectation_name="ExpectationValueNotNull")
    ), f"Expected success message but got: {result}"


def test_expectation_pandas_violations():
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationValueNotNull",
        column_name="col1",
    )
    data_frame = pd.DataFrame({"col1": [1, None, np.nan]})
    result = expectation.validate(data_frame=data_frame)

    expected_violations = pd.DataFrame({"col1": [None, np.nan]})
    assert str(result) == str(
        DataFrameExpectationFailureMessage(
            expectation_str=str(expectation),
            data_frame_type=DataFrameType.PANDAS,
            violations_data_frame=expected_violations,
            message="Found 2 row(s) where 'col1' is null.",
            limit_violations=5,
        )
    ), f"Expected failure message but got: {result}"


def test_expectation_pyspark_success(spark):
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationValueNotNull",
        column_name="col1",
    )
    data_frame = spark.createDataFrame([(1,), (2,), (3,)], ["col1"])
    result = expectation.validate(data_frame=data_frame)
    assert str(result) == str(
        DataFrameExpectationSuccessMessage(expectation_name="ExpectationValueNotNull")
    ), f"Expected success message but got: {result}"


def test_expectation_pyspark_violations(spark):
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationValueNotNull",
        column_name="col1",
    )
    data_frame = spark.createDataFrame([(1,), (None,), (None,)], ["col1"])
    result = expectation.validate(data_frame=data_frame)

    expected_violations = spark.createDataFrame([(None,), (None,)], "col1: int")
    assert str(result) == str(
        DataFrameExpectationFailureMessage(
            expectation_str=str(expectation),
            data_frame_type=DataFrameType.PYSPARK,
            violations_data_frame=expected_violations,
            message="Found 2 row(s) where 'col1' is null.",
            limit_violations=5,
        )
    ), f"Expected failure message but got: {result}"


def test_column_missing_error():
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationValueNotNull",
        column_name="col1",
    )
    data_frame = pd.DataFrame({"col2": [1, 2, 3]})
    result = expectation.validate(data_frame=data_frame)
    expected_failure_message = DataFrameExpectationFailureMessage(
        expectation_str=str(expectation),
        data_frame_type=DataFrameType.PANDAS,
        message="Column 'col1' does not exist in the DataFrame.",
    )
    assert str(result) == str(expected_failure_message), (
        f"Expected failure message but got: {result}"
    )


def test_suite_pandas_success():
    expectations_suite = DataFrameExpectationsSuite().expect_value_not_null(column_name="col1")
    data_frame = pd.DataFrame({"col1": [1, 2, 3]})
    result = expectations_suite.run(data_frame=data_frame)
    assert result is None, "Expected no exceptions to be raised"


def test_suite_pandas_violations():
    expectations_suite = DataFrameExpectationsSuite().expect_value_not_null(column_name="col1")
    data_frame = pd.DataFrame({"col1": [1, None, np.nan]})
    with pytest.raises(DataFrameExpectationsSuiteFailure):
        expectations_suite.run(data_frame=data_frame)


def test_suite_pyspark_success(spark):
    expectations_suite = DataFrameExpectationsSuite().expect_value_not_null(column_name="col1")
    data_frame = spark.createDataFrame([(1,), (2,), (3,)], ["col1"])
    result = expectations_suite.run(data_frame=data_frame)
    assert result is None, "Expected no exceptions to be raised"


def test_suite_pyspark_violations(spark):
    expectations_suite = DataFrameExpectationsSuite().expect_value_not_null(column_name="col1")
    data_frame = spark.createDataFrame([(1,), (None,), (None,)], ["col1"])
    with pytest.raises(DataFrameExpectationsSuiteFailure):
        expectations_suite.run(data_frame=data_frame)


def test_suite_pyspark_column_missing_error(spark):
    expectations_suite = DataFrameExpectationsSuite().expect_value_not_null(column_name="col1")
    data_frame = spark.createDataFrame([(1,), (2,), (3,)], ["col2"])
    with pytest.raises(DataFrameExpectationsSuiteFailure):
        expectations_suite.run(data_frame=data_frame)
