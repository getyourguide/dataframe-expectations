import pytest
import numpy as np
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


def test_expectation_name():
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationValueNull",
        column_name="col1",
    )
    assert expectation.get_expectation_name() == "ExpectationValueNull", (
        f"Expected 'ExpectationValueNull' but got: {expectation.get_expectation_name()}"
    )


def test_expectation_pandas_success():
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationValueNull",
        column_name="col1",
    )
    data_frame = pd.DataFrame({"col1": [None, np.nan, None]})
    result = expectation.validate(data_frame=data_frame)
    assert str(result) == str(
        DataFrameExpectationSuccessMessage(expectation_name="ExpectationValueNull")
    ), f"Expected success message but got: {result}"


def test_expectation_pandas_violations():
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationValueNull",
        column_name="col1",
    )
    data_frame = pd.DataFrame({"col1": [None, 1, 2]}, dtype="Int64")
    result = expectation.validate(data_frame=data_frame)

    expected_violations = pd.DataFrame({"col1": [1, 2]})
    assert str(result) == str(
        DataFrameExpectationFailureMessage(
            expectation_str=str(expectation),
            data_frame_type=DataFrameType.PANDAS,
            violations_data_frame=expected_violations,
            message="Found 2 row(s) where 'col1' is not null.",
            limit_violations=5,
        )
    ), f"Expected failure message but got: {result}"


def test_expectation_pyspark_success(spark):
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationValueNull",
        column_name="col1",
    )
    data_frame = spark.createDataFrame([(None,), (None,), (None,)], "col1: int")
    result = expectation.validate(data_frame=data_frame)
    assert str(result) == str(
        DataFrameExpectationSuccessMessage(expectation_name="ExpectationValueNull")
    ), f"Expected success message but got: {result}"


def test_expectation_pyspark_violations(spark):
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationValueNull",
        column_name="col1",
    )
    data_frame = spark.createDataFrame([(None,), (1,), (2,)], ["col1"])
    result = expectation.validate(data_frame=data_frame)

    expected_violations = spark.createDataFrame([(1,), (2,)], ["col1"])
    assert str(result) == str(
        DataFrameExpectationFailureMessage(
            expectation_str=str(expectation),
            data_frame_type=DataFrameType.PYSPARK,
            violations_data_frame=expected_violations,
            message="Found 2 row(s) where 'col1' is not null.",
            limit_violations=5,
        )
    ), f"Expected failure message but got: {result}"


def test_column_missing_error():
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationValueNull",
        column_name="col1",
    )
    data_frame = pd.DataFrame({"col2": [None, None, None]})
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
    expectations_suite = DataFrameExpectationsSuite().expect_value_null(column_name="col1")
    data_frame = pd.DataFrame({"col1": [None, None, None]})
    result = expectations_suite.build().run(data_frame=data_frame)
    assert result is None, "Expected no exceptions to be raised"


def test_suite_pandas_violations():
    expectations_suite = DataFrameExpectationsSuite().expect_value_null(column_name="col1")
    data_frame = pd.DataFrame({"col1": [None, 1, 2]})
    with pytest.raises(DataFrameExpectationsSuiteFailure):
        expectations_suite.build().run(data_frame=data_frame)


def test_suite_pyspark_success(spark):
    expectations_suite = DataFrameExpectationsSuite().expect_value_null(column_name="col1")
    data_frame = spark.createDataFrame([(None,), (None,), (None,)], "col1: int")
    result = expectations_suite.build().run(data_frame=data_frame)
    assert result is None, "Expected no exceptions to be raised"


def test_suite_pyspark_violations(spark):
    expectations_suite = DataFrameExpectationsSuite().expect_value_null(column_name="col1")
    data_frame = spark.createDataFrame([(None,), (1,), (2,)], ["col1"])
    with pytest.raises(DataFrameExpectationsSuiteFailure):
        expectations_suite.build().run(data_frame=data_frame)


def test_suite_pyspark_column_missing_error(spark):
    expectations_suite = DataFrameExpectationsSuite().expect_value_null(column_name="col1")
    data_frame = spark.createDataFrame([(None,), (None,), (None,)], "col2: int")
    with pytest.raises(DataFrameExpectationsSuiteFailure):
        expectations_suite.build().run(data_frame=data_frame)
