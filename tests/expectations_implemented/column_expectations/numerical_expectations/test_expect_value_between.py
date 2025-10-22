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
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationValueBetween",
        column_name="col1",
        min_value=2,
        max_value=5,
    )
    assert expectation.get_expectation_name() == "ExpectationValueBetween", (
        f"Expected 'ExpectationValueBetween' but got: {expectation.get_expectation_name()}"
    )


def test_expectation_pandas_success():
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationValueBetween",
        column_name="col1",
        min_value=2,
        max_value=5,
    )
    data_frame = pd.DataFrame({"col1": [2, 3, 4, 5]})
    result = expectation.validate(data_frame=data_frame)
    assert str(result) == str(
        DataFrameExpectationSuccessMessage(expectation_name="ExpectationValueBetween")
    ), f"Expected success message but got: {result}"


def test_expectation_pandas_violations():
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationValueBetween",
        column_name="col1",
        min_value=2,
        max_value=5,
    )
    data_frame = pd.DataFrame({"col1": [1, 2, 3, 6]})
    expected_violations = pd.DataFrame({"col1": [1, 6]})
    result = expectation.validate(data_frame=data_frame)
    expected_failure_message = DataFrameExpectationFailureMessage(
        expectation_str=str(expectation),
        data_frame_type=DataFrameType.PANDAS,
        violations_data_frame=expected_violations,
        message="Found 2 row(s) where 'col1' is not between 2 and 5.",
        limit_violations=5,
    )
    assert str(result) == str(expected_failure_message), (
        f"Expected failure message but got: {result}"
    )


def test_expectation_pyspark_success(spark):
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationValueBetween",
        column_name="col1",
        min_value=2,
        max_value=5,
    )
    data_frame = spark.createDataFrame([(2,), (3,), (4,), (5,)], ["col1"])
    result = expectation.validate(data_frame=data_frame)
    assert str(result) == str(
        DataFrameExpectationSuccessMessage(expectation_name="ExpectationValueBetween")
    ), f"Expected success message but got: {result}"


def test_expectation_pyspark_violations(spark):
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationValueBetween",
        column_name="col1",
        min_value=2,
        max_value=5,
    )
    data_frame = spark.createDataFrame([(1,), (2,), (3,), (6,)], ["col1"])
    expected_violations = spark.createDataFrame([(1,), (6,)], ["col1"])
    result = expectation.validate(data_frame=data_frame)
    expected_failure_message = DataFrameExpectationFailureMessage(
        expectation_str=str(expectation),
        data_frame_type=DataFrameType.PYSPARK,
        violations_data_frame=expected_violations,
        message="Found 2 row(s) where 'col1' is not between 2 and 5.",
        limit_violations=5,
    )
    assert str(result) == str(expected_failure_message), (
        f"Expected failure message but got: {result}"
    )


def test_column_missing_error():
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationValueBetween",
        column_name="col1",
        min_value=2,
        max_value=5,
    )
    data_frame = pd.DataFrame({"col2": [2, 3, 4]})
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
    expectations_suite = DataFrameExpectationsSuite().expect_value_between(
        column_name="col1", min_value=2, max_value=5
    )
    data_frame = pd.DataFrame({"col1": [2, 3, 4, 5]})
    result = expectations_suite.run(data_frame=data_frame)
    assert result is None, "Expected no exceptions to be raised"


def test_suite_pandas_violations():
    expectations_suite = DataFrameExpectationsSuite().expect_value_between(
        column_name="col1", min_value=2, max_value=5
    )
    data_frame = pd.DataFrame({"col1": [1, 2, 3, 6]})
    with pytest.raises(DataFrameExpectationsSuiteFailure):
        expectations_suite.run(data_frame=data_frame)


def test_suite_pyspark_success(spark):
    expectations_suite = DataFrameExpectationsSuite().expect_value_between(
        column_name="col1", min_value=2, max_value=5
    )
    data_frame = spark.createDataFrame([(2,), (3,), (4,), (5,)], ["col1"])
    result = expectations_suite.run(data_frame=data_frame)
    assert result is None, "Expected no exceptions to be raised"


def test_suite_pyspark_violations(spark):
    expectations_suite = DataFrameExpectationsSuite().expect_value_between(
        column_name="col1", min_value=2, max_value=5
    )
    data_frame = spark.createDataFrame([(1,), (2,), (3,), (6,)], ["col1"])
    with pytest.raises(DataFrameExpectationsSuiteFailure):
        expectations_suite.run(data_frame=data_frame)


def test_suite_column_missing_error():
    expectations_suite = DataFrameExpectationsSuite().expect_value_between(
        column_name="col1", min_value=2, max_value=5
    )
    data_frame = pd.DataFrame({"col2": [2, 3, 4]})
    with pytest.raises(DataFrameExpectationsSuiteFailure):
        expectations_suite.run(data_frame=data_frame)
