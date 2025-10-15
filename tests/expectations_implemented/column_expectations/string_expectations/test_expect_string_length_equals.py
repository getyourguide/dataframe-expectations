import pytest
import pandas as pd

from dataframe_expectations import DataFrameType
from dataframe_expectations.expectations.expectation_registry import (
    DataframeExpectationRegistry,
)
from dataframe_expectations.expectations_suite import (
    DataframeExpectationsSuite,
    DataframeExpectationsSuiteFailure,
)
from dataframe_expectations.result_message import (
    DataframeExpectationFailureMessage,
    DataframeExpectationSuccessMessage,
)



def test_expectation_name():
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationStringLengthEquals",
        column_name="col1",
        length=3,
    )
    assert expectation.get_expectation_name() == "ExpectationStringLengthEquals", f"Expected 'ExpectationStringLengthEquals' but got: {expectation.get_expectation_name()}"

def test_expectation_pandas_success():
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationStringLengthEquals",
        column_name="col1",
        length=3,
    )
    data_frame = pd.DataFrame({"col1": ["foo", "bar", "baz"]})
    result = expectation.validate(data_frame=data_frame)
    assert str(result) == str(
        DataframeExpectationSuccessMessage(
            expectation_name="ExpectationStringLengthEquals"
        )
    ), f"Expected success message but got: {result}"

def test_expectation_pandas_violations():
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationStringLengthEquals",
        column_name="col1",
        length=3,
    )
    data_frame = pd.DataFrame({"col1": ["foo", "bar", "bazz", "foobar"]})
    result = expectation.validate(data_frame=data_frame)

    expected_violations = pd.DataFrame({"col1": ["bazz", "foobar"]})
    assert str(result) == str(
        DataframeExpectationFailureMessage(
            expectation_str=str(expectation),
            data_frame_type=DataFrameType.PANDAS,
            violations_data_frame=expected_violations,
            message="Found 2 row(s) where 'col1' length is not equal to 3.",
            limit_violations=5,
        )
    ), f"Expected failure message but got: {result}"

def test_expectation_pyspark_success(spark):
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationStringLengthEquals",
        column_name="col1",
        length=3,
    )
    data_frame = spark.createDataFrame([("foo",), ("bar",), ("baz",)], ["col1"])
    result = expectation.validate(data_frame=data_frame)
    assert str(result) == str(
        DataframeExpectationSuccessMessage(
            expectation_name="ExpectationStringLengthEquals"
        )
    ), f"Expected success message but got: {result}"

def test_expectation_pyspark_violations(spark):
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationStringLengthEquals",
        column_name="col1",
        length=3,
    )
    data_frame = spark.createDataFrame(
        [("foo",), ("bar",), ("bazz",), ("foobar",)], ["col1"]
    )
    result = expectation.validate(data_frame=data_frame)

    expected_violations = spark.createDataFrame([("bazz",), ("foobar",)], ["col1"])
    assert str(result) == str(
        DataframeExpectationFailureMessage(
            expectation_str=str(expectation),
            data_frame_type=DataFrameType.PYSPARK,
            violations_data_frame=expected_violations,
            message="Found 2 row(s) where 'col1' length is not equal to 3.",
            limit_violations=5,
        )
    ), f"Expected failure message but got: {result}"

def test_column_missing_error():
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationStringLengthEquals",
        column_name="col1",
        length=3,
    )
    data_frame = pd.DataFrame({"col2": ["foo", "bar", "baz"]})
    result = expectation.validate(data_frame=data_frame)
    expected_failure_message = DataframeExpectationFailureMessage(
        expectation_str=str(expectation),
        data_frame_type=DataFrameType.PANDAS,
        message="Column 'col1' does not exist in the DataFrame.",
    )
    assert str(result) == str(expected_failure_message), f"Expected failure message but got: {result}"

def test_suite_pandas_success():
    expectations_suite = DataframeExpectationsSuite().expect_string_length_equals(
        column_name="col1", length=3
    )
    data_frame = pd.DataFrame({"col1": ["foo", "bar", "baz"]})
    result = expectations_suite.run(data_frame=data_frame)
    assert result is None, "Expected no exceptions to be raised"

def test_suite_pandas_violations():
    expectations_suite = DataframeExpectationsSuite().expect_string_length_equals(
        column_name="col1", length=3
    )
    data_frame = pd.DataFrame({"col1": ["foo", "bar", "bazz", "foobar"]})
    with pytest.raises(DataframeExpectationsSuiteFailure):
        expectations_suite.run(data_frame=data_frame)

def test_suite_pyspark_success(spark):
    expectations_suite = DataframeExpectationsSuite().expect_string_length_equals(
        column_name="col1", length=3
    )
    data_frame = spark.createDataFrame([("foo",), ("bar",), ("baz",)], ["col1"])
    result = expectations_suite.run(data_frame=data_frame)
    assert result is None, "Expected no exceptions to be raised"

def test_suite_pyspark_violations(spark):
    expectations_suite = DataframeExpectationsSuite().expect_string_length_equals(
        column_name="col1", length=3
    )
    data_frame = spark.createDataFrame(
        [("foo",), ("bar",), ("bazz",), ("foobar",)], ["col1"]
    )
    with pytest.raises(DataframeExpectationsSuiteFailure):
        expectations_suite.run(data_frame=data_frame)

def test_suite_pyspark_column_missing_error(spark):
    expectations_suite = DataframeExpectationsSuite().expect_string_length_equals(
        column_name="col1", length=3
    )
    data_frame = spark.createDataFrame([("foo",), ("bar",), ("baz",)], ["col2"])
    with pytest.raises(DataframeExpectationsSuiteFailure):
        expectations_suite.run(data_frame=data_frame)
