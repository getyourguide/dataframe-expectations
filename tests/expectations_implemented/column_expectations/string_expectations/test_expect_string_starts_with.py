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
        expectation_name="ExpectationStringStartsWith",
        column_name="col1",
        prefix="foo",
    )
    assert expectation.get_expectation_name() == "ExpectationStringStartsWith", (
        f"Expected 'ExpectationStringStartsWith' but got: {expectation.get_expectation_name()}"
    )


def test_expectation_pandas_success():
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationStringStartsWith",
        column_name="col1",
        prefix="foo",
    )
    data_frame = pd.DataFrame({"col1": ["foobar", "foo123", "foobaz"]})
    result = expectation.validate(data_frame=data_frame)
    assert str(result) == str(
        DataFrameExpectationSuccessMessage(expectation_name="ExpectationStringStartsWith")
    ), f"Expected success message but got: {result}"


def test_expectation_pandas_violations():
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationStringStartsWith",
        column_name="col1",
        prefix="foo",
    )
    data_frame = pd.DataFrame({"col1": ["foobar", "barfoo", "baz"]})
    result = expectation.validate(data_frame=data_frame)

    expected_violations = pd.DataFrame({"col1": ["barfoo", "baz"]})
    assert str(result) == str(
        DataFrameExpectationFailureMessage(
            expectation_str=str(expectation),
            data_frame_type=DataFrameType.PANDAS,
            violations_data_frame=expected_violations,
            message="Found 2 row(s) where 'col1' does not start with 'foo'.",
            limit_violations=5,
        )
    ), f"Expected failure message but got: {result}"


def test_expectation_pyspark_success(spark):
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationStringStartsWith",
        column_name="col1",
        prefix="foo",
    )
    data_frame = spark.createDataFrame([("foobar",), ("foo123",), ("foobaz",)], ["col1"])
    result = expectation.validate(data_frame=data_frame)
    assert str(result) == str(
        DataFrameExpectationSuccessMessage(expectation_name="ExpectationStringStartsWith")
    ), f"Expected success message but got: {result}"


def test_expectation_pyspark_violations(spark):
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationStringStartsWith",
        column_name="col1",
        prefix="foo",
    )
    data_frame = spark.createDataFrame([("foobar",), ("barfoo",), ("baz",)], ["col1"])
    result = expectation.validate(data_frame=data_frame)

    expected_violations = spark.createDataFrame([("barfoo",), ("baz",)], ["col1"])
    assert str(result) == str(
        DataFrameExpectationFailureMessage(
            expectation_str=str(expectation),
            data_frame_type=DataFrameType.PYSPARK,
            violations_data_frame=expected_violations,
            message="Found 2 row(s) where 'col1' does not start with 'foo'.",
            limit_violations=5,
        )
    ), f"Expected failure message but got: {result}"


def test_column_missing_error():
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name="ExpectationStringStartsWith",
        column_name="col1",
        prefix="foo",
    )
    data_frame = pd.DataFrame({"col2": ["foobar", "foo123", "foobaz"]})
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
    expectations_suite = DataFrameExpectationsSuite().expect_string_starts_with(
        column_name="col1", prefix="foo"
    )
    data_frame = pd.DataFrame({"col1": ["foobar", "foo123", "foobaz"]})
    result = expectations_suite.run(data_frame=data_frame)
    assert result is None, "Expected no exceptions to be raised"


def test_suite_pandas_violations():
    expectations_suite = DataFrameExpectationsSuite().expect_string_starts_with(
        column_name="col1", prefix="foo"
    )
    data_frame = pd.DataFrame({"col1": ["foobar", "barfoo", "baz"]})
    with pytest.raises(DataFrameExpectationsSuiteFailure):
        expectations_suite.run(data_frame=data_frame)


def test_suite_pyspark_success(spark):
    expectations_suite = DataFrameExpectationsSuite().expect_string_starts_with(
        column_name="col1", prefix="foo"
    )
    data_frame = spark.createDataFrame([("foobar",), ("foo123",), ("foobaz",)], ["col1"])
    result = expectations_suite.run(data_frame=data_frame)
    assert result is None, "Expected no exceptions to be raised"


def test_suite_pyspark_violations(spark):
    expectations_suite = DataFrameExpectationsSuite().expect_string_starts_with(
        column_name="col1", prefix="foo"
    )
    data_frame = spark.createDataFrame([("foobar",), ("barfoo",), ("baz",)], ["col1"])
    with pytest.raises(DataFrameExpectationsSuiteFailure):
        expectations_suite.run(data_frame=data_frame)


def test_suite_pyspark_column_missing_error(spark):
    expectations_suite = DataFrameExpectationsSuite().expect_string_starts_with(
        column_name="col1", prefix="foo"
    )
    data_frame = spark.createDataFrame([("foobar",), ("foo123",), ("foobaz",)], ["col2"])
    with pytest.raises(DataFrameExpectationsSuiteFailure):
        expectations_suite.run(data_frame=data_frame)
