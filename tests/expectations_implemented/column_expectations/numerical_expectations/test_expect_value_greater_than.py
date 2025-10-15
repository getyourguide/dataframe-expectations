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
    """
    Test that the expectation name is correctly returned.
    """
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationValueGreaterThan",
        column_name="col1",
        value=2,
    )
    assert (
        expectation.get_expectation_name() == "ExpectationValueGreaterThan"
    ), f"Expected 'ExpectationValueGreaterThan' but got: {expectation.get_expectation_name()}"


def test_expectation_pandas_success():
    """
    Test the greater than expectation for pandas dataframe.
    """
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationValueGreaterThan",
        column_name="col1",
        value=2,
    )
    data_frame = pd.DataFrame({"col1": [3, 4, 5]})
    result = expectation.validate(data_frame=data_frame)
    assert str(result) == str(
        DataframeExpectationSuccessMessage(expectation_name="ExpectationValueGreaterThan")
    ), f"Expected success message but got: {result}"


def test_expectation_pandas_violations():
    """
    Test the greater than expectation for pandas dataframe with violations.
    """
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationValueGreaterThan",
        column_name="col1",
        value=3,
    )
    data_frame = pd.DataFrame({"col1": [3, 4, 5]})
    result = expectation.validate(data_frame=data_frame)

    expected_violations = pd.DataFrame({"col1": [3]})
    expected_failure_message = DataframeExpectationFailureMessage(
        expectation_str=str(expectation),
        data_frame_type=DataFrameType.PANDAS,
        violations_data_frame=expected_violations,
        message="Found 1 row(s) where 'col1' is not greater than 3.",
        limit_violations=5,
    )

    assert str(result) == str(
        expected_failure_message
    ), f"Expected failure message but got: {result}"


def test_expectation_pyspark_success(spark):
    """
    Test the greater than expectation for pyspark dataframe.
    """
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationValueGreaterThan",
        column_name="col1",
        value=2,
    )
    data_frame = spark.createDataFrame([(3,), (4,), (5,)], ["col1"])
    result = expectation.validate(data_frame=data_frame)
    assert str(result) == str(
        DataframeExpectationSuccessMessage(expectation_name="ExpectationValueGreaterThan")
    ), f"Expected success message but got: {result}"


def test_expectation_pyspark_violations(spark):
    """
    Test the greater than expectation for pyspark dataframe with violations.
    """
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationValueGreaterThan",
        column_name="col1",
        value=3,
    )
    data_frame = spark.createDataFrame([(3,), (4,), (5,)], ["col1"])
    result = expectation.validate(data_frame=data_frame)

    expected_violations = spark.createDataFrame([(3,)], ["col1"])
    expected_failure_message = DataframeExpectationFailureMessage(
        expectation_str=str(expectation),
        data_frame_type=DataFrameType.PYSPARK,
        violations_data_frame=expected_violations,
        message="Found 1 row(s) where 'col1' is not greater than 3.",
        limit_violations=5,
    )

    assert str(result) == str(
        expected_failure_message
    ), f"Expected failure message but got: {result}"


def test_column_missing_error():
    """
    Test the error when the specified column is missing in the dataframe.
    """
    expectation = DataframeExpectationRegistry.get_expectation(
        expectation_name="ExpectationValueGreaterThan",
        column_name="col1",
        value=2,
    )
    data_frame = pd.DataFrame({"col2": [3, 4, 5]})

    result = expectation.validate(data_frame=data_frame)
    expected_failure_message = DataframeExpectationFailureMessage(
        expectation_str=str(expectation),
        data_frame_type=DataFrameType.PANDAS,
        message="Column 'col1' does not exist in the DataFrame.",
    )

    assert str(result) == str(
        expected_failure_message
    ), f"Expected failure message but got: {result}"


def test_suite_pandas_success():
    """
    Test the greater than expectation for pandas dataframe with no violations.
    """
    expectations_suite = DataframeExpectationsSuite().expect_value_greater_than(
        column_name="col1", value=2
    )
    data_frame = pd.DataFrame({"col1": [3, 4, 5]})
    result = expectations_suite.run(data_frame=data_frame)
    assert result is None, "Expected no exceptions to be raised"


def test_suite_pandas_violations():
    """
    Test the greater than expectation for pandas dataframe with violations.
    """
    expectations_suite = DataframeExpectationsSuite().expect_value_greater_than(
        column_name="col1", value=3
    )
    data_frame = pd.DataFrame({"col1": [3, 4, 5]})
    with pytest.raises(DataframeExpectationsSuiteFailure):
        expectations_suite.run(data_frame=data_frame)


def test_suite_pyspark_success(spark):
    """
    Test the greater than expectation for pyspark dataframe with no violations.
    """
    expectations_suite = DataframeExpectationsSuite().expect_value_greater_than(
        column_name="col1", value=2
    )
    data_frame = spark.createDataFrame([(3,), (4,), (5,)], ["col1"])
    result = expectations_suite.run(data_frame=data_frame)
    assert result is None, "Expected no exceptions to be raised"


def test_suite_pyspark_violations(spark):
    """
    Test the greater than expectation for pyspark dataframe with violations.
    """
    expectations_suite = DataframeExpectationsSuite().expect_value_greater_than(
        column_name="col1", value=3
    )
    data_frame = spark.createDataFrame([(3,), (4,), (5,)], ["col1"])
    with pytest.raises(DataframeExpectationsSuiteFailure):
        expectations_suite.run(data_frame=data_frame)


def test_suite_column_missing_error():
    """
    Test the greater than expectation for dataframe with missing column.
    """
    expectations_suite = DataframeExpectationsSuite().expect_value_greater_than(
        column_name="col1", value=2
    )
    data_frame = pd.DataFrame({"col2": [3, 4, 5]})
    with pytest.raises(DataframeExpectationsSuiteFailure):
        expectations_suite.run(data_frame=data_frame)
