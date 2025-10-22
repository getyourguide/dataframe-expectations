import pytest
import pandas as pd
from tabulate import tabulate  # type: ignore

from dataframe_expectations import DataFrameType
from dataframe_expectations.result_message import (
    DataFrameExpectationFailureMessage,
    DataFrameExpectationResultMessage,
    DataFrameExpectationSuccessMessage,
)
from tests.conftest import assert_pandas_df_equal


def test_result_message_empty():
    """
    By default the result message should be empty.
    """
    result_message = DataFrameExpectationResultMessage()

    assert str(result_message) == "", (
        f"Expected empty result message but got: {str(result_message)}"
    )


def test_data_frame_to_str_pandas():
    """
    Test the dataframe_to_str method with a mock DataFrame.
    """
    pandas_dataframe = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
    result_message = DataFrameExpectationResultMessage()

    expected_dataframe_str = tabulate(
        pandas_dataframe.head(2), headers="keys", tablefmt="pretty", showindex=False
    )

    actual_str = result_message.dataframe_to_str(
        data_frame_type=DataFrameType.PANDAS,
        data_frame=pandas_dataframe,
        rows=2,
    )
    assert actual_str == expected_dataframe_str, (
        f"Expected pandas dataframe string but got: {actual_str}"
    )


def test_dataframe_to_str_pyspark(spark):
    """
    Test the dataframe_to_str method with a mock PySpark DataFrame.
    """
    pyspark_dataframe = spark.createDataFrame([(1, "a"), (2, "b"), (3, "c")], ["col1", "col2"])

    result_message = DataFrameExpectationResultMessage()

    expected_dataframe_str = tabulate(
        pyspark_dataframe.limit(2).toPandas(),
        headers="keys",
        tablefmt="pretty",
        showindex=False,
    )

    actual_str = result_message.dataframe_to_str(
        data_frame_type=DataFrameType.PYSPARK,
        data_frame=pyspark_dataframe,
        rows=2,
    )
    assert actual_str == expected_dataframe_str, (
        f"Expected pyspark dataframe string but got: {actual_str}"
    )


def test_dataframe_to_str_invalid_type():
    """
    Test the dataframe_to_str method with an invalid DataFrame type.
    """
    result_message = DataFrameExpectationResultMessage()

    with pytest.raises(ValueError) as context:
        result_message.dataframe_to_str(
            data_frame_type="invalid_type", data_frame=pd.DataFrame(), rows=2
        )

    assert str(context.value) == "Unsupported DataFrame type: invalid_type", (
        f"Expected ValueError message but got: {str(context.value)}"
    )


def test_success_message_no_additional_message():
    """
    Test the success message initialization and string representation. Test with no additional message
    """
    expectation_name = "TestExpectation"
    success_message = DataFrameExpectationSuccessMessage(expectation_name)
    message_str = str(success_message)
    assert expectation_name in message_str, (
        f"Expectation name should be in the message: {message_str}"
    )


def test_success_message_with_additional_message():
    """
    Test the success message initialization and string representation. Test with an additional message
    """
    expectation_name = "TestExpectation"
    additional_message = "This is a success message."
    success_message_with_additional = DataFrameExpectationSuccessMessage(
        expectation_name, additional_message
    )
    message_str = str(success_message_with_additional)
    assert expectation_name in message_str, (
        f"Expectation name should be in the message: {message_str}"
    )
    assert additional_message in message_str, (
        f"Additional message should be in the success message: {message_str}"
    )


def test_failure_message_default_params():
    """
    Test the failure message initialization and string representation with default parameters.
    """
    expectation_name = "TestExpectation"
    data_frame_type = None
    failure_message = DataFrameExpectationFailureMessage(expectation_name, data_frame_type)

    message_str = str(failure_message)
    assert expectation_name in message_str, (
        f"Expectation name should be in the message: {message_str}"
    )

    violations_df = failure_message.get_violations_data_frame()
    assert violations_df is None, (
        f"Violations DataFrame should be None when not provided but got: {violations_df}"
    )


def test_failure_message_custom_message():
    """
    Test the failure message initialization and string representation with a custom message.
    """
    expectation_name = "TestExpectation"
    data_frame_type = None
    custom_message = "This is a custom failure message."
    failure_message = DataFrameExpectationFailureMessage(
        expectation_str=expectation_name,
        data_frame_type=data_frame_type,
        message=custom_message,
    )

    message_str = str(failure_message)
    assert expectation_name in message_str, (
        f"Expectation name should be in the message: {message_str}"
    )
    assert custom_message in message_str, (
        f"Custom message should be in the failure message: {message_str}"
    )

    violations_df = failure_message.get_violations_data_frame()
    assert violations_df is None, (
        f"Violations DataFrame should be None when not provided but got: {violations_df}"
    )


def test_failure_message_with_violations_dataframe():
    """
    Test the failure message initialization and string representation with a violations DataFrame.
    """
    expectation_name = "TestExpectation"
    data_frame_type = DataFrameType.PANDAS
    violations_dataframe = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
    failure_message = DataFrameExpectationFailureMessage(
        expectation_str=expectation_name,
        data_frame_type=data_frame_type,
        violations_data_frame=violations_dataframe,
        limit_violations=2,
    )

    expected_dataframe = violations_dataframe
    expected_dataframe_str = tabulate(
        expected_dataframe.head(2),
        headers="keys",
        tablefmt="pretty",
        showindex=False,
    )

    message_str = str(failure_message)
    assert expectation_name in message_str, (
        f"Expectation name should be in the message: {message_str}"
    )
    assert expected_dataframe_str in message_str, (
        f"Violations DataFrame should be included in the message: {message_str}"
    )

    actual_violations_df = failure_message.get_violations_data_frame()
    assert_pandas_df_equal(actual_violations_df, expected_dataframe)
