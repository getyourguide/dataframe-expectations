import pytest
import pandas as pd

from dataframe_expectations import DataFrameType
from dataframe_expectations.expectations_suite import (
    DataFrameExpectationsSuite,
    DataFrameExpectationsSuiteFailure,
)
from dataframe_expectations.result_message import (
    DataFrameExpectationFailureMessage,
)


def test_suite_success():
    """
    Test the ExpectationsSuite with a successful expectation.
    """

    # No expectations
    suite = DataFrameExpectationsSuite()
    result = suite.run(data_frame=pd.DataFrame())
    assert result is None, "Expected no result for empty suite"

    # All succeeding expectations
    suite = (
        DataFrameExpectationsSuite()
        .expect_value_greater_than(column_name="col1", value=2)
        .expect_value_less_than(column_name="col1", value=10)
    )
    data_Frame = pd.DataFrame({"col1": [3, 4, 5]})
    result = suite.run(data_frame=data_Frame)
    assert result is None, "Expected no result for successful suite"


def test_suite_failure():
    """
    Test the ExpectationsSuite with a failing expectation.
    """

    # Any 1 violation causes the suite to fail
    suite = (
        DataFrameExpectationsSuite()
        .expect_value_greater_than(column_name="col1", value=2)
        .expect_value_less_than(column_name="col1", value=3)
    )
    data_Frame = pd.DataFrame({"col1": [3, 4, 5]})

    with pytest.raises(DataFrameExpectationsSuiteFailure):
        suite.run(data_frame=data_Frame)


def test_invalid_data_frame_type():
    """
    Test that an invalid DataFrame type raises a ValueError.
    """

    suite = (
        DataFrameExpectationsSuite()
        .expect_value_greater_than(column_name="col1", value=2)
        .expect_value_less_than(column_name="col1", value=10)
    )
    data_Frame = None

    with pytest.raises(ValueError):
        suite.run(data_frame=data_Frame)


def test_suite_with_supported_dataframe_types(spark):
    """
    Test the ExpectationsSuite with all supported DataFrame types.
    """

    suite = DataFrameExpectationsSuite().expect_min_rows(min_rows=1)

    # Test with pandas DataFrame
    pandas_df = pd.DataFrame({"col1": [1, 2, 3]})
    result = suite.run(data_frame=pandas_df)
    assert result is None, "Expected success for pandas DataFrame"

    # Test with PySpark DataFrame
    spark_df = spark.createDataFrame([(1,), (2,), (3,)], ["col1"])
    result = suite.run(data_frame=spark_df)
    assert result is None, "Expected success for PySpark DataFrame"


def test_suite_with_unsupported_dataframe_types():
    """
    Test the ExpectationsSuite with unsupported DataFrame types.
    """
    suite = DataFrameExpectationsSuite().expect_min_rows(min_rows=1)

    # Test various unsupported types
    unsupported_types = [
        None,
        "not_a_dataframe",
        [1, 2, 3],
        {"col1": [1, 2, 3]},
        42,
        True,
    ]

    for unsupported_data in unsupported_types:
        with pytest.raises(ValueError) as context:
            suite.run(data_frame=unsupported_data)
        assert "Unsupported DataFrame type" in str(
            context.value
        ), f"Expected unsupported type error for {type(unsupported_data)}"


def test_suite_with_pyspark_connect_dataframe():
    """
    Test the ExpectationsSuite with PySpark Connect DataFrame (if available).
    """
    from unittest.mock import patch

    # Mock a Connect DataFrame
    class MockConnectDataFrame:
        def __init__(self):
            self.is_cached = False

        def cache(self):
            self.is_cached = True
            return self

        def unpersist(self):
            self.is_cached = False
            return self

    suite = DataFrameExpectationsSuite().expect_min_rows(min_rows=0)

    with patch(
        "dataframe_expectations.expectations.PySparkConnectDataFrame",
        MockConnectDataFrame,
    ):
        # Create mock expectation that can handle Connect DataFrame
        with patch.object(
            suite._DataFrameExpectationsSuite__expectations[0], "validate"
        ) as mock_validate:
            from dataframe_expectations.result_message import (
                DataFrameExpectationSuccessMessage,
            )

            mock_validate.return_value = DataFrameExpectationSuccessMessage(
                expectation_name="MockExpectation"
            )

            mock_connect_df = MockConnectDataFrame()
            result = suite.run(data_frame=mock_connect_df)
            assert result is None, "Expected success for mock Connect DataFrame"


def test_expectation_suite_failure_message():
    failed_expectation_messages = [
        DataFrameExpectationFailureMessage(
            expectation_str="ExpectationValueGreaterThan",
            data_frame_type=DataFrameType.PANDAS,
            message="Failed expectation 1",
        ),
        DataFrameExpectationFailureMessage(
            expectation_str="ExpectationValueGreaterThan",
            data_frame_type=DataFrameType.PANDAS,
            message="Failed expectation 2",
        ),
    ]

    suite_failure = DataFrameExpectationsSuiteFailure(
        total_expectations=4,
        failures=failed_expectation_messages,
    )

    expected_str = (
        "(2/4) expectations failed.\n\n"
        f"{'=' * 80}\n"
        "List of violations:\n"
        f"{'-' * 80}"
        "\n[Failed 1/2] ExpectationValueGreaterThan: Failed expectation 1\n"
        f"{'-' * 80}\n"
        "[Failed 2/2] ExpectationValueGreaterThan: Failed expectation 2\n"
        f"{'=' * 80}"
    )

    assert (
        str(suite_failure) == expected_str
    ), f"Expected suite failure message but got: {str(suite_failure)}"
