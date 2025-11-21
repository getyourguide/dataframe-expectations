import pytest
import pandas as pd

from dataframe_expectations.core.types import DataFrameType
from dataframe_expectations.suite import (
    DataFrameExpectationsSuite,
    DataFrameExpectationsSuiteFailure,
)
from dataframe_expectations.result_message import (
    DataFrameExpectationFailureMessage,
)
from dataframe_expectations.core.suite_result import SuiteExecutionResult


def test_suite_success():
    """
    Test the ExpectationsSuite with a successful expectation.
    """

    # All succeeding expectations
    suite = (
        DataFrameExpectationsSuite()
        .expect_value_greater_than(column_name="col1", value=2)
        .expect_value_less_than(column_name="col1", value=10)
    )
    data_Frame = pd.DataFrame({"col1": [3, 4, 5]})
    runner = suite.build()
    result = runner.run(data_frame=data_Frame)
    assert result is not None, "Expected SuiteExecutionResult"
    assert isinstance(result, SuiteExecutionResult), "Result should be SuiteExecutionResult"
    assert result.success, "Expected all expectations to pass"
    assert result.total_passed == 2, "Expected 2 passed expectations"
    assert result.total_failed == 0, "Expected 0 failed expectations"


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
    runner = suite.build()

    with pytest.raises(DataFrameExpectationsSuiteFailure):
        runner.run(data_frame=data_Frame)


def test_invalid_data_frame_type():
    """
    Test that an invalid DataFrame type raises a ValueError.
    """

    suite = (
        DataFrameExpectationsSuite()
        .expect_value_greater_than(column_name="col1", value=2)
        .expect_value_less_than(column_name="col1", value=10)
    )
    runner = suite.build()
    data_Frame = None

    with pytest.raises(ValueError):
        runner.run(data_frame=data_Frame)


def test_suite_with_supported_dataframe_types(spark):
    """
    Test the ExpectationsSuite with all supported DataFrame types.
    """

    suite = DataFrameExpectationsSuite().expect_min_rows(min_rows=1)
    runner = suite.build()

    # Test with pandas DataFrame
    pandas_df = pd.DataFrame({"col1": [1, 2, 3]})
    result = runner.run(data_frame=pandas_df)
    assert result is not None, "Expected SuiteExecutionResult for pandas DataFrame"
    assert isinstance(result, SuiteExecutionResult), "Result should be SuiteExecutionResult"
    assert result.success, "Expected success for pandas DataFrame"
    assert result.dataframe_type == DataFrameType.PANDAS

    # Test with PySpark DataFrame
    spark_df = spark.createDataFrame([(1,), (2,), (3,)], ["col1"])
    result = runner.run(data_frame=spark_df)
    assert result is not None, "Expected SuiteExecutionResult for PySpark DataFrame"
    assert isinstance(result, SuiteExecutionResult), "Result should be SuiteExecutionResult"
    assert result.success, "Expected success for PySpark DataFrame"
    assert result.dataframe_type == DataFrameType.PYSPARK


def test_suite_with_unsupported_dataframe_types():
    """
    Test the ExpectationsSuite with unsupported DataFrame types.
    """
    suite = DataFrameExpectationsSuite().expect_min_rows(min_rows=1)
    runner = suite.build()

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
            runner.run(data_frame=unsupported_data)
        assert "Unsupported DataFrame type" in str(context.value), (
            f"Expected unsupported type error for {type(unsupported_data)}"
        )


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

        def count(self):
            return 0

    suite = DataFrameExpectationsSuite().expect_min_rows(min_rows=0)
    runner = suite.build()

    with patch(
        "dataframe_expectations.core.expectation.PySparkConnectDataFrame",
        MockConnectDataFrame,
    ):
        # Create mock expectation that can handle Connect DataFrame
        with patch.object(
            runner._DataFrameExpectationsSuiteRunner__expectations[0], "validate"
        ) as mock_validate:
            from dataframe_expectations.result_message import (
                DataFrameExpectationSuccessMessage,
            )

            mock_validate.return_value = DataFrameExpectationSuccessMessage(
                expectation_name="MockExpectation"
            )

            mock_connect_df = MockConnectDataFrame()
            result = runner.run(data_frame=mock_connect_df)
            assert result is not None, "Expected SuiteExecutionResult for mock Connect DataFrame"
            assert isinstance(result, SuiteExecutionResult), "Result should be SuiteExecutionResult"
            assert result.success, "Expected success for mock Connect DataFrame"


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

    assert str(suite_failure) == expected_str, (
        f"Expected suite failure message but got: {str(suite_failure)}"
    )


def test_build_empty_suite():
    """
    Test that building a suite with no expectations raises an error.
    """
    suite = DataFrameExpectationsSuite()

    with pytest.raises(ValueError) as context:
        suite.build()

    assert "no expectations added" in str(context.value)


def test_builder_pattern_immutability():
    """
    Test that runners are immutable and independent.
    """
    suite = DataFrameExpectationsSuite()
    suite.expect_value_greater_than(column_name="col1", value=5)

    # Build first runner
    runner1 = suite.build()

    # Verify runner1 has exactly 1 expectation
    assert runner1.expectation_count == 1, "Runner1 should have 1 expectation"
    expectations_list = runner1.list_expectations()
    assert len(expectations_list) == 1
    assert expectations_list[0] == "ExpectationValueGreaterThan ('col1' is greater than 5)"

    # Add more expectations to suite
    suite.expect_value_less_than(column_name="col1", value=20)

    # Build second runner
    runner2 = suite.build()

    # Verify runner2 has 2 expectations but runner1 is unchanged
    assert runner1.expectation_count == 1, "Runner1 should still have 1 expectation (immutable)"
    assert runner2.expectation_count == 2, "Runner2 should have 2 expectations"
    expectations_list2 = runner2.list_expectations()
    assert len(expectations_list2) == 2
    assert expectations_list2[0] == "ExpectationValueGreaterThan ('col1' is greater than 5)"
    assert expectations_list2[1] == "ExpectationValueLessThan ('col1' is less than 20)"

    # Test data
    df = pd.DataFrame({"col1": [10, 15]})

    # Runner1 should only have 1 expectation (passes)
    result1 = runner1.run(data_frame=df)
    assert result1 is not None, "Runner1 should return SuiteExecutionResult"
    assert isinstance(result1, SuiteExecutionResult), "Result1 should be SuiteExecutionResult"
    assert result1.success, "Runner1 should pass with only 1 expectation"
    assert result1.total_passed == 1

    # Runner2 should have 2 expectations (passes)
    result2 = runner2.run(data_frame=df)
    assert result2 is not None, "Runner2 should return SuiteExecutionResult"
    assert isinstance(result2, SuiteExecutionResult), "Result2 should be SuiteExecutionResult"
    assert result2.success, "Runner2 should pass with 2 expectations"
    assert result2.total_passed == 2


def test_decorator_success():
    """
    Test the validate decorator with successful validation.
    """
    suite = DataFrameExpectationsSuite()
    suite.expect_min_rows(min_rows=1)
    suite.expect_value_greater_than(column_name="col1", value=0)

    runner = suite.build()

    @runner.validate
    def load_data():
        return pd.DataFrame({"col1": [1, 2, 3]})

    # Should not raise exception
    df = load_data()
    assert len(df) == 3, "Should return the original DataFrame"
    assert list(df["col1"]) == [1, 2, 3]


def test_decorator_failure():
    """
    Test the validate decorator with failing validation.
    """
    suite = DataFrameExpectationsSuite()
    suite.expect_value_greater_than(column_name="col1", value=10)

    runner = suite.build()

    @runner.validate
    def load_bad_data():
        return pd.DataFrame({"col1": [1, 2, 3]})

    # Should raise DataFrameExpectationsSuiteFailure
    with pytest.raises(DataFrameExpectationsSuiteFailure):
        load_bad_data()


def test_decorator_with_arguments():
    """
    Test the validate decorator with functions that take arguments.
    """
    suite = DataFrameExpectationsSuite()
    suite.expect_min_rows(min_rows=1)

    runner = suite.build()

    @runner.validate
    def load_filtered_data(min_value: int, max_value: int):
        data = [i for i in range(min_value, max_value + 1)]
        return pd.DataFrame({"col1": data})

    df = load_filtered_data(5, 10)
    assert len(df) == 6, "Should return filtered DataFrame"
    assert list(df["col1"]) == [5, 6, 7, 8, 9, 10]


def test_decorator_preserves_function_metadata():
    """
    Test that the validate decorator preserves function metadata.
    """
    suite = DataFrameExpectationsSuite()
    suite.expect_min_rows(min_rows=0)

    runner = suite.build()

    @runner.validate
    def my_function():
        """This is my docstring."""
        return pd.DataFrame({"col1": []})

    assert my_function.__name__ == "my_function"
    assert my_function.__doc__ == "This is my docstring."


def test_decorator_with_none_not_allowed():
    """
    Test that the decorator raises ValueError when function returns None by default.
    """
    suite = DataFrameExpectationsSuite()
    suite.expect_min_rows(min_rows=1)

    runner = suite.build()

    @runner.validate
    def returns_none():
        return None

    with pytest.raises(ValueError, match="returned None"):
        returns_none()


def test_decorator_with_none_allowed():
    """
    Test that the decorator allows None when allow_none=True.
    """
    suite = DataFrameExpectationsSuite()
    suite.expect_min_rows(min_rows=1)

    runner = suite.build()

    @runner.validate(allow_none=True)
    def returns_none():
        return None

    result = returns_none()
    assert result is None


def test_decorator_with_none_allowed_validates_dataframe():
    """
    Test that when allow_none=True, the decorator still validates DataFrames.
    """
    suite = DataFrameExpectationsSuite()
    suite.expect_min_rows(min_rows=2)

    runner = suite.build()

    @runner.validate(allow_none=True)
    def returns_dataframe():
        return pd.DataFrame({"col1": [1]})

    # Should fail validation because min_rows=2 but only 1 row
    with pytest.raises(DataFrameExpectationsSuiteFailure):
        returns_dataframe()


def test_decorator_with_none_allowed_conditional():
    """
    Test that allow_none=True works for conditional DataFrame returns.
    """
    suite = DataFrameExpectationsSuite()
    suite.expect_min_rows(min_rows=1)

    runner = suite.build()

    @runner.validate(allow_none=True)
    def conditional_load(load: bool):
        if load:
            return pd.DataFrame({"col1": [1, 2, 3]})
        return None

    # Should validate DataFrame
    result_with_data = conditional_load(True)
    assert isinstance(result_with_data, pd.DataFrame)
    assert len(result_with_data) == 3

    # Should allow None
    result_none = conditional_load(False)
    assert result_none is None
