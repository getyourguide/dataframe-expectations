"""Comprehensive unit tests for suite.py with tagging and result handling."""

import pytest
import pandas as pd
from dataframe_expectations.suite import (
    DataFrameExpectationsSuite,
    DataFrameExpectationsSuiteFailure,
)
from dataframe_expectations.core.suite_result import (
    SuiteExecutionResult,
    ExpectationStatus,
)
from dataframe_expectations.core.types import DataFrameType


def test_build_suite_no_tags():
    """Test building suite without tag filters."""
    suite = DataFrameExpectationsSuite()
    suite.expect_value_greater_than(column_name="col1", value=5)
    suite.expect_value_less_than(column_name="col1", value=20)

    runner = suite.build()

    assert runner.total_expectations == 2
    assert runner.selected_expectations_count == 2


def test_build_suite_with_any_tag_filter():
    """Test building suite with 'any' tag filter."""
    suite = DataFrameExpectationsSuite()
    suite.expect_value_greater_than(column_name="col1", value=5, tags=["priority:high"])
    suite.expect_value_less_than(column_name="col1", value=20, tags=["priority:medium"])
    suite.expect_min_rows(min_rows=1, tags=["priority:low"])

    # Filter for high or medium priority
    runner = suite.build(tags=["priority:high", "priority:medium"], tag_match_mode="any")

    assert runner.total_expectations == 3
    assert runner.selected_expectations_count == 2


def test_build_suite_with_all_tag_filter():
    """Test building suite with 'all' tag filter."""
    suite = DataFrameExpectationsSuite()
    suite.expect_value_greater_than(column_name="col1", value=5, tags=["priority:high", "env:prod"])
    suite.expect_value_less_than(column_name="col1", value=20, tags=["priority:high", "env:test"])
    suite.expect_min_rows(min_rows=1, tags=["priority:low"])

    # Filter for high priority AND prod environment
    runner = suite.build(tags=["priority:high", "env:prod"], tag_match_mode="all")

    assert runner.total_expectations == 3
    assert runner.selected_expectations_count == 1


def test_build_raises_error_tags_without_mode():
    """Test that building with tags but no mode raises ValueError."""
    suite = DataFrameExpectationsSuite()
    suite.expect_value_greater_than(column_name="col1", value=5, tags=["priority:high"])

    with pytest.raises(ValueError, match="tag_match_mode must be specified"):
        suite.build(tags=["priority:high"])


def test_build_raises_error_mode_without_tags():
    """Test that building with mode but no tags raises ValueError."""
    suite = DataFrameExpectationsSuite()
    suite.expect_value_greater_than(column_name="col1", value=5)

    with pytest.raises(ValueError, match="tag_match_mode cannot be provided"):
        suite.build(tag_match_mode="any")


def test_build_raises_error_all_filtered_out():
    """Test that filtering out all expectations raises ValueError."""
    suite = DataFrameExpectationsSuite()
    suite.expect_value_greater_than(column_name="col1", value=5, tags=["priority:low"])
    suite.expect_value_less_than(column_name="col1", value=20, tags=["priority:low"])

    with pytest.raises(ValueError, match="resulted in zero expectations"):
        suite.build(tags=["priority:high"], tag_match_mode="any")


@pytest.mark.parametrize(
    "filter_tags, match_mode, expected_executed, expected_skipped",
    [
        (None, None, 3, 0),  # No filter
        (["priority:high"], "any", 1, 2),  # Only high
        (["priority:medium"], "any", 1, 2),  # Only medium
        (["priority:high", "priority:medium"], "any", 2, 1),  # High or medium
        (["priority:high", "env:prod"], "all", 1, 2),  # High AND prod
        (["priority:high", "env:test"], "all", 0, 3),  # High AND test (none match)
    ],
)
def test_runner_tag_filtering(filter_tags, match_mode, expected_executed, expected_skipped):
    """Test runner with various tag filtering configurations."""
    suite = DataFrameExpectationsSuite()
    suite.expect_value_greater_than(column_name="col1", value=0, tags=["priority:high", "env:prod"])
    suite.expect_value_less_than(
        column_name="col1", value=100, tags=["priority:medium", "env:prod"]
    )
    suite.expect_min_rows(min_rows=1, tags=["priority:low"])

    df = pd.DataFrame({"col1": [10, 20, 30]})

    if filter_tags is None:
        runner = suite.build()
    else:
        try:
            runner = suite.build(tags=filter_tags, tag_match_mode=match_mode)
        except ValueError:
            # All filtered out
            assert expected_executed == 0
            return

    result = runner.run(data_frame=df, raise_on_failure=False)

    assert result is not None
    assert result.total_passed == expected_executed
    assert result.total_skipped == expected_skipped


def test_runner_lists_all_vs_selected_expectations():
    """Test list_all_expectations vs list_selected_expectations."""
    suite = DataFrameExpectationsSuite()
    suite.expect_value_greater_than(column_name="col1", value=0, tags=["priority:high"])
    suite.expect_value_less_than(column_name="col1", value=100, tags=["priority:medium"])
    suite.expect_min_rows(min_rows=1, tags=["priority:low"])

    runner = suite.build(tags=["priority:high", "priority:medium"], tag_match_mode="any")

    all_expectations = runner.list_all_expectations()
    selected_expectations = runner.list_selected_expectations()

    assert len(all_expectations) == 3
    assert len(selected_expectations) == 2


@pytest.mark.parametrize(
    "passing, failing, raise_on_failure, should_raise",
    [
        (3, 0, True, False),
        (3, 0, False, False),
        (2, 1, True, True),
        (2, 1, False, False),
        (0, 3, True, True),
        (0, 3, False, False),
    ],
)
def test_raise_on_failure_parametrized(passing, failing, raise_on_failure, should_raise):
    """Test raise_on_failure with various passing/failing combinations."""
    suite = DataFrameExpectationsSuite()

    # Add passing expectations
    for i in range(passing):
        suite.expect_value_greater_than(column_name="col1", value=0)

    # Add failing expectations
    for i in range(failing):
        suite.expect_value_greater_than(column_name="col1", value=100)

    df = pd.DataFrame({"col1": [10, 20, 30]})
    runner = suite.build()

    if should_raise:
        with pytest.raises(DataFrameExpectationsSuiteFailure) as exc_info:
            runner.run(data_frame=df, raise_on_failure=raise_on_failure)

        exception = exc_info.value
        assert len(exception.failures) == failing
        assert exception.result is not None
        assert isinstance(exception.result, SuiteExecutionResult)
        assert exception.result.total_passed == passing
        assert exception.result.total_failed == failing
    else:
        result = runner.run(data_frame=df, raise_on_failure=raise_on_failure)
        assert result is not None
        assert isinstance(result, SuiteExecutionResult)
        assert result.total_passed == passing
        assert result.total_failed == failing


def test_result_contains_execution_metadata():
    """Test that result contains execution metadata."""
    suite = DataFrameExpectationsSuite(suite_name="TestSuite")
    suite.expect_min_rows(min_rows=1)

    df = pd.DataFrame({"col1": [1, 2, 3]})
    runner = suite.build()

    result = runner.run(data_frame=df, context={"job_id": "123"})

    assert result is not None
    assert isinstance(result, SuiteExecutionResult)
    assert result.suite_name == "TestSuite"
    assert result.context == {"job_id": "123"}
    assert result.dataframe_type == DataFrameType.PANDAS
    assert result.dataframe_row_count == 3
    assert result.total_duration_seconds > 0


def test_result_contains_expectation_details():
    """Test that result contains detailed expectation results."""
    suite = DataFrameExpectationsSuite()
    suite.expect_value_greater_than(column_name="col1", value=0, tags=["priority:high"])
    suite.expect_value_greater_than(column_name="col1", value=100, tags=["priority:low"])

    df = pd.DataFrame({"col1": [10, 20, 30]})
    runner = suite.build()

    result = runner.run(data_frame=df, raise_on_failure=False)

    assert result is not None
    assert isinstance(result, SuiteExecutionResult)
    assert len(result.results) == 2

    # Check passed expectation
    passed = result.passed_expectations[0]
    assert passed.expectation_name == "ExpectationValueGreaterThan"
    assert passed.status == ExpectationStatus.PASSED
    assert passed.tags is not None
    assert len(passed.tags) == 1

    # Check failed expectation
    failed = result.failed_expectations[0]
    assert failed.expectation_name == "ExpectationValueGreaterThan"
    assert failed.status == ExpectationStatus.FAILED
    assert failed.error_message is not None
    assert failed.violation_count is not None


def test_result_with_skipped_expectations():
    """Test that result includes skipped expectations."""
    suite = DataFrameExpectationsSuite()
    suite.expect_value_greater_than(column_name="col1", value=0, tags=["priority:high"])
    suite.expect_value_less_than(column_name="col1", value=100, tags=["priority:low"])

    df = pd.DataFrame({"col1": [10, 20, 30]})
    runner = suite.build(tags=["priority:high"], tag_match_mode="any")

    result = runner.run(data_frame=df)

    assert result is not None
    assert isinstance(result, SuiteExecutionResult)
    assert result.total_expectations == 2
    assert result.total_passed == 1
    assert result.total_skipped == 1

    skipped = result.skipped_expectations[0]
    assert skipped.status == ExpectationStatus.SKIPPED
    assert skipped.expectation_name == "ExpectationValueLessThan"


def test_result_applied_filters():
    """Test that result captures applied tag filters."""
    suite = DataFrameExpectationsSuite()
    suite.expect_value_greater_than(column_name="col1", value=0, tags=["priority:high"])

    df = pd.DataFrame({"col1": [10]})
    runner = suite.build(tags=["priority:high"], tag_match_mode="any")

    result = runner.run(data_frame=df)

    assert result is not None
    assert len(result.applied_filters) == 1
    assert result.tag_match_mode == "any"


def test_pyspark_caching_enabled(spark):
    """Test that PySpark DataFrame is cached during execution."""
    suite = DataFrameExpectationsSuite()
    suite.expect_min_rows(min_rows=1)

    df = spark.createDataFrame([(1,), (2,), (3,)], ["col1"])
    runner = suite.build()

    # DataFrame should not be cached initially
    assert not df.is_cached

    result = runner.run(data_frame=df)

    # Should record that it wasn't cached before
    assert result is not None
    assert result.dataframe_was_cached is False
    assert result.dataframe_type == DataFrameType.PYSPARK

    # DataFrame should be uncached after execution
    assert not df.is_cached


def test_pyspark_already_cached(spark):
    """Test behavior when PySpark DataFrame is already cached."""
    suite = DataFrameExpectationsSuite()
    suite.expect_min_rows(min_rows=1)

    df = spark.createDataFrame([(1,), (2,), (3,)], ["col1"])
    df.cache()  # Pre-cache

    runner = suite.build()
    result = runner.run(data_frame=df)

    assert result is not None
    assert result.dataframe_was_cached is True
    assert df.is_cached  # Should still be cached

    df.unpersist()


def test_exception_contains_failures():
    """Test that exception contains failure details."""
    suite = DataFrameExpectationsSuite()
    suite.expect_value_greater_than(column_name="col1", value=100)
    suite.expect_value_less_than(column_name="col1", value=5)

    df = pd.DataFrame({"col1": [10, 20, 30]})
    runner = suite.build()

    with pytest.raises(DataFrameExpectationsSuiteFailure) as exc_info:
        runner.run(data_frame=df)

    exception = exc_info.value
    assert exception.total_expectations == 2
    assert len(exception.failures) == 2
    assert exception.result is not None


def test_exception_string_representation():
    """Test exception string representation."""
    suite = DataFrameExpectationsSuite()
    suite.expect_value_greater_than(column_name="col1", value=100)

    df = pd.DataFrame({"col1": [10, 20, 30]})
    runner = suite.build()

    with pytest.raises(DataFrameExpectationsSuiteFailure) as exc_info:
        runner.run(data_frame=df)

    exception_str = str(exc_info.value)
    assert "(1/1) expectations failed" in exception_str
    assert "List of violations:" in exception_str
