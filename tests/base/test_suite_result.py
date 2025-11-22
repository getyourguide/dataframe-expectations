"""Comprehensive unit tests for SuiteExecutionResult and ExpectationResult."""

import pytest
from datetime import datetime, timedelta
from dataframe_expectations.core.suite_result import (
    ExpectationResult,
    ExpectationStatus,
    SuiteExecutionResult,
    serialize_violations,
)
from dataframe_expectations.core.types import DataFrameType
from dataframe_expectations.core.tagging import TagSet
import pandas as pd


@pytest.mark.parametrize(
    "status, error_message, violation_count, violation_sample",
    [
        (ExpectationStatus.PASSED, None, None, None),
        (ExpectationStatus.FAILED, "Validation failed", 10, [{"col": "value"}]),
        (ExpectationStatus.SKIPPED, None, None, None),
    ],
)
def test_expectation_result_creation(status, error_message, violation_count, violation_sample):
    """Test creating ExpectationResult with various statuses."""
    result = ExpectationResult(
        expectation_name="TestExpectation",
        description="Test description",
        status=status,
        tags=TagSet(["priority:high"]),
        error_message=error_message,
        violation_count=violation_count,
        violation_sample=violation_sample,
    )

    assert result.expectation_name == "TestExpectation"
    assert result.description == "Test description"
    assert result.status == status
    assert result.error_message == error_message
    assert result.violation_count == violation_count
    assert result.violation_sample == violation_sample


def test_expectation_result_immutability():
    """Test that ExpectationResult is immutable."""
    result = ExpectationResult(
        expectation_name="TestExpectation",
        description="Test description",
        status=ExpectationStatus.PASSED,
    )

    with pytest.raises(Exception):  # Pydantic raises ValidationError or AttributeError
        result.status = ExpectationStatus.FAILED  # type: ignore


def test_expectation_result_with_tags():
    """Test ExpectationResult with tags."""
    tags = TagSet(["priority:high", "env:test"])
    result = ExpectationResult(
        expectation_name="TestExpectation",
        description="Test description",
        status=ExpectationStatus.PASSED,
        tags=tags,
    )

    assert result.tags is not None
    assert len(result.tags) == 2


def test_expectation_result_without_tags():
    """Test ExpectationResult without tags."""
    result = ExpectationResult(
        expectation_name="TestExpectation",
        description="Test description",
        status=ExpectationStatus.PASSED,
    )

    assert result.tags is None


def test_suite_result_basic_creation():
    """Test creating a basic SuiteExecutionResult."""
    start = datetime.now()
    end = start + timedelta(seconds=5)

    result = SuiteExecutionResult(
        suite_name="TestSuite",
        context={"job_id": "123"},
        applied_filters=TagSet(["priority:high"]),
        tag_match_mode="any",
        results=[],
        start_time=start,
        end_time=end,
        dataframe_type=DataFrameType.PANDAS,
        dataframe_row_count=100,
    )

    assert result.suite_name == "TestSuite"
    assert result.context == {"job_id": "123"}
    assert len(result.applied_filters) == 1
    assert result.tag_match_mode == "any"
    assert result.dataframe_type == DataFrameType.PANDAS
    assert result.dataframe_row_count == 100


def test_suite_result_immutability():
    """Test that SuiteExecutionResult is immutable."""
    start = datetime.now()
    end = start + timedelta(seconds=5)

    result = SuiteExecutionResult(
        results=[],
        start_time=start,
        end_time=end,
        dataframe_type=DataFrameType.PANDAS,
        dataframe_row_count=100,
    )

    with pytest.raises(Exception):
        result.suite_name = "NewName"  # type: ignore


@pytest.fixture
def sample_results():
    """Create sample expectation results."""
    return [
        ExpectationResult(
            expectation_name="Exp1",
            description="Desc1",
            status=ExpectationStatus.PASSED,
        ),
        ExpectationResult(
            expectation_name="Exp2",
            description="Desc2",
            status=ExpectationStatus.PASSED,
        ),
        ExpectationResult(
            expectation_name="Exp3",
            description="Desc3",
            status=ExpectationStatus.FAILED,
            error_message="Failed",
            violation_count=5,
        ),
        ExpectationResult(
            expectation_name="Exp4",
            description="Desc4",
            status=ExpectationStatus.SKIPPED,
        ),
    ]


def test_total_duration_seconds():
    """Test total_duration_seconds computation."""
    start = datetime(2024, 1, 1, 12, 0, 0)
    end = datetime(2024, 1, 1, 12, 0, 10)

    result = SuiteExecutionResult(
        results=[],
        start_time=start,
        end_time=end,
        dataframe_type=DataFrameType.PANDAS,
        dataframe_row_count=100,
    )

    assert result.total_duration_seconds == 10.0


def test_total_expectations(sample_results):
    """Test total_expectations computation."""
    result = SuiteExecutionResult(
        results=sample_results,
        start_time=datetime.now(),
        end_time=datetime.now(),
        dataframe_type=DataFrameType.PANDAS,
        dataframe_row_count=100,
    )

    assert result.total_expectations == 4


def test_total_counts(sample_results):
    """Test total_passed, total_failed, and total_skipped computation."""
    result = SuiteExecutionResult(
        results=sample_results,
        start_time=datetime.now(),
        end_time=datetime.now(),
        dataframe_type=DataFrameType.PANDAS,
        dataframe_row_count=100,
    )

    assert result.total_passed == 2
    assert result.total_failed == 1
    assert result.total_skipped == 1


@pytest.mark.parametrize(
    "passed, failed, expected_rate",
    [
        (3, 0, 1.0),
        (2, 1, 2 / 3),
        (1, 2, 1 / 3),
        (0, 3, 0.0),
        (0, 0, 1.0),  # Edge case: no executed expectations
    ],
)
def test_pass_rate(passed, failed, expected_rate):
    """Test pass_rate computation with various scenarios."""
    statuses = [ExpectationStatus.PASSED] * passed + [ExpectationStatus.FAILED] * failed
    results = []
    for i, status in enumerate(statuses):
        results.append(
            ExpectationResult(
                expectation_name=f"Exp{i}",
                description="Desc",
                status=status,
                error_message="Failed" if status == ExpectationStatus.FAILED else None,
            )
        )

    result = SuiteExecutionResult(
        results=results,
        start_time=datetime.now(),
        end_time=datetime.now(),
        dataframe_type=DataFrameType.PANDAS,
        dataframe_row_count=100,
    )

    assert result.pass_rate == pytest.approx(expected_rate)


@pytest.mark.parametrize(
    "passed, failed, skipped, expected_success",
    [
        (3, 0, 0, True),
        (3, 0, 2, True),  # Skipped doesn't affect success
        (2, 1, 0, False),
        (0, 0, 3, True),  # All skipped = success
    ],
)
def test_success(passed, failed, skipped, expected_success):
    """Test success computation."""
    statuses = (
        [ExpectationStatus.PASSED] * passed
        + [ExpectationStatus.FAILED] * failed
        + [ExpectationStatus.SKIPPED] * skipped
    )
    results = []
    for i, status in enumerate(statuses):
        results.append(
            ExpectationResult(
                expectation_name=f"Exp{i}",
                description="Desc",
                status=status,
                error_message="Failed" if status == ExpectationStatus.FAILED else None,
            )
        )

    result = SuiteExecutionResult(
        results=results,
        start_time=datetime.now(),
        end_time=datetime.now(),
        dataframe_type=DataFrameType.PANDAS,
        dataframe_row_count=100,
    )

    assert result.success == expected_success


def test_expectation_lists(sample_results):
    """Test passed_expectations, failed_expectations, skipped_expectations."""
    result = SuiteExecutionResult(
        results=sample_results,
        start_time=datetime.now(),
        end_time=datetime.now(),
        dataframe_type=DataFrameType.PANDAS,
        dataframe_row_count=100,
    )

    assert len(result.passed_expectations) == 2
    assert len(result.failed_expectations) == 1
    assert len(result.skipped_expectations) == 1

    assert all(r.status == ExpectationStatus.PASSED for r in result.passed_expectations)
    assert all(r.status == ExpectationStatus.FAILED for r in result.failed_expectations)
    assert all(r.status == ExpectationStatus.SKIPPED for r in result.skipped_expectations)


def test_serialize_pandas_violations():
    """Test serializing pandas DataFrame violations."""
    violations_df = pd.DataFrame({"col1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})

    count, sample = serialize_violations(violations_df, DataFrameType.PANDAS, limit=5)

    assert count == 10
    assert sample is not None
    assert len(sample) == 5
    assert sample[0] == {"col1": 1}


def test_serialize_pyspark_violations(spark):
    """Test serializing PySpark DataFrame violations."""
    violations_df = spark.createDataFrame([(i,) for i in range(1, 11)], ["col1"])

    count, sample = serialize_violations(violations_df, DataFrameType.PYSPARK, limit=5)

    assert count == 10
    assert sample is not None
    assert len(sample) == 5
    assert sample[0] == {"col1": 1}


def test_serialize_none_violations():
    """Test serializing None violations."""
    count, sample = serialize_violations(None, DataFrameType.PANDAS, limit=5)

    assert count is None
    assert sample is None


@pytest.mark.parametrize(
    "limit, expected_len",
    [
        (1, 1),
        (3, 3),
        (10, 10),
        (100, 10),  # More than available
    ],
)
def test_serialize_with_different_limits(limit, expected_len):
    """Test serializing with different limit values."""
    violations_df = pd.DataFrame({"col1": list(range(10))})

    count, sample = serialize_violations(violations_df, DataFrameType.PANDAS, limit=limit)

    assert count == 10
    assert sample is not None
    assert len(sample) == min(expected_len, 10)


def test_suite_result_with_tag_filters():
    """Test SuiteExecutionResult with tag filters applied."""
    results = [
        ExpectationResult(
            expectation_name="Exp1",
            description="Desc1",
            status=ExpectationStatus.PASSED,
            tags=TagSet(["priority:high"]),
        ),
        ExpectationResult(
            expectation_name="Exp2",
            description="Desc2",
            status=ExpectationStatus.SKIPPED,
            tags=TagSet(["priority:low"]),
        ),
    ]

    result = SuiteExecutionResult(
        suite_name="FilteredSuite",
        applied_filters=TagSet(["priority:high"]),
        tag_match_mode="any",
        results=results,
        start_time=datetime.now(),
        end_time=datetime.now(),
        dataframe_type=DataFrameType.PANDAS,
        dataframe_row_count=100,
    )

    assert result.total_expectations == 2
    assert result.total_passed == 1
    assert result.total_skipped == 1
    assert len(result.applied_filters) == 1


@pytest.mark.parametrize(
    "tag_match_mode",
    ["any", "all", None],
)
def test_suite_result_tag_match_modes(tag_match_mode):
    """Test SuiteExecutionResult with different tag match modes."""
    filters = TagSet(["priority:high"]) if tag_match_mode else TagSet()

    result = SuiteExecutionResult(
        applied_filters=filters,
        tag_match_mode=tag_match_mode,
        results=[],
        start_time=datetime.now(),
        end_time=datetime.now(),
        dataframe_type=DataFrameType.PANDAS,
        dataframe_row_count=100,
    )

    assert result.tag_match_mode == tag_match_mode


def test_suite_result_invalid_tag_match_mode():
    """Test that invalid tag_match_mode raises validation error."""
    with pytest.raises(Exception):  # Pydantic raises ValidationError
        SuiteExecutionResult(
            applied_filters=TagSet(["priority:high"]),
            tag_match_mode="invalid",  # type: ignore
            results=[],
            start_time=datetime.now(),
            end_time=datetime.now(),
            dataframe_type=DataFrameType.PANDAS,
            dataframe_row_count=100,
        )
