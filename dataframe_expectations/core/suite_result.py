"""Suite execution result models for capturing validation outcomes."""

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, computed_field

from dataframe_expectations.core.types import DataFrameType, DataFrameLike
from dataframe_expectations.core.tagging import TagSet
import logging

from enum import Enum

logger = logging.getLogger(__name__)


class ExpectationStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ExpectationResult(BaseModel):
    """
    Representation of a single expectation result within a suite execution.
    Captures the outcome (passed, failed, skipped) using status.
    Does not store raw dataframes, only serialized violation samples.
    """

    expectation_name: str = Field(..., description="Name of the expectation class")
    description: str = Field(..., description="Human-readable description of the expectation")
    status: ExpectationStatus = Field(..., description="Outcome status: passed, failed, or skipped")
    tags: Optional[TagSet] = Field(
        default=None, description="User-defined tags for this specific expectation"
    )
    error_message: Optional[str] = Field(
        default=None, description="Error message if expectation failed"
    )
    violation_count: Optional[int] = Field(
        default=None, description="Total count of violations (if applicable)"
    )
    violation_sample: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Sample of violations as list of dicts (limited by violation_sample_limit)",
    )

    model_config = {"frozen": True, "arbitrary_types_allowed": True}  # Make immutable, allow TagSet


class SuiteExecutionResult(BaseModel):
    """Result of a complete suite execution.
    Captures all metadata about the suite run including timing, dataframe info,
    and individual expectation results. Does not store raw dataframes.
    """

    suite_name: Optional[str] = Field(default=None, description="Optional name for the suite")
    context: Dict[str, Any] = Field(
        default_factory=dict, description="Additional runtime metadata (e.g., job_id, environment)"
    )
    applied_filters: TagSet = Field(
        default_factory=TagSet, description="Tag filters that were applied to select expectations"
    )
    tag_match_mode: Optional[Literal["any", "all"]] = Field(
        default=None, description="How tags were matched: 'any' (OR) or 'all' (AND)"
    )
    results: List[ExpectationResult] = Field(
        ..., description="Results for each expectation in execution order (including skipped)"
    )
    start_time: datetime = Field(..., description="Suite execution start timestamp")
    end_time: datetime = Field(..., description="Suite execution end timestamp")
    dataframe_type: DataFrameType = Field(..., description="Type of dataframe validated")
    dataframe_row_count: int = Field(..., description="Number of rows in validated dataframe")
    dataframe_was_cached: bool = Field(
        default=False, description="Whether PySpark dataframe was cached during execution"
    )

    model_config = {"frozen": True, "arbitrary_types_allowed": True}  # Make immutable, allow TagSet

    @computed_field  # type: ignore[misc]
    @property
    def total_duration_seconds(self) -> float:
        """Total execution time in seconds."""
        return (self.end_time - self.start_time).total_seconds()

    @computed_field  # type: ignore[misc]
    @property
    def total_expectations(self) -> int:
        """Total number of expectations in the suite (including skipped)."""
        return len(self.results)

    @computed_field  # type: ignore[misc]
    @property
    def total_passed(self) -> int:
        """Number of expectations that passed."""
        return sum(1 for r in self.results if r.status == ExpectationStatus.PASSED)

    @computed_field  # type: ignore[misc]
    @property
    def total_failed(self) -> int:
        """Number of expectations that failed."""
        return sum(1 for r in self.results if r.status == ExpectationStatus.FAILED)

    @computed_field  # type: ignore[misc]
    @property
    def total_skipped(self) -> int:
        """Number of expectations that were skipped due to tag filtering."""
        return sum(1 for r in self.results if r.status == ExpectationStatus.SKIPPED)

    @computed_field  # type: ignore[misc]
    @property
    def pass_rate(self) -> float:
        """Percentage of expectations that passed (0.0 to 1.0)."""
        executed = self.total_passed + self.total_failed
        if executed == 0:
            return 1.0
        return self.total_passed / executed

    @computed_field  # type: ignore[misc]
    @property
    def success(self) -> bool:
        """Whether all executed expectations passed (ignores skipped)."""
        return self.total_failed == 0

    @computed_field  # type: ignore[misc]
    @property
    def passed_expectations(self) -> List[ExpectationResult]:
        """List of expectations that passed."""
        return [r for r in self.results if r.status == ExpectationStatus.PASSED]

    @computed_field  # type: ignore[misc]
    @property
    def failed_expectations(self) -> List[ExpectationResult]:
        """List of expectations that failed."""
        return [r for r in self.results if r.status == ExpectationStatus.FAILED]

    @computed_field  # type: ignore[misc]
    @property
    def skipped_expectations(self) -> List[ExpectationResult]:
        """List of expectations that were skipped due to tag filtering."""
        return [r for r in self.results if r.status == ExpectationStatus.SKIPPED]


def serialize_violations(
    violations_df: Optional[DataFrameLike],
    df_type: DataFrameType,
    limit: int = 5,
) -> tuple[Optional[int], Optional[List[Dict[str, Any]]]]:
    """Serialize violation dataframe to count and sample for storage.

    Converts dataframes to JSON-serializable format without storing raw dataframes.

    :param violations_df: DataFrame containing violations (pandas or PySpark).
    :param df_type: Type of the violations dataframe.
    :param limit: Maximum number of violation rows to include in sample.
    :return: Tuple of (total_count, sample_as_list_of_dicts).
    """
    if violations_df is None:
        return None, None

    count: Optional[int] = None
    sample: Optional[list[dict[str, Any]]] = None

    try:
        if df_type == DataFrameType.PANDAS:
            pandas_df = violations_df  # type: ignore[assignment]
            count = len(pandas_df)  # type: ignore[arg-type]
            sample = pandas_df.head(limit).to_dict("records")  # type: ignore[assignment,union-attr]
        elif df_type == DataFrameType.PYSPARK:
            pyspark_df = violations_df  # type: ignore[assignment]
            count = pyspark_df.count()  # type: ignore[assignment]
            sample = pyspark_df.limit(limit).toPandas().to_dict("records")  # type: ignore[assignment,operator]

        return count, sample
    except Exception:
        # If serialization fails, return None to avoid breaking the suite
        logger.warning("Failed to serialize violations dataframe", exc_info=True)
        return None, None
