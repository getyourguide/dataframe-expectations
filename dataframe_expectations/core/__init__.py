"""Core base classes and interfaces for DataFrame expectations."""

from dataframe_expectations.core.suite_result import (
    ExpectationResult,
    ExpectationStatus,
    SuiteExecutionResult,
    serialize_violations,
)

__all__ = [
    "ExpectationResult",
    "ExpectationStatus",
    "SuiteExecutionResult",
    "serialize_violations",
]
