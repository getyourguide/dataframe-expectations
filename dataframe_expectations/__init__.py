"""DataFrame Expectations - A validation library for pandas and PySpark DataFrames."""

try:
    from importlib.metadata import version

    __version__ = version("dataframe-expectations")
except Exception:
    # Package is not installed (e.g., during development or linting)
    # Catch all exceptions to handle various edge cases in different environments
    __version__ = "0.0.0.dev0"

from dataframe_expectations.core.suite_result import (
    ExpectationResult,
    SuiteExecutionResult,
    serialize_violations,
)
from dataframe_expectations.suite import (
    DataFrameExpectationsSuite,
    DataFrameExpectationsSuiteRunner,
    DataFrameExpectationsSuiteFailure,
)

__all__ = [
    "ExpectationResult",
    "SuiteExecutionResult",
    "serialize_violations",
    "DataFrameExpectationsSuite",
    "DataFrameExpectationsSuiteRunner",
    "DataFrameExpectationsSuiteFailure",
]
