from functools import wraps
from typing import Any, Callable, Dict, List, Literal, Optional, cast

from dataframe_expectations.core.types import DataFrameLike
from dataframe_expectations.core.tagging import TagSet
from dataframe_expectations.registry import (
    DataFrameExpectationRegistry,
)
from dataframe_expectations.core.expectation import DataFrameExpectation
import logging

from dataframe_expectations.result_message import (
    DataFrameExpectationFailureMessage,
    DataFrameExpectationSuccessMessage,
)
from dataframe_expectations.core.suite_result import SuiteExecutionResult

logger = logging.getLogger(__name__)


class DataFrameExpectationsSuiteFailure(Exception):
    """Raised when one or more expectations in the suite fail."""

    def __init__(
        self,
        total_expectations: int,
        failures: List[DataFrameExpectationFailureMessage],
        result: Optional[SuiteExecutionResult] = None,
        *args,
    ):
        self.failures = failures
        self.total_expectations = total_expectations
        self.result = result
        super().__init__(*args)

    def __str__(self):
        margin_len = 80
        lines = [
            f"({len(self.failures)}/{self.total_expectations}) expectations failed.",
            "\n" + "=" * margin_len,
            "List of violations:",
            "-" * margin_len,
        ]

        for index, failure in enumerate(self.failures):
            lines.append(f"[Failed {index + 1}/{len(self.failures)}] {failure}")
            if index < len(self.failures) - 1:
                lines.append("-" * margin_len)

        lines.append("=" * margin_len)
        return "\n".join(lines)


class DataFrameExpectationsSuiteRunner:
    """
    Immutable runner for executing a fixed set of expectations.
    This class is created by DataFrameExpectationsSuite.build() and
    runs the expectations on provided DataFrames.
    """

    @staticmethod
    def _matches_tag_filter(
        expectation: Any,
        filter_tag_set: TagSet,
        tag_match_mode: Literal["any", "all"],
    ) -> bool:
        """
        Check if an expectation matches the tag filter criteria.

        :param expectation: Expectation instance to check.
        :param filter_tag_set: Tag filter to match against.
        :param tag_match_mode: Match mode - "any" (OR) or "all" (AND).
        :return: True if expectation matches filter, False otherwise.
        :raises ValueError: If tag_match_mode is invalid.
        """
        exp_tag_set = expectation.get_tags()

        # Check if expectation matches filter
        match tag_match_mode:
            case "any":
                return exp_tag_set.has_any_tag_from(filter_tag_set)
            case "all":
                return exp_tag_set.has_all_tags_from(filter_tag_set)
            case _:
                raise ValueError(f"Invalid tag_match_mode: {tag_match_mode}")

    def __init__(
        self,
        expectations: List[Any],
        suite_name: Optional[str] = None,
        violation_sample_limit: int = 5,
        tags: Optional[List[str]] = None,
        tag_match_mode: Optional[Literal["any", "all"]] = None,
    ):
        """
        Initialize the runner with a list of expectations and metadata.

        :param expectations: List of expectation instances.
        :param suite_name: Optional name for the suite.
        :param violation_sample_limit: Max number of violation rows to include in results.
        :param tags: Optional tag filters as list of strings in "key:value" format.
                    Example: ["priority:high", "priority:medium"]
                    If None or empty, all expectations will run.
        :param tag_match_mode: How to match tags - "any" (OR logic) or "all" (AND logic).
                              Required if tags are provided, must be None if tags are not provided.
                              - "any": Expectation matches if it has ANY of the filter tags
                              - "all": Expectation matches if it has ALL of the filter tags
        :raises ValueError: If tag_match_mode is provided without tags, or if tags are provided without tag_match_mode,
                           or if tag filters result in zero expectations to run.
        """
        self.__all_expectations = tuple(expectations)  # Store all expectations

        # Create filter TagSet from tags list
        self.__filter_tag_set = TagSet(tags)

        # Validate tags and tag_match_mode relationship
        if self.__filter_tag_set.is_empty() and tag_match_mode is not None:
            raise ValueError(
                "tag_match_mode cannot be provided when no tags are specified. "
                "Either provide tags or set tag_match_mode to None."
            )

        if not self.__filter_tag_set.is_empty() and tag_match_mode is None:
            raise ValueError(
                "tag_match_mode must be specified ('any' or 'all') when tags are provided."
            )

        self.__tag_match_mode = tag_match_mode

        # Filter expectations based on tags and track skipped ones
        if not self.__filter_tag_set.is_empty():
            # At this point, validation ensures tag_match_mode is not None
            assert tag_match_mode is not None
            filtered = []
            skipped = []
            for exp in self.__all_expectations:
                if self._matches_tag_filter(exp, self.__filter_tag_set, tag_match_mode):
                    filtered.append(exp)
                else:
                    skipped.append(exp)

            self.__expectations = tuple(filtered)
            self.__skipped_expectations = tuple(skipped)

            # Raise error if all expectations were filtered out
            if len(self.__expectations) == 0:
                error_message = (
                    f"Tag filter {self.__filter_tag_set} with mode '{tag_match_mode}' resulted in zero expectations to run. "
                    f"All {len(self.__all_expectations)} expectations were skipped. "
                    f"Please adjust your filter criteria."
                )
                logger.error(error_message)
                raise ValueError(error_message)

            logger.debug(
                f"Filtered {len(self.__all_expectations)} expectations to {len(self.__expectations)} "
                f"matching tags: {self.__filter_tag_set} (mode: {tag_match_mode}). Skipped: {len(self.__skipped_expectations)}"
            )
        else:
            self.__expectations = self.__all_expectations
            self.__skipped_expectations = tuple()  # No expectations skipped

        self.__suite_name = suite_name
        self.__violation_sample_limit = violation_sample_limit

    @property
    def selected_expectations_count(self) -> int:
        """Return the number of expectations that will run (after filtering)."""
        return len(self.__expectations)

    @property
    def total_expectations(self) -> int:
        """Return the total number of expectations before filtering."""
        return len(self.__all_expectations)

    @property
    def get_applied_tags(self) -> TagSet:
        """Return the applied tag filters for this runner."""
        return self.__filter_tag_set

    def list_all_expectations(self) -> List[str]:
        """
        Return a list of all expectation descriptions before filtering.

        :return: List of all expectation descriptions as strings in the format:
                 "ExpectationName (description)"
        """
        return [f"{exp}" for exp in self.__all_expectations]

    def list_selected_expectations(self) -> List[str]:
        """
        Return a list of selected expectation descriptions (after filtering).

        :return: List of selected expectation descriptions as strings in the format:
                 "ExpectationName (description)"
        """
        return [f"{exp}" for exp in self.__expectations]

    def run(
        self,
        data_frame: DataFrameLike,
        raise_on_failure: bool = True,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[SuiteExecutionResult]:
        """
        Run all expectations on the provided DataFrame with PySpark caching optimization.

        :param data_frame: The DataFrame to validate.
        :param raise_on_failure: If True (default), raises DataFrameExpectationsSuiteFailure on any failures.
                                If False, returns SuiteExecutionResult instead.
        :param context: Optional runtime context metadata (e.g., {"job_id": "123", "env": "prod"}).
        :return: None if raise_on_failure=True and all pass, SuiteExecutionResult if raise_on_failure=False.
        """
        from datetime import datetime
        from dataframe_expectations.core.types import DataFrameType
        from dataframe_expectations.core.suite_result import (
            ExpectationResult,
            SuiteExecutionResult,
            serialize_violations,
            ExpectationStatus,
        )

        # Track execution timing
        start_time = datetime.now()

        successes = []
        failures = []
        expectation_results = []
        margin_len = 80

        header_message = "Running expectations suite"
        header_prefix = "=" * ((margin_len - len(header_message) - 2) // 2)
        header_suffix = "=" * (
            (margin_len - len(header_message) - 2) // 2 - len(header_message) % 2
        )
        logger.info(f"{header_prefix} {header_message} {header_suffix}")

        # PySpark caching optimization
        data_frame_type = DataFrameExpectation.infer_data_frame_type(data_frame)
        was_already_cached = False
        dataframe_row_count = DataFrameExpectation.num_data_frame_rows(data_frame)

        if data_frame_type == DataFrameType.PYSPARK:
            from pyspark.sql import DataFrame as PySparkDataFrame

            pyspark_df = cast(PySparkDataFrame, data_frame)
            was_already_cached = pyspark_df.is_cached

            if not was_already_cached:
                logger.debug("Caching PySpark DataFrame for expectations suite execution")
                pyspark_df.cache()
                # Update the original reference for subsequent operations
                data_frame = pyspark_df

        try:
            # Run all expectations
            for expectation in self.__expectations:
                result = expectation.validate(data_frame=data_frame)
                # Get expectation's tags as TagSet
                exp_tag_set = expectation.get_tags()

                # Build ExpectationResult object using pattern matching
                match result:
                    case DataFrameExpectationSuccessMessage():
                        logger.debug(
                            f"{expectation.get_expectation_name()} ({expectation.get_description()}) ... OK"
                        )
                        successes.append(result)
                        expectation_results.append(
                            ExpectationResult(
                                expectation_name=expectation.get_expectation_name(),
                                description=expectation.get_description(),
                                status=ExpectationStatus.PASSED,
                                tags=exp_tag_set,
                                error_message=None,
                                violation_count=None,
                                violation_sample=None,
                            )
                        )
                    case DataFrameExpectationFailureMessage():
                        logger.warning(
                            f"{expectation.get_expectation_name()} ({expectation.get_description()}) ... FAIL"
                        )
                        failures.append(result)
                        # Serialize violations without storing raw dataframes
                        violations_df = result.get_violations_data_frame()
                        violation_count, violation_sample = serialize_violations(
                            violations_df, data_frame_type, self.__violation_sample_limit
                        )
                        expectation_results.append(
                            ExpectationResult(
                                expectation_name=expectation.get_expectation_name(),
                                description=expectation.get_description(),
                                status=ExpectationStatus.FAILED,
                                tags=exp_tag_set,
                                error_message=str(result),
                                violation_count=violation_count,
                                violation_sample=violation_sample,
                            )
                        )
                    case _:
                        raise ValueError(
                            f"Unexpected result type: {type(result)} for expectation: {expectation.get_expectation_name()}"
                        )
        finally:
            # Uncache the DataFrame if we cached it (and it wasn't already cached)
            if data_frame_type == DataFrameType.PYSPARK and not was_already_cached:
                from pyspark.sql import DataFrame as PySparkDataFrame

                logger.debug("Uncaching PySpark DataFrame after expectations suite execution")
                cast(PySparkDataFrame, data_frame).unpersist()

        # Track end time
        end_time = datetime.now()

        footer_message = f"{len(successes)} success, {len(failures)} failures"
        footer_prefix = "=" * ((margin_len - len(footer_message) - 2) // 2)
        footer_suffix = "=" * (
            (margin_len - len(footer_message) - 2) // 2 + len(footer_message) % 2
        )
        logger.info(f"{footer_prefix} {footer_message} {footer_suffix}")

        # Build skipped expectations list
        # Build skipped expectations as ExpectationResult with status="skipped"
        skipped_list = []
        for exp in self.__skipped_expectations:
            # Get expectation's tags as TagSet
            exp_tag_set = exp.get_tags()
            skipped_list.append(
                ExpectationResult(
                    expectation_name=exp.get_expectation_name(),
                    description=exp.get_description(),
                    status=ExpectationStatus.SKIPPED,
                    tags=exp_tag_set,
                    error_message=None,
                    violation_count=None,
                    violation_sample=None,
                )
            )

        # Build result object
        # Combine executed and skipped expectations
        all_results = expectation_results + skipped_list
        suite_result = SuiteExecutionResult(
            suite_name=self.__suite_name,
            context=context or {},
            applied_filters=self.__filter_tag_set,
            tag_match_mode=self.__tag_match_mode if not self.__filter_tag_set.is_empty() else None,
            results=all_results,
            start_time=start_time,
            end_time=end_time,
            dataframe_type=data_frame_type,
            dataframe_row_count=dataframe_row_count,
            dataframe_was_cached=was_already_cached,
        )

        # Dual-mode execution: raise exception or return result
        if len(failures) > 0 and raise_on_failure:
            raise DataFrameExpectationsSuiteFailure(
                total_expectations=len(self.__expectations),
                failures=failures,
                result=suite_result,
            )
        return suite_result

    def validate(self, func: Optional[Callable] = None, *, allow_none: bool = False) -> Callable:
        """
        Decorator to validate the DataFrame returned by a function.

        This decorator runs the expectations suite on the DataFrame returned
        by the decorated function. If validation fails, it raises
        DataFrameExpectationsSuiteFailure.

        Example:
            runner = suite.build()

            @runner.validate
            def load_data():
                return pd.read_csv("data.csv")

            df = load_data()  # Automatically validated

            # Allow None returns
            @runner.validate(allow_none=True)
            def maybe_load_data():
                if condition:
                    return pd.read_csv("data.csv")
                return None

        :param func: Function that returns a DataFrame.
        :param allow_none: If True, allows the function to return None without validation.
                          If False (default), None will raise a ValueError.
        :return: Wrapped function that validates the returned DataFrame.
        """

        def decorator(f: Callable) -> Callable:
            @wraps(f)
            def wrapper(*args, **kwargs):
                # Call the original function
                result = f(*args, **kwargs)

                # Handle None case
                if result is None:
                    if allow_none:
                        logger.debug(
                            f"Function '{f.__name__}' returned None, skipping validation (allow_none=True)"
                        )
                        return None
                    else:
                        raise ValueError(
                            f"Function '{f.__name__}' returned None. "
                            f"Use @runner.validate(allow_none=True) if this is intentional."
                        )

                # Validate the returned DataFrame
                logger.debug(f"Validating DataFrame returned from '{f.__name__}'")
                self.run(data_frame=result)

                return result

            return wrapper

        # Support both @validate and @validate(allow_none=True) syntax
        if func is None:
            # Called with arguments: @validate(allow_none=True)
            return decorator
        else:
            # Called without arguments: @validate
            return decorator(func)


class DataFrameExpectationsSuite:
    """
    A builder for creating expectation suites for validating DataFrames.

    Use this class to add expectations, then call build() to create an
    immutable runner that can execute the expectations on DataFrames.

    Example:
        suite = DataFrameExpectationsSuite(suite_name="user_validation")
        suite.expect_value_greater_than(
            column_name="age",
            value=18,
            tags=["priority:high", "category:compliance"]
        )
        suite.expect_value_less_than(
            column_name="salary",
            value=100000,
            tags=["priority:medium", "category:budget"]
        )
        suite.expect_min_rows(
            min_rows=10,
            tags=["priority:low", "category:data_quality"]
        )

        # Build runner for all expectations (no filtering)
        runner_all = suite.build()
        runner_all.run(df)  # Runs all 3 expectations

        # Build runner for high OR medium priority expectations (OR logic)
        runner_any = suite.build(tags=["priority:high", "priority:medium"], tag_match_mode="any")
        runner_any.run(df)  # Runs 2 expectations (age and salary checks)

        # Build runner for expectations with both high priority AND compliance category (AND logic)
        runner_and = suite.build(tags=["priority:high", "category:compliance"], tag_match_mode="all")
        runner_and.run(df)  # Runs 1 expectation (age check - has both tags)
    """

    def __init__(
        self,
        suite_name: Optional[str] = None,
        violation_sample_limit: int = 5,
    ):
        """
        Initialize the expectation suite builder.

        :param suite_name: Optional name for the suite (useful for logging/reporting).
        :param violation_sample_limit: Max number of violation rows to include in results (default 5).
        """
        self.__expectations: list[Any] = []  # List of expectation instances
        self.__suite_name = suite_name
        self.__violation_sample_limit = violation_sample_limit

    def __getattr__(self, name: str):
        """
        Dynamically create expectation methods.

        This is called when Python can't find an attribute through normal lookup.
        We use it to generate expect_* methods on-the-fly from the registry.
        """
        # Only handle expect_* methods
        if not name.startswith("expect_"):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

        # Create and return the dynamic method - validation happens in _create_expectation_method
        return self._create_expectation_method(name)

    def _create_expectation_method(self, suite_method_name: str):
        """
        Create a dynamic expectation method.

        Returns a closure that captures the suite_method_name and self.
        """

        def dynamic_method(tags: Optional[List[str]] = None, **kwargs):
            """Dynamically generated expectation method.

            :param tags: Optional tags as list of strings in "key:value" format.
                        Example: ["priority:high", "env:test"]
            :param **kwargs: Parameters for the expectation.
            """
            try:
                expectation = DataFrameExpectationRegistry.get_expectation_by_suite_method(
                    suite_method_name=suite_method_name, tags=tags, **kwargs
                )
            except ValueError as e:
                raise AttributeError(str(e)) from e

            logger.debug(f"Adding expectation: {expectation}")

            # Store expectation instance
            self.__expectations.append(expectation)
            return self

        # Set helpful name for debugging
        dynamic_method.__name__ = suite_method_name

        return dynamic_method

    def build(
        self,
        tags: Optional[List[str]] = None,
        tag_match_mode: Optional[Literal["any", "all"]] = None,
    ) -> DataFrameExpectationsSuiteRunner:
        """
        Build an immutable runner from the current expectations.

        This creates a snapshot of the current expectations in the suite.
        You can continue to add more expectations to this suite and build
        new runners without affecting previously built runners.

        :param tags: Optional tag filters as list of strings in "key:value" format.
                    Example: ["priority:high", "priority:medium"]
                    If None or empty, all expectations will be included.
        :param tag_match_mode: How to match tags - "any" (OR logic) or "all" (AND logic).
                              Required if tags are provided, must be None if tags are not provided.
                              - "any": Include expectations with ANY of the filter tags
                              - "all": Include expectations with ALL of the filter tags
        :return: An immutable DataFrameExpectationsSuiteRunner instance.
        :raises ValueError: If no expectations have been added, if tag_match_mode validation fails,
                           or if no expectations match the tag filters.
        """
        if not self.__expectations:
            raise ValueError(
                "Cannot build suite runner: no expectations added. "
                "Add at least one expectation using expect_* methods."
            )

        # Create a copy of expectations for the runner
        return DataFrameExpectationsSuiteRunner(
            expectations=list(self.__expectations),
            suite_name=self.__suite_name,
            violation_sample_limit=self.__violation_sample_limit,
            tags=tags,
            tag_match_mode=tag_match_mode,
        )
