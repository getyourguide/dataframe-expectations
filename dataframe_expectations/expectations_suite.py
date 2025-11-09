from functools import wraps
from typing import Callable, List, Optional, cast

from dataframe_expectations.expectations import DataFrameLike
from dataframe_expectations.expectations.expectation_registry import (
    DataFrameExpectationRegistry,
)
from dataframe_expectations.logging_utils import setup_logger
from dataframe_expectations.result_message import (
    DataFrameExpectationFailureMessage,
    DataFrameExpectationSuccessMessage,
)

logger = setup_logger(__name__)


class DataFrameExpectationsSuiteFailure(Exception):
    """Raised when one or more expectations in the suite fail."""

    def __init__(
        self,
        total_expectations: int,
        failures: List[DataFrameExpectationFailureMessage],
        *args,
    ):
        self.failures = failures
        self.total_expectations = total_expectations
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

    This class is created by DataFrameExpectationsSuite.build() and contains
    a snapshot of expectations that won't change during execution.
    """

    def __init__(self, expectations: List):
        """
        Initialize the runner with a list of expectations.

        :param expectations: List of expectation instances to run.
        """
        self.__expectations = tuple(expectations)  # Immutable tuple

    def run(
        self,
        data_frame: DataFrameLike,
    ) -> None:
        """
        Run all expectations on the provided DataFrame with PySpark caching optimization.

        :param data_frame: The DataFrame to validate.
        """
        from dataframe_expectations import DataFrameType
        from dataframe_expectations.expectations import DataFrameExpectation

        successes = []
        failures = []
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

        if data_frame_type == DataFrameType.PYSPARK:
            from pyspark.sql import DataFrame as PySparkDataFrame

            pyspark_df = cast(PySparkDataFrame, data_frame)
            was_already_cached = pyspark_df.is_cached

            # Cache the DataFrame if it wasn't already cached
            if not was_already_cached:
                logger.debug("Caching PySpark DataFrame for expectations suite execution")
                pyspark_df.cache()
                # Update the original reference for subsequent operations
                data_frame = pyspark_df

        try:
            # Run all expectations
            for expectation in self.__expectations:
                result = expectation.validate(data_frame=data_frame)
                if isinstance(result, DataFrameExpectationSuccessMessage):
                    logger.info(
                        f"{expectation.get_expectation_name()} ({expectation.get_description()}) ... OK"
                    )
                    successes.append(result)
                elif isinstance(result, DataFrameExpectationFailureMessage):
                    logger.info(
                        f"{expectation.get_expectation_name()} ({expectation.get_description()}) ... FAIL"
                    )
                    failures.append(result)
                else:
                    raise ValueError(
                        f"Unexpected result type: {type(result)} for expectation: {expectation.get_expectation_name()}"
                    )
        finally:
            # Uncache the DataFrame if we cached it (and it wasn't already cached)
            if data_frame_type == DataFrameType.PYSPARK and not was_already_cached:
                from pyspark.sql import DataFrame as PySparkDataFrame

                logger.debug("Uncaching PySpark DataFrame after expectations suite execution")
                cast(PySparkDataFrame, data_frame).unpersist()

        footer_message = f"{len(successes)} success, {len(failures)} failures"
        footer_prefix = "=" * ((margin_len - len(footer_message) - 2) // 2)
        footer_suffix = "=" * (
            (margin_len - len(footer_message) - 2) // 2 + len(footer_message) % 2
        )
        logger.info(f"{footer_prefix} {footer_message} {footer_suffix}")

        if len(failures) > 0:
            raise DataFrameExpectationsSuiteFailure(
                total_expectations=len(self.__expectations), failures=failures
            )

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
                        logger.info(
                            f"Function '{f.__name__}' returned None, skipping validation (allow_none=True)"
                        )
                        return None
                    else:
                        raise ValueError(
                            f"Function '{f.__name__}' returned None. "
                            f"Use @runner.validate(allow_none=True) if this is intentional."
                        )

                # Validate the returned DataFrame
                logger.info(f"Validating DataFrame returned from '{f.__name__}'")
                self.run(data_frame=result)

                # Return the original DataFrame if validation passes
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
        suite = DataFrameExpectationsSuite()
        suite.expect_value_greater_than(column_name="age", value=18)
        suite.expect_value_less_than(column_name="salary", value=100000)

        runner = suite.build()
        runner.run(df1)
        runner.run(df2)  # Same expectations, different DataFrame
    """

    def __init__(self):
        """
        Initialize the expectation suite builder.
        """
        self.__expectations = []

    def __getattr__(self, name: str):
        """
        Dynamically create expectation methods.

        This is called when Python can't find an attribute through normal lookup.
        We use it to generate expect_* methods on-the-fly from the registry.
        """
        # Only handle expect_* methods
        if not name.startswith("expect_"):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

        mapping = DataFrameExpectationRegistry.get_suite_method_mapping()

        # Check if this method exists in the registry
        if name not in mapping:
            available = list(mapping.keys())
            raise AttributeError(
                f"Unknown expectation method '{name}'. "
                f"Available methods: {', '.join(available[:5])}..."
            )

        expectation_name = mapping[name]

        # Create and return the dynamic method
        return self._create_expectation_method(expectation_name, name)

    def _create_expectation_method(self, expectation_name: str, method_name: str):
        """
        Create a dynamic expectation method.

        Returns a closure that captures the expectation_name and self.
        """

        def dynamic_method(**kwargs):
            """Dynamically generated expectation method."""
            expectation = DataFrameExpectationRegistry.get_expectation(
                expectation_name=expectation_name, **kwargs
            )

            logger.info(f"Adding expectation: {expectation}")

            # Add to internal list
            self.__expectations.append(expectation)
            return self

        # Set helpful name for debugging
        dynamic_method.__name__ = method_name

        return dynamic_method

    def build(self) -> DataFrameExpectationsSuiteRunner:
        """
        Build an immutable runner from the current expectations.

        The runner contains a snapshot of expectations at the time of building.
        You can continue to add more expectations to this suite and build
        new runners without affecting previously built runners.

        :return: An immutable DataFrameExpectationsSuiteRunner instance.
        :raises ValueError: If no expectations have been added.
        """
        if not self.__expectations:
            raise ValueError(
                "Cannot build suite runner: no expectations added. "
                "Add at least one expectation using expect_* methods."
            )

        # Create a copy of expectations for the runner
        return DataFrameExpectationsSuiteRunner(list(self.__expectations))


if __name__ == "__main__":
    import pandas as pd

    # Example 1: Direct usage
    print("=== Example 1: Direct Usage ===")
    suite = DataFrameExpectationsSuite()
    suite.expect_value_greater_than(column_name="age", value=18)
    suite.expect_value_less_than(column_name="salary", value=1000)
    suite.expect_unique_rows(column_names=["id"])
    suite.expect_column_mean_between(column_name="age", min_value=20, max_value=40)
    suite.expect_column_max_between(column_name="salary", min_value=80000, max_value=85000)

    # Create a sample DataFrame
    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "age": [20, 25, 30, 35],
            "salary": [50000, 90000, 80000, 85000],
        }
    )

    # Build the runner and execute
    runner = suite.build()
    runner.run(data_frame=df)

    # Example 2: Decorator usage
    print("\n=== Example 2: Decorator Usage ===")
    suite = DataFrameExpectationsSuite()
    suite.expect_value_greater_than(column_name="age", value=20)
    suite.expect_unique_rows(column_names=["id"])

    runner = suite.build()

    @runner.validate
    def load_employee_data():
        """Load employee data with automatic validation."""
        return pd.DataFrame(
            {
                "id": [1, 2, 3],
                "age": [18, 30, 35],
                "name": ["Alice", "Bob", "Charlie"],
            }
        )

    # Function is automatically validated when called
    validated_df = load_employee_data()
    print(f"Successfully loaded and validated DataFrame with {len(validated_df)} rows")
