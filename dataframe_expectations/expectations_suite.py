from typing import List, Union, cast

from dataframe_expectations.expectations import DataFrameLike
from dataframe_expectations.expectations.expectation_registry import (
    DataframeExpectationRegistry,
)
from dataframe_expectations.logging_utils import setup_logger
from dataframe_expectations.result_message import (
    DataframeExpectationFailureMessage,
    DataframeExpectationSuccessMessage,
)

logger = setup_logger(__name__)


class DataframeExpectationsSuiteFailure(Exception):
    """Raised when one or more expectations in the suite fail."""

    def __init__(
        self,
        total_expectations: int,
        failures: List[DataframeExpectationFailureMessage],
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


class DataframeExpectationsSuite:
    """
    A suite of expectations for validating DataFrames.
    """

    def __init__(self):
        """
        Initialize the expectation suite.
        """
        self.__expectations = []

    # Expectations for any data type

    def expect_value_equals(
        self,
        column_name: str,
        value: object,
    ):
        """
        Add an expectation to check if the values in a column equal a specified value.

        Categories:
          category: Column Expectations
          subcategory: Any Value

        :param column_name: The name of the column to check.
        :param value: The value to compare against.
        :return: an instance of DataframeExpectationsSuite.
        """
        expectation = DataframeExpectationRegistry.get_expectation(
            expectation_name="ExpectationValueEquals",
            column_name=column_name,
            value=value,
        )

        logger.info(f"Adding expectation: {expectation}")
        self.__expectations.append(expectation)
        return self

    def expect_value_not_equals(
        self,
        column_name: str,
        value: object,
    ):
        """
        Add an expectation to check if the values in a column do not equal a specified value.

        Categories:
          category: Column Expectations
          subcategory: Any Value

        :param column_name: The name of the column to check.
        :param value: The value to compare against.
        :return: an instance of DataframeExpectationsSuite.
        """
        expectation = DataframeExpectationRegistry.get_expectation(
            expectation_name="ExpectationValueNotEquals",
            column_name=column_name,
            value=value,
        )

        logger.info(f"Adding expectation: {expectation}")
        self.__expectations.append(expectation)
        return self

    def expect_value_null(
        self,
        column_name: str,
    ):
        """
        Add an expectation to check if the values in a column are null.

        Categories:
          category: Column Expectations
          subcategory: Any Value

        :param column_name: The name of the column to check.
        :return: an instance of DataframeExpectationsSuite.
        """
        expectation = DataframeExpectationRegistry.get_expectation(
            expectation_name="ExpectationValueNull",
            column_name=column_name,
        )

        logger.info(f"Adding expectation: {expectation}")
        self.__expectations.append(expectation)
        return self

    def expect_value_not_null(
        self,
        column_name: str,
    ):
        """
        Add an expectation to check if the values in a column are not null.

        Categories:
          category: Column Expectations
          subcategory: Any Value

        :param column_name: The name of the column to check.
        :return: an instance of DataframeExpectationsSuite.
        """
        expectation = DataframeExpectationRegistry.get_expectation(
            expectation_name="ExpectationValueNotNull",
            column_name=column_name,
        )

        logger.info(f"Adding expectation: {expectation}")
        self.__expectations.append(expectation)
        return self

    def expect_value_in(
        self,
        column_name: str,
        values: List[object],
    ):
        """
        Add an expectation to check if the values in a column are in a specified list of values.

        Categories:
          category: Column Expectations
          subcategory: Any Value

        :param column_name: The name of the column to check.
        :param values: The list of values to compare against.
        :return: an instance of DataframeExpectationsSuite.
        """
        expectation = DataframeExpectationRegistry.get_expectation(
            expectation_name="ExpectationValueIn",
            column_name=column_name,
            values=values,
        )

        logger.info(f"Adding expectation: {expectation}")
        self.__expectations.append(expectation)
        return self

    def expect_value_not_in(
        self,
        column_name: str,
        values: List[object],
    ):
        """
        Add an expectation to check if the values in a column are not in a specified list of values.

        Categories:
          category: Column Expectations
          subcategory: Any Value

        :param column_name: The name of the column to check.
        :param values: The list of values to compare against.
        :return: an instance of DataframeExpectationsSuite.
        """
        expectation = DataframeExpectationRegistry.get_expectation(
            expectation_name="ExpectationValueNotIn",
            column_name=column_name,
            values=values,
        )

        logger.info(f"Adding expectation: {expectation}")
        self.__expectations.append(expectation)
        return self

    # Expectations for numerical data types

    def expect_value_greater_than(
        self,
        column_name: str,
        value: float,
    ):
        """
        Add an expectation to check if the values in a column are greater than a specified value.

        Categories:
          category: Column Expectations
          subcategory: Numerical

        :param column_name: The name of the column to check.
        :param value: The value to compare against.
        :return: an instance of DataframeExpectationsSuite.
        """
        expectation = DataframeExpectationRegistry.get_expectation(
            expectation_name="ExpectationValueGreaterThan",
            column_name=column_name,
            value=value,
        )

        logger.info(f"Adding expectation: {expectation}")
        self.__expectations.append(expectation)
        return self

    def expect_value_less_than(
        self,
        column_name: str,
        value: float,
    ):
        """
        Add an expectation to check if the values in a column are less than a specified value.

        Categories:
          category: Column Expectations
          subcategory: Numerical

        :param column_name: The name of the column to check.
        :param value: The value to compare against.
        :return: an instance of DataframeExpectationsSuite.
        """
        expectation = DataframeExpectationRegistry.get_expectation(
            expectation_name="ExpectationValueLessThan",
            column_name=column_name,
            value=value,
        )
        logger.info(f"Adding expectation: {expectation}")
        self.__expectations.append(expectation)
        return self

    def expect_value_between(
        self,
        column_name: str,
        min_value: float,
        max_value: float,
    ):
        """
        Add an expectation to check if the values in a column are between two specified values.

        Categories:
          category: Column Expectations
          subcategory: Numerical

        :param column_name: The name of the column to check.
        :param min_value: The minimum value for the range.
        :param max_value: The maximum value for the range.
        :return: an instance of DataframeExpectationsSuite.
        """
        expectation = DataframeExpectationRegistry.get_expectation(
            expectation_name="ExpectationValueBetween",
            column_name=column_name,
            min_value=min_value,
            max_value=max_value,
        )

        logger.info(f"Adding expectation: {expectation}")
        self.__expectations.append(expectation)
        return self

    # Expectations for string data types

    def expect_string_contains(
        self,
        column_name: str,
        substring: str,
    ):
        """
        Add an expectation to check if the values in a string column contain a specified substring.

        Categories:
          category: Column Expectations
          subcategory: String

        :param column_name: The name of the column to check.
        :param substring: The substring to search for.
        :return: an instance of DataframeExpectationsSuite.
        """
        expectation = DataframeExpectationRegistry.get_expectation(
            expectation_name="ExpectationStringContains",
            column_name=column_name,
            substring=substring,
        )

        logger.info(f"Adding expectation: {expectation}")
        self.__expectations.append(expectation)
        return self

    def expect_string_not_contains(
        self,
        column_name: str,
        substring: str,
    ):
        """
        Add an expectation to check if the values in a string column do not contain a specified substring.

        Categories:
          category: Column Expectations
          subcategory: String

        :param column_name: The name of the column to check.
        :param substring: The substring to search for.
        :return: an instance of DataframeExpectationsSuite.
        """
        expectation = DataframeExpectationRegistry.get_expectation(
            expectation_name="ExpectationStringNotContains",
            column_name=column_name,
            substring=substring,
        )

        logger.info(f"Adding expectation: {expectation}")
        self.__expectations.append(expectation)
        return self

    def expect_string_starts_with(
        self,
        column_name: str,
        prefix: str,
    ):
        """
        Add an expectation to check if the values in a string column start with a specified prefix.

        Categories:
          category: Column Expectations
          subcategory: String

        :param column_name: The name of the column to check.
        :param prefix: The prefix to search for.
        :return: an instance of DataframeExpectationsSuite.
        """
        expectation = DataframeExpectationRegistry.get_expectation(
            expectation_name="ExpectationStringStartsWith",
            column_name=column_name,
            prefix=prefix,
        )

        logger.info(f"Adding expectation: {expectation}")
        self.__expectations.append(expectation)
        return self

    def expect_string_ends_with(
        self,
        column_name: str,
        suffix: str,
    ):
        """
        Add an expectation to check if the values in a string column end with a specified suffix.

        Categories:
          category: Column Expectations
          subcategory: String

        :param column_name: The name of the column to check.
        :param suffix: The suffix to search for.
        :return: an instance of DataframeExpectationsSuite.
        """
        expectation = DataframeExpectationRegistry.get_expectation(
            expectation_name="ExpectationStringEndsWith",
            column_name=column_name,
            suffix=suffix,
        )

        logger.info(f"Adding expectation: {expectation}")
        self.__expectations.append(expectation)
        return self

    def expect_string_length_less_than(
        self,
        column_name: str,
        length: int,
    ):
        """
        Add an expectation to check if the length of the values in a string column is less than a specified length.

        Categories:
          category: Column Expectations
          subcategory: String

        :param column_name: The name of the column to check.
        :param length: The length that the values should be less than.
        :return: an instance of DataframeExpectationsSuite.
        """
        expectation = DataframeExpectationRegistry.get_expectation(
            expectation_name="ExpectationStringLengthLessThan",
            column_name=column_name,
            length=length,
        )

        logger.info(f"Adding expectation: {expectation}")
        self.__expectations.append(expectation)
        return self

    def expect_string_length_greater_than(
        self,
        column_name: str,
        length: int,
    ):
        """
        Add an expectation to check if the length of the values in a string column is greater than a specified length.

        Categories:
          category: Column Expectations
          subcategory: String

        :param column_name: The name of the column to check.
        :param length: The length that the values should be greater than.
        :return: an instance of DataframeExpectationsSuite.
        """
        expectation = DataframeExpectationRegistry.get_expectation(
            expectation_name="ExpectationStringLengthGreaterThan",
            column_name=column_name,
            length=length,
        )

        logger.info(f"Adding expectation: {expectation}")
        self.__expectations.append(expectation)
        return self

    def expect_string_length_between(
        self,
        column_name: str,
        min_length: int,
        max_length: int,
    ):
        """
        Add an expectation to check if the length of the values in a string column is between two specified lengths.

        Categories:
          category: Column Expectations
          subcategory: String

        :param column_name: The name of the column to check.
        :param min_length: The minimum length that the values should be.
        :param max_length: The maximum length that the values should be.
        :return: an instance of DataframeExpectationsSuite.
        """
        expectation = DataframeExpectationRegistry.get_expectation(
            expectation_name="ExpectationStringLengthBetween",
            column_name=column_name,
            min_length=min_length,
            max_length=max_length,
        )

        logger.info(f"Adding expectation: {expectation}")
        self.__expectations.append(expectation)
        return self

    def expect_string_length_equals(
        self,
        column_name: str,
        length: int,
    ):
        """
        Add an expectation to check if the length of the values in a string column equals a specified length.

        Categories:
          category: Column Expectations
          subcategory: String

        :param column_name: The name of the column to check.
        :param length: The length that the values should equal.
        :return: an instance of DataframeExpectationsSuite.
        """
        expectation = DataframeExpectationRegistry.get_expectation(
            expectation_name="ExpectationStringLengthEquals",
            column_name=column_name,
            length=length,
        )

        logger.info(f"Adding expectation: {expectation}")
        self.__expectations.append(expectation)
        return self

    # Expectations for aggregation data types
    def expect_min_rows(
        self,
        min_rows: int,
    ):
        """
        Add an expectation to check if the DataFrame has at least a minimum number of rows.
        Categories:
          category: DataFrame Aggregation Expectations
          subcategory: Any Value
        :param min_rows: The minimum number of rows expected.
        :return: an instance of DataframeExpectationsSuite.
        """
        expectation = DataframeExpectationRegistry.get_expectation(
            expectation_name="ExpectationMinRows",
            min_rows=min_rows,
        )
        logger.info(f"Adding expectation: {expectation}")
        self.__expectations.append(expectation)
        return self

    def expect_max_rows(
        self,
        max_rows: int,
    ):
        """
        Add an expectation to check if the DataFrame has at most a maximum number of rows.
        Categories:
          category: DataFrame Aggregation Expectations
          subcategory: Any Value
        :param max_rows: The maximum number of rows expected.
        :return: an instance of DataframeExpectationsSuite.
        """
        expectation = DataframeExpectationRegistry.get_expectation(
            expectation_name="ExpectationMaxRows",
            max_rows=max_rows,
        )
        logger.info(f"Adding expectation: {expectation}")
        self.__expectations.append(expectation)
        return self

    def expect_max_null_percentage(
        self,
        column_name: str,
        max_percentage: float,
    ):
        """
        Add an expectation to check if the percentage of null/NaN values in a specific column is below a threshold.
        Categories:
          category: Column Aggregation Expectations
          subcategory: Any Value
        :param column_name: The name of the column to check for null percentage.
        :param max_percentage: The maximum allowed percentage of null/NaN values (0.0 to 100.0).
        :return: an instance of DataframeExpectationsSuite.
        """
        expectation = DataframeExpectationRegistry.get_expectation(
            expectation_name="ExpectationMaxNullPercentage",
            column_name=column_name,
            max_percentage=max_percentage,
        )
        logger.info(f"Adding expectation: {expectation}")
        self.__expectations.append(expectation)
        return self

    def expect_max_null_count(
        self,
        column_name: str,
        max_count: int,
    ):
        """
        Add an expectation to check if the count of null/NaN values in a specific column is below a threshold.
        Categories:
          category: Column Aggregation Expectations
          subcategory: Any Value
        :param column_name: The name of the column to check for null count.
        :param max_count: The maximum allowed count of null/NaN values.
        :return: an instance of DataframeExpectationsSuite.
        """
        expectation = DataframeExpectationRegistry.get_expectation(
            expectation_name="ExpectationMaxNullCount",
            column_name=column_name,
            max_count=max_count,
        )
        logger.info(f"Adding expectation: {expectation}")
        self.__expectations.append(expectation)
        return self

    def expect_unique_rows(
        self,
        column_names: List[str],
    ):
        """
        Add an expectation to check if the rows in the DataFrame are unique based on specified columns.

        Categories:
          category: Column Aggregation Expectations
          subcategory: Any Value

        :param column_names: The list of column names to check for uniqueness.
        :return: an instance of DataframeExpectationsSuite.
        """
        expectation = DataframeExpectationRegistry.get_expectation(
            expectation_name="ExpectationUniqueRows",
            column_names=column_names,
        )

        logger.info(f"Adding expectation: {expectation}")
        self.__expectations.append(expectation)
        return self

    def expect_distinct_column_values_equals(
        self,
        column_name: str,
        expected_value: int,
    ):
        """
        Add an expectation to check if the number of distinct values in a column equals an expected count.
        Categories:
          category: Column Aggregation Expectations
          subcategory: Any Value
        :param column_name: The name of the column to check.
        :param expected_value: The expected number of distinct values (exact match).
        :return: an instance of DataframeExpectationsSuite.
        """
        expectation = DataframeExpectationRegistry.get_expectation(
            expectation_name="ExpectationDistinctColumnValuesEquals",
            column_name=column_name,
            expected_value=expected_value,
        )
        logger.info(f"Adding expectation: {expectation}")
        self.__expectations.append(expectation)
        return self

    def expect_distinct_column_values_less_than(
        self,
        column_name: str,
        threshold: int,
    ):
        """
        Add an expectation to check if the number of distinct values in a column is less than a threshold.
        Categories:
          category: Column Aggregation Expectations
          subcategory: Any Value
        :param column_name: The name of the column to check.
        :param threshold: The threshold for distinct values count (exclusive upper bound).
        :return: an instance of DataframeExpectationsSuite.
        """
        expectation = DataframeExpectationRegistry.get_expectation(
            expectation_name="ExpectationDistinctColumnValuesLessThan",
            column_name=column_name,
            threshold=threshold,
        )
        logger.info(f"Adding expectation: {expectation}")
        self.__expectations.append(expectation)
        return self

    def expect_distinct_column_values_greater_than(
        self,
        column_name: str,
        threshold: int,
    ):
        """
        Add an expectation to check if the number of distinct values in a column is greater than a threshold.
        Categories:
          category: Column Aggregation Expectations
          subcategory: Any Value
        :param column_name: The name of the column to check.
        :param threshold: The threshold for distinct values count (exclusive lower bound).
        :return: an instance of DataframeExpectationsSuite.
        """
        expectation = DataframeExpectationRegistry.get_expectation(
            expectation_name="ExpectationDistinctColumnValuesGreaterThan",
            column_name=column_name,
            threshold=threshold,
        )
        logger.info(f"Adding expectation: {expectation}")
        self.__expectations.append(expectation)
        return self

    def expect_distinct_column_values_between(
        self,
        column_name: str,
        min_value: int,
        max_value: int,
    ):
        """
        Add an expectation to check if the number of distinct values in a column falls within a range.
        Categories:
          category: Column Aggregation Expectations
          subcategory: Any Value
        :param column_name: The name of the column to check.
        :param min_value: The minimum number of distinct values (inclusive lower bound).
        :param max_value: The maximum number of distinct values (inclusive upper bound).
        :return: an instance of DataframeExpectationsSuite.
        """
        expectation = DataframeExpectationRegistry.get_expectation(
            expectation_name="ExpectationDistinctColumnValuesBetween",
            column_name=column_name,
            min_value=min_value,
            max_value=max_value,
        )
        logger.info(f"Adding expectation: {expectation}")
        self.__expectations.append(expectation)
        return self

    def expect_column_quantile_between(
        self,
        column_name: str,
        quantile: float,
        min_value: Union[int, float],
        max_value: Union[int, float],
    ):
        """
        Add an expectation to check if a quantile of a column falls within a specified range.

        Categories:
          category: Column Aggregation Expectations
          subcategory: Numerical

        :param column_name: The name of the column to check.
        :param quantile: The quantile to compute (0.0 to 1.0, where 0.0=min, 0.5=median, 1.0=max).
        :param min_value: The minimum allowed value for the quantile.
        :param max_value: The maximum allowed value for the quantile.
        :return: an instance of DataframeExpectationsSuite.
        """
        expectation = DataframeExpectationRegistry.get_expectation(
            expectation_name="ExpectationColumnQuantileBetween",
            column_name=column_name,
            quantile=quantile,
            min_value=min_value,
            max_value=max_value,
        )

        logger.info(f"Adding expectation: {expectation}")
        self.__expectations.append(expectation)
        return self

    def expect_column_max_between(
        self,
        column_name: str,
        min_value: Union[int, float],
        max_value: Union[int, float],
    ):
        """
        Add an expectation to check if the maximum value of a column falls within a specified range.

        Categories:
          category: Column Aggregation Expectations
          subcategory: Numerical

        :param column_name: The name of the column to check.
        :param min_value: The minimum allowed value for the column maximum.
        :param max_value: The maximum allowed value for the column maximum.
        :return: an instance of DataframeExpectationsSuite.
        """
        expectation = DataframeExpectationRegistry.get_expectation(
            expectation_name="ExpectationColumnMaxBetween",
            column_name=column_name,
            min_value=min_value,
            max_value=max_value,
        )

        logger.info(f"Adding expectation: {expectation}")
        self.__expectations.append(expectation)
        return self

    def expect_column_min_between(
        self,
        column_name: str,
        min_value: Union[int, float],
        max_value: Union[int, float],
    ):
        """
        Add an expectation to check if the minimum value of a column falls within a specified range.

        Categories:
          category: Column Aggregation Expectations
          subcategory: Numerical

        :param column_name: The name of the column to check.
        :param min_value: The minimum allowed value for the column minimum.
        :param max_value: The maximum allowed value for the column minimum.
        :return: an instance of DataframeExpectationsSuite.
        """
        expectation = DataframeExpectationRegistry.get_expectation(
            expectation_name="ExpectationColumnMinBetween",
            column_name=column_name,
            min_value=min_value,
            max_value=max_value,
        )

        logger.info(f"Adding expectation: {expectation}")
        self.__expectations.append(expectation)
        return self

    def expect_column_mean_between(
        self,
        column_name: str,
        min_value: Union[int, float],
        max_value: Union[int, float],
    ):
        """
        Add an expectation to check if the mean value of a column falls within a specified range.

        Categories:
          category: Column Aggregation Expectations
          subcategory: Numerical

        :param column_name: The name of the column to check.
        :param min_value: The minimum allowed value for the column mean.
        :param max_value: The maximum allowed value for the column mean.
        :return: an instance of DataframeExpectationsSuite.
        """
        expectation = DataframeExpectationRegistry.get_expectation(
            expectation_name="ExpectationColumnMeanBetween",
            column_name=column_name,
            min_value=min_value,
            max_value=max_value,
        )

        logger.info(f"Adding expectation: {expectation}")
        self.__expectations.append(expectation)
        return self

    def expect_column_median_between(
        self,
        column_name: str,
        min_value: Union[int, float],
        max_value: Union[int, float],
    ):
        """
        Add an expectation to check if the median value of a column falls within a specified range.

        Categories:
          category: Column Aggregation Expectations
          subcategory: Numerical

        :param column_name: The name of the column to check.
        :param min_value: The minimum allowed value for the column median.
        :param max_value: The maximum allowed value for the column median.
        :return: an instance of DataframeExpectationsSuite.
        """
        expectation = DataframeExpectationRegistry.get_expectation(
            expectation_name="ExpectationColumnMedianBetween",
            column_name=column_name,
            min_value=min_value,
            max_value=max_value,
        )

        logger.info(f"Adding expectation: {expectation}")
        self.__expectations.append(expectation)
        return self

    def run(
        self,
        data_frame: DataFrameLike,
    ) -> None:
        """
        Run all expectations on the provided DataFrame with PySpark caching optimization.

        :param data_frame: The DataFrame to validate.
        """
        from dataframe_expectations import DataFrameType
        from dataframe_expectations.expectations import DataframeExpectation

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
        data_frame_type = DataframeExpectation.infer_data_frame_type(data_frame)
        was_already_cached = False

        if data_frame_type == DataFrameType.PYSPARK:
            # Import PySpark DataFrame for type casting
            from pyspark.sql import DataFrame as PySparkDataFrame

            # Cast to PySpark DataFrame since we know it's PySpark at this point
            pyspark_df = cast(PySparkDataFrame, data_frame)

            # Check if DataFrame is already cached
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
                if isinstance(result, DataframeExpectationSuccessMessage):
                    logger.info(
                        f"{expectation.get_expectation_name()} ({expectation.get_description()}) ... OK"
                    )
                    successes.append(result)
                elif isinstance(result, DataframeExpectationFailureMessage):
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
            raise DataframeExpectationsSuiteFailure(
                total_expectations=len(self.__expectations), failures=failures
            )


if __name__ == "__main__":
    # Example usage
    suite = DataframeExpectationsSuite()
    suite.expect_value_greater_than(column_name="age", value=18)
    suite.expect_value_less_than(column_name="salary", value=100000)
    suite.expect_unique_rows(column_names=["id"])
    suite.expect_column_mean_between(column_name="age", min_value=20, max_value=40)
    suite.expect_column_max_between(column_name="salary", min_value=80000, max_value=150000)

    import pandas as pd

    # Create a sample DataFrame
    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "age": [20, 25, 30, 35],
            "salary": [50000, 120000, 80000, 90000],
        }
    )

    suite.run(data_frame=df)
