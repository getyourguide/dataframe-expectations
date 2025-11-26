from typing import List, Optional, cast

import pandas as pd
from pandas import DataFrame as PandasDataFrame
from pyspark.sql import DataFrame as PySparkDataFrame
from pyspark.sql import functions as F
from dataframe_expectations.core.aggregation_expectation import (
    DataFrameAggregationExpectation,
)
from dataframe_expectations.core.types import (
    ExpectationCategory,
    ExpectationSubcategory,
    DataFrameLike,
    DataFrameType,
)
from dataframe_expectations.registry import register_expectation
from dataframe_expectations.core.utils import requires_params
from dataframe_expectations.result_message import (
    DataFrameExpectationFailureMessage,
    DataFrameExpectationResultMessage,
    DataFrameExpectationSuccessMessage,
)


class ExpectationUniqueRows(DataFrameAggregationExpectation):
    """
    Expectation that checks if there are no duplicate rows for the given column names.

    If columns list is empty, checks for duplicates across all columns.

    Example::

        For column_names ["col1", "col2"], given a DataFrame with columns col1, col2, col3:

        - If all rows have unique combinations of col1 and col2 values, there are no violations
        - If any rows have identical col1 AND col2 values, those are violations

        Example passing case (all unique)::

            col1=1, col2=10
            col1=2, col2=20
            col1=3, col2=30
            col1=1, col2=20  # Different col2, so unique

        Example failing case (duplicate found)::

            col1=1, col2=10
            col1=2, col2=20
            col1=3, col2=30
            col1=1, col2=10  # Duplicate of first row
    """

    def __init__(self, column_names: List[str], tags: Optional[List[str]] = None):
        """
        Initialize the unique expectation.

        :param column_names: List of column names to check for uniqueness.
                       If empty, checks all column_names.
        """
        description = (
            f"all rows unique for columns {column_names}"
            if column_names
            else "all rows unique across all columns"
        )

        self.column_names = column_names

        super().__init__(
            expectation_name="ExpectationUniqueRows",
            column_names=column_names,
            description=description,
            tags=tags,
        )

    def aggregate_and_validate_pandas(
        self, data_frame: DataFrameLike, **kwargs
    ) -> DataFrameExpectationResultMessage:
        """
        Validate uniqueness in a pandas DataFrame.
        """
        # Cast to PandasDataFrame for type safety
        pandas_df = cast(PandasDataFrame, data_frame)

        # If columns list is empty, use all columns
        check_columns = self.column_names if self.column_names else list(pandas_df.columns)

        # Find duplicates - dropna=False ensures null values are considered in duplicate detection
        # This means rows with null values can be duplicates of each other
        duplicates = pandas_df[pandas_df.duplicated(subset=check_columns, keep=False)]

        if len(duplicates) == 0:
            return DataFrameExpectationSuccessMessage(expectation_name=self.get_expectation_name())

        # Add duplicate count column and keep only one row per duplicate group
        duplicate_counts = (
            pandas_df.groupby(check_columns, dropna=False).size().reset_index(name="#duplicates")
        )
        # Filter to only keep groups with duplicates (count > 1)
        duplicate_counts = duplicate_counts[duplicate_counts["#duplicates"] > 1]

        # Order by #duplicates, then by the specified columns
        sort_columns = ["#duplicates"] + check_columns
        duplicates_with_counts = duplicate_counts.sort_values(sort_columns)

        # Replace NaN with None (use map for pandas >= 2.1.0, applymap for older versions)
        if hasattr(duplicates_with_counts, "map") and callable(
            getattr(duplicates_with_counts, "map")
        ):
            # pandas >= 2.1.0 uses .map() for element-wise operations on DataFrames
            duplicates_with_counts = duplicates_with_counts.map(lambda x: None if pd.isna(x) else x)
        else:
            # pandas < 2.1.0 uses .applymap() (deprecated in 2.1.0, removed in 3.0.0)
            duplicates_with_counts = duplicates_with_counts.applymap(  # type: ignore[operator]
                lambda x: None if pd.isna(x) else x
            )

        # Calculate total number of duplicate rows (not groups)
        total_duplicate_rows = duplicates_with_counts["#duplicates"].sum()  # type: ignore[misc]

        # Generate dynamic error message
        error_msg = (
            f"duplicate rows found for columns {self.column_names}"
            if self.column_names
            else "duplicate rows found"
        )

        return DataFrameExpectationFailureMessage(
            expectation_str=str(self),
            data_frame_type=DataFrameType.PANDAS,
            violations_data_frame=duplicates_with_counts,
            message=f"Found {total_duplicate_rows} duplicate row(s). {error_msg}",
        )

    def aggregate_and_validate_pyspark(
        self, data_frame: DataFrameLike, **kwargs
    ) -> DataFrameExpectationResultMessage:
        """
        Validate uniqueness in a PySpark DataFrame.
        """
        # Cast to PySparkDataFrame for type safety
        pyspark_df = cast(PySparkDataFrame, data_frame)

        # If columns list is empty, use all columns
        check_columns = self.column_names if self.column_names else pyspark_df.columns

        # Group by the specified columns and count duplicates
        duplicates_df = (
            pyspark_df.groupBy(*check_columns)
            .count()
            .filter(F.col("count") > 1)
            .withColumnRenamed("count", "#duplicates")
            .orderBy(F.col("#duplicates"), *check_columns)
        )

        duplicate_count = duplicates_df.count()

        if duplicate_count == 0:
            return DataFrameExpectationSuccessMessage(expectation_name=self.get_expectation_name())

        # Calculate total number of duplicate rows (not groups)
        total_duplicate_rows = duplicates_df.agg(F.sum("#duplicates")).collect()[0][0]

        # Generate dynamic error message
        error_msg = (
            f"duplicate rows found for columns {self.column_names}"
            if self.column_names
            else "duplicate rows found"
        )

        return DataFrameExpectationFailureMessage(
            expectation_str=str(self),
            data_frame_type=DataFrameType.PYSPARK,
            violations_data_frame=duplicates_df,
            message=f"Found {total_duplicate_rows} duplicate row(s). {error_msg}",
        )


class ExpectationDistinctColumnValuesEquals(DataFrameAggregationExpectation):
    """
    Expectation that validates a column has exactly a specified number of distinct values.

    This expectation counts the number of unique/distinct values in a specified column
    and checks if it equals the expected count.

    Examples:
        Column with values [1, 2, 3, 2, 1] has 3 distinct values:
        - ExpectationDistinctColumnValuesEquals(column_name="col1", expected_value=3) → PASS
        - ExpectationDistinctColumnValuesEquals(column_name="col1", expected_value=5) → FAIL

    Note: The comparison is exact equality (inclusive).
    """

    def __init__(self, column_name: str, expected_value: int, tags: Optional[List[str]] = None):
        """
        Initialize the distinct values equals expectation.

        :param column_name: Name of the column to check.
        :param expected_value: Expected number of distinct values (exact match).
        """
        if expected_value < 0:
            raise ValueError(f"expected_value must be non-negative, got {expected_value}")

        description = f"column '{column_name}' has exactly {expected_value} distinct values"

        self.column_name = column_name
        self.expected_value = expected_value

        super().__init__(
            expectation_name="ExpectationDistinctColumnValuesEquals",
            column_names=[column_name],
            description=description,
            tags=tags,
        )

    def aggregate_and_validate_pandas(
        self, data_frame: DataFrameLike, **kwargs
    ) -> DataFrameExpectationResultMessage:
        """Validate distinct values count in a pandas DataFrame."""
        try:
            # Cast to PandasDataFrame for type safety
            pandas_df = cast(PandasDataFrame, data_frame)
            # Count distinct values (dropna=False includes NaN as a distinct value)
            actual_count = pandas_df[self.column_name].nunique(dropna=False)

            if actual_count == self.expected_value:
                return DataFrameExpectationSuccessMessage(
                    expectation_name=self.get_expectation_name()
                )
            else:
                return DataFrameExpectationFailureMessage(
                    expectation_str=str(self),
                    data_frame_type=DataFrameType.PANDAS,
                    message=f"Column '{self.column_name}' has {actual_count} distinct values, expected exactly {self.expected_value}.",
                )

        except Exception as e:
            return DataFrameExpectationFailureMessage(
                expectation_str=str(self),
                data_frame_type=DataFrameType.PANDAS,
                message=f"Error counting distinct values: {str(e)}",
            )

    def aggregate_and_validate_pyspark(
        self, data_frame: DataFrameLike, **kwargs
    ) -> DataFrameExpectationResultMessage:
        """Validate distinct values count in a PySpark DataFrame."""
        try:
            # Cast to PySparkDataFrame for type safety
            pyspark_df = cast(PySparkDataFrame, data_frame)
            # Count distinct values including nulls
            actual_count = pyspark_df.select(self.column_name).distinct().count()

            if actual_count == self.expected_value:
                return DataFrameExpectationSuccessMessage(
                    expectation_name=self.get_expectation_name()
                )
            else:
                return DataFrameExpectationFailureMessage(
                    expectation_str=str(self),
                    data_frame_type=DataFrameType.PYSPARK,
                    message=f"Column '{self.column_name}' has {actual_count} distinct values, expected exactly {self.expected_value}.",
                )

        except Exception as e:
            return DataFrameExpectationFailureMessage(
                expectation_str=str(self),
                data_frame_type=DataFrameType.PYSPARK,
                message=f"Error counting distinct values: {str(e)}",
            )


class ExpectationDistinctColumnValuesLessThan(DataFrameAggregationExpectation):
    """
    Expectation that validates a column has fewer than a specified number of distinct values.

    This expectation counts the number of unique/distinct values in a specified column
    and checks if it's less than the specified threshold.

    Examples:
        Column with values [1, 2, 3, 2, 1] has 3 distinct values:
        - ExpectationDistinctColumnValuesLessThan(column_name="col1", threshold=5) → PASS (3 < 5)
        - ExpectationDistinctColumnValuesLessThan(column_name="col1", threshold=3) → FAIL (3 is not < 3)

    Note: The threshold is exclusive (actual_count < threshold).
    """

    def __init__(self, column_name: str, threshold: int, tags: Optional[List[str]] = None):
        """
        Initialize the distinct values less than expectation.

        :param column_name: Name of the column to check.
        :param threshold: Threshold for distinct values count (exclusive upper bound).
        """
        if threshold < 0:
            raise ValueError(f"threshold must be non-negative, got {threshold}")

        description = f"column '{column_name}' has fewer than {threshold} distinct values"

        self.column_name = column_name
        self.threshold = threshold

        super().__init__(
            expectation_name="ExpectationDistinctColumnValuesLessThan",
            column_names=[column_name],
            description=description,
            tags=tags,
        )

    def aggregate_and_validate_pandas(
        self, data_frame: DataFrameLike, **kwargs
    ) -> DataFrameExpectationResultMessage:
        """Validate distinct values count in a pandas DataFrame."""
        try:
            # Cast to PandasDataFrame for type safety
            pandas_df = cast(PandasDataFrame, data_frame)
            # Count distinct values (dropna=False includes NaN as a distinct value)
            actual_count = pandas_df[self.column_name].nunique(dropna=False)

            if actual_count < self.threshold:
                return DataFrameExpectationSuccessMessage(
                    expectation_name=self.get_expectation_name()
                )
            else:
                return DataFrameExpectationFailureMessage(
                    expectation_str=str(self),
                    data_frame_type=DataFrameType.PANDAS,
                    message=f"Column '{self.column_name}' has {actual_count} distinct values, expected fewer than {self.threshold}.",
                )

        except Exception as e:
            return DataFrameExpectationFailureMessage(
                expectation_str=str(self),
                data_frame_type=DataFrameType.PANDAS,
                message=f"Error counting distinct values: {str(e)}",
            )

    def aggregate_and_validate_pyspark(
        self, data_frame: DataFrameLike, **kwargs
    ) -> DataFrameExpectationResultMessage:
        """Validate distinct values count in a PySpark DataFrame."""
        try:
            # Cast to PySparkDataFrame for type safety
            pyspark_df = cast(PySparkDataFrame, data_frame)
            # Count distinct values including nulls
            actual_count = pyspark_df.select(self.column_name).distinct().count()

            if actual_count < self.threshold:
                return DataFrameExpectationSuccessMessage(
                    expectation_name=self.get_expectation_name()
                )
            else:
                return DataFrameExpectationFailureMessage(
                    expectation_str=str(self),
                    data_frame_type=DataFrameType.PYSPARK,
                    message=f"Column '{self.column_name}' has {actual_count} distinct values, expected fewer than {self.threshold}.",
                )

        except Exception as e:
            return DataFrameExpectationFailureMessage(
                expectation_str=str(self),
                data_frame_type=DataFrameType.PYSPARK,
                message=f"Error counting distinct values: {str(e)}",
            )


class ExpectationDistinctColumnValuesGreaterThan(DataFrameAggregationExpectation):
    """
    Expectation that validates a column has more than a specified number of distinct values.

    This expectation counts the number of unique/distinct values in a specified column
    and checks if it's greater than the specified threshold.

    Examples:
        Column with values [1, 2, 3, 2, 1] has 3 distinct values:
        - ExpectationDistinctColumnValuesGreaterThan(column_name="col1", threshold=2) → PASS (3 > 2)
        - ExpectationDistinctColumnValuesGreaterThan(column_name="col1", threshold=3) → FAIL (3 is not > 3)

    Note: The threshold is exclusive (actual_count > threshold).
    """

    def __init__(self, column_name: str, threshold: int, tags: Optional[List[str]] = None):
        """
        Initialize the distinct values greater than expectation.

        :param column_name: Name of the column to check.
        :param threshold: Threshold for distinct values count (exclusive lower bound).
        """
        if threshold < 0:
            raise ValueError(f"threshold must be non-negative, got {threshold}")

        description = f"column '{column_name}' has more than {threshold} distinct values"

        self.column_name = column_name
        self.threshold = threshold

        super().__init__(
            expectation_name="ExpectationDistinctColumnValuesGreaterThan",
            column_names=[column_name],
            description=description,
            tags=tags,
        )

    def aggregate_and_validate_pandas(
        self, data_frame: DataFrameLike, **kwargs
    ) -> DataFrameExpectationResultMessage:
        """Validate distinct values count in a pandas DataFrame."""
        try:
            # Cast to PandasDataFrame for type safety
            pandas_df = cast(PandasDataFrame, data_frame)
            # Count distinct values (dropna=False includes NaN as a distinct value)
            actual_count = pandas_df[self.column_name].nunique(dropna=False)

            if actual_count > self.threshold:
                return DataFrameExpectationSuccessMessage(
                    expectation_name=self.get_expectation_name()
                )
            else:
                return DataFrameExpectationFailureMessage(
                    expectation_str=str(self),
                    data_frame_type=DataFrameType.PANDAS,
                    message=f"Column '{self.column_name}' has {actual_count} distinct values, expected more than {self.threshold}.",
                )

        except Exception as e:
            return DataFrameExpectationFailureMessage(
                expectation_str=str(self),
                data_frame_type=DataFrameType.PANDAS,
                message=f"Error counting distinct values: {str(e)}",
            )

    def aggregate_and_validate_pyspark(
        self, data_frame: DataFrameLike, **kwargs
    ) -> DataFrameExpectationResultMessage:
        """Validate distinct values count in a PySpark DataFrame."""
        try:
            # Cast to PySparkDataFrame for type safety
            pyspark_df = cast(PySparkDataFrame, data_frame)
            # Count distinct values including nulls
            actual_count = pyspark_df.select(self.column_name).distinct().count()

            if actual_count > self.threshold:
                return DataFrameExpectationSuccessMessage(
                    expectation_name=self.get_expectation_name()
                )
            else:
                return DataFrameExpectationFailureMessage(
                    expectation_str=str(self),
                    data_frame_type=DataFrameType.PYSPARK,
                    message=f"Column '{self.column_name}' has {actual_count} distinct values, expected more than {self.threshold}.",
                )

        except Exception as e:
            return DataFrameExpectationFailureMessage(
                expectation_str=str(self),
                data_frame_type=DataFrameType.PYSPARK,
                message=f"Error counting distinct values: {str(e)}",
            )


class ExpectationDistinctColumnValuesBetween(DataFrameAggregationExpectation):
    """
    Expectation that validates a column has a number of distinct values within a specified range.

    This expectation counts the number of unique/distinct values in a specified column
    and checks if it's between the specified minimum and maximum values.

    Examples:
        Column with values [1, 2, 3, 2, 1] has 3 distinct values:
        - ExpectationDistinctColumnValuesBetween(column_name="col1", min_value=2, max_value=5) → PASS (2 ≤ 3 ≤ 5)
        - ExpectationDistinctColumnValuesBetween(column_name="col1", min_value=4, max_value=6) → FAIL (3 is not ≥ 4)

    Note: Both bounds are inclusive (min_value ≤ actual_count ≤ max_value).
    """

    def __init__(
        self, column_name: str, min_value: int, max_value: int, tags: Optional[List[str]] = None
    ):
        """
        Initialize the distinct values between expectation.

        :param column_name: Name of the column to check.
        :param min_value: Minimum number of distinct values (inclusive lower bound).
        :param max_value: Maximum number of distinct values (inclusive upper bound).
        """
        if min_value < 0:
            raise ValueError(f"min_value must be non-negative, got {min_value}")
        if max_value < 0:
            raise ValueError(f"max_value must be non-negative, got {max_value}")
        if min_value > max_value:
            raise ValueError(f"min_value ({min_value}) must be <= max_value ({max_value})")

        description = (
            f"column '{column_name}' has between {min_value} and {max_value} distinct values"
        )

        self.column_name = column_name
        self.min_value = min_value
        self.max_value = max_value

        super().__init__(
            expectation_name="ExpectationDistinctColumnValuesBetween",
            column_names=[column_name],
            description=description,
            tags=tags,
        )

    def aggregate_and_validate_pandas(
        self, data_frame: DataFrameLike, **kwargs
    ) -> DataFrameExpectationResultMessage:
        """Validate distinct values count in a pandas DataFrame."""
        try:
            # Cast to PandasDataFrame for type safety
            pandas_df = cast(PandasDataFrame, data_frame)
            # Count distinct values (dropna=False includes NaN as a distinct value)
            actual_count = pandas_df[self.column_name].nunique(dropna=False)

            if self.min_value <= actual_count <= self.max_value:
                return DataFrameExpectationSuccessMessage(
                    expectation_name=self.get_expectation_name()
                )
            else:
                return DataFrameExpectationFailureMessage(
                    expectation_str=str(self),
                    data_frame_type=DataFrameType.PANDAS,
                    message=f"Column '{self.column_name}' has {actual_count} distinct values, expected between {self.min_value} and {self.max_value}.",
                )

        except Exception as e:
            return DataFrameExpectationFailureMessage(
                expectation_str=str(self),
                data_frame_type=DataFrameType.PANDAS,
                message=f"Error counting distinct values: {str(e)}",
            )

    def aggregate_and_validate_pyspark(
        self, data_frame: DataFrameLike, **kwargs
    ) -> DataFrameExpectationResultMessage:
        """Validate distinct values count in a PySpark DataFrame."""
        try:
            # Cast to PySparkDataFrame for type safety
            pyspark_df = cast(PySparkDataFrame, data_frame)
            # Count distinct values including nulls
            actual_count = pyspark_df.select(self.column_name).distinct().count()

            if self.min_value <= actual_count <= self.max_value:
                return DataFrameExpectationSuccessMessage(
                    expectation_name=self.get_expectation_name()
                )
            else:
                return DataFrameExpectationFailureMessage(
                    expectation_str=str(self),
                    data_frame_type=DataFrameType.PYSPARK,
                    message=f"Column '{self.column_name}' has {actual_count} distinct values, expected between {self.min_value} and {self.max_value}.",
                )

        except Exception as e:
            return DataFrameExpectationFailureMessage(
                expectation_str=str(self),
                data_frame_type=DataFrameType.PYSPARK,
                message=f"Error counting distinct values: {str(e)}",
            )


# Register the expectations
@register_expectation(
    "ExpectationUniqueRows",
    pydoc="Check if all rows in the DataFrame are unique based on specified columns",
    category=ExpectationCategory.DATAFRAME_AGGREGATION_EXPECTATIONS,
    subcategory=ExpectationSubcategory.UNIQUE,
    params_doc={
        "column_names": "List of column names to check for uniqueness. Empty list checks all columns",
    },
)
@requires_params("column_names", types={"column_names": list})
def create_expectation_unique(
    column_names: List[str],
    tags: Optional[List[str]] = None,
) -> ExpectationUniqueRows:
    """
    Create an ExpectationUniqueRows instance.

    :param column_names: List of column names to check for uniqueness. If empty, checks all columns.
    :param tags: Optional list of tags for filtering expectations.
    :return: ExpectationUniqueRows instance
    """
    column_names = column_names
    return ExpectationUniqueRows(column_names=column_names, tags=tags)


@register_expectation(
    "ExpectationDistinctColumnValuesEquals",
    pydoc="Check if a column has exactly a specified number of distinct values",
    category=ExpectationCategory.COLUMN_AGGREGATION_EXPECTATIONS,
    subcategory=ExpectationSubcategory.UNIQUE,
    params_doc={
        "column_name": "The name of the column to check for distinct values",
        "expected_value": "The expected number of distinct values",
    },
)
@requires_params(
    "column_name",
    "expected_value",
    types={"column_name": str, "expected_value": int},
)
def create_expectation_distinct_column_values_equals(
    column_name: str,
    expected_value: int,
    tags: Optional[List[str]] = None,
) -> ExpectationDistinctColumnValuesEquals:
    """
    Create an ExpectationDistinctColumnValuesEquals instance.

    :param column_name: Name of the column to check.
    :param expected_value: Expected number of distinct values.
    :param tags: Optional list of tags for filtering expectations.
    :return: A configured expectation instance.
    """
    return ExpectationDistinctColumnValuesEquals(
        column_name=column_name,
        expected_value=expected_value,
        tags=tags,
    )


@register_expectation(
    "ExpectationDistinctColumnValuesLessThan",
    pydoc="Check if a column has at most a specified number of distinct values",
    category=ExpectationCategory.COLUMN_AGGREGATION_EXPECTATIONS,
    subcategory=ExpectationSubcategory.UNIQUE,
    params_doc={
        "column_name": "The name of the column to check for distinct values",
        "threshold": "The maximum number of distinct values (exclusive)",
    },
)
@requires_params(
    "column_name",
    "threshold",
    types={"column_name": str, "threshold": int},
)
def create_expectation_distinct_column_values_less_than(
    column_name: str,
    threshold: int,
    tags: Optional[List[str]] = None,
) -> ExpectationDistinctColumnValuesLessThan:
    """
    Create an ExpectationDistinctColumnValuesLessThan instance.

    :param column_name: Name of the column to check.
    :param threshold: Threshold for distinct values count (exclusive upper bound).
    :param tags: Optional list of tags for filtering expectations.
    :return: A configured expectation instance.
    """
    return ExpectationDistinctColumnValuesLessThan(
        column_name=column_name,
        threshold=threshold,
        tags=tags,
    )


@register_expectation(
    "ExpectationDistinctColumnValuesGreaterThan",
    pydoc="Check if a column has at least a specified number of distinct values",
    category=ExpectationCategory.COLUMN_AGGREGATION_EXPECTATIONS,
    subcategory=ExpectationSubcategory.UNIQUE,
    params_doc={
        "column_name": "The name of the column to check for distinct values",
        "threshold": "The minimum number of distinct values (exclusive)",
    },
)
@requires_params(
    "column_name",
    "threshold",
    types={"column_name": str, "threshold": int},
)
def create_expectation_distinct_column_values_greater_than(
    column_name: str,
    threshold: int,
    tags: Optional[List[str]] = None,
) -> ExpectationDistinctColumnValuesGreaterThan:
    """
    Create an ExpectationDistinctColumnValuesGreaterThan instance.

    :param column_name: Name of the column to check.
    :param threshold: Threshold for distinct values count (exclusive lower bound).
    :param tags: Optional list of tags for filtering expectations.
    :return: A configured expectation instance.
    """
    return ExpectationDistinctColumnValuesGreaterThan(
        column_name=column_name,
        threshold=threshold,
        tags=tags,
    )


@register_expectation(
    "ExpectationDistinctColumnValuesBetween",
    pydoc="Check if a column has a number of distinct values within a specified range",
    category=ExpectationCategory.COLUMN_AGGREGATION_EXPECTATIONS,
    subcategory=ExpectationSubcategory.UNIQUE,
    params_doc={
        "column_name": "The name of the column to check for distinct values",
        "min_value": "The minimum number of distinct values (inclusive)",
        "max_value": "The maximum number of distinct values (inclusive)",
    },
)
@requires_params(
    "column_name",
    "min_value",
    "max_value",
    types={"column_name": str, "min_value": int, "max_value": int},
)
def create_expectation_distinct_column_values_between(
    column_name: str,
    min_value: int,
    max_value: int,
    tags: Optional[List[str]] = None,
) -> ExpectationDistinctColumnValuesBetween:
    """
    Create an ExpectationDistinctColumnValuesBetween instance.

    :param column_name: Name of the column to check.
    :param min_value: Minimum number of distinct values (inclusive lower bound).
    :param max_value: Maximum number of distinct values (inclusive upper bound).
    :param tags: Optional key-value tags for this expectation.
    :return: A configured expectation instance.
    """
    return ExpectationDistinctColumnValuesBetween(
        column_name=column_name,
        min_value=min_value,
        max_value=max_value,
        tags=tags,
    )
