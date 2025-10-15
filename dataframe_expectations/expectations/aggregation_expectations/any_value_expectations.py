from typing import Union

from dataframe_expectations import DataFrameLike, DataFrameType
from dataframe_expectations.expectations.aggregation_expectation import (
    DataframeAggregationExpectation,
)
from dataframe_expectations.expectations.expectation_registry import (
    register_expectation,
)
from dataframe_expectations.expectations.utils import requires_params
from dataframe_expectations.result_message import (
    DataframeExpectationFailureMessage,
    DataframeExpectationResultMessage,
    DataframeExpectationSuccessMessage,
)


class ExpectationMinRows(DataframeAggregationExpectation):
    """
    Expectation that validates a DataFrame has at least a minimum number of rows.

    This expectation counts the total number of rows in the DataFrame and checks if it
    meets or exceeds the specified minimum threshold.

    Examples:
        DataFrame with 100 rows:
        - ExpectationMinRows(min_rows=50) → PASS
        - ExpectationMinRows(min_rows=150) → FAIL
    """

    def __init__(self, min_rows: int):
        """
        Initialize the minimum rows expectation.

        Args:
            min_rows (int): Minimum number of rows required (inclusive).
        """
        if min_rows < 0:
            raise ValueError(f"min_rows must be non-negative, got {min_rows}")

        description = f"DataFrame contains at least {min_rows} rows"

        self.min_rows = min_rows

        super().__init__(
            expectation_name="ExpectationMinRows",
            column_names=[],  # No specific columns required
            description=description,
        )

    def aggregate_and_validate_pandas(
        self, data_frame: DataFrameLike, **kwargs
    ) -> DataframeExpectationResultMessage:
        """Validate minimum rows in a pandas DataFrame."""
        try:
            row_count = len(data_frame)

            if row_count >= self.min_rows:
                return DataframeExpectationSuccessMessage(
                    expectation_name=self.get_expectation_name()
                )
            else:
                return DataframeExpectationFailureMessage(
                    expectation_str=str(self),
                    data_frame_type=DataFrameType.PANDAS,
                    message=f"DataFrame has {row_count} rows, expected at least {self.min_rows}.",
                )

        except Exception as e:
            return DataframeExpectationFailureMessage(
                expectation_str=str(self),
                data_frame_type=DataFrameType.PANDAS,
                message=f"Error counting DataFrame rows: {str(e)}",
            )

    def aggregate_and_validate_pyspark(
        self, data_frame: DataFrameLike, **kwargs
    ) -> DataframeExpectationResultMessage:
        """Validate minimum rows in a PySpark DataFrame."""
        try:
            row_count = data_frame.count()

            if row_count >= self.min_rows:
                return DataframeExpectationSuccessMessage(
                    expectation_name=self.get_expectation_name()
                )
            else:
                return DataframeExpectationFailureMessage(
                    expectation_str=str(self),
                    data_frame_type=DataFrameType.PYSPARK,
                    message=f"DataFrame has {row_count} rows, expected at least {self.min_rows}.",
                )

        except Exception as e:
            return DataframeExpectationFailureMessage(
                expectation_str=str(self),
                data_frame_type=DataFrameType.PYSPARK,
                message=f"Error counting DataFrame rows: {str(e)}",
            )


class ExpectationMaxRows(DataframeAggregationExpectation):
    """
    Expectation that validates a DataFrame has at most a maximum number of rows.

    This expectation counts the total number of rows in the DataFrame and checks if it
    does not exceed the specified maximum threshold.

    Examples:
        DataFrame with 100 rows:
        - ExpectationMaxRows(max_rows=150) → PASS
        - ExpectationMaxRows(max_rows=50) → FAIL
    """

    def __init__(self, max_rows: int):
        """
        Initialize the maximum rows expectation.

        Args:
            max_rows (int): Maximum number of rows allowed (inclusive).
        """
        if max_rows < 0:
            raise ValueError(f"max_rows must be non-negative, got {max_rows}")

        description = f"DataFrame contains at most {max_rows} rows"

        self.max_rows = max_rows

        super().__init__(
            expectation_name="ExpectationMaxRows",
            column_names=[],  # No specific columns required
            description=description,
        )

    def aggregate_and_validate_pandas(
        self, data_frame: DataFrameLike, **kwargs
    ) -> DataframeExpectationResultMessage:
        """Validate maximum rows in a pandas DataFrame."""
        try:
            row_count = len(data_frame)

            if row_count <= self.max_rows:
                return DataframeExpectationSuccessMessage(
                    expectation_name=self.get_expectation_name()
                )
            else:
                return DataframeExpectationFailureMessage(
                    expectation_str=str(self),
                    data_frame_type=DataFrameType.PANDAS,
                    message=f"DataFrame has {row_count} rows, expected at most {self.max_rows}.",
                )

        except Exception as e:
            return DataframeExpectationFailureMessage(
                expectation_str=str(self),
                data_frame_type=DataFrameType.PANDAS,
                message=f"Error counting DataFrame rows: {str(e)}",
            )

    def aggregate_and_validate_pyspark(
        self, data_frame: DataFrameLike, **kwargs
    ) -> DataframeExpectationResultMessage:
        """Validate maximum rows in a PySpark DataFrame."""
        try:
            row_count = data_frame.count()

            if row_count <= self.max_rows:
                return DataframeExpectationSuccessMessage(
                    expectation_name=self.get_expectation_name()
                )
            else:
                return DataframeExpectationFailureMessage(
                    expectation_str=str(self),
                    data_frame_type=DataFrameType.PYSPARK,
                    message=f"DataFrame has {row_count} rows, expected at most {self.max_rows}.",
                )

        except Exception as e:
            return DataframeExpectationFailureMessage(
                expectation_str=str(self),
                data_frame_type=DataFrameType.PYSPARK,
                message=f"Error counting DataFrame rows: {str(e)}",
            )


class ExpectationMaxNullPercentage(DataframeAggregationExpectation):
    """
    Expectation that validates the percentage of null/NaN values in a specific column
    is below a specified threshold.

    This expectation counts null values (including NaN for pandas) in the specified column
    and calculates the percentage relative to total rows, then checks if it's below the
    specified maximum threshold.

    Examples:
        Column with 100 rows and 5 null values (5% null):
        - ExpectationMaxNullPercentage(column_name="age", max_percentage=10.0) → PASS
        - ExpectationMaxNullPercentage(column_name="age", max_percentage=3.0) → FAIL

    Note: The percentage is expressed as a value between 0.0 and 100.0 (e.g., 5.5 for 5.5%).
    """

    def __init__(self, column_name: str, max_percentage: float):
        """
        Initialize the maximum null percentage expectation.

        Args:
            column_name (str): Name of the column to check for null percentage.
            max_percentage (float): Maximum percentage of null values allowed (0.0-100.0).
        """
        if not 0 <= max_percentage <= 100:
            raise ValueError(
                f"max_percentage must be between 0.0 and 100.0, got {max_percentage}"
            )

        description = (
            f"column '{column_name}' null percentage is at most {max_percentage}%"
        )

        self.column_name = column_name
        self.max_percentage = max_percentage

        super().__init__(
            expectation_name="ExpectationMaxNullPercentage",
            column_names=[column_name],  # Specify the required column
            description=description,
        )

    def aggregate_and_validate_pandas(
        self, data_frame: DataFrameLike, **kwargs
    ) -> DataframeExpectationResultMessage:
        """Validate maximum null percentage in a pandas DataFrame column."""
        try:
            # Get total number of rows
            total_rows = len(data_frame)

            if total_rows == 0:
                # Empty DataFrame has 0% null values
                actual_percentage = 0.0
            else:
                # Count null and NaN values in the specific column using isnull() which handles both
                null_count = data_frame[self.column_name].isnull().sum()
                actual_percentage = (null_count / total_rows) * 100

            if actual_percentage <= self.max_percentage:
                return DataframeExpectationSuccessMessage(
                    expectation_name=self.get_expectation_name()
                )
            else:
                return DataframeExpectationFailureMessage(
                    expectation_str=str(self),
                    data_frame_type=DataFrameType.PANDAS,
                    message=f"Column '{self.column_name}' has {actual_percentage:.2f}% null values, expected at most {self.max_percentage:.2f}%.",
                )

        except Exception as e:
            return DataframeExpectationFailureMessage(
                expectation_str=str(self),
                data_frame_type=DataFrameType.PANDAS,
                message=f"Error calculating null percentage for column '{self.column_name}': {str(e)}",
            )

    def aggregate_and_validate_pyspark(
        self, data_frame: DataFrameLike, **kwargs
    ) -> DataframeExpectationResultMessage:
        """Validate maximum null percentage in a PySpark DataFrame column."""
        try:
            from pyspark.sql import functions as F

            # Get total number of rows
            total_rows = data_frame.count()

            if total_rows == 0:
                # Empty DataFrame has 0% null values
                actual_percentage = 0.0
            else:
                # Count null values in the specific column
                null_count_result = data_frame.agg(
                    F.sum(
                        F.when(F.col(self.column_name).isNull(), 1).otherwise(0)
                    ).alias("null_count")
                ).collect()

                null_count = null_count_result[0]["null_count"]
                actual_percentage = (null_count / total_rows) * 100

            if actual_percentage <= self.max_percentage:
                return DataframeExpectationSuccessMessage(
                    expectation_name=self.get_expectation_name()
                )
            else:
                return DataframeExpectationFailureMessage(
                    expectation_str=str(self),
                    data_frame_type=DataFrameType.PYSPARK,
                    message=f"Column '{self.column_name}' has {actual_percentage:.2f}% null values, expected at most {self.max_percentage:.2f}%.",
                )

        except Exception as e:
            return DataframeExpectationFailureMessage(
                expectation_str=str(self),
                data_frame_type=DataFrameType.PYSPARK,
                message=f"Error calculating null percentage for column '{self.column_name}': {str(e)}",
            )


class ExpectationMaxNullCount(DataframeAggregationExpectation):
    """
    Expectation that validates the absolute count of null/NaN values in a specific column
    is below a specified threshold.

    This expectation counts null values (including NaN for pandas) in the specified column
    and checks if the absolute count is below the specified maximum threshold.

    Examples:
        Column with 100 rows and 5 null values:
        - ExpectationMaxNullCount(column_name="age", max_count=10) → PASS
        - ExpectationMaxNullCount(column_name="age", max_count=3) → FAIL

    Note: The count is the absolute number of null values, not a percentage.
    """

    def __init__(self, column_name: str, max_count: int):
        """
        Initialize the maximum null count expectation.

        Args:
            column_name (str): Name of the column to check for null count.
            max_count (int): Maximum number of null values allowed.
        """
        if max_count < 0:
            raise ValueError(f"max_count must be non-negative, got {max_count}")

        description = f"column '{column_name}' has at most {max_count} null values"

        self.column_name = column_name
        self.max_count = max_count

        super().__init__(
            expectation_name="ExpectationMaxNullCount",
            column_names=[column_name],  # Specify the required column
            description=description,
        )

    def aggregate_and_validate_pandas(
        self, data_frame: DataFrameLike, **kwargs
    ) -> DataframeExpectationResultMessage:
        """Validate maximum null count in a pandas DataFrame column."""
        try:
            # Count null and NaN values in the specific column using isnull() which handles both
            null_count = data_frame[self.column_name].isnull().sum()

            if null_count <= self.max_count:
                return DataframeExpectationSuccessMessage(
                    expectation_name=self.get_expectation_name()
                )
            else:
                return DataframeExpectationFailureMessage(
                    expectation_str=str(self),
                    data_frame_type=DataFrameType.PANDAS,
                    message=f"Column '{self.column_name}' has {null_count} null values, expected at most {self.max_count}.",
                )

        except Exception as e:
            return DataframeExpectationFailureMessage(
                expectation_str=str(self),
                data_frame_type=DataFrameType.PANDAS,
                message=f"Error calculating null count for column '{self.column_name}': {str(e)}",
            )

    def aggregate_and_validate_pyspark(
        self, data_frame: DataFrameLike, **kwargs
    ) -> DataframeExpectationResultMessage:
        """Validate maximum null count in a PySpark DataFrame column."""
        try:
            from pyspark.sql import functions as F

            # Count null values in the specific column
            null_count_result = data_frame.agg(
                F.sum(F.when(F.col(self.column_name).isNull(), 1).otherwise(0)).alias(
                    "null_count"
                )
            ).collect()

            # Handle the case where null_count might be None (e.g., empty DataFrame)
            null_count = null_count_result[0]["null_count"]
            if null_count is None:
                null_count = 0

            if null_count <= self.max_count:
                return DataframeExpectationSuccessMessage(
                    expectation_name=self.get_expectation_name()
                )
            else:
                return DataframeExpectationFailureMessage(
                    expectation_str=str(self),
                    data_frame_type=DataFrameType.PYSPARK,
                    message=f"Column '{self.column_name}' has {null_count} null values, expected at most {self.max_count}.",
                )

        except Exception as e:
            return DataframeExpectationFailureMessage(
                expectation_str=str(self),
                data_frame_type=DataFrameType.PYSPARK,
                message=f"Error calculating null count for column '{self.column_name}': {str(e)}",
            )


# Factory functions for the registry
@register_expectation("ExpectationMinRows")
@requires_params("min_rows", types={"min_rows": int})
def create_expectation_min_rows(**kwargs) -> ExpectationMinRows:
    """
    Create an ExpectMinRows instance.

    Args:
        min_rows (int): Minimum number of rows required.

    Returns:
        ExpectationMinRows: A configured expectation instance.
    """
    return ExpectationMinRows(min_rows=kwargs["min_rows"])


@register_expectation("ExpectationMaxRows")
@requires_params("max_rows", types={"max_rows": int})
def create_expectation_max_rows(**kwargs) -> ExpectationMaxRows:
    """
    Create an ExpectationMaxRows instance.

    Args:
        max_rows (int): Maximum number of rows allowed.

    Returns:
        ExpectationMaxRows: A configured expectation instance.
    """
    return ExpectationMaxRows(max_rows=kwargs["max_rows"])


@register_expectation("ExpectationMaxNullPercentage")
@requires_params(
    "column_name",
    "max_percentage",
    types={"column_name": str, "max_percentage": (int, float)},
)
def create_expectation_max_null_percentage(**kwargs) -> ExpectationMaxNullPercentage:
    """
    Create an ExpectationMaxNullPercentage instance.

    Args:
        column_name (str): Name of the column to check for null percentage.
        max_percentage (float): Maximum percentage of null values allowed (0.0-100.0).

    Returns:
        ExpectationMaxNullPercentage: A configured expectation instance.
    """
    return ExpectationMaxNullPercentage(
        column_name=kwargs["column_name"],
        max_percentage=kwargs["max_percentage"],
    )


@register_expectation("ExpectationMaxNullCount")
@requires_params(
    "column_name",
    "max_count",
    types={"column_name": str, "max_count": int},
)
def create_expectation_max_null_count(**kwargs) -> ExpectationMaxNullCount:
    """
    Create an ExpectationMaxNullCount instance.

    Args:
        column_name (str): Name of the column to check for null count.
        max_count (int): Maximum number of null values allowed.

    Returns:
        ExpectationMaxNullCount: A configured expectation instance.
    """
    return ExpectationMaxNullCount(
        column_name=kwargs["column_name"],
        max_count=kwargs["max_count"],
    )
