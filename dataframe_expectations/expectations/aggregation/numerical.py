from typing import Union, cast

import pandas as pd
from pandas import DataFrame as PandasDataFrame
from pyspark.sql import DataFrame as PySparkDataFrame
from pyspark.sql import functions as F

from dataframe_expectations.core.types import DataFrameLike, DataFrameType
from dataframe_expectations.core.aggregation_expectation import (
    DataFrameAggregationExpectation,
)
from dataframe_expectations.core.types import (
    ExpectationCategory,
    ExpectationSubcategory,
)
from dataframe_expectations.registry import register_expectation
from dataframe_expectations.core.utils import requires_params
from dataframe_expectations.result_message import (
    DataFrameExpectationFailureMessage,
    DataFrameExpectationResultMessage,
    DataFrameExpectationSuccessMessage,
)


class ExpectationColumnQuantileBetween(DataFrameAggregationExpectation):
    """
    Expectation that validates a quantile value of a column falls within a specified range.

    This expectation computes the specified quantile of the column and checks if it
    falls between the provided minimum and maximum bounds (inclusive).

    Quantile values:
    - 0.0 = minimum value
    - 0.5 = median value
    - 1.0 = maximum value
    - Any value between 0.0 and 1.0 for custom quantiles

    Examples:
        Column 'age' with values [20, 25, 30, 35]:
        - quantile=0.5 (median) = 27.5
        - ExpectationColumnQuantileBetween(column_name="age", quantile=0.5, min_value=25, max_value=30) → PASS
        - ExpectationColumnQuantileBetween(column_name="age", quantile=1.0, min_value=30, max_value=40) → PASS (max=35)
        - ExpectationColumnQuantileBetween(column_name="age", quantile=0.0, min_value=15, max_value=25) → PASS (min=20)
    """

    def __init__(
        self,
        column_name: str,
        quantile: float,
        min_value: Union[int, float],
        max_value: Union[int, float],
    ):
        """
        Initialize the column quantile between expectation.

        :param column_name: Name of the column to check.
        :param quantile: Quantile to compute (0.0 to 1.0, where 0.0=min, 0.5=median, 1.0=max).
        :param min_value: Minimum allowed value for the column quantile (inclusive).
        :param max_value: Maximum allowed value for the column quantile (inclusive).
        :raises ValueError: If quantile is not between 0.0 and 1.0.
        """
        if not (0.0 <= quantile <= 1.0):
            raise ValueError(f"Quantile must be between 0.0 and 1.0, got {quantile}")

        # Create descriptive names for common quantiles
        quantile_names = {
            0.0: "minimum",
            0.25: "25th percentile",
            0.5: "median",
            0.75: "75th percentile",
            1.0: "maximum",
        }
        self.quantile_desc = quantile_names.get(quantile, f"{quantile} quantile")

        description = (
            f"column '{column_name}' {self.quantile_desc} value between {min_value} and {max_value}"
        )

        self.column_name = column_name
        self.quantile = quantile
        self.min_value = min_value
        self.max_value = max_value

        super().__init__(
            expectation_name="ExpectationColumnQuantileBetween",
            column_names=[column_name],
            description=description,
        )

    def aggregate_and_validate_pandas(
        self, data_frame: DataFrameLike, **kwargs
    ) -> DataFrameExpectationResultMessage:
        """Validate column quantile in a pandas DataFrame."""
        try:
            # Cast to PandasDataFrame for type safety
            pandas_df = cast(PandasDataFrame, data_frame)
            # Calculate quantile
            if self.quantile == 0.0:
                quantile_val = pandas_df[self.column_name].min()
            elif self.quantile == 1.0:
                quantile_val = pandas_df[self.column_name].max()
            elif self.quantile == 0.5:
                quantile_val = pandas_df[self.column_name].median()
            else:
                quantile_val = pandas_df[self.column_name].quantile(self.quantile)

            # Handle case where all values are null
            if pd.isna(quantile_val):
                return DataFrameExpectationFailureMessage(
                    expectation_str=str(self),
                    data_frame_type=DataFrameType.PANDAS,
                    message=f"Column '{self.column_name}' contains only null values.",
                )

            # Check if quantile is within bounds
            if self.min_value <= quantile_val <= self.max_value:
                return DataFrameExpectationSuccessMessage(
                    expectation_name=self.get_expectation_name()
                )
            else:
                return DataFrameExpectationFailureMessage(
                    expectation_str=str(self),
                    data_frame_type=DataFrameType.PANDAS,
                    message=(
                        f"Column '{self.column_name}' {self.quantile_desc} value {quantile_val} is not between "
                        f"{self.min_value} and {self.max_value}."
                    ),
                )

        except Exception as e:
            return DataFrameExpectationFailureMessage(
                expectation_str=str(self),
                data_frame_type=DataFrameType.PANDAS,
                message=f"Error calculating {self.quantile} quantile for column '{self.column_name}': {str(e)}",
            )

    def aggregate_and_validate_pyspark(
        self, data_frame: DataFrameLike, **kwargs
    ) -> DataFrameExpectationResultMessage:
        """Validate column quantile in a PySpark DataFrame."""
        try:
            # Cast to PySparkDataFrame for type safety
            pyspark_df = cast(PySparkDataFrame, data_frame)
            # First check if all values are null to avoid edge cases
            non_null_count = pyspark_df.select(F.count(self.column_name)).collect()[0][0]
            if non_null_count == 0:
                return DataFrameExpectationFailureMessage(
                    expectation_str=str(self),
                    data_frame_type=DataFrameType.PYSPARK,
                    message=f"Column '{self.column_name}' contains only null values.",
                )

            # Calculate quantile
            if self.quantile == 0.0:
                result = pyspark_df.select(F.min(self.column_name).alias("quantile_val")).collect()
            elif self.quantile == 1.0:
                result = pyspark_df.select(F.max(self.column_name).alias("quantile_val")).collect()
            elif self.quantile == 0.5:
                result = pyspark_df.select(
                    F.median(self.column_name).alias("quantile_val")  # type: ignore
                ).collect()
            else:
                # Use percentile_approx for other quantiles
                result = pyspark_df.select(
                    F.percentile_approx(F.col(self.column_name), F.lit(self.quantile)).alias(  # type: ignore
                        "quantile_val"
                    )
                ).collect()

            quantile_val = result[0]["quantile_val"]

            # Defensive check: quantile_val should not be None after the non-null count check above,
            # but we keep this for extra safety in case of unexpected Spark behavior or schema issues.
            if quantile_val is None:
                return DataFrameExpectationFailureMessage(
                    expectation_str=str(self),
                    data_frame_type=DataFrameType.PYSPARK,
                    message=f"Column '{self.column_name}' contains only null values.",
                )

            # Check if quantile is within bounds
            if self.min_value <= quantile_val <= self.max_value:
                return DataFrameExpectationSuccessMessage(
                    expectation_name=self.get_expectation_name()
                )
            else:
                return DataFrameExpectationFailureMessage(
                    expectation_str=str(self),
                    data_frame_type=DataFrameType.PYSPARK,
                    message=f"Column '{self.column_name}' {self.quantile_desc} value {quantile_val} is not between {self.min_value} and {self.max_value}.",
                )

        except Exception as e:
            return DataFrameExpectationFailureMessage(
                expectation_str=str(self),
                data_frame_type=DataFrameType.PYSPARK,
                message=f"Error calculating {self.quantile} quantile for column '{self.column_name}': {str(e)}",
            )


class ExpectationColumnMeanBetween(DataFrameAggregationExpectation):
    """
    Expectation that validates the mean value of a column falls within a specified range.

    This expectation computes the mean (average) value of the specified column and checks if it
    falls between the provided minimum and maximum bounds (inclusive).

    Note: Mean is implemented separately since it's not a quantile operation.

    Examples:
        Column 'age' with values [20, 25, 30, 35]:
        - mean_value = 27.5
        - ExpectationColumnMeanBetween(column_name="age", min_value=25, max_value=30) → PASS
        - ExpectationColumnMeanBetween(column_name="age", min_value=30, max_value=35) → FAIL
    """

    def __init__(
        self,
        column_name: str,
        min_value: Union[int, float],
        max_value: Union[int, float],
    ):
        """
        Initialize the column mean between expectation.

        :param column_name: Name of the column to check.
        :param min_value: Minimum allowed value for the column mean (inclusive).
        :param max_value: Maximum allowed value for the column mean (inclusive).
        """
        description = f"column '{column_name}' mean value between {min_value} and {max_value}"

        self.column_name = column_name
        self.min_value = min_value
        self.max_value = max_value

        super().__init__(
            expectation_name="ExpectationColumnMeanBetween",
            column_names=[column_name],
            description=description,
        )

    def aggregate_and_validate_pandas(
        self, data_frame: DataFrameLike, **kwargs
    ) -> DataFrameExpectationResultMessage:
        """Validate column mean in a pandas DataFrame."""
        try:
            # Cast to PandasDataFrame for type safety
            pandas_df = cast(PandasDataFrame, data_frame)
            # Calculate mean
            mean_val = pandas_df[self.column_name].mean()

            # Handle case where all values are null
            if pd.isna(mean_val):
                return DataFrameExpectationFailureMessage(
                    expectation_str=str(self),
                    data_frame_type=DataFrameType.PANDAS,
                    message=f"Column '{self.column_name}' contains only null values.",
                )

            # Check if mean is within bounds
            if self.min_value <= mean_val <= self.max_value:
                return DataFrameExpectationSuccessMessage(
                    expectation_name=self.get_expectation_name()
                )
            else:
                return DataFrameExpectationFailureMessage(
                    expectation_str=str(self),
                    data_frame_type=DataFrameType.PANDAS,
                    message=f"Column '{self.column_name}' mean value {mean_val} is not between {self.min_value} and {self.max_value}.",
                )

        except Exception as e:
            return DataFrameExpectationFailureMessage(
                expectation_str=str(self),
                data_frame_type=DataFrameType.PANDAS,
                message=f"Error calculating mean for column '{self.column_name}': {str(e)}",
            )

    def aggregate_and_validate_pyspark(
        self, data_frame: DataFrameLike, **kwargs
    ) -> DataFrameExpectationResultMessage:
        """Validate column mean in a PySpark DataFrame."""
        try:
            # Cast to PySparkDataFrame for type safety
            pyspark_df = cast(PySparkDataFrame, data_frame)
            # Calculate mean
            mean_result = pyspark_df.select(F.avg(self.column_name).alias("mean_val")).collect()
            mean_val = mean_result[0]["mean_val"]

            # Handle case where all values are null
            if mean_val is None:
                return DataFrameExpectationFailureMessage(
                    expectation_str=str(self),
                    data_frame_type=DataFrameType.PYSPARK,
                    message=f"Column '{self.column_name}' contains only null values.",
                )

            # Check if mean is within bounds
            if self.min_value <= mean_val <= self.max_value:
                return DataFrameExpectationSuccessMessage(
                    expectation_name=self.get_expectation_name()
                )
            else:
                return DataFrameExpectationFailureMessage(
                    expectation_str=str(self),
                    data_frame_type=DataFrameType.PYSPARK,
                    message=f"Column '{self.column_name}' mean value {mean_val} is not between {self.min_value} and {self.max_value}.",
                )

        except Exception as e:
            return DataFrameExpectationFailureMessage(
                expectation_str=str(self),
                data_frame_type=DataFrameType.PYSPARK,
                message=f"Error calculating mean for column '{self.column_name}': {str(e)}",
            )


# Register the main expectation
@register_expectation(
    "ExpectationColumnQuantileBetween",
    pydoc="Check if a specific quantile of a numeric column falls within a specified range",
    category=ExpectationCategory.COLUMN_AGGREGATION_EXPECTATIONS,
    subcategory=ExpectationSubcategory.NUMERICAL,
    params_doc={
        "column_name": "The name of the numeric column to check",
        "quantile": "The quantile to calculate (0.0 to 1.0, e.g., 0.5 for median)",
        "min_value": "The minimum allowed value for the quantile",
        "max_value": "The maximum allowed value for the quantile",
    },
)
@requires_params(
    "column_name",
    "quantile",
    "min_value",
    "max_value",
    types={
        "column_name": str,
        "quantile": (int, float),
        "min_value": (int, float),
        "max_value": (int, float),
    },
)
def create_expectation_column_quantile_between(
    column_name: str,
    quantile: float,
    min_value: Union[int, float],
    max_value: Union[int, float],
) -> ExpectationColumnQuantileBetween:
    """
    Create an ExpectationColumnQuantileBetween instance.

    :param column_name: Name of the column to check.
    :param quantile: Quantile to compute (0.0 to 1.0).
    :param min_value: Minimum allowed value for the column quantile.
    :param max_value: Maximum allowed value for the column quantile.
    :return: A configured expectation instance.
    """
    return ExpectationColumnQuantileBetween(
        column_name=column_name,
        quantile=quantile,
        min_value=min_value,
        max_value=max_value,
    )


# Convenience functions for common quantiles
@register_expectation(
    "ExpectationColumnMaxBetween",
    pydoc="Check if the maximum value of a numeric column falls within a specified range",
    category=ExpectationCategory.COLUMN_AGGREGATION_EXPECTATIONS,
    subcategory=ExpectationSubcategory.NUMERICAL,
    params_doc={
        "column_name": "The name of the numeric column to check",
        "min_value": "The minimum allowed maximum value",
        "max_value": "The maximum allowed maximum value",
    },
)
@requires_params(
    "column_name",
    "min_value",
    "max_value",
    types={"column_name": str, "min_value": (int, float), "max_value": (int, float)},
)
def create_expectation_column_max_to_be_between(
    column_name: str,
    min_value: Union[int, float],
    max_value: Union[int, float],
) -> ExpectationColumnQuantileBetween:
    """
    Create an ExpectationColumnQuantileBetween instance for maximum values (quantile=1.0).

    :param column_name: Name of the column to check.
    :param min_value: Minimum allowed value for the column maximum.
    :param max_value: Maximum allowed value for the column maximum.
    :return: A configured expectation instance for maximum values.
    """
    return ExpectationColumnQuantileBetween(
        column_name=column_name,
        quantile=1.0,
        min_value=min_value,
        max_value=max_value,
    )


@register_expectation(
    "ExpectationColumnMinBetween",
    pydoc="Check if the minimum value of a numeric column falls within a specified range",
    category=ExpectationCategory.COLUMN_AGGREGATION_EXPECTATIONS,
    subcategory=ExpectationSubcategory.NUMERICAL,
    params_doc={
        "column_name": "The name of the numeric column to check",
        "min_value": "The minimum allowed minimum value",
        "max_value": "The maximum allowed minimum value",
    },
)
@requires_params(
    "column_name",
    "min_value",
    "max_value",
    types={"column_name": str, "min_value": (int, float), "max_value": (int, float)},
)
def create_expectation_column_min_to_be_between(
    column_name: str,
    min_value: Union[int, float],
    max_value: Union[int, float],
) -> ExpectationColumnQuantileBetween:
    """
    Create an ExpectationColumnQuantileBetween instance for minimum values (quantile=0.0).

    :param column_name: Name of the column to check.
    :param min_value: Minimum allowed value for the column minimum.
    :param max_value: Maximum allowed value for the column minimum.
    :return: A configured expectation instance for minimum values.
    """
    return ExpectationColumnQuantileBetween(
        column_name=column_name,
        quantile=0.0,
        min_value=min_value,
        max_value=max_value,
    )


@register_expectation(
    "ExpectationColumnMeanBetween",
    pydoc="Check if the mean (average) of a numeric column falls within a specified range",
    category=ExpectationCategory.COLUMN_AGGREGATION_EXPECTATIONS,
    subcategory=ExpectationSubcategory.NUMERICAL,
    params_doc={
        "column_name": "The name of the numeric column to check",
        "min_value": "The minimum allowed mean value",
        "max_value": "The maximum allowed mean value",
    },
)
@requires_params(
    "column_name",
    "min_value",
    "max_value",
    types={"column_name": str, "min_value": (int, float), "max_value": (int, float)},
)
def create_expectation_column_mean_to_be_between(
    column_name: str,
    min_value: Union[int, float],
    max_value: Union[int, float],
) -> ExpectationColumnMeanBetween:
    """
    Create a custom ExpectationColumnMeanBetween instance for mean values.
    Note: This uses a separate implementation since mean is not a quantile.

    :param column_name: Name of the column to check.
    :param min_value: Minimum allowed value for the column mean.
    :param max_value: Maximum allowed value for the column mean.
    :return: A configured expectation instance for mean values.
    """
    # For mean, we need a separate class since it's not a quantile
    return ExpectationColumnMeanBetween(
        column_name=column_name,
        min_value=min_value,
        max_value=max_value,
    )


@register_expectation(
    "ExpectationColumnMedianBetween",
    pydoc="Check if the median of a numeric column falls within a specified range",
    category=ExpectationCategory.COLUMN_AGGREGATION_EXPECTATIONS,
    subcategory=ExpectationSubcategory.NUMERICAL,
    params_doc={
        "column_name": "The name of the numeric column to check",
        "min_value": "The minimum allowed median value",
        "max_value": "The maximum allowed median value",
    },
)
@requires_params(
    "column_name",
    "min_value",
    "max_value",
    types={"column_name": str, "min_value": (int, float), "max_value": (int, float)},
)
def create_expectation_column_median_to_be_between(
    column_name: str,
    min_value: Union[int, float],
    max_value: Union[int, float],
) -> ExpectationColumnQuantileBetween:
    """
    Create an ExpectationColumnQuantileBetween instance for median values (quantile=0.5).

    :param column_name: Name of the column to check.
    :param min_value: Minimum allowed value for the column median.
    :param max_value: Maximum allowed value for the column median.
    :return: A configured expectation instance for median values.
    """
    return ExpectationColumnQuantileBetween(
        column_name=column_name,
        quantile=0.5,
        min_value=min_value,
        max_value=max_value,
    )
