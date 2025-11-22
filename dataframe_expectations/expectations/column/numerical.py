from typing import List, Optional

from pyspark.sql import functions as F

from dataframe_expectations.core.column_expectation import (
    DataFrameColumnExpectation,
)
from dataframe_expectations.core.types import (
    ExpectationCategory,
    ExpectationSubcategory,
)
from dataframe_expectations.registry import register_expectation
from dataframe_expectations.core.utils import requires_params


@register_expectation(
    "ExpectationValueGreaterThan",
    pydoc="Check if the values in a column are greater than a specified value",
    category=ExpectationCategory.COLUMN_EXPECTATIONS,
    subcategory=ExpectationSubcategory.NUMERICAL,
    params_doc={
        "column_name": "The name of the column to check",
        "value": "The value to compare against",
    },
)
@requires_params("column_name", "value", types={"column_name": str, "value": (int, float)})
def create_expectation_value_greater_than(
    column_name: str, value: float, tags: Optional[List[str]] = None
) -> DataFrameColumnExpectation:
    column_name = column_name
    value = value
    return DataFrameColumnExpectation(
        expectation_name="ExpectationValueGreaterThan",
        column_name=column_name,
        fn_violations_pandas=lambda df: df[df[column_name] <= value],
        fn_violations_pyspark=lambda df: df.filter(F.col(column_name) <= value),
        description=f"'{column_name}' is greater than {value}",
        error_message=f"'{column_name}' is not greater than {value}.",
        tags=tags,
    )


@register_expectation(
    "ExpectationValueLessThan",
    pydoc="Check if the values in a column are less than a specified value",
    category=ExpectationCategory.COLUMN_EXPECTATIONS,
    subcategory=ExpectationSubcategory.NUMERICAL,
    params_doc={
        "column_name": "The name of the column to check",
        "value": "The value to compare against",
    },
)
@requires_params("column_name", "value", types={"column_name": str, "value": (int, float)})
def create_expectation_value_less_than(
    column_name: str, value: float, tags: Optional[List[str]] = None
) -> DataFrameColumnExpectation:
    column_name = column_name
    value = value
    return DataFrameColumnExpectation(
        expectation_name="ExpectationValueLessThan",
        column_name=column_name,
        fn_violations_pandas=lambda df: df[df[column_name] >= value],
        fn_violations_pyspark=lambda df: df.filter(F.col(column_name) >= value),
        description=f"'{column_name}' is less than {value}",
        error_message=f"'{column_name}' is not less than {value}.",
        tags=tags,
    )


@register_expectation(
    "ExpectationValueBetween",
    pydoc="Check if the values in a column are between two specified values",
    category=ExpectationCategory.COLUMN_EXPECTATIONS,
    subcategory=ExpectationSubcategory.NUMERICAL,
    params_doc={
        "column_name": "The name of the column to check",
        "min_value": "The minimum value for the range",
        "max_value": "The maximum value for the range",
    },
)
@requires_params(
    "column_name",
    "min_value",
    "max_value",
    types={
        "column_name": str,
        "min_value": (int, float),
        "max_value": (int, float),
    },
)
def create_expectation_value_between(
    column_name: str, min_value: float, max_value: float, tags: Optional[List[str]] = None
) -> DataFrameColumnExpectation:
    column_name = column_name
    min_value = min_value
    max_value = max_value
    return DataFrameColumnExpectation(
        expectation_name="ExpectationValueBetween",
        column_name=column_name,
        fn_violations_pandas=lambda df: df[
            (df[column_name] < min_value) | (df[column_name] > max_value)
        ],
        fn_violations_pyspark=lambda df: df.filter(
            (F.col(column_name) < min_value) | (F.col(column_name) > max_value)
        ),
        description=f"'{column_name}' is between {min_value} and {max_value}",
        error_message=f"'{column_name}' is not between {min_value} and {max_value}.",
        tags=tags,
    )
