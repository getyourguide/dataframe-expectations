from pyspark.sql import functions as F

from dataframe_expectations.expectations.column_expectation import (
    DataFrameColumnExpectation,
)
from dataframe_expectations.expectations.expectation_registry import (
    register_expectation,
)
from dataframe_expectations.expectations.utils import requires_params


@register_expectation(
    "ExpectationValueEquals",
    description="Check if the values in a column equal a specified value",
    category="Column Expectations",
    subcategory="Any Value",
    params_doc={
        "column_name": "The name of the column to check",
        "value": "The value to compare against",
    },
)
@requires_params("column_name", "value", types={"column_name": str, "value": object})
def create_expectation_value_equals(**kwargs) -> DataFrameColumnExpectation:
    column_name = kwargs["column_name"]
    value = kwargs["value"]
    return DataFrameColumnExpectation(
        expectation_name="ExpectationValueEquals",
        column_name=column_name,
        fn_violations_pandas=lambda df: df[df[column_name] != value],
        fn_violations_pyspark=lambda df: df.filter(F.col(column_name) != value),
        description=f"'{column_name}' equals {value}",
        error_message=f"'{column_name}' is not equal to {value}.",
    )


@register_expectation(
    "ExpectationValueNotEquals",
    description="Check if the values in a column do not equal a specified value",
    category="Column Expectations",
    subcategory="Any Value",
    params_doc={
        "column_name": "The name of the column to check",
        "value": "The value to compare against",
    },
)
@requires_params("column_name", "value", types={"column_name": str, "value": object})
def create_expectation_value_not_equals(**kwargs) -> DataFrameColumnExpectation:
    column_name = kwargs["column_name"]
    value = kwargs["value"]
    return DataFrameColumnExpectation(
        expectation_name="ExpectationValueNotEquals",
        column_name=column_name,
        fn_violations_pandas=lambda df: df[df[column_name] == value],
        fn_violations_pyspark=lambda df: df.filter(F.col(column_name) == value),
        description=f"'{column_name}' is not equal to {value}",
        error_message=f"'{column_name}' is equal to {value}.",
    )


@register_expectation(
    "ExpectationValueNull",
    description="Check if the values in a column are null",
    category="Column Expectations",
    subcategory="Any Value",
    params_doc={
        "column_name": "The name of the column to check",
    },
)
@requires_params("column_name", types={"column_name": str})
def create_expectation_value_null(**kwargs) -> DataFrameColumnExpectation:
    column_name = kwargs["column_name"]
    return DataFrameColumnExpectation(
        expectation_name="ExpectationValueNull",
        column_name=column_name,
        fn_violations_pandas=lambda df: df[df[column_name].notnull()],
        fn_violations_pyspark=lambda df: df.filter(F.col(column_name).isNotNull()),
        description=f"'{column_name}' is null",
        error_message=f"'{column_name}' is not null.",
    )


@register_expectation(
    "ExpectationValueNotNull",
    description="Check if the values in a column are not null",
    category="Column Expectations",
    subcategory="Any Value",
    params_doc={
        "column_name": "The name of the column to check",
    },
)
@requires_params("column_name", types={"column_name": str})
def create_expectation_value_not_null(**kwargs) -> DataFrameColumnExpectation:
    column_name = kwargs["column_name"]
    return DataFrameColumnExpectation(
        expectation_name="ExpectationValueNotNull",
        column_name=column_name,
        fn_violations_pandas=lambda df: df[df[column_name].isnull()],
        fn_violations_pyspark=lambda df: df.filter(F.col(column_name).isNull()),
        description=f"'{column_name}' is not null",
        error_message=f"'{column_name}' is null.",
    )


@register_expectation(
    "ExpectationValueIn",
    description="Check if the values in a column are in a specified list of values",
    category="Column Expectations",
    subcategory="Any Value",
    params_doc={
        "column_name": "The name of the column to check",
        "values": "The list of values to compare against",
    },
)
@requires_params("column_name", "values", types={"column_name": str, "values": list})
def create_expectation_value_in(**kwargs) -> DataFrameColumnExpectation:
    column_name = kwargs["column_name"]
    values = kwargs["values"]
    return DataFrameColumnExpectation(
        expectation_name="ExpectationValueIn",
        column_name=column_name,
        fn_violations_pandas=lambda df: df[~df[column_name].isin(values)],
        fn_violations_pyspark=lambda df: df.filter(~F.col(column_name).isin(values)),
        description=f"'{column_name}' is in {values}",
        error_message=f"'{column_name}' is not in {values}.",
    )


@register_expectation(
    "ExpectationValueNotIn",
    description="Check if the values in a column are not in a specified list of values",
    category="Column Expectations",
    subcategory="Any Value",
    params_doc={
        "column_name": "The name of the column to check",
        "values": "The list of values to compare against",
    },
)
@requires_params("column_name", "values", types={"column_name": str, "values": list})
def create_expectation_value_not_in(**kwargs) -> DataFrameColumnExpectation:
    column_name = kwargs["column_name"]
    values = kwargs["values"]
    return DataFrameColumnExpectation(
        expectation_name="ExpectationValueNotIn",
        column_name=column_name,
        fn_violations_pandas=lambda df: df[df[column_name].isin(values)],
        fn_violations_pyspark=lambda df: df.filter(F.col(column_name).isin(values)),
        description=f"'{column_name}' is not in {values}",
        error_message=f"'{column_name}' is in {values}.",
    )
