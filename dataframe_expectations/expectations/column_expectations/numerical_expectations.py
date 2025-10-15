from pyspark.sql import functions as F

from dataframe_expectations.expectations.column_expectation import (
    DataframeColumnExpectation,
)
from dataframe_expectations.expectations.expectation_registry import (
    register_expectation,
)
from dataframe_expectations.expectations.utils import requires_params


@register_expectation("ExpectationValueGreaterThan")
@requires_params("column_name", "value", types={"column_name": str, "value": (int, float)})
def create_expectation_value_greater_than(**kwargs) -> DataframeColumnExpectation:
    column_name = kwargs["column_name"]
    value = kwargs["value"]
    return DataframeColumnExpectation(
        expectation_name="ExpectationValueGreaterThan",
        column_name=column_name,
        fn_violations_pandas=lambda df: df[df[column_name] <= value],
        fn_violations_pyspark=lambda df: df.filter(F.col(column_name) <= value),
        description=f"'{column_name}' is greater than {value}",
        error_message=f"'{column_name}' is not greater than {value}.",
    )


@register_expectation("ExpectationValueLessThan")
@requires_params("column_name", "value", types={"column_name": str, "value": (int, float)})
def create_expectation_value_less_than(**kwargs) -> DataframeColumnExpectation:
    column_name = kwargs["column_name"]
    value = kwargs["value"]
    return DataframeColumnExpectation(
        expectation_name="ExpectationValueLessThan",
        column_name=column_name,
        fn_violations_pandas=lambda df: df[df[column_name] >= value],
        fn_violations_pyspark=lambda df: df.filter(F.col(column_name) >= value),
        description=f"'{column_name}' is less than {value}",
        error_message=f"'{column_name}' is not less than {value}.",
    )


@register_expectation("ExpectationValueBetween")
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
def create_expectation_value_between(**kwargs) -> DataframeColumnExpectation:
    column_name = kwargs["column_name"]
    min_value = kwargs["min_value"]
    max_value = kwargs["max_value"]
    return DataframeColumnExpectation(
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
    )
