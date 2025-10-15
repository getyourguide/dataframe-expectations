from pyspark.sql import functions as F

from dataframe_expectations.expectations.column_expectation import (
    DataframeColumnExpectation,
)
from dataframe_expectations.expectations.expectation_registry import (
    register_expectation,
)
from dataframe_expectations.expectations.utils import requires_params


@register_expectation("ExpectationStringContains")
@requires_params("column_name", "substring", types={"column_name": str, "substring": str})
def create_expectation_string_contains(**kwargs) -> DataframeColumnExpectation:
    column_name = kwargs["column_name"]
    substring = kwargs["substring"]
    return DataframeColumnExpectation(
        expectation_name="ExpectationStringContains",
        column_name=column_name,
        fn_violations_pandas=lambda df: df[~df[column_name].str.contains(substring, na=False)],
        fn_violations_pyspark=lambda df: df.filter(~F.col(column_name).contains(substring)),
        description=f"'{column_name}' contains '{substring}'",
        error_message=f"'{column_name}' does not contain '{substring}'.",
    )


@register_expectation("ExpectationStringNotContains")
@requires_params("column_name", "substring", types={"column_name": str, "substring": str})
def create_expectation_string_not_contains(**kwargs) -> DataframeColumnExpectation:
    column_name = kwargs["column_name"]
    substring = kwargs["substring"]
    return DataframeColumnExpectation(
        expectation_name="ExpectationStringNotContains",
        column_name=column_name,
        fn_violations_pandas=lambda df: df[df[column_name].str.contains(substring, na=False)],
        fn_violations_pyspark=lambda df: df.filter(F.col(column_name).contains(substring)),
        description=f"'{column_name}' does not contain '{substring}'",
        error_message=f"'{column_name}' contains '{substring}'.",
    )


@register_expectation("ExpectationStringStartsWith")
@requires_params("column_name", "prefix", types={"column_name": str, "prefix": str})
def create_expectation_string_starts_with(**kwargs) -> DataframeColumnExpectation:
    column_name = kwargs["column_name"]
    prefix = kwargs["prefix"]
    return DataframeColumnExpectation(
        expectation_name="ExpectationStringStartsWith",
        column_name=column_name,
        fn_violations_pandas=lambda df: df[~df[column_name].str.startswith(prefix, na=False)],
        fn_violations_pyspark=lambda df: df.filter(~F.col(column_name).startswith(prefix)),
        description=f"'{column_name}' starts with '{prefix}'",
        error_message=f"'{column_name}' does not start with '{prefix}'.",
    )


@register_expectation("ExpectationStringEndsWith")
@requires_params("column_name", "suffix", types={"column_name": str, "suffix": str})
def create_expectation_string_ends_with(**kwargs) -> DataframeColumnExpectation:
    column_name = kwargs["column_name"]
    suffix = kwargs["suffix"]
    return DataframeColumnExpectation(
        expectation_name="ExpectationStringEndsWith",
        column_name=column_name,
        fn_violations_pandas=lambda df: df[~df[column_name].str.endswith(suffix, na=False)],
        fn_violations_pyspark=lambda df: df.filter(~F.col(column_name).endswith(suffix)),
        description=f"'{column_name}' ends with '{suffix}'",
        error_message=f"'{column_name}' does not end with '{suffix}'.",
    )


@register_expectation("ExpectationStringLengthLessThan")
@requires_params("column_name", "length", types={"column_name": str, "length": int})
def create_expectation_string_length_less_than(**kwargs) -> DataframeColumnExpectation:
    column_name = kwargs["column_name"]
    length = kwargs["length"]
    return DataframeColumnExpectation(
        expectation_name="ExpectationStringLengthLessThan",
        column_name=column_name,
        fn_violations_pandas=lambda df: df[df[column_name].str.len() >= length],
        fn_violations_pyspark=lambda df: df.filter(F.length(column_name) >= length),
        description=f"'{column_name}' length is less than {length}",
        error_message=f"'{column_name}' length is not less than {length}.",
    )


@register_expectation("ExpectationStringLengthGreaterThan")
@requires_params("column_name", "length", types={"column_name": str, "length": int})
def create_expectation_string_length_greater_than(
    **kwargs,
) -> DataframeColumnExpectation:
    column_name = kwargs["column_name"]
    length = kwargs["length"]
    return DataframeColumnExpectation(
        expectation_name="ExpectationStringLengthGreaterThan",
        column_name=column_name,
        fn_violations_pandas=lambda df: df[df[column_name].str.len() <= length],
        fn_violations_pyspark=lambda df: df.filter(F.length(F.col(column_name)) <= length),
        description=f"'{column_name}' length is greater than {length}",
        error_message=f"'{column_name}' length is not greater than {length}.",
    )


@register_expectation("ExpectationStringLengthBetween")
@requires_params(
    "column_name",
    "min_length",
    "max_length",
    types={"column_name": str, "min_length": int, "max_length": int},
)
def create_expectation_string_length_between(**kwargs) -> DataframeColumnExpectation:
    column_name = kwargs["column_name"]
    min_length = kwargs["min_length"]
    max_length = kwargs["max_length"]
    return DataframeColumnExpectation(
        expectation_name="ExpectationStringLengthBetween",
        column_name=column_name,
        fn_violations_pandas=lambda df: df[
            (df[column_name].str.len() < min_length) | (df[column_name].str.len() > max_length)
        ],
        fn_violations_pyspark=lambda df: df.filter(
            (F.length(F.col(column_name)) < min_length)
            | (F.length(F.col(column_name)) > max_length)
        ),
        description=f"'{column_name}' length is between {min_length} and {max_length}",
        error_message=f"'{column_name}' length is not between {min_length} and {max_length}.",
    )


@register_expectation("ExpectationStringLengthEquals")
@requires_params("column_name", "length", types={"column_name": str, "length": int})
def create_expectation_string_length_equals(**kwargs) -> DataframeColumnExpectation:
    column_name = kwargs["column_name"]
    length = kwargs["length"]
    return DataframeColumnExpectation(
        expectation_name="ExpectationStringLengthEquals",
        column_name=column_name,
        fn_violations_pandas=lambda df: df[df[column_name].str.len() != length],
        fn_violations_pyspark=lambda df: df.filter(F.length(F.col(column_name)) != length),
        description=f"'{column_name}' length equals {length}",
        error_message=f"'{column_name}' length is not equal to {length}.",
    )
