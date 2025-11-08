from pyspark.sql import functions as F

from dataframe_expectations.expectations.column_expectation import (
    DataFrameColumnExpectation,
)
from dataframe_expectations.expectations.expectation_registry import (
    ExpectationCategory,
    ExpectationSubcategory,
    register_expectation,
)
from dataframe_expectations.expectations.utils import requires_params


@register_expectation(
    "ExpectationStringContains",
    pydoc="Check if the values in a string column contain a specified substring",
    category=ExpectationCategory.COLUMN_EXPECTATIONS,
    subcategory=ExpectationSubcategory.STRING,
    params_doc={
        "column_name": "The name of the column to check",
        "substring": "The substring to search for",
    },
)
@requires_params("column_name", "substring", types={"column_name": str, "substring": str})
def create_expectation_string_contains(**kwargs) -> DataFrameColumnExpectation:
    column_name = kwargs["column_name"]
    substring = kwargs["substring"]
    return DataFrameColumnExpectation(
        expectation_name="ExpectationStringContains",
        column_name=column_name,
        fn_violations_pandas=lambda df: df[~df[column_name].str.contains(substring, na=False)],
        fn_violations_pyspark=lambda df: df.filter(~F.col(column_name).contains(substring)),
        description=f"'{column_name}' contains '{substring}'",
        error_message=f"'{column_name}' does not contain '{substring}'.",
    )


@register_expectation(
    "ExpectationStringNotContains",
    pydoc="Check if the values in a string column do not contain a specified substring",
    category=ExpectationCategory.COLUMN_EXPECTATIONS,
    subcategory=ExpectationSubcategory.STRING,
    params_doc={
        "column_name": "The name of the column to check",
        "substring": "The substring to search for",
    },
)
@requires_params("column_name", "substring", types={"column_name": str, "substring": str})
def create_expectation_string_not_contains(**kwargs) -> DataFrameColumnExpectation:
    column_name = kwargs["column_name"]
    substring = kwargs["substring"]
    return DataFrameColumnExpectation(
        expectation_name="ExpectationStringNotContains",
        column_name=column_name,
        fn_violations_pandas=lambda df: df[df[column_name].str.contains(substring, na=False)],
        fn_violations_pyspark=lambda df: df.filter(F.col(column_name).contains(substring)),
        description=f"'{column_name}' does not contain '{substring}'",
        error_message=f"'{column_name}' contains '{substring}'.",
    )


@register_expectation(
    "ExpectationStringStartsWith",
    pydoc="Check if the values in a string column start with a specified prefix",
    category=ExpectationCategory.COLUMN_EXPECTATIONS,
    subcategory=ExpectationSubcategory.STRING,
    params_doc={
        "column_name": "The name of the column to check",
        "prefix": "The prefix to search for",
    },
)
@requires_params("column_name", "prefix", types={"column_name": str, "prefix": str})
def create_expectation_string_starts_with(**kwargs) -> DataFrameColumnExpectation:
    column_name = kwargs["column_name"]
    prefix = kwargs["prefix"]
    return DataFrameColumnExpectation(
        expectation_name="ExpectationStringStartsWith",
        column_name=column_name,
        fn_violations_pandas=lambda df: df[~df[column_name].str.startswith(prefix, na=False)],
        fn_violations_pyspark=lambda df: df.filter(~F.col(column_name).startswith(prefix)),
        description=f"'{column_name}' starts with '{prefix}'",
        error_message=f"'{column_name}' does not start with '{prefix}'.",
    )


@register_expectation(
    "ExpectationStringEndsWith",
    pydoc="Check if the values in a string column end with a specified suffix",
    category=ExpectationCategory.COLUMN_EXPECTATIONS,
    subcategory=ExpectationSubcategory.STRING,
    params_doc={
        "column_name": "The name of the column to check",
        "suffix": "The suffix to search for",
    },
)
@requires_params("column_name", "suffix", types={"column_name": str, "suffix": str})
def create_expectation_string_ends_with(**kwargs) -> DataFrameColumnExpectation:
    column_name = kwargs["column_name"]
    suffix = kwargs["suffix"]
    return DataFrameColumnExpectation(
        expectation_name="ExpectationStringEndsWith",
        column_name=column_name,
        fn_violations_pandas=lambda df: df[~df[column_name].str.endswith(suffix, na=False)],
        fn_violations_pyspark=lambda df: df.filter(~F.col(column_name).endswith(suffix)),
        description=f"'{column_name}' ends with '{suffix}'",
        error_message=f"'{column_name}' does not end with '{suffix}'.",
    )


@register_expectation(
    "ExpectationStringLengthLessThan",
    pydoc="Check if the length of the values in a string column is less than a specified length",
    category=ExpectationCategory.COLUMN_EXPECTATIONS,
    subcategory=ExpectationSubcategory.STRING,
    params_doc={
        "column_name": "The name of the column to check",
        "length": "The length that the values should be less than",
    },
)
@requires_params("column_name", "length", types={"column_name": str, "length": int})
def create_expectation_string_length_less_than(**kwargs) -> DataFrameColumnExpectation:
    column_name = kwargs["column_name"]
    length = kwargs["length"]
    return DataFrameColumnExpectation(
        expectation_name="ExpectationStringLengthLessThan",
        column_name=column_name,
        fn_violations_pandas=lambda df: df[df[column_name].str.len() >= length],
        fn_violations_pyspark=lambda df: df.filter(F.length(column_name) >= length),
        description=f"'{column_name}' length is less than {length}",
        error_message=f"'{column_name}' length is not less than {length}.",
    )


@register_expectation(
    "ExpectationStringLengthGreaterThan",
    pydoc="Check if the length of the values in a string column is greater than a specified length",
    category=ExpectationCategory.COLUMN_EXPECTATIONS,
    subcategory=ExpectationSubcategory.STRING,
    params_doc={
        "column_name": "The name of the column to check",
        "length": "The length that the values should be greater than",
    },
)
@requires_params("column_name", "length", types={"column_name": str, "length": int})
def create_expectation_string_length_greater_than(
    **kwargs,
) -> DataFrameColumnExpectation:
    column_name = kwargs["column_name"]
    length = kwargs["length"]
    return DataFrameColumnExpectation(
        expectation_name="ExpectationStringLengthGreaterThan",
        column_name=column_name,
        fn_violations_pandas=lambda df: df[df[column_name].str.len() <= length],
        fn_violations_pyspark=lambda df: df.filter(F.length(F.col(column_name)) <= length),
        description=f"'{column_name}' length is greater than {length}",
        error_message=f"'{column_name}' length is not greater than {length}.",
    )


@register_expectation(
    "ExpectationStringLengthBetween",
    pydoc="Check if the length of the values in a string column is between two specified lengths",
    category=ExpectationCategory.COLUMN_EXPECTATIONS,
    subcategory=ExpectationSubcategory.STRING,
    params_doc={
        "column_name": "The name of the column to check",
        "min_length": "The minimum length that the values should be",
        "max_length": "The maximum length that the values should be",
    },
)
@requires_params(
    "column_name",
    "min_length",
    "max_length",
    types={"column_name": str, "min_length": int, "max_length": int},
)
def create_expectation_string_length_between(**kwargs) -> DataFrameColumnExpectation:
    column_name = kwargs["column_name"]
    min_length = kwargs["min_length"]
    max_length = kwargs["max_length"]
    return DataFrameColumnExpectation(
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


@register_expectation(
    "ExpectationStringLengthEquals",
    pydoc="Check if the length of the values in a string column equals a specified length",
    category=ExpectationCategory.COLUMN_EXPECTATIONS,
    subcategory=ExpectationSubcategory.STRING,
    params_doc={
        "column_name": "The name of the column to check",
        "length": "The length that the values should equal",
    },
)
@requires_params("column_name", "length", types={"column_name": str, "length": int})
def create_expectation_string_length_equals(**kwargs) -> DataFrameColumnExpectation:
    column_name = kwargs["column_name"]
    length = kwargs["length"]
    return DataFrameColumnExpectation(
        expectation_name="ExpectationStringLengthEquals",
        column_name=column_name,
        fn_violations_pandas=lambda df: df[df[column_name].str.len() != length],
        fn_violations_pyspark=lambda df: df.filter(F.length(F.col(column_name)) != length),
        description=f"'{column_name}' length equals {length}",
        error_message=f"'{column_name}' length is not equal to {length}.",
    )
