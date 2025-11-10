"""Core types, enums, and data models for dataframe-expectations."""

from enum import Enum
from typing import Any, Dict, Union

from pandas import DataFrame as PandasDataFrame
from pydantic import BaseModel, ConfigDict, Field
from pyspark.sql import DataFrame as PySparkDataFrame

# Type aliases
DataFrameLike = Union[PySparkDataFrame, PandasDataFrame]


class DataFrameType(str, Enum):
    """Enum for DataFrame types."""

    PANDAS = "pandas"
    PYSPARK = "pyspark"


class ExpectationCategory(str, Enum):
    """Categories for expectations."""

    COLUMN_EXPECTATIONS = "Column Expectations"
    COLUMN_AGGREGATION_EXPECTATIONS = "Column Aggregation Expectations"
    DATAFRAME_AGGREGATION_EXPECTATIONS = "DataFrame Aggregation Expectations"


class ExpectationSubcategory(str, Enum):
    """Subcategory of expectations."""

    ANY_VALUE = "Any Value"
    NUMERICAL = "Numerical"
    STRING = "String"
    UNIQUE = "Unique"


class ExpectationMetadata(BaseModel):
    """Metadata for a registered expectation."""

    suite_method_name: str = Field(
        ..., description="Method name in ExpectationsSuite (e.g., 'expect_value_greater_than')"
    )
    pydoc: str = Field(..., description="Human-readable description of the expectation")
    category: ExpectationCategory = Field(..., description="Category (e.g., 'Column Expectations')")
    subcategory: ExpectationSubcategory = Field(
        ..., description="Subcategory (e.g., 'Numerical', 'String')"
    )
    params_doc: Dict[str, str] = Field(..., description="Documentation for each parameter")
    params: list = Field(default_factory=list, description="List of required parameter names")
    param_types: Dict[str, Any] = Field(
        default_factory=dict, description="Type hints for parameters"
    )
    factory_func_name: str = Field(..., description="Name of the factory function")
    expectation_name: str = Field(..., description="Name of the expectation class")

    model_config = ConfigDict(frozen=True)  # Make model immutable
