from enum import Enum
from typing import Union

from pandas import DataFrame as PandasDataFrame
from pyspark.sql import DataFrame as PySparkDataFrame

DataFrameLike = Union[PySparkDataFrame, PandasDataFrame]


class DataFrameType(str, Enum):
    """
    Enum for DataFrame types.
    """

    PANDAS = "pandas"
    PYSPARK = "pyspark"
