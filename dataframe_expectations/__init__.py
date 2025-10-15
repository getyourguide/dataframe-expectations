from enum import Enum
from typing import Union

from pandas import DataFrame as PandasDataFrame
from pyspark.sql import DataFrame as PySparkDataFrame

DataFrameLike = Union[PySparkDataFrame, PandasDataFrame]


class DataFrameType(Enum):
    """
    Enum for DataFrame types.
    """

    PANDAS = "pandas"
    PYSPARK = "pyspark"

    def __str__(self):
        """
        Get the name of the DataFrame type.
        """
        return self.value
