from abc import ABC, abstractmethod
from typing import List, Optional, cast

from pandas import DataFrame as PandasDataFrame
from pyspark.sql import DataFrame as PySparkDataFrame

# Import the connect DataFrame type for Spark Connect
try:
    from pyspark.sql.connect.dataframe import DataFrame as PySparkConnectDataFrame
except ImportError:
    # Fallback for older PySpark versions that don't have connect
    PySparkConnectDataFrame = None  # type: ignore[misc,assignment]

from dataframe_expectations.core.types import DataFrameLike, DataFrameType
from dataframe_expectations.core.tagging import TagSet
from dataframe_expectations.result_message import (
    DataFrameExpectationResultMessage,
)


class DataFrameExpectation(ABC):
    """
    Base class for DataFrame expectations.
    """

    def __init__(self, tags: Optional[List[str]] = None):
        """
        Initialize the base expectation with optional tags.
        :param tags: Optional tags as list of strings in "key:value" format.
                    Example: ["priority:high", "env:test"]
        """
        self.__tags = TagSet(tags)

    def get_tags(self) -> TagSet:
        """
        Returns the tags for this expectation.
        """
        return self.__tags

    def get_expectation_name(self) -> str:
        """
        Returns the class name as the expectation name.
        """
        return type(self).__name__

    @abstractmethod
    def get_description(self) -> str:
        """
        Returns a description of the expectation.
        """
        raise NotImplementedError(
            f"description method must be implemented for {self.__class__.__name__}"
        )

    def __str__(self):
        """
        Returns a string representation of the expectation.
        """
        return f"{self.get_expectation_name()} ({self.get_description()})"

    @classmethod
    def infer_data_frame_type(cls, data_frame: DataFrameLike) -> DataFrameType:
        """
        Infer the DataFrame type based on the provided DataFrame.
        """
        match data_frame:
            case PandasDataFrame():
                return DataFrameType.PANDAS
            case PySparkDataFrame():
                return DataFrameType.PYSPARK
            case _ if PySparkConnectDataFrame is not None and isinstance(
                data_frame, PySparkConnectDataFrame
            ):
                return DataFrameType.PYSPARK
            case _:
                raise ValueError(f"Unsupported DataFrame type: {type(data_frame)}")

    def validate(self, data_frame: DataFrameLike, **kwargs):
        """
        Validate the DataFrame against the expectation.
        """
        data_frame_type = self.infer_data_frame_type(data_frame)

        match data_frame_type:
            case DataFrameType.PANDAS:
                return self.validate_pandas(data_frame=data_frame, **kwargs)
            case DataFrameType.PYSPARK:
                return self.validate_pyspark(data_frame=data_frame, **kwargs)
            case _:
                raise ValueError(f"Unsupported DataFrame type: {data_frame_type}")

    @abstractmethod
    def validate_pandas(
        self, data_frame: DataFrameLike, **kwargs
    ) -> DataFrameExpectationResultMessage:
        """
        Validate a pandas DataFrame against the expectation.
        """
        raise NotImplementedError(
            f"validate_pandas method must be implemented for {self.__class__.__name__}"
        )

    @abstractmethod
    def validate_pyspark(
        self, data_frame: DataFrameLike, **kwargs
    ) -> DataFrameExpectationResultMessage:
        """
        Validate a PySpark DataFrame against the expectation.
        """
        raise NotImplementedError(
            f"validate_pyspark method must be implemented for {self.__class__.__name__}"
        )

    @classmethod
    def num_data_frame_rows(cls, data_frame: DataFrameLike) -> int:
        """
        Count the number of rows in the DataFrame.
        """
        data_frame_type = cls.infer_data_frame_type(data_frame)
        if data_frame_type == DataFrameType.PANDAS:
            # Cast to PandasDataFrame since we know it's a Pandas DataFrame at this point
            return len(cast(PandasDataFrame, data_frame))
        elif data_frame_type == DataFrameType.PYSPARK:
            # Cast to PySparkDataFrame since we know it's a PySpark DataFrame at this point
            return cast(PySparkDataFrame, data_frame).count()
        else:
            raise ValueError(f"Unsupported DataFrame type: {data_frame_type}")
