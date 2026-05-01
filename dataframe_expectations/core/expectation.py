from abc import ABC, abstractmethod
from typing import Any, List, Optional, cast

from pandas import DataFrame as PandasDataFrame

from dataframe_expectations.core.pyspark_utils import is_pyspark_data_frame
from dataframe_expectations.core.types import DataFrameLike, DataFrameType
from dataframe_expectations.core.tagging import TagSet
from dataframe_expectations.result_message import (
    DataFrameExpectationResultMessage,
)
from dataframe_expectations.core.polars_utils import is_polars_data_frame

# Kept as module-level symbol for backward compatibility and test patching.
try:
    from pyspark.sql.connect.dataframe import DataFrame as PySparkConnectDataFrame
except ImportError:
    PySparkConnectDataFrame = None  # type: ignore[misc,assignment]


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
        if isinstance(data_frame, PandasDataFrame):
            return DataFrameType.PANDAS

        if PySparkConnectDataFrame is not None and isinstance(data_frame, PySparkConnectDataFrame):
            return DataFrameType.PYSPARK

        if is_pyspark_data_frame(data_frame):
            return DataFrameType.PYSPARK

        if is_polars_data_frame(data_frame):
            return DataFrameType.POLARS

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
            case DataFrameType.POLARS:
                return self.validate_polars(data_frame=data_frame, **kwargs)
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

    @abstractmethod
    def validate_polars(
        self, data_frame: DataFrameLike, **kwargs
    ) -> DataFrameExpectationResultMessage:
        """
        Validate a Polars DataFrame against the expectation.
        """
        raise NotImplementedError(
            f"validate_polars method must be implemented for {self.__class__.__name__}"
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
            # PySpark DataFrames expose count(), including runtime-provided patched variants.
            return cast(Any, data_frame).count()
        elif data_frame_type == DataFrameType.POLARS:
            # Polars DataFrames use height or len() for row count
            return len(cast(Any, data_frame))
        else:
            raise ValueError(f"Unsupported DataFrame type: {data_frame_type}")
