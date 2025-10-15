from abc import ABC
from typing import Optional

from tabulate import tabulate  # type: ignore

from dataframe_expectations import DataFrameLike, DataFrameType


class DataframeExpectationResultMessage(ABC):
    """
    Base class for expectation result message.
    """

    message: str = ""

    def __str__(self):
        """
        Print the result of the expectation.
        """
        return self.message

    def dataframe_to_str(self, data_frame_type: DataFrameType, data_frame, rows: int) -> str:
        """
        Print the DataFrame based on its type.
        """

        if data_frame_type == DataFrameType.PANDAS:
            data_frame = data_frame.head(rows)
        elif data_frame_type == DataFrameType.PYSPARK:
            data_frame = data_frame.limit(rows).toPandas()
        else:
            raise ValueError(f"Unsupported DataFrame type: {data_frame_type}")

        return tabulate(data_frame, headers="keys", tablefmt="pretty", showindex=False)


class DataframeExpectationSuccessMessage(DataframeExpectationResultMessage):
    def __init__(self, expectation_name: str, message: Optional[str] = None):
        """
        Initialize the expectation success message.
        """
        self.message = f"{expectation_name} succeeded."
        if message is not None:
            self.message = f"{self.message}: {message}"


class DataframeExpectationFailureMessage(DataframeExpectationResultMessage):
    def __init__(
        self,
        expectation_str: str,
        data_frame_type: DataFrameType,
        violations_data_frame: Optional[DataFrameLike] = None,
        message: Optional[str] = None,
        limit_violations: int = 5,
    ):
        self.message = expectation_str
        if message is not None:
            self.message = f"{self.message}: {message}"
        if violations_data_frame is not None:
            self.data_frame_type = data_frame_type

            self.violations_data_frame = violations_data_frame
            violations_dataframe_str = self.dataframe_to_str(
                data_frame_type=data_frame_type,
                data_frame=violations_data_frame,
                rows=limit_violations,
            )
            self.message = (
                f"{self.message} \nSome examples of violations: \n{violations_dataframe_str}"
            )

    def get_violations_data_frame(self) -> Optional[DataFrameLike]:
        """
        Get the DataFrame with violations.
        """
        return self.violations_data_frame if hasattr(self, "violations_data_frame") else None
