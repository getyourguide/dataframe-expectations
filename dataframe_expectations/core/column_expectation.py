from typing import Callable, List, Optional

from dataframe_expectations.core.types import DataFrameLike, DataFrameType
from dataframe_expectations.core.expectation import DataFrameExpectation
from dataframe_expectations.result_message import (
    DataFrameExpectationFailureMessage,
    DataFrameExpectationResultMessage,
    DataFrameExpectationSuccessMessage,
)


class DataFrameColumnExpectation(DataFrameExpectation):
    """
    Base class for DataFrame column expectations.
    This class is designed to validate a specific column in a DataFrame against a condition defined by
    `fn_violations_pandas` and `fn_violations_pyspark` functions."""

    def __init__(
        self,
        expectation_name: str,
        column_name: str,
        fn_violations_pandas: Callable,
        fn_violations_pyspark: Callable,
        description: str,
        error_message: str,
        tags: Optional[List[str]] = None,
    ):
        """
        Template for implementing DataFrame column expectations, where a column value is tested against a
        condition. The conditions are defined by the `fn_violations_pandas` and `fn_violations_pyspark` functions.

        :param expectation_name: The name of the expectation. This will be used during logging.
        :param column_name: The name of the column to check.
        :param fn_violations_pandas: Function to find violations in a pandas DataFrame.
        :param fn_violations_pyspark: Function to find violations in a PySpark DataFrame.
        :param description: A description of the expectation used in logging.
        :param error_message: The error message to return if the expectation fails.
        :param tags: Optional tags as list of strings in "key:value" format.
                    Example: ["priority:high", "env:test"]
        """
        super().__init__(tags=tags)
        self.column_name = column_name
        self.expectation_name = expectation_name
        self.fn_violations_pandas = fn_violations_pandas
        self.fn_violations_pyspark = fn_violations_pyspark
        self.description = description
        self.error_message = error_message

    def get_expectation_name(self) -> str:
        """
        Returns the expectation name.
        """
        return self.expectation_name

    def get_description(self) -> str:
        """
        Returns a description of the expectation.
        """
        return self.description

    def row_validation(
        self,
        data_frame_type: DataFrameType,
        data_frame: DataFrameLike,
        fn_violations: Callable,
        **kwargs,
    ) -> DataFrameExpectationResultMessage:
        """
        Validate the DataFrame against the expectation.

        :param data_frame_type: The type of DataFrame (Pandas or PySpark).
        :param data_frame: The DataFrame to validate.
        :param fn_violations: The function to find violations.
        :return: ExpectationResultMessage indicating success or failure.
        """

        if self.column_name not in data_frame.columns:
            return DataFrameExpectationFailureMessage(
                expectation_str=str(self),
                data_frame_type=data_frame_type,
                message=f"Column '{self.column_name}' does not exist in the DataFrame.",
            )

        violations = fn_violations(data_frame)

        # calculate number of violations based on DataFrame type
        num_violations = self.num_data_frame_rows(violations)

        if num_violations == 0:
            return DataFrameExpectationSuccessMessage(expectation_name=self.get_expectation_name())

        return DataFrameExpectationFailureMessage(
            expectation_str=str(self),
            data_frame_type=data_frame_type,
            violations_data_frame=violations,
            message=f"Found {num_violations} row(s) where {self.error_message}",
        )

    def validate_pandas(
        self, data_frame: DataFrameLike, **kwargs
    ) -> DataFrameExpectationResultMessage:
        return self.row_validation(
            data_frame_type=DataFrameType.PANDAS,
            data_frame=data_frame,
            fn_violations=self.fn_violations_pandas,
            **kwargs,
        )

    def validate_pyspark(
        self, data_frame: DataFrameLike, **kwargs
    ) -> DataFrameExpectationResultMessage:
        return self.row_validation(
            data_frame_type=DataFrameType.PYSPARK,
            data_frame=data_frame,
            fn_violations=self.fn_violations_pyspark,
            **kwargs,
        )
