from abc import abstractmethod
from typing import List, Union

from dataframe_expectations import DataFrameLike, DataFrameType
from dataframe_expectations.expectations import DataframeExpectation
from dataframe_expectations.result_message import (
    DataframeExpectationFailureMessage,
    DataframeExpectationResultMessage,
)


class DataframeAggregationExpectation(DataframeExpectation):
    """
    Base class for DataFrame aggregation expectations.
    This class is designed to first aggregate data and then validate the aggregation results.
    """

    def __init__(
        self,
        expectation_name: str,
        column_names: List[str],
        description: str,
    ):
        """
        Template for implementing DataFrame aggregation expectations, where data is first aggregated
        and then the aggregation results are validated.

        :param expectation_name: The name of the expectation. This will be used during logging.
        :param column_names: The list of column names to aggregate on.
        :param description: A description of the expectation used in logging.
        """
        self.expectation_name = expectation_name
        self.column_names = column_names
        self.description = description

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

    @abstractmethod
    def aggregate_and_validate_pandas(
        self, data_frame: DataFrameLike, **kwargs
    ) -> DataframeExpectationResultMessage:
        """
        Aggregate and validate a pandas DataFrame against the expectation.

        Note: This method should NOT check for column existence - that's handled
        automatically by the validate_pandas method.
        """
        raise NotImplementedError(
            f"aggregate_and_validate_pandas method must be implemented for {self.__class__.__name__}"
        )

    @abstractmethod
    def aggregate_and_validate_pyspark(
        self, data_frame: DataFrameLike, **kwargs
    ) -> DataframeExpectationResultMessage:
        """
        Aggregate and validate a PySpark DataFrame against the expectation.

        Note: This method should NOT check for column existence - that's handled
        automatically by the validate_pyspark method.
        """
        raise NotImplementedError(
            f"aggregate_and_validate_pyspark method must be implemented for {self.__class__.__name__}"
        )

    def validate_pandas(
        self, data_frame: DataFrameLike, **kwargs
    ) -> DataframeExpectationResultMessage:
        """
        Validate a pandas DataFrame against the expectation.
        Automatically checks column existence before calling the implementation.
        """
        # Check if all required columns exist
        column_error = self._check_columns_exist(data_frame)
        if column_error:
            return DataframeExpectationFailureMessage(
                expectation_str=str(self),
                data_frame_type=DataFrameType.PANDAS,
                message=column_error,
            )

        # Call the implementation-specific validation
        return self.aggregate_and_validate_pandas(data_frame, **kwargs)

    def validate_pyspark(
        self, data_frame: DataFrameLike, **kwargs
    ) -> DataframeExpectationResultMessage:
        """
        Validate a PySpark DataFrame against the expectation.
        Automatically checks column existence before calling the implementation.
        """
        # Check if all required columns exist
        column_error = self._check_columns_exist(data_frame)
        if column_error:
            return DataframeExpectationFailureMessage(
                expectation_str=str(self),
                data_frame_type=DataFrameType.PYSPARK,
                message=column_error,
            )

        # Call the implementation-specific validation
        return self.aggregate_and_validate_pyspark(data_frame, **kwargs)

    def _check_columns_exist(self, data_frame: DataFrameLike) -> Union[str, None]:
        """
        Check if all required columns exist in the DataFrame.
        Returns error message if columns are missing, None otherwise.
        """
        # Skip column check if no columns are required (e.g., for DataFrame-level expectations)
        if not self.column_names:
            return None

        missing_columns = [col for col in self.column_names if col not in data_frame.columns]
        if missing_columns:
            if len(missing_columns) == 1:
                return f"Column '{missing_columns[0]}' does not exist in the DataFrame."
            else:
                missing_columns_str = ", ".join([f"'{col}'" for col in missing_columns])
                return f"Columns [{missing_columns_str}] do not exist in the DataFrame."
        return None
