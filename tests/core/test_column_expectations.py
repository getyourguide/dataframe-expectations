import pytest
from unittest.mock import MagicMock

from dataframe_expectations.core.types import DataFrameType
from dataframe_expectations.core.column_expectation import (
    DataFrameColumnExpectation,
)


@pytest.fixture
def expectation():
    return DataFrameColumnExpectation(
        expectation_name="MyColumnExpectation",
        column_name="test_column",
        fn_violations_pandas=lambda df: df,
        fn_violations_pyspark=lambda df: df,
        fn_violations_polars=lambda df: df,
        description="Test column expectation",
        error_message="Test column expectation failed.",
    )


def test_get_expectation_name(expectation):
    """
    Test that the expectation name is the class name.
    """
    assert expectation.get_expectation_name() == "MyColumnExpectation", (
        f"Expected 'MyColumnExpectation' but got: {expectation.get_expectation_name()}"
    )


def test_validate_calls_row_validation(expectation, dataframe_factory):
    """
    Test that validate() calls row_validation() with the correct parameters
    for each supported DataFrame library.
    """
    df_lib, make_df = dataframe_factory

    expectation.row_validation = MagicMock(return_value="mock_result")
    data_frame = make_df({"col1": ([1, 2, 3], "long"), "col2": (["a", "b", "c"], "string")})

    _ = expectation.validate(data_frame=data_frame)

    match df_lib:
        case DataFrameType.PYSPARK:
            fn_violations = expectation.fn_violations_pyspark
        case DataFrameType.PANDAS:
            fn_violations = expectation.fn_violations_pandas
        case DataFrameType.POLARS:
            fn_violations = expectation.fn_violations_polars
    expectation.row_validation.assert_called_once_with(
        data_frame_type=df_lib,
        data_frame=data_frame,
        fn_violations=fn_violations,
    )
