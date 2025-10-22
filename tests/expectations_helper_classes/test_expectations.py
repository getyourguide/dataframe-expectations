import pytest
from unittest.mock import MagicMock, patch

import pandas as pd

from dataframe_expectations import DataFrameLike, DataFrameType
from dataframe_expectations.expectations import DataFrameExpectation


class MyTestExpectation(DataFrameExpectation):
    def validate_pandas(self, data_frame: DataFrameLike, **kwargs):
        """
        Mock implementation for pandas DataFrame validation.
        """
        return "pandas validation successful"

    def validate_pyspark(self, data_frame: DataFrameLike, **kwargs):
        """
        Mock implementation for PySpark DataFrame validation.
        """
        return "pyspark validation successful"

    def get_description(self):
        return "This is a test expectation for unit testing purposes."


class MockConnectDataFrame:
    """Mock class to simulate pyspark.sql.connect.dataframe.DataFrame"""

    def __init__(self):
        pass


def test_data_frame_type_enum():
    """
    Test that the DataFrameType enum has the correct values.
    """
    assert DataFrameType.PANDAS.value == "pandas", (
        f"Expected 'pandas' but got: {DataFrameType.PANDAS.value}"
    )
    assert DataFrameType.PYSPARK.value == "pyspark", (
        f"Expected 'pyspark' but got: {DataFrameType.PYSPARK.value}"
    )

    # Test string comparison (now works directly!)
    assert DataFrameType.PANDAS == "pandas", "Expected DataFrameType.PANDAS == 'pandas' to be True"
    assert DataFrameType.PYSPARK == "pyspark", (
        "Expected DataFrameType.PYSPARK == 'pyspark' to be True"
    )


def test_get_expectation_name():
    """
    Test that the expectation name is the class name.
    """
    expectation = MyTestExpectation()
    assert expectation.get_expectation_name() == "MyTestExpectation", (
        f"Expected 'MyTestExpectation' but got: {expectation.get_expectation_name()}"
    )


def test_validate_unsupported_dataframe_type():
    """
    Test that an error is raised for unsupported DataFrame types.
    """
    expectation = MyTestExpectation()
    with pytest.raises(ValueError):
        expectation.validate(None)


def test_validate_pandas_called():
    """
    Test that validate_pandas method is called and with right parameters.
    """
    expectation = MyTestExpectation()

    # Mock the validate_pandas method
    expectation.validate_pandas = MagicMock(return_value="mock_result")

    # Assert that validate_pandas was called with the correct arguments
    data_frame = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
    _ = expectation.validate(data_frame=data_frame)
    expectation.validate_pandas.assert_called_once_with(data_frame=data_frame)

    with pytest.raises(ValueError):
        expectation.validate(None)


def test_validate_pyspark_called(spark):
    """
    Test that validate_pyspark method is called with right parameters.
    """
    expectation = MyTestExpectation()

    # Mock the validate_pyspark method
    expectation.validate_pyspark = MagicMock(return_value="mock_result")

    # Assert that validate_pyspark was called with the correct arguments
    data_frame = spark.createDataFrame([(1, "a"), (2, "b"), (3, "c")], ["col1", "col2"])
    _ = expectation.validate(data_frame=data_frame)
    expectation.validate_pyspark.assert_called_once_with(data_frame=data_frame)

    with pytest.raises(ValueError):
        expectation.validate(None)


def test_num_data_frame_rows(spark):
    """
    Test that the number of rows in a DataFrame are counted correctly.
    """
    expectation = MyTestExpectation()

    # 1. Non empty DataFrames
    # Mock a pandas DataFrame
    pandas_df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
    num_rows = expectation.num_data_frame_rows(pandas_df)
    assert num_rows == 3, f"Expected 3 rows for pandas DataFrame but got: {num_rows}"

    # Mock a PySpark DataFrame
    spark_df = spark.createDataFrame([(1, "a"), (2, "b"), (3, "c")], ["col1", "col2"])
    num_rows = expectation.num_data_frame_rows(spark_df)
    assert num_rows == 3, f"Expected 3 rows for PySpark DataFrame but got: {num_rows}"

    # Test unsupported DataFrame type
    with pytest.raises(ValueError):
        expectation.num_data_frame_rows(None)

    # 2. Empty DataFrames
    # Mock an empty pandas DataFrame
    empty_pandas_df = pd.DataFrame(columns=["col1", "col2"])
    num_rows = expectation.num_data_frame_rows(empty_pandas_df)
    assert num_rows == 0, f"Expected 0 rows for empty pandas DataFrame but got: {num_rows}"

    # Mock an empty PySpark DataFrame
    empty_spark_df = spark.createDataFrame([], "col1 INT, col2 STRING")
    num_rows = expectation.num_data_frame_rows(empty_spark_df)
    assert num_rows == 0, f"Expected 0 rows for empty PySpark DataFrame but got: {num_rows}"

    # Test unsupported DataFrame type
    with pytest.raises(ValueError):
        expectation.num_data_frame_rows(None)


def test_infer_data_frame_type(spark):
    """
    Test that the DataFrame type is inferred correctly for all supported DataFrame types.
    """
    expectation = MyTestExpectation()

    # Test pandas DataFrame
    pandas_df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
    data_frame_type = expectation.infer_data_frame_type(pandas_df)
    assert data_frame_type == DataFrameType.PANDAS, (
        f"Expected PANDAS type but got: {data_frame_type}"
    )

    # Test PySpark DataFrame
    spark_df = spark.createDataFrame([(1, "a"), (2, "b"), (3, "c")], ["col1", "col2"])
    data_frame_type = expectation.infer_data_frame_type(spark_df)
    assert data_frame_type == DataFrameType.PYSPARK, (
        f"Expected PYSPARK type but got: {data_frame_type}"
    )

    # Test empty pandas DataFrame
    empty_pandas_df = pd.DataFrame(columns=["col1", "col2"])
    data_frame_type = expectation.infer_data_frame_type(empty_pandas_df)
    assert data_frame_type == DataFrameType.PANDAS, (
        f"Expected PANDAS type for empty DataFrame but got: {data_frame_type}"
    )

    # Test empty PySpark DataFrame
    empty_spark_df = spark.createDataFrame([], "col1 INT, col2 STRING")
    data_frame_type = expectation.infer_data_frame_type(empty_spark_df)
    assert data_frame_type == DataFrameType.PYSPARK, (
        f"Expected PYSPARK type for empty DataFrame but got: {data_frame_type}"
    )

    # Test unsupported DataFrame types
    with pytest.raises(ValueError) as context:
        expectation.infer_data_frame_type(None)
    assert "Unsupported DataFrame type" in str(context.value), (
        f"Expected 'Unsupported DataFrame type' in error message but got: {str(context.value)}"
    )

    with pytest.raises(ValueError) as context:
        expectation.infer_data_frame_type("not_a_dataframe")
    assert "Unsupported DataFrame type" in str(context.value), (
        f"Expected 'Unsupported DataFrame type' in error message but got: {str(context.value)}"
    )

    with pytest.raises(ValueError) as context:
        expectation.infer_data_frame_type([1, 2, 3])
    assert "Unsupported DataFrame type" in str(context.value), (
        f"Expected 'Unsupported DataFrame type' in error message but got: {str(context.value)}"
    )

    with pytest.raises(ValueError) as context:
        expectation.infer_data_frame_type({"col1": [1, 2, 3]})
    assert "Unsupported DataFrame type" in str(context.value), (
        f"Expected 'Unsupported DataFrame type' in error message but got: {str(context.value)}"
    )

    # Test with objects that might have similar attributes but aren't DataFrames
    class FakeDataFrame:
        def count(self):
            return 5

        def collect(self):
            return []

    fake_df = FakeDataFrame()
    with pytest.raises(ValueError):
        expectation.infer_data_frame_type(fake_df)

    # Test with numeric types
    with pytest.raises(ValueError):
        expectation.infer_data_frame_type(42)

    # Test with boolean
    with pytest.raises(ValueError):
        expectation.infer_data_frame_type(True)


def test_infer_data_frame_type_with_connect_dataframe_available():
    """
    Test that PySpark Connect DataFrame is correctly identified when available.
    """
    expectation = MyTestExpectation()

    # Patch the PySparkConnectDataFrame import to be our mock class
    with patch(
        "dataframe_expectations.expectations.PySparkConnectDataFrame",
        MockConnectDataFrame,
    ):
        # Create an instance of our mock Connect DataFrame
        mock_connect_df = MockConnectDataFrame()

        # Test that Connect DataFrame is identified as PYSPARK type
        data_frame_type = expectation.infer_data_frame_type(mock_connect_df)
        assert data_frame_type == DataFrameType.PYSPARK, (
            f"Expected PYSPARK type for Connect DataFrame but got: {data_frame_type}"
        )


@patch("dataframe_expectations.expectations.PySparkConnectDataFrame", None)
def test_infer_data_frame_type_without_connect_support(spark):
    """
    Test that the method works correctly when PySpark Connect is not available.
    """
    expectation = MyTestExpectation()

    # Test that regular DataFrames still work when Connect is not available
    pandas_df = pd.DataFrame({"col1": [1, 2, 3]})
    data_frame_type = expectation.infer_data_frame_type(pandas_df)
    assert data_frame_type == DataFrameType.PANDAS, (
        f"Expected PANDAS type but got: {data_frame_type}"
    )

    spark_df = spark.createDataFrame([(1,), (2,), (3,)], ["col1"])
    data_frame_type = expectation.infer_data_frame_type(spark_df)
    assert data_frame_type == DataFrameType.PYSPARK, (
        f"Expected PYSPARK type but got: {data_frame_type}"
    )


def test_infer_data_frame_type_connect_import_behavior(spark):
    """
    Test that the Connect DataFrame import behavior works as expected.
    """
    expectation = MyTestExpectation()

    # Test case 1: When PySparkConnectDataFrame is None (import failed)
    with patch("dataframe_expectations.expectations.PySparkConnectDataFrame", None):
        # Should still work with regular DataFrames
        pandas_df = pd.DataFrame({"col1": [1, 2, 3]})
        result_type = expectation.infer_data_frame_type(pandas_df)
        assert result_type == DataFrameType.PANDAS, f"Expected PANDAS type but got: {result_type}"

        spark_df = spark.createDataFrame([(1,), (2,), (3,)], ["col1"])
        result_type = expectation.infer_data_frame_type(spark_df)
        assert result_type == DataFrameType.PYSPARK, f"Expected PYSPARK type but got: {result_type}"

    # Test case 2: When PySparkConnectDataFrame is available (mocked)
    with patch(
        "dataframe_expectations.expectations.PySparkConnectDataFrame",
        MockConnectDataFrame,
    ):
        # Regular DataFrames should still work
        pandas_df = pd.DataFrame({"col1": [1, 2, 3]})
        result_type = expectation.infer_data_frame_type(pandas_df)
        assert result_type == DataFrameType.PANDAS, f"Expected PANDAS type but got: {result_type}"

        spark_df = spark.createDataFrame([(1,), (2,), (3,)], ["col1"])
        result_type = expectation.infer_data_frame_type(spark_df)
        assert result_type == DataFrameType.PYSPARK, f"Expected PYSPARK type but got: {result_type}"

        # Mock Connect DataFrame should be identified as PYSPARK
        mock_connect_df = MockConnectDataFrame()
        result_type = expectation.infer_data_frame_type(mock_connect_df)
        assert result_type == DataFrameType.PYSPARK, (
            f"Expected PYSPARK type for Connect DataFrame but got: {result_type}"
        )
