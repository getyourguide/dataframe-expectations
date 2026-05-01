import pytest
from unittest.mock import MagicMock, patch

from dataframe_expectations.core.column_expectation import DataFrameColumnExpectation
from dataframe_expectations.core.types import DataFrameLike, DataFrameType
from dataframe_expectations.core.expectation import DataFrameExpectation
from dataframe_expectations.core.tagging import TagSet
from dataframe_expectations.registry import DataFrameExpectationRegistry


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

    def validate_polars(self, data_frame: DataFrameLike, **kwargs):
        """
        Mock implementation for Polars DataFrame validation.
        """
        return "polars validation successful"

    def get_description(self):
        return "This is a test expectation for unit testing purposes."


class MockConnectDataFrame:
    """Mock class to simulate pyspark.sql.connect.dataframe.DataFrame"""

    def __init__(self):
        pass


class FakeDataFrame:
    def count(self):
        return 5

    def collect(self):
        return []


def test_data_frame_type_enum():
    """DataFrameType enum has the correct string values and supports direct string comparison."""
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
    """get_expectation_name() returns the class name."""
    expectation = MyTestExpectation()
    assert expectation.get_expectation_name() == "MyTestExpectation", (
        f"Expected 'MyTestExpectation' but got: {expectation.get_expectation_name()}"
    )


def test_validate_unsupported_dataframe_type():
    """validate() raises ValueError for unsupported DataFrame types."""
    expectation = MyTestExpectation()
    with pytest.raises(ValueError):
        expectation.validate(None)


def test_validate_called(dataframe_factory):
    """validate() dispatches to validate_pandas or validate_pyspark depending on the DataFrame type."""
    df_lib, make_df = dataframe_factory
    expectation = MyTestExpectation()

    match df_lib:
        case DataFrameType.PANDAS:
            expectation.validate_pandas = MagicMock(return_value="mock_result")
        case DataFrameType.PYSPARK:
            expectation.validate_pyspark = MagicMock(return_value="mock_result")

    data_frame = make_df({"col1": ([1, 2, 3], "long"), "col2": (["a", "b", "c"], "string")})
    _ = expectation.validate(data_frame=data_frame)

    match df_lib:
        case DataFrameType.PANDAS:
            expectation.validate_pandas.assert_called_once_with(data_frame=data_frame)
        case DataFrameType.PYSPARK:
            expectation.validate_pyspark.assert_called_once_with(data_frame=data_frame)

    with pytest.raises(ValueError):
        expectation.validate(None)


def test_num_data_frame_rows(dataframe_factory):
    """num_data_frame_rows() returns the correct row count for both empty and non-empty DataFrames."""
    df_lib, make_df = dataframe_factory
    expectation = MyTestExpectation()

    # 1. Non-empty DataFrame
    data_frame = make_df({"col1": ([1, 2, 3], "long"), "col2": (["a", "b", "c"], "string")})
    num_rows = expectation.num_data_frame_rows(data_frame)
    assert num_rows == 3, f"Expected 3 rows but got: {num_rows}"

    with pytest.raises(ValueError):
        expectation.num_data_frame_rows(None)

    # 2. Empty DataFrame
    empty_df = make_df({"col1": ([], "long"), "col2": ([], "string")})
    num_rows = expectation.num_data_frame_rows(empty_df)
    assert num_rows == 0, f"Expected 0 rows for empty DataFrame but got: {num_rows}"

    with pytest.raises(ValueError):
        expectation.num_data_frame_rows(None)


def test_infer_data_frame_type(dataframe_factory):
    """infer_data_frame_type() correctly identifies the library for both empty and non-empty DataFrames."""
    df_lib, make_df = dataframe_factory
    expectation = MyTestExpectation()

    # Non-empty DataFrame
    data_frame = make_df({"col1": ([1, 2, 3], "long"), "col2": (["a", "b", "c"], "string")})
    data_frame_type = expectation.infer_data_frame_type(data_frame)
    assert data_frame_type == df_lib, f"Expected {df_lib} type but got: {data_frame_type}"

    # Empty DataFrame
    empty_df = make_df({"col1": ([], "long"), "col2": ([], "string")})
    data_frame_type = expectation.infer_data_frame_type(empty_df)
    assert data_frame_type == df_lib, (
        f"Expected {df_lib} type for empty DataFrame but got: {data_frame_type}"
    )


@pytest.mark.parametrize(
    "invalid_input",
    [
        None,
        "not_a_dataframe",
        [1, 2, 3],
        {"col1": [1, 2, 3]},
        FakeDataFrame(),
        42,
        True,
    ],
)
def test_infer_data_frame_type_invalid(invalid_input):
    """infer_data_frame_type() raises ValueError for non-DataFrame inputs."""
    expectation = MyTestExpectation()
    with pytest.raises(ValueError, match="Unsupported DataFrame type"):
        expectation.infer_data_frame_type(invalid_input)


@pytest.mark.pyspark
def test_infer_data_frame_type_with_connect_dataframe_available():
    """PySpark Connect DataFrame is identified as PYSPARK when the Connect module is available."""
    expectation = MyTestExpectation()

    # Patch the PySparkConnectDataFrame import to be our mock class
    with patch(
        "dataframe_expectations.core.expectation.PySparkConnectDataFrame",
        MockConnectDataFrame,
    ):
        # Create an instance of our mock Connect DataFrame
        mock_connect_df = MockConnectDataFrame()

        # Test that Connect DataFrame is identified as PYSPARK type
        data_frame_type = expectation.infer_data_frame_type(mock_connect_df)
        assert data_frame_type == DataFrameType.PYSPARK, (
            f"Expected PYSPARK type for Connect DataFrame but got: {data_frame_type}"
        )


@patch("dataframe_expectations.core.expectation.PySparkConnectDataFrame", None)
def test_infer_data_frame_type_without_connect_support(dataframe_factory):
    """infer_data_frame_type() still works correctly when PySpark Connect is unavailable."""
    df_lib, make_df = dataframe_factory
    expectation = MyTestExpectation()

    data_frame = make_df({"col1": ([1, 2, 3], "long")})
    data_frame_type = expectation.infer_data_frame_type(data_frame)
    assert data_frame_type == df_lib, f"Expected {df_lib} type but got: {data_frame_type}"


def test_infer_data_frame_type_connect_import_behavior(dataframe_factory):
    """infer_data_frame_type() handles both presence and absence of the PySpark Connect module."""
    df_lib, make_df = dataframe_factory
    expectation = MyTestExpectation()

    # Test case 1: When PySparkConnectDataFrame is None (import failed)
    with patch("dataframe_expectations.core.expectation.PySparkConnectDataFrame", None):
        data_frame = make_df({"col1": ([1, 2, 3], "long")})
        result_type = expectation.infer_data_frame_type(data_frame)
        assert result_type == df_lib, f"Expected {df_lib} type but got: {result_type}"

    # Test case 2: When PySparkConnectDataFrame is available (mocked)
    with patch(
        "dataframe_expectations.core.expectation.PySparkConnectDataFrame",
        MockConnectDataFrame,
    ):
        data_frame = make_df({"col1": ([1, 2, 3], "long")})
        result_type = expectation.infer_data_frame_type(data_frame)
        assert result_type == df_lib, f"Expected {df_lib} type but got: {result_type}"

        # Mock Connect DataFrame should be identified as PYSPARK
        mock_connect_df = MockConnectDataFrame()
        result_type = expectation.infer_data_frame_type(mock_connect_df)
        assert result_type == DataFrameType.PYSPARK, (
            f"Expected PYSPARK type for Connect DataFrame but got: {result_type}"
        )


@pytest.mark.parametrize(
    "tag_list, expected_count, expected_empty",
    [
        (None, 0, True),
        ([], 0, True),
        (["priority:high"], 1, False),
        (["priority:high", "env:prod"], 2, False),
        (["priority:high", "env:prod", "team:data", "critical:true"], 4, False),
        (["priority:high", "priority:high", "env:prod"], 2, False),  # Deduplication
    ],
)
def test_tags_initialization(tag_list, expected_count, expected_empty):
    """Test that DataFrameExpectation properly initializes tags."""
    expectation = MyTestExpectation(tags=tag_list)
    tags = expectation.get_tags()

    assert isinstance(tags, TagSet)
    assert len(tags) == expected_count
    assert tags.is_empty() == expected_empty


def test_tags_propagation_to_subclass():
    """Test that tags are properly propagated to subclasses."""

    class MySubclassExpectation(MyTestExpectation):
        def __init__(self, custom_param: str, tags=None):
            super().__init__(tags=tags)
            self.custom_param = custom_param

    expectation_no_tags = MySubclassExpectation(custom_param="test")
    assert expectation_no_tags.get_tags().is_empty()

    expectation_with_tags = MySubclassExpectation(custom_param="test", tags=["priority:high"])
    assert len(expectation_with_tags.get_tags()) == 1


def test_tags_immutability():
    """Test that get_tags() returns the same TagSet instance."""
    expectation = MyTestExpectation(tags=["priority:high"])
    tags1 = expectation.get_tags()
    tags2 = expectation.get_tags()

    assert isinstance(tags1, TagSet)
    assert tags1 is tags2


def test_tags_with_invalid_format():
    """Test that invalid tag formats raise appropriate errors."""
    with pytest.raises(ValueError, match="Invalid tag format"):
        MyTestExpectation(tags=["invalid-tag-no-colon"])


def test_tags_sent_to_base_class_direct():
    """Tags propagate correctly when DataFrameColumnExpectation is instantiated directly.

    DataFrameColumnExpectation is not registered in the registry (it is a base class for
    factory-created column expectations), so it is not covered by test_tags_sent_to_base_class.
    Aggregation expectations are covered by test_tags_sent_to_base_class via the registry.
    """
    test_tags = ["priority:high", "env:test"]
    expectation = DataFrameColumnExpectation(
        expectation_name="TestColumn",
        column_name="test",
        fn_violations_pandas=lambda df: df,
        fn_violations_pyspark=lambda df: df,
        fn_violations_polars=lambda df: df,
        description="Test",
        error_message="Error",
        tags=test_tags,
    )
    assert len(expectation.get_tags()) == 2


# Maps Python types to sensible dummy values for constructing expectations in tests.
_TYPE_DEFAULTS: dict = {str: "col", int: 10, float: 0.5, list: ["col1", "col2"]}


def _build_kwargs(metadata) -> dict:
    def _dummy(param: str):
        raw = metadata.param_types.get(param, int)
        types = raw if isinstance(raw, tuple) else (raw,)
        # prefer float over int when both are valid (e.g. quantile, max_percentage)
        typ = float if float in types else types[0]
        return _TYPE_DEFAULTS.get(typ, 10)

    return {p: _dummy(p) for p in metadata.params}


DataFrameExpectationRegistry._ensure_loaded()


@pytest.fixture(
    params=[
        pytest.param((factory, _build_kwargs(metadata)), id=metadata.expectation_name)
        for _, (factory, metadata) in DataFrameExpectationRegistry._registry.items()
    ]
)
def registry_expectation(request):
    return request.param


def test_tags_sent_to_base_class(registry_expectation):
    """Every registered expectation correctly propagates tags to the base class.

    New expectations are picked up automatically from the registry —
    no manual list to maintain. If _build_kwargs fails, add the new
    param type to _TYPE_DEFAULTS.
    """
    factory, kwargs = registry_expectation
    test_tags = ["priority:high", "env:test"]
    expectation = factory(**kwargs, tags=test_tags)
    assert len(expectation.get_tags()) == 2
