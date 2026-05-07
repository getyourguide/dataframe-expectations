from typing import Any

import pandas as pd
import pandas.testing as pdt
import pyarrow as pa
import pytest

from dataframe_expectations.core.types import DataFrameType

# ---------------------------------------------------------------------------
# PyArrow type shorthands
# ---------------------------------------------------------------------------
# Use these string keys in make_dataframe() to avoid importing pyarrow types
# in every test file. Add new entries here as you need more types.
ARROW_TYPES: dict[str, pa.DataType] = {
    "long": pa.int64(),
    "string": pa.utf8(),
    "double": pa.float64(),
    "boolean": pa.bool_(),
    "timestamp": pa.timestamp("us"),
    "timestamp_utc": pa.timestamp("us", tz="UTC"),
}


# ---------------------------------------------------------------------------
# DataFrame creation helpers
# ---------------------------------------------------------------------------


def arrow_to_df(table: pa.Table, df_lib: DataFrameType, spark: Any = None) -> Any:
    """Convert a PyArrow Table to the specified DataFrame library format.

    Args:
        table:   Source PyArrow Table.
        df_lib:  Target library as a DataFrameType enum value.
        spark:   Active SparkSession (required when df_lib=DataFrameType.PYSPARK).

    Returns:
        A DataFrame in the requested library format.
    """
    match df_lib:
        case DataFrameType.PANDAS:
            # Use nullable dtypes so int columns with None stay as Int64
            # instead of converting to float64
            return table.to_pandas(types_mapper=pd.ArrowDtype)
        case DataFrameType.PYSPARK:
            if spark is None:
                raise ValueError("A Spark session is required for df_lib=DataFrameType.PYSPARK")
            from pyspark.sql.pandas.types import from_arrow_schema

            pdf = table.to_pandas(integer_object_nulls=True)

            # Replace NaN with None in numeric columns so PySpark sees proper SQL NULLs.
            # IMPORTANT: Only apply to numeric columns. Calling pdf.replace() on the
            # whole DataFrame converts datetime64 columns to object dtype containing
            # pd.Timestamp objects, which PySpark rejects.
            for col in pdf.columns:
                if pd.api.types.is_numeric_dtype(pdf[col]):
                    pdf[col] = pdf[col].replace({float("nan"): None})

            return spark.createDataFrame(
                pdf,
                schema=from_arrow_schema(table.schema),
            )
        case DataFrameType.POLARS:
            import polars as pl

            return pl.from_arrow(table)
        case _:
            raise ValueError(f"Unsupported df_lib: {df_lib!r}")


def make_dataframe(
    columns: dict[str, tuple[list, "str | pa.DataType"]],
    df_lib: DataFrameType,
    spark: Any = None,
) -> Any:
    """Create a multi-column DataFrame from a dict of column definitions.

    Defining data once as plain Python lists and type strings lets you run the
    same test logic against every supported library without duplicating fixtures.

    Args:
        columns:  Mapping of column name to ``(data, arrow_type)`` where
                  ``arrow_type`` is a string key from ARROW_TYPES
                  (e.g. "long", "string", "double", "boolean", "timestamp")
                  or a raw PyArrow DataType.
        df_lib:   Target library as a DataFrameType enum value.
        spark:    Active SparkSession (required when df_lib=DataFrameType.PYSPARK).

    Returns:
        A DataFrame in the requested library format.

    Example::

        df = make_dataframe({"col1": ([1, 2, 3], "long")}, DataFrameType.PANDAS)
        df = make_dataframe(
            {"col1": ([1, 2, 3], "long"), "col2": (["a", "b", "c"], "string")},
            DataFrameType.PYSPARK,
            spark,
        )
    """
    arrays = {
        name: pa.array(
            data, type=ARROW_TYPES[arrow_type] if isinstance(arrow_type, str) else arrow_type
        )
        for name, (data, arrow_type) in columns.items()
    }
    table = pa.table(arrays)
    return arrow_to_df(table, df_lib, spark)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def spark():
    """Create a Spark session reused for every test in the module."""
    from pyspark.sql import SparkSession

    return SparkSession.builder.master("local").appName("Test").getOrCreate()


@pytest.fixture(
    params=[
        pytest.param(DataFrameType.PANDAS, marks=pytest.mark.pandas),
        pytest.param(DataFrameType.PYSPARK, marks=pytest.mark.pyspark),
        pytest.param(DataFrameType.POLARS, marks=pytest.mark.polars),
    ]
)
def dataframe_factory(request):
    """Fixture parametrized over every supported DataFrame library.

    Yields ``(df_lib, factory)`` where ``df_lib`` is a ``DataFrameType`` enum
    value and ``factory(columns)`` creates a DataFrame in that library.
    To add polars support, append ``DataFrameType.POLARS`` to the ``params``
    list above.

    Usage::

        def test_my_expectation(dataframe_factory):
            df_lib, make_df = dataframe_factory
            df = make_df({"col1": ([1, 2, 3], "long")})
            ...
    """
    lib: DataFrameType = request.param
    match lib:
        case DataFrameType.PYSPARK:
            pytest.importorskip("pyspark")
            _spark = request.getfixturevalue("spark")

            def _factory(columns: dict[str, tuple[list, "str | pa.DataType"]]) -> Any:
                return make_dataframe(columns, DataFrameType.PYSPARK, _spark)

        case DataFrameType.PANDAS:

            def _factory(columns: dict[str, tuple[list, "str | pa.DataType"]]) -> Any:
                return make_dataframe(columns, DataFrameType.PANDAS)
        case DataFrameType.POLARS:
            pytest.importorskip("polars")

            def _factory(columns: dict[str, tuple[list, "str | pa.DataType"]]) -> Any:
                return make_dataframe(columns, DataFrameType.POLARS)
        case _:
            raise ValueError(f"Unsupported df_lib: {lib!r}")

    return lib, _factory


# ---------------------------------------------------------------------------
# Assertion helpers (kept for backward compatibility)
# ---------------------------------------------------------------------------


def assert_pandas_df_equal(df1: pd.DataFrame, df2: pd.DataFrame) -> None:
    df1_sorted = df1.sort_values(by=df1.columns.tolist()).reset_index(drop=True)
    df2_sorted = df2.sort_values(by=df2.columns.tolist()).reset_index(drop=True)
    pdt.assert_frame_equal(df1_sorted, df2_sorted, check_dtype=False)


def assert_pyspark_df_equal(df1: Any, df2: Any) -> None:
    df1_pd = df1.toPandas().sort_values(by=df1.columns).reset_index(drop=True)
    df2_pd = df2.toPandas().sort_values(by=df2.columns).reset_index(drop=True)
    pd.testing.assert_frame_equal(df1_pd, df2_pd, check_dtype=False)
