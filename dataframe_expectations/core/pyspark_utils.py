"""Utilities for optional PySpark imports and runtime type checks."""

import warnings
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Tuple

if TYPE_CHECKING:
    from pyspark.sql import DataFrame as PySparkDataFrame
else:
    PySparkDataFrame = Any

_pyspark_deprecation_warned = False


class _MissingPySparkFunctions:
    def __getattr__(self, name: str) -> Any:
        raise ImportError(
            "PySpark is required for PySpark validation paths. "
            "Install pyspark or use a runtime that provides pyspark."
        )


@lru_cache(maxsize=1)
def get_pyspark_functions() -> Any:
    """Return `pyspark.sql.functions` or a proxy raising a clear ImportError on use."""
    try:
        from pyspark.sql import functions as F

        return F
    except ImportError:
        return _MissingPySparkFunctions()


@lru_cache(maxsize=1)
def _get_pyspark_dataframe_types() -> Tuple[type, ...]:
    data_frame_types = []

    try:
        from pyspark.sql import DataFrame as PySparkDataFrameType

        data_frame_types.append(PySparkDataFrameType)
    except ImportError:
        pass

    try:
        from pyspark.sql.connect.dataframe import DataFrame as PySparkConnectDataFrameType

        data_frame_types.append(PySparkConnectDataFrameType)
    except ImportError:
        pass

    return tuple(data_frame_types)


def is_pyspark_data_frame(data_frame: Any) -> bool:
    """Return True when the input is a classic or connect PySpark DataFrame."""
    global _pyspark_deprecation_warned

    data_frame_types = _get_pyspark_dataframe_types()
    result = (data_frame_types and isinstance(data_frame, data_frame_types)) or (
        type(data_frame).__module__.startswith("pyspark.sql")
    )

    if result and not _pyspark_deprecation_warned:
        _pyspark_deprecation_warned = True
        warnings.warn(
            "In a future release, PySpark will be an optional dependency of dataframe-expectations. "
            "To avoid breakage, either: "
            "(1) install PySpark separately: pip install pyspark, "
            "or (2) install dataframe-expectations with PySpark included: "
            "pip install 'dataframe-expectations[pyspark]'.",
            FutureWarning,
            stacklevel=2,
        )

    return result
