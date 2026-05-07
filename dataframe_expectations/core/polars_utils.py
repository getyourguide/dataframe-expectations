from functools import lru_cache
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from polars import DataFrame as PolarsDataFrame
else:
    PolarsDataFrame = Any


class _MissingPolarsFunctions:
    def __getattr__(self, name: str) -> Any:
        raise ImportError(
            "Polars is required for Polars validation paths. "
            "Install polars or use a runtime that provides polars."
        )


@lru_cache(maxsize=1)
def get_polars_functions():
    """Return the polars module or a proxy raising a clear ImportError on use."""
    try:
        import polars as pl

        return pl
    except ImportError:
        return _MissingPolarsFunctions()


def is_polars_data_frame(data_frame: Any) -> bool:
    """Return True when the input is a Polars DataFrame."""
    pl = get_polars_functions()
    try:
        return isinstance(data_frame, pl.DataFrame)
    except ImportError:
        return False
