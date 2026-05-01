from functools import lru_cache
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from polars import DataFrame as PolarsDataFrame
else:
    PolarsDataFrame = Any


@lru_cache(maxsize=1)
def get_polars_functions():
    """Lazy import of polars module."""
    try:
        import polars as pl

        return pl
    except ImportError:
        return None


def is_polars_data_frame(data_frame: Any) -> bool:
    """Check if the given data frame is a Polars DataFrame."""
    pl = get_polars_functions()
    return pl is not None and isinstance(data_frame, pl.DataFrame)
