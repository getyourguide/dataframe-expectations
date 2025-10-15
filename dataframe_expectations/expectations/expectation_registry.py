from typing import Any, Callable, Dict

from dataframe_expectations.expectations import DataframeExpectation
from dataframe_expectations.logging_utils import setup_logger

logger = setup_logger(__name__)


class DataframeExpectationRegistry:
    """Registry for dataframe expectations."""

    _expectations: Dict[str, Callable[..., DataframeExpectation]] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register an expectation factory function."""

        def decorator(func: Callable[..., DataframeExpectation]):
            logger.debug(
                f"Registering expectation '{name}' with function {func.__name__}"
            )

            # check if the name is already registered
            if name in cls._expectations:
                error_message = f"Expectation '{name}' is already registered."
                logger.error(error_message)
                raise ValueError(error_message)

            cls._expectations[name] = func
            return func

        return decorator

    @classmethod
    def get_expectation(cls, expectation_name: str, **kwargs) -> DataframeExpectation:
        """Get an expectation instance by name."""
        logger.debug(
            f"Retrieving expectation '{expectation_name}' with arguments: {kwargs}"
        )
        if expectation_name not in cls._expectations:
            available = cls.list_expectations()
            error_message = (
                f"Unknown expectation '{expectation_name}'. "
                f"Available expectations: {', '.join(available)}"
            )
            logger.error(error_message)
            raise ValueError(error_message)
        return cls._expectations[expectation_name](**kwargs)

    @classmethod
    def list_expectations(cls) -> list:
        """List all registered expectation names."""
        return list(cls._expectations.keys())

    @classmethod
    def remove_expectation(cls, expectation_name: str):
        """Remove an expectation from the registry."""
        logger.debug(f"Removing expectation '{expectation_name}'")
        if expectation_name in cls._expectations:
            del cls._expectations[expectation_name]
        else:
            error_message = f"Expectation '{expectation_name}' not found."
            logger.error(error_message)
            raise ValueError(error_message)

    @classmethod
    def clear_expectations(cls):
        """Clear all registered expectations."""
        logger.debug(
            f"Clearing {len(cls._expectations)} expectations from the registry"
        )
        cls._expectations.clear()


# Convenience decorator
register_expectation = DataframeExpectationRegistry.register

from dataframe_expectations.expectations.aggregation_expectations import (
    any_value_expectations as aggregation_any_value_expectations,
)
from dataframe_expectations.expectations.aggregation_expectations import (
    numerical_expectations as aggregation_numerical_expectations,
)
from dataframe_expectations.expectations.aggregation_expectations import unique

# Import expectation modules AFTER defining the registry
# This ensures the registry class exists when the decorators are applied
from dataframe_expectations.expectations.column_expectations import (
    any_value_expectations as column_any_value_expectations,
)
from dataframe_expectations.expectations.column_expectations import (
    numerical_expectations as column_numerical_expectations,
)
from dataframe_expectations.expectations.column_expectations import (
    string_expectations,
)
