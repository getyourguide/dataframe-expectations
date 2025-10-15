from typing import Callable, Dict

from dataframe_expectations.expectations import DataframeExpectation
from dataframe_expectations.logging_utils import setup_logger

logger = setup_logger(__name__)


class DataframeExpectationRegistry:
    """Registry for dataframe expectations."""

    _expectations: Dict[str, Callable[..., DataframeExpectation]] = {}
    _loaded: bool = False

    @classmethod
    def register(cls, name: str):
        """Decorator to register an expectation factory function."""

        def decorator(func: Callable[..., DataframeExpectation]):
            logger.debug(f"Registering expectation '{name}' with function {func.__name__}")

            # check if the name is already registered
            if name in cls._expectations:
                error_message = f"Expectation '{name}' is already registered."
                logger.error(error_message)
                raise ValueError(error_message)

            cls._expectations[name] = func
            return func

        return decorator

    @classmethod
    def _ensure_loaded(cls):
        """Ensure all expectation modules are loaded (lazy loading)."""
        if not cls._loaded:
            cls._load_all_expectations()
            cls._loaded = True

    @classmethod
    def _load_all_expectations(cls):
        """Load all expectation modules to ensure their decorators are executed."""
        import importlib

        # Automatically discover all Python modules in expectations subdirectories
        # Explicitly import expectation modules
        modules_to_import = [
            "dataframe_expectations.expectations.column_expectations.null_expectation",
            "dataframe_expectations.expectations.column_expectations.type_expectation",
            "dataframe_expectations.expectations.column_expectations.any_value_expectations",
            "dataframe_expectations.expectations.column_expectations.numerical_expectations",
            "dataframe_expectations.expectations.column_expectations.string_expectations",
            "dataframe_expectations.expectations.aggregation_expectations.count_expectation",
            "dataframe_expectations.expectations.aggregation_expectations.sum_expectation",
            "dataframe_expectations.expectations.aggregation_expectations.any_value_expectations",
            "dataframe_expectations.expectations.aggregation_expectations.numerical_expectations",
            "dataframe_expectations.expectations.aggregation_expectations.unique",
            # Add more modules as needed
        ]

        for module_name in modules_to_import:
            try:
                importlib.import_module(module_name)
                logger.debug(f"Loaded expectation module: {module_name}")
            except ImportError as e:
                logger.warning(f"Failed to import expectation module {module_name}: {e}")

    @classmethod
    def get_expectation(cls, expectation_name: str, **kwargs) -> DataframeExpectation:
        """Get an expectation instance by name."""
        cls._ensure_loaded()  # Lazy load expectations
        logger.debug(f"Retrieving expectation '{expectation_name}' with arguments: {kwargs}")
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
        cls._ensure_loaded()  # Lazy load expectations
        return list(cls._expectations.keys())

    @classmethod
    def remove_expectation(cls, expectation_name: str):
        """Remove an expectation from the registry."""
        cls._ensure_loaded()  # Lazy load expectations
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
        logger.debug(f"Clearing {len(cls._expectations)} expectations from the registry")
        cls._expectations.clear()
        cls._loaded = False  # Allow reloading


# Convenience decorator
register_expectation = DataframeExpectationRegistry.register
