import re
from typing import Any, Callable, Dict, Optional, Tuple

from dataframe_expectations.core.expectation import DataFrameExpectation
from dataframe_expectations.core.types import (
    ExpectationCategory,
    ExpectationMetadata,
    ExpectationSubcategory,
)
from dataframe_expectations.logging_utils import setup_logger

logger = setup_logger(__name__)

# Type alias for registry entry (factory function + metadata)
FactoryFunction = Callable[..., DataFrameExpectation]
RegistryEntry = Tuple[FactoryFunction, ExpectationMetadata]


class DataFrameExpectationRegistry:
    """Registry for dataframe expectations."""

    # Primary registry: keyed by suite_method_name for O(1) suite access
    _registry: Dict[str, RegistryEntry] = {}

    # Secondary index: maps expectation_name -> suite_method_name for O(1) lookups
    _by_name: Dict[str, str] = {}

    _loaded: bool = False

    @classmethod
    def register(
        cls,
        name: str,
        pydoc: str,
        category: ExpectationCategory,
        subcategory: ExpectationSubcategory,
        params_doc: Dict[str, str],
        suite_method_name: Optional[str] = None,
    ):
        """Decorator to register an expectation factory function with metadata.

        :param name: Expectation name (e.g., 'ExpectationValueGreaterThan'). Required.
        :param pydoc: Human-readable description of the expectation. Required.
        :param category: Category from ExpectationCategory enum. Required.
        :param subcategory: Subcategory from ExpectationSubcategory enum. Required.
        :param params_doc: Documentation for each parameter. Required.
        :param suite_method_name: Override for suite method name.
                                  If not provided, auto-generated from expectation name.
        :return: Decorator function.
        """

        def decorator(func: FactoryFunction) -> FactoryFunction:
            expectation_name = name

            logger.debug(
                f"Registering expectation '{expectation_name}' with function {func.__name__}"
            )

            suite_method = suite_method_name or cls._convert_to_suite_method(expectation_name)

            # Check for duplicate suite method name
            if suite_method in cls._registry:
                existing_metadata = cls._registry[suite_method][1]
                error_message = (
                    f"Suite method '{suite_method}' is already registered by expectation '{existing_metadata.expectation_name}'. "
                    f"Cannot register '{expectation_name}'."
                )
                logger.error(error_message)
                raise ValueError(error_message)

            # Check for duplicate expectation name
            if expectation_name in cls._by_name:
                existing_suite_method = cls._by_name[expectation_name]
                error_message = f"Expectation '{expectation_name}' is already registered with suite method '{existing_suite_method}'."
                logger.error(error_message)
                raise ValueError(error_message)

            # Extract params from @requires_params if present
            extracted_params = []
            extracted_types: Dict[str, Any] = {}
            if hasattr(func, "_required_params"):
                extracted_params = list(func._required_params)
                extracted_types = getattr(func, "_param_types", {})

            metadata = ExpectationMetadata(
                suite_method_name=suite_method,
                pydoc=pydoc,
                category=category,
                subcategory=subcategory,
                params_doc=params_doc,
                params=extracted_params,
                param_types=extracted_types,
                factory_func_name=func.__name__,
                expectation_name=expectation_name,
            )

            # Store in primary registry
            cls._registry[suite_method] = (func, metadata)

            # Store in secondary index
            cls._by_name[expectation_name] = suite_method

            return func

        return decorator

    @classmethod
    def _convert_to_suite_method(cls, expectation_name: str) -> str:
        """Convert expectation name to suite method name.

        :param expectation_name: Expectation name (e.g., 'ExpectationValueGreaterThan').
        :return: Suite method name (e.g., 'expect_value_greater_than').

        Examples:
            ExpectationValueGreaterThan -> expect_value_greater_than
            ExpectationMinRows -> expect_min_rows
        """

        name = re.sub(r"^Expectation", "", expectation_name)

        # Convert CamelCase to snake_case
        snake = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
        snake = re.sub(r"([a-z\d])([A-Z])", r"\1_\2", snake)
        return "expect_" + snake.lower()

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
        import pkgutil
        from pathlib import Path

        import dataframe_expectations.expectations as expectations_pkg

        for importer, modname, ispkg in pkgutil.walk_packages(
            path=expectations_pkg.__path__,
            prefix=expectations_pkg.__name__ + ".",
            onerror=lambda x: None,
        ):
            # Skip if it's a package (directory with __init__.py)
            if ispkg:
                continue

            # Get the module file path to check if it contains the decorator
            try:
                spec = importlib.util.find_spec(modname)
                if spec and spec.origin:
                    module_file = Path(spec.origin)
                    # Quick check: does the file contain @register_expectation?
                    if "@register_expectation" in module_file.read_text():
                        importlib.import_module(modname)
                        logger.debug(f"Loaded expectation module: {modname}")
            except Exception as e:
                logger.warning(f"Failed to import module {modname}: {e}")

    @classmethod
    def get_expectation(cls, expectation_name: str, **kwargs) -> DataFrameExpectation:
        """Get an expectation instance by name.

        Note: This method is kept for backward compatibility with tests.
        The suite uses get_expectation_by_suite_method() for better performance.

        :param expectation_name: The name of the expectation.
        :param kwargs: Parameters to pass to the expectation factory function.
        :return: An instance of DataFrameExpectation.
        """
        cls._ensure_loaded()
        logger.debug(f"Retrieving expectation '{expectation_name}' with arguments: {kwargs}")

        if expectation_name not in cls._by_name:
            available = cls.list_expectations()
            error_message = (
                f"Unknown expectation '{expectation_name}'. "
                f"Available expectations: {', '.join(available)}"
            )
            logger.error(error_message)
            raise ValueError(error_message)

        suite_method = cls._by_name[expectation_name]
        factory, metadata = cls._registry[suite_method]
        return factory(**kwargs)

    @classmethod
    def get_metadata(cls, expectation_name: str) -> ExpectationMetadata:
        """Get metadata for a registered expectation.

        :param expectation_name: The name of the expectation.
        :return: Metadata for the expectation.
        :raises ValueError: If expectation not found.
        """
        cls._ensure_loaded()

        if expectation_name not in cls._by_name:
            raise ValueError(f"No metadata found for expectation '{expectation_name}'")

        suite_method = cls._by_name[expectation_name]
        factory, metadata = cls._registry[suite_method]
        return metadata

    @classmethod
    def get_all_metadata(cls) -> Dict[str, ExpectationMetadata]:
        """Get metadata for all registered expectations.

        :return: Dictionary mapping expectation names to their metadata.
        """
        cls._ensure_loaded()
        return {metadata.expectation_name: metadata for _, (_, metadata) in cls._registry.items()}

    @classmethod
    def get_expectation_by_suite_method(
        cls, suite_method_name: str, **kwargs
    ) -> DataFrameExpectation:
        """Get an expectation instance by suite method name.

        :param suite_method_name: The suite method name (e.g., 'expect_value_greater_than').
        :param kwargs: Parameters to pass to the expectation factory function.
        :return: An instance of DataFrameExpectation.
        :raises ValueError: If suite method not found.
        """
        cls._ensure_loaded()
        logger.debug(
            f"Retrieving expectation for suite method '{suite_method_name}' with arguments: {kwargs}"
        )

        if suite_method_name not in cls._registry:
            available = list(cls._registry.keys())
            error_message = (
                f"Unknown suite method '{suite_method_name}'. "
                f"Available methods: {', '.join(available[:10])}..."
            )
            logger.error(error_message)
            raise ValueError(error_message)

        factory, metadata = cls._registry[suite_method_name]
        return factory(**kwargs)

    @classmethod
    def get_suite_method_mapping(cls) -> Dict[str, str]:
        """Get mapping of suite method names to expectation names.

        :return: Dictionary mapping suite method names (e.g., 'expect_value_greater_than')
                 to expectation names (e.g., 'ExpectationValueGreaterThan').
        """
        cls._ensure_loaded()
        return {
            suite_method: metadata.expectation_name
            for suite_method, (_, metadata) in cls._registry.items()
        }

    @classmethod
    def list_expectations(cls) -> list:
        """List all registered expectation names.

        :return: List of registered expectation names.
        """
        cls._ensure_loaded()
        return [metadata.expectation_name for _, (_, metadata) in cls._registry.items()]

    @classmethod
    def remove_expectation(cls, expectation_name: str):
        """Remove an expectation from the registry.

        :param expectation_name: The name of the expectation to remove.
        :raises ValueError: If expectation not found.
        """
        cls._ensure_loaded()
        logger.debug(f"Removing expectation '{expectation_name}'")

        if expectation_name not in cls._by_name:
            error_message = f"Expectation '{expectation_name}' not found."
            logger.error(error_message)
            raise ValueError(error_message)

        # Remove from both dictionaries
        suite_method = cls._by_name[expectation_name]
        del cls._registry[suite_method]
        del cls._by_name[expectation_name]

    @classmethod
    def clear_expectations(cls):
        """Clear all registered expectations."""
        logger.debug(f"Clearing {len(cls._registry)} expectations from the registry")
        cls._registry.clear()
        cls._by_name.clear()
        cls._loaded = False  # Allow reloading


# Convenience decorator
register_expectation = DataFrameExpectationRegistry.register
