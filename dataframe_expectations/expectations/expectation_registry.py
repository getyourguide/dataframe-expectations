import re
from typing import Any, Callable, Dict, Optional

from dataframe_expectations.expectations import DataFrameExpectation
from dataframe_expectations.logging_utils import setup_logger

logger = setup_logger(__name__)


class DataFrameExpectationRegistry:
    """Registry for dataframe expectations."""

    _expectations: Dict[str, Callable[..., DataFrameExpectation]] = {}
    _metadata: Dict[str, Dict[str, Any]] = {}
    _loaded: bool = False

    @classmethod
    def register(
        cls,
        name: Optional[str] = None,
        suite_method_name: Optional[str] = None,
        description: Optional[str] = None,
        category: Optional[str] = None,
        subcategory: Optional[str] = None,
        params_doc: Optional[Dict[str, str]] = None,
    ):
        """Decorator to register an expectation factory function with metadata.

        :param name: Expectation name (e.g., 'ExpectationValueGreaterThan').
                     If not provided, auto-derived from function name.
        :param suite_method_name: Override for suite method name.
                                  If not provided, auto-generated from expectation name.
        :param description: Human-readable description of the expectation.
        :param category: Category (e.g., 'Column Expectations', 'Aggregation Expectations').
        :param subcategory: Subcategory (e.g., 'Numerical', 'String', 'Any Value').
        :param params_doc: Documentation for each parameter.
        :return: Decorator function.
        """

        def decorator(func: Callable[..., DataFrameExpectation]):
            # Auto-derive expectation name from function if not provided
            expectation_name = (
                name if name is not None else cls._derive_expectation_name(func.__name__)
            )

            logger.debug(
                f"Registering expectation '{expectation_name}' with function {func.__name__}"
            )

            # Check if the name is already registered
            if expectation_name in cls._expectations:
                error_message = f"Expectation '{expectation_name}' is already registered."
                logger.error(error_message)
                raise ValueError(error_message)

            # Register factory function
            cls._expectations[expectation_name] = func

            # Extract params from @requires_params if present
            extracted_params = []
            extracted_types: Dict[str, Any] = {}
            if hasattr(func, "_required_params"):
                extracted_params = list(func._required_params)
                extracted_types = getattr(func, "_param_types", {})

            # Store metadata
            cls._metadata[expectation_name] = {
                "suite_method_name": suite_method_name
                or cls._convert_to_suite_method(expectation_name),
                "description": description or "",
                "category": category or "Uncategorized",
                "subcategory": subcategory or "General",
                "params_doc": params_doc or {},
                "params": extracted_params,
                "param_types": extracted_types,
                "factory_func_name": func.__name__,
                "expectation_name": expectation_name,
            }

            return func

        return decorator

    @classmethod
    def _derive_expectation_name(cls, func_name: str) -> str:
        """Derive expectation name from factory function name.

        :param func_name: Factory function name (e.g., 'create_expectation_value_greater_than').
        :return: Expectation name (e.g., 'ExpectationValueGreaterThan').

        Examples:
            create_expectation_value_greater_than -> ExpectationValueGreaterThan
            create_expectation_min_rows -> ExpectationMinRows
        """
        # Remove 'create_expectation_' prefix
        name = re.sub(r"^create_expectation_", "", func_name)
        # Convert snake_case to CamelCase
        parts = name.split("_")
        camel_case = "".join(word.capitalize() for word in parts)
        return f"Expectation{camel_case}"

    @classmethod
    def _convert_to_suite_method(cls, expectation_name: str) -> str:
        """Convert expectation name to suite method name.

        :param expectation_name: Expectation name (e.g., 'ExpectationValueGreaterThan').
        :return: Suite method name (e.g., 'expect_value_greater_than').

        Examples:
            ExpectationValueGreaterThan -> expect_value_greater_than
            ExpectationMinRows -> expect_min_rows
        """
        # Remove 'Expectation' prefix
        name = re.sub(r"^Expectation", "", expectation_name)
        # Convert CamelCase to snake_case
        snake = re.sub("([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
        snake = re.sub("([a-z\d])([A-Z])", r"\1_\2", snake)
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
    def get_expectation(cls, expectation_name: str, **kwargs) -> DataFrameExpectation:
        """Get an expectation instance by name.

        :param expectation_name: The name of the expectation.
        :param kwargs: Parameters to pass to the expectation factory function.
        :return: An instance of DataFrameExpectation.
        """
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
    def get_metadata(cls, expectation_name: str) -> Dict[str, Any]:
        """Get metadata for a registered expectation.

        :param expectation_name: The name of the expectation.
        :return: Dictionary containing metadata for the expectation.
        :raises ValueError: If expectation not found.
        """
        cls._ensure_loaded()
        if expectation_name not in cls._metadata:
            raise ValueError(f"No metadata found for expectation '{expectation_name}'")
        return cls._metadata[expectation_name].copy()

    @classmethod
    def get_all_metadata(cls) -> Dict[str, Dict[str, Any]]:
        """Get metadata for all registered expectations.

        :return: Dictionary mapping expectation names to their metadata.
        """
        cls._ensure_loaded()
        return cls._metadata.copy()

    @classmethod
    def get_suite_method_mapping(cls) -> Dict[str, str]:
        """Get mapping of suite method names to expectation names.

        :return: Dictionary mapping suite method names (e.g., 'expect_value_greater_than')
                 to expectation names (e.g., 'ExpectationValueGreaterThan').
        """
        cls._ensure_loaded()
        return {meta["suite_method_name"]: exp_name for exp_name, meta in cls._metadata.items()}

    @classmethod
    def list_expectations(cls) -> list:
        """List all registered expectation names.

        :return: List of registered expectation names.
        """
        cls._ensure_loaded()  # Lazy load expectations
        return list(cls._expectations.keys())

    @classmethod
    def remove_expectation(cls, expectation_name: str):
        """Remove an expectation from the registry.

        :param expectation_name: The name of the expectation to remove.
        :raises ValueError: If expectation not found.
        """
        cls._ensure_loaded()  # Lazy load expectations
        logger.debug(f"Removing expectation '{expectation_name}'")
        if expectation_name in cls._expectations:
            del cls._expectations[expectation_name]
            if expectation_name in cls._metadata:
                del cls._metadata[expectation_name]
        else:
            error_message = f"Expectation '{expectation_name}' not found."
            logger.error(error_message)
            raise ValueError(error_message)

    @classmethod
    def clear_expectations(cls):
        """Clear all registered expectations."""
        logger.debug(f"Clearing {len(cls._expectations)} expectations from the registry")
        cls._expectations.clear()
        cls._metadata.clear()
        cls._loaded = False  # Allow reloading


# Convenience decorator
register_expectation = DataFrameExpectationRegistry.register
