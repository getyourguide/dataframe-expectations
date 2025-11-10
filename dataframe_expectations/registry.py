import re
from enum import Enum
from typing import Any, Callable, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field

from dataframe_expectations.core.expectation import DataFrameExpectation
from dataframe_expectations.logging_utils import setup_logger

logger = setup_logger(__name__)


class ExpectationCategory(str, Enum):
    """Categories for expectations."""

    COLUMN_EXPECTATIONS = "Column Expectations"
    COLUMN_AGGREGATION_EXPECTATIONS = "Column Aggregation Expectations"
    DATAFRAME_AGGREGATION_EXPECTATIONS = "DataFrame Aggregation Expectations"


class ExpectationSubcategory(str, Enum):
    """Subcategory of expectations."""

    ANY_VALUE = "Any Value"
    NUMERICAL = "Numerical"
    STRING = "String"
    UNIQUE = "Unique"


class ExpectationMetadata(BaseModel):
    """Metadata for a registered expectation."""

    suite_method_name: str = Field(
        ..., description="Method name in ExpectationsSuite (e.g., 'expect_value_greater_than')"
    )
    pydoc: str = Field(..., description="Human-readable description of the expectation")
    category: ExpectationCategory = Field(..., description="Category (e.g., 'Column Expectations')")
    subcategory: ExpectationSubcategory = Field(
        ..., description="Subcategory (e.g., 'Numerical', 'String')"
    )
    params_doc: Dict[str, str] = Field(..., description="Documentation for each parameter")
    params: list = Field(default_factory=list, description="List of required parameter names")
    param_types: Dict[str, Any] = Field(
        default_factory=dict, description="Type hints for parameters"
    )
    factory_func_name: str = Field(..., description="Name of the factory function")
    expectation_name: str = Field(..., description="Name of the expectation class")

    model_config = ConfigDict(frozen=True)  # Make model immutable


class DataFrameExpectationRegistry:
    """Registry for dataframe expectations."""

    _expectations: Dict[str, Callable[..., DataFrameExpectation]] = {}
    _metadata: Dict[str, ExpectationMetadata] = {}
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

        def decorator(func: Callable[..., DataFrameExpectation]):
            expectation_name = name

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
            cls._metadata[expectation_name] = ExpectationMetadata(
                suite_method_name=suite_method_name
                or cls._convert_to_suite_method(expectation_name),
                pydoc=pydoc,
                category=category,
                subcategory=subcategory,
                params_doc=params_doc,
                params=extracted_params,
                param_types=extracted_types,
                factory_func_name=func.__name__,
                expectation_name=expectation_name,
            )

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
        # Remove 'Expectation' prefix
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
    def get_metadata(cls, expectation_name: str) -> ExpectationMetadata:
        """Get metadata for a registered expectation.

        :param expectation_name: The name of the expectation.
        :return: Metadata for the expectation.
        :raises ValueError: If expectation not found.
        """
        cls._ensure_loaded()
        if expectation_name not in cls._metadata:
            raise ValueError(f"No metadata found for expectation '{expectation_name}'")
        return cls._metadata[expectation_name]

    @classmethod
    def get_all_metadata(cls) -> Dict[str, ExpectationMetadata]:
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
        return {meta.suite_method_name: exp_name for exp_name, meta in cls._metadata.items()}

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
