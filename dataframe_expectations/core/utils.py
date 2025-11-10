from functools import wraps
from typing import Any, Callable, Dict, Optional, Tuple, Type, Union, get_args

from dataframe_expectations.core.expectation import DataFrameExpectation


def requires_params(
    *required_params, types: Optional[Dict[str, Union[Type, Tuple[Type, ...]]]] = None
):
    """
    Decorator that validates required parameters and optionally checks their types.

    :param required_params: Required parameter names
    :param types: Optional dict mapping parameter names to expected types

    Usage:
        @requires_params("column_name", "value")
        def func(**kwargs): ...

        @requires_params("column_name", "value", types={"column_name": str, "value": int})
        def func(**kwargs): ...
    """

    def decorator(func: Callable[..., DataFrameExpectation]):
        @wraps(func)
        def wrapper(**kwargs):
            func_name = func.__name__

            # Check for missing parameters
            missing_params = [param for param in required_params if param not in kwargs]
            if missing_params:
                param_list = ", ".join(required_params)
                raise ValueError(
                    f"{func_name} missing required parameters: {', '.join(missing_params)}. "
                    f"Required: [{param_list}]"
                )

            # Type checking if types dict is provided
            if types:
                type_errors = []
                for param_name, expected_type in types.items():
                    if param_name in kwargs:
                        actual_value = kwargs[param_name]
                        if not _is_instance_of_type(actual_value, expected_type):
                            type_errors.append(
                                f"'{param_name}' expected {_get_type_name(expected_type)}, "
                                f"got {type(actual_value).__name__}"
                            )

                if type_errors:
                    raise TypeError(f"{func_name} type validation errors: {'; '.join(type_errors)}")

            return func(**kwargs)

        # Attach metadata to wrapper for registry to extract
        wrapper._required_params = required_params  # type: ignore[attr-defined]
        wrapper._param_types = types or {}  # type: ignore[attr-defined]

        return wrapper

    return decorator


def _is_instance_of_type(value: Any, expected_type: Type) -> bool:
    """Helper function to check if value is instance of expected_type, handling Union types."""
    # Handle Union types (like Optional[str] which is Union[str, None])
    if hasattr(expected_type, "__origin__") and expected_type.__origin__ is Union:
        # For Union types, check if value matches any of the union members
        union_args = get_args(expected_type)
        return any(isinstance(value, arg) for arg in union_args if arg is not type(None)) or (
            type(None) in union_args and value is None
        )

    # Handle regular types
    return isinstance(value, expected_type)


def _get_type_name(type_hint: Type) -> str:
    """Helper function to get a readable name for type hints."""
    if hasattr(type_hint, "__origin__") and type_hint.__origin__ is Union:
        union_args = get_args(type_hint)
        arg_names = [arg.__name__ if hasattr(arg, "__name__") else str(arg) for arg in union_args]
        return f"Union[{', '.join(arg_names)}]"

    return getattr(type_hint, "__name__", str(type_hint))
