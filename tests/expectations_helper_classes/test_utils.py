from typing import Union
import pytest

from dataframe_expectations.expectations.utils import requires_params


def test_requires_params_success():
    """
    Test that all required parameters are provided.
    """

    @requires_params("a", "b")
    def func(**kwargs):
        return kwargs["a"] + kwargs["b"]

    result = func(a=1, b=2)
    assert result == 3, f"Expected 3 but got: {result}"


def test_requires_params_missing_param():
    """
    Test that a ValueError is raised when a required parameter is missing.
    """

    @requires_params("a", "b")
    def func(**kwargs):
        return kwargs["a"] + kwargs["b"]

    with pytest.raises(ValueError) as context:
        func(a=1)
    assert "missing required parameters" in str(
        context.value
    ), f"Expected 'missing required parameters' in error message but got: {str(context.value)}"


def test_requires_params_type_success():
    """
    Test that type validation works correctly when types are specified.
    """

    @requires_params("a", "b", types={"a": int, "b": str})
    def func(**kwargs):
        return f"{kwargs['a']}-{kwargs['b']}"

    result = func(a=5, b="hello")
    assert result == "5-hello", f"Expected '5-hello' but got: {result}"


def test_requires_params_type_error():
    """
    Test that a TypeError is raised when a parameter does not match the expected type."""

    @requires_params("a", "b", types={"a": int, "b": str})
    def func(**kwargs):
        return f"{kwargs['a']}-{kwargs['b']}"

    with pytest.raises(TypeError) as context:
        func(a="not-an-int", b="hello")
    assert "type validation errors" in str(
        context.value
    ), f"Expected 'type validation errors' in error message but got: {str(context.value)}"


def test_requires_params_union_type_success():
    """
    Test that Union types are handled correctly.
    """

    @requires_params("a", types={"a": Union[int, str]})
    def func(**kwargs):
        return kwargs["a"]

    result1 = func(a=5)
    assert result1 == 5, f"Expected 5 but got: {result1}"

    result2 = func(a="foo")
    assert result2 == "foo", f"Expected 'foo' but got: {result2}"


def test_requires_params_union_type_error():
    """
    Test that a TypeError is raised when a parameter does not match any type in a Union.
    """

    @requires_params("a", types={"a": Union[int, str]})
    def func(**kwargs):
        return kwargs["a"]

    with pytest.raises(TypeError):
        func(a=3.14)
