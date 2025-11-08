import pytest

from dataframe_expectations.expectations.expectation_registry import (
    DataFrameExpectationRegistry,
    ExpectationCategory,
    ExpectationSubcategory,
)


class DummyExpectation:
    def __init__(self, foo=None):
        self.foo = foo


@pytest.fixture(autouse=True)
def cleanup_registry():
    # Save the original state of the registry
    original = set(DataFrameExpectationRegistry.list_expectations())

    yield

    # Remove any expectations added during the test
    current = set(DataFrameExpectationRegistry.list_expectations())
    for name in current - original:
        DataFrameExpectationRegistry.remove_expectation(name)


def test_register_and_get_expectation():
    """
    Test registering and retrieving an expectation.
    """

    @DataFrameExpectationRegistry.register(
        "DummyExpectation",
        pydoc="Test expectation",
        category=ExpectationCategory.COLUMN_EXPECTATIONS,
        subcategory=ExpectationSubcategory.ANY_VALUE,
        params_doc={"foo": "Test parameter"},
    )
    def dummy_expectation_factory(foo=None):
        return DummyExpectation(foo=foo)

    instance = DataFrameExpectationRegistry.get_expectation("DummyExpectation", foo=123)
    assert isinstance(instance, DummyExpectation), (
        f"Expected DummyExpectation instance but got: {type(instance)}"
    )
    assert instance.foo == 123, f"Expected foo=123 but got: {instance.foo}"


def test_duplicate_registration_raises():
    """
    Test that registering an expectation with the same name raises a ValueError.
    """

    @DataFrameExpectationRegistry.register(
        "DuplicateExpectation",
        pydoc="Test expectation",
        category=ExpectationCategory.COLUMN_EXPECTATIONS,
        subcategory=ExpectationSubcategory.ANY_VALUE,
        params_doc={"foo": "Test parameter"},
    )
    def dummy1(foo=None):
        return DummyExpectation(foo=foo)

    with pytest.raises(ValueError) as context:

        @DataFrameExpectationRegistry.register(
            "DuplicateExpectation",
            pydoc="Test expectation",
            category=ExpectationCategory.COLUMN_EXPECTATIONS,
            subcategory=ExpectationSubcategory.ANY_VALUE,
            params_doc={"foo": "Test parameter"},
        )
        def dummy2(foo=None):
            return DummyExpectation(foo=foo)

    assert "already registered" in str(context.value), (
        f"Expected 'already registered' in error message but got: {str(context.value)}"
    )


def test_get_unknown_expectation_raises():
    """
    Test that trying to get an unknown expectation raises a ValueError.
    """
    with pytest.raises(ValueError) as context:
        DataFrameExpectationRegistry.get_expectation("NonExistent")
    assert "Unknown expectation" in str(context.value), (
        f"Expected 'Unknown expectation' in error message but got: {str(context.value)}"
    )


def test_list_expectations():
    """
    Test listing all registered expectations.
    """

    @DataFrameExpectationRegistry.register(
        "First",
        pydoc="Test expectation",
        category=ExpectationCategory.COLUMN_EXPECTATIONS,
        subcategory=ExpectationSubcategory.ANY_VALUE,
        params_doc={"foo": "Test parameter"},
    )
    def dummy1(foo=None):
        return DummyExpectation(foo=foo)

    @DataFrameExpectationRegistry.register(
        "Second",
        pydoc="Test expectation",
        category=ExpectationCategory.COLUMN_EXPECTATIONS,
        subcategory=ExpectationSubcategory.ANY_VALUE,
        params_doc={"foo": "Test parameter"},
    )
    def dummy2(foo=None):
        return DummyExpectation(foo=foo)

    names = DataFrameExpectationRegistry.list_expectations()
    assert "First" in names, f"Expected 'First' in expectations list but got: {names}"
    assert "Second" in names, f"Expected 'Second' in expectations list but got: {names}"


def test_remove_expectation():
    """
    Test removing an expectation from the registry.
    """

    @DataFrameExpectationRegistry.register(
        "ToRemove",
        pydoc="Test expectation",
        category=ExpectationCategory.COLUMN_EXPECTATIONS,
        subcategory=ExpectationSubcategory.ANY_VALUE,
        params_doc={"foo": "Test parameter"},
    )
    def dummy(foo=None):
        return DummyExpectation(foo=foo)

    names_before = DataFrameExpectationRegistry.list_expectations()
    assert "ToRemove" in names_before, (
        f"Expected 'ToRemove' in expectations list before removal but got: {names_before}"
    )

    DataFrameExpectationRegistry.remove_expectation("ToRemove")

    names_after = DataFrameExpectationRegistry.list_expectations()
    assert "ToRemove" not in names_after, (
        f"Expected 'ToRemove' not in expectations list after removal but got: {names_after}"
    )


def test_remove_nonexistent_expectation_raises():
    """
    Test that trying to remove a non-existent expectation raises a ValueError.
    """
    with pytest.raises(ValueError) as context:
        DataFrameExpectationRegistry.remove_expectation("DefinitelyNotThere")
    assert "not found" in str(context.value), (
        f"Expected 'not found' in error message but got: {str(context.value)}"
    )
