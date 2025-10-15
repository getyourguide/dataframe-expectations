import pytest

from dataframe_expectations.expectations.expectation_registry import (
    DataframeExpectationRegistry,
)


class DummyExpectation:
    def __init__(self, foo=None):
        self.foo = foo


@pytest.fixture(autouse=True)
def cleanup_registry():
    # Save the original state of the registry
    original = set(DataframeExpectationRegistry.list_expectations())

    yield

    # Remove any expectations added during the test
    current = set(DataframeExpectationRegistry.list_expectations())
    for name in current - original:
        DataframeExpectationRegistry.remove_expectation(name)


def test_register_and_get_expectation():
    """
    Test registering and retrieving an expectation.
    """

    @DataframeExpectationRegistry.register("DummyExpectation")
    def dummy_expectation_factory(foo=None):
        return DummyExpectation(foo=foo)

    instance = DataframeExpectationRegistry.get_expectation("DummyExpectation", foo=123)
    assert isinstance(
        instance, DummyExpectation
    ), f"Expected DummyExpectation instance but got: {type(instance)}"
    assert instance.foo == 123, f"Expected foo=123 but got: {instance.foo}"


def test_duplicate_registration_raises():
    """
    Test that registering an expectation with the same name raises a ValueError.
    """

    @DataframeExpectationRegistry.register("DuplicateExpectation")
    def dummy1(foo=None):
        return DummyExpectation(foo=foo)

    with pytest.raises(ValueError) as context:

        @DataframeExpectationRegistry.register("DuplicateExpectation")
        def dummy2(foo=None):
            return DummyExpectation(foo=foo)

    assert "already registered" in str(
        context.value
    ), f"Expected 'already registered' in error message but got: {str(context.value)}"


def test_get_unknown_expectation_raises():
    """
    Test that trying to get an unknown expectation raises a ValueError.
    """
    with pytest.raises(ValueError) as context:
        DataframeExpectationRegistry.get_expectation("NonExistent")
    assert "Unknown expectation" in str(
        context.value
    ), f"Expected 'Unknown expectation' in error message but got: {str(context.value)}"


def test_list_expectations():
    """
    Test listing all registered expectations.
    """

    @DataframeExpectationRegistry.register("First")
    def dummy1(foo=None):
        return DummyExpectation(foo=foo)

    @DataframeExpectationRegistry.register("Second")
    def dummy2(foo=None):
        return DummyExpectation(foo=foo)

    names = DataframeExpectationRegistry.list_expectations()
    assert "First" in names, f"Expected 'First' in expectations list but got: {names}"
    assert "Second" in names, f"Expected 'Second' in expectations list but got: {names}"


def test_remove_expectation():
    """
    Test removing an expectation from the registry.
    """

    @DataframeExpectationRegistry.register("ToRemove")
    def dummy(foo=None):
        return DummyExpectation(foo=foo)

    names_before = DataframeExpectationRegistry.list_expectations()
    assert (
        "ToRemove" in names_before
    ), f"Expected 'ToRemove' in expectations list before removal but got: {names_before}"

    DataframeExpectationRegistry.remove_expectation("ToRemove")

    names_after = DataframeExpectationRegistry.list_expectations()
    assert (
        "ToRemove" not in names_after
    ), f"Expected 'ToRemove' not in expectations list after removal but got: {names_after}"


def test_remove_nonexistent_expectation_raises():
    """
    Test that trying to remove a non-existent expectation raises a ValueError.
    """
    with pytest.raises(ValueError) as context:
        DataframeExpectationRegistry.remove_expectation("DefinitelyNotThere")
    assert "not found" in str(
        context.value
    ), f"Expected 'not found' in error message but got: {str(context.value)}"
