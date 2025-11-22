"""Comprehensive unit tests for TagSet tagging system."""

import pytest
from dataframe_expectations.core.tagging import TagSet


@pytest.mark.parametrize(
    "tags, expected_count",
    [
        (["priority:high"], 1),
        (["priority:high", "env:test"], 2),
        (["priority:high", "priority:medium"], 2),
        (["priority:high", "env:test", "env:prod"], 3),
        ([], 0),
        (None, 0),
    ],
)
def test_initialization_valid_tags(tags, expected_count):
    """Test initialization with valid tags."""
    tag_set = TagSet(tags)
    assert len(tag_set) == expected_count


@pytest.mark.parametrize(
    "invalid_tag, error_pattern",
    [
        ("notag", "Invalid tag format"),
        ("no-colon", "Invalid tag format"),
        (":no-key", "key and value must be non-empty"),
        ("no-value:", "key and value must be non-empty"),
        ("  :  ", "key and value must be non-empty"),
    ],
)
def test_initialization_invalid_format(invalid_tag, error_pattern):
    """Test that invalid tag formats raise ValueError."""
    with pytest.raises(ValueError, match=error_pattern):
        TagSet([invalid_tag])


def test_multiple_colons_allowed():
    """Test that multiple colons are allowed (split on first colon)."""
    tag_set = TagSet(["url:https://example.com"])
    assert len(tag_set) == 1
    assert tag_set.tags["url"] == {"https://example.com"}


def test_multiple_values_same_key():
    """Test that multiple values for the same key are stored correctly."""
    tag_set = TagSet(["priority:high", "priority:medium", "priority:low"])
    assert len(tag_set) == 3
    # Verify internal structure
    assert "priority" in tag_set.tags
    assert tag_set.tags["priority"] == {"high", "medium", "low"}


def test_whitespace_handling():
    """Test that whitespace is properly handled."""
    tag_set = TagSet(["  priority : high  ", "env:test"])
    assert len(tag_set) == 2


@pytest.mark.parametrize(
    "self_tags, other_tags, expected",
    [
        # Single key matching
        (["priority:high"], ["priority:high"], True),
        (["priority:high"], ["priority:medium"], False),
        # Multiple keys, any matches
        (["priority:high", "env:test"], ["priority:high"], True),
        (["priority:high", "env:test"], ["env:test"], True),
        (["priority:high", "env:test"], ["priority:high", "env:test"], True),
        # Multiple keys, none match
        (["priority:high", "env:test"], ["priority:low", "env:prod"], False),
        # Multiple values per key
        (["priority:high", "priority:medium"], ["priority:high"], True),
        (["priority:high", "priority:medium"], ["priority:low"], False),
        (["priority:high", "priority:medium"], ["priority:medium", "priority:low"], True),
        # Empty cases
        (["priority:high"], [], True),  # Empty other matches everything
        ([], ["priority:high"], False),  # Empty self matches nothing
        # Complex scenarios
        (
            ["priority:high", "priority:medium", "env:test", "role:admin"],
            ["priority:low", "env:test"],
            True,
        ),  # env:test matches
        (
            ["priority:high", "env:test"],
            ["priority:medium", "env:prod", "role:admin"],
            False,
        ),
    ],
)
def test_has_any_tag_from(self_tags, other_tags, expected):
    """Test has_any_tag_from with various combinations (OR logic)."""
    tag_set = TagSet(self_tags)
    other = TagSet(other_tags)
    assert tag_set.has_any_tag_from(other) == expected


@pytest.mark.parametrize(
    "self_tags, other_tags, expected",
    [
        # Single key matching
        (["priority:high"], ["priority:high"], True),
        (["priority:high"], ["priority:medium"], False),
        # Multiple keys, all must match
        (["priority:high", "env:test"], ["priority:high"], True),  # All of other present
        (["priority:high", "env:test"], ["env:test"], True),  # All of other present
        (
            ["priority:high", "env:test"],
            ["priority:high", "env:test"],
            True,
        ),  # All of other present
        (
            ["priority:high", "env:test"],
            ["priority:high", "env:prod"],
            False,
        ),  # env:prod not present
        # Multiple values per key - ALL must be present
        (["priority:high", "priority:medium"], ["priority:high"], True),
        (
            ["priority:high", "priority:medium"],
            ["priority:high", "priority:medium"],
            True,
        ),
        (["priority:high"], ["priority:high", "priority:medium"], False),  # medium not present
        # Empty cases
        (["priority:high"], [], True),  # Empty other matches everything
        ([], ["priority:high"], False),  # Empty self matches nothing
        # Complex scenarios
        (
            ["priority:high", "priority:medium", "env:test", "role:admin"],
            ["priority:high", "env:test"],
            True,
        ),
        (
            ["priority:high", "priority:medium", "env:test", "role:admin"],
            ["priority:high", "priority:medium", "env:test"],
            True,
        ),
        (
            ["priority:high", "env:test"],
            ["priority:high", "priority:medium", "env:test"],
            False,
        ),  # priority:medium not present
        (
            ["priority:high", "env:test"],
            ["priority:high", "env:test", "role:admin"],
            False,
        ),  # role:admin not present
    ],
)
def test_has_all_tags_from(self_tags, other_tags, expected):
    """Test has_all_tags_from with various combinations (AND logic)."""
    tag_set = TagSet(self_tags)
    other = TagSet(other_tags)
    assert tag_set.has_all_tags_from(other) == expected


@pytest.mark.parametrize(
    "tags, expected_empty",
    [
        ([], True),
        (None, True),
        (["priority:high"], False),
        (["priority:high", "env:test"], False),
    ],
)
def test_is_empty(tags, expected_empty):
    """Test is_empty method."""
    tag_set = TagSet(tags)
    assert tag_set.is_empty() == expected_empty


@pytest.mark.parametrize(
    "tags, expected_len",
    [
        ([], 0),
        (None, 0),
        (["priority:high"], 1),
        (["priority:high", "env:test"], 2),
        (["priority:high", "priority:medium"], 2),
        (["priority:high", "priority:medium", "env:test"], 3),
    ],
)
def test_len(tags, expected_len):
    """Test __len__ method."""
    tag_set = TagSet(tags)
    assert len(tag_set) == expected_len


@pytest.mark.parametrize(
    "tags, expected_bool",
    [
        ([], False),
        (None, False),
        (["priority:high"], True),
        (["priority:high", "env:test"], True),
    ],
)
def test_bool(tags, expected_bool):
    """Test __bool__ method."""
    tag_set = TagSet(tags)
    assert bool(tag_set) == expected_bool


@pytest.mark.parametrize(
    "tags, expected_str",
    [
        ([], "TagSet(empty)"),
        (None, "TagSet(empty)"),
        (["priority:high"], "TagSet(priority:high)"),
        (["env:test", "priority:high"], "TagSet(env:test, priority:high)"),  # Alphabetically sorted
        (
            ["priority:medium", "priority:high"],
            "TagSet(priority:high, priority:medium)",
        ),  # Values sorted
    ],
)
def test_str_representation(tags, expected_str):
    """Test __str__ and __repr__ methods."""
    tag_set = TagSet(tags)
    assert str(tag_set) == expected_str
    assert repr(tag_set) == expected_str


def test_duplicate_tags():
    """Test that duplicate tags are deduplicated."""
    tag_set = TagSet(["priority:high", "priority:high", "env:test", "env:test"])
    assert len(tag_set) == 2  # Only unique tags counted


def test_case_sensitivity():
    """Test that tags are case-sensitive."""
    tag_set = TagSet(["priority:high", "priority:High", "Priority:high"])
    assert len(tag_set) == 3  # All three are different


def test_special_characters_in_values():
    """Test tags with special characters in values."""
    tag_set = TagSet(["url:https://example.com", "path:/usr/local/bin", "label:user-name"])
    assert len(tag_set) == 3


def test_numeric_values():
    """Test tags with numeric-looking values."""
    tag_set = TagSet(["version:1.0", "port:8080", "priority:1"])
    assert len(tag_set) == 3


def test_empty_string_handling():
    """Test that empty strings in key or value raise errors."""
    with pytest.raises(ValueError, match="key and value must be non-empty"):
        TagSet([":value"])

    with pytest.raises(ValueError, match="key and value must be non-empty"):
        TagSet(["key:"])
