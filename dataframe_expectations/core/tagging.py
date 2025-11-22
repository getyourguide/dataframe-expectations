"""Tagging system for filtering and organizing expectations.

Simple tag-based filtering using "key:value" format strings.
Tags are stored internally as Dict[key, Set[values]] for efficient matching.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set


class TagSet:
    """
    Collection of tags organized by key, supporting multiple values per key.

    Internal structure: Dict[key, Set[values]]
    Example: {"priority": {"high", "medium"}, "env": {"test", "prod"}}

    Tags are specified as strings in "key:value" format.
    """

    def __init__(self, tags: Optional[List[str]] = None):
        """
        Initialize TagSet from a list of tag strings.

        :param tags: List of tag strings in "key:value" format
                    Example: ["priority:high", "env:test", "priority:medium"]

        Examples:
            >>> TagSet(["priority:high", "env:test"])
            >>> TagSet(["priority:high", "priority:medium"])  # Multiple values for same key
        """
        self._tags: Dict[str, Set[str]] = {}

        if tags:
            for tag_string in tags:
                self._add_tag_string(tag_string)

    def _add_tag_string(self, tag_string: str) -> None:
        """
        Parse and add a tag string in "key:value" format.

        :param tag_string: Tag string to parse
        :raises ValueError: If format is invalid
        """
        tag_string = tag_string.strip()

        if ":" not in tag_string:
            raise ValueError(f"Invalid tag format '{tag_string}'. Expected 'key:value' format.")

        parts = tag_string.split(":", 1)
        if len(parts) != 2:
            raise ValueError(
                f"Invalid tag format '{tag_string}'. Expected exactly one ':' separator."
            )

        key, value = parts[0].strip(), parts[1].strip()

        if not key or not value:
            raise ValueError("Tag key and value must be non-empty strings")

        if key not in self._tags:
            self._tags[key] = set()
        self._tags[key].add(value)

    def has_any_tag_from(self, other: TagSet) -> bool:
        """
        Check if this TagSet has ANY tag from the other TagSet (OR logic).

        For each key in 'other', checks if there's any overlap in values.
        Returns True if ANY key has any overlapping values.

        :param other: TagSet to match against
        :return: True if any tag matches

        Examples:
            self = TagSet(["priority:high", "env:test"])
            other = TagSet(["priority:high"])
            self.has_any_tag_from(other) -> True (priority:high matches)

            other = TagSet(["priority:medium"])
            self.has_any_tag_from(other) -> False

            other = TagSet(["priority:medium", "env:test"])
            self.has_any_tag_from(other) -> True (env:test matches)
        """
        if not other._tags:
            return True  # Empty filter matches everything

        # OR logic: any key with overlapping values
        for key, required_values in other._tags.items():
            if key in self._tags:
                # Check if there's any overlap between required values and our values
                if required_values & self._tags[key]:
                    return True

        return False

    def has_all_tags_from(self, other: TagSet) -> bool:
        """
        Check if this TagSet has ALL tags from the other TagSet (AND logic).

        For each key in 'other', checks if there's any overlap in values.
        Returns True only if ALL keys from other have overlapping values.

        :param other: TagSet to match against
        :return: True if all tags match

        Examples:
            self = TagSet(["priority:high", "env:test", 'role:admin'])
            other = TagSet(["priority:high", "env:test"])
            self.has_all_tags_from(other) -> True (both match)

            other = TagSet(["priority:high"])
            self.has_all_tags_from(other) -> True (priority:high matches)

            other = TagSet(["priority:high", "env:prod"])
            self.has_all_tags_from(other) -> False (env:prod doesn't match)
        """
        if not other._tags:
            return True  # Empty filter matches everything

        # AND logic: all keys must have ALL required values present
        for key, required_values in other._tags.items():
            if key not in self._tags:
                return False
            # Check if ALL required values are present in our values
            if not required_values.issubset(self._tags[key]):
                return False

        return True

    def is_empty(self) -> bool:
        """Check if TagSet has no tags."""
        return len(self._tags) == 0

    def __len__(self) -> int:
        """Return total number of unique tags (key:value pairs)."""
        return sum(len(values) for values in self._tags.values())

    def __bool__(self) -> bool:
        """Return True if TagSet has any tags."""
        return bool(self._tags)

    def __str__(self) -> str:
        """String representation showing all tags."""
        tags = []
        for key in sorted(self._tags.keys()):
            for value in sorted(self._tags[key]):
                tags.append(f"{key}:{value}")
        return f"TagSet({', '.join(tags)})" if tags else "TagSet(empty)"

    def __repr__(self) -> str:
        return self.__str__()
