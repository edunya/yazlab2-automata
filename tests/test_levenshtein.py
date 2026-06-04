"""
Tests for Levenshtein-based unseen symbolic state mapping.
"""

import pytest

from src.automata.levenshtein import (
    LevenshteinStateMapper,
    levenshtein_distance
)


def test_levenshtein_distance_basic_operations():
    assert levenshtein_distance("abc", "abc") == 0
    assert levenshtein_distance("abc", "abd") == 1
    assert levenshtein_distance("abc", "ab") == 1
    assert levenshtein_distance("ab", "abc") == 1


def test_levenshtein_distance_supports_empty_strings():
    assert levenshtein_distance("", "abc") == 3
    assert levenshtein_distance("abc", "") == 3


def test_seen_state_maps_to_itself():
    mapper = LevenshteinStateMapper(["aab", "abb", "bcc"])

    mapping = mapper.map_state("abb")

    assert mapping.original_state == "abb"
    assert mapping.mapped_state == "abb"
    assert mapping.distance == 0
    assert mapping.status == "seen"
    assert mapping.is_unseen is False
    assert mapping.mapping_method == "exact_match"


def test_unseen_state_maps_to_nearest_known_state():
    mapper = LevenshteinStateMapper(["aaa", "abb", "ccc"])

    mapping = mapper.map_state("bbb")

    assert mapping.original_state == "bbb"
    assert mapping.mapped_state == "abb"
    assert mapping.distance == 1
    assert mapping.status == "unseen"
    assert mapping.is_unseen is True
    assert mapping.mapping_method == "levenshtein_nearest_state"


def test_tie_breaking_selects_alphabetically_first_state():
    mapper = LevenshteinStateMapper(["aac", "aaa"])

    mapping = mapper.map_state("aab")

    assert mapping.distance == 1
    assert mapping.mapped_state == "aaa"


def test_map_sequence_returns_mapped_path_and_records():
    mapper = LevenshteinStateMapper(["aaa", "abb", "ccc"])

    mapped_states, mappings = mapper.map_sequence(
        ["aaa", "bbb", "ccc"]
    )

    assert mapped_states == ("aaa", "abb", "ccc")
    assert [mapping.status for mapping in mappings] == [
        "seen", "unseen", "seen"
    ]


def test_mapper_rejects_empty_known_state_vocabulary():
    with pytest.raises(ValueError):
        LevenshteinStateMapper([])