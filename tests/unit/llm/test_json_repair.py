from __future__ import annotations

import pytest

from ccas.llm.json_repair import extract_json


def test_a_bare_object_parses() -> None:
    assert extract_json('{"a": 1}') == {"a": 1}


def test_whitespace_is_tolerated() -> None:
    assert extract_json('\n  {"a": 1}\n ') == {"a": 1}


@pytest.mark.parametrize(
    "text",
    [
        '```json\n{"a": 1}\n```',
        '```\n{"a": 1}\n```',
        'Here is the answer:\n```json\n{"a": 1}\n```\nHope that helps.',
    ],
)
def test_fenced_blocks_are_unwrapped(text: str) -> None:
    """Instructed models fence their output constantly, whatever the prompt says."""
    assert extract_json(text) == {"a": 1}


def test_an_object_embedded_in_prose_is_found() -> None:
    """Reasoning models narrate first and answer second."""
    text = 'Let me think. The intent is clearly tracking. {"a": 1} — that\'s my answer.'
    assert extract_json(text) == {"a": 1}


def test_nested_braces_are_balanced_correctly() -> None:
    assert extract_json('prefix {"a": {"b": [1, 2]}} suffix') == {"a": {"b": [1, 2]}}


def test_braces_inside_strings_do_not_confuse_the_scanner() -> None:
    assert extract_json('{"a": "a } brace", "b": 2}') == {"a": "a } brace", "b": 2}


def test_escaped_quotes_are_handled() -> None:
    assert extract_json(r'{"a": "say \"hi\""}') == {"a": 'say "hi"'}


@pytest.mark.parametrize("text", ["", "   ", "no json here at all", "{ broken", "[1, 2]", "null"])
def test_nothing_extractable_returns_none(text: str) -> None:
    """None is a real answer. The caller raises rather than inventing a result."""
    assert extract_json(text) is None


def test_an_object_wrapped_in_an_array_is_still_recovered() -> None:
    """A model that returns [{...}] has the right content in the wrong wrapper.

    Recovering it is right: the schema validation downstream catches a genuinely wrong
    shape, and refusing here would fail a case that is plainly recoverable.
    """
    assert extract_json('[{"a": 1}]') == {"a": 1}


def test_malformed_json_is_not_repaired() -> None:
    """A model that emitted broken JSON misunderstood the task. Guessing at its intent
    would put invented values into a caller's transcript."""
    assert extract_json('{"a": 1,}') is None
