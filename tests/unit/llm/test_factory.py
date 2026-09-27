"""The credential guard must treat an empty string as absent -- Rule 7."""

from __future__ import annotations

import pytest
from pydantic import SecretStr

from ccas.llm.factory import key_is_set


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        (None, False),
        ("", False),
        ("sk-...", True),
        (SecretStr(""), False),
        (SecretStr("sk-..."), True),
    ],
)
def test_key_is_set(key: object, expected: bool) -> None:
    assert key_is_set(key) is expected
