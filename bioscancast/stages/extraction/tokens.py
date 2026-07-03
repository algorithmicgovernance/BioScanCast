from __future__ import annotations

import tiktoken

_ENCODER = tiktoken.get_encoding("cl100k_base")


def approx_token_count(text: str) -> int:
    """Return an approximate token count using the cl100k_base encoding."""
    if not text:
        return 0
    return len(_ENCODER.encode(text))
