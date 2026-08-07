from __future__ import annotations

import math

try:
    import tiktoken
except Exception:  # pragma: no cover - network/proxy fallback
    tiktoken = None


def _fallback_token_count(text: str) -> int:
    """Fallback tokenizer based on whitespace-aware word counting."""
    if not text:
        return 0
    words = text.split()
    if not words:
        return 0
    # Roughly 1.3 tokens per word, with a small overhead for punctuation.
    return int(math.ceil(len(words) * 1.3))


if tiktoken is not None:
    try:
        _ENCODER = tiktoken.get_encoding("cl100k_base")
    except Exception:  # pragma: no cover - network/proxy fallback
        _ENCODER = None
else:
    _ENCODER = None


def approx_token_count(text: str) -> int:
    """Return an approximate token count using the cl100k_base encoding."""
    if not text:
        return 0
    if _ENCODER is not None:
        try:
            return len(_ENCODER.encode(text))
        except Exception:  # pragma: no cover - network/proxy fallback
            pass
    return _fallback_token_count(text)
