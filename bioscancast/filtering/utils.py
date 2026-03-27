from __future__ import annotations

import re
from collections import Counter
from typing import Iterable, List


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def tokenize(text: str) -> List[str]:
    text = normalize_text(text)
    return re.findall(r"[a-zA-Z0-9_\-]+", text)


def jaccard_similarity(a: str, b: str) -> float:
    set_a = set(tokenize(a))
    set_b = set(tokenize(b))
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def weighted_sum(score_map: dict[str, float], weight_map: dict[str, float]) -> float:
    total = 0.0
    denom = 0.0
    for key, value in score_map.items():
        w = weight_map.get(key, 0.0)
        total += value * w
        denom += w
    return total / denom if denom > 0 else 0.0


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def keyword_overlap_score(text: str, query_terms: Iterable[str]) -> float:
    text_tokens = set(tokenize(text))
    query_tokens = set(tokenize(" ".join(query_terms)))
    if not query_tokens:
        return 0.0
    overlap = len(text_tokens & query_tokens)
    return overlap / len(query_tokens)


def most_common_tokens(text: str, top_k: int = 20) -> list[str]:
    counts = Counter(tokenize(text))
    return [token for token, _ in counts.most_common(top_k)]