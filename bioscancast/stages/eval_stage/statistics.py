from __future__ import annotations

from dataclasses import dataclass
from math import comb
from typing import Iterable, Sequence

import numpy as np

try:
    from scipy.stats import binomtest, ttest_rel, wilcoxon
except Exception:  # pragma: no cover - fallback if scipy is unavailable
    binomtest = None
    ttest_rel = None
    wilcoxon = None


@dataclass(frozen=True)
class TestResult:
    test: str
    statistic: float
    p_value: float
    n: int


def paired_t_test(a: Sequence[float], b: Sequence[float]) -> TestResult:
    if ttest_rel is None:
        raise RuntimeError("scipy is required for paired t-test support.")
    stat, p = ttest_rel(a, b, nan_policy="omit")
    return TestResult("paired_t_test", float(stat), float(p), len(a))


def wilcoxon_signed_rank_test(a: Sequence[float], b: Sequence[float]) -> TestResult:
    if wilcoxon is None:
        raise RuntimeError("scipy is required for Wilcoxon test support.")
    stat, p = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided", mode="auto")
    return TestResult("wilcoxon_signed_rank", float(stat), float(p), len(a))


def exact_mcnemar_test(human_correct: Sequence[int], model_correct: Sequence[int]) -> TestResult:
    h = np.asarray(human_correct, dtype=int)
    m = np.asarray(model_correct, dtype=int)
    if h.shape != m.shape:
        raise ValueError("human_correct and model_correct must have the same length.")

    b = int(np.sum((h == 1) & (m == 0)))
    c = int(np.sum((h == 0) & (m == 1)))
    n = b + c

    if n == 0:
        return TestResult("mcnemar_exact", 0.0, 1.0, len(h))

    if binomtest is not None:
        p = float(binomtest(min(b, c), n=n, p=0.5, alternative="two-sided").pvalue)
    else:  # exact two-sided binomial fallback
        tail = sum(comb(n, k) for k in range(0, min(b, c) + 1)) / (2**n)
        p = float(min(1.0, 2.0 * tail))

    stat = float((abs(b - c) - 1) ** 2 / n) if n > 0 else 0.0
    return TestResult("mcnemar_exact", stat, p, len(h))


def permutation_p_value(differences: Iterable[float], n_permutations: int = 10000, seed: int = 0) -> float:
    diffs = np.asarray(list(differences), dtype=float)
    if diffs.size == 0:
        raise ValueError("differences cannot be empty.")

    observed = abs(np.mean(diffs))
    rng = np.random.default_rng(seed)
    more_extreme = 0
    for _ in range(n_permutations):
        flipped = diffs * rng.choice([-1.0, 1.0], size=diffs.size)
        if abs(np.mean(flipped)) >= observed:
            more_extreme += 1
    return float((more_extreme + 1) / (n_permutations + 1))
