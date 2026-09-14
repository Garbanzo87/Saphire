"""Metric aggregation, confidence intervals and A/B comparison for evaluation results."""
from __future__ import annotations

import math
import random
from typing import Any, Iterable, Sequence

import numpy as np


def percentile(values: Sequence[float], p: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=float), p))


def mean(values: Iterable[float]) -> float:
    v = list(values)
    return float(sum(v) / len(v)) if v else 0.0


def bootstrap_ci(values: Sequence[float], n_boot: int = 1000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    arr = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    means = arr[idx].mean(axis=1)
    return float(np.percentile(means, 100 * alpha / 2)), float(np.percentile(means, 100 * (1 - alpha / 2)))


def permutation_test(a: Sequence[float], b: Sequence[float], n_perm: int = 2000, seed: int = 0) -> float:
    """Two-sided permutation test p-value for difference in means."""
    if not a or not b:
        return 1.0
    a, b = list(a), list(b)
    obs = abs(mean(a) - mean(b))
    pooled = a + b
    rng = random.Random(seed)
    n = len(a)
    hits = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        if abs(mean(pooled[:n]) - mean(pooled[n:])) >= obs - 1e-12:
            hits += 1
    return (hits + 1) / (n_perm + 1)


def pass_hat_k(successes_per_task: dict[str, list[float]], k: int) -> float:
    """pass^k (tau-bench): probability that *all* k trials of a task succeed, averaged over tasks.

    Uses the unbiased estimator C(c,k)/C(n,k) where c = #successes among n trials.
    """
    vals = []
    for trials in successes_per_task.values():
        n = len(trials)
        c = int(sum(1 for t in trials if t >= 1.0))
        if n < k:
            continue
        vals.append(math.comb(c, k) / math.comb(n, k))
    return mean(vals)


def pass_at_k(successes_per_task: dict[str, list[float]], k: int) -> float:
    """pass@k: probability that at least one of k trials succeeds."""
    vals = []
    for trials in successes_per_task.values():
        n = len(trials)
        c = int(sum(1 for t in trials if t >= 1.0))
        if n < k:
            continue
        vals.append(1.0 - math.comb(n - c, k) / math.comb(n, k))
    return mean(vals)


def compare(baseline: dict[str, Any], candidate: dict[str, Any], metric_keys: Sequence[str]) -> dict[str, Any]:
    """Compare two EvalResult summaries (as dicts with `per_rollout` lists)."""
    out: dict[str, Any] = {}
    for key in metric_keys:
        a = [r.get(key) for r in baseline.get("per_rollout", []) if r.get(key) is not None]
        b = [r.get(key) for r in candidate.get("per_rollout", []) if r.get(key) is not None]
        if not a or not b:
            continue
        out[key] = {"baseline": mean(a), "candidate": mean(b), "delta": mean(b) - mean(a),
                    "p_value": permutation_test(a, b), "ci_candidate": bootstrap_ci(b)}
    return out
