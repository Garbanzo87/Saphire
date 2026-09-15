"""Drift and regression detection between a baseline window and a recent window of production records.

Input records are `RolloutRecord`-shaped dicts (what the server stores per rollout): task_success, latency_ms,
prompt_tokens/completion_tokens, called_tools, family, status, tool_error_free. Detects:

* metric regressions (success, tool-selection, context, error rate) with permutation-test p-values,
* latency / cost inflation (p95 ratio),
* tool-usage distribution shift (Jensen–Shannon divergence) and new / vanished tools,
* task-mix shift (family distribution),
* per-family regressions (the family that broke).
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Any, Iterable

from ..evaluation import metrics as M

METRICS = ("task_success", "tool_selection_f1", "context_preservation", "tool_error_free", "judge_score")


def _vals(records: list[dict[str, Any]], key: str) -> list[float]:
    return [float(r[key]) for r in records if r.get(key) is not None]


def _js_divergence(p: Counter, q: Counter) -> float:
    keys = set(p) | set(q)
    sp, sq = sum(p.values()) or 1, sum(q.values()) or 1
    P = {k: p.get(k, 0) / sp for k in keys}
    Q = {k: q.get(k, 0) / sq for k in keys}
    Mx = {k: (P[k] + Q[k]) / 2 for k in keys}

    def kl(a, b):
        return sum(a[k] * math.log(a[k] / b[k], 2) for k in keys if a[k] > 0)

    return 0.5 * kl(P, Mx) + 0.5 * kl(Q, Mx)


def drift_report(baseline: Iterable[dict[str, Any]], recent: Iterable[dict[str, Any]], p_value: float = 0.05,
                 min_effect: float = 0.03, latency_ratio: float = 1.3, js_threshold: float = 0.1) -> dict[str, Any]:
    base, rec = list(baseline), list(recent)
    alerts: list[dict[str, Any]] = []
    out: dict[str, Any] = {"n_baseline": len(base), "n_recent": len(rec), "metrics": {}, "alerts": alerts}
    if not base or not rec:
        out["note"] = "not enough data in one of the windows"
        return out
    for m in METRICS:
        a, b = _vals(base, m), _vals(rec, m)
        if len(a) < 5 or len(b) < 5:
            continue
        delta = M.mean(b) - M.mean(a)
        p = M.permutation_test(a, b)
        out["metrics"][m] = {"baseline": M.mean(a), "recent": M.mean(b), "delta": delta, "p_value": p}
        if delta < -min_effect and p < p_value:
            alerts.append({"type": "regression", "metric": m, "delta": delta, "p_value": p, "severity": "high" if delta < -0.1 else "medium",
                           "message": f"{m} dropped {abs(delta):.1%} (p={p:.3f})"})
    # error / timeout rate
    ea = M.mean(1.0 if r.get("status") in ("error", "timeout") else 0.0 for r in base)
    eb = M.mean(1.0 if r.get("status") in ("error", "timeout") else 0.0 for r in rec)
    out["metrics"]["error_rate"] = {"baseline": ea, "recent": eb, "delta": eb - ea}
    if eb - ea > min_effect:
        alerts.append({"type": "errors", "metric": "error_rate", "delta": eb - ea, "severity": "high", "message": f"error/timeout rate up {eb - ea:.1%}"})
    # latency & cost
    for key, label in (("latency_ms", "latency_p95"), ("tokens", "tokens_per_task")):
        if key == "tokens":
            a = [float(r.get("prompt_tokens", 0) + r.get("completion_tokens", 0)) for r in base]
            b = [float(r.get("prompt_tokens", 0) + r.get("completion_tokens", 0)) for r in rec]
            va, vb = M.mean(a), M.mean(b)
        else:
            va, vb = M.percentile(_vals(base, key), 95), M.percentile(_vals(rec, key), 95)
        ratio = (vb / va) if va else 1.0
        out["metrics"][label] = {"baseline": va, "recent": vb, "ratio": ratio}
        if ratio > latency_ratio:
            alerts.append({"type": "inflation", "metric": label, "ratio": ratio, "severity": "medium", "message": f"{label} up {ratio:.2f}x"})
    # distribution shifts
    ta = Counter(t for r in base for t in r.get("called_tools", []))
    tb = Counter(t for r in rec for t in r.get("called_tools", []))
    js = _js_divergence(ta, tb)
    out["tool_shift"] = {"js_divergence": js, "new_tools": sorted(set(tb) - set(ta)), "vanished_tools": sorted(set(ta) - set(tb)),
                         "top_recent": tb.most_common(8)}
    if js > js_threshold:
        alerts.append({"type": "tool_shift", "metric": "js_divergence", "value": js, "severity": "low", "message": f"tool usage distribution shifted (JS={js:.2f})"})
    fa, fb = Counter(r.get("family", "") for r in base), Counter(r.get("family", "") for r in rec)
    out["task_mix_shift"] = {"js_divergence": _js_divergence(fa, fb), "recent": fb.most_common(8)}
    # per-family success regressions
    fam: dict[str, Any] = {}
    for f in set(fa) | set(fb):
        a = [float(r["task_success"]) for r in base if r.get("family") == f and r.get("task_success") is not None]
        b = [float(r["task_success"]) for r in rec if r.get("family") == f and r.get("task_success") is not None]
        if len(a) >= 5 and len(b) >= 5:
            d = M.mean(b) - M.mean(a)
            fam[f] = {"baseline": M.mean(a), "recent": M.mean(b), "delta": d, "n_recent": len(b)}
            if d < -0.1 and M.permutation_test(a, b) < p_value:
                alerts.append({"type": "family_regression", "family": f, "delta": d, "severity": "high", "message": f"family '{f}' success dropped {abs(d):.1%}"})
    out["by_family"] = fam
    out["healthy"] = not any(a["severity"] == "high" for a in alerts)
    return out
