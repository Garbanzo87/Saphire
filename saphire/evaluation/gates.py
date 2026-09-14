"""Deployment gates: decide whether a candidate agent version may be promoted.

A gate combines absolute thresholds (e.g. task_success >= 0.8, error_rate <= 0.02) with a
non-regression test against the currently deployed baseline (delta >= -epsilon with a
permutation-test p-value guard so noisy small suites don't block or promote by chance).
"""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field

from .metrics import compare
from .runner import EvalResult


class GatePolicy(BaseModel):
    name: str = "default"
    min_metrics: dict[str, float] = Field(default_factory=lambda: {"task_success": 0.7, "tool_selection_f1": 0.8})
    max_metrics: dict[str, float] = Field(default_factory=lambda: {"error_rate": 0.05, "latency_ms_p95": 60000})
    non_regression_metrics: list[str] = Field(default_factory=lambda: ["task_success", "tool_selection_f1"])
    max_regression: float = 0.02  # allow at most 2pt drop
    require_significant_improvement: bool = False
    p_value: float = 0.1


class GateDecision(BaseModel):
    passed: bool
    reasons: list[str]
    checks: dict[str, Any]
    comparison: dict[str, Any] = Field(default_factory=dict)


def evaluate_gate(candidate: EvalResult, policy: GatePolicy, baseline: Optional[EvalResult] = None) -> GateDecision:
    reasons: list[str] = []
    checks: dict[str, Any] = {}
    m = candidate.metrics
    for k, v in policy.min_metrics.items():
        ok = m.get(k, 0.0) >= v
        checks[f"min:{k}"] = {"value": m.get(k), "threshold": v, "ok": ok}
        if not ok:
            reasons.append(f"{k}={m.get(k, 0.0):.3f} < {v}")
    for k, v in policy.max_metrics.items():
        ok = m.get(k, 0.0) <= v
        checks[f"max:{k}"] = {"value": m.get(k), "threshold": v, "ok": ok}
        if not ok:
            reasons.append(f"{k}={m.get(k, 0.0):.3f} > {v}")
    cmp: dict[str, Any] = {}
    if baseline is not None:
        cmp = compare(baseline.model_dump(), candidate.model_dump(), policy.non_regression_metrics)
        for k, c in cmp.items():
            regressed = c["delta"] < -policy.max_regression and c["p_value"] < policy.p_value
            checks[f"non_regression:{k}"] = {"delta": c["delta"], "p_value": c["p_value"], "ok": not regressed}
            if regressed:
                reasons.append(f"{k} regressed by {-c['delta']:.3f} (p={c['p_value']:.3f})")
            if policy.require_significant_improvement:
                improved = c["delta"] > 0 and c["p_value"] < policy.p_value
                checks[f"improvement:{k}"] = {"delta": c["delta"], "p_value": c["p_value"], "ok": improved}
                if not improved:
                    reasons.append(f"{k} not significantly improved (delta={c['delta']:.3f}, p={c['p_value']:.3f})")
    return GateDecision(passed=not reasons, reasons=reasons, checks=checks, comparison=cmp)
