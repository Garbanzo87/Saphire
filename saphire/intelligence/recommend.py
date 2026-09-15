"""Turn drift, failure clusters, coverage and attribution into ranked, executable recommendations.

Every recommendation carries an `action` the API/dashboard can execute directly:
  {"kind": "training", "algorithm": "online"|"router"|"exemplars"|"prompt_opt", "params": {...}}
  {"kind": "mine_tasks"}   {"kind": "attribution", "params": {...}}   {"kind": "gate", ...}   {"kind": "investigate"}
"""
from __future__ import annotations

from typing import Any, Optional


def recommend(drift: Optional[dict[str, Any]] = None, failures: Optional[dict[str, Any]] = None, coverage: Optional[dict[str, Any]] = None,
              attribution: Optional[dict[str, Any]] = None, multi_agent: bool = False) -> list[dict[str, Any]]:
    recs: list[dict[str, Any]] = []

    def add(priority: float, title: str, why: str, action: dict[str, Any]) -> None:
        recs.append({"priority": round(priority, 3), "title": title, "why": why, "action": action})

    if drift:
        for a in drift.get("alerts", []):
            if a["type"] == "regression" and a["metric"] == "task_success":
                add(0.95, "Success rate regressed in production — block promotions and retrain",
                    a["message"], {"kind": "training", "algorithm": "online", "params": {"iterations": 3, "learn_prompt_every": 1}})
            elif a["type"] == "regression" and a["metric"] == "tool_selection_f1":
                add(0.9, "Tool selection degraded — retrain the router on recent rollouts", a["message"],
                    {"kind": "training", "algorithm": "router", "params": {"max_rollouts": 2000}})
            elif a["type"] == "regression" and a["metric"] == "context_preservation":
                add(0.85, "Context preservation dropped — refresh exemplars and rules", a["message"],
                    {"kind": "training", "algorithm": "exemplars", "params": {}})
            elif a["type"] == "family_regression":
                add(0.85, f"Task family '{a['family']}' broke — targeted prompt optimisation", a["message"],
                    {"kind": "training", "algorithm": "prompt_opt", "params": {"iterations": 4, "families": [a["family"]]}})
            elif a["type"] == "errors":
                add(0.8, "Tool/infrastructure errors up — inspect failing tools before training", a["message"], {"kind": "investigate", "what": "tool_errors"})
            elif a["type"] == "inflation":
                add(0.5, f"{a['metric']} inflated {a['ratio']:.2f}x — check step efficiency and tool latency", a["message"], {"kind": "investigate", "what": a["metric"]})
            elif a["type"] == "tool_shift":
                add(0.4, "Tool usage distribution shifted — mine new production tasks into the eval set", a["message"], {"kind": "mine_tasks"})
    if failures:
        for c in failures.get("clusters", [])[:5]:
            if c["kind"] == "wrong_tool" and c["rate"] >= 0.05:
                role = c["role"]
                params = {"max_rollouts": 2000}
                if multi_agent and role not in ("main", None):
                    params["optimize_roles"] = [role]
                add(0.7 + c["rate"], f"'{c['tool']}' is the wrong tool in {c['rate']:.0%} of rollouts ({c['family'] or 'all families'})",
                    f"cluster of {c['count']} failures{' — ' + c['trend'] if c.get('trend') in ('new', 'growing') else ''}",
                    {"kind": "training", "algorithm": "router", "params": params})
            elif c["kind"] == "bad_handoff" and c["rate"] >= 0.05:
                add(0.7 + c["rate"], f"Orchestrator mis-routes {c['rate']:.0%} of requests", f"{c['count']} bad hand-offs",
                    {"kind": "training", "algorithm": "online", "params": {"optimize_roles": ["orchestrator"], "iterations": 3}})
            elif c["kind"] == "tool_error" and c["rate"] >= 0.05:
                add(0.6 + c["rate"], f"Tool '{c['tool']}' errors in {c['rate']:.0%} of rollouts (bad arguments or flaky tool)",
                    f"{c['count']} failures", {"kind": "investigate", "what": f"tool:{c['tool']}"})
            elif c["kind"] in ("timeout", "incomplete") and c["rate"] >= 0.05:
                add(0.5 + c["rate"], f"{c['rate']:.0%} of rollouts end {c['kind']} — raise max_steps or add exemplars", f"{c['count']} failures",
                    {"kind": "training", "algorithm": "exemplars", "params": {}})
    if coverage and coverage.get("uncovered_traffic_share", 0) > 0.2:
        add(0.65, f"{coverage['uncovered_traffic_share']:.0%} of production traffic has no matching eval task — mine tasks",
            f"{coverage['uncovered_patterns']} uncovered tool patterns", {"kind": "mine_tasks"})
    if attribution:
        roles = attribution.get("ablation", {}).get("roles", {}) or {}
        best = sorted(roles.items(), key=lambda kv: -(kv[1].get("headroom") or 0))[:2]
        if best and (best[0][1].get("headroom") or 0) > 0.05:
            add(0.75, f"Optimise role(s) {[r for r, _ in best]} — largest counterfactual headroom",
                ", ".join(f"{r}: +{e.get('headroom', 0):.0%}" for r, e in best),
                {"kind": "training", "algorithm": "online", "params": {"optimize_roles": [r for r, _ in best], "iterations": 3}})
    elif multi_agent and failures and failures.get("n_failed", 0) >= 10:
        add(0.45, "Run counterfactual attribution to find which role to optimise", "multi-agent system with enough failures to attribute",
            {"kind": "attribution", "params": {"reference_model": None, "degraded_model": None}})
    if not recs:
        add(0.1, "No action needed", "no regressions, clusters or coverage gaps above thresholds", {"kind": "none"})
    recs.sort(key=lambda r: -r["priority"])
    return recs
