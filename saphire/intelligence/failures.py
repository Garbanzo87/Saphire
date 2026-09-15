"""Failure clustering: group failed rollouts into signatures a human (or the optimiser) can act on."""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Iterable, Optional

from ..sdk.attribution import blame
from ..sdk.types import Rollout


def failure_signature(ro: Rollout, family: str = "") -> Optional[tuple[str, str, str, str]]:
    b = blame(ro)
    if b["kind"] == "none":
        return None
    return (b["kind"], b["role"] or "main", b["tool"] or "-", family)


def cluster_failures(rollouts: Iterable[Rollout], families: Optional[dict[str, str]] = None, baseline: Optional[Iterable[Rollout]] = None,
                     top: int = 10) -> dict[str, Any]:
    """Cluster by (fault kind, role, tool, task family). With `baseline` rollouts, mark clusters that are new or growing."""
    families = families or {}
    ros = list(rollouts)
    clusters: dict[tuple, list[Rollout]] = defaultdict(list)
    for ro in ros:
        sig = failure_signature(ro, families.get(ro.task_id, ""))
        if sig:
            clusters[sig].append(ro)
    base_counts: Counter = Counter()
    n_base = 0
    if baseline is not None:
        base_list = list(baseline)
        n_base = len(base_list)
        for ro in base_list:
            sig = failure_signature(ro, families.get(ro.task_id, ""))
            if sig:
                base_counts[sig] += 1
    n = len(ros)
    items = []
    for sig, members in sorted(clusters.items(), key=lambda kv: -len(kv[1])):
        kind, role, tool, fam = sig
        rate = len(members) / n if n else 0.0
        base_rate = (base_counts[sig] / n_base) if n_base else None
        # wrong tools: what was called instead of what was expected, for the router / prompt optimiser
        confusions = Counter()
        for ro in members:
            b = blame(ro)
            if b["kind"] == "wrong_tool":
                confusions[b["tool"]] += 1
        items.append({"kind": kind, "role": role, "tool": tool, "family": fam, "count": len(members), "rate": rate,
                      "baseline_rate": base_rate, "trend": ("new" if base_rate == 0 else "growing" if base_rate is not None and rate > 1.5 * base_rate else "stable") if base_rate is not None else None,
                      "examples": [m.id for m in members[:5]], "tasks": sorted({m.task_id for m in members})[:5]})
    failed = sum(len(v) for v in clusters.values())
    return {"n_rollouts": n, "n_failed": failed, "failure_rate": failed / n if n else 0.0, "clusters": items[:top],
            "by_kind": Counter(sig[0] for sig, m in clusters.items() for _ in m), "by_role": Counter(sig[1] for sig, m in clusters.items() for _ in m)}
