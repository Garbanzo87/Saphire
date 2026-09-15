"""Multi-agent attribution: who deserves credit or blame in a multi-agent rollout?

Three complementary views, from cheap to expensive:

1. **Blame analysis** (`blame`, per rollout, free): the first faulty step — wrong tool, tool error, bad hand-off,
   step limit — and the role that produced it. Aggregated over an evaluation this gives each role's *fault share*.
2. **Advantage credit** (`advantage_credit`, over stored rollouts, free): for every role, success rate when the role
   made no mistake vs. when it made at least one; the difference is the role's observed leverage on the outcome.
3. **Counterfactual attribution** (`role_ablation`, `shapley_attribution`, needs new rollouts): swap one role's policy
   for a *reference* (stronger) or *degraded* policy and re-run the tasks. Headroom = gain from upgrading the role,
   criticality = loss from degrading it, and Monte-Carlo Shapley values over "upgrade coalitions" give a fair split of
   the total improvement between roles. These use the same machinery as evaluation, so they run on any backend.

All results are plain dicts so the server can store them on jobs and the dashboard can render them.
"""
from __future__ import annotations

import itertools
import random
from collections import defaultdict
from typing import Any, Iterable, Optional

from .multi_agent import ORCHESTRATOR
from .types import AgentConfig, Rollout, TaskSpec

HANDOFF = "delegate_to_"


def blame(rollout: Rollout) -> dict[str, Any]:
    """Locate the first fault in a rollout. Uses step-level `tool_correct` rewards and tool errors."""
    step_rewards: dict[int, list[Any]] = defaultdict(list)
    for r in rollout.rewards:
        if r.step_index is not None:
            step_rewards[r.step_index].append(r)
    success = rollout.total_reward >= 0.99
    for s in rollout.steps:
        for tr in s.tool_results:
            if tr.error and not tr.name.startswith(HANDOFF):
                return {"role": s.role, "step_index": s.index, "tool": tr.name, "kind": "tool_error", "detail": tr.error[:200], "success": success}
        for r in step_rewards.get(s.index, []):
            if r.name == "tool_correct" and r.value < 0:
                kind = "bad_handoff" if r.metadata.get("handoff") else "wrong_tool"
                return {"role": s.role, "step_index": s.index, "tool": r.metadata.get("tool"), "kind": kind, "detail": "", "success": success}
    if rollout.status.value in ("timeout", "error"):
        last = rollout.steps[-1].role if rollout.steps else ORCHESTRATOR
        return {"role": last, "step_index": len(rollout.steps) - 1, "tool": None, "kind": rollout.status.value, "detail": rollout.metadata.get("error", ""), "success": success}
    if not success:
        return {"role": ORCHESTRATOR, "step_index": None, "tool": None, "kind": "incomplete", "detail": "expected work not done", "success": success}
    return {"role": None, "step_index": None, "tool": None, "kind": "none", "detail": "", "success": success}


def advantage_credit(rollouts: Iterable[Rollout]) -> dict[str, Any]:
    """Per-role fault share and outcome advantage over a set of rollouts."""
    rollouts = list(rollouts)
    roles: set[str] = set()
    faults: dict[str, int] = defaultdict(int)
    fault_kinds: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    clean_success: dict[str, list[float]] = defaultdict(list)
    faulty_success: dict[str, list[float]] = defaultdict(list)
    n_failed = 0
    for ro in rollouts:
        succ = 1.0 if ro.total_reward >= 0.99 else 0.0
        if not succ:
            n_failed += 1
        step_role = {s.index: s.role for s in ro.steps}
        roles |= set(step_role.values())
        bad_roles = {step_role[r.step_index] for r in ro.rewards if r.step_index is not None and r.name == "tool_correct" and r.value < 0 and r.step_index in step_role}
        bad_roles |= {s.role for s in ro.steps for tr in s.tool_results if tr.error and not tr.name.startswith(HANDOFF)}
        for role in set(step_role.values()):
            (faulty_success if role in bad_roles else clean_success)[role].append(succ)
        b = blame(ro)
        if b["role"]:
            faults[b["role"]] += 1
            fault_kinds[b["role"]][b["kind"]] += 1
    out = {}
    for role in sorted(roles):
        cs, fs = clean_success[role], faulty_success[role]
        out[role] = {
            "rollouts": len(cs) + len(fs),
            "fault_share": faults[role] / n_failed if n_failed else 0.0,
            "first_faults": faults[role],
            "fault_kinds": dict(fault_kinds[role]),
            "success_when_clean": sum(cs) / len(cs) if cs else None,
            "success_when_faulty": sum(fs) / len(fs) if fs else None,
            "advantage": ((sum(cs) / len(cs)) - (sum(fs) / len(fs))) if cs and fs else None,
            "mistake_rate": len(fs) / (len(cs) + len(fs)) if cs or fs else 0.0,
        }
    return {"n_rollouts": len(rollouts), "n_failed": n_failed, "roles": out}


# ---------------------------------------------------------------------------
# Counterfactual attribution
# ---------------------------------------------------------------------------
def _with_role_models(config: AgentConfig, overrides: dict[str, str]) -> AgentConfig:
    cfg = config.model_copy(deep=True)
    for role, model in overrides.items():
        if role == ORCHESTRATOR:
            cfg.model = model  # orchestrator inherits the system model
        else:
            cfg.roles[role].model = model
    return cfg


def _success_rate(config: AgentConfig, tasks: list[TaskSpec], envs, k: int, backend: Optional[str], workers: Optional[int]) -> float:
    if backend:
        from ..distributed.rollouts import evaluate_distributed

        return evaluate_distributed(config, tasks, k=k, backend=backend, workers=workers).metrics["task_success"]
    from ..evaluation.runner import evaluate
    from .multi_agent import build_agent

    return evaluate(build_agent(config), tasks, envs=envs, k=k).metrics["task_success"]


def role_ablation(config: AgentConfig, tasks: list[TaskSpec], envs=None, reference_model: Optional[str] = None,
                  degraded_model: Optional[str] = None, roles: Optional[list[str]] = None, k: int = 1,
                  backend: Optional[str] = None, workers: Optional[int] = None) -> dict[str, Any]:
    """Headroom (upgrade one role to `reference_model`) and criticality (degrade one role to `degraded_model`)."""
    roles = roles or [ORCHESTRATOR] + list(config.roles)
    base = _success_rate(config, tasks, envs, k, backend, workers)
    out: dict[str, Any] = {"baseline": base, "roles": {}, "reference_model": reference_model, "degraded_model": degraded_model}
    for role in roles:
        entry: dict[str, Any] = {}
        if reference_model:
            up = _success_rate(_with_role_models(config, {role: reference_model}), tasks, envs, k, backend, workers)
            entry["upgraded"] = up
            entry["headroom"] = up - base
        if degraded_model:
            down = _success_rate(_with_role_models(config, {role: degraded_model}), tasks, envs, k, backend, workers)
            entry["degraded"] = down
            entry["criticality"] = base - down
        out["roles"][role] = entry
    if reference_model:
        all_up = _success_rate(_with_role_models(config, {r: reference_model for r in roles}), tasks, envs, k, backend, workers)
        out["all_upgraded"] = all_up
        out["total_headroom"] = all_up - base
    return out


def shapley_attribution(config: AgentConfig, tasks: list[TaskSpec], envs=None, reference_model: str = "mock",
                        roles: Optional[list[str]] = None, n_permutations: int = 6, k: int = 1, seed: int = 0,
                        backend: Optional[str] = None, workers: Optional[int] = None, exact: bool = False) -> dict[str, Any]:
    """Monte-Carlo (or exact for small role sets) Shapley values of upgrading each role to `reference_model`.

    v(S) = success rate with roles in S upgraded. phi_r = E_pi[ v(S_pi(r) ∪ {r}) - v(S_pi(r)) ]. Sums to v(all) - v(∅).
    Coalition values are cached so the number of evaluations is far below n_permutations * |roles|.
    """
    roles = roles or [ORCHESTRATOR] + list(config.roles)
    rng = random.Random(seed)
    cache: dict[frozenset, float] = {}

    def v(coalition: frozenset) -> float:
        if coalition not in cache:
            cache[coalition] = _success_rate(_with_role_models(config, {r: reference_model for r in coalition}), tasks, envs, k, backend, workers)
        return cache[coalition]

    phi: dict[str, float] = {r: 0.0 for r in roles}
    perms = list(itertools.permutations(roles)) if exact or len(roles) <= 3 else [rng.sample(roles, len(roles)) for _ in range(n_permutations)]
    for perm in perms:
        s: frozenset = frozenset()
        for r in perm:
            before = v(s)
            s = s | {r}
            phi[r] += v(s) - before
    phi = {r: val / len(perms) for r, val in phi.items()}
    return {"baseline": v(frozenset()), "all_upgraded": v(frozenset(roles)), "shapley": phi, "n_permutations": len(perms),
            "n_evaluations": len(cache), "reference_model": reference_model,
            "share": {r: (val / sum(phi.values())) if sum(phi.values()) else 0.0 for r, val in phi.items()}}
