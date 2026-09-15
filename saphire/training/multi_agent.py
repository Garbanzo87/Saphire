"""Selective optimisation of multi-agent systems.

`train_roles` updates the policy surface (router, exemplars, optionally prompt) of the *selected*
roles only, using the steps those roles produced; other roles are frozen. Returns a new
`AgentConfig` with the role artifacts pointed at the new files.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Optional

from ..environments.base import Environment
from ..sdk.exemplars import ExemplarStore
from ..sdk.multi_agent import ORCHESTRATOR, handoff_spec, role_rewards
from ..sdk.router import ToolRouter
from ..sdk.types import AgentConfig, Rollout
from ..signals.generate import router_examples


def role_tool_names(config: AgentConfig, role: str, envs: dict[str, Environment]) -> list[str]:
    if role == ORCHESTRATOR:
        return [handoff_spec(r, c).name for r, c in config.roles.items()] + list(config.orchestrator_tools)
    rc = config.roles[role]
    if rc.tool_names:
        return list(rc.tool_names)
    names: set[str] = set()
    for env in envs.values():
        for t in env.tools.specs():
            if not rc.tool_tags or set(rc.tool_tags) & set(t.tags):
                names.add(t.name)
    return sorted(names)


def selectable_roles(config: AgentConfig, roles: Optional[Iterable[str]] = None) -> list[str]:
    all_roles = [ORCHESTRATOR] + [r for r, c in config.roles.items() if c.trainable]
    if roles is None:
        return all_roles
    return [r for r in roles if r in all_roles]


def train_roles(config: AgentConfig, rollouts: list[Rollout], envs: dict[str, Environment], out_dir: str | Path,
                roles: Optional[Iterable[str]] = None, routers: Optional[dict[str, ToolRouter]] = None,
                exemplars: Optional[dict[str, ExemplarStore]] = None, learn_router: bool = True, learn_exemplars: bool = True,
                epochs: int = 30) -> tuple[AgentConfig, dict[str, Any]]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    routers = routers if routers is not None else {}
    exemplars = exemplars if exemplars is not None else {}
    report: dict[str, Any] = {"roles": {}, "frozen": []}
    cfg = config.model_copy(deep=True)
    selected = selectable_roles(config, roles)
    for role in [ORCHESTRATOR] + list(config.roles):
        if role not in selected:
            report["frozen"].append(role)
            continue
        rrep: dict[str, Any] = {}
        names = role_tool_names(config, role, envs)
        if learn_router and len(names) > 1:
            router = routers.get(role)
            if router is None or set(router.tool_names) != set(names):
                router = ToolRouter(names)
            ex = router_examples(rollouts, role=role)
            fit = router.fit(ex, epochs=epochs)
            path = router.save(out_dir / f"router_{role}")
            routers[role] = ToolRouter.load(path)
            rrep["router"] = {"artifact": path, "examples": len(ex), "positives": sum(1 for e in ex if e[2] > 0), "fit": fit}
            if role == ORCHESTRATOR:
                cfg.tool_router = path
            else:
                cfg.roles[role].tool_router = path
        if learn_exemplars:
            store = exemplars.get(role) or ExemplarStore()
            added = sum(1 for ro in rollouts if store.add_rollout(ro, role=role))
            path = store.save(out_dir / f"exemplars_{role}.json")
            exemplars[role] = store
            rrep["exemplars"] = {"artifact": path, "added": added, "size": len(store.items)}
            if role == ORCHESTRATOR:
                cfg.exemplar_store = path
            else:
                cfg.roles[role].exemplar_store = path
        report["roles"][role] = rrep
    # per-role reward summary for reporting / credit assignment
    agg: dict[str, list[float]] = {}
    for ro in rollouts:
        for role, r in role_rewards(ro).items():
            agg.setdefault(role, []).append(r["step_mean"])
    report["role_step_reward"] = {k: sum(v) / len(v) for k, v in agg.items() if v}
    return cfg, report
