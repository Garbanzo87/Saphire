"""Lightweight learners updated online: the tool router and the exemplar store.

Both train in seconds on CPU from reward-labelled rollouts and are stored as artifacts that an
`AgentConfig` points at (`tool_router`, `exemplar_store`).
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from ..environments.base import Environment
from ..sdk.exemplars import ExemplarStore
from ..sdk.router import ToolRouter
from ..sdk.types import Rollout
from ..signals.generate import router_examples


def train_router(rollouts: Iterable[Rollout], envs: dict[str, Environment], out_path: str | Path,
                 router: ToolRouter | None = None, epochs: int = 30, lr: float = 0.5) -> dict:
    """Fit (or continue fitting) a ToolRouter over the union of all environments' tools."""
    rollouts = list(rollouts)
    names = sorted({n for e in envs.values() for n in e.tools.names()})
    if router is None:
        router = ToolRouter(names)
    elif set(router.tool_names) != set(names):  # catalogue changed: re-init preserving nothing (rare)
        router = ToolRouter(names)
    ex = router_examples(rollouts)
    stats = router.fit(ex, epochs=epochs, lr=lr)
    from .tools_v2 import tool_stats

    router.set_prior(tool_stats(rollouts))
    # held-in top-k accuracy per env for reporting
    acc = {}
    for env_name, env in envs.items():
        env_ex = [(q, n, w) for q, n, w in ex if n in env.tools]
        if env_ex:
            acc[env_name] = {"top1": router.accuracy(env_ex, env.tools.specs(), 1), "top8": router.accuracy(env_ex, env.tools.specs(), 8)}
    path = router.save(out_path)
    return {"artifact": path, "examples": len(ex), "positives": sum(1 for e in ex if e[2] > 0), "fit": stats, "accuracy": acc}


def update_exemplars(rollouts: Iterable[Rollout], out_path: str | Path, store: ExemplarStore | None = None) -> dict:
    store = store or ExemplarStore()
    added = sum(1 for ro in rollouts if store.add_rollout(ro))
    path = store.save(out_path)
    return {"artifact": path, "added": added, "size": len(store.items)}
