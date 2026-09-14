"""Environment abstraction.

An Environment bundles (1) a tool registry, (2) per-task world state built from first-party
data, (3) a task generator, (4) an optional simulated user for multi-turn tasks and (5) a
verifier that turns the final world state + trajectory into ground-truth rewards.

Companies plug their own systems in by subclassing `Environment` or by using the connectors
(`saphire.connectors`) to build a `ToolRegistry` from SQL tables, OpenAPI specs or MCP servers.
"""
from __future__ import annotations

import copy
from typing import Any, Optional

from ..sdk.tools import ToolRegistry
from ..sdk.types import Reward, Rollout, TaskSpec


class EnvState:
    """Mutable world state for one rollout (a deep copy of the environment's seed data)."""

    def __init__(self, data: dict[str, Any]):
        self.data = data
        self.log: list[dict[str, Any]] = []  # side-effects performed by tools

    def record(self, action: str, **kw: Any) -> None:
        self.log.append({"action": action, **kw})


class Environment:
    name: str = "base"
    description: str = ""

    def __init__(self) -> None:
        self._registry: Optional[ToolRegistry] = None

    # ---- to override ----
    def build_tools(self) -> ToolRegistry:
        raise NotImplementedError

    def seed_data(self) -> dict[str, Any]:
        return {}

    def generate_tasks(self, n: int = 20, seed: int = 0) -> list[TaskSpec]:
        raise NotImplementedError

    def verify(self, task: TaskSpec, rollout: Rollout, state: EnvState) -> list[Reward]:
        """Ground-truth verification. Default: no reward."""
        return []

    # ---- provided ----
    @property
    def tools(self) -> ToolRegistry:
        if self._registry is None:
            self._registry = self.build_tools()
        return self._registry

    def reset(self, task: TaskSpec) -> EnvState:
        return EnvState(copy.deepcopy(self.seed_data()))

    def next_user_turn(self, task: TaskSpec, turn: int) -> Optional[str]:
        """Simulated user: reveal the next scripted follow-up (None when the script is exhausted)."""
        if turn < len(task.user_script):
            return task.user_script[turn]
        return None

    def info(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "n_tools": len(self.tools),
                "tools": self.tools.names()}


_REGISTRY: dict[str, type[Environment]] = {}


def register_environment(cls: type[Environment]) -> type[Environment]:
    _REGISTRY[cls.name] = cls
    return cls


def get_environment(name: str, **kw: Any) -> Environment:
    _ensure_builtin()
    if name not in _REGISTRY:
        raise KeyError(f"unknown environment '{name}'. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**kw)


def list_environments() -> list[str]:
    _ensure_builtin()
    return sorted(_REGISTRY)


def _ensure_builtin() -> None:
    from . import (
        data_ops,  # noqa: F401
        support_desk,  # noqa: F401  (registers on import)
    )
