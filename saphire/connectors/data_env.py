"""DataEnvironment: build a Saphire Environment from any ToolRegistry + hand-written / generated tasks.

    env = DataEnvironment("crm", registry=connector.registry(), seed_data=connector.seed_data(),
                          tasks=[TaskSpec(env_name="crm", instruction="...", expected={"tools": ["get_customers"],
                                                                                      "final_state": {...}})],
                          verifier=my_verifier)   # optional custom verifier
    register_environment(env.__class__)  # or pass the instance directly to evaluate()

The default `TaskVerifier` understands the same `expected` keys as the built-in environments:
  tools            – expected tool names (set F1 + coverage; `ordered: true` enforces order)
  answer_contains  – substrings that must appear in the final answer
  state_checks     – list of {"path": "tables.orders.0.status", "equals": "cancelled"} against the final state
  log_contains     – list of {"table": ..., "update": {...}} entries the state log must contain
"""
from __future__ import annotations

import copy
from typing import Any, Callable, Optional

from ..environments.base import Environment, EnvState
from ..environments.verify_utils import effective_calls, ordered_ok, step_tool_rewards, tool_selection_scores
from ..sdk.tools import ToolRegistry
from ..sdk.types import Reward, Rollout, TaskSpec


def _get_path(obj: Any, path: str) -> Any:
    for part in path.split("."):
        if isinstance(obj, list):
            obj = obj[int(part)]
        elif isinstance(obj, dict):
            obj = obj.get(part)
        else:
            return None
    return obj


class TaskVerifier:
    def __call__(self, task: TaskSpec, rollout: Rollout, state: EnvState) -> list[Reward]:
        exp = task.expected
        called = effective_calls(rollout)
        checks: dict[str, bool] = {}
        exp_tools = exp.get("tools", [])
        f1 = 1.0
        if exp_tools:
            f1, checks["all_expected_tools_called"] = tool_selection_scores(exp_tools, called)
            if exp.get("ordered"):
                checks["tool_order"] = ordered_ok(exp_tools, called)
        for s in exp.get("answer_contains", []):
            checks[f"answer_contains:{s}"] = s.lower() in rollout.final_answer.lower()
        for i, chk in enumerate(exp.get("state_checks", [])):
            val = _get_path(state.data, chk["path"])
            if "equals" in chk:
                checks[f"state:{chk['path']}"] = val == chk["equals"]
            elif "contains" in chk:
                checks[f"state:{chk['path']}"] = chk["contains"] in (val or [])
        for i, entry in enumerate(exp.get("log_contains", [])):
            checks[f"log:{i}"] = any(all(x.get(k) == v for k, v in entry.items()) for x in state.data.get("log", []) + state.log)
        n_err = sum(1 for s in rollout.steps for r in s.tool_results if r.error)
        success = all(checks.values()) and rollout.status.value == "succeeded"
        rewards = [Reward(value=float(success), source="verifier", name="task_success",
                          rationale="; ".join(f"{k}={'ok' if v else 'FAIL'}" for k, v in checks.items()), metadata={"checks": checks}),
                   Reward(value=f1, source="verifier", name="tool_selection_f1"),
                   Reward(value=1.0 if n_err == 0 else max(0.0, 1 - 0.5 * n_err), source="verifier", name="tool_error_free")]
        if exp_tools:
            rewards.append(Reward(value=min(1.0, len(exp_tools) / max(1, len(called))), source="verifier", name="step_efficiency"))
            rewards.extend(step_tool_rewards(rollout, exp_tools))
        return rewards


class DataEnvironment(Environment):
    def __init__(self, name: str, registry: ToolRegistry, seed_data: Optional[dict[str, Any]] = None,
                 tasks: Optional[list[TaskSpec]] = None, verifier: Optional[Callable[[TaskSpec, Rollout, EnvState], list[Reward]]] = None,
                 description: str = ""):
        super().__init__()
        self.name = name
        self.description = description or f"Data environment '{name}' with {len(registry)} tools"
        self._registry = registry
        self._seed = seed_data or {}
        self._tasks = tasks or []
        self._verifier = verifier or TaskVerifier()

    def build_tools(self) -> ToolRegistry:
        return self._registry

    def seed_data(self) -> dict[str, Any]:
        return self._seed

    def generate_tasks(self, n: int = 20, seed: int = 0, families: list[str] | None = None) -> list[TaskSpec]:
        tasks = [t for t in self._tasks if not families or (t.tags and t.tags[0] in families)]
        return [t.model_copy(update={"env_name": self.name}) for t in tasks[:n]]

    def verify(self, task: TaskSpec, rollout: Rollout, state: EnvState) -> list[Reward]:
        return self._verifier(task, rollout, state)

    def reset(self, task: TaskSpec) -> EnvState:
        return EnvState(copy.deepcopy(self._seed))
