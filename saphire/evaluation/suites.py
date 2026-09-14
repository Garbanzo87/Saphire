"""Named evaluation suites built from environment task families.

  tool_selection        – catalogue with distractors; measures tool_selection_f1 and success
  context_preservation  – multi-turn + cross-step argument passing
  long_horizon          – 6-step ordered workflows
  cross_env             – both environments mixed
  full                  – everything
"""
from __future__ import annotations

from ..environments.base import get_environment
from ..sdk.types import TaskSpec

SUITES: dict[str, dict] = {
    "tool_selection": {"support_desk": ["lookup", "cancel_email", "refund_resolve", "kb_email"], "data_ops": ["metric_alert", "report_post"]},
    "context_preservation": {"support_desk": ["multiturn", "find_list"]},
    "long_horizon": {"support_desk": ["damage_workflow"], "data_ops": ["quality_schedule"]},
    "cross_env": {"support_desk": None, "data_ops": None},
    "full": {"support_desk": None, "data_ops": None},
    "smoke": {"support_desk": ["lookup", "cancel_email"]},
}


def build_suite(name: str, n_per_env: int = 14, seed: int = 0) -> list[TaskSpec]:
    if name not in SUITES:
        raise KeyError(f"unknown suite {name}; available: {list(SUITES)}")
    tasks: list[TaskSpec] = []
    for env_name, fams in SUITES[name].items():
        env = get_environment(env_name)
        tasks.extend(env.generate_tasks(n_per_env, seed=seed, families=fams))
    return tasks


def list_suites() -> list[str]:
    return list(SUITES)
