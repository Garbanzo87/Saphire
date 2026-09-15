"""Training-signal generation: rollouts + rewards -> datasets for every learner.

  router_examples(rollouts)      -> (query, tool, weight)          for the ToolRouter
  sft_examples(rollouts)         -> {prompt, completion}           successful trajectories, step level
  preference_pairs(rollouts)     -> {prompt, chosen, rejected}     DPO, paired on the same task/prefix
  grpo_prompts(rollouts, tasks)  -> {prompt, task_id, step_index, prefix_calls, expected_tools}   for GRPO
  step_rewards(rollouts)         -> flat table of per-step rewards (RL transitions)

All functions are pure and operate on in-memory Rollouts; `export_jsonl` writes them out.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from ..sdk.llm import assistant_to_text, messages_to_text
from ..sdk.types import Role, Rollout, TaskSpec, ToolSpec


def _tools_for(step, tools: list[ToolSpec] | None) -> list[ToolSpec] | None:
    """Only the tools that were exposed to the policy at that step (keeps prompts short and faithful)."""
    if tools is None:
        return None
    if step.exposed_tools:
        by = {t.name: t for t in tools}
        return [by[n] for n in step.exposed_tools if n in by]
    return tools


def _instruction(rollout: Rollout) -> str:
    for m in rollout.steps[0].prompt_messages if rollout.steps else []:
        if m.role == Role.user:
            return m.content
    return ""


def _steps(rollout: Rollout, role: str | None):
    """Steps of one role (multi-agent) or all steps (single agent / role=None)."""
    return [s for s in rollout.steps if role is None or s.role == role]


def _step_query(rollout: Rollout, step, role: str | None) -> str:
    """The text the router saw for this step: the (delegated) task plus the latest user message."""
    if role is not None:
        task = next((m.content for m in step.prompt_messages if m.role == Role.user), "")
        return task.split("\n\nContext:")[0]
    instr = _instruction(rollout)
    last_user = next((m.content for m in reversed(step.prompt_messages) if m.role == Role.user), "")
    return instr + " " + (last_user if last_user != instr else "")


def _step_rewards(rollout: Rollout) -> dict[int, list[float]]:
    out: dict[int, list[float]] = defaultdict(list)
    for r in rollout.rewards:
        if r.step_index is not None:
            out[r.step_index].append(r.value)
    return out


def router_examples(rollouts: Iterable[Rollout], success_weight: float = 1.0, step_weight: float = 0.5,
                    role: str | None = None) -> list[tuple[str, str, float]]:
    """Positive examples from successful trajectories; per-step tool_correct rewards as +/- weighted examples.
    With `role`, only that role's steps are used (selective optimisation in multi-agent systems)."""
    ex: list[tuple[str, str, float]] = []
    for ro in rollouts:
        sr = _step_rewards(ro)
        success = ro.total_reward >= 0.99
        for s in _steps(ro, role):
            query = _step_query(ro, s, role)
            for tc in s.response.tool_calls:
                if success:
                    ex.append((query, tc.name, success_weight))
                for v in sr.get(s.index, []):
                    ex.append((query, tc.name, step_weight * (1.0 if v > 0 else -1.0)))
    return ex


def sft_examples(rollouts: Iterable[Rollout], tools_by_env: dict[str, list[ToolSpec]] | None = None,
                 min_reward: float = 0.99, role: str | None = None) -> list[dict[str, Any]]:
    out = []
    for ro in rollouts:
        if ro.total_reward < min_reward:
            continue
        tools = (tools_by_env or {}).get(ro.env_name)
        for s in _steps(ro, role):
            out.append({"prompt": messages_to_text(s.prompt_messages, _tools_for(s, tools)), "completion": assistant_to_text(s.response),
                        "task_id": ro.task_id, "rollout_id": ro.id, "step_index": s.index, "env": ro.env_name, "role": s.role})
    return out


def preference_pairs(rollouts: Iterable[Rollout], tools_by_env: dict[str, list[ToolSpec]] | None = None,
                     role: str | None = None) -> list[dict[str, Any]]:
    """Pair rollouts of the same task: at the first step where a good and a bad rollout diverge,
    emit (prompt, chosen, rejected). Also uses per-step rewards inside a single rollout."""
    by_task: dict[str, list[Rollout]] = defaultdict(list)
    for ro in rollouts:
        by_task[ro.task_id].append(ro)
    pairs = []
    for task_id, ros in by_task.items():
        good = [r for r in ros if r.total_reward >= 0.99]
        bad = [r for r in ros if r.total_reward < 0.99]
        for g in good:
            tools = (tools_by_env or {}).get(g.env_name)
            for b in bad:
                for sg, sb in zip(_steps(g, role), _steps(b, role)):
                    tg, tb = assistant_to_text(sg.response), assistant_to_text(sb.response)
                    if tg != tb:
                        pairs.append({"prompt": messages_to_text(sg.prompt_messages, _tools_for(sg, tools)), "chosen": tg, "rejected": tb,
                                      "task_id": task_id, "env": g.env_name, "role": sg.role})
                        break
    # intra-rollout: a step with negative tool_correct vs a positive step for the same prefix length is rare,
    # so we only use cross-rollout pairs here.
    return pairs


def grpo_prompts(rollouts: Iterable[Rollout], tasks: dict[str, TaskSpec], tools_by_env: dict[str, list[ToolSpec]] | None = None,
                 role: str | None = None) -> list[dict[str, Any]]:
    """One prompt per (task, step prefix) with the information needed to compute an environment-grounded reward
    for *new* completions: replay `prefix_calls` in a fresh env, then score the sampled action."""
    out = []
    seen = set()
    for ro in rollouts:
        task = tasks.get(ro.task_id)
        if task is None:
            continue
        tools = (tools_by_env or {}).get(ro.env_name)
        prefix: list[dict] = []
        for s in ro.steps:
            key = (ro.task_id, s.index, json.dumps(prefix, sort_keys=True))
            if key not in seen and (role is None or s.role == role):
                seen.add(key)
                out.append({"prompt": messages_to_text(s.prompt_messages, _tools_for(s, tools)), "task_id": ro.task_id, "env": ro.env_name,
                            "step_index": s.index, "prefix_calls": list(prefix), "expected_tools": task.expected.get("tools", []), "role": s.role})
            for tc in s.response.tool_calls:
                if not tc.name.startswith("delegate_to_"):
                    prefix.append({"name": tc.name, "arguments": tc.arguments})
    return out


def step_rewards(rollouts: Iterable[Rollout]) -> list[dict[str, Any]]:
    rows = []
    for ro in rollouts:
        sr = _step_rewards(ro)
        for s in ro.steps:
            rows.append({"rollout_id": ro.id, "task_id": ro.task_id, "step_index": s.index,
                         "action": assistant_to_text(s.response), "step_reward": sum(sr.get(s.index, [])) if sr.get(s.index) else None,
                         "trajectory_reward": ro.total_reward, "exposed_tools": s.exposed_tools})
    return rows


def export_jsonl(rows: Iterable[dict[str, Any]], path: str | Path) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")
    return str(p)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]
