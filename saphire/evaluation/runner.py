"""Evaluation runner: executes an agent on a task set, verifies, aggregates.

Produces an `EvalResult` with:
  * per-rollout records (success, tool-selection F1, context preservation, latency, tokens, ...)
  * aggregate metrics incl. reliability (pass@1, pass^k, consistency), throughput
    (tasks/min, tokens/task), and breakdowns by task family / difficulty / environment
  * bootstrap confidence intervals
"""
from __future__ import annotations

import concurrent.futures as cf
import time
from collections import defaultdict
from typing import Any, Callable, Optional

from pydantic import BaseModel, Field

from ..environments.base import Environment, get_environment
from ..sdk.agent import ToolAgent
from ..sdk.types import AgentConfig, Reward, Rollout, TaskSpec
from . import metrics as M

TRAJ_METRICS = ["task_success", "tool_selection_f1", "step_efficiency", "context_preservation", "tool_error_free", "judge_score"]


class RolloutRecord(BaseModel):
    rollout_id: str
    task_id: str
    env_name: str
    family: str = ""
    difficulty: str = ""
    trial: int = 0
    status: str
    n_steps: int
    n_tool_calls: int
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    task_success: float = 0.0
    tool_selection_f1: Optional[float] = None
    step_efficiency: Optional[float] = None
    context_preservation: Optional[float] = None
    tool_error_free: Optional[float] = None
    judge_score: Optional[float] = None
    rationale: str = ""
    called_tools: list[str] = Field(default_factory=list)
    trace_id: Optional[str] = None


class EvalResult(BaseModel):
    agent: str
    agent_version: str
    suite: str = "custom"
    n_tasks: int
    n_rollouts: int
    k: int = 1
    wall_time_s: float
    metrics: dict[str, Any]
    by_family: dict[str, dict[str, float]] = Field(default_factory=dict)
    by_env: dict[str, dict[str, float]] = Field(default_factory=dict)
    by_difficulty: dict[str, dict[str, float]] = Field(default_factory=dict)
    by_role: dict[str, dict[str, float]] = Field(default_factory=dict)  # multi-agent systems: credit per role
    per_rollout: list[RolloutRecord] = Field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {k: v for k, v in self.model_dump().items() if k != "per_rollout"}


def record_from(rollout: Rollout, task: TaskSpec, rewards: list[Reward], trial: int = 0) -> RolloutRecord:
    traj = {r.name: r for r in rewards if r.step_index is None}
    u = rollout.usage
    rec = RolloutRecord(
        rollout_id=rollout.id, task_id=task.id, env_name=task.env_name, family=(task.tags[0] if task.tags else ""),
        difficulty=task.difficulty, trial=trial, status=rollout.status.value, n_steps=len(rollout.steps),
        n_tool_calls=len(rollout.tool_calls), latency_ms=rollout.duration_ms, prompt_tokens=u.prompt_tokens,
        completion_tokens=u.completion_tokens, called_tools=[tc.name for tc in rollout.tool_calls], trace_id=rollout.trace_id,
        rationale=(traj["task_success"].rationale or "") if "task_success" in traj else "",
    )
    for m in TRAJ_METRICS:
        if m in traj:
            setattr(rec, m, float(traj[m].value))
    return rec


def _aggregate(records: list[RolloutRecord]) -> dict[str, float]:
    out: dict[str, float] = {"n": float(len(records))}
    for m in TRAJ_METRICS:
        vals = [getattr(r, m) for r in records if getattr(r, m) is not None]
        if vals:
            out[m] = M.mean(vals)
    if records:
        out["latency_ms_p50"] = M.percentile([r.latency_ms for r in records], 50)
        out["latency_ms_p95"] = M.percentile([r.latency_ms for r in records], 95)
        out["steps_mean"] = M.mean(r.n_steps for r in records)
        out["tokens_per_task"] = M.mean(r.prompt_tokens + r.completion_tokens for r in records)
        out["error_rate"] = M.mean(1.0 if r.status in ("error", "timeout") else 0.0 for r in records)
    return out


def evaluate(agent: ToolAgent, tasks: list[TaskSpec], envs: dict[str, Environment] | None = None, k: int = 1,
             concurrency: int = 1, suite: str = "custom", judge: Optional[Callable[[TaskSpec, Rollout], list[Reward]]] = None,
             on_rollout: Optional[Callable[[Rollout, TaskSpec, list[Reward]], None]] = None) -> EvalResult:
    """Run `agent` on every task `k` times and aggregate.

    `judge` (optional) adds model-graded rewards (e.g. `saphire.signals.judges.RubricJudge`).
    `on_rollout` is invoked for each finished rollout (used by the server to persist traces).
    """
    envs = envs or {}

    def env_for(name: str) -> Environment:
        if name not in envs:
            envs[name] = get_environment(name)
        return envs[name]

    jobs = [(t, trial) for t in tasks for trial in range(k)]
    records: list[RolloutRecord] = []
    role_stats: dict[str, list[dict[str, float]]] = {}
    t0 = time.perf_counter()

    def _one(job):
        task, trial = job
        env = env_for(task.env_name)
        state = env.reset(task)
        ro = agent.run(task, env, state)
        rewards = env.verify(task, ro, state)
        if judge is not None:
            rewards.extend(judge(task, ro))
        ro.rewards = rewards
        if ro.metadata.get("multi_agent"):
            from ..sdk.multi_agent import role_rewards

            for role, rr in role_rewards(ro).items():
                role_stats.setdefault(role, []).append(rr)
        if on_rollout is not None:
            on_rollout(ro, task, rewards)
        return record_from(ro, task, rewards, trial)

    if concurrency > 1:
        with cf.ThreadPoolExecutor(max_workers=concurrency) as ex:
            records = list(ex.map(_one, jobs))
    else:
        records = [_one(j) for j in jobs]
    wall = time.perf_counter() - t0
    return build_result(agent.config.name, agent.config.version, suite, len(tasks), records, k, wall, role_stats)


def build_result(agent_name: str, agent_version: str, suite: str, n_tasks: int, records: list[RolloutRecord], k: int, wall: float,
                 role_stats: dict[str, list[dict[str, float]]] | None = None) -> EvalResult:
    """Aggregate per-rollout records into an EvalResult (shared by local and distributed evaluation)."""
    agg = _aggregate(records)
    per_task: dict[str, list[float]] = defaultdict(list)
    for r in records:
        per_task[r.task_id].append(r.task_success)
    agg["pass_at_1"] = agg.get("task_success", 0.0)
    if k > 1:
        agg[f"pass_hat_{k}"] = M.pass_hat_k(per_task, k)
        agg[f"pass_at_{k}"] = M.pass_at_k(per_task, k)
        agg["consistency"] = M.mean(1.0 if len(set(v)) == 1 else 0.0 for v in per_task.values())
    agg["throughput_tasks_per_min"] = len(records) / wall * 60 if wall > 0 else 0.0
    lo, hi = M.bootstrap_ci([r.task_success for r in records])
    agg["task_success_ci_low"], agg["task_success_ci_high"] = lo, hi

    def group(keyfn):
        g: dict[str, list[RolloutRecord]] = defaultdict(list)
        for r in records:
            g[keyfn(r)].append(r)
        return {kk: _aggregate(v) for kk, v in sorted(g.items())}

    by_role = {role: {"step_reward": M.mean(x["step_mean"] for x in xs), "steps_per_rollout": M.mean(x["n_steps"] for x in xs),
                      "tool_calls_per_rollout": M.mean(x["n_tool_calls"] for x in xs), "n": float(len(xs))}
               for role, xs in sorted((role_stats or {}).items())}
    return EvalResult(agent=agent_name, agent_version=agent_version, suite=suite, n_tasks=n_tasks,
                      n_rollouts=len(records), k=k, wall_time_s=wall, metrics=agg, by_family=group(lambda r: r.family),
                      by_env=group(lambda r: r.env_name), by_difficulty=group(lambda r: r.difficulty), by_role=by_role, per_rollout=records)


def evaluate_config(config: AgentConfig, tasks: list[TaskSpec], **kw: Any) -> EvalResult:
    return evaluate(ToolAgent(config), tasks, **kw)
