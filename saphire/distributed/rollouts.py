"""Distributed rollout collection and evaluation.

Rollouts are embarrassingly parallel: each (task, trial) needs an environment copy and a policy.
`RolloutEngine` shards them across

* **Ray** actors (`backend="ray"`, `pip install saphire[distributed]`) — scales across a cluster; each actor holds its
  own environments and agent (built from the serialised `AgentConfig`; artifacts must be on a shared filesystem/S3),
* a local **process pool** (`backend="process"`) — multi-core on one node without Ray,
* **threads** (`backend="thread"`) — for I/O-bound hosted-model calls,

and returns fully verified `Rollout`s plus per-rollout records so `build_result` can aggregate exactly as the
local runner does. The same engine feeds `OnlineLoop` (`collect`) and the trainers' data collection.
"""
from __future__ import annotations

import concurrent.futures as cf
import os
import time
from typing import Any, Callable, Iterable, Optional

from ..environments.base import Environment, get_environment
from ..evaluation.runner import EvalResult, RolloutRecord, build_result, record_from
from ..sdk.multi_agent import build_agent, role_rewards
from ..sdk.types import AgentConfig, Reward, Rollout, TaskSpec


# ---------------------------------------------------------------------------
# Work unit executed inside a worker (process, thread or Ray actor)
# ---------------------------------------------------------------------------
class _Worker:
    def __init__(self, config: dict[str, Any], judge_model: Optional[str] = None):
        os.environ.setdefault("SAPHIRE_TRACING", "off")
        self.config = AgentConfig.model_validate(config)
        self.agent = build_agent(self.config)
        self.envs: dict[str, Environment] = {}
        from ..signals.judges import make_judge

        self.judge = make_judge(judge_model)

    def run(self, task: dict[str, Any], trial: int) -> dict[str, Any]:
        t = TaskSpec.model_validate(task)
        env = self.envs.setdefault(t.env_name, get_environment(t.env_name))
        state = env.reset(t)
        ro = self.agent.run(t, env, state)
        rewards = env.verify(t, ro, state)
        if self.judge is not None:
            rewards.extend(self.judge(t, ro))
        ro.rewards = rewards
        rec = record_from(ro, t, rewards, trial)
        return {"rollout": ro.model_dump(mode="json"), "record": rec.model_dump(), "task": task,
                "roles": role_rewards(ro) if ro.metadata.get("multi_agent") else None}


_PROC_WORKER: dict[str, _Worker] = {}


def _proc_init(config: dict[str, Any], judge_model: Optional[str]) -> None:
    _PROC_WORKER["w"] = _Worker(config, judge_model)


def _proc_run(args):
    task, trial = args
    return _PROC_WORKER["w"].run(task, trial)


# ---------------------------------------------------------------------------
class RolloutEngine:
    def __init__(self, config: AgentConfig, backend: str = "auto", workers: Optional[int] = None, judge_model: Optional[str] = None,
                 ray_address: Optional[str] = None, actor_options: Optional[dict[str, Any]] = None):
        self.config = config
        self.judge_model = judge_model
        self.workers = workers or max(1, (os.cpu_count() or 2))
        self.actor_options = actor_options or {}
        if backend == "auto":
            backend = "ray" if _ray_available() else "process"
        self.backend = backend
        self._ray_actors: list[Any] = []
        if backend == "ray":
            import ray

            if not ray.is_initialized():
                ray.init(address=ray_address or os.getenv("RAY_ADDRESS"), ignore_reinit_error=True, log_to_driver=False,
                         include_dashboard=False)
            actor_cls = ray.remote(**self.actor_options)(_Worker) if self.actor_options else ray.remote(_Worker)
            self._ray_actors = [actor_cls.remote(config.model_dump(), judge_model) for _ in range(self.workers)]

    # -------------------------------------------------------------------
    def collect(self, tasks: list[TaskSpec], k: int = 1, on_rollout: Optional[Callable[[Rollout, TaskSpec, list[Reward]], None]] = None
                ) -> tuple[list[Rollout], list[RolloutRecord], dict[str, list[dict[str, float]]], float]:
        jobs = [(t.model_dump(), trial) for t in tasks for trial in range(k)]
        t0 = time.perf_counter()
        if self.backend == "ray":
            import ray

            futures = [self._ray_actors[i % len(self._ray_actors)].run.remote(task, trial) for i, (task, trial) in enumerate(jobs)]
            outs = ray.get(futures)
        elif self.backend == "process":
            ctx = __import__("multiprocessing").get_context("spawn")
            with cf.ProcessPoolExecutor(max_workers=self.workers, mp_context=ctx, initializer=_proc_init,
                                        initargs=(self.config.model_dump(), self.judge_model)) as ex:
                outs = list(ex.map(_proc_run, jobs, chunksize=max(1, len(jobs) // (self.workers * 4) or 1)))
        else:  # thread
            w = _Worker(self.config.model_dump(), self.judge_model)
            with cf.ThreadPoolExecutor(max_workers=self.workers) as ex:
                outs = list(ex.map(lambda a: w.run(*a), jobs))
        wall = time.perf_counter() - t0
        rollouts, records, role_stats = [], [], {}
        for o in outs:
            ro = Rollout.model_validate(o["rollout"])
            rec = RolloutRecord.model_validate(o["record"])
            rollouts.append(ro)
            records.append(rec)
            if o.get("roles"):
                for role, rr in o["roles"].items():
                    role_stats.setdefault(role, []).append(rr)
            if on_rollout is not None:
                on_rollout(ro, TaskSpec.model_validate(o["task"]), ro.rewards)
        return rollouts, records, role_stats, wall

    def evaluate(self, tasks: list[TaskSpec], k: int = 1, suite: str = "distributed", **kw: Any) -> EvalResult:
        _, records, role_stats, wall = self.collect(tasks, k=k, **kw)
        res = build_result(self.config.name, self.config.version, suite, len(tasks), records, k, wall, role_stats)
        res.metrics["workers"] = float(self.workers)
        res.metrics["backend"] = self.backend  # type: ignore[assignment]
        return res

    def shutdown(self) -> None:
        if self.backend == "ray":
            import ray

            for a in self._ray_actors:
                ray.kill(a)
            self._ray_actors = []


def _ray_available() -> bool:
    try:
        import ray  # noqa: F401

        return True
    except ImportError:
        return False


def evaluate_distributed(config: AgentConfig, tasks: Iterable[TaskSpec], k: int = 1, backend: str = "auto", workers: Optional[int] = None,
                         judge_model: Optional[str] = None, suite: str = "distributed", **kw: Any) -> EvalResult:
    eng = RolloutEngine(config, backend=backend, workers=workers, judge_model=judge_model, **kw)
    try:
        return eng.evaluate(list(tasks), k=k, suite=suite)
    finally:
        eng.shutdown()
