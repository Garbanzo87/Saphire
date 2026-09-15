"""Online learning loop: collect -> verify -> generate signals -> update learners -> evaluate.

Each iteration produces a new agent *version* whose artifacts (router, exemplars, prompt) are
written under `artifacts/<agent>/<version>/`, plus a metrics snapshot on a fixed held-out
suite so improvement over time is measured against the same yardstick.

Weight updates (SFT/DPO/GRPO via TRL) are optional periodic steps (`weight_update_every`);
they are skipped when the `train` extra is not installed.
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any, Callable, Optional

from ..environments.base import Environment, get_environment
from ..evaluation.runner import EvalResult, evaluate
from ..sdk.agent import ToolAgent
from ..sdk.exemplars import ExemplarStore
from ..sdk.llm import LLM, get_llm
from ..sdk.multi_agent import ORCHESTRATOR, AgentSystem, RolePolicy
from ..sdk.router import ToolRouter
from ..sdk.types import AgentConfig, Rollout, TaskSpec
from .learners import train_router, update_exemplars
from .multi_agent import train_roles
from .prompt_opt import optimize_prompt


class OnlineLoop:
    def __init__(self, config: AgentConfig, train_tasks: list[TaskSpec], eval_tasks: list[TaskSpec],
                 artifacts_dir: str | Path = "artifacts", envs: dict[str, Environment] | None = None,
                 llm: Optional[LLM] = None, batch_size: int = 16, learn_router: bool = True, learn_exemplars: bool = True,
                 learn_prompt_every: int = 0, weight_update_every: int = 0, weight_update: Optional[Callable[..., dict]] = None,
                 seed: int = 0, on_iteration: Optional[Callable[[dict[str, Any]], None]] = None,
                 on_rollout: Optional[Callable[[Rollout, TaskSpec, list], None]] = None, eval_k: int = 1,
                 optimize_roles: Optional[list[str]] = None, rollout_backend: Optional[str] = None, rollout_workers: Optional[int] = None):
        self.config = config
        self.train_tasks = train_tasks
        self.eval_tasks = eval_tasks
        self.envs = envs or {}
        for t in train_tasks + eval_tasks:
            self.envs.setdefault(t.env_name, get_environment(t.env_name))
        self.artifacts_dir = Path(artifacts_dir) / config.name
        self.llm = llm or get_llm(config.model)
        self.batch_size = batch_size
        self.learn_router = learn_router
        self.learn_exemplars = learn_exemplars
        self.learn_prompt_every = learn_prompt_every
        self.weight_update_every = weight_update_every
        self.weight_update = weight_update
        self.rng = random.Random(seed)
        self.on_iteration = on_iteration
        self.on_rollout = on_rollout
        self.eval_k = eval_k
        self.router: Optional[ToolRouter] = ToolRouter.load(config.tool_router) if config.tool_router else None
        self.exemplars: ExemplarStore = ExemplarStore.load(config.exemplar_store) if config.exemplar_store else ExemplarStore()
        # multi-agent: per-role learners; `optimize_roles=None` means every trainable role
        self.optimize_roles = optimize_roles
        # distributed collection / evaluation (ray | process | thread); artifacts must be on a shared filesystem
        self.rollout_backend = rollout_backend
        self.rollout_workers = rollout_workers
        self.role_routers: dict[str, ToolRouter] = {}
        self.role_exemplars: dict[str, ExemplarStore] = {}
        if config.is_multi_agent:
            for role, rc in [(ORCHESTRATOR, None)] + list(config.roles.items()):
                rp = config.tool_router if rc is None else rc.tool_router
                ep = config.exemplar_store if rc is None else rc.exemplar_store
                if rp:
                    self.role_routers[role] = ToolRouter.load(rp)
                if ep:
                    self.role_exemplars[role] = ExemplarStore.load(ep)
        self.buffer: list[Rollout] = []  # replay buffer of verified rollouts
        self.history: list[dict[str, Any]] = []
        self.version_no = 0

    # ------------------------------------------------------------------
    def _agent(self):
        if self.config.is_multi_agent:
            policies = {}
            for role in [ORCHESTRATOR] + list(self.config.roles):
                rc = self.config.roles.get(role)
                from ..sdk.types import RoleConfig

                rcfg = rc if rc is not None else RoleConfig(
                    system_prompt=self.config.orchestrator_prompt or self.config.system_prompt, tool_names=self.config.orchestrator_tools,
                    router_top_k=self.config.router_top_k, exemplar_k=self.config.exemplar_k, max_steps=self.config.max_steps)
                policies[role] = RolePolicy(role, rcfg, self.config, llm=self.llm if (rc is None or rc.model is None) else None,
                                            router=self.role_routers.get(role),
                                            exemplars=self.role_exemplars.get(role) if self.learn_exemplars else ExemplarStore())
            return AgentSystem(self.config, llm=self.llm, policies=policies)
        return ToolAgent(self.config, llm=self.llm, router=self.router, exemplars=self.exemplars if self.learn_exemplars else None)

    def _version_dir(self) -> Path:
        d = self.artifacts_dir / self.config.version
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _engine(self):
        from ..distributed.rollouts import RolloutEngine

        return RolloutEngine(self.config, backend=self.rollout_backend, workers=self.rollout_workers)

    def evaluate(self, suite: str = "heldout") -> EvalResult:
        if self.rollout_backend:
            eng = self._engine()
            try:
                return eng.evaluate(self.eval_tasks, k=self.eval_k, suite=suite, on_rollout=self.on_rollout)
            finally:
                eng.shutdown()
        return evaluate(self._agent(), self.eval_tasks, envs=self.envs, suite=suite, k=self.eval_k, on_rollout=self.on_rollout)

    def collect(self, n: int) -> list[Rollout]:
        batch = self.rng.sample(self.train_tasks, min(n, len(self.train_tasks)))
        out: list[Rollout] = []

        def _cb(ro, task, rewards):
            out.append(ro)
            if self.on_rollout:
                self.on_rollout(ro, task, rewards)

        if self.rollout_backend:
            eng = self._engine()
            try:
                eng.collect(batch, on_rollout=_cb)
            finally:
                eng.shutdown()
            return out
        evaluate(self._agent(), batch, envs=self.envs, suite="collect", on_rollout=_cb)
        return out

    def step(self) -> dict[str, Any]:
        """One iteration: collect a batch, update learners, bump version, evaluate."""
        t0 = time.perf_counter()
        it = len(self.history) if self.history and self.history[0]["iteration"] == 0 else len(self.history) + 1
        rollouts = self.collect(self.batch_size)
        self.buffer.extend(rollouts)
        self.buffer = self.buffer[-2000:]
        self.version_no += 1
        self.config = self.config.model_copy(update={"version": f"v{self.version_no}"})
        vdir = self._version_dir()
        updates: dict[str, Any] = {}
        if self.config.is_multi_agent:
            self.config, updates["roles"] = train_roles(self.config, self.buffer, self.envs, vdir, roles=self.optimize_roles,
                                                        routers=self.role_routers, exemplars=self.role_exemplars,
                                                        learn_router=self.learn_router, learn_exemplars=self.learn_exemplars)
        elif self.learn_router:
            updates["router"] = train_router(self.buffer, self.envs, vdir / "router", router=self.router)
            self.router = ToolRouter.load(updates["router"]["artifact"])
            self.config = self.config.model_copy(update={"tool_router": updates["router"]["artifact"]})
        if self.learn_exemplars and not self.config.is_multi_agent:
            updates["exemplars"] = update_exemplars(rollouts, vdir / "exemplars.json", store=self.exemplars)
            self.config = self.config.model_copy(update={"exemplar_store": updates["exemplars"]["artifact"]})
        if self.learn_prompt_every and it % self.learn_prompt_every == 0 and not self.config.is_multi_agent:
            po = optimize_prompt(self.config, self.train_tasks, envs=self.envs, reflection_model=self.config.model if self.config.model.startswith("mock") else self.config.metadata.get("reflection_model", "mock"),
                                 iterations=2, minibatch=min(8, len(self.train_tasks)), seed=self.rng.randint(0, 10**6), policy_llm=self.llm)
            if po["best_score"] >= po["baseline_score"]:
                self.config = self.config.model_copy(update={"system_prompt": po["best_prompt"]})
            updates["prompt"] = {k: v for k, v in po.items() if k != "history"}
        if self.weight_update_every and self.weight_update and it % self.weight_update_every == 0:
            updates["weights"] = self.weight_update(self.buffer, self.config, vdir)
            if updates["weights"].get("model"):
                self.config = self.config.model_copy(update={"model": updates["weights"]["model"]})
                self.llm = get_llm(self.config.model)
        (vdir / "agent.json").write_text(self.config.model_dump_json(indent=2))
        res = self.evaluate()
        (vdir / "eval.json").write_text(json.dumps(res.summary(), indent=2))
        rec = {"iteration": it, "version": self.config.version, "collected": len(rollouts),
               "collect_success": sum(r.total_reward >= 0.99 for r in rollouts) / max(1, len(rollouts)),
               "buffer": len(self.buffer), "updates": updates, "eval": res.metrics, "by_family": res.by_family,
               "seconds": time.perf_counter() - t0, "artifact_dir": str(vdir)}
        self.history.append(rec)
        (self.artifacts_dir / "history.json").write_text(json.dumps(self.history, indent=2, default=str))
        if self.on_iteration:
            self.on_iteration(rec)
        return rec

    def run(self, iterations: int) -> list[dict[str, Any]]:
        if not self.history:
            base = self.evaluate()
            vdir = self._version_dir()
            (vdir / "agent.json").write_text(self.config.model_dump_json(indent=2))
            (vdir / "eval.json").write_text(json.dumps(base.summary(), indent=2))
            rec = {"iteration": 0, "version": self.config.version, "collected": 0, "collect_success": None, "buffer": 0,
                   "updates": {}, "eval": base.metrics, "by_family": base.by_family, "seconds": base.wall_time_s, "artifact_dir": str(vdir)}
            self.history.append(rec)
            if self.on_iteration:
                self.on_iteration(rec)
        for _ in range(iterations):
            self.step()
        return self.history
