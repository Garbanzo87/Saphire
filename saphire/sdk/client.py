"""Thin HTTP client for the Saphire server (used by the CLI, tests, notebooks and BYO agents)."""
from __future__ import annotations

import os
import time
from typing import Any, Optional

import httpx

from .types import AgentConfig, Reward, Rollout, TaskSpec


class SaphireClient:
    def __init__(self, host: Optional[str] = None, api_key: Optional[str] = None, project: Optional[str] = None, timeout: float = 60.0):
        self.host = (host or os.getenv("SAPHIRE_HOST", "http://localhost:8000")).rstrip("/")
        self.api_key = api_key or os.getenv("SAPHIRE_API_KEY", "dev-key")
        self.project = project or os.getenv("SAPHIRE_PROJECT", "default")
        self.http = httpx.Client(base_url=self.host, timeout=timeout, headers={"x-api-key": self.api_key})

    # ---- low level ----
    def _r(self, method: str, path: str, **kw) -> Any:
        params = kw.pop("params", {}) or {}
        params.setdefault("project", self.project)
        r = self.http.request(method, f"/v1{path}", params=params, **kw)
        if r.status_code >= 400:
            raise RuntimeError(f"{method} {path} -> {r.status_code}: {r.text[:500]}")
        return r.json()

    def get(self, path: str, **params) -> Any:
        return self._r("GET", path, params=params)

    def post(self, path: str, json: Any = None, **params) -> Any:
        return self._r("POST", path, json=json, params=params)

    # ---- convenience ----
    def health(self) -> dict:
        return self.http.get("/health").json()

    def create_agent(self, config: AgentConfig, status: str = "candidate", parent_id: Optional[str] = None) -> dict:
        return self.post("/agents", json={"config": config.model_dump(), "status": status, "parent_id": parent_id})

    def create_dataset(self, name: str, suite: Optional[str] = None, tasks: Optional[list[TaskSpec]] = None, n_per_env: int = 14,
                       seed: int = 0, split: str = "eval") -> dict:
        return self.post("/datasets", json={"name": name, "suite": suite, "tasks": [t.model_dump() for t in (tasks or [])],
                                            "n_per_env": n_per_env, "seed": seed, "split": split})

    def run_eval(self, agent_id: str, dataset_id: str, k: int = 1, judge_model: Optional[str] = None, gate: bool = False,
                 auto_promote: bool = False, gate_policy: Optional[dict] = None, wait: bool = False, concurrency: int = 1) -> dict:
        run = self.post("/evals", json={"agent_id": agent_id, "dataset_id": dataset_id, "k": k, "judge_model": judge_model, "gate": gate,
                                        "auto_promote": auto_promote, "gate_policy": gate_policy, "concurrency": concurrency})
        return self.wait_job(run["job_id"]) and self.get(f"/evals/{run['id']}") if wait else run

    def train(self, agent_id: str, algorithm: str, wait: bool = False, **params) -> dict:
        run = self.post("/training", json={"agent_id": agent_id, "algorithm": algorithm, "params": params})
        return self.wait_job(run["job_id"]) and self.get(f"/training/{run['id']}") if wait else run

    def wait_job(self, job_id: str, timeout: float = 3600, poll: float = 1.0) -> dict:
        t0 = time.time()
        while True:
            j = self.get(f"/jobs/{job_id}")
            if j["status"] in ("succeeded", "failed", "cancelled"):
                if j["status"] != "succeeded":
                    raise RuntimeError(f"job {job_id} {j['status']}: {j.get('error', '')[:800]}")
                return j
            if time.time() - t0 > timeout:
                raise TimeoutError(f"job {job_id} still {j['status']} after {timeout}s")
            time.sleep(poll)

    def log_rollout(self, rollout: Rollout, task: TaskSpec, rewards: list[Reward] | None = None, agent_id: Optional[str] = None) -> dict:
        return self.post("/rollouts", json={"rollout": rollout.model_dump(mode="json"), "task": task.model_dump(),
                                            "rewards": [r.model_dump() for r in (rewards or [])], "agent_id": agent_id})

    def score(self, name: str, value: float, trace_id: Optional[str] = None, rollout_id: Optional[str] = None, source: str = "sdk",
              rationale: str = "") -> dict:
        return self.post("/scores", json={"name": name, "value": value, "trace_id": trace_id, "rollout_id": rollout_id,
                                          "source": source, "rationale": rationale})

    def assign_variant(self, experiment_id: str, unit: str) -> dict:
        return self.get(f"/experiments/{experiment_id}/assign", unit=unit)

    def record_outcome(self, experiment_id: str, unit: str, value: float, variant: Optional[str] = None) -> dict:
        return self.post(f"/experiments/{experiment_id}/outcomes", json={"unit": unit, "value": value, "variant": variant})
