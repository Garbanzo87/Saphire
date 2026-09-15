"""Learned reward model: predict task success from a trajectory when no verifier exists.

A pure-numpy logistic regression over hashed trajectory features (instruction n-grams, tool sequence bigrams,
step/tool/error counts, status, answer length), trained from any labels the platform has — verifier outcomes,
human/product scores, or judge scores. Fast to train (seconds), cheap at inference, and exposed as a judge
(`judge_model="rm:<artifact>"`) so it can score production rollouts that have no ground truth, feed the online loop
(`RewardModelJudge`), and be re-calibrated as human labels arrive.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

from ..sdk.router import hashed_features
from ..sdk.types import Reward, Role, Rollout, TaskSpec

HANDOFF = "delegate_to_"


def trajectory_features(ro: Rollout, dim: int = 4096) -> np.ndarray:
    instr = next((m.content for s in ro.steps[:1] for m in s.prompt_messages if m.role == Role.user), "")
    tools = [tc.name for tc in ro.tool_calls if not tc.name.startswith(HANDOFF)]
    tool_text = " ".join(tools) + " " + " ".join(f"{a}__{b}" for a, b in zip(tools, tools[1:]))
    x = hashed_features(instr, dim) * 0.5 + hashed_features(tool_text, dim)
    n_err = sum(1 for s in ro.steps for r in s.tool_results if r.error)
    dense = np.array([len(ro.steps) / 10, len(tools) / 10, n_err, float(ro.status.value == "succeeded"), float(ro.status.value in ("timeout", "error")),
                      min(len(ro.final_answer), 500) / 500, float(len(set(tools)) < len(tools)), 1.0], dtype=np.float32)
    return np.concatenate([x, dense])


class RewardModel:
    def __init__(self, dim: int = 4096, seed: int = 0):
        self.dim = dim
        self.w = np.zeros(dim + 8, dtype=np.float32)
        self.b = 0.0
        self.n_trained = 0
        self.metrics: dict[str, Any] = {}
        self.rng = np.random.default_rng(seed)

    def _X(self, rollouts: list[Rollout]) -> np.ndarray:
        return np.stack([trajectory_features(r, self.dim) for r in rollouts]) if rollouts else np.zeros((0, self.dim + 8))

    def fit(self, rollouts: list[Rollout], labels: list[float], epochs: int = 40, lr: float = 0.3, l2: float = 1e-4) -> dict[str, Any]:
        X = self._X(rollouts)
        y = np.asarray(labels, dtype=np.float32)
        n = len(y)
        if n == 0:
            return {"n": 0}
        for _ in range(epochs):
            idx = self.rng.permutation(n)
            for s in range(0, n, 32):
                bi = idx[s:s + 32]
                p = 1 / (1 + np.exp(-(X[bi] @ self.w + self.b)))
                g = p - y[bi]
                self.w -= lr * (X[bi].T @ g / len(bi) + l2 * self.w)
                self.b -= lr * float(g.mean())
        self.n_trained += n
        self.metrics = self.evaluate(rollouts, labels) | {"n_train": n, "positive_rate": float(y.mean())}
        return self.metrics

    def predict(self, rollouts: list[Rollout]) -> np.ndarray:
        X = self._X(rollouts)
        return 1 / (1 + np.exp(-(X @ self.w + self.b)))

    def evaluate(self, rollouts: list[Rollout], labels: list[float]) -> dict[str, float]:
        p = self.predict(rollouts)
        y = np.asarray(labels, dtype=np.float32)
        acc = float(((p >= 0.5) == (y >= 0.5)).mean()) if len(y) else 0.0
        # AUC via rank statistic
        pos, neg = p[y >= 0.5], p[y < 0.5]
        if len(pos) and len(neg):
            auc = float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())
        else:
            auc = float("nan")
        return {"accuracy": acc, "auc": auc, "n": int(len(y))}

    def save(self, path: str | Path) -> str:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(path.with_suffix(".npz")), w=self.w, b=np.array([self.b]))
        path.with_suffix(".json").write_text(json.dumps({"dim": self.dim, "n_trained": self.n_trained, "metrics": self.metrics}))
        return str(path.with_suffix(".json"))

    @classmethod
    def load(cls, path: str | Path) -> "RewardModel":
        from ..sdk.artifacts import resolve

        p = Path(resolve(str(path), siblings=("npz",)))
        meta = json.loads(p.with_suffix(".json").read_text())
        m = cls(dim=meta["dim"])
        arr = np.load(str(p.with_suffix(".npz")))
        m.w, m.b = arr["w"], float(arr["b"][0])
        m.n_trained, m.metrics = meta.get("n_trained", 0), meta.get("metrics", {})
        return m


class RewardModelJudge:
    """Judge adapter: emits `rm_score` (probability of success) for any rollout."""

    def __init__(self, path: str):
        self.model = RewardModel.load(path[3:] if path.startswith("rm:") else path)
        self.model_name = f"rm:{path}"

    def __call__(self, task: TaskSpec, rollout: Rollout) -> list[Reward]:
        p = float(self.model.predict([rollout])[0])
        return [Reward(value=p, source=self.model_name, name="rm_score", rationale=f"learned reward model p(success)={p:.2f}")]


def labels_from_rollouts(rollouts: Iterable[Rollout], human_scores: Optional[dict[str, float]] = None) -> tuple[list[Rollout], list[float]]:
    """Label = human/product score when available (>=0.5 positive), else the verifier outcome."""
    xs, ys = [], []
    for ro in rollouts:
        if human_scores and ro.id in human_scores:
            y = 1.0 if human_scores[ro.id] >= 0.5 else 0.0
        else:
            traj = [r.value for r in ro.rewards if r.name == "task_success"]
            if not traj:
                continue
            y = 1.0 if traj[0] >= 1.0 else 0.0
        xs.append(ro)
        ys.append(y)
    return xs, ys
