"""Exemplar store: retrieval of successful past trajectories as in-context demonstrations.

The cheapest "online learning" mechanism for an agent stack: every high-reward rollout is
compressed into a short (instruction -> tool sequence) demonstration and indexed; at inference
the k most similar demonstrations are injected into the system prompt. Updated continuously by
the online loop without touching model weights.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .router import hashed_features
from .types import Rollout


class ExemplarStore:
    def __init__(self, dim: int = 4096, max_items: int = 500, min_reward: float = 0.99):
        self.dim = dim
        self.max_items = max_items
        self.min_reward = min_reward
        self.items: list[dict] = []
        self._X: np.ndarray | None = None

    @staticmethod
    def summarize(rollout: Rollout, role: str | None = None) -> list[dict]:
        """Compress a rollout into (instruction -> tool sequence) items.

        Single-agent rollouts yield one item. With `role`, each delegation episode of that role (identified by the
        task message the role received) yields one item, using only that role's steps."""
        steps = [s for s in rollout.steps if role is None or s.role == role]
        if not steps:
            return []
        if role is None:
            calls = [{"tool": tc.name, "arguments": tc.arguments} for s in steps for tc in s.response.tool_calls]
            instr = next((m.content for m in steps[0].prompt_messages if m.role.value == "user"), "")
            return [{"instruction": instr, "calls": calls, "reward": rollout.total_reward, "rollout_id": rollout.id}]
        episodes: dict[str, list] = {}
        for s in steps:
            task_msg = next((m.content for m in s.prompt_messages if m.role.value == "user"), "")
            episodes.setdefault(task_msg.split("\n\nContext:")[0], []).extend(
                {"tool": tc.name, "arguments": tc.arguments} for tc in s.response.tool_calls)
        return [{"instruction": k, "calls": v, "reward": rollout.total_reward, "rollout_id": rollout.id, "role": role}
                for k, v in episodes.items()]

    def add_rollout(self, rollout: Rollout, role: str | None = None) -> bool:
        if rollout.total_reward < self.min_reward or not rollout.tool_calls:
            return False
        added = False
        for item in self.summarize(rollout, role):
            if not item["calls"] or any(i["instruction"] == item["instruction"] for i in self.items):
                continue
            self.items.append(item)
            added = True
        if len(self.items) > self.max_items:
            self.items = self.items[-self.max_items:]
        self._X = None
        return added

    def _matrix(self) -> np.ndarray:
        if self._X is None:
            self._X = np.stack([hashed_features(i["instruction"], self.dim) for i in self.items]) if self.items else np.zeros((0, self.dim))
        return self._X

    def retrieve(self, query: str, k: int = 2) -> list[dict]:
        if not self.items or k <= 0:
            return []
        q = hashed_features(query, self.dim)
        sims = self._matrix() @ q
        order = np.argsort(-sims)[:k]
        return [self.items[i] for i in order if sims[i] > 0]

    def render(self, query: str, k: int = 2) -> str:
        ex = self.retrieve(query, k)
        if not ex:
            return ""
        lines = ["Here are examples of similar tasks solved correctly:"]
        for e in ex:
            seq = " -> ".join(f"{c['tool']}({json.dumps(c['arguments'])})" for c in e["calls"])
            lines.append(f"- Task: {e['instruction']}\n  Tools: {seq}")
        return "\n".join(lines)

    def save(self, path: str | Path) -> str:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"dim": self.dim, "max_items": self.max_items, "min_reward": self.min_reward,
                                 "items": self.items}))
        return str(p)

    @classmethod
    def load(cls, path: str | Path) -> "ExemplarStore":
        from .artifacts import resolve

        d = json.loads(Path(resolve(str(path))).read_text())
        s = cls(dim=d["dim"], max_items=d["max_items"], min_reward=d["min_reward"])
        s.items = d["items"]
        return s
