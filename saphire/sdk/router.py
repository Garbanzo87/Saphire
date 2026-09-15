"""Learnable tool router.

Large tool catalogues (hundreds of MCP servers / API endpoints) hurt tool selection: models
pick look-alike tools or blow their context. A `ToolRouter` ranks the catalogue for the
current request and only the top-k are exposed to the policy.

The router combines a zero-shot lexical prior (token overlap with tool name/description) with
a multinomial logistic-regression head over hashed n-gram features that is *trained from
reward-labelled rollouts* (`saphire.training.router_trainer`). It is pure numpy, trains in
seconds on CPU, and is one of the online-learnable components of an agent version.
"""
from __future__ import annotations

import json
import re
import zlib
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from .types import ToolSpec

_STOP = set("the a an to of for and then please with on in at is it this that my me i you".split())


def tokenize(text: str) -> list[str]:
    toks = [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOP and len(w) > 1]
    toks = [re.sub(r"\d+", "#", w) for w in toks]  # normalise ids
    return toks


def hashed_features(text: str, dim: int) -> np.ndarray:
    toks = tokenize(text)
    grams = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
    x = np.zeros(dim, dtype=np.float32)
    for g in grams:
        x[zlib.crc32(g.encode()) % dim] += 1.0
    n = np.linalg.norm(x)
    return x / n if n > 0 else x


class ToolRouter:
    def __init__(self, tool_names: list[str], dim: int = 4096, alpha: float = 0.5, seed: int = 0):
        self.tool_names = list(tool_names)
        self.index = {n: i for i, n in enumerate(self.tool_names)}
        self.dim = dim
        self.alpha = alpha  # weight of learned head vs lexical prior
        rng = np.random.default_rng(seed)
        self.W = rng.normal(0, 0.01, size=(dim, len(self.tool_names))).astype(np.float32)
        self.b = np.zeros(len(self.tool_names), dtype=np.float32)
        self.trained_examples = 0
        self.history: list[dict] = []

    # ---------- scoring ----------
    def lexical_scores(self, query: str, specs: list[ToolSpec]) -> np.ndarray:
        q = set(tokenize(query))
        out = np.zeros(len(specs), dtype=np.float32)
        for i, s in enumerate(specs):
            tt = set(tokenize(s.name.replace("_", " ") + " " + s.description))
            out[i] = len(q & tt) / (len(tt) ** 0.5 + 1e-6)
        if out.max() > 0:
            out = out / out.max()
        return out

    def learned_probs(self, query: str) -> np.ndarray:
        x = hashed_features(query, self.dim)
        logits = x @ self.W + self.b
        logits -= logits.max()
        p = np.exp(logits)
        return p / p.sum()

    def rank(self, query: str, specs: list[ToolSpec], top_k: int = 8, exclude: Iterable[str] = ()) -> list[ToolSpec]:
        """Return the top-k specs for `query` (always returns at least the lexical ranking)."""
        if top_k <= 0 or top_k >= len(specs):
            return list(specs)
        lex = self.lexical_scores(query, specs)
        learned = self.learned_probs(query) if self.trained_examples > 0 else None
        scores = []
        for i, s in enumerate(specs):
            sc = (1 - self.alpha) * lex[i]
            if learned is not None and s.name in self.index:
                sc += self.alpha * float(learned[self.index[s.name]]) * len(specs) ** 0.5
            scores.append(sc)
        order = np.argsort(-np.asarray(scores), kind="stable")
        excl = set(exclude)
        picked = [specs[i] for i in order if specs[i].name not in excl][:top_k]
        return picked

    # ---------- training ----------
    def fit(self, examples: list[tuple[str, str, float]], epochs: int = 30, lr: float = 0.5, l2: float = 1e-4,
            seed: int = 0) -> dict:
        """Train the head on (query, tool_name, weight) examples with weighted softmax cross-entropy."""
        ex = [(q, n, w) for q, n, w in examples if n in self.index and w != 0]
        if not ex:
            return {"examples": 0}
        X = np.stack([hashed_features(q, self.dim) for q, _, _ in ex])
        y = np.array([self.index[n] for _, n, _ in ex])
        w = np.array([float(wt) for _, _, wt in ex], dtype=np.float32)
        rng = np.random.default_rng(seed)
        n = len(ex)
        loss = 0.0
        for _ in range(epochs):
            idx = rng.permutation(n)
            for start in range(0, n, 32):
                bi = idx[start:start + 32]
                xb, yb, wb = X[bi], y[bi], w[bi]
                logits = xb @ self.W + self.b
                logits -= logits.max(axis=1, keepdims=True)
                p = np.exp(logits)
                p /= p.sum(axis=1, keepdims=True)
                onehot = np.zeros_like(p)
                onehot[np.arange(len(bi)), yb] = 1.0
                # positive weight: push towards label; negative weight: push away from label
                grad = (p - onehot) * wb[:, None]
                loss = float(-(np.log(p[np.arange(len(bi)), yb] + 1e-9) * wb).mean())
                self.W -= lr * (xb.T @ grad / len(bi) + l2 * self.W)
                self.b -= lr * grad.mean(axis=0)
        self.trained_examples += n
        rec = {"examples": n, "final_loss": loss, "epochs": epochs}
        self.history.append(rec)
        return rec

    def accuracy(self, examples: list[tuple[str, str, float]], specs: list[ToolSpec], top_k: int = 1) -> float:
        hits = 0
        tot = 0
        for q, n, w in examples:
            if w <= 0:
                continue
            tot += 1
            if n in {s.name for s in self.rank(q, specs, top_k=top_k)}:
                hits += 1
        return hits / tot if tot else 0.0

    # ---------- persistence ----------
    def save(self, path: str | Path) -> str:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(path.with_suffix(".npz")), W=self.W, b=self.b)
        meta = {"tool_names": self.tool_names, "dim": self.dim, "alpha": self.alpha,
                "trained_examples": self.trained_examples, "history": self.history}
        path.with_suffix(".json").write_text(json.dumps(meta))
        return str(path.with_suffix(".json"))

    @classmethod
    def load(cls, path: str | Path) -> "ToolRouter":
        from .artifacts import resolve

        path = Path(resolve(str(path), siblings=("npz",)))
        meta = json.loads(path.with_suffix(".json").read_text())
        r = cls(meta["tool_names"], dim=meta["dim"], alpha=meta["alpha"])
        arr = np.load(str(path.with_suffix(".npz")))
        r.W, r.b = arr["W"], arr["b"]
        r.trained_examples = meta.get("trained_examples", 0)
        r.history = meta.get("history", [])
        return r


def load_router(path: Optional[str]) -> Optional[ToolRouter]:
    if not path:
        return None
    return ToolRouter.load(path)
