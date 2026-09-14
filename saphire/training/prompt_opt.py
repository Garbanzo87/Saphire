"""Reflective prompt optimisation (GEPA-style).

Loop: evaluate the current best prompt on a minibatch -> collect failure evidence (verifier
rationales, wrong/expected tools) -> ask a reflection model to rewrite the system prompt ->
re-evaluate -> keep candidates on the per-task-family Pareto front.

With a hosted model the reflection is free-form; with the mock provider a deterministic
rule-mining reflection is used (it appends `Rule: when the request mentions "<phrase>", use
<tool>` lines that the mock policy honours) so the optimiser is fully testable offline.
"""
from __future__ import annotations

import random
import re
from typing import Any, Optional

from ..environments.base import Environment, get_environment
from ..evaluation.runner import evaluate
from ..sdk.agent import ToolAgent
from ..sdk.llm import LLM, MockLLM, get_llm
from ..sdk.types import AgentConfig, Message, Role, Rollout, TaskSpec

_STOP = set("the a an to of for and then please with on in at is it this that my me i you all order customer".split())


def _keywords(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z]+", text.lower()) if w not in _STOP and len(w) > 3]


class Candidate:
    def __init__(self, prompt: str, parent: Optional[int] = None):
        self.prompt = prompt
        self.parent = parent
        self.scores: dict[str, float] = {}  # per family
        self.mean = 0.0
        self.n = 0


def mine_rules(failures: list[dict[str, Any]], max_rules: int = 3) -> list[str]:
    """Deterministic reflection: for each (clause -> wrong tool) failure find the expected tool and a
    distinguishing keyword."""
    rules = []
    seen = set()
    for f in failures:
        for wrong, expected, clause in f.get("confusions", []):
            kws = _keywords(clause)
            for kw in kws:
                if kw in expected.replace("_", " ").split() or kw in clause.lower():
                    key = (kw, expected)
                    if key in seen:
                        continue
                    seen.add(key)
                    rules.append(f'Rule: when the request mentions "{kw}", use {expected}.')
                    break
            if len(rules) >= max_rules:
                return rules
    return rules


def collect_failures(rollouts: list[tuple[TaskSpec, Rollout]]) -> list[dict[str, Any]]:
    out = []
    for task, ro in rollouts:
        if ro.total_reward >= 0.99:
            continue
        exp = task.expected.get("tools", [])
        called = [tc.name for tc in ro.tool_calls]
        # clause-level confusion pairs (align by position, best effort)
        from ..sdk.llm import _clauses

        clauses = _clauses(task.instruction + "\n" + "\n".join(task.user_script))
        confusions = []
        for i, c in enumerate(called):
            if c not in exp:
                # find expected tool not yet called
                missing = [e for e in exp if e not in called]
                if missing and i < len(clauses):
                    confusions.append((c, missing[0], clauses[i]))
        rationale = next((r.rationale for r in ro.rewards if r.name == "task_success"), "")
        out.append({"task": task.instruction, "expected": exp, "called": called, "rationale": rationale, "confusions": confusions})
    return out


def reflect(llm: LLM, prompt: str, failures: list[dict[str, Any]]) -> str:
    if isinstance(llm, MockLLM):
        rules = mine_rules(failures)
        if not rules:
            return prompt
        existing = set(prompt.splitlines())
        new = [r for r in rules if r not in existing]
        return prompt + ("\n" + "\n".join(new) if new else "")
    evidence = "\n\n".join(
        f"TASK: {f['task']}\nEXPECTED TOOLS: {f['expected']}\nCALLED: {f['called']}\nVERIFIER: {f['rationale']}" for f in failures[:8])
    msg = (
        "You are optimising the system prompt of a tool-using agent. Below is the current prompt and evidence of failures. "
        "Rewrite the prompt so the agent avoids these failures. Keep it concise, keep everything that already works, and "
        "add explicit rules about which tool to use for which request. Reply with the new prompt only.\n\n"
        f"CURRENT PROMPT:\n{prompt}\n\nFAILURES:\n{evidence}\n\nNEW PROMPT:")
    resp = llm.complete([Message(role=Role.user, content=msg)], temperature=0.7)
    return resp.message.content.strip() or prompt


def optimize_prompt(config: AgentConfig, tasks: list[TaskSpec], envs: dict[str, Environment] | None = None,
                    reflection_model: str = "mock", iterations: int = 4, minibatch: int = 8, seed: int = 0,
                    policy_llm: Optional[LLM] = None) -> dict[str, Any]:
    rng = random.Random(seed)
    envs = envs or {}
    for t in tasks:
        envs.setdefault(t.env_name, get_environment(t.env_name))
    reflector = get_llm(reflection_model)
    pool: list[Candidate] = [Candidate(config.system_prompt)]
    history = []

    def score(cand: Candidate, batch: list[TaskSpec]) -> list[tuple[TaskSpec, Rollout]]:
        cfg = config.model_copy(update={"system_prompt": cand.prompt})
        agent = ToolAgent(cfg, llm=policy_llm)
        pairs: list[tuple[TaskSpec, Rollout]] = []

        def _cb(ro, task, rewards):
            pairs.append((task, ro))

        res = evaluate(agent, batch, envs=envs, on_rollout=_cb)
        for fam, m in res.by_family.items():
            prev = cand.scores.get(fam)
            cand.scores[fam] = m["task_success"] if prev is None else (prev + m["task_success"]) / 2
        cand.mean = res.metrics["task_success"] if cand.n == 0 else (cand.mean * cand.n + res.metrics["task_success"] * len(batch)) / (cand.n + len(batch))
        cand.n += len(batch)
        return pairs

    batch = rng.sample(tasks, min(minibatch, len(tasks)))
    pairs = score(pool[0], batch)
    history.append({"iteration": 0, "candidate": 0, "task_success": pool[0].mean, "prompt": pool[0].prompt})
    for it in range(1, iterations + 1):
        # Pareto-ish parent selection: best on a random family, tie -> best mean
        fams = sorted({f for c in pool for f in c.scores})
        fam = rng.choice(fams) if fams else None
        parent_idx = max(range(len(pool)), key=lambda i: (pool[i].scores.get(fam, 0.0) if fam else 0.0, pool[i].mean))
        parent = pool[parent_idx]
        failures = collect_failures(pairs)
        child = Candidate(reflect(reflector, parent.prompt, failures), parent=parent_idx)
        batch = rng.sample(tasks, min(minibatch, len(tasks)))
        pairs = score(child, batch)
        if child.prompt != parent.prompt:
            pool.append(child)
        history.append({"iteration": it, "candidate": len(pool) - 1, "parent": parent_idx, "task_success": child.mean,
                        "prompt": child.prompt, "n_failures": len(failures)})
    best = max(pool, key=lambda c: (c.mean, -len(c.prompt)))
    return {"best_prompt": best.prompt, "best_score": best.mean, "baseline_score": pool[0].mean,
            "n_candidates": len(pool), "history": history}
