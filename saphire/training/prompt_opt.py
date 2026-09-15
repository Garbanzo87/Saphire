"""Reflective, Pareto-based prompt optimisation (GEPA-class) for single agents and multi-agent systems.

Algorithm (per iteration)
  1. Pick a parent from the **per-task Pareto front**: candidates that are the best on at least one task, sampled in
     proportion to how many tasks they win (GEPA's selection rule) — this preserves diverse partial solutions.
  2. Evaluate the parent on a minibatch, collect **failure evidence** (verifier rationales, wrong→expected tool
     confusions, per-role in multi-agent systems).
  3. **Reflect**: a reflection model rewrites the target prompt(s) given the evidence (with the mock provider a
     deterministic rule miner is used so the optimiser is fully testable offline).
  4. Every few iterations, **merge** two front candidates (union of their distinct instruction lines) — system-aware
     crossover.
  5. Score the child on the minibatch; keep it if it enters the front. A rollout **budget** bounds the total cost.

Targets
  * single agent: the system prompt
  * multi-agent: `target="orchestrator"`, a role name, or `"joint"` (all trainable roles + orchestrator optimised
    together, each with its own evidence — MAMUT-style system-level optimisation)
"""
from __future__ import annotations

import random
import re
from collections import defaultdict
from typing import Any, Optional

from ..environments.base import Environment, get_environment
from ..evaluation.runner import evaluate
from ..sdk.llm import LLM, MockLLM, _clauses, get_llm
from ..sdk.multi_agent import ORCHESTRATOR, build_agent
from ..sdk.types import AgentConfig, Message, Role, Rollout, TaskSpec

_STOP = set("the a an to of for and then please with on in at is it this that my me i you all order customer".split())
HANDOFF = "delegate_to_"


def _keywords(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z]+", text.lower()) if w not in _STOP and len(w) > 3]


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------
def collect_failures(rollouts: list[tuple[TaskSpec, Rollout]], role: Optional[str] = None,
                     allowed_tools: Optional[set[str]] = None) -> list[dict[str, Any]]:
    """Failure evidence. With `role`, confusions are computed from that role's steps only (orchestrator: hand-offs) and the
    suggested tool is restricted to `allowed_tools` (the role's own tool subset)."""
    out = []
    for task, ro in rollouts:
        if ro.total_reward >= 0.99:
            continue
        exp = task.expected.get("tools", [])
        if role is None:
            called = [tc.name for tc in ro.tool_calls if not tc.name.startswith(HANDOFF)]
            clauses = _clauses(task.instruction + "\n" + "\n".join(task.user_script))
            confusions = []
            for i, c in enumerate(called):
                if c not in exp:
                    missing = [e for e in exp if e not in called]
                    if missing and i < len(clauses):
                        confusions.append((c, missing[0], clauses[i]))
        else:
            confusions = []
            step_rewards = {(r.step_index, r.metadata.get("tool")): r.value for r in ro.rewards if r.step_index is not None and r.name == "tool_correct"}
            for s in ro.steps:
                if s.role != role:
                    continue
                task_text = next((m.content for m in s.prompt_messages if m.role == Role.user), task.instruction).split("\n\nContext:")[0]
                for tc in s.response.tool_calls:
                    if step_rewards.get((s.index, tc.name), 1.0) < 0:
                        if role == ORCHESTRATOR:
                            confusions.append((tc.name, "the right specialist", task_text))
                        else:
                            missing = [e for e in exp if e not in [x.name for x in ro.tool_calls] and (allowed_tools is None or e in allowed_tools)]
                            if missing:
                                confusions.append((tc.name, missing[0], task_text))
        rationale = next((r.rationale for r in ro.rewards if r.name == "task_success"), "")
        out.append({"task": task.instruction, "expected": exp, "called": [tc.name for tc in ro.tool_calls], "rationale": rationale,
                    "confusions": confusions, "role": role})
    return out


def mine_rules(failures: list[dict[str, Any]], max_rules: int = 3) -> list[str]:
    """Deterministic reflection: `Rule: when the request mentions "<keyword>", use <tool>.`"""
    rules, seen = [], set()
    for f in failures:
        for wrong, expected, clause in f.get("confusions", []):
            if expected.startswith("the right"):
                continue
            for kw in _keywords(clause):
                key = (kw, expected)
                if key not in seen:
                    seen.add(key)
                    rules.append(f'Rule: when the request mentions "{kw}", use {expected}.')
                    break
            if len(rules) >= max_rules:
                return rules
    return rules


def reflect(llm: LLM, prompt: str, failures: list[dict[str, Any]], role: Optional[str] = None) -> str:
    if isinstance(llm, MockLLM):
        new = [r for r in mine_rules(failures) if r not in set(prompt.splitlines())]
        return prompt + ("\n" + "\n".join(new) if new else "")
    evidence = "\n\n".join(
        f"TASK: {f['task']}\nEXPECTED TOOLS: {f['expected']}\nCALLED: {f['called']}\nVERIFIER: {f['rationale']}\nCONFUSIONS: {f['confusions'][:3]}"
        for f in failures[:8])
    who = f"the '{role}' agent of a multi-agent system" if role else "a tool-using agent"
    msg = (f"You are optimising the system prompt of {who}. Below is the current prompt and evidence of failures. Rewrite the prompt so "
           "the agent avoids these failures. Keep what already works, be concise, add explicit rules about which tool (or specialist) to "
           f"use for which request. Reply with the new prompt only.\n\nCURRENT PROMPT:\n{prompt}\n\nFAILURES:\n{evidence}\n\nNEW PROMPT:")
    resp = llm.complete([Message(role=Role.user, content=msg)], temperature=0.7)
    return resp.message.content.strip() or prompt


def merge_prompts(a: str, b: str) -> str:
    """System-aware crossover: keep a's text and append b's distinct instruction lines."""
    seen = set(a.splitlines())
    extra = [ln for ln in b.splitlines() if ln.strip() and ln not in seen]
    return a + ("\n" + "\n".join(extra) if extra else "")


# ---------------------------------------------------------------------------
# Candidates and Pareto front
# ---------------------------------------------------------------------------
class Candidate:
    def __init__(self, prompts: dict[str, str], parent: Optional[int] = None, origin: str = "init"):
        self.prompts = prompts  # target -> prompt
        self.parent = parent
        self.origin = origin
        self.task_scores: dict[str, float] = {}
        self.n = 0

    @property
    def mean(self) -> float:
        return sum(self.task_scores.values()) / len(self.task_scores) if self.task_scores else 0.0


def pareto_front(pool: list[Candidate]) -> dict[int, int]:
    """candidate index -> number of tasks on which it is (tied-)best. Only candidates winning ≥1 task are returned."""
    tasks = {t for c in pool for t in c.task_scores}
    wins: dict[int, int] = defaultdict(int)
    for t in tasks:
        scored = [(c.task_scores[t], i) for i, c in enumerate(pool) if t in c.task_scores]
        best = max(s for s, _ in scored)
        for s, i in scored:
            if s == best:
                wins[i] += 1
    return dict(wins)


def _apply(config: AgentConfig, prompts: dict[str, str]) -> AgentConfig:
    cfg = config.model_copy(deep=True)
    for target, prompt in prompts.items():
        if target == "main":
            cfg.system_prompt = prompt
        elif target == ORCHESTRATOR:
            cfg.orchestrator_prompt = prompt
        else:
            cfg.roles[target].system_prompt = prompt
    return cfg


def _targets(config: AgentConfig, target: str) -> list[str]:
    if not config.is_multi_agent:
        return ["main"]
    if target == "joint":
        return [ORCHESTRATOR] + [r for r, c in config.roles.items() if c.trainable]
    if target in ("main", "orchestrator"):
        return [ORCHESTRATOR]
    if target not in config.roles:
        raise KeyError(f"unknown role {target}")
    return [target]


def _initial_prompts(config: AgentConfig, targets: list[str]) -> dict[str, str]:
    out = {}
    for t in targets:
        if t == "main":
            out[t] = config.system_prompt
        elif t == ORCHESTRATOR:
            out[t] = config.orchestrator_prompt or config.system_prompt
        else:
            out[t] = config.roles[t].system_prompt or config.system_prompt
    return out


# ---------------------------------------------------------------------------
def optimize_prompt(config: AgentConfig, tasks: list[TaskSpec], envs: dict[str, Environment] | None = None, reflection_model: str = "mock",
                    iterations: int = 4, minibatch: int = 8, seed: int = 0, policy_llm: Optional[LLM] = None, target: str = "main",
                    merge_every: int = 3, max_rollouts: Optional[int] = None, families: Optional[list[str]] = None) -> dict[str, Any]:
    rng = random.Random(seed)
    envs = envs or {}
    if families:
        tasks = [t for t in tasks if t.tags and t.tags[0] in families] or tasks
    for t in tasks:
        envs.setdefault(t.env_name, get_environment(t.env_name))
    reflector = get_llm(reflection_model)
    targets = _targets(config, target)
    pool: list[Candidate] = [Candidate(_initial_prompts(config, targets))]
    history: list[dict[str, Any]] = []
    used = {"rollouts": 0}

    def score(cand: Candidate, batch: list[TaskSpec]) -> list[tuple[TaskSpec, Rollout]]:
        cfg = _apply(config, cand.prompts)
        agent = build_agent(cfg, llm=policy_llm) if not cfg.is_multi_agent else build_agent(cfg, llm=policy_llm)
        pairs: list[tuple[TaskSpec, Rollout]] = []
        res = evaluate(agent, batch, envs=envs, on_rollout=lambda ro, task, _r: pairs.append((task, ro)))
        for rec in res.per_rollout:
            prev = cand.task_scores.get(rec.task_id)
            cand.task_scores[rec.task_id] = rec.task_success if prev is None else max(prev, rec.task_success)
        cand.n += len(batch)
        used["rollouts"] += len(batch)
        return pairs

    batch = rng.sample(tasks, min(minibatch, len(tasks)))
    pairs = score(pool[0], batch)
    history.append({"iteration": 0, "candidate": 0, "origin": "init", "mean": pool[0].mean, "tasks_scored": len(pool[0].task_scores)})
    for it in range(1, iterations + 1):
        if max_rollouts and used["rollouts"] >= max_rollouts:
            history.append({"iteration": it, "stopped": "budget"})
            break
        front = pareto_front(pool)
        idxs, weights = zip(*front.items()) if front else ((0,), (1,))
        parent_idx = rng.choices(idxs, weights=weights, k=1)[0]
        parent = pool[parent_idx]
        if merge_every and it % merge_every == 0 and len(front) >= 2:
            other_idx = rng.choice([i for i in front if i != parent_idx])
            child = Candidate({t: merge_prompts(parent.prompts[t], pool[other_idx].prompts[t]) for t in targets}, parent=parent_idx, origin=f"merge:{parent_idx}+{other_idx}")
        else:
            # evidence from the parent's most recent minibatch (re-score the parent if it has none)
            if not pairs or pairs[0][1].agent_version != config.version:
                pairs = score(parent, rng.sample(tasks, min(minibatch, len(tasks))))
            new_prompts = {}
            for t in targets:
                role = None if t == "main" else t
                allowed = set(config.roles[t].tool_names) if t in config.roles and config.roles[t].tool_names else None
                new_prompts[t] = reflect(reflector, parent.prompts[t], collect_failures(pairs, role=role, allowed_tools=allowed), role=role)
            child = Candidate(new_prompts, parent=parent_idx, origin="reflect")
        batch = rng.sample(tasks, min(minibatch, len(tasks)))
        pairs = score(child, batch)
        accepted = child.prompts != parent.prompts and (len(pool) < 2 or child.mean >= min(c.mean for c in pool) or child.n < minibatch)
        if accepted:
            pool.append(child)
        history.append({"iteration": it, "origin": child.origin, "parent": parent_idx, "mean": child.mean, "accepted": accepted,
                        "front_size": len(pareto_front(pool)), "rollouts_used": used["rollouts"]})
    # final: pick by mean over scored tasks, tie-break shorter prompts
    best = max(pool, key=lambda c: (c.mean, -sum(len(p) for p in c.prompts.values())))
    return {"best_prompts": best.prompts, "best_prompt": best.prompts.get("main") or next(iter(best.prompts.values())),
            "best_score": best.mean, "baseline_score": pool[0].mean, "n_candidates": len(pool), "front": pareto_front(pool),
            "targets": targets, "rollouts_used": used["rollouts"], "history": history, "best_config": _apply(config, best.prompts).model_dump()}
