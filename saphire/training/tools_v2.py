"""Tool learning beyond routing: reliability statistics as router priors, and tool-description optimisation.

* `tool_stats(rollouts)`   – per tool: calls, error rate, success rate of rollouts that used it, mean latency, confusions
                             (which expected tool it was mistaken for). Stored on the router as a prior: unreliable or
                             frequently-confused tools are ranked lower unless the learned head is confident.
* `optimize_tool_descriptions(env, rollouts)` – for confusable pairs, append disambiguating hints
                             ("Use for …; not for …") to descriptions. Applied through `AgentConfig.tool_description_overrides`
                             so the change is versioned with the agent, not the environment.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Iterable

from ..environments.base import Environment
from ..sdk.llm import _clauses, _tokens
from ..sdk.types import Rollout

HANDOFF = "delegate_to_"


def tool_stats(rollouts: Iterable[Rollout]) -> dict[str, dict[str, Any]]:
    calls: Counter = Counter()
    errors: Counter = Counter()
    latency: dict[str, list[float]] = defaultdict(list)
    used_success: dict[str, list[float]] = defaultdict(list)
    confused: dict[str, Counter] = defaultdict(Counter)  # wrong tool -> Counter(expected tool)
    for ro in rollouts:
        succ = 1.0 if ro.total_reward >= 0.99 else 0.0
        bad = {(r.step_index, r.metadata.get("tool")) for r in ro.rewards if r.step_index is not None and r.name == "tool_correct" and r.value < 0}
        for s in ro.steps:
            for tr in s.tool_results:
                if tr.name.startswith(HANDOFF):
                    continue
                calls[tr.name] += 1
                latency[tr.name].append(tr.latency_ms)
                used_success[tr.name].append(succ)
                if tr.error:
                    errors[tr.name] += 1
            for tc in s.response.tool_calls:
                if (s.index, tc.name) in bad and not tc.name.startswith(HANDOFF):
                    exp = ro.metadata.get("expected_tools") or []
                    for e in exp:
                        confused[tc.name][e] += 1
    return {t: {"calls": c, "error_rate": errors[t] / c, "success_rate": sum(used_success[t]) / len(used_success[t]),
                "latency_ms": sum(latency[t]) / len(latency[t]), "confused_with": dict(confused[t].most_common(3))} for t, c in calls.items()}


def confusion_pairs(rollouts: Iterable[Rollout], expected_by_task: dict[str, list[str]]) -> Counter:
    """(wrong_tool, expected_tool, clause_keywords) counts from wrong-tool steps."""
    pairs: Counter = Counter()
    for ro in rollouts:
        exp = expected_by_task.get(ro.task_id, [])
        if not exp:
            continue
        called = [tc.name for tc in ro.tool_calls if not tc.name.startswith(HANDOFF)]
        missing = [e for e in exp if e not in called]
        bad = {(r.step_index, r.metadata.get("tool")) for r in ro.rewards if r.step_index is not None and r.name == "tool_correct" and r.value < 0}
        for s in ro.steps:
            task_text = next((m.content for m in s.prompt_messages if m.role.value == "user"), "").split("\n\nContext:")[0]
            for tc in s.response.tool_calls:
                if (s.index, tc.name) in bad and missing and not tc.name.startswith(HANDOFF):
                    toks = _tokens(_clauses(task_text)[0] if _clauses(task_text) else task_text)
                    kws = tuple(sorted(t for t in toks if not any(ch.isdigit() for ch in t))[:4])
                    pairs[(tc.name, missing[0], kws)] += 1
    return pairs


def optimize_tool_descriptions(env: Environment, rollouts: list[Rollout], expected_by_task: dict[str, list[str]], max_hints: int = 8) -> dict[str, str]:
    """Return {tool_name: improved description} for the most confused tools."""
    specs = {s.name: s for s in env.tools.specs()}
    pairs = confusion_pairs(rollouts, expected_by_task)
    overrides: dict[str, str] = {}
    for (wrong, expected, kws), n in pairs.most_common(max_hints):
        if expected not in specs or wrong not in specs:
            continue
        kw_text = ", ".join(kws) if kws else "this kind of request"
        e_desc = overrides.get(expected, specs[expected].description)
        if kw_text not in e_desc:
            overrides[expected] = f"{e_desc} Use for requests mentioning {kw_text}."
        w_desc = overrides.get(wrong, specs[wrong].description)
        if expected not in w_desc:
            overrides[wrong] = f"{w_desc} Not for {kw_text}; use {expected} for that."
    return overrides
