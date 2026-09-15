"""Task mining: production traffic -> evaluation tasks, and coverage of production by the eval sets."""
from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Optional

from ..sdk.types import Role, Rollout, TaskSpec

HANDOFF = "delegate_to_"


def _instruction(ro: Rollout) -> str:
    for m in (ro.steps[0].prompt_messages if ro.steps else []):
        if m.role == Role.user:
            return m.content
    return ""


def _tool_sequence(ro: Rollout) -> list[str]:
    seq = []
    for tc in ro.tool_calls:
        if not tc.name.startswith(HANDOFF) and (not seq or seq[-1] != tc.name):
            seq.append(tc.name)
    return seq


def mine_tasks_from_rollouts(rollouts: Iterable[Rollout], min_reward: float = 0.99, human_scores: Optional[dict[str, float]] = None,
                             min_human: float = 0.5, max_tasks: int = 500) -> list[TaskSpec]:
    """Successful production rollouts (by verifier reward, or by a human/product score when supplied) become tasks whose
    ground truth is the tool sequence that succeeded. Duplicates by instruction are collapsed."""
    seen: set[str] = set()
    out: list[TaskSpec] = []
    for ro in rollouts:
        ok = ro.total_reward >= min_reward
        if human_scores is not None and ro.id in human_scores:
            ok = human_scores[ro.id] >= min_human
        instr = _instruction(ro)
        seq = _tool_sequence(ro)
        if not ok or not instr or not seq or instr in seen:
            continue
        seen.add(instr)
        out.append(TaskSpec(env_name=ro.env_name, instruction=instr, expected={"tools": seq, "ordered": len(seq) > 1},
                            tags=["mined", "production"], difficulty="hard" if len(seq) >= 4 else "medium" if len(seq) > 1 else "easy",
                            max_steps=max(4, 2 * len(seq) + 2), metadata={"source_rollout": ro.id, "source_agent": ro.agent_id}))
        if len(out) >= max_tasks:
            break
    return out


def mine_tasks_from_traces(traces: Iterable[dict[str, Any]], spans_by_trace: dict[str, list[dict[str, Any]]],
                           scores_by_trace: Optional[dict[str, float]] = None, env_name: str = "production", min_score: float = 0.5,
                           max_tasks: int = 500) -> list[TaskSpec]:
    """Bring-your-own-agent traces (OTel spans) -> tasks: root span input = instruction, tool spans = expected tools.
    Only traces with a score >= min_score (thumbs-up, CSAT, judge) are used when scores are supplied."""
    seen: set[str] = set()
    out: list[TaskSpec] = []
    for t in traces:
        tid = t["id"]
        if scores_by_trace is not None and scores_by_trace.get(tid, 0.0) < min_score:
            continue
        spans = spans_by_trace.get(tid, [])
        root = next((s for s in spans if not s.get("parent_span_id")), None)
        instr = str((root or {}).get("attributes", {}).get("input.value", "")) or t.get("name", "")
        tools = [s["attributes"].get("tool.name") or s["name"] for s in sorted(spans, key=lambda s: s.get("start_time") or 0) if s.get("kind") == "tool"]
        seq = [x for i, x in enumerate(tools) if i == 0 or tools[i - 1] != x]
        if not instr or not seq or instr in seen:
            continue
        seen.add(instr)
        out.append(TaskSpec(env_name=env_name, instruction=instr[:2000], expected={"tools": seq, "ordered": len(seq) > 1},
                            tags=["mined", "trace"], metadata={"source_trace": tid}))
        if len(out) >= max_tasks:
            break
    return out


def coverage_gaps(production: Iterable[Rollout], eval_tasks: Iterable[TaskSpec]) -> dict[str, Any]:
    """Which production tool patterns are not represented in the evaluation datasets?"""
    prod = Counter(tuple(_tool_sequence(ro)) for ro in production)
    prod.pop((), None)
    covered = {tuple(t.expected.get("tools", [])) for t in eval_tasks}
    covered_tools = {x for c in covered for x in c}
    missing = [(list(p), n) for p, n in prod.most_common() if p not in covered]
    total = sum(prod.values())
    uncovered_traffic = sum(n for _, n in missing)
    return {"production_patterns": len(prod), "covered_patterns": len(prod) - len(missing), "uncovered_patterns": len(missing),
            "uncovered_traffic_share": (uncovered_traffic / total) if total else 0.0,
            "top_uncovered": [{"tools": p, "count": n} for p, n in missing[:10]],
            "tools_never_evaluated": sorted({x for p, _ in missing for x in p} - covered_tools)}
