"""Helpers shared by verifiers: multi-agent aware tool-call accounting and step-level credit assignment."""
from __future__ import annotations

from ..sdk.types import Reward, Rollout

HANDOFF_PREFIX = "delegate_to_"


def effective_calls(rollout: Rollout) -> list[str]:
    """Tool names actually executed against the environment (hand-offs between agents excluded)."""
    return [tc.name for tc in rollout.tool_calls if not tc.name.startswith(HANDOFF_PREFIX)]


def tool_selection_scores(expected: list[str], called: list[str]) -> tuple[float, bool]:
    es, cs = set(expected), set(called)
    tp = len(es & cs)
    prec = tp / len(cs) if cs else 0.0
    rec = tp / len(es) if es else 1.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return f1, es <= cs


def ordered_ok(expected: list[str], called: list[str]) -> bool:
    es = set(expected)
    it = iter([n for n in called if n in es])
    return all(any(x == e for x in it) for e in expected)


def step_tool_rewards(rollout: Rollout, expected: list[str]) -> list[Reward]:
    """Per-step `tool_correct` rewards.

    * environment tool calls: +1 if the tool is expected, -1 otherwise;
    * orchestrator hand-offs: +1 if the delegated episode (the specialist steps that follow until control returns)
      executed at least one expected tool, -1 otherwise — credit assignment for the routing decision.
    """
    out: list[Reward] = []
    steps = rollout.steps
    es = set(expected)
    for i, s in enumerate(steps):
        for tc in s.response.tool_calls:
            if tc.name.startswith(HANDOFF_PREFIX):
                good = False
                for nxt in steps[i + 1:]:
                    if nxt.role == s.role:
                        break
                    if any(t.name in es for t in nxt.response.tool_calls):
                        good = True
                        break
                out.append(Reward(value=1.0 if good else -1.0, source="verifier", name="tool_correct", step_index=s.index,
                                  metadata={"tool": tc.name, "handoff": True, "role": s.role}))
            else:
                out.append(Reward(value=1.0 if tc.name in es else -1.0, source="verifier", name="tool_correct", step_index=s.index,
                                  metadata={"tool": tc.name, "role": s.role}))
    return out
