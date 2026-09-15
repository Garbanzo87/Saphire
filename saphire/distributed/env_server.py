"""Environment / reward server: lets *remote* RL trainers (verl, OpenRLHF, SkyRL, custom Ray jobs) use Saphire
environments without importing Saphire into the training image.

Stateless (works behind any number of API replicas):
  POST /v1/env/reward        {env, task, prefix_calls, completion} -> environment-grounded reward for one sampled action
  POST /v1/env/reward/batch  [{...}, ...]                          -> rewards for a whole GRPO group
  POST /v1/env/verify        {env, task, rollout}                  -> verifier rewards for a full trajectory (replayed)

Stateful sessions (pin to one replica, or run `saphire env-server` as a dedicated service):
  POST /v1/env/sessions {env, task}            -> {session_id, tools, instruction}
  POST /v1/env/sessions/{id}/step {tool_call}  -> tool result
  POST /v1/env/sessions/{id}/user_turn         -> next simulated user message (or null)
  POST /v1/env/sessions/{id}/finish {rollout}  -> rewards from the live state; session closed
"""
from __future__ import annotations

import threading
import time
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..environments.base import get_environment, list_environments
from ..sdk.types import Rollout, TaskSpec, ToolCall
from ..training.trl_trainers import replay_reward

router = APIRouter(prefix="/env", tags=["environment-server"])


def _auth():  # replaced with the real dependency when mounted inside the main app
    return None


class RewardIn(BaseModel):
    env: str
    task: TaskSpec
    prefix_calls: list[dict[str, Any]] = Field(default_factory=list)
    completion: str


class VerifyIn(BaseModel):
    env: str
    task: TaskSpec
    rollout: Rollout


class SessionIn(BaseModel):
    env: str
    task: TaskSpec


class StepIn(BaseModel):
    tool_call: ToolCall


_SESSIONS: dict[str, dict[str, Any]] = {}
_LOCK = threading.Lock()
SESSION_TTL_S = 3600


@router.get("/environments")
def environments():
    return [get_environment(n).info() for n in list_environments()]


@router.post("/reward")
def reward(body: RewardIn):
    r = replay_reward(body.completion, body.env, body.task.id, body.prefix_calls, body.task.expected.get("tools", []),
                      tasks_by_id={body.task.id: body.task})
    return {"reward": r}


@router.post("/reward/batch")
def reward_batch(body: list[RewardIn]):
    return {"rewards": [replay_reward(b.completion, b.env, b.task.id, b.prefix_calls, b.task.expected.get("tools", []),
                                      tasks_by_id={b.task.id: b.task}) for b in body]}


@router.post("/verify")
def verify(body: VerifyIn):
    """Replay every tool call of `rollout` in a fresh environment and run the verifier on the resulting state."""
    env = get_environment(body.env)
    state = env.reset(body.task)
    for tc in body.rollout.tool_calls:
        if not tc.name.startswith("delegate_to_"):
            env.tools.call(tc, state=state)
    rewards = env.verify(body.task, body.rollout, state)
    return {"rewards": [r.model_dump() for r in rewards], "total_reward": sum(r.value for r in rewards if r.step_index is None) /
            max(1, sum(1 for r in rewards if r.step_index is None))}


@router.post("/sessions", status_code=201)
def create_session(body: SessionIn):
    env = get_environment(body.env)
    sid = f"sess_{int(time.time() * 1000)}_{len(_SESSIONS)}"
    with _LOCK:
        _gc()
        _SESSIONS[sid] = {"env": env, "task": body.task, "state": env.reset(body.task), "turn": 0, "created": time.time()}
    return {"session_id": sid, "instruction": body.task.instruction, "tools": [t.model_dump() for t in env.tools.specs()],
            "max_steps": body.task.max_steps}


def _get(sid: str) -> dict[str, Any]:
    s = _SESSIONS.get(sid)
    if s is None:
        raise HTTPException(404, "unknown or expired session")
    return s


@router.post("/sessions/{sid}/step")
def step(sid: str, body: StepIn):
    s = _get(sid)
    res = s["env"].tools.call(body.tool_call, state=s["state"])
    return res.model_dump()


@router.post("/sessions/{sid}/user_turn")
def user_turn(sid: str):
    s = _get(sid)
    msg = s["env"].next_user_turn(s["task"], s["turn"])
    if msg is not None:
        s["turn"] += 1
    return {"message": msg}


@router.post("/sessions/{sid}/finish")
def finish(sid: str, body: dict[str, Any]):
    s = _get(sid)
    ro = Rollout.model_validate(body["rollout"]) if "rollout" in body else Rollout(task_id=s["task"].id, env_name=s["env"].name)
    rewards = s["env"].verify(s["task"], ro, s["state"])
    with _LOCK:
        _SESSIONS.pop(sid, None)
    return {"rewards": [r.model_dump() for r in rewards], "state_log": s["state"].log}


def _gc() -> None:
    cutoff = time.time() - SESSION_TTL_S
    for k in [k for k, v in _SESSIONS.items() if v["created"] < cutoff]:
        _SESSIONS.pop(k, None)


def create_app(api_key: Optional[str] = None):
    """Standalone environment server (`saphire env-server`)."""
    from fastapi import FastAPI

    from ..server.deps import require_api_key

    app = FastAPI(title="Saphire environment server")
    app.include_router(router, prefix="/v1", dependencies=[Depends(require_api_key)])

    @app.get("/health")
    def health():
        return {"status": "ok", "sessions": len(_SESSIONS)}

    return app
