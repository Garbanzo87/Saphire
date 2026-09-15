"""OpenRLHF / SkyRL-style agent function: run a Saphire environment episode step-by-step for an external trainer.

OpenRLHF's async agent RL (`--agent_func_path this_file.py`) expects an `AgentInstance` with async `reset` and `step`
methods. The trainer owns the policy (vLLM); Saphire owns the environment, tools, simulated user and verifier —
either in-process or over HTTP (`SAPHIRE_ENV_SERVER`) so the trainer image needs only `httpx`.

Episode protocol
  reset(states)  -> {"observation_text": <prompt>}                # system prompt with tools + user instruction
  step(states)   -> {"rewards": r, "scores": r, "environment_feedback": <tool result text>, "done": bool,
                     "sampling_params": ..., "extra_logs": {...}}
"""
from __future__ import annotations

import json
import os
from typing import Any, Optional

from ..sdk.llm import messages_to_text, parse_assistant_text, render_tools_text
from ..sdk.types import Message, Role, Rollout, RolloutStatus, Step, TaskSpec, ToolCall, ToolResult, ToolSpec


class AgentInstance:
    def __init__(self, *args: Any, **kwargs: Any):
        self.server = os.getenv("SAPHIRE_ENV_SERVER")
        self.api_key = os.getenv("SAPHIRE_API_KEY", "dev-key")
        self.env = None
        self.state = None
        self.task: Optional[TaskSpec] = None
        self.tools: list[ToolSpec] = []
        self.messages: list[Message] = []
        self.rollout: Optional[Rollout] = None
        self.session: Optional[str] = None
        self.max_steps = 8

    # ---- transport ----
    def _http(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        import httpx

        r = httpx.post(f"{self.server.rstrip('/')}/v1/env{path}", json=body, headers={"x-api-key": self.api_key}, timeout=60)
        r.raise_for_status()
        return r.json()

    # ---- protocol ----
    async def reset(self, states: dict[str, Any], **kw: Any) -> dict[str, Any]:
        """`states["label"]` carries the task (json) and env name, as emitted by `export_openrlhf_prompts`."""
        label = states.get("label") or states.get("extra_info") or {}
        if isinstance(label, str):
            label = json.loads(label)
        self.task = TaskSpec.model_validate(label["task"])
        env_name = label.get("env") or self.task.env_name
        self.max_steps = self.task.max_steps
        if self.server:
            s = self._http("/sessions", {"env": env_name, "task": self.task.model_dump()})
            self.session = s["session_id"]
            self.tools = [ToolSpec.model_validate(t) for t in s["tools"]]
        else:
            from ..environments.base import get_environment

            self.env = get_environment(env_name)
            self.state = self.env.reset(self.task)
            self.tools = self.env.tools.specs()
        self.rollout = Rollout(task_id=self.task.id, env_name=env_name, agent_id="openrlhf")
        self.messages = [Message(role=Role.system, content="You are a helpful assistant. Use tools, then answer.\n\n" + render_tools_text(self.tools)),
                         Message(role=Role.user, content=self.task.instruction)]
        return {"observation_text": messages_to_text(self.messages)}

    async def step(self, states: dict[str, Any], **kw: Any) -> dict[str, Any]:
        action = states.get("action_text") or states.get("action") or ""
        msg = parse_assistant_text(action)
        step = Step(index=len(self.rollout.steps), prompt_messages=list(self.messages), response=msg, exposed_tools=[t.name for t in self.tools])
        self.messages.append(msg)
        feedback = ""
        done = False
        if msg.tool_calls:
            tc: ToolCall = msg.tool_calls[0]
            if self.server:
                res = ToolResult.model_validate(self._http(f"/sessions/{self.session}/step", {"tool_call": tc.model_dump()}))
            else:
                res = self.env.tools.call(tc, state=self.state)
            step.tool_results.append(res)
            feedback = json.dumps(res.output if res.error is None else {"error": res.error}, default=str)
            self.messages.append(Message(role=Role.tool, tool_call_id=tc.id, name=tc.name, content=feedback))
        else:
            nxt = self._http(f"/sessions/{self.session}/user_turn", {})["message"] if self.server else self.env.next_user_turn(self.task, self.rollout.metadata.get("user_turns", 0))
            if nxt:
                self.rollout.metadata["user_turns"] = self.rollout.metadata.get("user_turns", 0) + 1
                self.messages.append(Message(role=Role.user, content=nxt))
                feedback = nxt
            else:
                done = True
        self.rollout.steps.append(step)
        if len(self.rollout.steps) >= self.max_steps:
            done = True
        reward = 0.0
        if done:
            self.rollout.final_answer = msg.content
            self.rollout.status = RolloutStatus.succeeded if not msg.tool_calls else RolloutStatus.timeout
            if self.server:
                out = self._http(f"/sessions/{self.session}/finish", {"rollout": self.rollout.model_dump(mode="json")})
                rewards = out["rewards"]
            else:
                rewards = [r.model_dump() for r in self.env.verify(self.task, self.rollout, self.state)]
            traj = [r["value"] for r in rewards if r.get("step_index") is None]
            reward = sum(traj) / len(traj) if traj else 0.0
        return {"rewards": reward, "scores": reward, "environment_feedback": feedback, "done": done,
                "sampling_params": states.get("sampling_params"), "extra_logs": {"steps": len(self.rollout.steps)},
                "observation_text": messages_to_text(self.messages)}


def export_openrlhf_prompts(tasks: list[TaskSpec], path: str) -> str:
    """JSONL with `prompt` + `label` (task json) as consumed by OpenRLHF's agent dataset."""
    with open(path, "w") as f:
        for t in tasks:
            f.write(json.dumps({"prompt": t.instruction, "label": json.dumps({"task": t.model_dump(), "env": t.env_name})}) + "\n")
    return path
