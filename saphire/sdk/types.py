"""Core data types shared by the SDK, environments, signals, training and evaluation.

Everything is a plain Pydantic model so it can be serialised to JSON, stored in the
Saphire server, exported to training datasets, and round-tripped between processes.
"""
from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


def new_id(prefix: str = "") -> str:
    return f"{prefix}{uuid.uuid4().hex[:16]}"


class Role(str, Enum):
    system = "system"
    user = "user"
    assistant = "assistant"
    tool = "tool"


class ToolCall(BaseModel):
    """A tool invocation requested by the policy (LLM)."""

    id: str = Field(default_factory=lambda: new_id("call_"))
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class Message(BaseModel):
    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_call_id: Optional[str] = None  # for role=tool
    name: Optional[str] = None  # tool name for role=tool

    def to_openai(self) -> dict[str, Any]:
        d: dict[str, Any] = {"role": self.role.value, "content": self.content}
        if self.tool_calls:
            d["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": _json(tc.arguments)},
                }
                for tc in self.tool_calls
            ]
        if self.role == Role.tool:
            d["tool_call_id"] = self.tool_call_id
            d["name"] = self.name
        return d


def _json(obj: Any) -> str:
    import json

    return json.dumps(obj, ensure_ascii=False)


class ToolSpec(BaseModel):
    """JSON-schema description of a callable tool (OpenAI function-calling compatible)."""

    name: str
    description: str = ""
    parameters: dict[str, Any] = Field(default_factory=lambda: {"type": "object", "properties": {}})
    tags: list[str] = Field(default_factory=list)
    source: str = "python"  # python | mcp | openapi | sql

    def to_openai(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {"name": self.name, "description": self.description, "parameters": self.parameters},
        }


class ToolResult(BaseModel):
    call_id: str
    name: str
    output: Any = None
    error: Optional[str] = None
    latency_ms: float = 0.0


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class LLMResponse(BaseModel):
    message: Message
    usage: Usage = Field(default_factory=Usage)
    model: str = ""
    latency_ms: float = 0.0
    raw: Optional[dict[str, Any]] = None


class Step(BaseModel):
    """One policy decision inside a rollout: the prompt state, the LLM output, the tool results.

    A Step is the atomic unit that RL algorithms operate on ("transition"): the
    `prompt_messages` are the state, `response` is the action, `reward` the signal.
    """

    index: int
    prompt_messages: list[Message]
    response: Message
    tool_results: list[ToolResult] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    latency_ms: float = 0.0
    exposed_tools: list[str] = Field(default_factory=list)  # tool names visible to the policy at this step
    role: str = "main"  # which agent (in a multi-agent system) produced this step
    reward: Optional[float] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Reward(BaseModel):
    """A scalar training signal with provenance."""

    value: float
    source: str  # e.g. "verifier", "judge:gpt-4o", "rubric", "human", "tool_correctness"
    name: str = "reward"
    step_index: Optional[int] = None  # None => trajectory-level
    rationale: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RolloutStatus(str, Enum):
    running = "running"
    succeeded = "succeeded"
    failed = "failed"
    timeout = "timeout"
    error = "error"


class Rollout(BaseModel):
    """A full trajectory of an agent attempting one task in one environment."""

    id: str = Field(default_factory=lambda: new_id("ro_"))
    task_id: str
    env_name: str
    agent_id: str = ""
    agent_version: str = ""
    steps: list[Step] = Field(default_factory=list)
    final_answer: str = ""
    status: RolloutStatus = RolloutStatus.running
    rewards: list[Reward] = Field(default_factory=list)
    started_at: float = Field(default_factory=time.time)
    ended_at: Optional[float] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    trace_id: Optional[str] = None

    # ----- convenience -----
    @property
    def duration_ms(self) -> float:
        end = self.ended_at or time.time()
        return (end - self.started_at) * 1000.0

    @property
    def total_reward(self) -> float:
        traj = [r.value for r in self.rewards if r.step_index is None]
        if traj:
            return float(sum(traj)) / len(traj)
        step = [r.value for r in self.rewards]
        return float(sum(step)) / len(step) if step else 0.0

    @property
    def tool_calls(self) -> list[ToolCall]:
        return [tc for s in self.steps for tc in s.response.tool_calls]

    @property
    def usage(self) -> Usage:
        u = Usage()
        for s in self.steps:
            u.prompt_tokens += s.usage.prompt_tokens
            u.completion_tokens += s.usage.completion_tokens
        return u

    def messages(self) -> list[Message]:
        """Reconstruct the full conversation from the last step."""
        if not self.steps:
            return []
        last = self.steps[-1]
        msgs = list(last.prompt_messages) + [last.response]
        for tr in last.tool_results:
            msgs.append(Message(role=Role.tool, content=_json(tr.output if tr.error is None else {"error": tr.error}), tool_call_id=tr.call_id, name=tr.name))
        return msgs


class TaskSpec(BaseModel):
    """A unit of work an agent must complete inside an environment.

    `expected` holds ground truth that verifiers use (expected tool calls, final state,
    answer substrings...). `user_script` drives a simulated user for multi-turn tasks.
    """

    id: str = Field(default_factory=lambda: new_id("task_"))
    env_name: str
    instruction: str
    expected: dict[str, Any] = Field(default_factory=dict)
    user_script: list[str] = Field(default_factory=list)  # follow-up user turns revealed over time
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    tags: list[str] = Field(default_factory=list)
    max_steps: int = 12
    metadata: dict[str, Any] = Field(default_factory=dict)


class RoleConfig(BaseModel):
    """One agent inside a multi-agent system (orchestrator or specialist).

    Each role has its own policy surface (prompt, model, router, exemplars) and its own tool subset, so
    it can be optimised independently ("selective optimisation") while the others stay frozen.
    """

    description: str = ""  # shown to the orchestrator as the hand-off tool description
    system_prompt: Optional[str] = None  # None -> inherit the system-level prompt
    model: Optional[str] = None  # None -> inherit
    tool_names: list[str] = Field(default_factory=list)  # explicit subset of environment tools
    tool_tags: list[str] = Field(default_factory=list)  # ... or by tag
    tool_router: Optional[str] = None
    router_top_k: int = 0
    exemplar_store: Optional[str] = None
    exemplar_k: int = 2
    max_steps: int = 6
    trainable: bool = True  # False -> frozen during selective optimisation
    temperature: float = 0.0


class AgentConfig(BaseModel):
    """Everything that defines a deployable agent version (single agent or multi-agent system)."""

    name: str = "agent"
    version: str = "v0"
    model: str = "mock"  # e.g. "openai/gpt-4o-mini", "anthropic/claude-sonnet-4", "hf:./checkpoints/x", "mock"
    system_prompt: str = "You are a helpful assistant. Use tools to complete the user's task, then reply with a final answer."
    temperature: float = 0.0
    max_steps: int = 12
    tool_router: Optional[str] = None  # artifact path of a trained ToolRouter
    router_top_k: int = 8  # how many tools to expose per step (0 = expose all)
    exemplar_store: Optional[str] = None  # artifact path of ExemplarStore
    exemplar_k: int = 2
    # multi-agent: when `roles` is non-empty the agent is an orchestrator that hands off to role agents
    roles: dict[str, RoleConfig] = Field(default_factory=dict)
    orchestrator_prompt: Optional[str] = None
    orchestrator_tools: list[str] = Field(default_factory=list)  # env tools the orchestrator may call directly
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_multi_agent(self) -> bool:
        return bool(self.roles)
