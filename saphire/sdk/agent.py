"""Generic tool-using agent loop with tracing, routing, exemplars and simulated users.

`ToolAgent.run(task, env)` produces a fully-populated `Rollout` — the unit consumed by
signal generation, training and evaluation. Bring-your-own agents (LangGraph, OpenAI Agents
SDK, custom loops) can instead build Rollouts through `RolloutRecorder`.
"""
from __future__ import annotations

import json
import time
from typing import Any, Optional

from . import tracing
from .exemplars import ExemplarStore
from .llm import LLM, get_llm
from .router import ToolRouter
from .types import AgentConfig, Message, Role, Rollout, RolloutStatus, Step, TaskSpec, ToolResult, Usage

if False:  # pragma: no cover  (typing only)
    from ..environments.base import Environment


class ToolAgent:
    def __init__(self, config: AgentConfig, llm: Optional[LLM] = None, router: Optional[ToolRouter] = None,
                 exemplars: Optional[ExemplarStore] = None):
        self.config = config
        self.llm = llm or get_llm(config.model)
        self.router = router if router is not None else (ToolRouter.load(config.tool_router) if config.tool_router else None)
        self.exemplars = exemplars if exemplars is not None else (
            ExemplarStore.load(config.exemplar_store) if config.exemplar_store else None)

    # -------------------------------------------------------------
    def system_prompt(self, task: TaskSpec) -> str:
        sp = self.config.system_prompt
        if self.exemplars is not None and self.config.exemplar_k > 0:
            ex = self.exemplars.render(task.instruction, self.config.exemplar_k)
            if ex:
                sp = sp + "\n\n" + ex
        return sp

    def exposed_tools(self, env: "Environment", query: str, exclude: list[str] | None = None):
        specs = self.config.apply_overrides(env.tools.specs())
        k = self.config.router_top_k
        if self.router is not None and k and k < len(specs):
            return self.router.rank(query, specs, top_k=k, exclude=exclude or [])
        if k and k < len(specs) and self.router is None:
            # untrained lexical-only routing
            return ToolRouter([s.name for s in specs], alpha=0.0).rank(query, specs, top_k=k, exclude=exclude or [])
        return specs

    # -------------------------------------------------------------
    def run(self, task: TaskSpec, env: "Environment", state=None) -> Rollout:
        state = state if state is not None else env.reset(task)
        rollout = Rollout(task_id=task.id, env_name=env.name, agent_id=self.config.name, agent_version=self.config.version)
        messages: list[Message] = [Message(role=Role.system, content=self.system_prompt(task)),
                                   Message(role=Role.user, content=task.instruction)]
        max_steps = min(self.config.max_steps, task.max_steps) if task.max_steps else self.config.max_steps
        user_turn = 0
        with tracing.span(f"agent:{self.config.name}", kind="agent", input=task.instruction,
                          **{"agent.version": self.config.version, "task.id": task.id, "env": env.name}) as root:
            rollout.trace_id = root.trace_id
            try:
                for i in range(max_steps):
                    query = task.instruction + " " + (messages[-1].content if messages[-1].role == Role.user else "")
                    exposed = self.exposed_tools(env, query)
                    t0 = time.perf_counter()
                    with tracing.span("llm", kind="llm", input=[m.to_openai() for m in messages[-4:]],
                                      **{"llm.model": getattr(self.llm, "model_name", self.config.model),
                                         "llm.exposed_tools": [t.name for t in exposed]}) as ls:
                        resp = self.llm.complete(messages, tools=exposed, temperature=self.config.temperature)
                        ls.set(**{"llm.token_count.prompt": resp.usage.prompt_tokens,
                                  "llm.token_count.completion": resp.usage.completion_tokens})
                        ls.output(resp.message.to_openai())
                    step = Step(index=i, prompt_messages=list(messages), response=resp.message, usage=resp.usage,
                                latency_ms=(time.perf_counter() - t0) * 1000, exposed_tools=[t.name for t in exposed])
                    messages.append(resp.message)
                    if resp.message.tool_calls:
                        for tc in resp.message.tool_calls:
                            with tracing.span(tc.name, kind="tool", input=tc.arguments, **{"tool.name": tc.name}) as ts:
                                res: ToolResult = env.tools.call(tc, state=state)
                                ts.set(**{"tool.latency_ms": res.latency_ms, "tool.error": res.error or ""})
                                ts.output(res.output if res.error is None else {"error": res.error})
                            step.tool_results.append(res)
                            messages.append(Message(role=Role.tool, tool_call_id=tc.id, name=tc.name,
                                                    content=json.dumps(res.output if res.error is None else {"error": res.error}, default=str)))
                        rollout.steps.append(step)
                        continue
                    # final (text) answer: does the simulated user have another turn?
                    rollout.steps.append(step)
                    follow = env.next_user_turn(task, user_turn)
                    if follow is not None:
                        user_turn += 1
                        messages.append(Message(role=Role.user, content=follow))
                        continue
                    rollout.final_answer = resp.message.content
                    rollout.status = RolloutStatus.succeeded
                    break
                else:
                    rollout.status = RolloutStatus.timeout
                    rollout.final_answer = messages[-1].content if messages[-1].role == Role.assistant else ""
            except Exception as e:  # provider/tool infrastructure failure
                rollout.status = RolloutStatus.error
                rollout.metadata["error"] = f"{type(e).__name__}: {e}"
            rollout.ended_at = time.time()
            rollout.metadata["state_log"] = state.log
            rollout.metadata["user_turns"] = user_turn
            root.set(**{"rollout.id": rollout.id, "rollout.status": rollout.status.value,
                        "rollout.steps": len(rollout.steps)})
            root.output(rollout.final_answer)
        return rollout


class RolloutRecorder:
    """Build a Rollout by hand from a custom agent loop (for bring-your-own-agent integrations).

        rec = RolloutRecorder(task_id, env_name, agent_id="my-agent")
        rec.step(prompt_messages, response_message, tool_results=[...], usage=Usage(...))
        rollout = rec.finish(final_answer="...")
    """

    def __init__(self, task_id: str, env_name: str, agent_id: str = "", agent_version: str = ""):
        self.rollout = Rollout(task_id=task_id, env_name=env_name, agent_id=agent_id, agent_version=agent_version)

    def step(self, prompt_messages: list[Message], response: Message, tool_results: list[ToolResult] | None = None,
             usage: Usage | None = None, latency_ms: float = 0.0, exposed_tools: list[str] | None = None) -> Step:
        s = Step(index=len(self.rollout.steps), prompt_messages=prompt_messages, response=response,
                 tool_results=tool_results or [], usage=usage or Usage(), latency_ms=latency_ms,
                 exposed_tools=exposed_tools or [])
        self.rollout.steps.append(s)
        return s

    def finish(self, final_answer: str = "", status: RolloutStatus = RolloutStatus.succeeded, **metadata: Any) -> Rollout:
        self.rollout.final_answer = final_answer
        self.rollout.status = status
        self.rollout.ended_at = time.time()
        self.rollout.metadata.update(metadata)
        return self.rollout
