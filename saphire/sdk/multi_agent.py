"""Multi-agent systems with selective optimisation.

An `AgentSystem` is an orchestrator agent plus named specialist *roles*. The orchestrator sees one
hand-off tool per role (`delegate_to_<role>(task, context)`) plus any environment tools it is
allowed to call directly; each specialist sees only its own subset of environment tools and runs
its own short loop on the delegated task, sharing the environment state.

Every `Step` in the resulting `Rollout` is tagged with the role that produced it, so:

* rewards can be attributed per role (`role_rewards`), and
* learners (router, exemplars, prompt optimiser, SFT/DPO/GRPO datasets) can be trained on the
  steps of *selected* roles while the other roles stay frozen — "selective optimisation".

`AgentSystem` has the same `run(task, env, state) -> Rollout` interface as `ToolAgent`, so
evaluation, the online loop and the server work unchanged (`build_agent(config)` picks the class).
"""
from __future__ import annotations

import json
import time
from typing import Any, Optional

from . import tracing
from .exemplars import ExemplarStore
from .llm import LLM, get_llm
from .router import ToolRouter
from .types import AgentConfig, Message, Role, RoleConfig, Rollout, RolloutStatus, Step, TaskSpec, ToolCall, ToolResult, ToolSpec

if False:  # pragma: no cover
    from ..environments.base import Environment

ORCHESTRATOR = "orchestrator"


def handoff_spec(role: str, cfg: RoleConfig) -> ToolSpec:
    return ToolSpec(
        name=f"delegate_to_{role}",
        description=f"Delegate to the {role} specialist: {cfg.description}",
        parameters={"type": "object",
                    "properties": {"task": {"type": "string", "description": "What the specialist should do, in full"},
                                   "context": {"type": "string", "description": "Identifiers and facts the specialist needs (order ids, emails...)"}},
                    "required": ["task"]},
        tags=["handoff"], source="handoff")


class RolePolicy:
    """Per-role policy surface: llm + router + exemplars + prompt, resolved from RoleConfig with system defaults."""

    def __init__(self, name: str, cfg: RoleConfig, system: AgentConfig, llm: Optional[LLM] = None,
                 router: Optional[ToolRouter] = None, exemplars: Optional[ExemplarStore] = None):
        self.name = name
        self.cfg = cfg
        self.model = cfg.model or system.model
        self.llm = llm or get_llm(self.model)
        self.system_prompt = cfg.system_prompt or system.system_prompt
        self.router = router if router is not None else (ToolRouter.load(cfg.tool_router) if cfg.tool_router else None)
        self.exemplars = exemplars if exemplars is not None else (ExemplarStore.load(cfg.exemplar_store) if cfg.exemplar_store else None)
        self.top_k = cfg.router_top_k
        self.exemplar_k = cfg.exemplar_k
        self.max_steps = cfg.max_steps
        self.temperature = cfg.temperature

    def tools(self, env: "Environment") -> list[ToolSpec]:
        if self.cfg.tool_names:
            return env.tools.specs(self.cfg.tool_names)
        if self.cfg.tool_tags:
            tags = set(self.cfg.tool_tags)
            return [t for t in env.tools.specs() if tags & set(t.tags)]
        return env.tools.specs()

    def exposed(self, specs: list[ToolSpec], query: str) -> list[ToolSpec]:
        if self.top_k and self.top_k < len(specs):
            r = self.router or ToolRouter([s.name for s in specs], alpha=0.0)
            return r.rank(query, specs, top_k=self.top_k)
        return specs

    def prompt(self, task_text: str) -> str:
        sp = self.system_prompt
        if self.exemplars is not None and self.exemplar_k > 0:
            ex = self.exemplars.render(task_text, self.exemplar_k)
            if ex:
                sp += "\n\n" + ex
        return sp


class AgentSystem:
    def __init__(self, config: AgentConfig, llm: Optional[LLM] = None, policies: Optional[dict[str, RolePolicy]] = None):
        if not config.roles:
            raise ValueError("AgentConfig.roles is empty; use ToolAgent for single agents")
        self.config = config
        orch_cfg = RoleConfig(system_prompt=config.orchestrator_prompt or config.system_prompt, tool_names=config.orchestrator_tools,
                              tool_router=config.tool_router, router_top_k=config.router_top_k, exemplar_store=config.exemplar_store,
                              exemplar_k=config.exemplar_k, max_steps=config.max_steps)
        self.policies: dict[str, RolePolicy] = dict(policies or {})
        self.policies.setdefault(ORCHESTRATOR, RolePolicy(ORCHESTRATOR, orch_cfg, config, llm=llm))
        for name, rc in config.roles.items():
            self.policies.setdefault(name, RolePolicy(name, rc, config, llm=llm if rc.model is None else None))
        self.llm = self.policies[ORCHESTRATOR].llm

    @property
    def roles(self) -> list[str]:
        return list(self.config.roles)

    # ------------------------------------------------------------------
    def _llm_step(self, pol: RolePolicy, messages: list[Message], exposed: list[ToolSpec], rollout: Rollout, role: str) -> Step:
        t0 = time.perf_counter()
        with tracing.span(f"llm:{role}", kind="llm", input=[m.to_openai() for m in messages[-3:]],
                          **{"llm.model": getattr(pol.llm, "model_name", pol.model), "agent.role": role,
                             "llm.exposed_tools": [t.name for t in exposed]}) as ls:
            resp = pol.llm.complete(messages, tools=exposed, temperature=pol.temperature)
            ls.set(**{"llm.token_count.prompt": resp.usage.prompt_tokens, "llm.token_count.completion": resp.usage.completion_tokens})
            ls.output(resp.message.to_openai())
        step = Step(index=len(rollout.steps), prompt_messages=list(messages), response=resp.message, usage=resp.usage,
                    latency_ms=(time.perf_counter() - t0) * 1000, exposed_tools=[t.name for t in exposed], role=role)
        rollout.steps.append(step)
        return step

    def _run_role(self, role: str, task_text: str, context: str, env: "Environment", state, rollout: Rollout) -> str:
        pol = self.policies[role]
        specs = pol.tools(env)
        user = task_text + (f"\n\nContext: {context}" if context else "")
        messages = [Message(role=Role.system, content=pol.prompt(task_text)), Message(role=Role.user, content=user)]
        with tracing.span(f"agent:{role}", kind="agent", input=user, **{"agent.role": role}) as sp:
            for _ in range(pol.max_steps):
                exposed = pol.exposed(specs, task_text)
                step = self._llm_step(pol, messages, exposed, rollout, role)
                messages.append(step.response)
                if not step.response.tool_calls:
                    sp.output(step.response.content)
                    return step.response.content
                for tc in step.response.tool_calls:
                    res = self._call_tool(env, tc, state, allowed={s.name for s in specs})
                    step.tool_results.append(res)
                    messages.append(Message(role=Role.tool, tool_call_id=tc.id, name=tc.name,
                                            content=json.dumps(res.output if res.error is None else {"error": res.error}, default=str)))
            out = "Specialist reached its step limit. " + "; ".join(m.content[:200] for m in messages if m.role == Role.tool)[-800:]
            sp.output(out)
            return out

    @staticmethod
    def _call_tool(env: "Environment", tc: ToolCall, state, allowed: set[str]) -> ToolResult:
        with tracing.span(tc.name, kind="tool", input=tc.arguments, **{"tool.name": tc.name}) as ts:
            if tc.name not in allowed:
                res = ToolResult(call_id=tc.id, name=tc.name, error=f"tool '{tc.name}' is not available to this agent")
            else:
                res = env.tools.call(tc, state=state)
            ts.set(**{"tool.latency_ms": res.latency_ms, "tool.error": res.error or ""})
            ts.output(res.output if res.error is None else {"error": res.error})
        return res

    # ------------------------------------------------------------------
    def run(self, task: TaskSpec, env: "Environment", state=None) -> Rollout:
        state = state if state is not None else env.reset(task)
        rollout = Rollout(task_id=task.id, env_name=env.name, agent_id=self.config.name, agent_version=self.config.version,
                          metadata={"multi_agent": True, "roles": self.roles})
        orch = self.policies[ORCHESTRATOR]
        handoffs = [handoff_spec(r, c) for r, c in self.config.roles.items()]
        direct = orch.tools(env) if self.config.orchestrator_tools else []
        specs = handoffs + direct
        messages = [Message(role=Role.system, content=orch.prompt(task.instruction)), Message(role=Role.user, content=task.instruction)]
        max_steps = min(orch.max_steps, task.max_steps) if task.max_steps else orch.max_steps
        user_turn = 0
        with tracing.span(f"agent:{self.config.name}", kind="agent", input=task.instruction,
                          **{"agent.version": self.config.version, "task.id": task.id, "env": env.name, "agent.role": ORCHESTRATOR}) as root:
            rollout.trace_id = root.trace_id
            try:
                for _ in range(max_steps):
                    query = task.instruction + " " + (messages[-1].content if messages[-1].role == Role.user else "")
                    exposed = orch.exposed(specs, query)
                    step = self._llm_step(orch, messages, exposed, rollout, ORCHESTRATOR)
                    messages.append(step.response)
                    if step.response.tool_calls:
                        for tc in step.response.tool_calls:
                            if tc.name.startswith("delegate_to_") and tc.name[len("delegate_to_"):] in self.config.roles:
                                role = tc.name[len("delegate_to_"):]
                                t0 = time.perf_counter()
                                out = self._run_role(role, str(tc.arguments.get("task", "")), str(tc.arguments.get("context", "") or ""),
                                                     env, state, rollout)
                                res = ToolResult(call_id=tc.id, name=tc.name, output={"specialist": role, "result": out},
                                                 latency_ms=(time.perf_counter() - t0) * 1000)
                            else:
                                res = self._call_tool(env, tc, state, allowed={s.name for s in direct})
                            step.tool_results.append(res)
                            messages.append(Message(role=Role.tool, tool_call_id=tc.id, name=tc.name,
                                                    content=json.dumps(res.output if res.error is None else {"error": res.error}, default=str)))
                        continue
                    follow = env.next_user_turn(task, user_turn)
                    if follow is not None:
                        user_turn += 1
                        messages.append(Message(role=Role.user, content=follow))
                        continue
                    rollout.final_answer = step.response.content
                    rollout.status = RolloutStatus.succeeded
                    break
                else:
                    rollout.status = RolloutStatus.timeout
            except Exception as e:  # noqa: BLE001
                rollout.status = RolloutStatus.error
                rollout.metadata["error"] = f"{type(e).__name__}: {e}"
            rollout.ended_at = time.time()
            rollout.metadata["state_log"] = state.log
            rollout.metadata["user_turns"] = user_turn
            rollout.metadata["steps_by_role"] = {r: sum(1 for s in rollout.steps if s.role == r) for r in [ORCHESTRATOR] + self.roles}
            root.set(**{"rollout.id": rollout.id, "rollout.status": rollout.status.value, "rollout.steps": len(rollout.steps)})
            root.output(rollout.final_answer)
        return rollout


# ----------------------------------------------------------------------
def role_rewards(rollout: Rollout) -> dict[str, dict[str, float]]:
    """Credit assignment per role: trajectory reward shared, plus the mean of that role's step rewards."""
    out: dict[str, dict[str, float]] = {}
    step_role = {s.index: s.role for s in rollout.steps}
    per_role_steps: dict[str, list[float]] = {}
    for r in rollout.rewards:
        if r.step_index is not None and r.step_index in step_role:
            per_role_steps.setdefault(step_role[r.step_index], []).append(r.value)
    roles = {s.role for s in rollout.steps}
    for role in roles:
        vals = per_role_steps.get(role, [])
        out[role] = {"trajectory": rollout.total_reward, "step_mean": (sum(vals) / len(vals)) if vals else 0.0,
                     "n_steps": float(sum(1 for s in rollout.steps if s.role == role)),
                     "n_tool_calls": float(sum(len(s.response.tool_calls) for s in rollout.steps if s.role == role))}
    return out


def build_agent(config: AgentConfig, llm: Optional[LLM] = None, **kw: Any):
    """Factory: ToolAgent for single-agent configs, AgentSystem when roles are defined."""
    if config.is_multi_agent:
        return AgentSystem(config, llm=llm, policies=kw.get("policies"))
    from .agent import ToolAgent

    return ToolAgent(config, llm=llm, router=kw.get("router"), exemplars=kw.get("exemplars"))
