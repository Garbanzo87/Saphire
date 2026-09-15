"""Gymnasium-style adapter: a Saphire environment as `reset()/step()` for RL frameworks that speak gym (SkyRL-gym,
AgentGym, custom PPO loops). Observations are text (the prompt the policy should continue); actions are text
(an assistant message in Saphire's text tool-call format); rewards are the verifier's trajectory reward at episode end
(plus optional dense per-step tool-correctness shaping)."""
from __future__ import annotations

import json
from typing import Any, Optional

from ..environments.base import Environment, get_environment
from ..sdk.llm import messages_to_text, parse_assistant_text
from ..sdk.types import Message, Role, Rollout, RolloutStatus, Step, TaskSpec


class SaphireGymEnv:
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, env: Environment | str, dense_reward: bool = True, system_prompt: str = "You are a helpful assistant. Use tools, then answer."):
        self.env = get_environment(env) if isinstance(env, str) else env
        self.dense = dense_reward
        self.system_prompt = system_prompt
        self.task: Optional[TaskSpec] = None
        self.state = None
        self.rollout: Optional[Rollout] = None
        self.messages: list[Message] = []
        self.user_turn = 0

    # gym API ---------------------------------------------------------------
    def reset(self, task: Optional[TaskSpec] = None, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None) -> tuple[str, dict[str, Any]]:
        self.task = task or self.env.generate_tasks(1, seed=seed or 0)[0]
        self.state = self.env.reset(self.task)
        self.rollout = Rollout(task_id=self.task.id, env_name=self.env.name, agent_id="gym")
        self.messages = [Message(role=Role.system, content=self.system_prompt), Message(role=Role.user, content=self.task.instruction)]
        self.user_turn = 0
        return self._obs(), {"task": self.task.model_dump(), "tools": [t.model_dump() for t in self.env.tools.specs()]}

    def step(self, action: str) -> tuple[str, float, bool, bool, dict[str, Any]]:
        assert self.rollout is not None and self.task is not None, "call reset() first"
        msg = parse_assistant_text(action)
        step = Step(index=len(self.rollout.steps), prompt_messages=list(self.messages), response=msg,
                    exposed_tools=[t.name for t in self.env.tools.specs()])
        self.messages.append(msg)
        reward, terminated, truncated = 0.0, False, False
        expected = set(self.task.expected.get("tools", []))
        if msg.tool_calls:
            for tc in msg.tool_calls:
                res = self.env.tools.call(tc, state=self.state)
                step.tool_results.append(res)
                self.messages.append(Message(role=Role.tool, tool_call_id=tc.id, name=tc.name,
                                             content=json.dumps(res.output if res.error is None else {"error": res.error}, default=str)))
                if self.dense and expected:
                    reward += (0.1 if tc.name in expected else -0.1) if res.error is None else -0.05
        else:
            nxt = self.env.next_user_turn(self.task, self.user_turn)
            if nxt is not None:
                self.user_turn += 1
                self.messages.append(Message(role=Role.user, content=nxt))
            else:
                terminated = True
        self.rollout.steps.append(step)
        if len(self.rollout.steps) >= self.task.max_steps and not terminated:
            truncated = True
        info: dict[str, Any] = {"step": len(self.rollout.steps)}
        if terminated or truncated:
            self.rollout.final_answer = msg.content if not msg.tool_calls else ""
            self.rollout.status = RolloutStatus.succeeded if terminated else RolloutStatus.timeout
            rewards = self.env.verify(self.task, self.rollout, self.state)
            self.rollout.rewards = rewards
            reward += self.rollout.total_reward
            info["rewards"] = {r.name: r.value for r in rewards if r.step_index is None}
            info["rollout"] = self.rollout
        return self._obs(), reward, terminated, truncated, info

    def render(self) -> str:
        return messages_to_text(self.messages, self.env.tools.specs())

    def _obs(self) -> str:
        return messages_to_text(self.messages, self.env.tools.specs())

    @property
    def action_space_description(self) -> str:
        return 'free text; tool calls as <tool_call>{"name": ..., "arguments": {...}}</tool_call>'
