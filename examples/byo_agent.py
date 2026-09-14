"""Bring-your-own agent: instrument any agent loop, record rollouts, and ship them to Saphire.

Run the server first (`saphire serve --inline-jobs`) then `python examples/byo_agent.py`.
Swap `MockLLM()` for `get_llm("openai/gpt-4o-mini")` once OPENAI_API_KEY is set.
"""
from saphire.environments import get_environment
from saphire.sdk import AgentConfig, MockLLM, RolloutRecorder, SaphireClient, ToolCall, tracing
from saphire.sdk.types import Message, Role

tracing.init(service_name="byo-agent")  # spans -> $SAPHIRE_HOST (default http://localhost:8000)
client = SaphireClient()
env = get_environment("support_desk")
llm = MockLLM()
task = env.generate_tasks(1, seed=42)[0]
state = env.reset(task)

agent = client.create_agent(AgentConfig(name="byo-agent", model="mock"))
rec = RolloutRecorder(task.id, env.name, agent_id="byo-agent")
messages = [Message(role=Role.system, content="You are a support agent."), Message(role=Role.user, content=task.instruction)]

with tracing.span("byo-agent", kind="agent", input=task.instruction) as root:
    for _ in range(8):
        resp = llm.complete(messages, tools=env.tools.specs())
        results = []
        messages.append(resp.message)
        for tc in resp.message.tool_calls:
            with tracing.span(tc.name, kind="tool", input=tc.arguments):
                res = env.tools.call(ToolCall(id=tc.id, name=tc.name, arguments=tc.arguments), state=state)
            results.append(res)
            messages.append(Message(role=Role.tool, content=str(res.output if res.error is None else res.error), tool_call_id=tc.id, name=tc.name))
        rec.step(messages[:-1 - len(results)], resp.message, tool_results=results, usage=resp.usage)
        if not resp.message.tool_calls:
            break
    rollout = rec.finish(final_answer=messages[-1].content)
    rollout.trace_id = root.trace_id
    root.set(**{"rollout.id": rollout.id})  # lets the server link trace <-> rollout

rewards = env.verify(task, rollout, state)  # ground-truth signals from the environment
print(client.log_rollout(rollout, task, rewards, agent_id=agent["id"]))
client.score("thumbs_up", 1.0, rollout_id=rollout.id, source="human")  # any extra signal
tracing.flush()
