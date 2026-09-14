import json

from saphire.sdk import (
    AgentConfig,
    ExemplarStore,
    Message,
    MockLLM,
    Role,
    RolloutRecorder,
    ToolAgent,
    ToolCall,
    ToolRegistry,
    ToolRouter,
    ToolSpec,
    tracing,
)
from saphire.sdk.llm import get_llm, messages_to_text, parse_assistant_text
from saphire.sdk.tracing import InMemorySpanExporter


def test_tool_registry_schema_and_call():
    reg = ToolRegistry()

    @reg.register
    def add(a: int, b: int = 1) -> int:
        """Add two numbers.

        Args:
            a: first
            b: second
        """
        return a + b

    spec = reg.specs()[0]
    assert spec.name == "add" and spec.parameters["required"] == ["a"]
    assert spec.parameters["properties"]["a"] == {"type": "integer", "description": "first"}
    assert reg.call(ToolCall(name="add", arguments={"a": 2})).output == 3
    assert reg.call(ToolCall(name="add", arguments={"x": 2})).error.startswith("invalid arguments")
    assert reg.call(ToolCall(name="nope")).error.startswith("unknown tool")


def test_mock_llm_picks_tool_and_fills_args():
    llm = MockLLM()
    tools = [ToolSpec(name="lookup_order", description="Look up an order", parameters={"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]}),
             ToolSpec(name="send_email", description="Send an email", parameters={"type": "object", "properties": {"email": {"type": "string"}, "subject": {"type": "string"}, "body": {"type": "string"}}, "required": ["email"]})]
    msgs = [Message(role=Role.system, content="sys"), Message(role=Role.user, content="Look up order ORD-2001, then send an email to a@b.com")]
    r = llm.complete(msgs, tools)
    assert r.message.tool_calls[0].name == "lookup_order"
    assert r.message.tool_calls[0].arguments == {"order_id": "ORD-2001"}
    msgs += [r.message, Message(role=Role.tool, content=json.dumps({"order_id": "ORD-2001", "status": "shipped"}), tool_call_id=r.message.tool_calls[0].id, name="lookup_order")]
    r2 = llm.complete(msgs, tools)
    assert r2.message.tool_calls[0].name == "send_email"
    assert r2.message.tool_calls[0].arguments["email"] == "a@b.com"
    msgs += [r2.message, Message(role=Role.tool, content="{}", tool_call_id=r2.message.tool_calls[0].id, name="send_email")]
    r3 = llm.complete(msgs, tools)
    assert not r3.message.tool_calls and "shipped" in r3.message.content


def test_mock_llm_honours_rules_and_exemplars():
    tools = [ToolSpec(name="track_return", description="Track the status of a product return shipment"),
             ToolSpec(name="track_shipment", description="Get delivery estimate of the shipment for an order")]
    user = Message(role=Role.user, content="Track the shipment for me")
    plain = MockLLM().complete([Message(role=Role.system, content="x"), user], tools)
    assert plain.message.tool_calls[0].name == "track_return"  # lexical confusion
    ruled = MockLLM().complete([Message(role=Role.system, content='Rule: when the request mentions "shipment", use track_shipment.'), user], tools)
    assert ruled.message.tool_calls[0].name == "track_shipment"
    demo = MockLLM().complete([Message(role=Role.system, content="Examples:\n- Task: track\n  Tools: track_shipment({})"), user], tools)
    assert demo.message.tool_calls[0].name == "track_shipment"


def test_get_llm_parses_mock_options():
    llm = get_llm("mock:error=0.3,ctx=4,seed=7")
    assert isinstance(llm, MockLLM) and llm.error_rate == 0.3 and llm.context_window == 4


def test_text_format_roundtrip():
    m = Message(role=Role.assistant, tool_calls=[ToolCall(name="f", arguments={"a": 1})])
    txt = messages_to_text([Message(role=Role.system, content="s"), Message(role=Role.user, content="u"), m], [ToolSpec(name="f", description="d")])
    assert "<tool_call>" in txt and "- f: d" in txt
    parsed = parse_assistant_text('<tool_call>{"name": "f", "arguments": {"a": 1}}</tool_call>')
    assert parsed.tool_calls[0].name == "f" and parsed.tool_calls[0].arguments == {"a": 1}
    assert parse_assistant_text("hello").content == "hello"


def test_router_learns(tmp_path):
    specs = [ToolSpec(name="a_tool", description="alpha things"), ToolSpec(name="b_tool", description="beta things"),
             ToolSpec(name="c_tool", description="gamma things")]
    r = ToolRouter([s.name for s in specs], alpha=1.0)
    ex = [("please do the alpha thing", "a_tool", 1.0), ("beta please now", "b_tool", 1.0), ("gamma time", "c_tool", 1.0)] * 5
    r.fit(ex, epochs=40)
    assert r.accuracy(ex, specs, top_k=1) == 1.0
    assert r.rank("beta please now", specs, top_k=1)[0].name == "b_tool"
    p = r.save(tmp_path / "router")
    r2 = ToolRouter.load(p)
    assert r2.rank("gamma time", specs, top_k=1)[0].name == "c_tool"


def test_exemplar_store(tmp_path):
    from saphire.environments import get_environment

    env = get_environment("support_desk")
    task = env.generate_tasks(2, seed=1)[1]
    agent = ToolAgent(AgentConfig(router_top_k=0), llm=MockLLM())
    st = env.reset(task)
    ro = agent.run(task, env, st)
    ro.rewards = env.verify(task, ro, st)
    store = ExemplarStore()
    assert store.add_rollout(ro) == (ro.total_reward >= 0.99)
    if store.items:
        assert "Tools:" in store.render(task.instruction)
    store.save(tmp_path / "ex.json")
    assert len(ExemplarStore.load(tmp_path / "ex.json").items) == len(store.items)


def test_tracing_exports_spans():
    exp = InMemorySpanExporter()
    tracing.init(exporter=exp, batch=False, force=True)
    with tracing.span("outer", kind="agent", input={"q": 1}) as s:
        with tracing.span("inner", kind="tool", **{"tool.name": "x"}):
            pass
        s.output("done")
    tracing.flush()
    names = {sp["name"]: sp for sp in exp.spans}
    assert names["inner"]["parent_span_id"] == names["outer"]["span_id"]
    assert names["outer"]["attributes"]["output.value"] == "done"
    assert names["inner"]["kind"] == "tool"
    tracing.shutdown()


def test_rollout_recorder():
    rec = RolloutRecorder("t1", "custom", agent_id="byo")
    rec.step([Message(role=Role.user, content="hi")], Message(role=Role.assistant, content="hello"))
    ro = rec.finish(final_answer="hello")
    assert ro.status.value == "succeeded" and len(ro.steps) == 1 and ro.messages()[-1].content == "hello"
