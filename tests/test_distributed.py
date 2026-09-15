import asyncio
import json
import os

import pytest

from saphire.distributed import openrlhf, verl
from saphire.distributed.rollouts import RolloutEngine
from saphire.environments import get_environment
from saphire.evaluation.runner import evaluate
from saphire.evaluation.suites import build_suite
from saphire.sdk import AgentConfig, MockLLM, ToolAgent
from saphire.sdk.types import ToolCall
from saphire.signals.generate import grpo_prompts


def test_thread_engine_matches_local():
    tasks = build_suite("smoke", n_per_env=6, seed=2)
    cfg = AgentConfig(model="mock", router_top_k=0)
    eng = RolloutEngine(cfg, backend="thread", workers=3)
    res = eng.evaluate(tasks, k=2)
    local = evaluate(ToolAgent(cfg, llm=MockLLM()), tasks, k=2)
    assert res.n_rollouts == local.n_rollouts == 12
    assert res.metrics["task_success"] == pytest.approx(local.metrics["task_success"])
    assert "pass_hat_2" in res.metrics and res.metrics["workers"] == 3


@pytest.mark.skipif(os.getenv("CI") == "true", reason="process pool needs a spawn-safe entrypoint in CI runners")
def test_process_engine():
    tasks = build_suite("smoke", n_per_env=4, seed=2)
    eng = RolloutEngine(AgentConfig(model="mock"), backend="process", workers=2)
    rollouts, records, _, wall = eng.collect(tasks)
    assert len(rollouts) == len(records) == len(tasks) and wall > 0
    assert all(ro.rewards for ro in rollouts)


def test_ray_engine():
    pytest.importorskip("ray")
    tasks = build_suite("smoke", n_per_env=4, seed=2)
    eng = RolloutEngine(AgentConfig(model="mock"), backend="ray", workers=2)
    try:
        res = eng.evaluate(tasks)
        assert res.n_rollouts == len(tasks) and res.metrics["task_success"] > 0.5
    finally:
        eng.shutdown()


def test_verl_export_and_compute_score(tmp_path):
    env = get_environment("support_desk")
    tasks = build_suite("smoke", n_per_env=4, seed=3)
    ros = []
    evaluate(ToolAgent(AgentConfig(model="mock"), llm=MockLLM()), tasks, envs={"support_desk": env}, on_rollout=lambda ro, t, r: ros.append(ro))
    rows = grpo_prompts(ros, {t.id: t for t in tasks}, {"support_desk": env.tools.specs()})
    files = verl.export_verl_dataset(rows, tmp_path / "verl", {t.id: t for t in tasks})
    assert files["train"].endswith(".parquet") and int(files["n_train"]) > 0
    import pyarrow.parquet as pq

    table = pq.read_table(files["train"]).to_pylist()
    row = next(r for r in table if json.loads(json.loads(r["reward_model"])["ground_truth"])["prefix_calls"] == [])
    gt = json.loads(row["reward_model"])["ground_truth"]
    task_d = json.loads(gt)["task"]
    good = '<tool_call>{"name": "%s", "arguments": {"order_id": "%s"}}</tool_call>' % (task_d["expected"]["tools"][0], task_d["instruction"].split()[3])
    assert verl.compute_score(row["data_source"], good, gt, json.loads(row["extra_info"])) > 0
    assert verl.compute_score(row["data_source"], "nonsense", gt) < 0
    msgs = json.loads(row["prompt"])
    assert msgs[0]["role"] == "system" and any(m["role"] == "user" for m in msgs)


def test_env_server_endpoints(api):
    env = get_environment("support_desk")
    task = env.generate_tasks(3, seed=1)[0]  # lookup family
    r = api.post("/v1/env/reward", json={"env": "support_desk", "task": task.model_dump(), "prefix_calls": [],
                                         "completion": '<tool_call>{"name": "lookup_order", "arguments": {"order_id": "%s"}}</tool_call>' % task.instruction.split()[3]})
    assert r.json()["reward"] == 1.0
    s = api.post("/v1/env/sessions", json={"env": "support_desk", "task": task.model_dump()}).json()
    assert len(s["tools"]) == 30
    step = api.post(f"/v1/env/sessions/{s['session_id']}/step", json={"tool_call": ToolCall(name="lookup_order", arguments={"order_id": task.instruction.split()[3]}).model_dump()}).json()
    assert step["error"] is None and step["output"]["order_id"] == task.instruction.split()[3]
    assert api.post(f"/v1/env/sessions/{s['session_id']}/user_turn").json()["message"] is None
    fin = api.post(f"/v1/env/sessions/{s['session_id']}/finish", json={}).json()
    assert any(x["name"] == "task_success" for x in fin["rewards"])
    assert api.post(f"/v1/env/sessions/{s['session_id']}/step", json={"tool_call": ToolCall(name="x").model_dump()}).status_code == 404
    # verify by replay
    ros = []
    evaluate(ToolAgent(AgentConfig(model="mock"), llm=MockLLM()), [task], envs={"support_desk": env}, on_rollout=lambda ro, t, r: ros.append(ro))
    v = api.post("/v1/env/verify", json={"env": "support_desk", "task": task.model_dump(), "rollout": ros[0].model_dump(mode="json")}).json()
    assert v["total_reward"] == pytest.approx(ros[0].total_reward)


def test_openrlhf_agent_instance_in_process(tmp_path):
    os.environ.pop("SAPHIRE_ENV_SERVER", None)
    env = get_environment("support_desk")
    task = env.generate_tasks(3, seed=1)[0]
    path = openrlhf.export_openrlhf_prompts([task], str(tmp_path / "p.jsonl"))
    label = json.loads(open(path).readline())["label"]
    inst = openrlhf.AgentInstance()
    obs = asyncio.run(inst.reset({"label": label}))
    assert task.instruction in obs["observation_text"]
    oid = task.instruction.split()[3]
    out = asyncio.run(inst.step({"action_text": '<tool_call>{"name": "lookup_order", "arguments": {"order_id": "%s"}}</tool_call>' % oid}))
    assert not out["done"] and oid in out["environment_feedback"]
    out = asyncio.run(inst.step({"action_text": "The order is " + json.loads(out["environment_feedback"])["status"]}))
    assert out["done"] and out["rewards"] > 0.5
