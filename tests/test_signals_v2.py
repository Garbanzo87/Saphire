"""Reward model, judge calibration, tool statistics/priors, tool-description optimisation."""
from conftest import wait_job

from saphire.environments import get_environment
from saphire.evaluation.runner import evaluate
from saphire.evaluation.suites import build_suite
from saphire.sdk import AgentConfig, MockLLM, ToolAgent, ToolRouter
from saphire.sdk.types import ToolSpec
from saphire.signals.calibration import calibrate
from saphire.signals.judges import make_judge
from saphire.signals.reward_model import RewardModel, labels_from_rollouts
from saphire.training.tools_v2 import optimize_tool_descriptions, tool_stats


def _rollouts(err, n=28, seed=1):
    env = get_environment("support_desk")
    tasks = [t for t in build_suite("full", n_per_env=n, seed=seed) if t.env_name == "support_desk"]
    ros = []
    evaluate(ToolAgent(AgentConfig(model="mock", router_top_k=8), llm=MockLLM(error_rate=err, seed=seed)), tasks, envs={"support_desk": env},
             on_rollout=lambda ro, t, r: ros.append(ro))
    for ro, t in zip(ros, tasks):
        ro.metadata["expected_tools"] = t.expected.get("tools", [])
    return env, tasks, ros


def test_reward_model_learns_and_serves_as_judge(tmp_path):
    _, _, ros = _rollouts(0.5, n=56)
    xs, ys = labels_from_rollouts(ros)
    assert 0 < sum(ys) < len(ys)
    rm = RewardModel()
    m = rm.fit(xs[10:], ys[10:])
    val = rm.evaluate(xs[:10], ys[:10])
    assert m["accuracy"] > 0.7 and val["n"] == 10
    path = rm.save(tmp_path / "rm")
    judge = make_judge(f"rm:{path}")
    r = judge(None, ros[0])
    assert r[0].name == "rm_score" and 0 <= r[0].value <= 1
    # human labels override verifier labels
    xs2, ys2 = labels_from_rollouts(ros[:5], human_scores={ros[0].id: 0.0, ros[1].id: 1.0})
    assert ys2[0] == 0.0 and ys2[1] == 1.0


def test_calibration_metrics():
    c = calibrate([0.9, 0.8, 0.2, 0.1, 0.6], [1, 1, 0, 0, 0])
    assert c["n"] == 5 and c["agreement"] == 0.8 and c["kappa"] > 0.5 and c["best_threshold"] >= 0.6 and c["reliability"]
    assert calibrate([], [])["n"] == 0


def test_tool_stats_and_router_prior():
    _, _, ros = _rollouts(0.4)
    st = tool_stats(ros)
    assert st and all(0 <= v["error_rate"] <= 1 for v in st.values()) and any(v["confused_with"] for v in st.values())
    specs = [ToolSpec(name="a", description="alpha"), ToolSpec(name="b", description="alpha beta")]
    r = ToolRouter(["a", "b"], alpha=0.0)
    assert r.rank("alpha", specs, top_k=1)[0].name in ("a", "b")
    r.set_prior({"b": {"calls": 50, "error_rate": 1.0}, "a": {"calls": 50, "error_rate": 0.0}})
    assert r.rank("alpha beta", specs, top_k=1)[0].name == "a"  # unreliable b is demoted


def test_tool_description_optimisation_improves_mock():
    env, tasks, noisy = _rollouts(0.5, n=42, seed=3)
    expected = {t.id: t.expected.get("tools", []) for t in tasks}
    overrides = optimize_tool_descriptions(env, noisy, expected)
    assert overrides and all("Use for" in v or "Not for" in v for v in overrides.values())
    cfg = AgentConfig(model="mock", router_top_k=8, tool_description_overrides=overrides)
    specs = cfg.apply_overrides(env.tools.specs())
    assert any(s.description != env.tools.get(s.name).spec.description for s in specs)


def test_training_jobs_reward_model_and_tool_descriptions(api):
    a = api.post("/v1/agents", json={"config": AgentConfig(name="a", model="mock:error=0.4,seed=1", router_top_k=8).model_dump()}).json()
    ds = api.post("/v1/datasets", json={"name": "d", "suite": "full", "n_per_env": 14, "seed": 1}).json()
    wait_job(api, api.post("/v1/evals", json={"agent_id": a["id"], "dataset_id": ds["id"], "judge_model": "mock"}).json()["job_id"])
    t = api.post("/v1/training", json={"agent_id": a["id"], "algorithm": "reward_model", "params": {}}).json()
    wait_job(api, t["job_id"])
    res = api.get(f"/v1/training/{t['id']}").json()["result"]
    assert res["judge_model"].startswith("rm:") and "accuracy" in res["validation"]
    ev = api.post("/v1/evals", json={"agent_id": a["id"], "dataset_id": ds["id"], "judge_model": res["judge_model"]}).json()
    wait_job(api, ev["job_id"])
    assert any(s["name"] == "rm_score" for s in api.get("/v1/scores", params={"name": "rm_score"}).json())
    t2 = api.post("/v1/training", json={"agent_id": a["id"], "algorithm": "tool_descriptions", "params": {}}).json()
    wait_job(api, t2["job_id"])
    out = api.get(f"/v1/training/{t2['id']}").json()
    assert out["result"]["n_overrides"] >= 1 and api.get(f"/v1/agents/{out['output_agent_id']}").json()["config"]["tool_description_overrides"]
    # calibration: attach human scores to some rollouts then compare with the judge
    first_eval = api.get("/v1/evals").json()[-1]["id"]
    rows = api.get("/v1/rollouts", params={"limit": 10, "eval_run_id": first_eval}).json()["items"]
    for r in rows:
        api.post("/v1/scores", json={"name": "thumbs", "value": 1.0 if r["task_success"] else 0.0, "rollout_id": r["id"], "source": "human"})
    cal = api.get("/v1/signals/calibration", params={"judge": "judge_score"}).json()
    assert cal["paired_rollouts"] >= 5 and "kappa" in cal
    assert api.get("/v1/signals/tool-stats").json()
