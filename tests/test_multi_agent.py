import json

from saphire.environments import get_environment
from saphire.environments.support_desk import multi_agent_config
from saphire.evaluation.runner import evaluate
from saphire.evaluation.suites import build_suite
from saphire.sdk import ExemplarStore
from saphire.sdk.multi_agent import ORCHESTRATOR, AgentSystem, build_agent, role_rewards
from saphire.signals.generate import router_examples, sft_examples
from saphire.training.multi_agent import train_roles
from saphire.training.online import OnlineLoop


def _tasks(n=21, seed=5):
    return [t for t in build_suite("full", n_per_env=n, seed=seed) if t.env_name == "support_desk"]


def test_build_agent_picks_system():
    cfg = multi_agent_config()
    assert cfg.is_multi_agent and isinstance(build_agent(cfg), AgentSystem)


def test_system_solves_tasks_and_tags_roles():
    env = get_environment("support_desk")
    res = evaluate(build_agent(multi_agent_config(model="mock")), _tasks(), envs={"support_desk": env})
    assert res.metrics["task_success"] >= 0.8
    assert set(res.by_role) >= {ORCHESTRATOR, "orders"}
    assert res.by_role[ORCHESTRATOR]["step_reward"] > 0  # hand-off credit assignment


def test_specialists_cannot_use_other_roles_tools():
    env = get_environment("support_desk")
    cfg = multi_agent_config(model="mock")
    cfg.roles["billing"].tool_names = ["issue_refund"]
    sys_ = build_agent(cfg)
    task = [t for t in _tasks() if t.tags[0] == "refund_resolve"][0]
    st = env.reset(task)
    ro = sys_.run(task, env, st)
    for s in ro.steps:
        if s.role == "billing":
            assert set(s.exposed_tools) <= {"issue_refund"}
    rr = role_rewards(ro)
    assert "billing" in rr and rr["billing"]["n_tool_calls"] >= 1


def test_role_filtered_signals_and_exemplars():
    env = get_environment("support_desk")
    ros = []
    evaluate(build_agent(multi_agent_config(model="mock")), _tasks(), envs={"support_desk": env}, on_rollout=lambda ro, t, r: ros.append(ro))
    orders = router_examples(ros, role="orders")
    assert orders and all(n in env.tools for _, n, _ in orders)
    assert all("delegate_to_" not in n for _, n, _ in orders)
    orch = router_examples(ros, role=ORCHESTRATOR)
    assert orch and all(n.startswith("delegate_to_") for _, n, _ in orch)
    assert all(r["role"] == "tickets" for r in sft_examples(ros, role="tickets"))
    store = ExemplarStore()
    assert any(store.add_rollout(ro, role="orders") for ro in ros)
    assert all(i["role"] == "orders" and "Context:" not in i["instruction"] for i in store.items)


def test_selective_training_freezes_other_roles(tmp_path):
    env = get_environment("support_desk")
    ros = []
    evaluate(build_agent(multi_agent_config(model="mock:error=0.4,seed=1")), _tasks(), envs={"support_desk": env},
             on_rollout=lambda ro, t, r: ros.append(ro))
    cfg = multi_agent_config(model="mock")
    new_cfg, rep = train_roles(cfg, ros, {"support_desk": env}, tmp_path, roles=["orders", "tickets"])
    assert set(rep["frozen"]) == {ORCHESTRATOR, "customers", "billing"}
    assert new_cfg.roles["orders"].tool_router and new_cfg.roles["orders"].exemplar_store
    assert new_cfg.roles["customers"].tool_router is None and new_cfg.tool_router is None
    router_meta = json.loads(open(new_cfg.roles["orders"].tool_router).read())
    assert set(router_meta["tool_names"]) == set(cfg.roles["orders"].tool_names)
    assert set(rep["role_step_reward"]) >= {"orders", ORCHESTRATOR}


def test_online_loop_multi_agent_improves(tmp_path):
    train, ev = _tasks(42, 100), _tasks(28, 200)
    loop = OnlineLoop(multi_agent_config(model="mock:error=0.3,seed=3", router_top_k=4), train, ev, artifacts_dir=tmp_path,
                      batch_size=24, seed=1)
    hist = loop.run(4)
    assert hist[-1]["eval"]["task_success"] > hist[0]["eval"]["task_success"]
    assert loop.config.roles["orders"].exemplar_store and loop.config.exemplar_store
    assert hist[-1]["updates"]["roles"]["frozen"] == []


def test_server_multi_agent_eval_and_selective_training(api):
    from conftest import wait_job

    cfg = multi_agent_config(model="mock:error=0.3,seed=1", name="sys")
    a = api.post("/v1/agents", json={"config": cfg.model_dump()}).json()
    ds = api.post("/v1/datasets", json={"name": "d", "suite": "tool_selection", "n_per_env": 8, "seed": 1}).json()
    run = api.post("/v1/evals", json={"agent_id": a["id"], "dataset_id": ds["id"]}).json()
    wait_job(api, run["job_id"])
    det = api.get(f"/v1/evals/{run['id']}").json()
    assert det["by_role"] and ORCHESTRATOR in det["by_role"]
    t = api.post("/v1/training", json={"agent_id": a["id"], "algorithm": "router", "params": {"optimize_roles": ["orders"]}}).json()
    wait_job(api, t["job_id"])
    tr = api.get(f"/v1/training/{t['id']}").json()
    assert tr["status"] == "succeeded" and "customers" in tr["result"]["frozen"]
    out = api.get(f"/v1/agents/{tr['output_agent_id']}").json()
    assert out["config"]["roles"]["orders"]["tool_router"] and not out["config"]["roles"]["customers"]["tool_router"]


def test_attribution_functions_and_api(api):
    from conftest import wait_job

    from saphire.sdk.attribution import advantage_credit, blame, role_ablation, shapley_attribution

    env = get_environment("support_desk")
    envs = {"support_desk": env}
    tasks = _tasks(14, 3)
    cfg = multi_agent_config(model="mock:error=0.3,seed=2")
    ros = []
    evaluate(build_agent(cfg), tasks, envs=envs, on_rollout=lambda ro, t, r: ros.append(ro))
    kinds = {blame(ro)["kind"] for ro in ros}
    assert kinds & {"wrong_tool", "bad_handoff", "tool_error", "incomplete"}
    credit = advantage_credit(ros)
    assert set(credit["roles"]) >= {ORCHESTRATOR, "orders"} and abs(sum(v["fault_share"] for v in credit["roles"].values()) - 1) < 1e-6
    ab = role_ablation(cfg, tasks, envs, reference_model="mock", degraded_model="mock:error=0.9", roles=[ORCHESTRATOR, "orders"])
    assert ab["roles"][ORCHESTRATOR]["headroom"] > 0 and "criticality" in ab["roles"]["orders"]
    sh = shapley_attribution(cfg, tasks, envs, reference_model="mock", roles=[ORCHESTRATOR, "orders", "tickets"], exact=True)
    assert abs(sum(sh["shapley"].values()) - (sh["all_upgraded"] - sh["baseline"])) < 1e-9
    # API: cheap attribution on a stored eval + counterfactual job
    a = api.post("/v1/agents", json={"config": cfg.model_dump()}).json()
    ds = api.post("/v1/datasets", json={"name": "d", "suite": "tool_selection", "n_per_env": 6, "seed": 1}).json()
    run = api.post("/v1/evals", json={"agent_id": a["id"], "dataset_id": ds["id"]}).json()
    wait_job(api, run["job_id"])
    att = api.get(f"/v1/evals/{run['id']}/attribution").json()
    assert ORCHESTRATOR in att["credit"]["roles"] and "by_kind" in att
    job = api.post("/v1/attribution", json={"agent_id": a["id"], "dataset_id": ds["id"], "reference_model": "mock", "degraded_model": "mock:error=0.9",
                                             "roles": [ORCHESTRATOR, "orders"], "n_permutations": 2}).json()
    j = wait_job(api, job["id"])
    assert "shapley" in j["result"] and j["result"]["recommendation"]["optimize_roles"]
    assert api.get("/v1/attribution").json()[0]["id"] == job["id"]
