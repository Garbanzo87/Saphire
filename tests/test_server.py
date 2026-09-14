from conftest import wait_job

from saphire.sdk import AgentConfig, Message, Role, RolloutRecorder, TaskSpec


def _agent(api, **kw):
    cfg = AgentConfig(name="a", model=kw.pop("model", "mock:error=0.3,seed=1"), router_top_k=8, **kw)
    return api.post("/v1/agents", json={"config": cfg.model_dump()}).json()


def test_health_and_auth(api):
    assert api.get("/health").json()["status"] == "ok"
    r = api.get("/v1/agents", headers={"x-api-key": "wrong"})
    assert r.status_code == 401


def test_trace_ingest_and_query(api):
    spans = [{"trace_id": "t1", "span_id": "s1", "parent_span_id": None, "name": "agent", "kind": "agent", "start_time": 1.0, "end_time": 2.0,
              "status": "ok", "attributes": {"rollout.id": "ro_x"}, "service": "svc", "events": []},
             {"trace_id": "t1", "span_id": "s2", "parent_span_id": "s1", "name": "llm", "kind": "llm", "start_time": 1.1, "end_time": 1.5,
              "status": "ok", "attributes": {}, "service": "svc", "events": []}]
    r = api.post("/v1/traces/ingest", json={"project": "default", "spans": spans})
    assert r.json() == {"traces": 1, "spans": 2}
    lst = api.get("/v1/traces").json()
    assert lst["total"] == 1 and lst["items"][0]["n_spans"] == 2 and lst["items"][0]["rollout_id"] == "ro_x"
    det = api.get("/v1/traces/t1").json()
    assert len(det["spans"]) == 2
    api.post("/v1/scores", json={"name": "thumbs", "value": 1.0, "trace_id": "t1", "source": "human"})
    assert api.get("/v1/traces/t1").json()["scores"][0]["name"] == "thumbs"
    assert api.get("/v1/scores/summary").json()[0]["count"] == 1


def test_eval_training_gate_flow(api):
    a = _agent(api)
    assert a["version"] == "v0"
    api.post(f"/v1/agents/{a['id']}/promote")
    train = api.post("/v1/datasets", json={"name": "train", "suite": "full", "n_per_env": 21, "seed": 100, "split": "train"}).json()
    ev = api.post("/v1/datasets", json={"name": "eval", "suite": "full", "n_per_env": 14, "seed": 200}).json()
    assert train["n_tasks"] == 42
    run = api.post("/v1/evals", json={"agent_id": a["id"], "dataset_id": ev["id"], "k": 2, "judge_model": "mock"}).json()
    wait_job(api, run["job_id"])
    det = api.get(f"/v1/evals/{run['id']}").json()
    assert det["status"] == "succeeded" and "pass_hat_2" in det["metrics"] and det["metrics"]["judge_score"] > 0
    assert len(det["rollouts"]) == 28 * 2
    assert api.get("/v1/rollouts", params={"eval_run_id": run["id"]}).json()["total"] == 56
    assert api.get("/v1/traces").json()["total"] >= 56
    tr = api.post("/v1/training", json={"agent_id": a["id"], "algorithm": "online",
                                        "params": {"train_dataset_id": train["id"], "eval_dataset_id": ev["id"], "iterations": 3, "batch_size": 16, "learn_prompt_every": 2}}).json()
    wait_job(api, tr["job_id"])
    trd = api.get(f"/v1/training/{tr['id']}").json()
    assert trd["status"] == "succeeded" and len(trd["history"]) == 4 and trd["output_agent_id"]
    assert trd["result"]["improvement"]["task_success"] > 0
    versions = api.get("/v1/agents", params={"name": "a"}).json()
    assert len(versions) == 4
    out = trd["output_agent_id"]
    run2 = api.post("/v1/evals", json={"agent_id": out, "dataset_id": ev["id"], "gate": True, "auto_promote": True,
                                       "gate_policy": {"min_metrics": {"task_success": 0.5}}}).json()
    j = wait_job(api, run2["job_id"])
    assert j["result"]["gate"]["passed"] is True
    deps = api.get("/v1/deployments").json()
    assert deps[0]["promoted"] and deps[0]["baseline_version"] == "v0"
    assert api.get("/v1/deployments/current").json()[0]["id"] == out
    cmp = api.get(f"/v1/evals/{run['id']}/compare/{run2['id']}").json()
    assert cmp["comparison"]["task_success"]["delta"] > 0
    ts = api.get("/v1/metrics/timeseries", params={"agent_name": "a", "name": "task_success"}).json()
    assert len(ts) >= 5 and ts[-1]["value"] >= ts[0]["value"]
    ov = api.get("/v1/metrics/overview").json()
    assert ov["agents"]["a"]["deployed_agent_id"] == out and ov["n_training_runs"] == 1
    # derived learners from stored rollouts
    for algo in ("router", "exemplars", "signals"):
        t = api.post("/v1/training", json={"agent_id": out, "algorithm": algo, "params": {}}).json()
        wait_job(api, t["job_id"])
        assert api.get(f"/v1/training/{t['id']}").json()["status"] == "succeeded"
    po = api.post("/v1/training", json={"agent_id": out, "algorithm": "prompt_opt", "params": {"train_dataset_id": train["id"], "iterations": 1}}).json()
    wait_job(api, po["job_id"])


def test_experiments(api):
    a, b = _agent(api), _agent(api)
    e = api.post("/v1/experiments", json={"name": "ab", "variants": [{"name": "c", "agent_id": a["id"], "weight": 1}, {"name": "t", "agent_id": b["id"], "weight": 1}]}).json()
    seen = set()
    for i in range(40):
        v = api.get(f"/v1/experiments/{e['id']}/assign", params={"unit": f"u{i}"}).json()
        assert v == api.get(f"/v1/experiments/{e['id']}/assign", params={"unit": f"u{i}"}).json()  # sticky
        seen.add(v["variant"])
        api.post(f"/v1/experiments/{e['id']}/outcomes", json={"unit": f"u{i}", "value": 1.0 if v["variant"] == "t" else 0.0})
    assert seen == {"c", "t"}
    res = api.get(f"/v1/experiments/{e['id']}").json()["results"]
    assert res["t"]["mean"] == 1.0 and res["c"]["mean"] == 0.0 and res["p_value"] < 0.01


def test_byo_rollout_ingest(api):
    a = _agent(api)
    task = TaskSpec(env_name="custom", instruction="say hi", expected={})
    rec = RolloutRecorder(task.id, "custom", agent_id="a")
    rec.step([Message(role=Role.user, content="say hi")], Message(role=Role.assistant, content="hi"))
    ro = rec.finish("hi")
    r = api.post("/v1/rollouts", json={"rollout": ro.model_dump(mode="json"), "task": task.model_dump(), "agent_id": a["id"],
                                       "rewards": [{"value": 1.0, "source": "human", "name": "task_success"}]}).json()
    assert r["task_success"] == 1.0
    assert api.get(f"/v1/rollouts/{ro.id}").json()["scores"][0]["value"] == 1.0


def test_bad_requests(api):
    assert api.post("/v1/datasets", json={"name": "x"}).status_code == 400
    a = _agent(api)
    assert api.post("/v1/training", json={"agent_id": a["id"], "algorithm": "online", "params": {}}).status_code == 400
    assert api.get("/v1/agents/nope").status_code == 404
