"""Production intelligence (drift, failure clusters, task mining, coverage, recommendations, one-click apply) + webhooks."""
import hashlib
import hmac
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

from conftest import wait_job

from saphire.environments import get_environment
from saphire.environments.support_desk import multi_agent_config
from saphire.evaluation.runner import evaluate, record_from
from saphire.evaluation.suites import build_suite
from saphire.intelligence import cluster_failures, coverage_gaps, drift_report, mine_tasks_from_rollouts, mine_tasks_from_traces, recommend
from saphire.sdk import AgentConfig, MockLLM, ToolAgent


def _run(model_llm, tasks, envs):
    ros, recs = [], []

    def cb(ro, t, r):
        ros.append(ro)
        recs.append(record_from(ro, t, r).model_dump())

    evaluate(ToolAgent(AgentConfig(model="mock", router_top_k=8), llm=model_llm), tasks, envs=envs, on_rollout=cb)
    return ros, recs


def test_drift_detects_regression_and_shift():
    env = get_environment("support_desk")
    envs = {"support_desk": env}
    tasks = build_suite("tool_selection", n_per_env=24, seed=2)
    _, good = _run(MockLLM(), tasks, envs)
    _, bad = _run(MockLLM(error_rate=0.6, seed=3), tasks, envs)
    rep = drift_report(good, bad)
    assert not rep["healthy"] and any(a["type"] == "regression" and a["metric"] == "task_success" for a in rep["alerts"])
    assert rep["tool_shift"]["js_divergence"] > 0
    assert drift_report(good, good)["healthy"]
    assert "note" in drift_report([], good)


def test_failure_clusters_and_mining_and_coverage():
    env = get_environment("support_desk")
    envs = {"support_desk": env}
    tasks = build_suite("full", n_per_env=28, seed=4)
    tasks = [t for t in tasks if t.env_name == "support_desk"]
    ros_bad, _ = _run(MockLLM(error_rate=0.5, seed=1), tasks, envs)
    ros_good, _ = _run(MockLLM(), tasks, envs)
    fam = {t.id: t.tags[0] for t in tasks}
    fc = cluster_failures(ros_bad, families=fam, baseline=ros_good)
    assert fc["n_failed"] > 0 and fc["clusters"] and fc["clusters"][0]["count"] >= fc["clusters"][-1]["count"]
    assert all(c["trend"] in ("new", "growing", "stable") for c in fc["clusters"])
    mined = mine_tasks_from_rollouts(ros_good)
    assert mined and all(t.expected["tools"] and "mined" in t.tags for t in mined)
    # a mined task is solvable by the same policy
    st = env.reset(mined[0])
    ro = ToolAgent(AgentConfig(model="mock", router_top_k=0), llm=MockLLM()).run(mined[0], env, st)
    assert {r.name: r.value for r in env.verify(mined[0], ro, st) if r.step_index is None}["tool_selection_f1"] > 0.5
    # human score overrides the verifier
    scored = mine_tasks_from_rollouts(ros_bad, human_scores={ro.id: 1.0 for ro in ros_bad[:3]})
    assert len(scored) >= 1
    cov = coverage_gaps(ros_good, tasks[:5])
    assert cov["uncovered_patterns"] > 0 and cov["uncovered_traffic_share"] > 0
    assert coverage_gaps(ros_good, tasks)["uncovered_traffic_share"] < 0.6


def test_mine_from_traces():
    traces = [{"id": "t1", "name": "agent"}, {"id": "t2", "name": "agent"}]
    spans = {"t1": [{"span_id": "r", "parent_span_id": None, "kind": "agent", "attributes": {"input.value": "Refund order ORD-1"}, "start_time": 1, "name": "agent"},
                    {"span_id": "a", "parent_span_id": "r", "kind": "tool", "attributes": {"tool.name": "lookup_order"}, "start_time": 2, "name": "lookup_order"},
                    {"span_id": "b", "parent_span_id": "r", "kind": "tool", "attributes": {"tool.name": "issue_refund"}, "start_time": 3, "name": "issue_refund"}],
             "t2": [{"span_id": "r2", "parent_span_id": None, "kind": "agent", "attributes": {"input.value": "x"}, "start_time": 1, "name": "agent"}]}
    tasks = mine_tasks_from_traces(traces, spans, scores_by_trace={"t1": 1.0, "t2": 0.0})
    assert len(tasks) == 1 and tasks[0].expected["tools"] == ["lookup_order", "issue_refund"] and tasks[0].expected["ordered"]


def test_recommendations_are_actionable():
    recs = recommend(drift={"alerts": [{"type": "regression", "metric": "tool_selection_f1", "message": "m"}]},
                     failures={"clusters": [{"kind": "wrong_tool", "role": "orders", "tool": "track_return", "family": "multiturn", "count": 9, "rate": 0.3, "trend": "new"}], "n_failed": 12},
                     coverage={"uncovered_traffic_share": 0.5, "uncovered_patterns": 4}, multi_agent=True)
    kinds = [r["action"]["kind"] for r in recs]
    assert "training" in kinds and "mine_tasks" in kinds and recs[0]["priority"] >= recs[-1]["priority"]
    router_rec = next(r for r in recs if r["action"].get("algorithm") == "router" and "optimize_roles" in r["action"]["params"])
    assert router_rec["action"]["params"]["optimize_roles"] == ["orders"]
    assert recommend()[0]["action"]["kind"] == "none"


def test_intelligence_api_end_to_end(api):
    cfg = multi_agent_config(model="mock:error=0.4,seed=2", name="sys")
    a = api.post("/v1/agents", json={"config": cfg.model_dump()}).json()
    ds = api.post("/v1/datasets", json={"name": "d", "suite": "tool_selection", "n_per_env": 10, "seed": 1}).json()
    wait_job(api, api.post("/v1/evals", json={"agent_id": a["id"], "dataset_id": ds["id"]}).json()["job_id"])
    rep = api.post("/v1/intelligence/run", json={"sync": True, "recent_hours": 1}).json()
    assert rep["n_recent"] == 20 and rep["failures"]["n_failed"] > 0 and rep["recommendations"]
    assert api.get("/v1/intelligence/latest").json()["id"] == rep["id"]
    assert api.get("/v1/intelligence/reports").json()[0]["n_recommendations"] >= 1
    job = api.post("/v1/intelligence/run", json={"recent_hours": 1}).json()
    assert wait_job(api, job["id"])["result"]["report_id"]
    mined = api.post("/v1/intelligence/mine-tasks", json={"since_hours": 1, "name": "prod tasks"}).json()
    assert mined["n_tasks"] >= 1 and api.get(f"/v1/datasets/{mined['dataset_id']}").json()["suite"] == "mined"
    rec = next(r for r in rep["recommendations"] if r["action"]["kind"] == "training")
    applied = api.post("/v1/intelligence/apply", json={"recommendation": rec, "agent_id": a["id"], "train_dataset_id": ds["id"], "eval_dataset_id": ds["id"]}).json()
    assert applied["kind"] == "training"
    wait_job(api, applied["job_id"])
    assert api.get(f"/v1/training/{applied['training_run_id']}").json()["status"] == "succeeded"


class _Receiver(BaseHTTPRequestHandler):
    received: list = []
    secret = "s3cret"

    def do_POST(self):
        body = self.rfile.read(int(self.headers["content-length"]))
        sig = "sha256=" + hmac.new(self.secret.encode(), body, hashlib.sha256).hexdigest()
        _Receiver.received.append({"ok": sig == self.headers.get("x-saphire-signature"), "event": self.headers.get("x-saphire-event"), "body": json.loads(body)})
        self.send_response(200)
        self.end_headers()

    def log_message(self, *a):  # silence
        pass


def test_webhooks_signed_delivery(api):
    srv = HTTPServer(("127.0.0.1", 0), _Receiver)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_port}/hook"
    assert api.post("/v1/orgs/current/webhooks", json={"url": url, "events": ["nope"]}).status_code == 400
    w = api.post("/v1/orgs/current/webhooks", json={"url": url, "events": ["test", "job.succeeded", "gate.decided"], "secret": _Receiver.secret}).json()
    t = api.post(f"/v1/orgs/current/webhooks/{w['id']}/test").json()
    assert t[0]["status_code"] == 200 and _Receiver.received[-1]["ok"] and _Receiver.received[-1]["event"] == "test"
    # a finished job triggers job.succeeded
    a = api.post("/v1/agents", json={"config": AgentConfig(name="a", model="mock").model_dump()}).json()
    api.post(f"/v1/agents/{a['id']}/promote")
    ds = api.post("/v1/datasets", json={"name": "d", "suite": "smoke", "n_per_env": 2}).json()
    wait_job(api, api.post("/v1/evals", json={"agent_id": a["id"], "dataset_id": ds["id"], "gate": True}).json()["job_id"])
    deadline = time.time() + 5
    while time.time() < deadline and not {r["event"] for r in _Receiver.received} >= {"job.succeeded", "gate.decided"}:
        time.sleep(0.1)
    events = {r["event"] for r in _Receiver.received}
    assert {"job.succeeded", "gate.decided"} <= events and all(r["ok"] for r in _Receiver.received)
    lst = api.get("/v1/orgs/current/webhooks").json()
    assert lst[0]["last_delivery"]["status_code"] == 200 and "secret" not in lst[0]
    assert len(api.get(f"/v1/orgs/current/webhooks/{w['id']}/deliveries").json()) >= 3
    api.delete(f"/v1/orgs/current/webhooks/{w['id']}")
    srv.shutdown()
