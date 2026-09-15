"""Artifact store mirroring, retention, bulk ingest, benchmark harness smoke."""

from saphire.evaluation.suites import build_suite
from saphire.sdk import AgentConfig, ExemplarStore, ToolRouter
from saphire.sdk import artifacts as A
from saphire.sdk.types import ToolSpec


def test_artifact_store_roundtrip_via_file_scheme(tmp_path, monkeypatch):
    monkeypatch.setenv("SAPHIRE_ARTIFACT_STORE", f"file://{tmp_path}/mirror")
    monkeypatch.setattr(A, "CACHE_DIR", tmp_path / "cache")
    A._STORE = None
    specs = [ToolSpec(name="a", description="alpha"), ToolSpec(name="b", description="beta")]
    r = ToolRouter(["a", "b"])
    r.fit([("alpha", "a", 1.0), ("beta", "b", 1.0)] * 3)
    local = tmp_path / "v1"
    p = r.save(local / "router")
    ExemplarStore().save(local / "exemplars.json")
    mapping = A.publish_dir(local, "proj/agent/run/v1")
    assert all(v.startswith("file://") for v in mapping.values()) and p in mapping
    remote = A.remap(p, mapping)
    # loading from the remote URI pulls router.json + router.npz into the cache
    r2 = ToolRouter.load(remote)
    assert r2.rank("beta", specs, top_k=1)[0].name == "b"
    assert (tmp_path / "cache").exists()
    ExemplarStore.load(A.remap(str(local / "exemplars.json"), mapping))


def test_training_job_publishes_artifacts(api, tmp_path, monkeypatch):
    from conftest import wait_job

    monkeypatch.setenv("SAPHIRE_ARTIFACT_STORE", f"file://{tmp_path}/mirror")
    monkeypatch.setattr(A, "CACHE_DIR", tmp_path / "cache")
    A._STORE = None
    a = api.post("/v1/agents", json={"config": AgentConfig(name="a", model="mock:error=0.3,seed=1", router_top_k=8).model_dump()}).json()
    ds = api.post("/v1/datasets", json={"name": "d", "suite": "smoke", "n_per_env": 6}).json()
    wait_job(api, api.post("/v1/evals", json={"agent_id": a["id"], "dataset_id": ds["id"]}).json()["job_id"])
    t = api.post("/v1/training", json={"agent_id": a["id"], "algorithm": "router", "params": {}}).json()
    wait_job(api, t["job_id"])
    out_id = api.get(f"/v1/training/{t['id']}").json()["output_agent_id"]
    out = api.get(f"/v1/agents/{out_id}").json()
    assert out["config"]["tool_router"].startswith("file://")
    # the published version is usable by a worker on another node (fresh cache)
    ev = api.post("/v1/evals", json={"agent_id": out["id"], "dataset_id": ds["id"]}).json()
    wait_job(api, ev["job_id"])
    assert api.get(f"/v1/evals/{ev['id']}").json()["status"] == "succeeded"


def test_retention_purges_old_data(api):
    from saphire.server import db as D
    from saphire.server.retention import run_retention

    tasks = build_suite("smoke", n_per_env=4, seed=1)
    api.post("/v1/datasets", json={"name": "d", "suite": "smoke", "n_per_env": 4})
    a = api.post("/v1/agents", json={"config": AgentConfig(name="a", model="mock").model_dump()}).json()
    from conftest import wait_job

    ds = api.get("/v1/datasets").json()[0]
    wait_job(api, api.post("/v1/evals", json={"agent_id": a["id"], "dataset_id": ds["id"]}).json()["job_id"])
    db = D.session()
    try:
        assert db.query(D.RolloutRow).count() == len(tasks) and db.query(D.Trace).count() >= len(tasks)
        # nothing configured -> skipped
        assert "skipped" in run_retention(db, 0)["default"]
        # age everything by 100 days, keep 30
        for row in db.query(D.RolloutRow).all():
            row.created_at -= 100 * 86400
        for row in db.query(D.Trace).all():
            row.created_at -= 100 * 86400
        db.commit()
        rep = run_retention(db, 30, dry_run=True)["default"]
        assert rep["rollouts"] == len(tasks) and db.query(D.RolloutRow).count() == len(tasks)
        rep = run_retention(db, 30)["default"]
        assert rep["rollouts"] == len(tasks) and db.query(D.RolloutRow).count() == 0 and db.query(D.Span).count() == 0
        assert db.query(D.EvalRun).count() == 1 and db.query(D.MetricPoint).count() > 0  # aggregates survive
    finally:
        db.close()


def test_bulk_ingest_upserts(api):
    import uuid

    tid = uuid.uuid4().hex
    root = {"trace_id": tid, "span_id": "r1", "parent_span_id": None, "name": "agent", "kind": "agent", "start_time": 1.0, "end_time": 3.0,
            "status": "ok", "attributes": {}, "service": "s", "events": []}
    kids = [{**root, "span_id": f"k{i}", "parent_span_id": "r1", "name": f"tool{i}", "kind": "tool", "start_time": 1.5, "end_time": 2.0} for i in range(50)]
    assert api.post("/v1/traces/ingest", json={"spans": [root] + kids}).json() == {"traces": 1, "spans": 51}
    # re-sending the same spans updates rather than duplicates
    kids[0]["name"] = "renamed"
    api.post("/v1/traces/ingest", json={"spans": [root] + kids})
    t = api.get(f"/v1/traces/{tid}").json()
    assert t["n_spans"] == 51 and any(s["name"] == "renamed" for s in t["spans"])


def test_bench_harness_smoke():
    from saphire.bench import bench_db, bench_rollouts, to_markdown

    r = {"machine": {"cpu_count": 2, "platform": "x", "python": "3"}, "timestamp": "t",
         "rollouts": bench_rollouts(n_per_env=3, k=1, backends=("local", "thread")), "db": bench_db(20)}
    assert r["rollouts"]["local"]["rollouts_per_min"] > 0 and r["db"]["insert_per_s"] > 0
    assert "Rollout throughput" in to_markdown(r)
