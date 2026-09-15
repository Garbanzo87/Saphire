"""Load & scale benchmark harness (`saphire bench`).

Measures, with real numbers on the machine it runs on:
  * ingest   – spans/s and request p50/p95 for POST /v1/traces/ingest under concurrency
  * rollouts – rollouts/min for local, thread, process and Ray backends
  * api      – read latency for list/detail endpoints on a populated database
  * db       – bulk insert + query timings for rollouts

Writes JSON + a Markdown table (docs/BENCHMARKS.md is generated from this).
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import os
import platform
import statistics
import time
import uuid
from pathlib import Path
from typing import Any, Optional

import httpx


def _pct(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(len(xs) * p))] if xs else 0.0


def _span(trace_id: str, i: int, parent: Optional[str]) -> dict[str, Any]:
    t = time.time()
    return {"trace_id": trace_id, "span_id": uuid.uuid4().hex[:16], "parent_span_id": parent, "name": f"span-{i}", "kind": "tool" if i else "agent",
            "start_time": t, "end_time": t + 0.01, "status": "ok", "attributes": {"input.value": "x" * 200, "output.value": "y" * 300, "tool.name": "lookup"},
            "service": "bench", "events": []}


def bench_ingest(host: str, api_key: str, n_requests: int = 200, spans_per_request: int = 20, concurrency: int = 8) -> dict[str, Any]:
    client = httpx.Client(base_url=host, timeout=60, headers={"x-api-key": api_key})
    payloads = []
    for _ in range(n_requests):
        tid = uuid.uuid4().hex
        spans = [_span(tid, 0, None)]
        spans += [_span(tid, i, spans[0]["span_id"]) for i in range(1, spans_per_request)]
        payloads.append({"project": "bench", "spans": spans})
    lat: list[float] = []

    def _one(p):
        t0 = time.perf_counter()
        r = client.post("/v1/traces/ingest", json=p)
        lat.append((time.perf_counter() - t0) * 1000)
        return r.status_code

    t0 = time.perf_counter()
    with cf.ThreadPoolExecutor(max_workers=concurrency) as ex:
        codes = list(ex.map(_one, payloads))
    wall = time.perf_counter() - t0
    return {"requests": n_requests, "spans": n_requests * spans_per_request, "concurrency": concurrency, "wall_s": round(wall, 2),
            "spans_per_s": round(n_requests * spans_per_request / wall), "req_per_s": round(n_requests / wall, 1),
            "p50_ms": round(_pct(lat, 0.5), 1), "p95_ms": round(_pct(lat, 0.95), 1), "errors": sum(1 for c in codes if c >= 300)}


def bench_api_reads(host: str, api_key: str, n: int = 100) -> dict[str, Any]:
    client = httpx.Client(base_url=host, timeout=60, headers={"x-api-key": api_key})
    out = {}
    for path in ["/v1/traces?limit=50&project=bench", "/v1/metrics/overview?project=bench", "/v1/agents?project=bench"]:
        lat = []
        for _ in range(n):
            t0 = time.perf_counter()
            client.get(path)
            lat.append((time.perf_counter() - t0) * 1000)
        out[path.split("?")[0]] = {"p50_ms": round(_pct(lat, 0.5), 1), "p95_ms": round(_pct(lat, 0.95), 1)}
    tid = client.get("/v1/traces?limit=1&project=bench").json()["items"][0]["id"]
    lat = []
    for _ in range(n):
        t0 = time.perf_counter()
        client.get(f"/v1/traces/{tid}")
        lat.append((time.perf_counter() - t0) * 1000)
    out["/v1/traces/{id}"] = {"p50_ms": round(_pct(lat, 0.5), 1), "p95_ms": round(_pct(lat, 0.95), 1)}
    return out


def bench_rollouts(n_per_env: int = 56, k: int = 2, backends: tuple[str, ...] = ("local", "thread", "process", "ray"), workers: Optional[int] = None) -> dict[str, Any]:
    os.environ.setdefault("SAPHIRE_TRACING", "off")
    from .evaluation.runner import evaluate
    from .evaluation.suites import build_suite
    from .sdk.multi_agent import build_agent
    from .sdk.types import AgentConfig

    tasks = build_suite("full", n_per_env=n_per_env, seed=7)
    cfg = AgentConfig(model="mock:error=0.2,seed=1", router_top_k=8)
    out: dict[str, Any] = {"n_rollouts": len(tasks) * k}
    for be in backends:
        try:
            t0 = time.perf_counter()
            if be == "local":
                res = evaluate(build_agent(cfg), tasks, k=k)
            else:
                from .distributed.rollouts import evaluate_distributed

                res = evaluate_distributed(cfg, tasks, k=k, backend=be, workers=workers)
            out[be] = {"wall_s": round(time.perf_counter() - t0, 2), "rollouts_per_min": round(res.metrics["throughput_tasks_per_min"]),
                       "task_success": round(res.metrics["task_success"], 3)}
        except Exception as e:  # noqa: BLE001
            out[be] = {"error": f"{type(e).__name__}: {str(e)[:120]}"}
    # a multi-agent system (5 policies per task) for reference
    from .environments.support_desk import multi_agent_config

    t0 = time.perf_counter()
    res = evaluate(build_agent(multi_agent_config(model="mock:error=0.2,seed=1")), [t for t in tasks if t.env_name == "support_desk"], k=1)
    out["multi_agent_local"] = {"wall_s": round(time.perf_counter() - t0, 2), "rollouts_per_min": round(res.metrics["throughput_tasks_per_min"]),
                                "steps_mean": round(res.metrics["steps_mean"], 1)}
    return out


def bench_db(n: int = 2000) -> dict[str, Any]:
    """Direct DB path: persist_rollout throughput and a paged list query on a temp SQLite DB."""
    import tempfile

    from .environments.base import get_environment
    from .sdk import AgentConfig, MockLLM, ToolAgent
    from .server import config
    from .server import db as D
    from .server import service as S

    tmp = tempfile.mkdtemp()
    old = config.settings.database_url
    config.settings.database_url = f"sqlite:///{tmp}/bench.db"
    D.reset_engine()
    try:
        db = D.session()
        proj = D.ensure_project(db, "bench")
        env = get_environment("support_desk")
        task = env.generate_tasks(1, seed=1)[0]
        st = env.reset(task)
        ro = ToolAgent(AgentConfig(model="mock"), llm=MockLLM()).run(task, env, st)
        rewards = env.verify(task, ro, st)
        t0 = time.perf_counter()
        for _ in range(n):
            ro.id = "ro_" + uuid.uuid4().hex[:16]
            S.persist_rollout(db, proj, ro, task, rewards)
        w = time.perf_counter() - t0
        lat = []
        for _ in range(50):
            t1 = time.perf_counter()
            db.query(D.RolloutRow).filter_by(project_id=proj.id).order_by(D.RolloutRow.created_at.desc()).offset(500).limit(50).all()
            lat.append((time.perf_counter() - t1) * 1000)
        db.close()
        return {"rollouts_inserted": n, "insert_per_s": round(n / w), "list_query_p50_ms": round(statistics.median(lat), 2),
                "list_query_p95_ms": round(_pct(lat, 0.95), 2), "db": "sqlite (WAL)"}
    finally:
        config.settings.database_url = old
        D.reset_engine()


def run_all(host: Optional[str], api_key: str = "dev-key", quick: bool = False) -> dict[str, Any]:
    report: dict[str, Any] = {"machine": {"cpu_count": os.cpu_count(), "platform": platform.platform(), "python": platform.python_version()},
                              "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())}
    report["rollouts"] = bench_rollouts(n_per_env=28 if quick else 56, k=1 if quick else 2)
    report["db"] = bench_db(500 if quick else 2000)
    if host:
        report["ingest"] = bench_ingest(host, api_key, n_requests=50 if quick else 200)
        report["api_reads"] = bench_api_reads(host, api_key, n=30 if quick else 100)
    return report


def to_markdown(r: dict[str, Any]) -> str:
    m = r["machine"]
    lines = [f"# Benchmarks\n\nMeasured {r['timestamp']} on {m['cpu_count']} vCPU, {m['platform']}, Python {m['python']}. "
             "Numbers are from the mock policy (no LLM latency) and therefore measure the *platform overhead* — tracing, verification, "
             "persistence, aggregation — not model speed. With hosted models, throughput is bounded by provider latency and scales with "
             "concurrency/workers.\n"]
    ro = r["rollouts"]
    lines.append(f"## Rollout throughput ({ro['n_rollouts']} rollouts, support_desk + data_ops, k trials)\n")
    lines.append("| backend | wall s | rollouts/min | task_success |\n|---|---|---|---|")
    for be in ("local", "thread", "process", "ray"):
        v = ro.get(be, {})
        lines.append(f"| {be} | {v.get('wall_s', '-')} | {v.get('rollouts_per_min', v.get('error', '-'))} | {v.get('task_success', '-')} |")
    ma = ro["multi_agent_local"]
    lines.append(f"\nMulti-agent system (orchestrator + 4 specialists, {ma['steps_mean']} steps/rollout): {ma['rollouts_per_min']} rollouts/min local.\n")
    d = r["db"]
    lines.append(f"## Persistence ({d['db']})\n\n* `persist_rollout` (rollout + scores + metering): **{d['insert_per_s']} rollouts/s**\n"
                 f"* paged rollout list query (offset 500, limit 50): p50 {d['list_query_p50_ms']} ms, p95 {d['list_query_p95_ms']} ms\n")
    if "ingest" in r:
        i = r["ingest"]
        lines.append(f"## Trace ingest (HTTP, {i['concurrency']} concurrent clients, {i['spans'] // i['requests']} spans/request)\n\n"
                     f"* **{i['spans_per_s']} spans/s** ({i['req_per_s']} req/s), p50 {i['p50_ms']} ms, p95 {i['p95_ms']} ms, errors {i['errors']}\n")
    if "api_reads" in r:
        lines.append("## API read latency (populated DB)\n\n| endpoint | p50 ms | p95 ms |\n|---|---|---|")
        for k, v in r["api_reads"].items():
            lines.append(f"| `{k}` | {v['p50_ms']} | {v['p95_ms']} |")
    lines.append("\nReproduce: `saphire serve --inline-jobs &` then `saphire bench --host http://localhost:8000 --out docs/benchmarks.json`.\n")
    return "\n".join(lines)


def main(host: Optional[str], api_key: str, out: Optional[Path], quick: bool) -> dict[str, Any]:
    r = run_all(host, api_key, quick)
    md = to_markdown(r)
    if out:
        out.write_text(json.dumps(r, indent=2))
        out.with_suffix(".md").write_text(md)
    print(md)
    return r
