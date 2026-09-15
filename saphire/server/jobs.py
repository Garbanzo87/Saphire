"""Background job handlers (evaluation runs, training runs) and the worker loop.

Jobs live in the `jobs` table; any number of `saphire worker` processes poll and claim them
(atomic UPDATE ... WHERE status='pending'). With `SAPHIRE_INLINE_JOBS=1` the API process runs
them in a background thread instead (used in tests and single-container deployments).
"""
from __future__ import annotations

import json
import socket
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Optional

from sqlalchemy import update
from sqlalchemy.orm import Session

from ..environments.base import get_environment
from ..evaluation.gates import GatePolicy, evaluate_gate
from ..evaluation.runner import EvalResult, evaluate
from ..sdk.multi_agent import build_agent
from ..sdk.types import AgentConfig, Rollout, TaskSpec
from ..signals.generate import export_jsonl, grpo_prompts, preference_pairs, sft_examples, step_rewards
from ..signals.judges import make_judge
from ..training.learners import train_router, update_exemplars
from ..training.online import OnlineLoop
from ..training.prompt_opt import optimize_prompt
from . import db as D
from . import service as S
from .config import settings

HANDLERS: dict[str, Callable[[Session, D.Job], dict[str, Any]]] = {}


class DBSpanExporter:
    """Writes SDK spans produced inside jobs straight into the traces/spans tables (no HTTP hop)."""

    def __init__(self, project_id: str):
        self.project_id = project_id

    def export(self, spans):
        from opentelemetry.sdk.trace.export import SpanExportResult

        from ..sdk.tracing import span_to_dict

        db = D.session()
        try:
            project = db.get(D.Project, self.project_id)
            S.ingest_spans(db, project, [span_to_dict(s) for s in spans])
            return SpanExportResult.SUCCESS
        except Exception:  # noqa: BLE001
            db.rollback()
            return SpanExportResult.FAILURE
        finally:
            db.close()

    def shutdown(self) -> None:
        pass


def _init_tracing(project_id: str) -> None:
    from ..sdk import tracing

    tracing.init(service_name="saphire-jobs", exporter=DBSpanExporter(project_id), batch=True, force=True)


def handler(name: str):
    def deco(fn):
        HANDLERS[name] = fn
        return fn

    return deco


class JobLog:
    def __init__(self, db: Session, job: D.Job):
        self.db, self.job = db, job

    def __call__(self, msg: str, progress: Optional[float] = None) -> None:
        self.job.logs = (self.job.logs or "") + f"[{time.strftime('%H:%M:%S')}] {msg}\n"
        if progress is not None:
            self.job.progress = progress
        self.db.commit()


# ---------------------------------------------------------------------------
@handler("eval")
def run_eval(db: Session, job: D.Job) -> dict[str, Any]:
    log = JobLog(db, job)
    run = db.get(D.EvalRun, job.params["eval_run_id"])
    project = db.get(D.Project, job.project_id)
    agent_row = db.get(D.Agent, run.agent_id)
    ds = db.get(D.Dataset, run.dataset_id)
    tasks = S.tasks_from_dataset(ds)
    cfg = AgentConfig.model_validate(agent_row.config)
    run.status = "running"
    db.commit()
    log(f"evaluating {cfg.name}/{cfg.version} on {len(tasks)} tasks (k={run.k})", 0.05)
    judge = make_judge(run.judge_model)
    done = {"n": 0}

    def _cb(ro: Rollout, task: TaskSpec, rewards):
        S.persist_rollout(db, project, ro, task, rewards, agent_id=agent_row.id, eval_run_id=run.id)
        done["n"] += 1
        if done["n"] % 10 == 0:
            log(f"{done['n']}/{len(tasks) * run.k} rollouts", 0.05 + 0.85 * done["n"] / (len(tasks) * run.k))

    res: EvalResult = evaluate(build_agent(cfg), tasks, k=run.k, suite=run.suite, judge=judge, on_rollout=_cb,
                               concurrency=int(job.params.get("concurrency", 1)))
    run.metrics, run.by_family, run.by_env, run.by_difficulty = res.metrics, res.by_family, res.by_env, res.by_difficulty
    run.by_role = res.by_role
    run.n_tasks, run.status, run.finished_at = res.n_tasks, "succeeded", time.time()
    db.commit()
    S.record_metrics(db, project, agent_row, res.metrics, tags={"suite": run.suite, "eval_run_id": run.id, "dataset_id": ds.id})
    out: dict[str, Any] = {"eval_run_id": run.id, "metrics": res.metrics}
    # optional deployment gate
    if job.params.get("gate"):
        policy = GatePolicy.model_validate(job.params.get("gate_policy") or {})
        baseline_row = S.get_deployed(db, project.id, agent_row.name)
        baseline_res = None
        if baseline_row is not None and baseline_row.id != agent_row.id:
            b = (db.query(D.EvalRun).filter_by(project_id=project.id, agent_id=baseline_row.id, dataset_id=ds.id, status="succeeded")
                 .order_by(D.EvalRun.created_at.desc()).first())
            if b is not None:
                baseline_res = _eval_result_from_run(db, b)
        decision = evaluate_gate(res, policy, baseline_res)
        dep = D.Deployment(id=D.uid("dep"), project_id=project.id, agent_name=agent_row.name, agent_id=agent_row.id,
                           baseline_agent_id=baseline_row.id if baseline_row else None, eval_run_id=run.id, policy=policy.model_dump(),
                           decision=decision.model_dump(), promoted=False)
        db.add(dep)
        if decision.passed and job.params.get("auto_promote"):
            promote(db, project.id, agent_row)
            dep.promoted = dep.active = True
        db.commit()
        out["gate"] = decision.model_dump()
        out["deployment_id"] = dep.id
    log("done", 1.0)
    return out


def _eval_result_from_run(db: Session, run: D.EvalRun) -> EvalResult:
    rows = db.query(D.RolloutRow).filter_by(eval_run_id=run.id).all()
    from ..evaluation.runner import RolloutRecord

    agent = db.get(D.Agent, run.agent_id)
    return EvalResult(agent=agent.name, agent_version=agent.version, suite=run.suite, n_tasks=run.n_tasks, n_rollouts=len(rows), k=run.k,
                      wall_time_s=0.0, metrics=run.metrics, by_family=run.by_family, by_env=run.by_env, by_difficulty=run.by_difficulty,
                      per_rollout=[RolloutRecord.model_validate(r.record) for r in rows])


def promote(db: Session, project_id: str, agent: D.Agent) -> None:
    for a in db.query(D.Agent).filter_by(project_id=project_id, name=agent.name, status="deployed"):
        a.status = "retired"
    for d in db.query(D.Deployment).filter_by(project_id=project_id, agent_name=agent.name, active=True):
        d.active = False
    agent.status = "deployed"
    db.commit()


# ---------------------------------------------------------------------------
@handler("training")
def run_training(db: Session, job: D.Job) -> dict[str, Any]:
    log = JobLog(db, job)
    run = db.get(D.TrainingRun, job.params["training_run_id"])
    project = db.get(D.Project, job.project_id)
    agent_row = db.get(D.Agent, run.agent_id)
    cfg = AgentConfig.model_validate(agent_row.config)
    p = run.params or {}
    run.status = "running"
    db.commit()
    art = settings.artifacts_path / project.name / cfg.name / run.id
    art.mkdir(parents=True, exist_ok=True)
    algo = run.algorithm
    log(f"training run {run.id}: {algo} from {cfg.name}/{cfg.version}", 0.02)

    def _publish(new_cfg: AgentConfig, local_dir: Path) -> AgentConfig:
        """Mirror artifacts to the configured object store and point the config at the remote URIs."""
        from ..sdk.artifacts import publish_dir, remap

        mapping = publish_dir(local_dir, f"{project.name}/{cfg.name}/{run.id}/{local_dir.name}")
        if not mapping:
            return new_cfg
        new_cfg = new_cfg.model_copy(update={"tool_router": remap(new_cfg.tool_router, mapping), "exemplar_store": remap(new_cfg.exemplar_store, mapping)})
        for r in new_cfg.roles.values():
            r.tool_router, r.exemplar_store = remap(r.tool_router, mapping), remap(r.exemplar_store, mapping)
        return new_cfg

    def _new_version(new_cfg: AgentConfig, origin: str, parent: D.Agent, local_dir: Optional[Path] = None) -> D.Agent:
        if local_dir is not None:
            new_cfg = _publish(new_cfg, local_dir)
        new_cfg = new_cfg.model_copy(update={"version": S.next_version(db, project.id, cfg.name)})
        a = S.create_agent_version(db, project, new_cfg, parent_id=parent.id, origin=origin)
        log(f"created agent version {a.name}/{a.version} ({a.id})")
        return a

    if algo == "online":
        train_ds, eval_ds = db.get(D.Dataset, p["train_dataset_id"]), db.get(D.Dataset, p["eval_dataset_id"])
        train_tasks, eval_tasks = S.tasks_from_dataset(train_ds), S.tasks_from_dataset(eval_ds)
        iterations = int(p.get("iterations", 5))
        state = {"parent": agent_row, "last": agent_row}
        start_version = S.next_version(db, project.id, cfg.name)

        def _on_rollout(ro, task, rewards):
            S.persist_rollout(db, project, ro, task, rewards, agent_id=state["last"].id, training_run_id=run.id)

        def _on_iter(rec: dict[str, Any]):
            new_cfg = AgentConfig.model_validate(json.loads((Path(rec["artifact_dir"]) / "agent.json").read_text()))
            if rec["iteration"] == 0:
                a = agent_row
            else:
                a = _new_version(new_cfg, "online", state["last"], local_dir=Path(rec["artifact_dir"]))
                state["last"] = a
            S.record_metrics(db, project, a, rec["eval"], tags={"training_run_id": run.id, "iteration": rec["iteration"]})
            run.history = (run.history or []) + [{k: v for k, v in rec.items() if k != "updates"} | {"agent_id": a.id, "updates": _slim(rec["updates"])}]
            log(f"iteration {rec['iteration']}: task_success={rec['eval'].get('task_success', 0):.3f}", 0.05 + 0.9 * rec["iteration"] / max(1, iterations))
            db.commit()

        weight_fn = None
        if p.get("weight_update_every"):
            weight_fn = _make_weight_update(p, log)
        loop = OnlineLoop(cfg, train_tasks, eval_tasks, artifacts_dir=art, batch_size=int(p.get("batch_size", 16)),
                          learn_router=bool(p.get("learn_router", True)), learn_exemplars=bool(p.get("learn_exemplars", True)),
                          learn_prompt_every=int(p.get("learn_prompt_every", 0)), weight_update_every=int(p.get("weight_update_every", 0)),
                          weight_update=weight_fn, seed=int(p.get("seed", 0)), on_iteration=_on_iter, on_rollout=_on_rollout,
                          eval_k=int(p.get("eval_k", 1)), optimize_roles=p.get("optimize_roles"),
                          rollout_backend=p.get("rollout_backend"), rollout_workers=p.get("rollout_workers"))
        # the loop numbers versions v1..vN internally; align with the project's version counter
        loop.version_no = int(start_version[1:]) - 1
        hist = loop.run(iterations)
        run.output_agent_id = state["last"].id
        run.result = {"iterations": len(hist) - 1, "baseline": hist[0]["eval"], "final": hist[-1]["eval"],
                      "improvement": {k: hist[-1]["eval"].get(k, 0) - hist[0]["eval"].get(k, 0) for k in ("task_success", "tool_selection_f1", "context_preservation")}}
    elif algo in ("router", "exemplars", "signals", "sft", "dpo", "grpo"):
        rollouts = S.load_rollouts(db, project.id, agent_name=cfg.name, limit=int(p.get("max_rollouts", 2000)))
        log(f"loaded {len(rollouts)} stored rollouts for agent '{cfg.name}'", 0.1)
        if not rollouts:
            raise RuntimeError("no stored rollouts for this agent; run an evaluation or online loop first")
        envs = {e: get_environment(e) for e in {r.env_name for r in rollouts}}
        tools_by_env = {k: v.tools.specs() for k, v in envs.items()}
        if algo in ("router", "exemplars") and cfg.is_multi_agent:
            from ..training.multi_agent import train_roles

            new_cfg, r = train_roles(cfg, rollouts, envs, art, roles=p.get("optimize_roles"), learn_router=algo == "router",
                                     learn_exemplars=algo == "exemplars")
            a = _new_version(new_cfg, algo, agent_row, local_dir=art)
            run.output_agent_id, run.result = a.id, _slim(r)
        elif algo == "router":
            r = train_router(rollouts, envs, art / "router")
            a = _new_version(cfg.model_copy(update={"tool_router": r["artifact"], "router_top_k": int(p.get("router_top_k", cfg.router_top_k or 8))}), "router", agent_row, local_dir=art)
            run.output_agent_id, run.result = a.id, r
        elif algo == "exemplars":
            r = update_exemplars(rollouts, art / "exemplars.json")
            a = _new_version(cfg.model_copy(update={"exemplar_store": r["artifact"], "exemplar_k": int(p.get("exemplar_k", 2))}), "exemplars", agent_row, local_dir=art)
            run.output_agent_id, run.result = a.id, r
        else:
            task_ids = {r.task_id for r in rollouts}
            tasks: dict[str, TaskSpec] = {}
            for ds in db.query(D.Dataset).filter_by(project_id=project.id):
                for t in ds.tasks:
                    if t["id"] in task_ids:
                        tasks[t["id"]] = TaskSpec.model_validate(t)
            role = p.get("role")  # multi-agent: restrict datasets to one role's steps
            files = {"sft": export_jsonl(sft_examples(rollouts, tools_by_env, role=role), art / "sft.jsonl"),
                     "dpo": export_jsonl(preference_pairs(rollouts, tools_by_env, role=role), art / "dpo.jsonl"),
                     "grpo": export_jsonl(grpo_prompts(rollouts, tasks, tools_by_env, role=role), art / "grpo.jsonl"),
                     "step_rewards": export_jsonl(step_rewards(rollouts), art / "step_rewards.jsonl")}
            counts = {k: sum(1 for _ in open(v)) for k, v in files.items()}
            log(f"signals: {counts}", 0.3)
            run.result = {"files": files, "counts": counts}
            if algo in ("sft", "dpo", "grpo"):
                from ..signals.generate import load_jsonl
                from ..training import trl_trainers as T

                rows = load_jsonl(files[algo])
                if not rows:
                    raise RuntimeError(f"no {algo} rows could be generated from stored rollouts")
                kw = {k: v for k, v in p.items() if k in ("model", "epochs", "lr", "lora_r", "max_steps", "batch_size", "num_generations",
                                                            "max_completion_length", "beta", "max_length")}
                kw.setdefault("model", T.DEFAULT_TINY)
                log(f"{algo}: {len(rows)} rows, model={kw['model']}", 0.4)
                r = getattr(T, algo)(rows, art / algo, **kw)
                run.result.update(r)
                a = _new_version(cfg.model_copy(update={"model": r["model"]}), algo, agent_row)
                run.output_agent_id = a.id
    elif algo == "prompt_opt":
        ds = db.get(D.Dataset, p["train_dataset_id"])
        r = optimize_prompt(cfg, S.tasks_from_dataset(ds), reflection_model=p.get("reflection_model", "mock"),
                            iterations=int(p.get("iterations", 4)), minibatch=int(p.get("minibatch", 8)), seed=int(p.get("seed", 0)))
        a = _new_version(cfg.model_copy(update={"system_prompt": r["best_prompt"]}), "prompt_opt", agent_row)
        run.output_agent_id, run.result = a.id, {k: v for k, v in r.items()}
    else:
        raise ValueError(f"unknown algorithm {algo}")
    run.status, run.finished_at = "succeeded", time.time()
    db.commit()
    log("done", 1.0)
    return {"training_run_id": run.id, "output_agent_id": run.output_agent_id, "result": _slim(run.result)}


def _slim(d: Any) -> Any:
    try:
        s = json.dumps(d, default=str)
    except Exception:
        return str(d)[:2000]
    return json.loads(s) if len(s) < 20000 else {"_truncated": s[:2000]}


def _make_weight_update(p: dict[str, Any], log: JobLog):
    from ..training import trl_trainers as T

    algo = p.get("weight_algorithm", "sft")

    def _fn(buffer: list[Rollout], cfg: AgentConfig, vdir: Path) -> dict[str, Any]:
        envs = {e: get_environment(e) for e in {r.env_name for r in buffer}}
        tools_by_env = {k: v.tools.specs() for k, v in envs.items()}
        rows = sft_examples(buffer, tools_by_env) if algo == "sft" else preference_pairs(buffer, tools_by_env)
        if not rows:
            return {"skipped": "no rows"}
        log(f"weight update ({algo}) on {len(rows)} rows")
        base = cfg.model[3:] if cfg.model.startswith("hf:") else p.get("model", T.DEFAULT_TINY)
        return getattr(T, algo)(rows, vdir / algo, model=base, max_steps=int(p.get("max_steps", 20)), lora_r=int(p.get("lora_r", 0)))

    return _fn


@handler("retention")
def run_retention_job(db: Session, job: D.Job) -> dict[str, Any]:
    from .retention import run_retention

    return run_retention(db, (job.params or {}).get("days"), dry_run=bool((job.params or {}).get("dry_run")))


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------
def claim_job(db: Session, worker: str) -> Optional[D.Job]:
    job = db.query(D.Job).filter_by(status="pending").order_by(D.Job.created_at).first()
    if job is None:
        return None
    res = db.execute(update(D.Job).where(D.Job.id == job.id, D.Job.status == "pending")
                     .values(status="running", worker=worker, started_at=time.time()))
    db.commit()
    if res.rowcount != 1:
        return None
    db.refresh(job)
    return job


def execute_job(job_id: str) -> None:
    db = D.session()
    try:
        job = db.get(D.Job, job_id)
        try:
            _init_tracing(job.project_id)
            fn = HANDLERS[job.type]
            job.result = fn(db, job)
            job.status = "succeeded"
        except Exception as e:  # noqa: BLE001
            db.rollback()
            job = db.get(D.Job, job_id)
            job.status = "failed"
            job.error = f"{type(e).__name__}: {e}\n{traceback.format_exc()[-3000:]}"
            for model, key in ((D.EvalRun, "eval_run_id"), (D.TrainingRun, "training_run_id")):
                rid = (job.params or {}).get(key)
                if rid:
                    row = db.get(model, rid)
                    if row is not None:
                        row.status = "failed"
        from ..sdk import tracing

        tracing.flush()  # spans must be persisted before the job is reported as finished
        job.finished_at = time.time()
        db.commit()
    finally:
        db.close()


def submit_job(db: Session, project_id: str, type_: str, params: dict[str, Any]) -> D.Job:
    job = D.Job(id=D.uid("job"), project_id=project_id, type=type_, params=params)
    db.add(job)
    db.commit()
    from .auth import meter

    proj = db.get(D.Project, project_id)
    meter(db, proj.org_id if proj else None, "jobs", 1)
    if settings.inline_jobs:
        _run_inline(job.id)
    return job


def _run_inline(job_id: str) -> None:
    def _t():
        db = D.session()
        try:
            j = claim_job_by_id(db, job_id)
        finally:
            db.close()
        if j:
            execute_job(job_id)

    threading.Thread(target=_t, daemon=True).start()


def claim_job_by_id(db: Session, job_id: str) -> bool:
    res = db.execute(update(D.Job).where(D.Job.id == job_id, D.Job.status == "pending")
                     .values(status="running", worker=f"inline@{socket.gethostname()}", started_at=time.time()))
    db.commit()
    return res.rowcount == 1


def worker_loop(poll_s: Optional[float] = None, once: bool = False) -> int:
    name = f"worker@{socket.gethostname()}:{threading.get_ident()}"
    poll_s = poll_s or settings.worker_poll_s
    n = 0
    while True:
        db = D.session()
        try:
            job = claim_job(db, name)
        finally:
            db.close()
        if job is None:
            if once:
                return n
            time.sleep(poll_s)
            continue
        execute_job(job.id)
        n += 1
        if once:
            return n
