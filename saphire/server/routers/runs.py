"""Evaluation runs, training runs, jobs, deployments, experiments, metrics."""
from __future__ import annotations

import random
import time
from collections import defaultdict
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ...evaluation import metrics as M
from ...evaluation.gates import GatePolicy
from .. import db as D
from .. import jobs as J
from ..deps import get_or_404, get_project, require_api_key

router = APIRouter(dependencies=[Depends(require_api_key)])


# ---------------- evals ----------------
class EvalIn(BaseModel):
    agent_id: str
    dataset_id: str
    k: int = 1
    judge_model: Optional[str] = None  # e.g. "openai/gpt-4o-mini" or "mock"
    concurrency: int = 1
    gate: bool = False
    gate_policy: Optional[GatePolicy] = None
    auto_promote: bool = False


@router.post("/evals", status_code=201)
def create_eval(body: EvalIn, project: D.Project = Depends(get_project), db: Session = Depends(D.get_db)):
    agent = get_or_404(db, D.Agent, body.agent_id)
    ds = get_or_404(db, D.Dataset, body.dataset_id)
    run = D.EvalRun(id=D.uid("eval"), project_id=project.id, agent_id=agent.id, dataset_id=ds.id, suite=ds.suite, k=body.k,
                    judge_model=body.judge_model, n_tasks=len(ds.tasks))
    db.add(run)
    db.commit()
    job = J.submit_job(db, project.id, "eval", {"eval_run_id": run.id, "concurrency": body.concurrency, "gate": body.gate,
                                                "gate_policy": body.gate_policy.model_dump() if body.gate_policy else None,
                                                "auto_promote": body.auto_promote})
    run.job_id = job.id
    db.commit()
    return D.to_dict(run)


@router.get("/evals")
def list_evals(agent_id: Optional[str] = None, agent_name: Optional[str] = None, limit: int = Query(50, le=500),
               project: D.Project = Depends(get_project), db: Session = Depends(D.get_db)):
    q = db.query(D.EvalRun).filter_by(project_id=project.id)
    if agent_id:
        q = q.filter_by(agent_id=agent_id)
    if agent_name:
        ids = [a.id for a in db.query(D.Agent).filter_by(project_id=project.id, name=agent_name)]
        q = q.filter(D.EvalRun.agent_id.in_(ids))
    runs = q.order_by(D.EvalRun.created_at.desc()).limit(limit).all()
    agents = {a.id: a for a in db.query(D.Agent).filter_by(project_id=project.id)}
    out = []
    for r in runs:
        d = D.to_dict(r)
        a = agents.get(r.agent_id)
        d["agent_name"], d["agent_version"] = (a.name, a.version) if a else (None, None)
        out.append(d)
    return out


@router.get("/evals/{eval_id}")
def get_eval(eval_id: str, db: Session = Depends(D.get_db)):
    r = get_or_404(db, D.EvalRun, eval_id)
    a = db.get(D.Agent, r.agent_id)
    job = db.get(D.Job, r.job_id) if r.job_id else None
    rollouts = db.query(D.RolloutRow).filter_by(eval_run_id=eval_id).order_by(D.RolloutRow.created_at).all()
    return D.to_dict(r) | {"agent": D.to_dict(a) if a else None, "job": D.to_dict(job) if job else None,
                           "rollouts": [r_.record | {"id": r_.id, "trace_id": r_.trace_id} for r_ in rollouts]}


@router.get("/evals/{eval_id}/compare/{other_id}")
def compare_evals(eval_id: str, other_id: str, db: Session = Depends(D.get_db)):
    a, b = get_or_404(db, D.EvalRun, eval_id), get_or_404(db, D.EvalRun, other_id)
    ra = J._eval_result_from_run(db, a)
    rb = J._eval_result_from_run(db, b)
    keys = ["task_success", "tool_selection_f1", "step_efficiency", "context_preservation", "tool_error_free", "judge_score"]
    return {"baseline": eval_id, "candidate": other_id, "comparison": M.compare(ra.model_dump(), rb.model_dump(), keys)}


# ---------------- training ----------------
class TrainingIn(BaseModel):
    agent_id: str
    algorithm: str = Field(description="online | router | exemplars | prompt_opt | signals | sft | dpo | grpo")
    params: dict[str, Any] = Field(default_factory=dict)


@router.post("/training", status_code=201)
def create_training(body: TrainingIn, project: D.Project = Depends(get_project), db: Session = Depends(D.get_db)):
    agent = get_or_404(db, D.Agent, body.agent_id)
    if body.algorithm not in ("online", "router", "exemplars", "prompt_opt", "signals", "sft", "dpo", "grpo"):
        raise HTTPException(400, "unknown algorithm")
    if body.algorithm == "online" and not ({"train_dataset_id", "eval_dataset_id"} <= set(body.params)):
        raise HTTPException(400, "online training needs params.train_dataset_id and params.eval_dataset_id")
    if body.algorithm == "prompt_opt" and "train_dataset_id" not in body.params:
        raise HTTPException(400, "prompt_opt needs params.train_dataset_id")
    run = D.TrainingRun(id=D.uid("train"), project_id=project.id, agent_id=agent.id, algorithm=body.algorithm, params=body.params)
    db.add(run)
    db.commit()
    job = J.submit_job(db, project.id, "training", {"training_run_id": run.id})
    run.job_id = job.id
    db.commit()
    return D.to_dict(run)


@router.get("/training")
def list_training(limit: int = Query(50, le=500), project: D.Project = Depends(get_project), db: Session = Depends(D.get_db)):
    runs = db.query(D.TrainingRun).filter_by(project_id=project.id).order_by(D.TrainingRun.created_at.desc()).limit(limit).all()
    agents = {a.id: a for a in db.query(D.Agent).filter_by(project_id=project.id)}
    out = []
    for r in runs:
        d = D.to_dict(r)
        d["history"] = [{k: v for k, v in h.items() if k in ("iteration", "version", "agent_id", "eval", "collect_success", "seconds")} for h in (r.history or [])]
        a = agents.get(r.agent_id)
        d["agent_name"] = a.name if a else None
        d["input_version"] = a.version if a else None
        o = agents.get(r.output_agent_id) if r.output_agent_id else None
        d["output_version"] = o.version if o else None
        out.append(d)
    return out


@router.get("/training/{run_id}")
def get_training(run_id: str, db: Session = Depends(D.get_db)):
    r = get_or_404(db, D.TrainingRun, run_id)
    job = db.get(D.Job, r.job_id) if r.job_id else None
    return D.to_dict(r) | {"job": D.to_dict(job) if job else None}


# ---------------- jobs ----------------
@router.get("/jobs")
def list_jobs(status: Optional[str] = None, limit: int = Query(50, le=500), project: D.Project = Depends(get_project),
              db: Session = Depends(D.get_db)):
    q = db.query(D.Job).filter_by(project_id=project.id)
    if status:
        q = q.filter_by(status=status)
    return [D.to_dict(j) for j in q.order_by(D.Job.created_at.desc()).limit(limit).all()]


@router.get("/jobs/{job_id}")
def get_job(job_id: str, db: Session = Depends(D.get_db)):
    return D.to_dict(get_or_404(db, D.Job, job_id))


@router.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str, db: Session = Depends(D.get_db)):
    j = get_or_404(db, D.Job, job_id)
    if j.status == "pending":
        j.status = "cancelled"
        db.commit()
    return D.to_dict(j)


# ---------------- deployments ----------------
@router.get("/deployments")
def list_deployments(project: D.Project = Depends(get_project), db: Session = Depends(D.get_db)):
    rows = db.query(D.Deployment).filter_by(project_id=project.id).order_by(D.Deployment.created_at.desc()).limit(100).all()
    agents = {a.id: a for a in db.query(D.Agent).filter_by(project_id=project.id)}
    out = []
    for r in rows:
        d = D.to_dict(r)
        a = agents.get(r.agent_id)
        d["agent_version"] = a.version if a else None
        b = agents.get(r.baseline_agent_id) if r.baseline_agent_id else None
        d["baseline_version"] = b.version if b else None
        out.append(d)
    return out


@router.post("/deployments/{deployment_id}/promote")
def promote_deployment(deployment_id: str, db: Session = Depends(D.get_db)):
    d = get_or_404(db, D.Deployment, deployment_id)
    a = get_or_404(db, D.Agent, d.agent_id)
    J.promote(db, d.project_id, a)
    d.promoted = d.active = True
    db.commit()
    return D.to_dict(d)


@router.get("/deployments/current")
def current_deployments(project: D.Project = Depends(get_project), db: Session = Depends(D.get_db)):
    return [D.to_dict(a) for a in db.query(D.Agent).filter_by(project_id=project.id, status="deployed").all()]


# ---------------- experiments (online A/B) ----------------
class VariantIn(BaseModel):
    name: str
    agent_id: str
    weight: float = 1.0


class ExperimentIn(BaseModel):
    name: str
    variants: list[VariantIn]


class OutcomeIn(BaseModel):
    unit: str  # user/session id
    value: float
    variant: Optional[str] = None  # optional: server re-derives from unit hash if omitted


@router.post("/experiments", status_code=201)
def create_experiment(body: ExperimentIn, project: D.Project = Depends(get_project), db: Session = Depends(D.get_db)):
    for v in body.variants:
        get_or_404(db, D.Agent, v.agent_id)
    e = D.Experiment(id=D.uid("exp"), project_id=project.id, name=body.name, variants=[v.model_dump() for v in body.variants])
    db.add(e)
    db.commit()
    return D.to_dict(e)


@router.get("/experiments")
def list_experiments(project: D.Project = Depends(get_project), db: Session = Depends(D.get_db)):
    return [_exp_summary(e, db) for e in db.query(D.Experiment).filter_by(project_id=project.id).order_by(D.Experiment.created_at.desc())]


def _assign(e: D.Experiment, unit: str) -> dict[str, Any]:
    rng = random.Random(f"{e.id}:{unit}")
    total = sum(v["weight"] for v in e.variants)
    x = rng.random() * total
    for v in e.variants:
        x -= v["weight"]
        if x <= 0:
            return v
    return e.variants[-1]


@router.get("/experiments/{exp_id}/assign")
def assign_variant(exp_id: str, unit: str, db: Session = Depends(D.get_db)):
    """Deterministic, sticky assignment of a traffic unit to a variant (SDK calls this at request time)."""
    e = get_or_404(db, D.Experiment, exp_id)
    v = _assign(e, unit)
    a = db.get(D.Agent, v["agent_id"])
    return {"variant": v["name"], "agent_id": v["agent_id"], "config": a.config if a else None}


@router.post("/experiments/{exp_id}/outcomes", status_code=201)
def record_outcome(exp_id: str, body: OutcomeIn, db: Session = Depends(D.get_db)):
    e = get_or_404(db, D.Experiment, exp_id)
    variant = body.variant or _assign(e, body.unit)["name"]
    e.outcomes = list(e.outcomes) + [{"variant": variant, "unit": body.unit, "value": body.value, "ts": time.time()}]
    db.commit()
    return {"variant": variant, "n": len(e.outcomes)}


@router.get("/experiments/{exp_id}")
def get_experiment(exp_id: str, db: Session = Depends(D.get_db)):
    return _exp_summary(get_or_404(db, D.Experiment, exp_id), db)


def _exp_summary(e: D.Experiment, db: Session) -> dict[str, Any]:
    by: dict[str, list[float]] = defaultdict(list)
    for o in e.outcomes:
        by[o["variant"]].append(o["value"])
    results = {}
    names = [v["name"] for v in e.variants]
    for n in names:
        vals = by.get(n, [])
        results[n] = {"n": len(vals), "mean": M.mean(vals), "ci": M.bootstrap_ci(vals) if vals else (0, 0)}
    if len(names) >= 2 and by.get(names[0]) and by.get(names[1]):
        results["p_value"] = M.permutation_test(by[names[0]], by[names[1]])
        results["delta"] = M.mean(by[names[1]]) - M.mean(by[names[0]])
    d = D.to_dict(e)
    d.pop("outcomes")
    d["n_outcomes"] = len(e.outcomes)
    d["results"] = results
    return d


@router.post("/experiments/{exp_id}/stop")
def stop_experiment(exp_id: str, db: Session = Depends(D.get_db)):
    e = get_or_404(db, D.Experiment, exp_id)
    e.status = "stopped"
    db.commit()
    return _exp_summary(e, db)


# ---------------- metrics ----------------
@router.get("/metrics/timeseries")
def timeseries(agent_name: str, name: str = "task_success", limit: int = Query(500, le=5000),
               project: D.Project = Depends(get_project), db: Session = Depends(D.get_db)):
    rows = (db.query(D.MetricPoint).filter_by(project_id=project.id, agent_name=agent_name, name=name)
            .order_by(D.MetricPoint.ts).limit(limit).all())
    agents = {a.id: a.version for a in db.query(D.Agent).filter_by(project_id=project.id, name=agent_name)}
    return [{"ts": r.ts, "value": r.value, "agent_id": r.agent_id, "version": agents.get(r.agent_id), "tags": r.tags} for r in rows]


@router.get("/metrics/overview")
def overview(project: D.Project = Depends(get_project), db: Session = Depends(D.get_db)):
    """Dashboard summary: counts, deployed agents, latest eval per agent name, recent improvement."""
    agents = db.query(D.Agent).filter_by(project_id=project.id).all()
    n_traces = db.query(D.Trace).filter_by(project_id=project.id).count()
    n_rollouts = db.query(D.RolloutRow).filter_by(project_id=project.id).count()
    n_success = db.query(D.RolloutRow).filter(D.RolloutRow.project_id == project.id, D.RolloutRow.task_success >= 1.0).count()
    jobs = db.query(D.Job).filter_by(project_id=project.id).order_by(D.Job.created_at.desc()).limit(10).all()
    latest: dict[str, Any] = {}
    for name in sorted({a.name for a in agents}):
        ids = [a.id for a in agents if a.name == name]
        runs = (db.query(D.EvalRun).filter(D.EvalRun.project_id == project.id, D.EvalRun.agent_id.in_(ids), D.EvalRun.status == "succeeded")
                .order_by(D.EvalRun.created_at).all())
        pts = (db.query(D.MetricPoint).filter_by(project_id=project.id, agent_name=name, name="task_success").order_by(D.MetricPoint.ts).all())
        deployed = next((a for a in agents if a.name == name and a.status == "deployed"), None)
        latest[name] = {
            "versions": len(ids), "deployed_version": deployed.version if deployed else None, "deployed_agent_id": deployed.id if deployed else None,
            "latest_eval": D.to_dict(runs[-1]) if runs else None,
            "first_task_success": pts[0].value if pts else None, "latest_task_success": pts[-1].value if pts else None,
            "n_points": len(pts),
        }
    return {"project": D.to_dict(project), "n_agents": len(agents), "n_traces": n_traces, "n_rollouts": n_rollouts,
            "rollout_success_rate": (n_success / n_rollouts) if n_rollouts else None,
            "n_datasets": db.query(D.Dataset).filter_by(project_id=project.id).count(),
            "n_eval_runs": db.query(D.EvalRun).filter_by(project_id=project.id).count(),
            "n_training_runs": db.query(D.TrainingRun).filter_by(project_id=project.id).count(),
            "agents": latest, "recent_jobs": [D.to_dict(j) for j in jobs]}
