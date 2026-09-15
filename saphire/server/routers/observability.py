"""Traces (OTel ingest), rollouts, scores."""
from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field
from sqlalchemy import func
from sqlalchemy.orm import Session

from ...sdk.types import Reward, Rollout, TaskSpec
from .. import db as D
from .. import service as S
from ..auth import Principal, check_quota, get_principal, meter
from ..deps import get_or_404, get_project, require_api_key

router = APIRouter(dependencies=[Depends(require_api_key)])


# ---------------- traces ----------------
class IngestIn(BaseModel):
    project: Optional[str] = None
    spans: list[dict[str, Any]]


@router.post("/traces/ingest")
def ingest(body: IngestIn, p: Principal = Depends(get_principal), db: Session = Depends(D.get_db)):
    from ..config import settings

    project = D.ensure_project(db, body.project or settings.default_project, org_id=(p.org or D.ensure_org(db)).id)
    out = S.ingest_spans(db, project, body.spans)
    meter(db, project.org_id, "spans", out["spans"])
    return out


@router.get("/traces")
def list_traces(limit: int = Query(50, le=500), offset: int = 0, status: Optional[str] = None, kind: Optional[str] = None,
                project: D.Project = Depends(get_project), db: Session = Depends(D.get_db)):
    q = db.query(D.Trace).filter_by(project_id=project.id)
    if status:
        q = q.filter_by(status=status)
    total = q.count()
    rows = q.order_by(D.Trace.start_time.desc().nullslast()).offset(offset).limit(limit).all()
    out = []
    for t in rows:
        d = D.to_dict(t)
        d["duration_ms"] = (t.end_time - t.start_time) * 1000 if t.start_time and t.end_time else None
        out.append(d)
    return {"total": total, "items": out}


@router.get("/traces/{trace_id}")
def get_trace(trace_id: str, db: Session = Depends(D.get_db)):
    t = get_or_404(db, D.Trace, trace_id)
    spans = db.query(D.Span).filter_by(trace_id=trace_id).order_by(D.Span.start_time).all()
    scores = db.query(D.Score).filter_by(trace_id=trace_id).all()
    return D.to_dict(t) | {"spans": [D.to_dict(s) for s in spans], "scores": [D.to_dict(s) for s in scores]}


# ---------------- rollouts ----------------
class RolloutIn(BaseModel):
    rollout: Rollout
    task: TaskSpec
    rewards: list[Reward] = Field(default_factory=list)
    agent_id: Optional[str] = None
    verify: bool = True  # run the environment verifier server-side when it can be reconstructed


@router.post("/rollouts", status_code=201)
def create_rollout(body: RolloutIn, project: D.Project = Depends(get_project), db: Session = Depends(D.get_db)):
    """Bring-your-own-agent ingestion: store a rollout produced by the SDK's RolloutRecorder."""
    check_quota(db, db.get(D.Organization, project.org_id), "rollouts_per_month")
    rewards = list(body.rewards)
    row = S.persist_rollout(db, project, body.rollout, body.task, rewards, agent_id=body.agent_id)
    return {"id": row.id, "task_success": row.task_success, "n_rewards": len(rewards)}


@router.get("/rollouts")
def list_rollouts(limit: int = Query(50, le=500), offset: int = 0, agent_id: Optional[str] = None, eval_run_id: Optional[str] = None,
                  training_run_id: Optional[str] = None, env: Optional[str] = None, success: Optional[bool] = None,
                  project: D.Project = Depends(get_project), db: Session = Depends(D.get_db)):
    q = db.query(D.RolloutRow).filter_by(project_id=project.id)
    if agent_id:
        q = q.filter_by(agent_id=agent_id)
    if eval_run_id:
        q = q.filter_by(eval_run_id=eval_run_id)
    if training_run_id:
        q = q.filter_by(training_run_id=training_run_id)
    if env:
        q = q.filter_by(env_name=env)
    if success is not None:
        q = q.filter(D.RolloutRow.task_success >= 1.0) if success else q.filter(D.RolloutRow.task_success < 1.0)
    total = q.count()
    rows = q.order_by(D.RolloutRow.created_at.desc()).offset(offset).limit(limit).all()
    items = []
    for r in rows:
        d = D.to_dict(r)
        d.pop("payload")
        items.append(d)
    return {"total": total, "items": items}


@router.get("/rollouts/{rollout_id}")
def get_rollout(rollout_id: str, db: Session = Depends(D.get_db)):
    r = get_or_404(db, D.RolloutRow, rollout_id)
    scores = db.query(D.Score).filter_by(rollout_id=rollout_id).all()
    return D.to_dict(r) | {"scores": [D.to_dict(s) for s in scores]}


# ---------------- scores ----------------
class ScoreIn(BaseModel):
    name: str
    value: float
    trace_id: Optional[str] = None
    rollout_id: Optional[str] = None
    source: str = "api"
    rationale: str = ""
    step_index: Optional[int] = None


@router.post("/scores", status_code=201)
def create_score(body: ScoreIn, project: D.Project = Depends(get_project), db: Session = Depends(D.get_db)):
    """Attach a human / external / judge score to a trace or rollout — a training signal."""
    s = D.Score(id=D.uid("sc"), project_id=project.id, **body.model_dump())
    db.add(s)
    db.commit()
    return D.to_dict(s)


@router.post("/scores/batch", status_code=201)
def create_scores(body: list[ScoreIn], project: D.Project = Depends(get_project), db: Session = Depends(D.get_db)):
    ids = []
    for b in body:
        s = D.Score(id=D.uid("sc"), project_id=project.id, **b.model_dump())
        db.add(s)
        ids.append(s.id)
    db.commit()
    return {"ids": ids}


@router.get("/scores")
def list_scores(name: Optional[str] = None, limit: int = Query(200, le=5000), project: D.Project = Depends(get_project),
                db: Session = Depends(D.get_db)):
    q = db.query(D.Score).filter_by(project_id=project.id)
    if name:
        q = q.filter_by(name=name)
    return [D.to_dict(s) for s in q.order_by(D.Score.created_at.desc()).limit(limit).all()]


@router.get("/scores/summary")
def score_summary(project: D.Project = Depends(get_project), db: Session = Depends(D.get_db)):
    rows = (db.query(D.Score.name, func.count(D.Score.id), func.avg(D.Score.value)).filter_by(project_id=project.id)
            .group_by(D.Score.name).all())
    return [{"name": n, "count": c, "mean": float(a or 0)} for n, c, a in rows]
