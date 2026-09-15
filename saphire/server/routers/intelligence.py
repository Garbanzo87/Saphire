"""Production intelligence API + webhooks management."""
from __future__ import annotations

import time
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ...intelligence.mining import mine_tasks_from_rollouts
from ...intelligence.report import _human_scores, _rows, build_report
from ...sdk.types import Rollout
from .. import db as D
from .. import jobs as J
from ..auth import Principal, audit, check_quota, require
from ..deps import get_or_404, get_project, require_api_key
from ..webhooks import EVENTS, deliver

router = APIRouter(dependencies=[Depends(require_api_key)])


# ---------------- reports ----------------
class IntelIn(BaseModel):
    agent_name: Optional[str] = None
    recent_hours: float = 24
    baseline_hours: float = 168
    production_only: bool = False  # only rollouts that were not produced by eval/training jobs
    sync: bool = False  # compute inline instead of as a job (small projects / dashboards)


@router.post("/intelligence/run", status_code=201)
def run_intelligence(body: IntelIn, project: D.Project = Depends(get_project), p: Principal = Depends(require("write")), db: Session = Depends(D.get_db)):
    if body.sync:
        from ...intelligence.report import persist_report

        rep = build_report(db, project, body.agent_name, body.recent_hours, body.baseline_hours, body.production_only)
        row = persist_report(db, project, rep)
        return D.to_dict(row)
    check_quota(db, db.get(D.Organization, project.org_id), "jobs_per_day")
    job = J.submit_job(db, project.id, "intelligence", body.model_dump())
    return D.to_dict(job)


@router.get("/intelligence/reports")
def list_reports(limit: int = Query(20, le=200), project: D.Project = Depends(get_project), db: Session = Depends(D.get_db)):
    rows = db.query(D.IntelligenceReport).filter_by(project_id=project.id).order_by(D.IntelligenceReport.created_at.desc()).limit(limit).all()
    return [{k: v for k, v in D.to_dict(r).items() if k not in ("drift", "failures", "coverage")} | {"n_alerts": len((r.drift or {}).get("alerts", [])),
             "n_recommendations": len(r.recommendations or [])} for r in rows]


@router.get("/intelligence/latest")
def latest_report(project: D.Project = Depends(get_project), db: Session = Depends(D.get_db)):
    r = db.query(D.IntelligenceReport).filter_by(project_id=project.id).order_by(D.IntelligenceReport.created_at.desc()).first()
    if r is None:
        raise HTTPException(404, "no intelligence report yet — POST /v1/intelligence/run")
    return D.to_dict(r)


@router.get("/intelligence/reports/{report_id}")
def get_report(report_id: str, db: Session = Depends(D.get_db)):
    return D.to_dict(get_or_404(db, D.IntelligenceReport, report_id))


# ---------------- task mining ----------------
class MineIn(BaseModel):
    name: str = "mined from production"
    agent_name: Optional[str] = None
    since_hours: float = 24 * 30
    min_reward: float = 0.99
    use_human_scores: bool = True
    production_only: bool = False
    max_tasks: int = 500
    split: str = "eval"


@router.post("/intelligence/mine-tasks", status_code=201)
def mine_tasks(body: MineIn, project: D.Project = Depends(get_project), p: Principal = Depends(require("write")), db: Session = Depends(D.get_db)):
    now = time.time()
    rows = _rows(db, project.id, body.agent_name, now - body.since_hours * 3600, now + 1, body.production_only)
    ros = [Rollout.model_validate(r.payload) for r in rows]
    human = _human_scores(db, [r.id for r in rows]) if body.use_human_scores else None
    tasks = mine_tasks_from_rollouts(ros, min_reward=body.min_reward, human_scores=human or None, max_tasks=body.max_tasks)
    if not tasks:
        raise HTTPException(400, "no successful production rollouts to mine in this window")
    ds = D.Dataset(id=D.uid("ds"), project_id=project.id, name=body.name, suite="mined", tasks=[t.model_dump() for t in tasks], split=body.split)
    db.add(ds)
    db.commit()
    audit(db, p, "intelligence.mine_tasks", "dataset", ds.id, details={"n_tasks": len(tasks), "from_rollouts": len(rows)})
    return {"dataset_id": ds.id, "n_tasks": len(tasks), "from_rollouts": len(rows), "families": sorted({t.difficulty for t in tasks})}


# ---------------- one-click actions ----------------
class ApplyIn(BaseModel):
    recommendation: dict[str, Any]
    agent_id: Optional[str] = None
    train_dataset_id: Optional[str] = None
    eval_dataset_id: Optional[str] = None


@router.post("/intelligence/apply", status_code=201)
def apply_recommendation(body: ApplyIn, project: D.Project = Depends(get_project), p: Principal = Depends(require("write")), db: Session = Depends(D.get_db)):
    """Execute a recommendation's action: create the training run / mine tasks / attribution job it describes."""
    action = body.recommendation.get("action") or {}
    kind = action.get("kind")
    if kind == "mine_tasks":
        return mine_tasks(MineIn(), project, p, db)
    if kind in ("training", "attribution"):
        if not body.agent_id:
            agent = db.query(D.Agent).filter_by(project_id=project.id, status="deployed").order_by(D.Agent.created_at.desc()).first()
            if agent is None:
                raise HTTPException(400, "agent_id required (no deployed agent to default to)")
        else:
            agent = get_or_404(db, D.Agent, body.agent_id)
        check_quota(db, db.get(D.Organization, project.org_id), "jobs_per_day")
        if kind == "attribution":
            ds_id = body.eval_dataset_id or _latest_dataset(db, project)
            job = J.submit_job(db, project.id, "attribution", {"agent_id": agent.id, "dataset_id": ds_id, "reference_model": action.get("params", {}).get("reference_model") or "mock",
                                                                 "degraded_model": action.get("params", {}).get("degraded_model"), "shapley": True, "n_permutations": 4})
            return {"job_id": job.id, "kind": "attribution"}
        params = dict(action.get("params") or {})
        algo = action.get("algorithm", "online")
        if algo == "online":
            params.setdefault("train_dataset_id", body.train_dataset_id or _latest_dataset(db, project, split="train") or _latest_dataset(db, project))
            params.setdefault("eval_dataset_id", body.eval_dataset_id or _latest_dataset(db, project))
        if algo == "prompt_opt":
            params.setdefault("train_dataset_id", body.train_dataset_id or _latest_dataset(db, project))
        run = D.TrainingRun(id=D.uid("train"), project_id=project.id, agent_id=agent.id, algorithm=algo, params=params)
        db.add(run)
        db.commit()
        job = J.submit_job(db, project.id, "training", {"training_run_id": run.id})
        run.job_id = job.id
        db.commit()
        audit(db, p, "intelligence.apply", "training_run", run.id, details={"algorithm": algo, "title": body.recommendation.get("title")})
        return {"training_run_id": run.id, "job_id": job.id, "kind": "training", "algorithm": algo, "params": params}
    raise HTTPException(400, f"recommendation kind '{kind}' is informational; nothing to execute")


def _latest_dataset(db: Session, project: D.Project, split: Optional[str] = None) -> Optional[str]:
    q = db.query(D.Dataset).filter_by(project_id=project.id)
    if split:
        q = q.filter_by(split=split)
    d = q.order_by(D.Dataset.created_at.desc()).first()
    return d.id if d else None


# ---------------- webhooks ----------------
class WebhookIn(BaseModel):
    url: str
    events: list[str] = Field(default_factory=list)
    description: str = ""
    secret: Optional[str] = None


@router.get("/orgs/current/webhooks")
def list_webhooks(p: Principal = Depends(require("keys")), db: Session = Depends(D.get_db)):
    rows = db.query(D.Webhook).filter_by(org_id=p.org.id).order_by(D.Webhook.created_at.desc()).all()
    out = []
    for w in rows:
        last = db.query(D.WebhookDelivery).filter_by(webhook_id=w.id).order_by(D.WebhookDelivery.created_at.desc()).first()
        d = {k: v for k, v in D.to_dict(w).items() if k != "secret"}
        d["last_delivery"] = D.to_dict(last) if last else None
        out.append(d)
    return out


@router.get("/orgs/current/webhooks/events")
def webhook_events():
    return EVENTS


@router.post("/orgs/current/webhooks", status_code=201)
def create_webhook(body: WebhookIn, p: Principal = Depends(require("keys")), db: Session = Depends(D.get_db)):
    import secrets

    bad = [e for e in body.events if e not in EVENTS]
    if bad:
        raise HTTPException(400, f"unknown events {bad}; valid: {EVENTS}")
    w = D.Webhook(id=D.uid("wh"), org_id=p.org.id, url=body.url, secret=body.secret or secrets.token_urlsafe(24), events=body.events,
                  description=body.description)
    db.add(w)
    db.commit()
    audit(db, p, "webhooks.create", "webhook", w.id, details={"url": w.url, "events": w.events})
    return D.to_dict(w) | {"note": "store the secret; signatures are HMAC-SHA256 over the raw body"}


@router.delete("/orgs/current/webhooks/{webhook_id}")
def delete_webhook(webhook_id: str, p: Principal = Depends(require("keys")), db: Session = Depends(D.get_db)):
    w = db.query(D.Webhook).filter_by(id=webhook_id, org_id=p.org.id).one_or_none()
    if w is None:
        raise HTTPException(404, "webhook not found")
    db.delete(w)
    db.commit()
    audit(db, p, "webhooks.delete", "webhook", webhook_id)
    return {"deleted": webhook_id}


@router.post("/orgs/current/webhooks/{webhook_id}/test")
def test_webhook(webhook_id: str, p: Principal = Depends(require("keys")), db: Session = Depends(D.get_db)):
    w = db.query(D.Webhook).filter_by(id=webhook_id, org_id=p.org.id).one_or_none()
    if w is None:
        raise HTTPException(404, "webhook not found")
    ids = deliver(p.org.id, "test", {"webhook_id": w.id, "message": "hello from saphire"}, sync=True)
    rows = db.query(D.WebhookDelivery).filter(D.WebhookDelivery.id.in_(ids), D.WebhookDelivery.webhook_id == w.id).all()
    return [D.to_dict(r) for r in rows]


@router.get("/orgs/current/webhooks/{webhook_id}/deliveries")
def deliveries(webhook_id: str, limit: int = Query(50, le=500), p: Principal = Depends(require("keys")), db: Session = Depends(D.get_db)):
    w = db.query(D.Webhook).filter_by(id=webhook_id, org_id=p.org.id).one_or_none()
    if w is None:
        raise HTTPException(404, "webhook not found")
    return [D.to_dict(r) for r in db.query(D.WebhookDelivery).filter_by(webhook_id=w.id).order_by(D.WebhookDelivery.created_at.desc()).limit(limit)]
