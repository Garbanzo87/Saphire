"""Build a production-intelligence report for a project from stored rollouts, traces, datasets and attribution jobs."""
from __future__ import annotations

import time
from typing import Any, Optional

from sqlalchemy.orm import Session

from ..sdk.types import Rollout, TaskSpec
from ..server import db as D
from .drift import drift_report
from .failures import cluster_failures
from .mining import coverage_gaps, mine_tasks_from_rollouts
from .recommend import recommend


def _rows(db: Session, project_id: str, agent_name: Optional[str], since: float, until: float, production_only: bool) -> list[D.RolloutRow]:
    q = db.query(D.RolloutRow).filter(D.RolloutRow.project_id == project_id, D.RolloutRow.created_at >= since, D.RolloutRow.created_at < until)
    if agent_name:
        ids = [a.id for a in db.query(D.Agent).filter_by(project_id=project_id, name=agent_name)]
        q = q.filter(D.RolloutRow.agent_id.in_(ids))
    if production_only:
        q = q.filter(D.RolloutRow.eval_run_id.is_(None), D.RolloutRow.training_run_id.is_(None))
    return q.order_by(D.RolloutRow.created_at).limit(20000).all()


def _human_scores(db: Session, rollout_ids: list[str]) -> dict[str, float]:
    if not rollout_ids:
        return {}
    rows = db.query(D.Score).filter(D.Score.rollout_id.in_(rollout_ids), D.Score.source.in_(("human", "sdk", "api", "product"))).all()
    out: dict[str, list[float]] = {}
    for r in rows:
        out.setdefault(r.rollout_id, []).append(r.value)
    return {k: sum(v) / len(v) for k, v in out.items()}


def build_report(db: Session, project: D.Project, agent_name: Optional[str] = None, recent_hours: float = 24, baseline_hours: float = 168,
                 production_only: bool = False, now: Optional[float] = None) -> dict[str, Any]:
    now = now or time.time()
    recent_rows = _rows(db, project.id, agent_name, now - recent_hours * 3600, now, production_only)
    base_rows = _rows(db, project.id, agent_name, now - (recent_hours + baseline_hours) * 3600, now - recent_hours * 3600, production_only)
    recent = [Rollout.model_validate(r.payload) for r in recent_rows]
    base = [Rollout.model_validate(r.payload) for r in base_rows]
    fam = {r.task_id: (r.record or {}).get("family", "") for r in recent_rows + base_rows}
    drift = drift_report([r.record for r in base_rows], [r.record for r in recent_rows])
    failures = cluster_failures(recent, families=fam, baseline=base if base else None)
    eval_tasks: list[TaskSpec] = []
    for ds in db.query(D.Dataset).filter_by(project_id=project.id):
        eval_tasks += [TaskSpec.model_validate(t) for t in ds.tasks]
    coverage = coverage_gaps(recent, eval_tasks)
    multi_agent = any(r.metadata.get("multi_agent") for r in recent)
    attribution = None
    att_job = (db.query(D.Job).filter_by(project_id=project.id, type="attribution", status="succeeded").order_by(D.Job.created_at.desc()).first())
    if att_job and att_job.finished_at and att_job.finished_at > now - 7 * 86400:
        attribution = att_job.result
    recs = recommend(drift, failures, coverage, attribution, multi_agent=multi_agent)
    minable = len(mine_tasks_from_rollouts(recent, human_scores=_human_scores(db, [r.id for r in recent_rows]) or None))
    return {"agent_name": agent_name, "recent_hours": recent_hours, "baseline_hours": baseline_hours, "n_recent": len(recent), "n_baseline": len(base),
            "healthy": drift.get("healthy", True) and failures.get("failure_rate", 0) < 0.5, "drift": drift, "failures": failures,
            "coverage": coverage | {"minable_tasks": minable}, "recommendations": recs, "attribution_used": att_job.id if attribution else None,
            "production_only": production_only}


def persist_report(db: Session, project: D.Project, rep: dict[str, Any], job_id: Optional[str] = None) -> D.IntelligenceReport:
    row = D.IntelligenceReport(id=D.uid("intel"), project_id=project.id, agent_name=rep["agent_name"], recent_hours=rep["recent_hours"],
                               baseline_hours=rep["baseline_hours"], n_recent=rep["n_recent"], n_baseline=rep["n_baseline"], healthy=rep["healthy"],
                               drift=rep["drift"], failures=rep["failures"], coverage=rep["coverage"], recommendations=rep["recommendations"], job_id=job_id)
    db.add(row)
    db.commit()
    return row
