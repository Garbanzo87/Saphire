"""Data retention: delete traces/spans/rollouts/scores/audit older than the org's retention window.

Order: org.settings.retention_days > SAPHIRE_RETENTION_DAYS > keep forever. Runs as the `retention` job
(schedule it from cron/K8s CronJob via `saphire retention`) and deletes in batches to keep transactions short.
Aggregated data (eval runs, metric points, agent versions, usage counters) is never deleted.
"""
from __future__ import annotations

import time
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from . import db as D
from .config import settings


def purge_org(db: Session, org: D.Organization, days: int, batch: int = 5000, dry_run: bool = False) -> dict[str, int]:
    cutoff = time.time() - days * 86400
    project_ids = [p.id for p in db.query(D.Project).filter_by(org_id=org.id)]
    counts = {"traces": 0, "spans": 0, "rollouts": 0, "scores": 0, "audit": 0}
    if not project_ids:
        return counts
    # rollouts + their scores
    while True:
        ids = [r[0] for r in db.execute(select(D.RolloutRow.id).where(D.RolloutRow.project_id.in_(project_ids), D.RolloutRow.created_at < cutoff).limit(batch))]
        if not ids:
            break
        counts["rollouts"] += len(ids)
        if dry_run:
            break
        counts["scores"] += db.execute(delete(D.Score).where(D.Score.rollout_id.in_(ids))).rowcount
        db.execute(delete(D.RolloutRow).where(D.RolloutRow.id.in_(ids)))
        db.commit()
    # traces + spans + trace scores
    while True:
        ids = [r[0] for r in db.execute(select(D.Trace.id).where(D.Trace.project_id.in_(project_ids), D.Trace.created_at < cutoff).limit(batch))]
        if not ids:
            break
        counts["traces"] += len(ids)
        if dry_run:
            break
        counts["spans"] += db.execute(delete(D.Span).where(D.Span.trace_id.in_(ids))).rowcount
        counts["scores"] += db.execute(delete(D.Score).where(D.Score.trace_id.in_(ids))).rowcount
        db.execute(delete(D.Trace).where(D.Trace.id.in_(ids)))
        db.commit()
    if not dry_run:
        counts["audit"] = db.execute(delete(D.AuditLog).where(D.AuditLog.org_id == org.id, D.AuditLog.created_at < cutoff - 365 * 86400)).rowcount
        db.commit()
    return counts


def run_retention(db: Session, default_days: int | None = None, dry_run: bool = False) -> dict[str, Any]:
    default_days = settings.retention_days if default_days is None else default_days
    report: dict[str, Any] = {}
    for org in db.query(D.Organization).all():
        days = int((org.settings or {}).get("retention_days") or default_days or 0)
        if days <= 0:
            report[org.slug] = {"skipped": "no retention configured"}
            continue
        report[org.slug] = {"days": days, **purge_org(db, org, days, dry_run=dry_run)}
    return report
