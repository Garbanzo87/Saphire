"""Service layer shared by API routers and background jobs."""
from __future__ import annotations

import json
from typing import Any, Optional

from sqlalchemy.orm import Session

from ..evaluation.runner import RolloutRecord, record_from
from ..sdk.types import AgentConfig, Reward, Rollout, TaskSpec
from . import db as D


def ingest_spans(db: Session, project: D.Project, spans: list[dict[str, Any]]) -> dict[str, int]:
    """Bulk upsert spans and roll them up into traces (set-based: 3 queries + 2 bulk writes per batch)."""
    if not spans:
        return {"traces": 0, "spans": 0}
    by_trace: dict[str, list[dict]] = {}
    for s in spans:
        by_trace.setdefault(s["trace_id"], []).append(s)
    trace_ids = list(by_trace)
    span_ids = [s["span_id"] for s in spans]
    existing_traces = {t.id: t for t in db.query(D.Trace).filter(D.Trace.id.in_(trace_ids)).all()}
    existing_spans = {r[0] for r in db.query(D.Span.id).filter(D.Span.id.in_(span_ids)).all()}
    new_span_rows: list[dict[str, Any]] = []
    upd_span_rows: list[dict[str, Any]] = []
    for tid, ss in by_trace.items():
        tr = existing_traces.get(tid)
        if tr is None:
            tr = D.Trace(id=tid, project_id=project.id, n_spans=0, attributes={}, status="unset", name="", service="")
            db.add(tr)
            existing_traces[tid] = tr
        for s in ss:
            row = {"id": s["span_id"], "trace_id": tid, "parent_span_id": s.get("parent_span_id"), "name": s.get("name", ""),
                   "kind": s.get("kind", "span"), "start_time": s.get("start_time"), "end_time": s.get("end_time"),
                   "status": s.get("status", "unset"), "attributes": s.get("attributes", {}) or {}, "events": s.get("events", []) or []}
            if s["span_id"] in existing_spans:
                upd_span_rows.append(row)
            else:
                new_span_rows.append(row)
                existing_spans.add(s["span_id"])
                tr.n_spans += 1
            if not s.get("parent_span_id"):
                tr.name = row["name"]
                tr.attributes = row["attributes"]
                tr.status = row["status"]
                if row["attributes"].get("rollout.id"):
                    tr.rollout_id = row["attributes"]["rollout.id"]
            tr.service = s.get("service") or tr.service
            tr.start_time = min([t for t in [tr.start_time, row["start_time"]] if t is not None], default=None)
            tr.end_time = max([t for t in [tr.end_time, row["end_time"]] if t is not None], default=None)
            if row["status"] == "error":
                tr.status = "error"
    db.flush()  # traces exist before spans reference them
    if new_span_rows:
        db.bulk_insert_mappings(D.Span, new_span_rows)
    if upd_span_rows:
        db.bulk_update_mappings(D.Span, upd_span_rows)
    db.commit()
    from . import metrics as MX

    MX.inc("saphire_spans_ingested_total", len(spans))
    return {"traces": len(by_trace), "spans": len(spans)}


def persist_rollout(db: Session, project: D.Project, rollout: Rollout, task: TaskSpec, rewards: list[Reward],
                    agent_id: Optional[str] = None, eval_run_id: Optional[str] = None, training_run_id: Optional[str] = None) -> D.RolloutRow:
    rec: RolloutRecord = record_from(rollout, task, rewards)
    row = D.RolloutRow(id=rollout.id, project_id=project.id, agent_id=agent_id, task_id=task.id, env_name=task.env_name,
                       status=rollout.status.value, task_success=rec.task_success, total_reward=rollout.total_reward,
                       record=rec.model_dump(), payload=json.loads(rollout.model_dump_json()), trace_id=rollout.trace_id,
                       eval_run_id=eval_run_id, training_run_id=training_run_id)
    db.add(row)
    for r in rewards:
        if r.step_index is None:
            db.add(D.Score(id=D.uid("sc"), project_id=project.id, trace_id=rollout.trace_id, rollout_id=rollout.id, name=r.name,
                           value=float(r.value), source=r.source, rationale=r.rationale or "", step_index=None))
    if rollout.trace_id:
        tr = db.get(D.Trace, rollout.trace_id)
        if tr is not None:
            tr.rollout_id = rollout.id
    db.commit()
    from .auth import meter

    meter(db, project.org_id, "rollouts", 1)
    meter(db, project.org_id, "tokens", float(rollout.usage.total_tokens))
    from . import metrics as MX

    MX.inc("saphire_rollouts_persisted_total", env=rollout.env_name)
    return row


def load_rollouts(db: Session, project_id: str, agent_name: Optional[str] = None, limit: int = 2000,
                  eval_run_id: Optional[str] = None, training_run_id: Optional[str] = None) -> list[Rollout]:
    q = db.query(D.RolloutRow).filter_by(project_id=project_id)
    if agent_name:
        ids = [a.id for a in db.query(D.Agent).filter_by(project_id=project_id, name=agent_name)]
        q = q.filter(D.RolloutRow.agent_id.in_(ids))
    if eval_run_id:
        q = q.filter_by(eval_run_id=eval_run_id)
    if training_run_id:
        q = q.filter_by(training_run_id=training_run_id)
    rows = q.order_by(D.RolloutRow.created_at.desc()).limit(limit).all()
    return [Rollout.model_validate(r.payload) for r in rows]


def create_agent_version(db: Session, project: D.Project, config: AgentConfig, parent_id: Optional[str] = None,
                         origin: str = "manual", status: str = "candidate") -> D.Agent:
    existing = db.query(D.Agent).filter_by(project_id=project.id, name=config.name, version=config.version).one_or_none()
    if existing is not None:
        existing.config = config.model_dump()
        db.commit()
        return existing
    a = D.Agent(id=D.uid("agent"), project_id=project.id, name=config.name, version=config.version, config=config.model_dump(),
                parent_id=parent_id, origin=origin, status=status)
    db.add(a)
    db.commit()
    return a


def next_version(db: Session, project_id: str, name: str) -> str:
    n = db.query(D.Agent).filter_by(project_id=project_id, name=name).count()
    return f"v{n}"


def record_metrics(db: Session, project: D.Project, agent: D.Agent, metrics: dict[str, Any], tags: dict[str, Any] | None = None,
                   ts: Optional[float] = None) -> None:
    for k, v in metrics.items():
        if isinstance(v, (int, float)) and k != "n":
            db.add(D.MetricPoint(id=D.uid("mp"), project_id=project.id, agent_name=agent.name, agent_id=agent.id, name=k, value=float(v),
                                 tags=tags or {}, **({"ts": ts} if ts else {})))
    db.commit()


def tasks_from_dataset(ds: D.Dataset) -> list[TaskSpec]:
    return [TaskSpec.model_validate(t) for t in ds.tasks]


def get_deployed(db: Session, project_id: str, agent_name: str) -> Optional[D.Agent]:
    return db.query(D.Agent).filter_by(project_id=project_id, name=agent_name, status="deployed").order_by(D.Agent.created_at.desc()).first()
