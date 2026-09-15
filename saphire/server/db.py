from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any, Iterator, Optional

from sqlalchemy import JSON, Boolean, Float, ForeignKey, Index, Integer, String, Text, UniqueConstraint, create_engine, event
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from .config import settings


def now() -> float:
    return time.time()


def uid(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class Base(DeclarativeBase):
    pass


class Organization(Base):
    """Tenant. Every project, API key, membership, usage record and audit entry belongs to exactly one org."""

    __tablename__ = "organizations"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    slug: Mapped[str] = mapped_column(String(100), unique=True)
    plan: Mapped[str] = mapped_column(String(32), default="free")  # free | team | enterprise
    quotas: Mapped[dict] = mapped_column(JSON, default=dict)  # e.g. {"rollouts_per_month": 5000, "jobs_per_day": 50, "requests_per_minute": 600}
    settings: Mapped[dict] = mapped_column(JSON, default=dict)  # {"allowed_email_domains": [...], "default_role": "member", "retention_days": 90}
    created_at: Mapped[float] = mapped_column(Float, default=now)


class User(Base):
    __tablename__ = "users"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    email: Mapped[str] = mapped_column(String(320), unique=True)
    name: Mapped[str] = mapped_column(String(200), default="")
    sso_provider: Mapped[str] = mapped_column(String(64), default="")
    sso_subject: Mapped[str] = mapped_column(String(320), default="")
    is_superadmin: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[float] = mapped_column(Float, default=now)
    last_login_at: Mapped[Optional[float]] = mapped_column(Float, nullable=True)


class Membership(Base):
    __tablename__ = "memberships"
    __table_args__ = (UniqueConstraint("org_id", "user_id", name="uq_membership"),)
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    org_id: Mapped[str] = mapped_column(ForeignKey("organizations.id"), index=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    role: Mapped[str] = mapped_column(String(16), default="member")  # owner | admin | member | viewer
    created_at: Mapped[float] = mapped_column(Float, default=now)


class ApiKey(Base):
    """Hashed API keys scoped to an org with a role. The raw key is shown once at creation."""

    __tablename__ = "api_keys"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    org_id: Mapped[str] = mapped_column(ForeignKey("organizations.id"), index=True)
    name: Mapped[str] = mapped_column(String(200))
    key_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    prefix: Mapped[str] = mapped_column(String(16))
    role: Mapped[str] = mapped_column(String(16), default="member")
    allowed_ips: Mapped[list] = mapped_column(JSON, default=list)  # CIDRs / IPs; [] = any
    created_by: Mapped[str] = mapped_column(String(64), default="")
    created_at: Mapped[float] = mapped_column(Float, default=now)
    last_used_at: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    expires_at: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    revoked_at: Mapped[Optional[float]] = mapped_column(Float, nullable=True)


class AuditLog(Base):
    __tablename__ = "audit_log"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    org_id: Mapped[Optional[str]] = mapped_column(String(64), index=True, nullable=True)
    actor_type: Mapped[str] = mapped_column(String(16))  # user | api_key | root | system
    actor_id: Mapped[str] = mapped_column(String(64), default="")
    actor_label: Mapped[str] = mapped_column(String(320), default="")
    action: Mapped[str] = mapped_column(String(100), index=True)  # e.g. "agents.promote", "evals.create"
    resource_type: Mapped[str] = mapped_column(String(64), default="")
    resource_id: Mapped[str] = mapped_column(String(64), default="")
    method: Mapped[str] = mapped_column(String(8), default="")
    path: Mapped[str] = mapped_column(String(300), default="")
    status_code: Mapped[int] = mapped_column(Integer, default=0)
    ip: Mapped[str] = mapped_column(String(64), default="")
    details: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[float] = mapped_column(Float, default=now, index=True)


class UsageCounter(Base):
    """Metered usage per org, metric and UTC day (rollouts, spans, tokens, jobs, api_requests)."""

    __tablename__ = "usage_counters"
    __table_args__ = (UniqueConstraint("org_id", "metric", "day", name="uq_usage"),)
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    org_id: Mapped[str] = mapped_column(String(64), index=True)
    metric: Mapped[str] = mapped_column(String(48), index=True)
    day: Mapped[str] = mapped_column(String(10), index=True)  # YYYY-MM-DD
    value: Mapped[float] = mapped_column(Float, default=0.0)


class Webhook(Base):
    """Org-level outbound webhooks (HMAC-SHA256 signed) for job, gate, quota and intelligence events."""

    __tablename__ = "webhooks"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    org_id: Mapped[str] = mapped_column(ForeignKey("organizations.id"), index=True)
    url: Mapped[str] = mapped_column(String(1000))
    secret: Mapped[str] = mapped_column(String(128))
    events: Mapped[list] = mapped_column(JSON, default=list)  # [] = all
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    description: Mapped[str] = mapped_column(String(300), default="")
    created_at: Mapped[float] = mapped_column(Float, default=now)


class WebhookDelivery(Base):
    __tablename__ = "webhook_deliveries"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    webhook_id: Mapped[str] = mapped_column(String(64), index=True)
    event: Mapped[str] = mapped_column(String(64))
    payload: Mapped[dict] = mapped_column(JSON, default=dict)
    status_code: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str] = mapped_column(Text, default="")
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[float] = mapped_column(Float, default=now)


class IntelligenceReport(Base):
    """Snapshot of production intelligence for a project (drift, failure clusters, coverage, recommendations)."""

    __tablename__ = "intelligence_reports"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    agent_name: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    recent_hours: Mapped[float] = mapped_column(Float, default=24)
    baseline_hours: Mapped[float] = mapped_column(Float, default=168)
    n_recent: Mapped[int] = mapped_column(Integer, default=0)
    n_baseline: Mapped[int] = mapped_column(Integer, default=0)
    healthy: Mapped[bool] = mapped_column(Boolean, default=True)
    drift: Mapped[dict] = mapped_column(JSON, default=dict)
    failures: Mapped[dict] = mapped_column(JSON, default=dict)
    coverage: Mapped[dict] = mapped_column(JSON, default=dict)
    recommendations: Mapped[list] = mapped_column(JSON, default=list)
    job_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    created_at: Mapped[float] = mapped_column(Float, default=now, index=True)


class Project(Base):
    __tablename__ = "projects"
    __table_args__ = (UniqueConstraint("org_id", "name", name="uq_project_org_name"),)
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    org_id: Mapped[str] = mapped_column(ForeignKey("organizations.id"), index=True)
    name: Mapped[str] = mapped_column(String(200))
    description: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[float] = mapped_column(Float, default=now)


class Agent(Base):
    """An agent *version*. Versions of the same logical agent share `name`."""

    __tablename__ = "agents"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    name: Mapped[str] = mapped_column(String(200), index=True)
    version: Mapped[str] = mapped_column(String(64))
    config: Mapped[dict] = mapped_column(JSON)  # AgentConfig
    parent_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="candidate")  # candidate | deployed | retired
    origin: Mapped[str] = mapped_column(String(64), default="manual")  # manual | online | sft | dpo | grpo | prompt_opt
    created_at: Mapped[float] = mapped_column(Float, default=now)


class Dataset(Base):
    __tablename__ = "datasets"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    name: Mapped[str] = mapped_column(String(200))
    suite: Mapped[str] = mapped_column(String(64), default="custom")
    tasks: Mapped[list] = mapped_column(JSON)  # list[TaskSpec]
    split: Mapped[str] = mapped_column(String(16), default="eval")  # train | eval
    created_at: Mapped[float] = mapped_column(Float, default=now)


class Trace(Base):
    __tablename__ = "traces"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)  # otel trace id
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    name: Mapped[str] = mapped_column(String(300), default="")
    service: Mapped[str] = mapped_column(String(200), default="")
    start_time: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    end_time: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(String(16), default="unset")
    n_spans: Mapped[int] = mapped_column(Integer, default=0)
    attributes: Mapped[dict] = mapped_column(JSON, default=dict)
    rollout_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    created_at: Mapped[float] = mapped_column(Float, default=now)


class Span(Base):
    __tablename__ = "spans"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    trace_id: Mapped[str] = mapped_column(ForeignKey("traces.id"), index=True)
    parent_span_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    name: Mapped[str] = mapped_column(String(300))
    kind: Mapped[str] = mapped_column(String(32), default="span")
    start_time: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    end_time: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(String(16), default="unset")
    attributes: Mapped[dict] = mapped_column(JSON, default=dict)
    events: Mapped[list] = mapped_column(JSON, default=list)


class RolloutRow(Base):
    __tablename__ = "rollouts"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    agent_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    task_id: Mapped[str] = mapped_column(String(64), index=True)
    env_name: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(16))
    task_success: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    total_reward: Mapped[float] = mapped_column(Float, default=0.0)
    record: Mapped[dict] = mapped_column(JSON, default=dict)  # RolloutRecord
    payload: Mapped[dict] = mapped_column(JSON)  # full Rollout
    trace_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    eval_run_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    training_run_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    created_at: Mapped[float] = mapped_column(Float, default=now)


class Score(Base):
    """A scalar signal attached to a trace or rollout (verifier, judge, human, SDK)."""

    __tablename__ = "scores"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    trace_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    rollout_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    name: Mapped[str] = mapped_column(String(100))
    value: Mapped[float] = mapped_column(Float)
    source: Mapped[str] = mapped_column(String(100), default="api")
    rationale: Mapped[str] = mapped_column(Text, default="")
    step_index: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    created_at: Mapped[float] = mapped_column(Float, default=now)


class Job(Base):
    __tablename__ = "jobs"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    type: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(16), default="pending", index=True)  # pending|running|succeeded|failed|cancelled
    params: Mapped[dict] = mapped_column(JSON, default=dict)
    result: Mapped[dict] = mapped_column(JSON, default=dict)
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    logs: Mapped[str] = mapped_column(Text, default="")
    error: Mapped[str] = mapped_column(Text, default="")
    worker: Mapped[str] = mapped_column(String(100), default="")
    created_at: Mapped[float] = mapped_column(Float, default=now)
    started_at: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    finished_at: Mapped[Optional[float]] = mapped_column(Float, nullable=True)


class EvalRun(Base):
    __tablename__ = "eval_runs"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    agent_id: Mapped[str] = mapped_column(String(64), index=True)
    dataset_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    suite: Mapped[str] = mapped_column(String(64), default="custom")
    k: Mapped[int] = mapped_column(Integer, default=1)
    judge_model: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    status: Mapped[str] = mapped_column(String(16), default="pending")
    job_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    metrics: Mapped[dict] = mapped_column(JSON, default=dict)
    by_family: Mapped[dict] = mapped_column(JSON, default=dict)
    by_env: Mapped[dict] = mapped_column(JSON, default=dict)
    by_difficulty: Mapped[dict] = mapped_column(JSON, default=dict)
    by_role: Mapped[dict] = mapped_column(JSON, default=dict)
    n_tasks: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[float] = mapped_column(Float, default=now)
    finished_at: Mapped[Optional[float]] = mapped_column(Float, nullable=True)


class TrainingRun(Base):
    __tablename__ = "training_runs"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    agent_id: Mapped[str] = mapped_column(String(64), index=True)  # input agent
    algorithm: Mapped[str] = mapped_column(String(32))  # online | router | exemplars | prompt_opt | sft | dpo | grpo | signals
    params: Mapped[dict] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(String(16), default="pending")
    job_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    result: Mapped[dict] = mapped_column(JSON, default=dict)
    output_agent_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    history: Mapped[list] = mapped_column(JSON, default=list)  # per-iteration metrics for online runs
    created_at: Mapped[float] = mapped_column(Float, default=now)
    finished_at: Mapped[Optional[float]] = mapped_column(Float, nullable=True)


class Experiment(Base):
    """Online A/B experiment between agent versions on live traffic (SDK-assigned variants)."""

    __tablename__ = "experiments"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    name: Mapped[str] = mapped_column(String(200))
    variants: Mapped[list] = mapped_column(JSON)  # [{"name","agent_id","weight"}]
    status: Mapped[str] = mapped_column(String(16), default="running")
    outcomes: Mapped[list] = mapped_column(JSON, default=list)  # [{"variant","unit","value","ts"}]
    created_at: Mapped[float] = mapped_column(Float, default=now)


class MetricPoint(Base):
    __tablename__ = "metric_points"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    agent_name: Mapped[str] = mapped_column(String(200), index=True)
    agent_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    name: Mapped[str] = mapped_column(String(100), index=True)
    value: Mapped[float] = mapped_column(Float)
    ts: Mapped[float] = mapped_column(Float, default=now, index=True)
    tags: Mapped[dict] = mapped_column(JSON, default=dict)


class Deployment(Base):
    __tablename__ = "deployments"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    agent_name: Mapped[str] = mapped_column(String(200), index=True)
    agent_id: Mapped[str] = mapped_column(String(64))
    baseline_agent_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    eval_run_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    policy: Mapped[dict] = mapped_column(JSON, default=dict)
    decision: Mapped[dict] = mapped_column(JSON, default=dict)
    promoted: Mapped[bool] = mapped_column(Boolean, default=False)
    active: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[float] = mapped_column(Float, default=now)


Index("ix_metric_points_lookup", MetricPoint.project_id, MetricPoint.agent_name, MetricPoint.name)
Index("ix_rollouts_project_created", RolloutRow.project_id, RolloutRow.created_at)
Index("ix_traces_project_start", Trace.project_id, Trace.start_time)
Index("ix_traces_project_created", Trace.project_id, Trace.created_at)
Index("ix_spans_trace_start", Span.trace_id, Span.start_time)
Index("ix_scores_project_name", Score.project_id, Score.name)
Index("ix_audit_org_created", AuditLog.org_id, AuditLog.created_at)
Index("ix_jobs_status_created", Job.status, Job.created_at)

# ---------------------------------------------------------------------------
_engine = None
_Session = None


def get_engine(url: Optional[str] = None):
    global _engine, _Session
    if _engine is None:
        url = url or settings.database_url
        if url.startswith("sqlite"):
            path = url.split("///", 1)[-1]
            if path and path != ":memory:":
                Path(path).parent.mkdir(parents=True, exist_ok=True)
            _engine = create_engine(url, connect_args={"check_same_thread": False, "timeout": 30})

            @event.listens_for(_engine, "connect")
            def _pragma(dbapi_conn, _):
                cur = dbapi_conn.cursor()
                cur.execute("PRAGMA journal_mode=WAL")
                cur.execute("PRAGMA busy_timeout=30000")
                cur.close()
        else:
            _engine = create_engine(url, pool_pre_ping=True)
        _Session = sessionmaker(bind=_engine, expire_on_commit=False)
        Base.metadata.create_all(_engine)
    return _engine


def reset_engine() -> None:
    global _engine, _Session
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _Session = None


def session() -> Session:
    get_engine()
    return _Session()


def get_db() -> Iterator[Session]:
    db = session()
    try:
        yield db
    finally:
        db.close()


DEFAULT_ORG_SLUG = "default"


def _get_or_create(db: Session, model, filters: dict[str, Any], defaults: dict[str, Any]):
    """Race-safe get-or-create (concurrent first requests may both try to insert)."""
    from sqlalchemy.exc import IntegrityError

    obj = db.query(model).filter_by(**filters).one_or_none()
    if obj is not None:
        return obj
    try:
        with db.begin_nested():
            obj = model(**filters, **defaults)
            db.add(obj)
        db.commit()
        return obj
    except IntegrityError:
        db.rollback()
        return db.query(model).filter_by(**filters).one()


def ensure_org(db: Session, slug: str = DEFAULT_ORG_SLUG, name: Optional[str] = None, plan: str = "enterprise") -> Organization:
    return _get_or_create(db, Organization, {"slug": slug}, {"id": uid("org"), "name": name or slug, "plan": plan, "quotas": {}, "settings": {}})


def ensure_project(db: Session, name: str, org_id: Optional[str] = None) -> Project:
    org_id = org_id or ensure_org(db).id
    return _get_or_create(db, Project, {"name": name, "org_id": org_id}, {"id": uid("proj")})


def to_dict(obj: Any) -> dict[str, Any]:
    return {c.name: getattr(obj, c.name) for c in obj.__table__.columns}
