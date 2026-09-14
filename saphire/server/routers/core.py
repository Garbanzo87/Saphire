"""Projects, environments, agents, datasets."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ...environments.base import get_environment, list_environments
from ...evaluation.suites import build_suite, list_suites
from ...sdk.types import AgentConfig, TaskSpec
from .. import db as D
from .. import service as S
from ..deps import get_or_404, get_project, require_api_key

router = APIRouter(dependencies=[Depends(require_api_key)])


# ---------------- projects ----------------
class ProjectIn(BaseModel):
    name: str
    description: str = ""


@router.get("/projects")
def list_projects(db: Session = Depends(D.get_db)):
    return [D.to_dict(p) for p in db.query(D.Project).order_by(D.Project.created_at).all()]


@router.post("/projects", status_code=201)
def create_project(body: ProjectIn, db: Session = Depends(D.get_db)):
    p = D.ensure_project(db, body.name)
    p.description = body.description
    db.commit()
    return D.to_dict(p)


# ---------------- environments ----------------
@router.get("/environments")
def environments():
    return [get_environment(n).info() for n in list_environments()]


@router.get("/environments/{name}/tools")
def environment_tools(name: str):
    env = get_environment(name)
    return [t.model_dump() for t in env.tools.specs()]


@router.get("/suites")
def suites():
    return list_suites()


# ---------------- agents ----------------
class AgentIn(BaseModel):
    config: AgentConfig
    parent_id: Optional[str] = None
    status: str = "candidate"


@router.get("/agents")
def list_agents(name: Optional[str] = None, project: D.Project = Depends(get_project), db: Session = Depends(D.get_db)):
    q = db.query(D.Agent).filter_by(project_id=project.id)
    if name:
        q = q.filter_by(name=name)
    return [D.to_dict(a) for a in q.order_by(D.Agent.created_at.desc()).all()]


@router.post("/agents", status_code=201)
def create_agent(body: AgentIn, project: D.Project = Depends(get_project), db: Session = Depends(D.get_db)):
    cfg = body.config
    if not cfg.version or cfg.version == "v0":
        cfg = cfg.model_copy(update={"version": S.next_version(db, project.id, cfg.name)})
    a = S.create_agent_version(db, project, cfg, parent_id=body.parent_id, status=body.status)
    return D.to_dict(a)


@router.get("/agents/{agent_id}")
def get_agent(agent_id: str, db: Session = Depends(D.get_db)):
    a = get_or_404(db, D.Agent, agent_id)
    evals = db.query(D.EvalRun).filter_by(agent_id=agent_id).order_by(D.EvalRun.created_at.desc()).limit(20).all()
    return D.to_dict(a) | {"eval_runs": [D.to_dict(e) for e in evals]}


@router.post("/agents/{agent_id}/promote")
def promote_agent(agent_id: str, db: Session = Depends(D.get_db)):
    from ..jobs import promote

    a = get_or_404(db, D.Agent, agent_id)
    promote(db, a.project_id, a)
    return D.to_dict(a)


@router.delete("/agents/{agent_id}")
def retire_agent(agent_id: str, db: Session = Depends(D.get_db)):
    a = get_or_404(db, D.Agent, agent_id)
    a.status = "retired"
    db.commit()
    return D.to_dict(a)


# ---------------- datasets ----------------
class DatasetIn(BaseModel):
    name: str
    suite: Optional[str] = None  # build from a named suite ...
    tasks: list[TaskSpec] = Field(default_factory=list)  # ... or provide tasks
    n_per_env: int = 14
    seed: int = 0
    split: str = "eval"


@router.get("/datasets")
def list_datasets(project: D.Project = Depends(get_project), db: Session = Depends(D.get_db)):
    out = []
    for d in db.query(D.Dataset).filter_by(project_id=project.id).order_by(D.Dataset.created_at.desc()).all():
        row = D.to_dict(d)
        row["n_tasks"] = len(d.tasks)
        row.pop("tasks")
        out.append(row)
    return out


@router.post("/datasets", status_code=201)
def create_dataset(body: DatasetIn, project: D.Project = Depends(get_project), db: Session = Depends(D.get_db)):
    tasks = body.tasks or (build_suite(body.suite, n_per_env=body.n_per_env, seed=body.seed) if body.suite else [])
    if not tasks:
        raise HTTPException(400, "provide `suite` or `tasks`")
    d = D.Dataset(id=D.uid("ds"), project_id=project.id, name=body.name, suite=body.suite or "custom",
                  tasks=[t.model_dump() for t in tasks], split=body.split)
    db.add(d)
    db.commit()
    row = D.to_dict(d)
    row["n_tasks"] = len(tasks)
    row.pop("tasks")
    return row


@router.get("/datasets/{dataset_id}")
def get_dataset(dataset_id: str, limit: int = Query(100, le=2000), db: Session = Depends(D.get_db)):
    d = get_or_404(db, D.Dataset, dataset_id)
    row = D.to_dict(d)
    row["n_tasks"] = len(d.tasks)
    row["tasks"] = d.tasks[:limit]
    return row


@router.post("/datasets/{dataset_id}/tasks", status_code=201)
def add_tasks(dataset_id: str, tasks: list[TaskSpec], db: Session = Depends(D.get_db)):
    d = get_or_404(db, D.Dataset, dataset_id)
    d.tasks = list(d.tasks) + [t.model_dump() for t in tasks]
    db.commit()
    return {"n_tasks": len(d.tasks)}
