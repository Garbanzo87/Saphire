from __future__ import annotations

from fastapi import Depends, HTTPException, Query
from sqlalchemy.orm import Session

from . import db as D
from .auth import Principal, current_principal, get_principal, method_permission, require  # noqa: F401  (re-exported)
from .config import settings


def require_api_key(p: Principal = Depends(method_permission)) -> Principal:
    """Router-level auth: any valid principal; GET needs `read`, mutations need `write`."""
    return p


def get_project(project: str = Query(default=None), p: Principal = Depends(get_principal), db: Session = Depends(D.get_db)) -> D.Project:
    """Resolve the project inside the caller's org (created on first use). Cross-org access is impossible by construction."""
    org = p.org or D.ensure_org(db)
    return D.ensure_project(db, project or settings.default_project, org_id=org.id)


def get_or_404(db: Session, model, id_: str, principal: Principal | None = None):
    obj = db.get(model, id_)
    if obj is None:
        raise HTTPException(status_code=404, detail=f"{model.__name__} {id_} not found")
    principal = principal or current_principal.get()
    if principal is not None and not principal.is_superadmin and hasattr(obj, "project_id"):
        proj = db.get(D.Project, obj.project_id)
        if proj is None or (principal.org and proj.org_id != principal.org.id):
            raise HTTPException(status_code=404, detail=f"{model.__name__} {id_} not found")
    return obj
