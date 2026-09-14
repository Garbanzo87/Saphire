from __future__ import annotations

from typing import Optional

from fastapi import Depends, Header, HTTPException, Query
from sqlalchemy.orm import Session

from . import db as D
from .config import settings


def require_api_key(x_api_key: Optional[str] = Header(default=None), authorization: Optional[str] = Header(default=None)) -> str:
    key = x_api_key or (authorization.split(" ", 1)[1] if authorization and " " in authorization else None)
    if settings.api_key and key != settings.api_key:
        raise HTTPException(status_code=401, detail="invalid or missing API key (x-api-key header)")
    return key or ""


def get_project(project: str = Query(default=None), db: Session = Depends(D.get_db)) -> D.Project:
    return D.ensure_project(db, project or settings.default_project)


def get_or_404(db: Session, model, id_: str):
    obj = db.get(model, id_)
    if obj is None:
        raise HTTPException(status_code=404, detail=f"{model.__name__} {id_} not found")
    return obj
