"""SCIM 2.0 user provisioning (RFC 7644 subset) for identity providers (Okta, Entra ID, OneLogin ...).

Authenticate with an org API key of role admin/owner as the SCIM bearer token. Users are provisioned into that org
with `org.settings.default_role` (or `member`); deactivating (`active: false`) or deleting removes the membership.
Endpoints: GET/POST /scim/v2/Users, GET/PUT/PATCH/DELETE /scim/v2/Users/{id}, GET /scim/v2/ServiceProviderConfig.
"""
from __future__ import annotations

import re
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from sqlalchemy.orm import Session

from .. import db as D
from ..auth import Principal, audit, require, upsert_user

router = APIRouter(prefix="/scim/v2", tags=["scim"])
SCHEMA_USER = "urn:ietf:params:scim:schemas:core:2.0:User"
SCHEMA_LIST = "urn:ietf:params:scim:api:messages:2.0:ListResponse"
SCHEMA_PATCH = "urn:ietf:params:scim:api:messages:2.0:PatchOp"
SCHEMA_ERROR = "urn:ietf:params:scim:api:messages:2.0:Error"


def _scim_error(status: int, detail: str):
    raise HTTPException(status, {"schemas": [SCHEMA_ERROR], "status": str(status), "detail": detail})


def _user_json(u: D.User, m: Optional[D.Membership], org: D.Organization) -> dict[str, Any]:
    return {"schemas": [SCHEMA_USER], "id": u.id, "userName": u.email, "displayName": u.name or u.email,
            "name": {"formatted": u.name or u.email}, "emails": [{"value": u.email, "primary": True}],
            "active": m is not None, "roles": [{"value": m.role, "primary": True}] if m else [],
            "externalId": u.sso_subject or None,
            "meta": {"resourceType": "User", "created": u.created_at, "location": f"/scim/v2/Users/{u.id}", "organization": org.slug}}


def _get(db: Session, org: D.Organization, user_id: str) -> tuple[D.User, Optional[D.Membership]]:
    u = db.get(D.User, user_id)
    if u is None:
        _scim_error(404, "User not found")
    return u, db.query(D.Membership).filter_by(org_id=org.id, user_id=u.id).one_or_none()


@router.get("/ServiceProviderConfig")
def service_provider_config():
    return {"schemas": ["urn:ietf:params:scim:schemas:core:2.0:ServiceProviderConfig"], "patch": {"supported": True}, "bulk": {"supported": False},
            "filter": {"supported": True, "maxResults": 200}, "changePassword": {"supported": False}, "sort": {"supported": False},
            "etag": {"supported": False}, "authenticationSchemes": [{"type": "oauthbearertoken", "name": "Org API key (admin)", "description": "x-api-key or Bearer"}]}


@router.get("/Users")
def list_users(filter: Optional[str] = None, startIndex: int = Query(1, ge=1), count: int = Query(100, le=200),
               p: Principal = Depends(require("members")), db: Session = Depends(D.get_db)):
    org = p.org
    q = db.query(D.User, D.Membership).join(D.Membership, D.Membership.user_id == D.User.id).filter(D.Membership.org_id == org.id)
    if filter:
        m = re.match(r'\s*(userName|emails?(?:\.value)?)\s+eq\s+"([^"]+)"', filter, re.I)
        if not m:
            _scim_error(400, "unsupported filter; use userName eq \"...\"")
        q = q.filter(D.User.email == m.group(2).lower())
    rows = q.order_by(D.User.email).all()
    page = rows[startIndex - 1:startIndex - 1 + count]
    return {"schemas": [SCHEMA_LIST], "totalResults": len(rows), "startIndex": startIndex, "itemsPerPage": len(page),
            "Resources": [_user_json(u, mem, org) for u, mem in page]}


@router.post("/Users", status_code=201)
def create_user(body: dict[str, Any], p: Principal = Depends(require("members")), db: Session = Depends(D.get_db)):
    org = p.org
    email = (body.get("userName") or next((e.get("value") for e in body.get("emails", []) if e.get("primary")), None) or "").lower()
    if not email or "@" not in email:
        _scim_error(400, "userName (email) required")
    name = (body.get("displayName") or (body.get("name") or {}).get("formatted") or "")
    u = upsert_user(db, email, name, provider="scim", subject=str(body.get("externalId") or ""))
    m = db.query(D.Membership).filter_by(org_id=org.id, user_id=u.id).one_or_none()
    if m is not None:
        _scim_error(409, "User already provisioned in this organization")
    role = (org.settings or {}).get("default_role", "member")
    roles = body.get("roles") or []
    if roles and roles[0].get("value") in ("viewer", "member", "admin"):
        role = roles[0]["value"]
    m = D.Membership(id=D.uid("mem"), org_id=org.id, user_id=u.id, role=role)
    db.add(m)
    db.commit()
    audit(db, p, "scim.users.create", "user", u.id, details={"email": u.email, "role": role})
    return _user_json(u, m, org)


@router.get("/Users/{user_id}")
def get_user(user_id: str, p: Principal = Depends(require("members")), db: Session = Depends(D.get_db)):
    u, m = _get(db, p.org, user_id)
    if m is None:
        _scim_error(404, "User not found in this organization")
    return _user_json(u, m, p.org)


@router.put("/Users/{user_id}")
def replace_user(user_id: str, body: dict[str, Any], p: Principal = Depends(require("members")), db: Session = Depends(D.get_db)):
    u, m = _get(db, p.org, user_id)
    if body.get("displayName"):
        u.name = body["displayName"]
    if body.get("active") is False and m is not None:
        db.delete(m)
        m = None
    elif body.get("active", True) and m is None:
        m = D.Membership(id=D.uid("mem"), org_id=p.org.id, user_id=u.id, role=(p.org.settings or {}).get("default_role", "member"))
        db.add(m)
    db.commit()
    audit(db, p, "scim.users.replace", "user", u.id, details={"active": m is not None})
    return _user_json(u, m, p.org)


@router.patch("/Users/{user_id}")
def patch_user(user_id: str, body: dict[str, Any], p: Principal = Depends(require("members")), db: Session = Depends(D.get_db)):
    u, m = _get(db, p.org, user_id)
    for op in body.get("Operations", []):
        name = (op.get("op") or "").lower()
        path = (op.get("path") or "").lower()
        value = op.get("value")
        if name not in ("replace", "add", "remove"):
            _scim_error(400, f"unsupported op {name}")
        values = value if isinstance(value, dict) and not path else {path: value}
        for k, v in values.items():
            k = k.lower()
            if k == "active":
                active = v in (True, "true", "True")
                if not active and m is not None:
                    db.delete(m)
                    m = None
                elif active and m is None:
                    m = D.Membership(id=D.uid("mem"), org_id=p.org.id, user_id=u.id, role=(p.org.settings or {}).get("default_role", "member"))
                    db.add(m)
            elif k in ("displayname", "name.formatted"):
                u.name = str(v)
            elif k.startswith("roles") and m is not None:
                role = v[0]["value"] if isinstance(v, list) and v else (v if isinstance(v, str) else None)
                if role in ("viewer", "member", "admin"):
                    m.role = role
    db.commit()
    audit(db, p, "scim.users.patch", "user", u.id, details={"active": m is not None, "ops": len(body.get("Operations", []))})
    return _user_json(u, m, p.org)


@router.delete("/Users/{user_id}", status_code=204)
def delete_user(user_id: str, request: Request, p: Principal = Depends(require("members")), db: Session = Depends(D.get_db)):
    u, m = _get(db, p.org, user_id)
    if m is not None:
        db.delete(m)
        db.commit()
    audit(db, p, "scim.users.delete", "user", u.id)
    return Response(status_code=204)
