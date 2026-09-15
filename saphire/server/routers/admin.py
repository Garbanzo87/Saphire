"""Organizations, members, API keys, usage, audit log, and authentication (OIDC SSO / JWT)."""
from __future__ import annotations

import secrets
import time
from typing import Any, Optional
from urllib.parse import urlencode

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from .. import db as D
from ..auth import (
    PLAN_DEFAULTS,
    Principal,
    audit,
    auto_join,
    create_api_key,
    decode_jwt,
    effective_quotas,
    get_principal,
    issue_jwt,
    require,
    upsert_user,
    usage_summary,
)
from ..config import settings

router = APIRouter()
auth_router = APIRouter(prefix="/auth", tags=["auth"])


def _org(p: Principal, db: Session) -> D.Organization:
    if p.org is None:
        raise HTTPException(400, "no org in context (send x-org or ?org=)")
    return p.org


# ---------------- organizations ----------------
class OrgIn(BaseModel):
    name: str
    slug: str
    plan: str = "free"
    quotas: dict[str, Any] = Field(default_factory=dict)
    settings: dict[str, Any] = Field(default_factory=dict)


class OrgPatch(BaseModel):
    name: Optional[str] = None
    plan: Optional[str] = None
    quotas: Optional[dict[str, Any]] = None
    settings: Optional[dict[str, Any]] = None


@router.get("/orgs", tags=["orgs"])
def list_orgs(p: Principal = Depends(get_principal), db: Session = Depends(D.get_db)):
    if p.is_superadmin:
        orgs = db.query(D.Organization).order_by(D.Organization.created_at).all()
    elif p.user is not None:
        ids = [m.org_id for m in db.query(D.Membership).filter_by(user_id=p.user.id)]
        orgs = db.query(D.Organization).filter(D.Organization.id.in_(ids)).all()
    else:
        orgs = [p.org] if p.org else []
    return [_org_view(o, db) for o in orgs]


@router.post("/orgs", status_code=201, tags=["orgs"])
def create_org(body: OrgIn, p: Principal = Depends(get_principal), db: Session = Depends(D.get_db)):
    """Superadmins (root key / superadmin users) create orgs; a user creator becomes its owner."""
    if not p.is_superadmin and p.actor_type != "user":
        raise HTTPException(403, "only superadmins or signed-in users can create orgs")
    if db.query(D.Organization).filter_by(slug=body.slug).one_or_none():
        raise HTTPException(409, "slug already exists")
    if body.plan not in PLAN_DEFAULTS:
        raise HTTPException(400, f"plan must be one of {list(PLAN_DEFAULTS)}")
    o = D.Organization(id=D.uid("org"), name=body.name, slug=body.slug, plan=body.plan, quotas=body.quotas, settings=body.settings)
    db.add(o)
    db.commit()
    if p.user is not None:
        db.add(D.Membership(id=D.uid("mem"), org_id=o.id, user_id=p.user.id, role="owner"))
        db.commit()
    D.ensure_project(db, settings.default_project, org_id=o.id)
    audit(db, p, "orgs.create", "org", o.id, details={"slug": o.slug, "plan": o.plan})
    return _org_view(o, db)


@router.get("/orgs/current", tags=["orgs"])
def current_org(p: Principal = Depends(require("read")), db: Session = Depends(D.get_db)):
    return _org_view(_org(p, db), db) | {"you": {"actor_type": p.actor_type, "label": p.label, "role": p.role}}


@router.patch("/orgs/current", tags=["orgs"])
def patch_org(body: OrgPatch, p: Principal = Depends(require("org.write")), db: Session = Depends(D.get_db)):
    o = _org(p, db)
    for k, v in body.model_dump(exclude_none=True).items():
        setattr(o, k, v)
    db.commit()
    audit(db, p, "orgs.update", "org", o.id, details=body.model_dump(exclude_none=True))
    return _org_view(o, db)


def _org_view(o: D.Organization, db: Session) -> dict[str, Any]:
    return D.to_dict(o) | {"effective_quotas": effective_quotas(o),
                           "n_members": db.query(D.Membership).filter_by(org_id=o.id).count(),
                           "n_projects": db.query(D.Project).filter_by(org_id=o.id).count(),
                           "n_api_keys": db.query(D.ApiKey).filter_by(org_id=o.id, revoked_at=None).count()}


# ---------------- members ----------------
class MemberIn(BaseModel):
    email: str
    role: str = "member"
    name: str = ""


@router.get("/orgs/current/members", tags=["orgs"])
def list_members(p: Principal = Depends(require("org.read")), db: Session = Depends(D.get_db)):
    o = _org(p, db)
    rows = db.query(D.Membership, D.User).join(D.User, D.User.id == D.Membership.user_id).filter(D.Membership.org_id == o.id).all()
    return [{"membership_id": m.id, "user_id": u.id, "email": u.email, "name": u.name, "role": m.role, "sso_provider": u.sso_provider,
             "last_login_at": u.last_login_at, "created_at": m.created_at} for m, u in rows]


@router.post("/orgs/current/members", status_code=201, tags=["orgs"])
def add_member(body: MemberIn, p: Principal = Depends(require("members")), db: Session = Depends(D.get_db)):
    o = _org(p, db)
    if body.role not in ("viewer", "member", "admin", "owner"):
        raise HTTPException(400, "invalid role")
    if body.role == "owner" and not p.has_role("owner"):
        raise HTTPException(403, "only owners can add owners")
    q = effective_quotas(o)
    if q.get("members") and db.query(D.Membership).filter_by(org_id=o.id).count() >= q["members"]:
        raise HTTPException(402, f"member limit reached for plan {o.plan}")
    u = upsert_user(db, body.email, body.name)
    m = db.query(D.Membership).filter_by(org_id=o.id, user_id=u.id).one_or_none()
    if m is None:
        m = D.Membership(id=D.uid("mem"), org_id=o.id, user_id=u.id, role=body.role)
        db.add(m)
    else:
        m.role = body.role
    db.commit()
    audit(db, p, "members.upsert", "membership", m.id, details={"email": u.email, "role": m.role})
    return {"membership_id": m.id, "user_id": u.id, "email": u.email, "role": m.role}


@router.delete("/orgs/current/members/{membership_id}", tags=["orgs"])
def remove_member(membership_id: str, p: Principal = Depends(require("members")), db: Session = Depends(D.get_db)):
    o = _org(p, db)
    m = db.query(D.Membership).filter_by(id=membership_id, org_id=o.id).one_or_none()
    if m is None:
        raise HTTPException(404, "membership not found")
    if m.role == "owner" and db.query(D.Membership).filter_by(org_id=o.id, role="owner").count() <= 1:
        raise HTTPException(400, "cannot remove the last owner")
    db.delete(m)
    db.commit()
    audit(db, p, "members.remove", "membership", membership_id)
    return {"removed": membership_id}


# ---------------- API keys ----------------
class KeyIn(BaseModel):
    name: str
    role: str = "member"
    expires_in_days: Optional[int] = None
    allowed_ips: list[str] = Field(default_factory=list)  # IPs or CIDRs; empty = any


@router.get("/orgs/current/keys", tags=["orgs"])
def list_keys(p: Principal = Depends(require("keys")), db: Session = Depends(D.get_db)):
    o = _org(p, db)
    return [{k: v for k, v in D.to_dict(r).items() if k != "key_hash"} for r in
            db.query(D.ApiKey).filter_by(org_id=o.id).order_by(D.ApiKey.created_at.desc()).all()]


@router.post("/orgs/current/keys", status_code=201, tags=["orgs"])
def create_key(body: KeyIn, p: Principal = Depends(require("keys")), db: Session = Depends(D.get_db)):
    o = _org(p, db)
    if not p.has_role(body.role) and not p.is_superadmin:
        raise HTTPException(403, "cannot create a key with a role above your own")
    row, raw = create_api_key(db, o, body.name, body.role, created_by=p.actor_id, expires_in_days=body.expires_in_days, allowed_ips=body.allowed_ips)
    audit(db, p, "keys.create", "api_key", row.id, details={"name": row.name, "role": row.role, "prefix": row.prefix})
    return {k: v for k, v in D.to_dict(row).items() if k != "key_hash"} | {"key": raw, "note": "store this key now; it is not shown again"}


@router.delete("/orgs/current/keys/{key_id}", tags=["orgs"])
def revoke_key(key_id: str, p: Principal = Depends(require("keys")), db: Session = Depends(D.get_db)):
    o = _org(p, db)
    row = db.query(D.ApiKey).filter_by(id=key_id, org_id=o.id).one_or_none()
    if row is None:
        raise HTTPException(404, "key not found")
    row.revoked_at = time.time()
    db.commit()
    audit(db, p, "keys.revoke", "api_key", row.id, details={"prefix": row.prefix})
    return {"revoked": key_id}


# ---------------- usage & audit ----------------
@router.get("/orgs/current/usage", tags=["orgs"])
def usage(days: int = Query(30, le=365), p: Principal = Depends(require("usage")), db: Session = Depends(D.get_db)):
    o = _org(p, db)
    return usage_summary(db, o.id, days) | {"quotas": effective_quotas(o), "plan": o.plan}


@router.get("/audit", tags=["audit"])
def list_audit(limit: int = Query(100, le=1000), offset: int = 0, action: Optional[str] = None, actor: Optional[str] = None,
               since: Optional[float] = None, p: Principal = Depends(require("audit")), db: Session = Depends(D.get_db)):
    q = db.query(D.AuditLog)
    if not p.is_superadmin or p.org is not None:
        q = q.filter_by(org_id=_org(p, db).id)
    if action:
        q = q.filter(D.AuditLog.action.like(f"{action}%"))
    if actor:
        q = q.filter(D.AuditLog.actor_label.like(f"%{actor}%"))
    if since:
        q = q.filter(D.AuditLog.created_at >= since)
    total = q.count()
    rows = q.order_by(D.AuditLog.created_at.desc()).offset(offset).limit(limit).all()
    return {"total": total, "items": [D.to_dict(r) for r in rows]}


# ---------------- auth ----------------
class DevLoginIn(BaseModel):
    email: str
    name: str = ""
    org: Optional[str] = None  # join this org as `role` (dev only)
    role: str = "member"


@auth_router.get("/me")
def me(p: Principal = Depends(get_principal), db: Session = Depends(D.get_db)):
    orgs = []
    if p.user is not None:
        for m in db.query(D.Membership).filter_by(user_id=p.user.id):
            o = db.get(D.Organization, m.org_id)
            orgs.append({"slug": o.slug, "name": o.name, "role": m.role})
    return {"actor_type": p.actor_type, "label": p.label, "role": p.role, "org": p.org.slug if p.org else None,
            "user": {"id": p.user.id, "email": p.user.email, "name": p.user.name, "is_superadmin": p.user.is_superadmin} if p.user else None,
            "orgs": orgs, "permissions": [k for k in ("read", "write", "deploy", "keys", "members", "org.read", "org.write", "audit", "usage") if p.can(k)]}


@auth_router.get("/config")
def auth_config():
    """Public: tells the dashboard which login methods exist."""
    return {"oidc": bool(settings.oidc_issuer and settings.oidc_client_id), "dev_login": settings.dev_login,
            "login_url": "/v1/auth/login" if settings.oidc_issuer else None}


@auth_router.post("/dev-login")
def dev_login(body: DevLoginIn, db: Session = Depends(D.get_db)):
    """Mint a JWT without an identity provider. Enabled only with SAPHIRE_DEV_LOGIN=1 (never in production)."""
    if not settings.dev_login:
        raise HTTPException(404, "dev login disabled")
    u = upsert_user(db, body.email, body.name, provider="dev")
    auto_join(db, u)
    if body.org:
        o = db.query(D.Organization).filter_by(slug=body.org).one_or_none() or D.ensure_org(db, body.org, body.org, plan="free")
        if db.query(D.Membership).filter_by(org_id=o.id, user_id=u.id).one_or_none() is None:
            db.add(D.Membership(id=D.uid("mem"), org_id=o.id, user_id=u.id, role=body.role))
            db.commit()
    token = issue_jwt(u)
    audit(db, Principal("user", u.id, u.email, None, "member", user=u), "auth.dev_login", "user", u.id)
    return {"token": token, "user": {"id": u.id, "email": u.email}}


_OIDC_CACHE: dict[str, Any] = {}
_STATES: dict[str, float] = {}


def _oidc_meta() -> dict[str, Any]:
    if not settings.oidc_issuer:
        raise HTTPException(404, "SSO not configured (set SAPHIRE_OIDC_ISSUER / CLIENT_ID / CLIENT_SECRET)")
    if "meta" not in _OIDC_CACHE:
        r = httpx.get(settings.oidc_issuer.rstrip("/") + "/.well-known/openid-configuration", timeout=15)
        r.raise_for_status()
        _OIDC_CACHE["meta"] = r.json()
    return _OIDC_CACHE["meta"]


@auth_router.get("/login")
def oidc_login(next: str = "/"):
    meta = _oidc_meta()
    state = secrets.token_urlsafe(24)
    _STATES[state] = time.time()
    params = {"response_type": "code", "client_id": settings.oidc_client_id, "redirect_uri": settings.oidc_redirect_url,
              "scope": settings.oidc_scopes, "state": state}
    return RedirectResponse(meta["authorization_endpoint"] + "?" + urlencode(params))


@auth_router.get("/callback")
def oidc_callback(code: str, state: str, request: Request, db: Session = Depends(D.get_db)):
    if state not in _STATES or time.time() - _STATES.pop(state) > 600:
        raise HTTPException(400, "invalid or expired state")
    meta = _oidc_meta()
    tok = httpx.post(meta["token_endpoint"], data={"grant_type": "authorization_code", "code": code, "redirect_uri": settings.oidc_redirect_url,
                                                    "client_id": settings.oidc_client_id, "client_secret": settings.oidc_client_secret}, timeout=15)
    if tok.status_code >= 400:
        raise HTTPException(401, f"token exchange failed: {tok.text[:200]}")
    access = tok.json().get("access_token")
    info = httpx.get(meta["userinfo_endpoint"], headers={"Authorization": f"Bearer {access}"}, timeout=15).json()
    email = info.get("email")
    if not email:
        raise HTTPException(401, "identity provider did not return an email")
    u = upsert_user(db, email, info.get("name", ""), provider=settings.oidc_issuer, subject=str(info.get("sub", "")))
    auto_join(db, u)
    audit(db, Principal("user", u.id, u.email, None, "member", user=u), "auth.sso_login", "user", u.id, ip=request.client.host if request.client else "")
    token = issue_jwt(u)
    return RedirectResponse(f"{settings.frontend_url.rstrip('/')}/login/#token={token}")


@auth_router.post("/token/refresh")
def refresh(p: Principal = Depends(get_principal)):
    if p.user is None:
        raise HTTPException(400, "only user sessions can be refreshed")
    return {"token": issue_jwt(p.user)}


@auth_router.get("/validate")
def validate(token: str):
    return {"valid": True, "claims": decode_jwt(token)}
