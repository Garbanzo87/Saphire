"""SCIM provisioning, key IP allow-lists, Prometheus metrics, gym adapter, migrations."""
import subprocess
import sys

from saphire.distributed.gym import SaphireGymEnv
from saphire.environments import get_environment


def _org(api, slug="acme"):
    api.post("/v1/orgs", json={"name": slug, "slug": slug, "plan": "team", "settings": {"default_role": "viewer"}})
    return api.post("/v1/orgs/current/keys", json={"name": "scim", "role": "admin"}, headers={"x-org": slug}).json()["key"]


def test_scim_lifecycle(api):
    h = {"x-api-key": _org(api)}
    assert api.get("/scim/v2/ServiceProviderConfig").json()["patch"]["supported"]
    u = api.post("/scim/v2/Users", json={"schemas": ["urn:ietf:params:scim:schemas:core:2.0:User"], "userName": "Ann@Acme.com", "displayName": "Ann",
                                         "externalId": "okta-1", "roles": [{"value": "member"}]}, headers=h)
    assert u.status_code == 201 and u.json()["active"] and u.json()["roles"][0]["value"] == "member"
    uid = u.json()["id"]
    assert api.post("/scim/v2/Users", json={"userName": "ann@acme.com"}, headers=h).status_code == 409
    lst = api.get("/scim/v2/Users", params={"filter": 'userName eq "ann@acme.com"'}, headers=h).json()
    assert lst["totalResults"] == 1 and lst["Resources"][0]["userName"] == "ann@acme.com"
    p = api.patch(f"/scim/v2/Users/{uid}", json={"schemas": ["urn:ietf:params:scim:api:messages:2.0:PatchOp"],
                                                 "Operations": [{"op": "replace", "value": {"active": False}}]}, headers=h).json()
    assert p["active"] is False
    assert api.get(f"/scim/v2/Users/{uid}", headers=h).status_code == 404  # no longer a member
    p = api.put(f"/scim/v2/Users/{uid}", json={"active": True, "displayName": "Ann L"}, headers=h).json()
    assert p["active"] and p["displayName"] == "Ann L"
    assert api.delete(f"/scim/v2/Users/{uid}", headers=h).status_code == 204
    assert api.get("/scim/v2/Users", headers=h).json()["totalResults"] == 0
    # viewer keys cannot provision
    v = api.post("/v1/orgs/current/keys", json={"name": "v", "role": "viewer"}, headers={"x-org": "acme"}).json()["key"]
    assert api.get("/scim/v2/Users", headers={"x-api-key": v}).status_code == 403
    assert any(i["action"].startswith("scim.users") for i in api.get("/v1/audit", headers=h).json()["items"])


def test_api_key_ip_allowlist(api):
    api.post("/v1/orgs", json={"name": "g", "slug": "globex", "plan": "team"})
    assert api.post("/v1/orgs/current/keys", json={"name": "k", "role": "member", "allowed_ips": ["not-an-ip"]}, headers={"x-org": "globex"}).status_code == 400
    ok = api.post("/v1/orgs/current/keys", json={"name": "k", "role": "member", "allowed_ips": ["testclient", "10.0.0.0/8"]}, headers={"x-org": "globex"})
    # TestClient's client host is "testclient" (not an IP) -> refused; a real IP range key works for matching hosts
    bad = api.post("/v1/orgs/current/keys", json={"name": "k2", "role": "member", "allowed_ips": ["10.0.0.0/8"]}, headers={"x-org": "globex"}).json()
    assert api.get("/v1/agents", headers={"x-api-key": bad["key"]}).status_code == 403
    assert ok.status_code == 400  # "testclient" is not a valid CIDR
    # behind a load balancer: X-Forwarded-For is honoured only when SAPHIRE_TRUST_PROXY=1
    fwd = {"x-api-key": bad["key"], "x-forwarded-for": "10.1.2.3, 192.0.2.1"}
    assert api.get("/v1/agents", headers=fwd).status_code == 403
    from saphire.server.config import settings

    settings.trust_proxy = True
    try:
        assert api.get("/v1/agents", headers=fwd).status_code == 200
        assert api.get("/v1/agents", headers={"x-api-key": bad["key"], "x-forwarded-for": "172.16.0.1"}).status_code == 403
    finally:
        settings.trust_proxy = False


def test_prometheus_metrics(api):
    api.get("/v1/agents")
    txt = api.get("/metrics").text
    assert "saphire_http_requests_total" in txt and 'route="/v1/agents"' in txt and "saphire_jobs_pending" in txt
    assert "saphire_http_request_seconds_bucket" in txt


def test_gym_adapter():
    env = get_environment("support_desk")
    g = SaphireGymEnv(env)
    obs, info = g.reset(seed=1)
    task = info["task"]
    assert task["instruction"] in obs and len(info["tools"]) == 30
    oid = task["instruction"].split()[3]
    obs, r, term, trunc, info = g.step('<tool_call>{"name": "lookup_order", "arguments": {"order_id": "%s"}}</tool_call>' % oid)
    assert not term and r > 0 and oid in obs
    obs, r, term, trunc, info = g.step("The order is " + ("processing" if "processing" in obs else "shipped" if "shipped" in obs else "delivered"))
    assert term and "rewards" in info and info["rewards"]["task_success"] in (0.0, 1.0)
    g.reset(seed=2)
    for _ in range(4):
        obs, r, term, trunc, info = g.step('<tool_call>{"name": "lookup_order", "arguments": {"order_id": "%s"}}</tool_call>' % oid)
        if term or trunc:
            break
    assert trunc or term


def test_migrations_apply(tmp_path):
    import os

    env = dict(os.environ, SAPHIRE_DATABASE_URL=f"sqlite:///{tmp_path}/m.db")
    r = subprocess.run([sys.executable, "-m", "alembic", "upgrade", "head"], cwd=os.path.dirname(os.path.dirname(__file__)), env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    import sqlite3

    tables = {row[0] for row in sqlite3.connect(f"{tmp_path}/m.db").execute("select name from sqlite_master where type='table'")}
    assert {"organizations", "rollouts", "intelligence_reports", "webhooks", "alembic_version"} <= tables
