import os

import pytest

os.environ.setdefault("SAPHIRE_TRACING", "off")


@pytest.fixture()
def api(tmp_path, monkeypatch):
    """A TestClient against a fresh SQLite DB with inline jobs."""
    from saphire.server import config, db

    monkeypatch.setattr(config.settings, "database_url", f"sqlite:///{tmp_path}/test.db")
    monkeypatch.setattr(config.settings, "inline_jobs", True)
    monkeypatch.setattr(config.settings, "artifacts_dir", str(tmp_path / "artifacts"))
    db.reset_engine()
    from fastapi.testclient import TestClient

    from saphire.server.main import create_app

    with TestClient(create_app()) as c:
        c.headers["x-api-key"] = "dev-key"
        yield c
    db.reset_engine()


def wait_job(api, job_id, timeout=120):
    import time

    t0 = time.time()
    while time.time() - t0 < timeout:
        j = api.get(f"/v1/jobs/{job_id}").json()
        if j["status"] in ("succeeded", "failed", "cancelled"):
            assert j["status"] == "succeeded", j.get("error")
            return j
        time.sleep(0.2)
    raise TimeoutError(job_id)
