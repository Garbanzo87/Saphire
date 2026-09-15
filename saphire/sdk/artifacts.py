"""Artifact storage that works across nodes.

Learners write artifacts (routers, exemplar stores, checkpoints) to a local directory. When workers and API
replicas do not share a filesystem, an `ArtifactStore` mirrors them to object storage and agents resolve
remote URIs back to a local cache on demand:

    SAPHIRE_ARTIFACT_STORE=s3://bucket/prefix      (boto3, `pip install saphire[s3]`; MinIO/R2 via AWS_ENDPOINT_URL)
    SAPHIRE_ARTIFACT_STORE=file:///shared/mirror   (any mounted path — also what the tests use)

`resolve(uri)` is called by `ToolRouter.load`, `ExemplarStore.load` and `HFLocalProvider`, so an `AgentConfig`
may point at `s3://.../router.json` and still run anywhere.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

CACHE_DIR = Path(os.getenv("SAPHIRE_ARTIFACT_CACHE", os.path.expanduser("~/.cache/saphire/artifacts")))


class ArtifactStore:
    def __init__(self, base: str):
        self.base = base.rstrip("/")
        u = urlparse(self.base)
        self.scheme = u.scheme or "file"
        if self.scheme == "s3":
            import boto3  # type: ignore

            self.bucket = u.netloc
            self.prefix = u.path.strip("/")
            self.s3 = boto3.client("s3", endpoint_url=os.getenv("AWS_ENDPOINT_URL") or None)
        elif self.scheme == "file":
            self.root = Path(u.path if u.netloc == "" else "/" + u.netloc + u.path)
            self.root.mkdir(parents=True, exist_ok=True)
        else:
            raise ValueError(f"unsupported artifact store scheme {self.scheme}")

    # ---- upload ----
    def upload_dir(self, local_dir: str | Path, key_prefix: str) -> dict[str, str]:
        """Upload every file under local_dir; returns {local_path: remote_uri}."""
        local_dir = Path(local_dir)
        out: dict[str, str] = {}
        for f in sorted(p for p in local_dir.rglob("*") if p.is_file()):
            rel = f.relative_to(local_dir).as_posix()
            key = f"{key_prefix.strip('/')}/{rel}"
            out[str(f)] = self.put(f, key)
        return out

    def put(self, local_file: Path, key: str) -> str:
        if self.scheme == "s3":
            full = f"{self.prefix}/{key}".strip("/")
            self.s3.upload_file(str(local_file), self.bucket, full)
            return f"s3://{self.bucket}/{full}"
        dest = self.root / key
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_file, dest)
        return f"file://{dest}"

    # ---- download ----
    def get(self, uri: str, dest: Path) -> Path:
        u = urlparse(uri)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if u.scheme == "s3":
            self.s3.download_file(u.netloc, u.path.lstrip("/"), str(dest))
        else:
            shutil.copy2(u.path if u.netloc == "" else "/" + u.netloc + u.path, dest)
        return dest

    def exists(self, uri: str) -> bool:
        u = urlparse(uri)
        if u.scheme == "s3":
            try:
                self.s3.head_object(Bucket=u.netloc, Key=u.path.lstrip("/"))
                return True
            except Exception:  # noqa: BLE001
                return False
        return Path(u.path if u.netloc == "" else "/" + u.netloc + u.path).exists()


_STORE: Optional[ArtifactStore] = None


def get_store() -> Optional[ArtifactStore]:
    global _STORE
    base = os.getenv("SAPHIRE_ARTIFACT_STORE", "")
    if not base:
        return None
    if _STORE is None or _STORE.base != base.rstrip("/"):
        _STORE = ArtifactStore(base)
    return _STORE


def is_remote(path: str | None) -> bool:
    return bool(path) and (path.startswith("s3://") or path.startswith("file://"))


def resolve(path: str | Path, siblings: tuple[str, ...] = ()) -> str:
    """Return a local path for `path`. Remote URIs are downloaded into the cache (together with sibling files that
    share the stem, e.g. router.json + router.npz)."""
    p = str(path)
    if not is_remote(p):
        return p
    store = get_store() or ArtifactStore(p.rsplit("/", 1)[0])
    u = urlparse(p)
    local = CACHE_DIR / u.netloc / u.path.lstrip("/")
    if not local.exists():
        store.get(p, local)
    stem = p.rsplit(".", 1)[0]
    for ext in siblings:
        sib = f"{stem}.{ext}"
        loc = CACHE_DIR / u.netloc / urlparse(sib).path.lstrip("/")
        if not loc.exists() and store.exists(sib):
            store.get(sib, loc)
    return str(local)


def publish_dir(local_dir: str | Path, key_prefix: str) -> dict[str, str]:
    """Mirror a version directory to the configured store (no-op without SAPHIRE_ARTIFACT_STORE)."""
    store = get_store()
    if store is None:
        return {}
    return store.upload_dir(local_dir, key_prefix)


def remap(path: Optional[str], mapping: dict[str, str]) -> Optional[str]:
    """Rewrite a local artifact path to its published URI (router artifacts are addressed by their .json)."""
    if not path:
        return path
    return mapping.get(str(path), path)
