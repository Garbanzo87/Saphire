"""OpenAPI connector: every operation in an OpenAPI 3 document becomes a tool that calls the API via httpx."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional

import httpx
import yaml

from ..sdk.tools import ToolRegistry
from ..sdk.types import ToolSpec


class OpenAPIConnector:
    def __init__(self, spec: str | dict, base_url: Optional[str] = None, headers: Optional[dict[str, str]] = None,
                 include: Optional[list[str]] = None, timeout: float = 30.0, tag: str = "openapi", client: Optional[httpx.Client] = None):
        self.spec = self._load(spec)
        servers = self.spec.get("servers") or [{"url": ""}]
        self.base_url = (base_url or servers[0]["url"]).rstrip("/")
        self.include = include
        self.client = client or httpx.Client(base_url=self.base_url, headers=headers or {}, timeout=timeout)
        self.tag = tag

    @staticmethod
    def _load(spec: str | dict) -> dict:
        if isinstance(spec, dict):
            return spec
        if spec.startswith("http"):
            r = httpx.get(spec, timeout=30)
            r.raise_for_status()
            txt = r.text
        else:
            txt = Path(spec).read_text()
        return yaml.safe_load(txt) if not txt.lstrip().startswith("{") else json.loads(txt)

    def _resolve(self, obj: Any) -> Any:
        if isinstance(obj, dict) and "$ref" in obj:
            node = self.spec
            for part in obj["$ref"].lstrip("#/").split("/"):
                node = node[part]
            return self._resolve(node)
        if isinstance(obj, dict):
            return {k: self._resolve(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._resolve(v) for v in obj]
        return obj

    def registry(self) -> ToolRegistry:
        reg = ToolRegistry()
        for path, ops in (self.spec.get("paths") or {}).items():
            for method, op in ops.items():
                if method.lower() not in ("get", "post", "put", "patch", "delete"):
                    continue
                op = self._resolve(op)
                name = op.get("operationId") or re.sub(r"[^a-z0-9]+", "_", f"{method}_{path}".lower()).strip("_")
                if self.include and name not in self.include:
                    continue
                props: dict[str, Any] = {}
                required: list[str] = []
                locs: dict[str, str] = {}
                for p in op.get("parameters", []):
                    props[p["name"]] = {**p.get("schema", {"type": "string"}), "description": p.get("description", "")}
                    locs[p["name"]] = p.get("in", "query")
                    if p.get("required"):
                        required.append(p["name"])
                body = op.get("requestBody", {}).get("content", {}).get("application/json", {}).get("schema")
                body_props = list((body or {}).get("properties", {}).keys()) if body else []
                if body:
                    for k, v in body.get("properties", {}).items():
                        props[k] = v
                        locs[k] = "body"
                    required += body.get("required", [])
                spec = ToolSpec(name=name, description=op.get("summary") or op.get("description") or f"{method.upper()} {path}",
                                parameters={"type": "object", "properties": props, "required": required}, tags=[self.tag] + op.get("tags", []),
                                source="openapi")
                reg.add_spec(spec, self._make_call(method.upper(), path, locs, body_props))
        return reg

    def _make_call(self, method: str, path: str, locs: dict[str, str], body_props: list[str]):
        def _call(**kw):
            url = path
            params, body, headers = {}, {}, {}
            for k, v in kw.items():
                loc = locs.get(k, "query")
                if loc == "path":
                    url = url.replace("{" + k + "}", str(v))
                elif loc == "query":
                    params[k] = v
                elif loc == "header":
                    headers[k] = str(v)
                else:
                    body[k] = v
            r = self.client.request(method, url, params=params, json=body or None, headers=headers)
            if r.status_code >= 400:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
            try:
                return r.json()
            except ValueError:
                return {"status": r.status_code, "text": r.text[:1000]}

        return _call
